"""Rewritten tensor operations. Every kernel is provided by installed PyTorch."""
import torch
from torch import nn
from torch.nn import functional as F
from .quantization import NVFP4Linear


class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__();self.weight=nn.Parameter(torch.ones(dim));self.eps=eps

    def forward(self,x):
        dtype=x.dtype
        x=x.float()
        x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
        return (x*self.weight.float()).to(dtype)


class NumbatGatedDeltaNet(nn.Module):
    def __init__(self,config,quantized=True):
        super().__init__();g=config.gdn_config
        self.num_heads=g['num_heads'];self.head_dim_k=g['head_dim_k'];self.head_dim_v=g['head_dim_v']
        self.kernel=g['short_conv_kernel'];self.qk_l2norm=g.get('qk_l2norm',True)
        hidden=config.hidden_size

        def linear(inputs,outputs):
            return NVFP4Linear(inputs,outputs) if quantized else nn.Linear(inputs,outputs,bias=False)

        self.q_proj=linear(hidden,self.num_heads*self.head_dim_k)
        self.k_proj=linear(hidden,self.num_heads*self.head_dim_k)
        self.v_proj=linear(hidden,self.num_heads*self.head_dim_v)
        self.g_proj=linear(hidden,self.num_heads*self.head_dim_v)
        self.o_proj=linear(hidden,self.num_heads*self.head_dim_v)
        self.a_proj=nn.Linear(hidden,self.num_heads,bias=False)
        self.b_proj=nn.Linear(hidden,self.num_heads,bias=False)
        for name,width in (('q_conv1d',self.num_heads*self.head_dim_k),('k_conv1d',self.num_heads*self.head_dim_k),('v_conv1d',self.num_heads*self.head_dim_v)):
            setattr(self,name,nn.Conv1d(width,width,kernel_size=self.kernel,groups=width,bias=g.get('conv_bias',False)))
        self.A_log=nn.Parameter(torch.zeros(self.num_heads,dtype=torch.float32))
        self.dt_bias=nn.Parameter(torch.zeros(self.num_heads,dtype=torch.float32))
        self.o_norm=RMSNorm(self.head_dim_v,config.rms_norm_eps)

    def _short_conv(self,module,x):
        batch,length=x.shape[0],x.shape[1]
        filtered=module(F.pad(x.reshape(batch,length,-1).transpose(1,2),(self.kernel-1,0)))
        return F.silu(filtered.transpose(1,2).reshape(x.shape))

    def forward(self,x,position_ids=None,attention_mask=None):
        batch,length,_=x.shape
        heads,keys,values=self.num_heads,self.head_dim_k,self.head_dim_v
        q=self._short_conv(self.q_conv1d,self.q_proj(x).view(batch,length,heads,keys))
        k=self._short_conv(self.k_conv1d,self.k_proj(x).view(batch,length,heads,keys))
        v=self._short_conv(self.v_conv1d,self.v_proj(x).view(batch,length,heads,values))
        if self.qk_l2norm:
            q=F.normalize(q,dim=-1)
            k=F.normalize(k,dim=-1)
        q=q.permute(2,0,1,3)
        k=k.permute(2,0,1,3)
        v=v.permute(2,0,1,3)
        alpha=torch.exp(-torch.exp(self.A_log)*F.softplus(self.a_proj(x)+self.dt_bias)).permute(2,0,1)
        beta=torch.sigmoid(self.b_proj(x)).permute(2,0,1)
        if attention_mask is not None: beta=beta*attention_mask.to(beta.dtype)[None,:,:]
        state=x.new_zeros(heads,batch,values,keys,dtype=torch.float32);outputs=[]
        for t in range(length):
            key=k[:,:,t].to(torch.float32);value=v[:,:,t].to(torch.float32)
            decay=alpha[:,:,t:t+1].unsqueeze(-1);write=beta[:,:,t:t+1].unsqueeze(-1)
            state=decay*(state-write*state.matmul(key.unsqueeze(-1)).matmul(key.unsqueeze(-2)))
            state=state+write*value.unsqueeze(-1)*key.unsqueeze(-2)
            outputs.append(state.matmul(q[:,:,t].to(torch.float32).unsqueeze(-1)).squeeze(-1))
        output=torch.stack(outputs,dim=2)
        output=self.o_norm(output).permute(1,2,0,3).reshape(batch,length,heads*values)
        output=output.to(x.dtype)
        return self.o_proj(output*F.silu(self.g_proj(x)))


class NumbatLatentAttention(nn.Module):
    def __init__(self,config,quantized=True):
        super().__init__();m=config.mla_config
        self.num_heads=m['num_heads']
        self.qk_nope_head_dim=m['qk_nope_head_dim']
        self.qk_rope_head_dim=m['qk_rope_head_dim']
        self.v_head_dim=m['v_head_dim']
        self.q_lora_rank=m['q_lora_rank']
        self.kv_lora_rank=m['kv_lora_rank']
        self.rope_theta=m['rope_theta']
        hidden=config.hidden_size

        def linear(inputs,outputs):
            return NVFP4Linear(inputs,outputs) if quantized else nn.Linear(inputs,outputs,bias=False)

        self.q_a_proj=linear(hidden,self.q_lora_rank)
        self.q_a_layernorm=RMSNorm(self.q_lora_rank,config.rms_norm_eps)
        self.q_b_proj=linear(self.q_lora_rank,self.num_heads*(self.qk_nope_head_dim+self.qk_rope_head_dim))
        self.kv_a_proj_with_mqa=linear(hidden,self.kv_lora_rank+self.qk_rope_head_dim)
        self.kv_a_layernorm=RMSNorm(self.kv_lora_rank,config.rms_norm_eps)
        self.kv_b_proj=linear(self.kv_lora_rank,self.num_heads*(self.qk_nope_head_dim+self.v_head_dim))
        self.o_gate_proj=linear(hidden,hidden)
        self.o_proj=linear(hidden,hidden)

    @staticmethod
    def _rope(x,cos,sin):
        first,second=x[...,:x.shape[-1]//2],x[...,x.shape[-1]//2:]
        cos,sin=cos.to(x.dtype),sin.to(x.dtype)
        return torch.cat([first*cos-second*sin,second*cos+first*sin],dim=-1)

    def forward(self,x,position_ids=None,attention_mask=None):
        batch,length,_=x.shape
        if position_ids is None: position_ids=torch.arange(length,device=x.device,dtype=torch.float32).expand(batch,length)
        q=self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q=q.view(batch,length,self.num_heads,-1).permute(0,2,1,3)
        compressed=self.kv_a_proj_with_mqa(x)
        kv=self.kv_b_proj(self.kv_a_layernorm(compressed[...,:self.kv_lora_rank]))
        kv=kv.view(batch,length,self.num_heads,-1).permute(0,2,1,3)
        rotary_key=compressed[...,self.kv_lora_rank:].unsqueeze(1).expand(-1,self.num_heads,-1,-1)
        phase=position_ids.float()[:,None,:,None]*(self.rope_theta**(-torch.arange(0,self.qk_rope_head_dim,2,dtype=torch.float32,device=x.device)/self.qk_rope_head_dim)).view(1,1,1,-1)
        cosine,sine=phase.cos(),phase.sin()
        q=torch.cat([q[...,:self.qk_nope_head_dim],self._rope(q[...,self.qk_nope_head_dim:],cosine,sine)],dim=-1)
        k=torch.cat([kv[...,:self.qk_nope_head_dim],self._rope(rotary_key,cosine,sine)],dim=-1)
        v=kv[...,self.qk_nope_head_dim:]
        if attention_mask is None: output=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        else:
            keep=attention_mask.to(torch.bool)[:,None,None,:]
            mask=torch.zeros((batch,1,1,length),dtype=q.dtype,device=q.device)
            mask=mask.masked_fill(~keep,float('-inf'))+torch.triu(torch.full((length,length),float('-inf'),dtype=q.dtype,device=q.device),1)
            output=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        output=output.transpose(1,2).reshape(batch,length,-1)
        return self.o_proj(output*torch.sigmoid(self.o_gate_proj(x)))


class NumbatMLP(nn.Module):
    def __init__(self,config,intermediate_size,linear):
        super().__init__();hidden=config.hidden_size
        self.gate_proj=linear(hidden,intermediate_size);self.up_proj=linear(hidden,intermediate_size);self.down_proj=linear(intermediate_size,hidden)
        self.act_fn={'silu':F.silu,'relu':F.relu,'gelu':F.gelu}[config.hidden_act]

    def forward(self,x):
        return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))


class NumbatRouter(nn.Module):
    def __init__(self,config):
        super().__init__();m=config.moe_config
        self.num_experts=m['n_routed_experts'];self.top_k=m['num_experts_per_tok']
        self.n_group=m['group_limited_routing']['n_group'];self.topk_group=m['group_limited_routing']['topk_group']
        self.norm_topk_prob=m['norm_topk_prob'];self.routed_scaling_factor=m['routed_scaling_factor']
        self.weight=nn.Parameter(torch.empty(self.num_experts,config.hidden_size))
        self.register_buffer('e_score_correction_bias',torch.zeros(self.num_experts,dtype=torch.float32))

    def forward(self,x):
        scores=torch.sigmoid(F.linear(x.float(),self.weight.float()));corrected=scores+self.e_score_correction_bias
        group_scores=corrected.view(*scores.shape[:-1],self.n_group,-1).topk(2,dim=-1).values.sum(dim=-1)
        retained=group_scores.topk(self.topk_group,dim=-1).indices
        membership=torch.arange(self.num_experts,device=x.device)//(self.num_experts//self.n_group)
        candidates=(retained.unsqueeze(-1)==membership.view(1,1,-1)).any(dim=2)
        selected=corrected.masked_fill(~candidates,float('-inf')).topk(self.top_k,dim=-1).indices
        weights=scores.gather(-1,selected)
        if self.norm_topk_prob: weights=weights/weights.sum(dim=-1,keepdim=True)*self.routed_scaling_factor
        return selected,weights


class NumbatMoE(nn.Module):
    def __init__(self,config,quantized=True):
        super().__init__();self.config=config;self.gate=NumbatRouter(config)
        self.shared_experts=NumbatMLP(config,config.moe_config['moe_intermediate_size'],NVFP4Linear)
        self.experts=nn.ModuleList([NumbatMLP(config,config.moe_config['moe_intermediate_size'],NVFP4Linear) for _ in range(config.moe_config['n_routed_experts'])])

    def forward(self,x):
        batch,length,_=x.shape;selected,weights=self.gate(x)
        flat=x.view(-1,x.shape[-1]);selection=selected.view(-1,selected.shape[-1]);mixture_weights=weights.view(-1,weights.shape[-1]).to(x.dtype)
        output=torch.zeros_like(flat)
        for index,expert in enumerate(self.experts):
            hits=selection==index;rows=hits.any(dim=-1).nonzero(as_tuple=True)[0]
            if rows.numel()==0:
                continue
            row_weights=mixture_weights[rows][hits[rows]].unsqueeze(-1)
            output.index_add_(0,rows,expert(flat[rows])*row_weights)
        return self.shared_experts(x)+output.view(batch,length,-1)


class NumbatDecoderLayer(nn.Module):
    def __init__(self,config,layer_idx):
        super().__init__();dense=layer_idx in config.dense_layer_indices
        if layer_idx in config.mla_layer_indices:
            self.self_attn=NumbatLatentAttention(config,quantized=not dense)
        else:
            self.self_attn=NumbatGatedDeltaNet(config,quantized=not dense)
        if dense:
            self.mlp=NumbatMLP(config,config.intermediate_size,lambda i,o:nn.Linear(i,o,bias=False))
        else:
            self.mlp=NumbatMoE(config)
        self.input_layernorm=RMSNorm(config.hidden_size,config.rms_norm_eps)
        self.post_attention_layernorm=RMSNorm(config.hidden_size,config.rms_norm_eps)

    def forward(self,x,position_ids=None,attention_mask=None):
        x=x+self.self_attn(self.input_layernorm(x),position_ids,attention_mask)
        return x+self.mlp(self.post_attention_layernorm(x))


class NumbatMTP(nn.Module):
    def __init__(self,config):
        super().__init__();hidden=config.hidden_size
        self.enorm=RMSNorm(hidden,config.rms_norm_eps);self.hnorm=RMSNorm(hidden,config.rms_norm_eps)
        self.eh_proj=nn.Linear(hidden*2,hidden,bias=False)
        self.input_layernorm=RMSNorm(hidden,config.rms_norm_eps);self.post_attention_layernorm=RMSNorm(hidden,config.rms_norm_eps)
        self.self_attn=NumbatLatentAttention(config,quantized=False)
        self.mlp=NumbatMLP(config,config.intermediate_size,lambda i,o:nn.Linear(i,o,bias=False))

    def forward(self,h,e):
        fused=self.eh_proj(torch.cat([self.enorm(e),self.hnorm(h)],dim=-1))
        fused=fused+self.self_attn(self.input_layernorm(fused))
        fused=fused+self.mlp(self.post_attention_layernorm(fused))
        return fused
