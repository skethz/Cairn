"""Tensor layout and allocation accounting without importing a tensor library."""
from math import prod


def tensor_layout(config):
    h=config.hidden_size;g=config.gdn_config;a=config.mla_config;m=config.moe_config
    def linear(name,inputs,outputs,packed):
        if packed:
            yield name+'.weight',(outputs,inputs//2),'uint8',True
            yield name+'.weight_scale',(outputs,inputs//16),'float8_e4m3fn',False
            yield name+'.weight_scale_2',(),'float32',False
            yield name+'.input_scale',(),'float32',False
        else: yield name+'.weight',(outputs,inputs),'bfloat16',True
    def norm(name,width): yield name+'.weight',(width,),'bfloat16',True
    def mlp(name,width,packed):
        yield from linear(name+'.gate_proj',h,width,packed)
        yield from linear(name+'.up_proj',h,width,packed)
        yield from linear(name+'.down_proj',width,h,packed)
    def mixer(name,mla,packed):
        if mla:
            for label,inputs,outputs in [('q_a_proj',h,a['q_lora_rank']),('q_b_proj',a['q_lora_rank'],a['num_heads']*(a['qk_nope_head_dim']+a['qk_rope_head_dim'])),('kv_a_proj_with_mqa',h,a['kv_lora_rank']+a['qk_rope_head_dim']),('kv_b_proj',a['kv_lora_rank'],a['num_heads']*(a['qk_nope_head_dim']+a['v_head_dim'])),('o_gate_proj',h,h),('o_proj',h,h)]:
                yield from linear(name+'.'+label,inputs,outputs,packed)
            yield from norm(name+'.q_a_layernorm',a['q_lora_rank'])
            yield from norm(name+'.kv_a_layernorm',a['kv_lora_rank'])
        else:
            for label,width in [('q',g['num_heads']*g['head_dim_k']),('k',g['num_heads']*g['head_dim_k']),('v',g['num_heads']*g['head_dim_v']),('g',h),('o',h)]:
                yield from linear(name+'.'+label+'_proj',h,width,packed)
            for label in ('a','b'): yield from linear(name+'.'+label+'_proj',h,g['num_heads'],False)
            for label,width in [('q',g['num_heads']*g['head_dim_k']),('k',g['num_heads']*g['head_dim_k']),('v',h)]:
                yield name+'.'+label+'_conv1d.weight',(width,1,g['short_conv_kernel']),'bfloat16',True
                if g.get('conv_bias',False): yield name+'.'+label+'_conv1d.bias',(width,),'bfloat16',True
            for label in ('A_log','dt_bias'): yield name+'.'+label,(g['num_heads'],),'float32',True
            yield from norm(name+'.o_norm',g['head_dim_v'])
    yield 'model.embed_tokens.weight',(config.vocab_size,h),'bfloat16',True
    for index in range(config.num_hidden_layers):
        name=f'model.layers.{index}';dense=index in config.dense_layer_indices
        yield from mixer(name+'.self_attn',index in config.mla_layer_indices,not dense)
        for label in ('input_layernorm','post_attention_layernorm'): yield from norm(name+'.'+label,h)
        if dense: yield from mlp(name+'.mlp',config.intermediate_size,False)
        else:
            yield from linear(name+'.mlp.gate',h,m['n_routed_experts'],False)
            yield name+'.mlp.gate.e_score_correction_bias',(m['n_routed_experts'],),'float32',False
            yield from mlp(name+'.mlp.shared_experts',m['moe_intermediate_size'],True)
            for expert in range(m['n_routed_experts']):
                yield from mlp(name+f'.mlp.experts.{expert}',m['moe_intermediate_size'],True)
    name=f'model.layers.{config.num_hidden_layers}'
    for label in ('enorm','hnorm','input_layernorm','post_attention_layernorm'): yield from norm(name+'.'+label,h)
    yield from linear(name+'.eh_proj',2*h,h,False)
    yield from mixer(name+'.self_attn',True,False)
    yield from mlp(name+'.mlp',config.intermediate_size,False)
    yield from norm('model.norm',h)
    yield from linear('lm_head',h,config.vocab_size,False)


def constructor_bytes(config):
    """Conservative tensor allocation estimate with default FP32 plain weights.

    O(number of layers), including packed buffers. Excludes activation memory
    and Python/allocator overhead; the launcher separately limits input work.
    """
    h=config.hidden_size;g=config.gdn_config;a=config.mla_config;m=config.moe_config
    def linear(inputs,outputs,packed):
        return outputs*(inputs//2+inputs//16)+8 if packed else 4*inputs*outputs
    def mixer(mla,packed):
        if mla:
            pairs=[(h,a['q_lora_rank']),(a['q_lora_rank'],a['num_heads']*(a['qk_nope_head_dim']+a['qk_rope_head_dim'])),(h,a['kv_lora_rank']+a['qk_rope_head_dim']),(a['kv_lora_rank'],a['num_heads']*(a['qk_nope_head_dim']+a['v_head_dim'])),(h,h),(h,h)]
            return sum(linear(i,o,packed) for i,o in pairs)+4*(a['q_lora_rank']+a['kv_lora_rank'])
        key=g['num_heads']*g['head_dim_k'];value=g['num_heads']*g['head_dim_v']
        return (sum(linear(h,width,packed) for width in (key,key,value,value,value))+
                8*h*g['num_heads']+4*(2*key+value)*(g['short_conv_kernel']+int(g.get('conv_bias',False)))+
                8*g['num_heads']+4*g['head_dim_v'])
    dense_ffn=12*h*config.intermediate_size
    expert=2*linear(h,m['moe_intermediate_size'],True)+linear(m['moe_intermediate_size'],h,True)
    moe_ffn=4*h*m['n_routed_experts']+4*m['n_routed_experts']+(m['n_routed_experts']+1)*expert
    total=8*config.vocab_size*h+4*h
    for index in range(config.num_hidden_layers):
        dense=index in config.dense_layer_indices
        total+=mixer(index in config.mla_layer_indices,not dense)+8*h+(dense_ffn if dense else moe_ffn)
    return total+mixer(True,False)+dense_ffn+8*h*h+16*h


def layout_summary(config):
    sizes={'uint8':1,'float8_e4m3fn':1,'bfloat16':2,'float32':4}
    count=stored=bytes_total=index_count=0
    for name,shape,dtype,_ in tensor_layout(config):
        number=prod(shape);count+=1;stored+=number;bytes_total+=number*sizes[dtype]
        if not name.endswith('.weight_scale'): index_count+=number
    return {'tensor_entries':count,'stored_elements_including_buffers':stored,
            'index_count_excluding_fp8_block_scales':index_count,'checkpoint_storage_bytes':bytes_total,
            'constructor_tensor_bytes':constructor_bytes(config),'model_allocated':False}
