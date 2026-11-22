"""Data-only architecture configuration. Unknown keys are rejected."""
from dataclasses import dataclass, field, fields
from math import isfinite
from types import MappingProxyType


def _integer(value, label, upper=1_000_000):
    if type(value) is not int or not 0 < value <= upper:
        raise ValueError(f'{label} must be a positive bounded integer')


def _mapping(value, allowed, required, name):
    if type(value) not in (dict, MappingProxyType):
        raise ValueError(f'{name} must be a plain mapping')
    if set(value) - allowed or required - set(value):
        raise ValueError(f'Unsupported or missing {name} fields')
    return dict(value)


@dataclass(frozen=True)
class NumbatConfig:
    vocab_size: int = 131072
    hidden_size: int = 12288
    num_hidden_layers: int = 114
    hidden_act: str = 'silu'
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = False
    intermediate_size: int | None = None
    gdn_config: dict = field(default_factory=dict)
    mla_config: dict = field(default_factory=dict)
    moe_config: dict = field(default_factory=dict)
    mtp_config: dict = field(default_factory=dict)
    dense_layer_indices: tuple = ()
    mla_layer_indices: tuple = ()

    def __post_init__(self):
        for name in ('vocab_size', 'hidden_size', 'num_hidden_layers', 'intermediate_size', 'max_position_embeddings'):
            _integer(getattr(self, name), name, 4096 if name == 'num_hidden_layers' else 1_000_000)
        if type(self.rms_norm_eps) not in (int, float) or not isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError('rms_norm_eps must be finite and positive')
        if self.hidden_act not in ('silu', 'relu', 'gelu') or self.tie_word_embeddings is not False:
            raise ValueError('Unsupported activation or tied embeddings')
        for name in ('dense_layer_indices', 'mla_layer_indices'):
            values=getattr(self, name)
            if type(values) not in (list, tuple) or len(set(values)) != len(values):
                raise ValueError('Layer indices must be unique integers')
            if any(type(i) is not int or not 0 <= i < self.num_hidden_layers for i in values):
                raise ValueError('Invalid layer index')
            object.__setattr__(self, name, tuple(values))
        g_required={'num_heads','head_dim_k','head_dim_v','short_conv_kernel'}
        g=_mapping(self.gdn_config,g_required|{'qk_l2norm','conv_bias','chunk_size','state_dtype'},g_required,'gdn_config')
        for name in g_required: _integer(g[name], name)
        for name in ('qk_l2norm','conv_bias'):
            if name in g and type(g[name]) is not bool: raise ValueError('Invalid convolution/normalization flag')
        if g.get('state_dtype','float32') != 'float32': raise ValueError('State must be float32')
        a_keys={'num_heads','q_lora_rank','kv_lora_rank','qk_nope_head_dim','qk_rope_head_dim','v_head_dim','rope_theta'}
        a=_mapping(self.mla_config,a_keys,a_keys,'mla_config')
        for name in a_keys-{'rope_theta'}: _integer(a[name],name)
        if type(a['rope_theta']) not in (int,float) or not isfinite(a['rope_theta']) or a['rope_theta'] <= 0:
            raise ValueError('Invalid rotary base')
        m_keys={'n_routed_experts','n_shared_experts','num_experts_per_tok','moe_intermediate_size','group_limited_routing','norm_topk_prob','routed_scaling_factor'}
        m=_mapping(self.moe_config,m_keys,m_keys,'moe_config')
        for name in ('n_routed_experts','num_experts_per_tok','moe_intermediate_size'): _integer(m[name],name,65536)
        if type(m['n_shared_experts']) is not int or m['n_shared_experts'] != 1: raise ValueError('Exactly one shared expert is implemented')
        grouping=_mapping(m['group_limited_routing'],{'n_group','topk_group'},{'n_group','topk_group'},'routing groups')
        for name in grouping: _integer(grouping[name],name,65536)
        groups=grouping['n_group']; kept=grouping['topk_group']; experts=m['n_routed_experts']
        if experts % groups or experts//groups < 2 or kept > groups or m['num_experts_per_tok'] > kept*(experts//groups):
            raise ValueError('Incompatible top-two group routing dimensions')
        if type(m['norm_topk_prob']) is not bool: raise ValueError('Invalid normalization flag')
        scale=m['routed_scaling_factor']
        if type(scale) not in (int,float) or not isfinite(scale) or scale <= 0: raise ValueError('Invalid routed scale')
        m['group_limited_routing']=MappingProxyType(grouping)
        mtp=_mapping(self.mtp_config,{'depth','layer_index_in_checkpoint','loss_weight'},set(),'mtp_config')
        mtp.setdefault('layer_index_in_checkpoint',114)
        if type(mtp.get('depth',1)) is not int or mtp.get('depth',1) != 1: raise ValueError('Only one MTP layer is implemented')
        if g['num_heads']*g['head_dim_v'] != self.hidden_size or a['num_heads']*a['v_head_dim'] != self.hidden_size:
            raise ValueError('Mixer value widths must equal hidden_size')
        if a['qk_rope_head_dim'] % 2: raise ValueError('RoPE dimension must be even')
        if any(n % 16 for n in (self.hidden_size,a['q_lora_rank'],a['kv_lora_rank'],m['moe_intermediate_size'])):
            raise ValueError('Packed linear input widths must be divisible by 16')
        for name,value in [('gdn_config',g),('mla_config',a),('moe_config',m),('mtp_config',mtp)]:
            object.__setattr__(self,name,MappingProxyType(value))

    @property
    def num_moe_layers(self):
        return sum(1 for index in range(self.num_hidden_layers) if index not in self.dense_layer_indices)

    def to_dict(self):
        def plain(value):
            if isinstance(value, (dict, MappingProxyType)): return {k:plain(v) for k,v in value.items()}
            if isinstance(value, tuple): return list(value)
            return value
        return {item.name:plain(getattr(self,item.name)) for item in fields(self)}

    @classmethod
    def from_dict(cls, data):
        if type(data) is not dict or set(data)-{item.name for item in fields(cls)}:
            raise ValueError('Unknown configuration fields; remote/plugin metadata is not accepted')
        return cls(**data)
