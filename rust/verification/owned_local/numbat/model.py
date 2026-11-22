"""Standalone nn.Module classes. No Transformers inheritance or remote APIs."""
from torch import nn
from .config import NumbatConfig
from .layers import RMSNorm,NumbatDecoderLayer,NumbatMTP
from .layout import constructor_bytes


class CausalLMOutputWithPast(dict):
    """Local logits container retaining the accessed output fields."""
    loss=None
    past_key_values=None
    hidden_states=None
    attentions=None

    def __init__(self,logits):
        super().__init__(logits=logits)

    @property
    def logits(self): return self['logits']

    def to_tuple(self): return tuple(self.values())

    def __getitem__(self,key):
        return super().__getitem__(key) if isinstance(key,str) else self.to_tuple()[key]


class NumbatPreTrainedModel(nn.Module):
    """Historical class name; only nn.Module behavior is inherited."""
    config_class=NumbatConfig
    base_model_prefix='model'

    def __init__(self,config,*,max_parameter_bytes=512*1024*1024):
        super().__init__()
        if type(config) is not NumbatConfig: raise TypeError('Expected the local NumbatConfig')
        if type(max_parameter_bytes) is not int or max_parameter_bytes<=0: raise ValueError('Invalid allocation budget')
        estimated=constructor_bytes(config)
        if estimated>max_parameter_bytes: raise ValueError(f'Model tensors need about {estimated} bytes; allocation budget is {max_parameter_bytes}')
        self.config=config

    def _init_weights(self,module):
        pass


class NumbatModel(NumbatPreTrainedModel):
    def __init__(self,config,*,max_parameter_bytes=512*1024*1024):
        super().__init__(config,max_parameter_bytes=max_parameter_bytes)
        self.config=config
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        self.layers=nn.ModuleList([NumbatDecoderLayer(config,index) for index in range(config.num_hidden_layers)]+[NumbatMTP(config)])
        self.norm=RMSNorm(config.hidden_size,config.rms_norm_eps)

    def forward(self,input_ids,position_ids=None,attention_mask=None):
        hidden=self.embed_tokens(input_ids)
        for index in range(self.config.num_hidden_layers):
            hidden=self.layers[index](hidden,position_ids,attention_mask)
        return hidden


class NumbatForCausalLM(NumbatPreTrainedModel):
    def __init__(self,config,*,max_parameter_bytes=512*1024*1024):
        super().__init__(config,max_parameter_bytes=max_parameter_bytes)
        self.config=config
        self.model=NumbatModel(config,max_parameter_bytes=max_parameter_bytes)
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)

    def forward(self,input_ids,position_ids=None,attention_mask=None):
        hidden=self.model(input_ids,position_ids,attention_mask)
        logits=self.lm_head(self.model.norm(hidden))
        return CausalLMOutputWithPast(logits=logits)

    def draft(self,input_ids,position_ids=None,attention_mask=None):
        hidden=self.model(input_ids,position_ids,attention_mask)
        next_embedding=self.model.embed_tokens(input_ids)[:,1:]
        hidden=hidden[:,:-1]
        prediction=self.model.layers[-1](hidden,next_embedding)
        return CausalLMOutputWithPast(logits=self.lm_head(self.model.norm(prediction)))
