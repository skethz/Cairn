"""Explicit synthetic test initialization, not pretrained weights or training."""
import torch
from torch import nn
from .layers import RMSNorm,NumbatRouter
from .quantization import NVFP4Linear


@torch.no_grad()
def fill_synthetic(module,*,seed=17,checkpoint_dtypes=False):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    for child in module.modules():
        if isinstance(child,NVFP4Linear):
            child.weight.copy_(torch.randint(0,256,child.weight.shape,generator=generator,dtype=torch.uint8))
            child.weight_scale.fill_(.5);child.weight_scale_2.fill_(.025);child.input_scale.fill_(.015625)
        elif isinstance(child,RMSNorm): child.weight.fill_(1.)
        elif isinstance(child,(nn.Linear,nn.Embedding,nn.Conv1d,NumbatRouter)):
            child.weight.copy_(torch.randn(child.weight.shape,generator=generator)*.03)
            if getattr(child,'bias',None) is not None: child.bias.zero_()
    if checkpoint_dtypes:
        for name,parameter in module.named_parameters():
            if parameter.is_floating_point() and not name.endswith(('A_log','dt_bias')):
                parameter.data=parameter.data.bfloat16()
    return module
