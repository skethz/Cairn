"""Local E2M1 unpacking and stored-scale NVFP4 reference arithmetic."""
import torch
from torch import nn
from torch.nn import functional as F

E2M1_LUT=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],dtype=torch.float32)
E2M1_BOUNDS=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.])
E2M1_BOUNDS[[1,3,5]]=torch.nextafter(E2M1_BOUNDS[[1,3,5]],torch.zeros(3))


def unpack_e2m1(packed):
    high=((packed>>4)&15).long()
    low=(packed&15).long()
    sign=(torch.stack([low,high],dim=-1)&8)!=0
    codes=(torch.stack([low,high],dim=-1)&7).long()
    values=E2M1_LUT.to(packed.device)[codes]*torch.where(sign,-1.,1.)
    return values.flatten(-2)


class NVFP4Linear(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        if in_features % 16: raise ValueError('NVFP4 input width must be divisible by 16')
        self.weight=nn.Parameter(torch.zeros(out_features,in_features//2,dtype=torch.uint8),requires_grad=False)
        self.register_buffer('weight_scale',torch.zeros(out_features,in_features//16,dtype=torch.float8_e4m3fn))
        self.register_buffer('weight_scale_2',torch.zeros((),dtype=torch.float32))
        self.register_buffer('input_scale',torch.zeros((),dtype=torch.float32))
        self.out_features=out_features
        self.in_features=in_features

    def dequantize(self):
        weight=unpack_e2m1(self.weight).float()
        weight=weight.view(self.out_features,-1,16)*self.weight_scale.to(torch.float32).unsqueeze(-1)
        return (weight.view(self.out_features,-1)*self.weight_scale_2).to(torch.bfloat16)

    def forward(self,x):
        weight=self.dequantize()
        if self.input_scale:
            shape=x.shape
            x=x.to(torch.float32)/self.input_scale.to(torch.float32)
            blocks=x.reshape(*x.shape[:-1],-1,16)
            scale=(blocks.abs().amax(-1,keepdim=True)/6.).clamp(min=1e-8)
            scale=scale.to(torch.float8_e4m3fn).to(torch.float32).clamp(min=1e-8)
            codes=torch.bucketize(blocks.abs()/scale,E2M1_BOUNDS.to(blocks.device))
            x=E2M1_LUT.to(codes.device)[codes]*torch.sign(blocks)*scale
            x=x.to(weight.dtype).reshape(shape)
            return F.linear(x,weight)*self.input_scale.to(weight.dtype)
        return F.linear(x.to(weight.dtype),weight)
