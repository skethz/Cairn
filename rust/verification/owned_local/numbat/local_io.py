"""Bounded regular-file JSON and safetensors I/O; no pickle or URI resolver."""
import json
import os
from pathlib import Path
import stat
from .config import NumbatConfig


def read_local_bytes(path,*,max_bytes):
    text=os.fspath(path)
    if not isinstance(text,str) or '://' in text: raise ValueError('Only a local file path is accepted')
    flags=os.O_RDONLY|os.O_NONBLOCK|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_CLOEXEC',0)
    descriptor=os.open(text,flags)
    with os.fdopen(descriptor,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>max_bytes: raise ValueError('Expected a bounded regular file')
        data=stream.read(max_bytes+1)
    if len(data)>max_bytes: raise ValueError('File grew beyond the size limit')
    return data


def load_config(path):
    return NumbatConfig.from_dict(json.loads(read_local_bytes(path,max_bytes=128*1024)))


def save_weights(model,path):
    from safetensors.torch import save_file
    destination=Path(path)
    if destination.suffix!='.safetensors' or destination.is_symlink(): raise ValueError('Expected a local .safetensors destination')
    save_file({name:tensor.detach().cpu().contiguous() for name,tensor in model.state_dict().items()},str(destination))


def load_weights(model,path,*,max_bytes=512*1024*1024):
    from safetensors.torch import load
    from .layout import tensor_layout
    if Path(path).suffix!='.safetensors': raise ValueError('Only safetensors are accepted')
    state=load(read_local_bytes(path,max_bytes=max_bytes))
    expected={name:(shape,dtype) for name,shape,dtype,_ in tensor_layout(model.config)}
    if set(state)!=set(expected): raise ValueError('Checkpoint tensor names do not match the architecture')
    for name,tensor in state.items():
        shape,dtype=expected[name]
        if tuple(tensor.shape)!=shape or str(tensor.dtype)!='torch.'+dtype:
            raise ValueError('Checkpoint tensor shape/dtype mismatch: '+name)
    model.load_state_dict(state,strict=True,assign=True)
    return model
