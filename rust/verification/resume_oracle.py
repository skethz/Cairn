"""Test-only continuation of our frozen Python training code, with final snapshots.

No code from the original external repository is imported. The production Rust
worker never calls this file. Dependencies must already be present locally.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument('--reference', required=True)
p.add_argument('--sites', required=True)
p.add_argument('--config', required=True)
p.add_argument('--run-dir', required=True)
p.add_argument('--updates', type=int, default=3)
p.add_argument('--trace', action='store_true')
p.add_argument('--deterministic', action='store_true')
p.add_argument('--no-snapshot', action='store_true')
args = p.parse_args()
reference = Path(args.reference).resolve(strict=True)
sys.path.insert(0, str(reference / 'src'))
from isolation import activate
assert os.environ.get('CUDA_VISIBLE_DEVICES') in {'4', '5', '6', '7'}
activate(local_gpu_initialization=True)
sys.path.extend(json.loads(args.sites))
import torch
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(args.deterministic)
torch.backends.cudnn.deterministic = args.deterministic
torch.backends.cudnn.benchmark = False
torch.cuda.init()
activate()
import trainer
from checkpoint import save
from safetensors.torch import save_file

directory = Path(args.run_dir)
pointer = json.loads((directory / 'latest.json').read_text())
initial = json.loads((directory / pointer['checkpoint'] / 'state.json').read_text())
target = initial['step'] + args.updates
captured = {}
original_optimizer = trainer.optimizer
original_status = trainer.status

def capture_optimizer(model, lr):
    result = original_optimizer(model, lr)
    captured.update(model=model, optimizer=result)
    return result

def capture_status(path, state):
    original_status(path, state)
    captured['state'] = state
    if args.trace and state.get('step') == target - 1 and state.get('status') == 'running':
        captured['profiler'] = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False)
        captured['profiler'].__enter__()
    if args.trace and state.get('step') == target and state.get('status') == 'running':
        prof = captured['profiler']
        prof.__exit__(None, None, None)
        prof.export_chrome_trace(str(directory / 'trace.json'))
        (directory / 'operator_profile.txt').write_text(
            prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=50))

trainer.optimizer = capture_optimizer
trainer.status = capture_status
trainer.run(reference, SimpleNamespace(config=args.config, run_dir=args.run_dir,
            command='profile', resume=True, steps=target),
            {'device': 'cuda:0', 'os_negative_controls': 'mandatory before torch import and after CUDA init'})
if not args.no_snapshot:
    save(directory, captured['model'], captured['optimizer'], captured['state'])
    save_file({n: t.grad.detach().cpu().contiguous() for n, t in captured['model'].named_parameters()
               if t.grad is not None}, str(directory / 'final_gradients.safetensors'))
print(json.dumps({'status': 'profile_complete' if args.no_snapshot else 'saved', 'step': target}), flush=True)
