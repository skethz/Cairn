"""Compare Rust retention probes using saved hashes and full gradient bytes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def count(path):
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        assert size < 16 * 1024 * 1024
        return len([k for k in json.loads(f.read(size)) if k != '__metadata__'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    roots = [a.baseline, a.candidate]
    names = [read(r / 'latest.json')['checkpoint'] for r in roots]
    assert names[0] == names[1]
    states = [read(r / n / 'state.json') for r, n in zip(roots, names)]
    differences = []
    for key in ('step', 'valid_tokens', 'data_cursor', 'config_sha256', 'data_sha256'):
        if states[0][key] != states[1][key]:
            differences.append({'state': key})
    files = []
    for name in ('model.safetensors', 'optimizer.safetensors', 'final_gradients.safetensors'):
        paths = [r / name if name.startswith('final') else r / n / name
                 for r, n in zip(roots, names)]
        hashes = ([sha(path) for path in paths] if name.startswith('final') else
                  [s['sha256'][name] for s in states])
        counts = [count(path) for path in paths]
        equal = hashes[0] == hashes[1] and counts[0] == counts[1]
        files.append({'file': name, 'tensors': counts[0], 'byte_identical': equal,
                      'hashes': hashes,
                      'hash_source': 'computed_now' if name.startswith('final') else 'frozen_worker_checkpoint_save'})
    events = [[v for v in (json.loads(line) for line in (r / 'metrics.jsonl').read_text().splitlines())
               if v.get('event') == 'train'] for r in roots]
    assert len(events[0]) == len(events[1]) and len(events[0]) > 0
    for x, y in zip(*events):
        for key in ('step', 'valid_tokens', 'batch_valid_tokens', 'nll', 'gradient_norm',
                    'router_aux', 'lr', 'mean_expert_load'):
            if x[key] != y[key]:
                differences.append({'step': x['step'], 'metric': key})
        assert all(math.isfinite(x[k]) and math.isfinite(y[k])
                   for k in ('nll', 'gradient_norm', 'router_aux', 'lr'))
    result = {'status': 'passed' if not differences and all(f['byte_identical'] for f in files) else 'different',
              'checkpoint': names[0], 'updates': len(events[0]), 'files': files,
              'tensors': sum(f['tensors'] for f in files), 'differences': differences}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
