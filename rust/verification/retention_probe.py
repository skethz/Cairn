"""Local administrative benchmark of the frozen Rust worker; no Python training.

Run only on a GPU released by its production worker after a saved checkpoint.
Model workers retain the existing offline launcher and network-denial policy.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path('/scratch/hongse/mythos/cairn-rust')
RELEASE = ROOT / 'releases/20261103-v3'
EVIDENCE = ROOT / 'evidence/retention_170gib_20261103'
GPUS = {'cairn': 4, 'gdn_h_recipe_proxy': 5,
        'kda_mla_operator': 6, 'gdn_gqa': 7,
        'dense_gqa': 5, 'full_mla': 7}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('family', choices=GPUS)
    p.add_argument('blocks', type=int, choices=range(25))
    p.add_argument('label')
    p.add_argument('--updates', type=int, default=10)
    p.add_argument('--save', action='store_true')
    a = p.parse_args()
    assert a.label and all(c.isalnum() or c in '_-' for c in a.label)
    assert 3 <= a.updates <= 30
    gpu = GPUS[a.family]
    config = RELEASE / 'configs' / (a.family + '_1p3b.json')
    c = read(config)
    production = ROOT / 'runs' / c['run_name']
    fresh = a.family in ('dense_gqa', 'full_mla')
    if fresh:
        assert not production.exists(), 'Queued model already has a production run'
    else:
        status = read(production / 'status.json')
        assert status['status'] == 'stopped_at_checkpoint', status['status']
        assert not (production / 'run.lock').exists(), 'Production worker still owns GPU'
    uuid = subprocess.check_output(
        ['/usr/bin/nvidia-smi', '--id', str(gpu), '--query-gpu=uuid',
         '--format=csv,noheader'], text=True).strip()
    occupants = subprocess.check_output(
        ['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
         '--format=csv,noheader'], text=True)
    assert not any(line.split(',')[0].strip() == uuid
                   for line in occupants.splitlines()), 'GPU is occupied'
    name = None
    state = {'step': 0, 'valid_tokens': 0, 'sha256': {}}
    if not fresh:
        pointer = read(production / 'latest.json')
        name = pointer['checkpoint']
        state = read(production / name / 'state.json')
        assert state['step'] == status['step']
    case = EVIDENCE / a.family / a.label
    case.mkdir(parents=True, exist_ok=False)
    if not fresh:
        checkpoint = case / name
        checkpoint.mkdir()
        for source in (production / name).iterdir():
            assert source.is_file() and not source.is_symlink()
            os.link(source, checkpoint / source.name)
        write(case / 'latest.json', pointer)
    command = [str(RELEASE / 'bin/cairn-run'), str(gpu), '--optimized',
               '--fused-scan', '--deterministic', '--retained-blocks', str(a.blocks),
               'train', str(config), str(case), '--profile',
               '--steps', str(state['step'] + a.updates)]
    if not fresh:
        command.append('--resume')
    if a.save:
        command.append('--save-profile')
    record = {'status': 'running', 'started_utc': now(), 'gpu': gpu,
              'family': a.family, 'retained_blocks': a.blocks,
              'source_checkpoint': name, 'source_valid_tokens': state['valid_tokens'],
              'source_hashes': state['sha256'], 'command': command,
              'worker_sha256': hashlib.sha256((RELEASE / 'bin/cairn').read_bytes()).hexdigest(),
              'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
              'production_training_unmodified': True,
              'fresh_queued_model_timing': fresh}
    write(case / 'probe.json', record)
    began = time.monotonic()
    samples = []
    with (case / 'console.log').open('w') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            inventory = subprocess.check_output(
                ['/usr/bin/nvidia-smi', '--id', str(gpu),
                 '--query-gpu=memory.used,memory.free,utilization.gpu',
                 '--format=csv,noheader,nounits'], text=True).strip()
            used, free, utilization = [float(x.strip()) for x in inventory.split(',')]
            samples.append({'elapsed_seconds': time.monotonic() - began,
                            'used_GiB': used / 1024, 'free_GiB': free / 1024,
                            'utilization_percent': utilization})
            time.sleep(1)
    events = ([json.loads(line) for line in (case / 'metrics.jsonl').read_text().splitlines()]
              if (case / 'metrics.jsonl').exists() else [])
    train = [e for e in events if e.get('event') == 'train']
    measured = train[2:]
    record.update({'status': 'passed' if process.returncode == 0 and len(train) == a.updates else 'failed',
                   'ended_utc': now(), 'returncode': process.returncode,
                   'wall_seconds': time.monotonic() - began,
                   'updates': len(train), 'warmup_updates': 2,
                   'measured_updates': len(measured),
                   'sampled_peak_device_GiB': max((s['used_GiB'] for s in samples), default=0),
                   'peak_allocated_GiB': max((e['peak_allocated_GiB'] for e in train), default=0),
                   'peak_reserved_GiB': max((e['peak_reserved_GiB'] for e in train), default=0),
                   'timing_scope': 'Same checkpoint/data/order, Rust worker only; initialization and save excluded; shared host and other GPUs'})
    if measured:
        record.update({'median_update_seconds': statistics.median(e['seconds'] for e in measured),
                       'measured_tokens': sum(e['batch_valid_tokens'] for e in measured),
                       'measured_seconds': sum(e['seconds'] for e in measured)})
        record['valid_tokens_per_second'] = record['measured_tokens'] / record['measured_seconds']
    write(case / 'device_memory_samples.json', samples)
    write(case / 'probe.json', record)
    print(json.dumps(record), flush=True)
    if record['status'] != 'passed':
        print((case / 'console.log').read_text()[-5000:], flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
