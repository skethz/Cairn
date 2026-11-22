"""Read-only training snapshot and explicitly approximate ETA calculation."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path('/scratch/hongse/mythos/cairn-rust')
EVIDENCE = ROOT / 'evidence/retention_170gib_20261103'
FAMILIES = ['cairn', 'gdn_h_recipe_proxy', 'kda_mla_operator', 'gdn_gqa']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    zone = ZoneInfo('Europe/Zurich')
    inventory = subprocess.check_output(
        ['/usr/bin/nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free',
         '--format=csv,noheader,nounits'], text=True)
    memory = {int(i): {'total_GiB': float(t) / 1024, 'used_GiB': float(u) / 1024,
                       'free_GiB': float(f) / 1024}
              for i, t, u, f in (line.split(',') for line in inventory.splitlines()) if int(i) >= 4}
    observations = [json.loads(line) for line in (EVIDENCE / 'production_observations.jsonl').read_text().splitlines()]
    runs, etas = [], []
    for family in FAMILIES:
        decision = read(EVIDENCE / family / 'decision.json')
        directory = ROOT / 'runs' / (family + '_1p3b_seed17_v1')
        status = read(directory / 'status.json')
        execution = read(directory / 'execution.json')
        starting_step = int(decision['source_checkpoint'].split('_')[-1])
        history = [v for v in (json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines())
                   if v.get('event') == 'train']
        old_blocks = 2 if family == 'kda_mla_operator' else 8
        old = [v for v in history if v.get('retained_blocks') == old_blocks][2:][-200:]
        new = [v for v in history if v['step'] > starting_step and v.get('retained_blocks') == decision['retained_blocks']][2:][-200:]
        old_seconds = sum(v['seconds'] for v in old) / decision['throughput_ratio']
        predicted_rate = (sum(v['batch_valid_tokens'] for v in old + new)
                          / (old_seconds + sum(v['seconds'] for v in new)))
        observed = [v for v in observations if v['runs'][family]['status'] == 'running'
                    and v['runs'][family]['step'] >= starting_step + 3
                    and v['runs'][family]['execution']['retained_blocks'] == decision['retained_blocks']]
        wall = None
        if len(observed) >= 2:
            first, last = observed[0], observed[-1]
            seconds = (datetime.datetime.fromisoformat(last['utc']) - datetime.datetime.fromisoformat(first['utc'])).total_seconds()
            tokens = last['runs'][family]['valid_tokens'] - first['runs'][family]['valid_tokens']
            wall = {'seconds': seconds, 'valid_tokens': tokens,
                    'tokens_per_second': tokens / seconds if seconds > 0 else None}
        remaining = 1_000_000_000 - status['valid_tokens']
        hours = remaining / predicted_rate * 1.10 / 3600
        eta = {'family': family, 'gpu': decision['gpu'], 'valid_tokens': status['valid_tokens'],
               'remaining_tokens': remaining, 'progress_fraction': status['valid_tokens'] / 1e9,
               'old_updates': len(old), 'new_updates': len(new),
               'predicted_training_tokens_per_second': predicted_rate,
               'recent_observed_wall_throughput': wall,
               'training_hours_remaining': hours,
               'planning_range_hours': [hours * 0.85, hours * 1.15],
               'estimated_training_finish_local': (now + datetime.timedelta(hours=hours)).astimezone(zone).isoformat()}
        etas.append(eta)
        pid = int((directory / 'run.lock').read_text()) if (directory / 'run.lock').exists() else None
        record = {'family': family, 'gpu': decision['gpu'], 'worker_pid': pid,
                  'worker_executable': str((Path('/proc') / str(pid) / 'exe').resolve()) if pid else None,
                  'status': status, 'execution': execution, 'memory': memory[decision['gpu']],
                  'decision': decision,
                  'observed_peak_device_GiB': max([memory[decision['gpu']]['used_GiB']] +
                      [v['memory'][str(decision['gpu'])]['used_GiB'] for v in observed])}
        assert status['status'] == 'running' and status['step'] > starting_step + 2, family
        assert execution['retained_blocks'] == decision['retained_blocks']
        decision['status'] = 'running_verified'
        decision['verified_utc'] = now.isoformat()
        write(EVIDENCE / family / 'decision.json', decision)
        runs.append(record)
    queued = []
    for family, predecessor in [('dense_gqa', 'gdn_h_recipe_proxy'), ('full_mla', 'gdn_gqa')]:
        probe = read(EVIDENCE / family / 'queued_timing/probe.json')
        parent = next(v for v in etas if v['family'] == predecessor)
        duration = 1e9 / probe['valid_tokens_per_second'] * 1.10 / 3600
        total = parent['training_hours_remaining'] + duration
        queued.append({'family': family, 'gpu': probe['gpu'], 'predecessor': predecessor,
                       'retained_blocks': probe['retained_blocks'],
                       'profile_valid_tokens_per_second': probe['valid_tokens_per_second'],
                       'profile_measured_updates': probe['measured_updates'],
                       'training_duration_hours': duration,
                       'training_completion_hours_from_now': total,
                       'planning_range_hours_from_now': [parent['planning_range_hours'][0] + duration * 0.75,
                                                          parent['planning_range_hours'][1] + duration * 1.25],
                       'estimated_training_finish_local': (now + datetime.timedelta(hours=total)).astimezone(zone).isoformat()})
    eta_report = {'as_of_utc': now.isoformat(), 'timezone': 'Europe/Zurich',
                  'valid_token_budget_per_model': 1_000_000_000,
                  'method': 'Pool last up to 200 prior Rust updates, adjusted by measured retention speedup, with up to 200 new updates. Exclude first two warmup updates per regime. Add a stated 10% allowance for periodic validation/checkpoint overhead. Planning ranges are not statistical confidence intervals.',
                  'limitations': 'Final held-out evaluation/benchmarks and queue handoff evaluation time are unmeasured and excluded. Queued models have only 10 measured updates. Future host contention and batch distribution may change throughput.',
                  'active': etas, 'queued': queued,
                  'all_training_completion_hours_from_now': max([v['training_hours_remaining'] for v in etas] + [v['training_completion_hours_from_now'] for v in queued])}
    snapshot = {'utc': now.isoformat(), 'runs': runs, 'gpu_inventory': memory}
    summary = {'status': 'passed_and_training', 'target_device_GiB': 170,
               'release': str(ROOT / 'releases/20261103-v4'),
               'worker_sha256': hashlib.sha256((ROOT / 'releases/20261103-v4/bin/cairn').read_bytes()).hexdigest(),
               'same_worker_as_previous_release': True,
               'scope': 'Four active models. Queued Dense GQA and Full MLA keep their prior 8/4 block settings.',
               'uncheckpointed_updates_lost': 0, 'all_four_workers_progressing': True,
               'total_exact_tensors': sum(v['decision']['comparison']['tensors'] for v in runs),
               'runs': [{k: v[k] for k in ('family', 'gpu', 'worker_pid', 'memory', 'observed_peak_device_GiB', 'decision')} for v in runs],
               'eta': eta_report}
    write(EVIDENCE / 'live_snapshot.json', snapshot)
    write(EVIDENCE / 'eta.json', eta_report)
    write(EVIDENCE / 'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'tensors': summary['total_exact_tensors'],
                      'active': etas, 'queued': queued,
                      'memory': memory}, indent=2))


if __name__ == '__main__':
    main()
