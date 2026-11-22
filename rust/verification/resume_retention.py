"""Resume a checkpointed run only after its retention comparison passes."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/scratch/hongse/mythos/cairn-rust')
EVIDENCE = ROOT / 'evidence/retention_170gib_20261103'
SETTINGS = {'cairn': (4, 14, None), 'gdn_h_recipe_proxy': (5, 14, 'dense_gqa'),
            'kda_mla_operator': (6, 5, None), 'gdn_gqa': (7, 14, 'full_mla')}


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('family', choices=SETTINGS)
    a = p.parse_args()
    gpu, blocks, queued = SETTINGS[a.family]
    folder = EVIDENCE / a.family
    candidate = 'retain' + str(blocks)
    comparison = read(folder / (candidate + '_comparison.json'))
    assert comparison['status'] == 'passed'
    cases = ['baseline_a', candidate, 'candidate_b', 'baseline_b']
    probes = [read(folder / case / 'probe.json') for case in cases]
    assert all(p['status'] == 'passed' and p['updates'] == 10 for p in probes)
    assert len({p['source_checkpoint'] for p in probes}) == 1
    metric_fields = ('step', 'valid_tokens', 'batch_valid_tokens', 'nll',
                     'gradient_norm', 'router_aux', 'lr', 'mean_expert_load')
    reference = None
    for case in cases:
        metrics = [v for v in (json.loads(line) for line in (folder / case / 'metrics.jsonl').read_text().splitlines())
                   if v.get('event') == 'train']
        reduced = [{k: v[k] for k in metric_fields} for v in metrics]
        if reference is None:
            reference = reduced
        assert reduced == reference, 'Repeated run changed numerical results'
    baseline, optimized = [probes[0], probes[3]], [probes[1], probes[2]]
    rate = lambda values: sum(v['measured_tokens'] for v in values) / sum(v['measured_seconds'] for v in values)
    gain = rate(optimized) / rate(baseline)
    peak = max(max(v['sampled_peak_device_GiB'], v['peak_reserved_GiB'] + 1.0) for v in optimized)
    assert peak <= 170.0, ('Memory exceeds tested target', peak)
    assert gain > 1.0, ('No speed improvement', gain)
    if queued:
        assert read(EVIDENCE / queued / 'queued_timing/probe.json')['status'] == 'passed'
    release = ROOT / 'releases/20261103-v4'
    assert hashlib.sha256((release / 'bin/cairn').read_bytes()).hexdigest() == probes[0]['worker_sha256']
    config = release / 'configs' / (a.family + '_1p3b.json')
    assert hashlib.sha256(config.read_bytes()).hexdigest() == probes[0]['config_sha256']
    production = ROOT / 'runs' / read(config)['run_name']
    status = read(production / 'status.json')
    assert status['status'] == 'stopped_at_checkpoint'
    assert not (production / 'run.lock').exists()
    assert not (ROOT / 'runs' / ('queue_gpu' + str(gpu) + '.lock')).exists()
    assert read(production / 'latest.json')['checkpoint'] == probes[0]['source_checkpoint']
    request = production / 'stop.request'
    assert request.read_text().startswith('Authorized retention-budget benchmark:')
    destination = folder / 'stop.request.archived'
    assert not destination.exists()
    request.rename(destination)
    command = [str(release / 'bin/cairn-control'), 'queue', str(ROOT / 'runs'), str(gpu),
               '--resume', '--optimized', '--fused-scan', '--deterministic',
               '--retained-blocks', str(blocks), str(config)]
    if queued:
        command.append(str(release / 'configs' / (queued + '_1p3b.json')))
    decision = {'family': a.family, 'gpu': gpu, 'retained_blocks': blocks,
                'target_device_GiB': 170, 'conservative_peak_GiB': peak,
                'baseline_tokens_per_second': rate(baseline),
                'optimized_tokens_per_second': rate(optimized), 'throughput_ratio': gain,
                'measured_updates_per_setting': 16,
                'comparison': comparison, 'all_40_update_metrics_identical': True,
                'source_checkpoint': probes[0]['source_checkpoint'],
                'source_valid_tokens': probes[0]['source_valid_tokens'],
                'uncheckpointed_updates_lost': 0, 'command': command,
                'resumed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'status': 'launch_requested'}
    with (folder / 'resume_console.log').open('w') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    decision['controller_pid'] = process.pid
    (folder / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps({k: v for k, v in decision.items() if k != 'comparison'}), flush=True)


if __name__ == '__main__':
    main()
