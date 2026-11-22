"""Test-only oracle: imports ONLY our frozen rewrite, never the suspect repository.

This is deliberately not part of the Rust runtime. Run with -I -S -B and explicit
local dependency paths. OS network denial is installed before importing torch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('command', choices=['generate', 'compare', 'packed-fixtures', 'packed-compare', 'extra-fixtures', 'extra-compare', 'full-fixture', 'full-compare', 'timing', 'smoke-compare'])
p.add_argument('--reference', required=True)
p.add_argument('--sites', required=True)
p.add_argument('--directory', required=True)
p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
p.add_argument('--actual-suffix', default='rust')
p.add_argument('--length', type=int, default=17)
p.add_argument('--family', choices=['cairn', 'gdn_h_recipe_proxy', 'kda_mla_operator', 'gdn_gqa', 'full_mla', 'dense_gqa'])
args = p.parse_args()
reference = Path(args.reference).resolve(strict=True)
assert (reference / 'src/model.py').is_file()
sys.path.insert(0, str(reference / 'src'))
from isolation import activate
activate(local_gpu_initialization=args.device == 'cuda')
sys.path.extend(json.loads(args.sites))
import torch
from safetensors.torch import save_file, load_file
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if args.device == 'cuda':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in {'4', '5', '6', '7'}
    torch.cuda.init()
activate()
from model import LanguageModel, initialize_paired, optimizer
from verification import tiny

root = Path(args.directory)
families = ('cairn', 'gdn_h_recipe_proxy', 'kda_mla_operator', 'gdn_gqa', 'full_mla', 'dense_gqa')
if args.family:
    families = (args.family,)
case_suffix = '' if args.length == 17 else f'_t{args.length}'

def snapshot(x):
    return x.detach().cpu().contiguous().clone()

if args.command == 'smoke-compare':
    expected_dir = root.parent / 'smoke_python'
    actual_dir = root.parent / ('smoke_' + args.actual_suffix)
    checks = []
    for step in (1, 2):
        folder = f'checkpoint_{step:08d}'
        for name in ('model', 'optimizer'):
            expected = load_file(str(expected_dir / folder / (name + '.safetensors')))
            actual = load_file(str(actual_dir / folder / (name + '.safetensors')))
            assert set(expected) == set(actual)
            unequal = [n for n in expected if not torch.equal(expected[n], actual[n])]
            checks.append(dict(step=step, file=name, tensors=len(expected), unequal=unequal))
    a = [json.loads(x) for x in (expected_dir / 'metrics.jsonl').read_text().splitlines()]
    b = [json.loads(x) for x in (actual_dir / 'metrics.jsonl').read_text().splitlines()]
    a = [x for x in a if x.get('event') == 'train']
    b = [x for x in b if x.get('event') == 'train']
    expected_by_step = {x['step']: x for x in a}
    for y in b:
        x = expected_by_step[y['step']]
        for field in ('step', 'valid_tokens', 'batch_valid_tokens', 'nll', 'router_aux', 'lr', 'gradient_norm', 'mean_expert_load'):
            assert x[field] == y[field], (field, x[field], y[field])
    passed = all(not c['unequal'] for c in checks)
    row = dict(status='passed' if passed else 'failed', checkpoints=checks, training_metrics_equal=True, valid_tokens=400,
               coverage='real corpus slice, BF16, two microbatches, partial final-token budget, validation/test, model+optimizer+CPU/CUDA RNG checkpoints')
    report_name = 'cross_resume_comparison.json' if 'cross_resume' in args.actual_suffix else 'smoke_comparison.json'
    (root / report_name).write_text(json.dumps(row, indent=2))
    print(json.dumps(row))
    raise SystemExit(0 if passed else 1)

if args.command == 'timing':
    import time
    import statistics
    model = LanguageModel(tiny())
    initialize_paired(model, 17)
    model.to(args.device).train()
    opt = optimizer(model, 4e-4)
    ids = ((torch.arange(256).reshape(2, 128) * 29 + 11) % 256 + 3).to(args.device)
    labels = (ids * 11 + 7) % 259
    seconds, losses = [], []
    for i in range(12):
        if args.device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == 'cuda'):
            loss, count, aux, _ = model(ids, labels != -100, labels)
            objective = loss / count + .01 * aux
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if args.device == 'cuda':
            torch.cuda.synchronize()
        if i >= 2:
            seconds.append(time.perf_counter() - started)
            losses.append(float(loss.detach()) / 256.)
    row = dict(implementation='Python frozen reference + native LibTorch 2.11.0', device=args.device, threads=8,
               parameters=sum(p.numel() for p in model.parameters()), shape=[2, 128], warmup_updates=2, measured_updates=10,
               seconds=seconds, median_seconds=statistics.median(seconds), median_tokens_per_second=256 / statistics.median(seconds),
               nll=losses, scope='small-model pilot on shared host; not an isolated 1.3B GPU speed claim')
    (root / f'timing_python_{args.device}.json').write_text(json.dumps(row, indent=2))
    print(json.dumps(row))
    raise SystemExit(0)

if args.command.startswith('extra-'):
    directory = root / 'protocol'
    data = Path('/scratch/hongse/mythos/cairn-training/data')
    if args.command == 'extra-fixtures':
        directory.mkdir(parents=True, exist_ok=True)
        from data import Documents
        from tokenizers import Tokenizer, models, pre_tokenizers
        from benchmarks import choice_tensors
        sys.path.insert(0, str(Path(__file__).parent / 'owned_local'))
        from numbat.tokenizer import ByteBPETokenizer
        texts = ['Hello Cairn! Hello Cairn!', '한국어 토크나이저를 검증합니다.', 'a a a b b b', 'naïve café — Zürich', '🤖🪨\n\t<bos> & <eos>', '', 'abcabcabcabc', 'Μάϊος Σ ΟΣ ß İ']
        batches = [dict(seed=17, context=128, microbatch=4, cursor=0), dict(seed=0, context=512, microbatch=2, cursor=13), dict(seed=91, context=4096, microbatch=2, cursor=100)]
        result, cursors = {}, []
        for i, c in enumerate(batches):
            d = Documents(data, 'train', c['seed'], c['context'])
            d.cursor = c['cursor']
            for j in range(2):
                ids, labels = d.batch(c['microbatch'], 'cpu')
                result[f'ids::{i}::{j}'] = ids
                result[f'labels::{i}::{j}'] = labels
            cursors.append(d.cursor)
        token = Tokenizer.from_file(str(data / 'tokenizer.json'))
        byte = ByteBPETokenizer.train(texts, vocabulary_size=300, minimum_frequency=2)
        ids = [byte.encode(s, bos=True, eos=True) for s in texts]
        meta = dict(batches=batches, texts=texts, choice=dict(prompt='a a a', choices=[' b', ' b c']))
        (directory / 'protocol_case.json').write_text(json.dumps(meta, ensure_ascii=False))
        expected = dict(cursors=cursors, token_ids=[token.encode(s, add_special_tokens=False).ids for s in texts], byte_merges=byte.merges, byte_ids=ids, decoded=[byte.decode(x) for x in ids])
        (directory / 'expected.json').write_text(json.dumps(expected, ensure_ascii=False))
        word = Tokenizer(models.WordLevel({'<unk>': 0, '<bos>': 1, 'a': 2, 'b': 3, 'c': 4}, unk_token='<unk>'))
        word.pre_tokenizer = pre_tokenizers.Whitespace()
        word.save(str(directory / 'choice_tokenizer.json'))
        for i, (ids, labels, _) in enumerate(choice_tensors(word, meta['choice'], 4, 'cpu')):
            result[f'choice_ids::{i}'] = ids
            result[f'choice_labels::{i}'] = labels
        save_file(result, str(directory / 'expected.safetensors'))
        from recurrence import recurrent_reference
        torch.manual_seed(123)
        expected = {}
        for channels in (1, 4):
            xs = [torch.nn.functional.normalize(torch.randn(2, 2, 11, 4, dtype=torch.float64), dim=-1),
                  torch.nn.functional.normalize(torch.randn(2, 2, 11, 4, dtype=torch.float64), dim=-1),
                  torch.randn(2, 2, 11, 5, dtype=torch.float64), -torch.rand(2, 2, 11, channels, dtype=torch.float64),
                  torch.rand(2, 2, 11, dtype=torch.float64), torch.randn(2, 2, 4, 5, dtype=torch.float64)]
            save_file(dict(zip(('q', 'k', 'v', 'decay', 'beta', 'initial'), xs)), str(directory / f'recurrence_{channels}.safetensors'))
            xs = [x.to(args.device).requires_grad_() for x in xs]
            out, state = recurrent_reference(*xs[:5], xs[5])
            gradients = torch.autograd.grad(out.square().sum() + state.square().sum(), xs)
            for chunk in (1, 4, 8, 16):
                expected[f'out::{channels}::{chunk}'] = snapshot(out)
                expected[f'state::{channels}::{chunk}'] = snapshot(state)
                for i, g in enumerate(gradients):
                    expected[f'grad::{channels}::{chunk}::{i}'] = snapshot(g)
        save_file(expected, str(directory / 'recurrence_expected.safetensors'))
        print('Data/tokenizer/candidate-mask and sequential delta-rule fixtures generated', flush=True)
    else:
        assert json.loads((directory / 'expected.json').read_text()) == json.loads((directory / 'rust/actual.json').read_text())
        expected = load_file(str(directory / 'expected.safetensors'))
        actual = load_file(str(directory / 'rust/actual.safetensors'))
        assert set(expected) == set(actual)
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
        actual = load_file(str(directory / f'recurrence_actual_{args.device}.safetensors'))
        expected = load_file(str(directory / 'recurrence_expected.safetensors'))
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=1e-7, atol=1e-8)
        row = dict(status='passed', device=args.device, dataset_batches=6, unicode_tokenizer_texts=8, byte_bpe_training_and_roundtrip=True, continuation_boundary=True, recurrence_tensors=len(expected), recurrence_reference='sequential update; scalar/channel decay; nonzero initial state; lengths not divisible by chunk; all input gradients')
        (directory / f'comparison_{args.device}.json').write_text(json.dumps(row, indent=2))
        print(json.dumps(row))
    raise SystemExit(0)

if args.command.startswith('full-'):
    directory = root / 'fullscale'
    if args.command == 'full-fixture':
        import shutil
        directory.mkdir(parents=True, exist_ok=True)
        config_path = reference / 'configs/cairn_1p3b.json'
        config = json.loads(config_path.read_text())
        run = reference.parents[1] / 'runs' / config['run_name']
        pointer = json.loads((run / 'latest.json').read_text())
        name = pointer['checkpoint']
        assert name.startswith('checkpoint_') and name[11:].isdigit()
        source = run / name / 'model.safetensors'
        shutil.copyfile(source, directory / 'weights.safetensors')
        shutil.copyfile(config_path, directory / 'config.json')
        model = LanguageModel(config['model'])
        model.load_state_dict(load_file(str(directory / 'weights.safetensors')), strict=True)
        model.to(args.device).train()
        ids = ((torch.arange(65).reshape(1, 65) * 37 + 11) % 31997 + 3).to(args.device)
        labels = (ids * 13 + 17) % 32000
        labels[:, -9:] = -100
        save_file({'ids': snapshot(ids), 'labels': snapshot(labels)}, str(directory / 'inputs.safetensors'))
        with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == 'cuda'):
            logits, aux, load = model(ids, labels != -100)
        result = dict(logits=snapshot(logits), aux=snapshot(aux), load=snapshot(load))
        with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == 'cuda'):
            loss, count, aux, _ = model(ids, labels != -100, labels)
            objective = loss / count + .01 * aux
        result['loss'] = torch.tensor([float(loss.detach()), float(aux.detach())], dtype=torch.float64)
        objective.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                result['grad::' + n] = snapshot(p.grad)
        save_file(result, str(directory / 'expected.safetensors'))
        metadata = dict(source_checkpoint=str(source), parameters=sum(p.numel() for p in model.parameters()), tensors=len(result), shape=list(ids.shape), source='Our independently trained Cairn checkpoint; no original repository weights')
        (directory / 'case.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)
    else:
        expected = load_file(str(directory / 'expected.safetensors'))
        actual = load_file(str(directory / 'actual.safetensors'))
        assert set(expected) == set(actual)
        exact, errors, failures = 0, [], []
        for n, x in expected.items():
            y = actual[n]
            if torch.equal(x, y):
                exact += 1
                continue
            errors.append(dict(tensor=n, maximum_absolute_error=float((x.double() - y.double()).abs().max())))
            try:
                torch.testing.assert_close(x, y, rtol=2e-4, atol=2e-6)
            except AssertionError:
                failures.append(n)
        row = dict(status='passed' if not failures else 'failed', tensors=len(expected), bitwise_equal=exact, failures=failures, largest_differences=sorted(errors, key=lambda x: x['maximum_absolute_error'], reverse=True)[:20])
        (directory / 'comparison.json').write_text(json.dumps(row, indent=2))
        print(json.dumps(row))
        if failures:
            raise SystemExit(1)
    raise SystemExit(0)

if args.command.startswith('packed-'):
    local = Path(__file__).parent / 'owned_local'
    sys.path.insert(0, str(local))
    from numbat.config import NumbatConfig
    from numbat.model import NumbatForCausalLM
    from numbat.diagnostics import fill_synthetic
    from numbat.quantization import NVFP4Linear
    reports = []
    for i, scale in enumerate((.015625, 0., -.015625)):
        directory = root / f'packed_{args.device}_{i}'
        if args.command == 'packed-fixtures':
            directory.mkdir(parents=True, exist_ok=True)
            c = json.loads((local / 'tiny.json').read_text())
            if i == 1:
                c['gdn_config']['qk_l2norm'] = False
                c['gdn_config']['conv_bias'] = True
                c['hidden_act'] = 'gelu'
            if i == 2:
                c['hidden_act'] = 'relu'
                c['moe_config']['norm_topk_prob'] = False
            model = fill_synthetic(NumbatForCausalLM(NumbatConfig.from_dict(c)), checkpoint_dtypes=True).eval()
            for layer in model.modules():
                if isinstance(layer, NVFP4Linear):
                    layer.input_scale.fill_(scale)
            save_file({n: snapshot(t) for n, t in model.state_dict().items()}, str(directory / 'weights.safetensors'))
            (directory / 'config.json').write_text(json.dumps(c, indent=2))
            model.to(args.device)
            ids = torch.tensor([[1, 5, 3, 20, 7, 11, 2], [1, 9, 22, 7, 11, 0, 0]], device=args.device)
            inputs = {'ids': snapshot(ids)}
            kwargs = {}
            if i != 1:
                kwargs['attention_mask'] = torch.tensor([[1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]], device=args.device)
                kwargs['position_ids'] = torch.arange(13, 20, device=args.device).expand_as(ids)
                inputs['mask'] = snapshot(kwargs['attention_mask'])
                inputs['positions'] = snapshot(kwargs['position_ids'])
            save_file(inputs, str(directory / 'inputs.safetensors'))
            with torch.no_grad():
                save_file({'logits': snapshot(model(ids, **kwargs).logits)}, str(directory / 'expected.safetensors'))
                save_file({'logits': snapshot(model.draft(ids, **kwargs).logits)}, str(directory / 'expected_draft.safetensors'))
            print(json.dumps({'generated': str(directory)}), flush=True)
        else:
            for suffix in ('', '_draft'):
                a = load_file(str(directory / f'actual{suffix}.safetensors'))['logits']
                b = load_file(str(directory / f'expected{suffix}.safetensors'))['logits']
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                reports.append({'case': directory.name + suffix, 'bitwise_equal': torch.equal(a, b), 'elements': a.numel()})
    if reports:
        (root / f'packed_comparison_{args.device}.json').write_text(json.dumps(reports, indent=2))
        print(json.dumps(reports))
    raise SystemExit(0)

if args.command == 'generate':
    for family in families:
        directory = root / f'{args.device}_{family}{case_suffix}'
        directory.mkdir(parents=True, exist_ok=True)
        c = tiny(family)
        model = LanguageModel(c)
        initialize_paired(model, 17)
        initialized = {f'initialized::{n}': snapshot(t) for n, t in model.named_parameters()}
        save_file({n: snapshot(t) for n, t in model.state_dict().items()}, str(directory / 'weights.safetensors'))
        model.to(args.device)
        ids = ((torch.arange(2 * args.length).reshape(2, args.length) * 29 + 11) % 256 + 3).to(args.device)
        labels = (ids * 11 + 7) % 259
        labels[0, 9:] = -100
        labels[1, -2:] = -100
        ids[0, 12:] = 0
        save_file({'ids': snapshot(ids), 'labels': snapshot(labels)}, str(directory / 'inputs.safetensors'))
        bf16 = args.device == 'cuda'
        result = initialized
        with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16, enabled=bf16):
            hidden = model.embedding(ids)
            result['embedding'] = snapshot(hidden)
            for i, block in enumerate(model.blocks):
                hidden, aux, load = block(hidden, labels != -100)
                result[f'block::{i}'] = snapshot(hidden)
                result[f'aux::{i}'] = snapshot(aux)
                result[f'load::{i}'] = snapshot(load)
            logits, aux, load = model(ids, labels != -100)
            result.update(logits=snapshot(logits), aux=snapshot(aux), load=snapshot(load))
        opt = optimizer(model, 4e-4)
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=bf16):
                loss, count, aux, _ = model(ids, labels != -100, labels)
                objective = loss / count + aux * .01
            result[f'loss::{step}'] = torch.tensor([float(loss.detach()), float(aux.detach())], dtype=torch.float64)
            objective.backward()
            for n, t in model.named_parameters():
                if t.grad is not None:
                    result[f'grad::{step}::{n}'] = snapshot(t.grad)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            result[f'gradient_norm::{step}'] = torch.tensor([float(norm)], dtype=torch.float64)
            opt.step()
            for n, t in model.named_parameters():
                result[f'updated::{step}::{n}'] = snapshot(t)
                result[f'mean::{step}::{n}'] = snapshot(opt.state[t]['exp_avg'])
                result[f'variance::{step}::{n}'] = snapshot(opt.state[t]['exp_avg_sq'])
        save_file(result, str(directory / 'expected.safetensors'))
        meta = {'model': c, 'bf16': bf16, 'torch': torch.__version__, 'reference': str(reference),
                'source_sha256': {n: hashlib.sha256((reference / 'src' / n).read_bytes()).hexdigest()
                                  for n in ('model.py', 'recurrence.py')},
                'scope': 'owned frozen Python rewrite; initialized weights, blocks, logits, router, all gradients, three fused AdamW updates'}
        (directory / 'case.json').write_text(json.dumps(meta, indent=2))
        print(json.dumps({'generated': str(directory), 'tensors': len(result)}), flush=True)
else:
    report = []
    for family in families:
        directory = root / f'{args.device}_{family}{case_suffix}'
        expected = load_file(str(directory / 'expected.safetensors'))
        actual = load_file(str(directory / args.actual_suffix / 'actual.safetensors'))
        assert set(expected) == set(actual), (set(expected) - set(actual), set(actual) - set(expected))
        differences, bitwise, failures = [], 0, []
        for name, x in expected.items():
            y = actual[name]
            if torch.equal(x, y):
                bitwise += 1
                continue
            assert x.shape == y.shape and x.dtype == y.dtype, name
            error = float((x.double() - y.double()).abs().max())
            differences.append({'tensor': name, 'max_absolute_error': error})
            # Fixed in advance: FP32 tolerances for different autograd traversal;
            # BF16 is reported separately. Exact initialized weights always required.
            rtol, atol = ((2e-4, 2e-6) if args.device == 'cpu' else (2e-2, 2e-4))
            if name.startswith('initialized::'):
                rtol = atol = 0
            try:
                torch.testing.assert_close(x, y, rtol=rtol, atol=atol)
            except AssertionError:
                failures.append(name)
        row = {'case': directory.name, 'tensors': len(expected), 'bitwise_equal': bitwise,
               'different': len(differences), 'failures': failures,
               'forward_differences': [d for d in differences if d['tensor'].startswith(('block::','aux::','load::','logits','embedding','loss::'))],
               'largest_differences': sorted(differences, key=lambda x: x['max_absolute_error'], reverse=True)[:15]}
        report.append(row)
        print(json.dumps(row), flush=True)
    (root / f'comparison_{args.device}_{args.actual_suffix}{case_suffix}.json').write_text(json.dumps(report, indent=2))
    if any(row['failures'] for row in report):
        raise SystemExit(1)
