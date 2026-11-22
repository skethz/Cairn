"""Offline test-only streaming comparison of complete continuation checkpoints."""
import argparse
import json
from pathlib import Path
import sys
p=argparse.ArgumentParser()
p.add_argument('--reference',required=True)
p.add_argument('--sites',required=True)
p.add_argument('--expected',required=True)
p.add_argument('--actual',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
sys.path.insert(0,str(Path(a.reference)/'src'))
from isolation import activate
activate()
sys.path.extend(json.loads(a.sites))
import torch
from safetensors import safe_open
torch.set_num_threads(8)
roots=[Path(a.expected),Path(a.actual)]
names=[json.loads((r/'latest.json').read_text())['checkpoint'] for r in roots]
assert names[0]==names[1]
records=[]
for file in ('model.safetensors','optimizer.safetensors','final_gradients.safetensors'):
    paths=[r/file if file.startswith('final') else r/n/file for r,n in zip(roots,names)]
    exact,total=0,0
    differences=[]
    with safe_open(str(paths[0]),framework='pt',device='cpu') as x, safe_open(str(paths[1]),framework='pt',device='cpu') as y:
        assert set(x.keys())==set(y.keys())
        for key in x.keys():
            left,right=x.get_tensor(key),y.get_tensor(key)
            assert left.shape==right.shape and left.dtype==right.dtype
            total+=1
            if torch.equal(left,right):
                exact+=1
            else:
                diff=(left.double()-right.double()).abs()
                differences.append({'tensor':key,'max_absolute':float(diff.max()),
                                    'relative_l2':float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(left.double()).clamp_min(1e-30)),
                                    'nonfinite':not bool(torch.isfinite(right).all())})
    records.append({'file':file,'tensors':total,'bitwise_equal':exact,'differences':differences})
def metrics(r):
    return {v['step']:v for v in (json.loads(l) for l in (r/'metrics.jsonl').read_text().splitlines()) if v.get('event')=='train'}
x,y=map(metrics,roots)
metric_diffs=[]
for step in x.keys() & y.keys():
    for key in ('valid_tokens','batch_valid_tokens','nll','gradient_norm','router_aux','lr','mean_expert_load'):
        if x[step][key]!=y[step][key]:
            metric_diffs.append({'step':step,'metric':key,'expected':x[step][key],'actual':y[step][key]})
result={'status':'passed' if all(r['tensors']==r['bitwise_equal'] for r in records) and not metric_diffs else 'different',
        'checkpoint':names[0],'records':records,'metric_differences':metric_diffs}
Path(a.output).write_text(json.dumps(result,indent=2))
print(json.dumps({'status':result['status'],'checkpoint':names[0],
                  'records':[{k:v for k,v in r.items() if k!='differences'} for r in records],
                  'metric_differences':len(metric_diffs)}))
