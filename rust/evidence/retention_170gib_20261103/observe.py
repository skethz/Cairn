import pathlib,time,json,datetime,subprocess
r=pathlib.Path('/scratch/hongse/mythos/cairn-rust');e=r/'evidence/retention_170gib_20261103'
families=['cairn','gdn_h_recipe_proxy','kda_mla_operator','gdn_gqa']
with (e/'production_observations.jsonl').open('x') as log:
 for _ in range(61):
  runs={}
  for family in families:
   d=r/'runs'/(family+'_1p3b_seed17_v1');s=json.loads((d/'status.json').read_text())
   runs[family]={k:s.get(k) for k in ('status','step','valid_tokens','data_cursor')}
   runs[family]['last_step']=s.get('last_step')
   runs[family]['execution']=json.loads((d/'execution.json').read_text())
  gpu=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
  memory={int(i):{'used_GiB':float(u)/1024,'free_GiB':float(f)/1024} for i,u,f in (line.split(',') for line in gpu.splitlines()) if int(i)>=4}
  log.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':runs,'memory':memory})+'\n');log.flush()
  time.sleep(5)
