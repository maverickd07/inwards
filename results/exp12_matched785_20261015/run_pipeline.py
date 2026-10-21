"""Twelve sequential GPU fits, CPU audit in parallel; reuse all source predictions."""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
from reuse import HERE,read,key,js,sha,verify_source,verify_environment
from protocol import verify_freeze
from engine_io import write

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--model',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke-test',action='store_true');a=ap.parse_args()
 a.source=a.source.resolve();a.out=a.out.resolve()
 if a.out.exists():raise FileExistsError('No overwrite: '+str(a.out))
 verify_freeze();hashes=verify_source(a.source)
 if not a.smoke_test:verify_environment(a.source)
 a.out.mkdir(parents=True);start=time.time();snapshot=a.out/'source_700454'
 for name,digest in hashes.items():
  dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.source/name,dest);assert sha(dest)==digest
 inputs=a.out/'inputs';shutil.copytree(HERE/'frozen_inputs',inputs)
 for name in ['ALLOCATION_FREEZE.json','DESIGN_FREEZE.json','PROTOCOL.txt','training_label_policy.json','training_label_scorer.py','scorer_v3.py','SCORER_FREEZE.json','aliases.json','reference_catalog.json']:
  shutil.copyfile(HERE/name,inputs/name)
 (inputs/'context_s0').mkdir();shutil.copyfile(snapshot/'context_s0/runtime.json',inputs/'context_s0/runtime.json')
 base=read(snapshot/'baseline/eval_D_reservedC.jsonl')
 if a.smoke_test:
  # Full 785 examples and 198 steps; only evaluation reduced to two fixed-primary items.
  mask=json.loads((inputs/'primary_D303_keys.json').read_text());mask['identities']=mask['identities'][:2];mask['n']=2
  keep=set(map(tuple,mask['identities']));(inputs/'primary_D303_keys.json').write_text(json.dumps(mask))
  for path in [inputs/'plan/items_D_reservedC.jsonl']+list((inputs/'reference_v3').glob('*.jsonl')):
   rows=[r for r in read(path) if key(r) in keep];path.unlink();write(path,rows)
  base=[r for r in base if key(r) in keep]
 (a.out/'baseline').mkdir();write(a.out/'baseline/eval_D_reservedC.jsonl',base)
 js(a.out/'reuse_receipt.json',{'source':str(a.source),'source_job':700454,'source_hashes':hashes,'smoke_test':a.smoke_test,'baseline_regenerated':False,'targets_regenerated':False,'training_items':785,'smoke_evaluation_only_reduced':a.smoke_test})
 js(a.out/'source_hashes.json',{p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.json','.txt')})
 audit_cmd=[sys.executable,str(HERE/'audit_training_labels.py'),'--inputs',str(inputs),'--source',str(snapshot),'--out',str(a.out/'training_label_audit')]
 commands=[]
 with (a.out/'training_label_audit.log').open('x') as log:
  audit=subprocess.Popen(audit_cmd,stdout=log,stderr=subprocess.STDOUT)
  try:
   for seed in range(6):
    for c in ['context','direct']:
     cmd=[sys.executable,'-u',str(HERE/'train_seed.py'),'--source',str(inputs),'--model',a.model,'--out',str(a.out/f'{c}_s{seed}'),'--train-seed',str(seed),'--condition',c]
     if a.smoke_test:cmd.append('--smoke-test')
     commands.append(cmd);js(a.out/f'command_{c}_s{seed}.json',cmd)
     print('RUN:', ' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
    digests=[json.loads((a.out/f'{c}_s{seed}/initialization.json').read_text())['lora_initial_sha256'] for c in ['context','direct']]
    if digests[0]!=digests[1]:raise ValueError('Paired initialization mismatch')
   if audit.wait()!=0:raise RuntimeError('Parallel training-label audit failed: inspect training_label_audit.log')
  finally:
   if audit.poll() is None:audit.terminate();audit.wait()
 cmd=[sys.executable,str(HERE/'analyze.py'),'--run',str(a.out)]
 if a.smoke_test:cmd.append('--smoke-test')
 subprocess.run(cmd,check=True)
 verify_freeze();assert verify_source(a.source)==hashes
 js(a.out/'commands.json',commands)
 js(a.out/'complete.json',{'training_runs':12,'training_seeds':list(range(6)),'items_per_fit':785,'steps_per_fit':198,'epochs':2,'batch':8,'primary_split':'fixed_D303','eval_full_split':'D_reservedC','baseline_regenerated':False,'targets_regenerated':False,'smoke_test':a.smoke_test,'seconds':time.time()-start,'training_label_audit_complete':True,'next_experiment_authorized':False})
 print('MATCHED785 COMPLETE. Stop at this 12-fit decision checkpoint.',flush=True)
if __name__=='__main__':main()
