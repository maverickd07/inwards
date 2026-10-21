"""Serial orchestration, fresh outputs, fixed 3-round design; no HPC submission."""
import argparse,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--items',required=True);ap.add_argument('--data',required=True);ap.add_argument('--reference-evals',default=str(HERE/'reference_exp8'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke-test',action='store_true');ap.add_argument('--gen-batch',type=int,default=8);a=ap.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);commands=[]
 def run(stage,out,**kw):
  cmd=[sys.executable,str(HERE/'exp11_iterate.py'),'--stage',stage,'--out',str(out)]
  for k,v in kw.items():cmd+=['--'+k.replace('_','-'),str(v)]
  if a.smoke_test:cmd+=['--smoke-test']
  commands.append(cmd);(a.out/'commands.json').write_text(json.dumps(commands,indent=2));print('RUN:', ' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
 plan=a.out/'plan';run('plan',plan,items=a.items,data=a.data,reference_evals=a.reference_evals)
 model=str(plan/'tiny_model') if a.smoke_test else a.model
 common={'model':model,'plan':plan,'gen_batch':a.gen_batch}
 base_targets=a.out/'base_targets';run('targets',base_targets,**common);run('baseline',a.out/'baseline',**common)
 stages=[]
 def fit(condition,round_,seed,targets,adapter=None):
  out=a.out/f'{condition}_r{round_}_s{seed}';kw=dict(common,targets=targets,condition=condition,round=round_,train_seed=seed,epochs=2*round_ if condition=='frozen_budget' else 2,batch=8)
  if adapter:kw['adapter']=adapter
  run('train',out,**kw);stages.append({'condition':condition,'round':round_,'seed':seed,'path':out.name});return out
 for seed in range(3):
  r1=fit('shared',1,seed,base_targets/'targets.jsonl')
  t1=a.out/f'shared_targets_r1_s{seed}';run('targets',t1,**common,adapter=r1/'adapter',train_seed=seed)
  for branch in ['reset','continue']:
   previous=r1;cache=t1
   for round_ in [2,3]:
    out=fit(branch,round_,seed,cache/'targets.jsonl',previous/'adapter' if branch=='continue' else None)
    next_cache=a.out/f'{branch}_targets_r{round_}_s{seed}';run('targets',next_cache,**common,adapter=out/'adapter',train_seed=seed)
    previous=out;cache=next_cache
  for round_ in [2,3]:fit('frozen_budget',round_,seed,base_targets/'targets.jsonl')
 (a.out/'stages.json').write_text(json.dumps(stages,indent=2));subprocess.run([sys.executable,str(HERE/'analyze_exp11.py'),'--runs',str(a.out),'--out',str(a.out/'analysis')],check=True)
 (a.out/'complete.json').write_text(json.dumps({'training_runs':len(stages),'smoke_test':a.smoke_test,'rounds':3,'seeds':[0,1,2],'exp12_authorized':False},indent=2));print('EXP11 COMPLETE:',a.out,flush=True)
if __name__=='__main__':main()
