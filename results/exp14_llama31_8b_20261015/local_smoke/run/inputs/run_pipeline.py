"""One serial run: baseline, teachers, pretrain seal, twelve fits, analysis."""
import argparse,shutil,subprocess,sys,time
from common import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--smoke-test',action='store_true');ap.add_argument('--smoke-plan',type=Path);a=ap.parse_args()
 frozen=verify_freeze();a.out=a.out.resolve();a.model=a.model.resolve();a.out.mkdir(parents=True,exist_ok=False);started=time.time()
 plan=a.smoke_plan if a.smoke_test else HERE/'plan'
 if not a.smoke_test and a.smoke_plan:raise ValueError('Production plan is immutable')
 shutil.copytree(plan,a.out/'plan');(a.out/'inputs').mkdir()
 for name in list(frozen['files'])+['FREEZE.json']:
  p=a.out/'inputs'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HERE/name,p)
 if not a.smoke_test:
  from preflight import model_check,environment
  environment();model_check(a.model,HERE/'MODEL_DOWNLOAD.json',weights=True)
  shutil.copy2(HERE/'MODEL_DOWNLOAD.json',a.out/'inputs/MODEL_DOWNLOAD.json')
  shutil.copy2(HERE/'submission.json',a.out/'inputs/submission.json')
  shutil.copy2(HERE/'quota_ticket.json',a.out/'inputs/quota_ticket.json')
 train=read(a.out/'plan/train.jsonl');d=read(a.out/'plan/items_D_reservedC.jsonl')
 if not a.smoke_test and (len(train)!=864 or len(d)!=500):raise ValueError('Allocation changed')
 assert not({r['person'] for r in train}&{r['person'] for r in d})
 assert not({r['family'] for r in train}&{r['family'] for r in d})
 def stage(stage,name,**kwargs):
  cmd=[sys.executable,'-u',str(HERE/'exp14.py'),'--stage',stage,'--model',str(a.model),'--plan',str(a.out/'plan'),'--out',str(a.out/name)]
  for k,v in kwargs.items():cmd+=['--'+k.replace('_','-'),str(v)]
  if a.smoke_test:cmd+=['--smoke-test']
  with (a.out/'commands.jsonl').open('a') as f:f.write(json.dumps({'stage':name,'command':cmd,'started_unix':time.time()})+'\n')
  print('RUN',name,flush=True);subprocess.run(cmd,check=True)
 stage('baseline','baseline');stage('targets','paired_targets')
 from seal_evaluation import seal
 seal(a.out);seal_digest=sha(a.out/'EVALUATION_FREEZE.json')
 for seed in range(6):
  for c in ['context','direct']:
   if sha(a.out/'EVALUATION_FREEZE.json')!=seal_digest:raise ValueError('Seal document changed')
   verify_seal(a.out)
   stage('train',f'{c}_s{seed}',condition=c,train_seed=seed,targets=a.out/'paired_targets'/f'{c}_targets.jsonl')
   if seed==0 and c=='context':
    cfg=json.loads((a.out/'context_s0/run_config.json').read_text())
    js(a.out/'timing_forecast.json',{'one_fit_train_seconds':cfg['training_seconds'],'one_fit_eval_seconds':cfg['evaluation_seconds'],
     'approx_remaining_11_fits_seconds':11*(cfg['training_seconds']+cfg['evaluation_seconds']),
     'scope':'rough forecast only; excludes load and analysis; no adaptive budget change'})
 from analyze import analyze
 analyze(a.out,a.smoke_test)
 if not a.smoke_test:model_check(a.model,HERE/'MODEL_DOWNLOAD.json',weights=True)
 verify_seal(a.out)
 if sha(a.out/'EVALUATION_FREEZE.json')!=seal_digest:raise ValueError('Seal changed')
 js(a.out/'complete.json',{'training_runs':12,'smoke_test':a.smoke_test,'seconds':time.time()-started,
  'evaluation_freeze_sha256':seal_digest,'analysis_sha256':sha(a.out/'analysis_v3/report.json'),
  'manual_flip_review':'pending','next_experiment_authorized':False})
 print('EXP14 COMPLETE: 12 fits; stop and review results.',flush=True)
if __name__=='__main__':main()
