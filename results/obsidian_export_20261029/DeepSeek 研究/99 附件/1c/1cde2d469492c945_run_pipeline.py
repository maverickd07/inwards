"""13 sequential inference processes, baseline first; immutable evaluation, no training."""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
from common import HERE,NAMES,read,write,js,sha,verify_freeze
from check_source import check,environment

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke-test',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('No overwrite: '+str(a.out))
 verify_freeze()
 if not a.smoke_test:environment(a.source)
 hashes=check(a.source,a.smoke_test,weights=True)
 a.out.mkdir(parents=True);started=time.time();inp=a.out/'inputs';shutil.copytree(HERE/'data',inp)
 for name in ['FREEZE.json','PROTOCOL.txt','source_manifest.json','inference_config.json','upstream_commit.json']:
  shutil.copyfile(HERE/name,inp/name)
 shutil.copytree(HERE/'upstream/metrics',inp/'official_metrics',ignore=shutil.ignore_patterns('__pycache__'))
 shutil.copyfile(HERE/'upstream/LICENSE',inp/'DATA_LICENSE.txt')
 if a.smoke_test:
  rows=read(inp/'items.jsonl');small=[next(r for r in rows if r['n_hops']==n) for n in [2,3,4]]
  ids={q['query_id'] for r in small for q in [r]+r['hops']}
  for file,rows in [('items.jsonl',small),('queries.jsonl',[r for r in read(inp/'queries.jsonl') if r['query_id'] in ids]),('timing_queries.jsonl',read(inp/'timing_queries.jsonl')[:2])]:
   (inp/file).unlink();write(inp/file,rows)
 js(a.out/'source_receipt.json',{'source':str(a.source),'source_job':701792,'sha256':hashes,'smoke_test':a.smoke_test})
 js(a.out/'code_hashes.json',{p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.json','.txt','.sh']})
 commands=[]
 for name in NAMES:
  cmd=[sys.executable,'-u',str(HERE/'infer.py'),'--model',a.model,'--queries',str(inp/'queries.jsonl'),'--out',str(a.out/name),'--name',name]
  if name=='baseline':cmd+=['--timing-queries',str(inp/'timing_queries.jsonl')]
  else:cmd+=['--adapter',str(a.source/name/'adapter')]
  if a.smoke_test:cmd+=['--smoke-test']
  js(a.out/f'command_{name}.json',cmd);commands.append(cmd);print('RUN',name,flush=True);subprocess.run(cmd,check=True)
  scorecmd=[sys.executable,str(HERE/'score_model.py'),'--folder',str(a.out/name),'--items',str(inp/'items.jsonl')]
  if name=='baseline':scorecmd+=['--baseline']
  subprocess.run(scorecmd,check=True)
  if name=='baseline':
   timing=json.loads((a.out/name/'timing.json').read_text());complete=json.loads((a.out/name/'complete.json').read_text())
   est=12*complete['seconds'];js(a.out/'timing_estimate.json',{'baseline_timing':timing,'baseline_full_seconds':complete['seconds'],'remaining12_generation_seconds_linear_estimate':est,'excludes':'future model loading, adapter overhead, scheduler and varying output lengths','policy':'Timing only. No data/prompt/scoring changes or extra runs.'})
   print('Estimated remaining generation seconds (rough):',est,flush=True)
 subprocess.run([sys.executable,str(HERE/'analyze.py'),'--run',str(a.out)],check=True)
 assert check(a.source,a.smoke_test,weights=True)==hashes
 verify_freeze();js(a.out/'commands.json',commands)
 js(a.out/'complete.json',{'models_evaluated':13,'training_runs':0,'source_job':701792,'smoke_test':a.smoke_test,'items':len(read(inp/'items.jsonl')),'queries_per_model':len(read(inp/'queries.jsonl')),'primary':'two-hop EM','seconds':time.time()-started,'source_unchanged':True,'next_experiment_authorized':False})
 print('EXP13 COMPLETE. Fixed stage ends here.',flush=True)
if __name__=='__main__':main()
