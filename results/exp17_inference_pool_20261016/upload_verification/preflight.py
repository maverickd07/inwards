"""Login-node metadata checks only: no ML imports, no inference or roster selection."""
import json,re,time,importlib.metadata,shutil,argparse,os
from pathlib import Path
from shared import HERE,verify_package,sha,js

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path.home()/'inward');ap.add_argument('--quota-file',type=Path);ap.add_argument('--write-ticket',type=Path);ap.add_argument('--ticket',type=Path);a=ap.parse_args();verify_package()
 meta=json.loads((HERE/'inputs/roster_summary.json').read_text())
 if meta['new_people']!=400 or not meta['source_review_complete']:raise ValueError('Roster audit incomplete')
 model=a.root/'models/Qwen3-14B';idx=json.loads((model/'model.safetensors.index.json').read_text());cfg=json.loads((model/'config.json').read_text())
 if cfg.get('model_type')!='qwen3':raise ValueError('Wrong model type')
 shards=sorted(set(idx['weight_map'].values()))
 if not shards or any(Path(s).name!=s or not(model/s).is_file() or(model/s).stat().st_size==0 for s in shards):raise ValueError('Missing/empty model shard')
 total=sum((model/s).stat().st_size for s in shards)
 if total<20*1024**3 or total<idx.get('metadata',{}).get('total_size',0):raise ValueError('Unexpected/incomplete 14B weights')
 for f in ['tokenizer_config.json','tokenizer.json']:
  if not(model/f).is_file():raise ValueError('Missing tokenizer '+f)
 versions={p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate']}
 if a.quota_file:
  age=time.time()-a.quota_file.stat().st_mtime
  if age < -60 or age>300:raise ValueError('Refresh lquota immediately before submission')
  quota=a.quota_file.read_text()
 elif a.ticket:
  t=json.loads(a.ticket.read_text())
  if time.time()-t['checked_unix']>48*3600:raise ValueError('Quota ticket expired; ask for manual review')
  quota=t['quota_text']
 else:raise ValueError('Missing quota capture/ticket')
 m=re.search(r'USED\s*=\s*([\d.]+)\s*([KMGT])B\s+LIMIT\s*=\s*([\d.]+)\s*([KMGT])B',quota,re.I)
 if not m:raise ValueError('Unrecognized quota output')
 factor={'K':1/1024**2,'M':1/1024,'G':1,'T':1024};free=float(m[3])*factor[m[4].upper()]-float(m[1])*factor[m[2].upper()]
 if free<2 or shutil.disk_usage(a.root).free<2*1024**3:raise ValueError('Require 2GB free headroom')
 ticket={'checked_unix':time.time(),'quota_text':quota,'quota_free_gib':free,'versions':versions,'model_config_sha256':sha(model/'config.json'),'weight_index_sha256':sha(model/'model.safetensors.index.json'),'package_freeze_sha256':sha(HERE/'PACKAGE_FREEZE.json')}
 if a.write_ticket:js(a.write_ticket,ticket)
 print('Preflight passed. New people=400; no training; model shards=',len(shards),'free_GB=',round(free,2));print(versions)
if __name__=='__main__':main()
