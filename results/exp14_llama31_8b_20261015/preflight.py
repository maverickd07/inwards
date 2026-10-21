"""Metadata-only login checks; weight hashes are compute-node work."""
import argparse,importlib.metadata,os,platform,shutil,time
from common import *
from preflight_source import parse_quota

def model_check(model,receipt,weights=False):
 r=json.loads(receipt.read_text())
 if r['repo']!=CONFIG['model_repo'] or r['revision']!=CONFIG['model_revision']:raise ValueError('Wrong model revision')
 for name,meta in r['files'].items():
  p=model/name
  if Path(name).name!=name or not p.is_file() or p.stat().st_size!=meta['size']:raise ValueError('Missing/truncated model file: '+name)
  if weights or not name.endswith('.safetensors'):
   if sha(p)!=meta['sha256']:raise ValueError('Model checksum mismatch: '+name)
 cfg=json.loads((model/'config.json').read_text())
 if (cfg['model_type'],cfg['hidden_size'],cfg['num_hidden_layers'],cfg['vocab_size'])!=('llama',4096,32,128256):raise ValueError('Not Llama 3.1 8B')
 gen=json.loads((model/'generation_config.json').read_text())
 if set(gen['eos_token_id'])!={128001,128008,128009}:raise ValueError('Unexpected EOS configuration')
 idx=json.loads((model/'model.safetensors.index.json').read_text())
 if not set(idx['weight_map'].values())<=set(r['files']):raise ValueError('Weight index missing shards')
 if sum((model/n).stat().st_size for n in set(idx['weight_map'].values()))<idx['metadata']['total_size']:raise ValueError('Truncated weights')
 return r

def environment():
 versions={p:importlib.metadata.version(p) for p in CONFIG['expected_versions']}
 if versions!=CONFIG['expected_versions']:raise ValueError('Environment differs from exp12: '+str(versions))
 if platform.python_version()!='3.11.16':raise ValueError('Expected original Python 3.11.16')
 return versions

def check(root,quota_file=None,ticket=None):
 verify_freeze();versions=environment();model_check(root/CONFIG['model_relative_path'],HERE/'MODEL_DOWNLOAD.json')
 if quota_file:
  if not -60<=time.time()-quota_file.stat().st_mtime<=300:raise ValueError('Refresh lquota immediately before submission')
  quota=quota_file.read_text();created=time.time()
 else:
  t=json.loads(ticket.read_text());created=t['checked_unix'];quota=t['quota_text']
  if time.time()-created>48*3600:raise ValueError('Quota ticket expired after 48h; no training started')
 free=parse_quota(quota)
 if shutil.disk_usage(root).free<6*1024**3:raise ValueError('Filesystem lacks 6 GiB output headroom')
 return {'checked_unix':created,'quota_text':quota,'free_gb':free,'versions':versions,
  'model_download_receipt_sha256':sha(HERE/'MODEL_DOWNLOAD.json'),'freeze_sha256':sha(HERE/'FREEZE.json')}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path.home()/'inward');ap.add_argument('--quota-file',type=Path);ap.add_argument('--ticket',type=Path);ap.add_argument('--write-ticket',type=Path);a=ap.parse_args()
 result=check(a.root,a.quota_file,a.ticket)
 if a.write_ticket:js(a.write_ticket,result)
 if a.ticket:
  original=json.loads(a.ticket.read_text())
  for name in ['model_download_receipt_sha256','freeze_sha256']:
   if original[name]!=result[name]:raise ValueError('Inputs changed since submission')
 print('Lightweight preflight passed; no model loaded.',flush=True)
