"""Lightweight checks, no model inference. Quota failures stop before expensive work."""
import json,re,shutil,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
for p in [Path('models/Qwen3-14B/config.json'),Path('results/exp1v2_Qwen3-14B/items.jsonl'),Path('data/compositional_celebrities.json'),*[H/'reference_exp8'/f'eval_{s}.jsonl' for s in ['A_newPerson','B_newCategory','C_bothNew']]]:
 if not p.is_file():raise FileNotFoundError(p)
idx=Path('models/Qwen3-14B/model.safetensors.index.json')
if idx.exists():
 for fn in set(json.loads(idx.read_text())['weight_map'].values()):
  if not (idx.parent/fn).is_file():raise FileNotFoundError(fn)
elif not Path('models/Qwen3-14B/model.safetensors').is_file():raise FileNotFoundError('Model weights missing')
if shutil.which('lquota'):
 p=subprocess.run(['lquota'],capture_output=True,text=True,check=True);text=p.stdout+p.stderr;print(text)
 m=re.search(r'USED\s*=\s*([\d.]+)\s*([KMGT])B\s+LIMIT\s*=\s*([\d.]+)\s*([KMGT])B',text,re.I)
 if not m:raise RuntimeError('Cannot parse quota; verify lquota output before running')
 factors={'K':1/1024**2,'M':1/1024,'G':1,'T':1024};free=float(m[3])*factors[m[4].upper()]-float(m[1])*factors[m[2].upper()]
 if free<8:raise RuntimeError(f'Only {free:.1f} GB quota free; require >=8 GB for adapters/results and headroom')
 print('Quota free:',free,'GB')
else:raise RuntimeError('lquota unavailable: verify quota explicitly before proceeding')
print('Input files/model shards/quota checks passed.')
