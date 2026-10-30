"""Read-only source and environment validation; large weight hashes only on compute node."""
import importlib.metadata,json,sys
from pathlib import Path
from common import HERE,NAMES,sha
MANIFEST=json.loads((HERE/'source_manifest.json').read_text())
def environment(source):
 expected=MANIFEST['expected_versions'];actual={p:importlib.metadata.version(p) for p in expected}
 if actual!=expected:raise ValueError(f'Runtime drift: expected {expected}, got {actual}')
 prior=json.loads((source/'context_s0/runtime.json').read_text())
 if sys.version.split()[0]!=prior['python'].split()[0]:raise ValueError('Python differs from source job')
 return actual

def check(source,smoke=False,weights=False):
 source=Path(source);complete=json.loads((source/'complete.json').read_text())
 if complete['smoke_test']!=smoke or complete['training_runs']!=12:raise ValueError('Wrong source completion/smoke status')
 hashes={}
 if not smoke:
  for rel,digest in MANIFEST['files'].items():
   if sha(source/rel)!=digest:raise ValueError('Source metadata changed: '+rel)
   hashes[rel]=digest
 for name in NAMES[1:]:
  folder=source/name;cfg=json.loads((folder/'run_config.json').read_text());done=json.loads((folder/'complete.json').read_text())
  if cfg['n_unique']!=785 or cfg['optimizer_steps']!=198 or done['steps']!=198:raise ValueError('Wrong matched785 source')
  adapter=folder/'adapter/adapter_model.safetensors';config=folder/'adapter/adapter_config.json'
  if not adapter.is_file() or not adapter.stat().st_size or not config.is_file():raise FileNotFoundError(adapter)
  ac=json.loads(config.read_text())
  if ac['r']!=16 or ac['lora_alpha']!=32 or ac['lora_dropout']!=.05:raise ValueError('Adapter configuration drift')
  hashes[str(config.relative_to(source))]=sha(config)
  if weights:
   expected=done['adapter_sha256'] if smoke else MANIFEST['adapter_sha256'][name]
   if sha(adapter)!=expected:raise ValueError('Adapter bytes differ: '+name)
   hashes[str(adapter.relative_to(source))]=expected
 return hashes
