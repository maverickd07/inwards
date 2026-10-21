"""Lightweight frozen allocation checks, shared by preflight and execution."""
from reuse import HERE,sha,read,key
import json

def verify_freeze():
 freeze=json.loads((HERE/'ALLOCATION_FREEZE.json').read_text())
 for rel,digest in freeze['files'].items():
  if sha(HERE/rel)!=digest:raise ValueError('Frozen allocation changed: '+rel)
 for rel,digest in json.loads((HERE/'DESIGN_FREEZE.json').read_text())['files'].items():
  if sha(HERE/rel)!=digest:raise ValueError('Frozen design changed: '+rel)
 rows=read(HERE/'frozen_inputs/plan/train.jsonl')
 assert len(rows)==len(set(map(key,rows)))==785
 for c in ['context','direct']:
  assert list(map(key,read(HERE/f'frozen_inputs/paired_targets/{c}_targets.jsonl')))==list(map(key,rows))
 return freeze
