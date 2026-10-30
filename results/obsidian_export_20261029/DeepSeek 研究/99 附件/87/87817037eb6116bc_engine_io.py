"""Small stdlib IO; avoids importing torch in the analysis subprocess."""
import json
from pathlib import Path

def write(path,rows):
 with Path(path).open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
