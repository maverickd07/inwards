"""Run locally once after editing; never run on HPC to repair a mismatch."""
from common import *
def freeze():
 files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.json','.txt','.sh'] and p.name not in ['FREEZE.json','PACKAGE_RECEIPT.json','LOCAL_VERIFICATION.json','SHA256SUMS'] and '_draft' not in p.stem]
 files+=list((HERE/'plan').glob('*.json*'))
 js(HERE/'FREEZE.json',{'experiment':CONFIG['experiment'],'files':{str(p.relative_to(HERE)):sha(p) for p in sorted(files)},'decision_rule':CONFIG['success']})
if __name__=='__main__':freeze()
