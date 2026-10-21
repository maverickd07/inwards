"""Meaningful regressions: paired statistics, no overwrite, and corrupted prediction rejection.
Run AFTER completed local smoke: python test_audit.py --smoke smoke_v2
"""
import argparse,json,shutil,subprocess,sys,tempfile
from pathlib import Path
from analyze_exp11 import paired,holm
from scoring_v2_v1 import is_correct
HERE=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',type=Path,required=True);a=ap.parse_args()
 assert paired([True]*4,[False]*4)['mcnemar_p']==.125
 ts=[{'mcnemar_p':.01},{'mcnemar_p':.04},{'mcnemar_p':.03}];holm(ts);assert [t['mcnemar_p_holm'] for t in ts]==[.03,.06,.06]
 assert not is_correct('-2',['2']);assert is_correct('-2.5',['-3']);assert not is_correct('-2.5',['-2']);assert is_correct('-2.5',['-2'],'truncate')
 done=json.loads((a.smoke/'complete.json').read_text());assert done['training_runs']==21
 rep=json.loads((a.smoke/'analysis/report.json').read_text());assert rep['C_primary_test_family_n']==54 and rep['C_hop_test_family_n']==42
 # Existing outputs must fail before model loading or file writes.
 original=(a.smoke/'complete.json').read_bytes()
 r=subprocess.run([sys.executable,str(HERE/'exp11_iterate.py'),'--stage','baseline','--out',str(a.smoke)],capture_output=True,text=True)
 assert r.returncode!=0 and 'No overwrite' in r.stderr;assert original==(a.smoke/'complete.json').read_bytes()
 with tempfile.TemporaryDirectory(prefix='exp11-audit-',dir='/private/tmp') as t:
  clone=Path(t)/'runs';shutil.copytree(a.smoke,clone)
  p=clone/'reset_r2_s0/eval_C_bothNew.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()];rows[0]['composed_ok']=not rows[0]['composed_ok'];p.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
  r=subprocess.run([sys.executable,str(HERE/'analyze_exp11.py'),'--runs',str(clone),'--out',str(Path(t)/'analysis')],capture_output=True,text=True)
  assert r.returncode!=0 and 'AssertionError' in r.stderr,'Corrupted scoring escaped audit'
 print('PASS: full flow, 54/42 test families, scoring, Holm/McNemar, no overwrite, corruption rejection.')
if __name__=='__main__':main()
