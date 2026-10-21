"""Focused scientific and scheduler regression tests, no network/GPU."""
import importlib,json,tempfile,unittest,math
from pathlib import Path
from unittest.mock import patch
import numpy as np
from common import *
from analyze import compare
from seal_evaluation import seal
from submit import dependencies

class Integrity(unittest.TestCase):
 def test_real_allocation_and_scorer_unchanged(self):
  verify_freeze()
  prov=json.loads((HERE/'SOURCE_PROVENANCE.json').read_text())
  verify_manifest(HERE,prov['copied_files'])
  train=read(HERE/'plan/train.jsonl');d=read(HERE/'plan/items_D_reservedC.jsonl')
  self.assertEqual(len(train),864);self.assertEqual(len(d),500)
  self.assertFalse({r['person'] for r in train}&{r['person'] for r in d})
  self.assertFalse({r['family'] for r in train}&{r['family'] for r in d})
  self.assertEqual(len({key(r) for r in train}),864)
 def test_statistics_and_sensitivity(self):
  rng=np.random.default_rng(14)
  groups={n:[{'scores':{'composed':{'correct':bool(c) and not bool(f),'needs_review':bool(f)}}} for c,f in zip(rng.integers(2,size=41),rng.integers(2,size=41))] for n in NAMES}
  for policy in ['all_wrong','all_correct','exclude_pair_union']:
   actual=compare(groups,list(range(41)),'composed',policy);expected=[]
   for s in range(6):
    aa=groups[f'context_s{s}'];bb=groups[f'direct_s{s}'];mask=[i for i in range(41) if policy!='exclude_pair_union' or not(aa[i]['scores']['composed']['needs_review'] or bb[i]['scores']['composed']['needs_review'])]
    def arr(rows):return np.array([rows[i]['scores']['composed']['correct'] or (policy=='all_correct' and rows[i]['scores']['composed']['needs_review']) for i in mask],dtype=int)
    expected.append(100*np.mean(arr(aa)-arr(bb)))
   np.testing.assert_allclose(actual['per_seed_pp'],expected)
   center=np.mean(expected);half=2.570581835636314*np.std(expected,ddof=1)/np.sqrt(6)
   np.testing.assert_allclose(actual['t95_pp'],[center-half,center+half])
  self.assertIsNone(compare(groups,[],'composed','all_wrong')['mean_pp'])
 def test_scheduler_guards(self):
  self.assertEqual(dependencies('123;cluster\n',[]),('123','123'))
  self.assertEqual(dependencies('123',['123']),('123','123'))
  self.assertEqual(dependencies('123',['456']),('123','123:456'))
  for receipt,jobs in [('123',['123','456']),('abc',[]),('123',['456_1']),('0',[])]:
   with self.assertRaises(ValueError):dependencies(receipt,jobs)
 def test_fixed_pretrain_selection_and_tamper(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'plan').mkdir();(root/'baseline').mkdir();(root/'inputs').mkdir();(root/'paired_targets').mkdir()
   items=[{'category':'birthplace_currency','person':str(i),'e':{'A1':['France'],'A2':['Euro'],'Answer':['Euro']}} for i in range(5)]
   # Keep initially right AND wrong; exclude baseline review and non-known hops.
   raw=[{'category':r['category'],'person':r['person'],'pred':p,'q1_pred':q1,'q2_pred':q2} for r,p,q1,q2 in zip(items,['Euro','US Dollar','some prose','Euro','Euro'],['France','France','France','Italy','France'],['Euro','Euro','Euro','Euro','US Dollar'])]
   write(root/'plan/items_D_reservedC.jsonl',items);write(root/'baseline/eval_D_reservedC.jsonl',raw)
   seal(root);ids=json.loads((root/'evaluation_ids.json').read_text())
   self.assertEqual([k[1] for k in ids['known']],['0','1','2']);self.assertEqual([k[1] for k in ids['clean']],['0','1'])
   verify_seal(root)
   with (root/'baseline/eval_D_reservedC.jsonl').open('a') as f:f.write(' \n')
   with self.assertRaises(ValueError):verify_seal(root)
 def test_submission_must_include_exp13(self):
  import submit
  with tempfile.TemporaryDirectory() as t:
   home=Path(t);root=home/'inward';root.mkdir();pkg=root/'new';pkg.mkdir()
   receipt=root/CONFIG['dependency_receipt'];receipt.parent.mkdir(parents=True);receipt.write_text('321\n')
   (pkg/'quota_login.txt').write_text('USED=60GB LIMIT=100GB')
   calls=[]
   def output(cmd,**kw):
    calls.append(cmd)
    if cmd[0]=='squeue':return ''
    if cmd[0]=='sacct':return '321|testuser|inward-exp13-musi|\n'
    if cmd[0]=='sbatch':return '654\n'
    raise AssertionError(cmd)
   import os
   previous=os.getcwd()
   try:
    with patch.object(submit,'HERE',pkg),patch('pathlib.Path.home',return_value=home),patch('getpass.getuser',return_value='testuser'),patch.object(submit,'check',return_value={'ok':True}),patch('subprocess.check_output',side_effect=output),patch('sys.argv',['submit.py','--configuration-reviewed']):
     submit.main()
     self.assertIn('--dependency=afterany:321',calls[-1]);self.assertEqual((pkg/'submitted_job.txt').read_text(),'654\n')
     with self.assertRaises(FileExistsError):submit.main()
   finally:os.chdir(previous)
if __name__=='__main__':unittest.main()
