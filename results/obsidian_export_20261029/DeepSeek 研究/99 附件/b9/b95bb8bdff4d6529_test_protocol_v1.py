"""Targeted scoring and paired-statistic invariants, plus local random Qwen smoke."""
import copy,json,sys,unittest
from pathlib import Path
import numpy as np
from scoring import score_answer,score_row,parts,country_score
from analyze import summarize
from prompts import CONDITIONS
HERE=Path(__file__).resolve().parent
ROWS=[json.loads(x) for x in (HERE/'inputs/items.jsonl').read_text().splitlines()]
def item(country,cat='birthplace_currency'):
 return {'birth_country':country,'category':cat,'A1':[country],'Answer':['INR']}
class ProtocolTests(unittest.TestCase):
 def test_currency_equivalence(self):
  for cat in ['birthplace_currency','birthplace_currency_short','birthplace_currency_symbol']:
   for pred in ['INR','₹','Indian rupee','Indian rupees','Indian rupee (INR)','Currency: INR','INR = ₹']:
    self.assertTrue(score_answer(pred,item('India',cat))['correct'],(cat,pred))
 def test_historical(self):
  for country,pred in [('Italy','ITL'),('Italy','Italian lira'),('Italy','Lira'),('Germany','Mark'),('Germany','DEM'),('Germany','Deutsche marks'),('Bulgaria','BGN'),('Cuba','CUC')]:
   self.assertEqual(score_answer(pred,item(country))['status'],'HISTORICAL',(country,pred))
 def test_live_currency_not_historical(self):
  for country,pred in [('Turkey','Turkish lira'),('Bosnia and Herzegovina','convertible mark'),('India','Indian rupee')]:
   self.assertTrue(score_answer(pred,item(country))['correct'],(country,pred))
 def test_conflict_no_substring_credit(self):
  for pred in ['INR or USD','Not INR','Indian rupee and US dollar','Indian rupee (USD)']:
   self.assertFalse(score_answer(pred,item('India'))['correct'],pred)
 def test_typed_answers(self):
  e=dict(item('India','birthplace_ccn3'),Answer=['004'])
  self.assertTrue(score_answer('4',e)['correct']);self.assertFalse(score_answer('14',e)['correct']);self.assertFalse(score_answer('4.9',e)['correct'])
  e=dict(e,category='birthplace_tld',Answer=['.in']);self.assertTrue(score_answer('IN',e)['correct']);self.assertFalse(score_answer('.uk',e)['correct'])
 def test_country_and_final_separation(self):
  c,a,bad=parts('Country: India\nAnswer: USD','C2');self.assertFalse(bad);self.assertTrue(country_score(c,item('India')));self.assertFalse(score_answer(a,item('India'))['correct'])
  self.assertTrue(parts('India INR','C2')[2])
 def records(self,mat):
  result=[]
  for i,answers in enumerate(mat):
   for j,c in enumerate(CONDITIONS):
    result.append(dict(id=i,condition=c,D500_known313=True,fixed303=True,score={'correct':bool(answers[j]),'status':'CORRECT' if answers[j] else 'INCORRECT'},country_correct=None,wrong_country_proxy_candidates=[],truncated=False))
  return result
 def test_paired_bootstrap(self):
  rows=self.records([[0,1,0,1,1]]*12+[[1,1,1,1,1]]*8)
  s=summarize(rows,1000)['sets']['fixed303'];self.assertEqual(s['conditions']['C1']['recovery'],1);self.assertEqual(s['conditions']['C1']['recovery_ci95'],[1,1]);self.assertGreater(s['C1_minus_ctrl']['accuracy_difference_ci95'][0],0)
 def test_zero_and_unstable_denominator(self):
  rows=self.records([[1,1,0,1,1]]*4)
  s=summarize(rows,100)['sets']['fixed303'];self.assertIsNone(s['conditions']['C1']['recovery']);self.assertIsNone(s['conditions']['C1']['recovery_ci95'])
  rows=self.records([[0,1,0,1,1],[1,1,0,1,1]])
  s=summarize(rows,1000)['sets']['fixed303'];self.assertGreater(s['bootstrap_nonpositive_denominator_n'],0);self.assertIsNone(s['conditions']['C1']['recovery_ci95'])
 def test_fixed_rosters(self):
  self.assertEqual(len(ROWS),402);self.assertEqual(sum(r['fixed303'] for r in ROWS),303);self.assertEqual(sum(r['D500_known313'] for r in ROWS),313)
if __name__=='__main__':unittest.main()
