"""Numerical edge cases that could reverse the scientific interpretation."""
import unittest
import numpy as np
from analyze import paired,emergence_stats
from scoring import score_row
from run import read,HERE
class Protocol(unittest.TestCase):
 def test_censoring_is_retained(self):
  rr=[{'id':i,'num_layers':40,'emergence':{'bridge':v}} for i,v in enumerate([1,None,3,None])]
  s=emergence_stats(rr,'bridge',1000,np.random.default_rng(21));self.assertEqual(s['censored_L_plus_1']['n'],4);self.assertEqual(s['censored_L_plus_1']['median'],22);self.assertEqual(s['censored_L_plus_1']['mean'],21.5);self.assertEqual(s['never_hit_n'],2);self.assertEqual(s['hit_only']['mean'],2)
 def test_paired_direction_and_censoring(self):
  a=[{'id':i,'num_layers':40,'emergence':{'bridge':v}} for i,v in enumerate([10,None,30])]
  b=[dict(r,emergence={'bridge':v}) for r,v in zip(a,[5,31,20])]
  s=paired(a,b,'bridge',1000,np.random.default_rng(21));self.assertEqual(s['earlier_n'],3);self.assertLess(s['delta_mean_ci95'][1],0);self.assertEqual(s['both_hit_n'],2)
 def test_roster_scoring_identity(self):
  rr=read(HERE/'inputs/items.jsonl');self.assertEqual(len(rr),402);self.assertEqual(sum(r['fixed303'] for r in rr),303);self.assertEqual(sum(r['D500_known313'] for r in rr),313);self.assertEqual(sum(r['fixed303'] and r['D500_known313'] for r in rr),214)
  r=rr[0];self.assertTrue(score_row(dict(r,raw=r['e']['Answer'][0],condition='C0',truncated=False))['score']['correct'])
  self.assertTrue(score_row(dict(r,raw='Country: '+r['e']['birth_country']+'\nAnswer: '+r['e']['Answer'][0],condition='C2',truncated=False))['score']['correct'])
if __name__=='__main__':unittest.main()
