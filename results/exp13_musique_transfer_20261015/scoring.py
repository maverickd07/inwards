"""Exact upstream answer EM/F1, without extra stripping or v3 country rules."""
import sys
from common import HERE
sys.path.insert(0,str(HERE/'upstream'))
from metrics.answer import compute_exact,compute_f1,metric_max_over_ground_truths

def scores(pred,answers):
 if not answers:raise ValueError('No reference answers')
 return {'em':metric_max_over_ground_truths(compute_exact,pred,answers),'f1':metric_max_over_ground_truths(compute_f1,pred,answers)}
