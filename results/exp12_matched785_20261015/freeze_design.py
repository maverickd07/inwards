"""Run once before tests/submission. Changes require a new version, never overwrite."""
from pathlib import Path
from reuse import js,sha
HERE=Path(__file__).parent
if __name__=='__main__':
 names=['ALLOCATION_FREEZE.json','PROTOCOL.txt','training_label_scorer.py','training_label_policy.json','audit_training_labels.py','analyze.py','run_pipeline.py','train_seed.py','protocol.py','scorer_v3.py','aliases.json','reference_catalog.json','SCORER_FREEZE.json','exp12_v1.py','engine_exp11_v1.py','exp1_gap_v2_v1.py','scoring_v2_v1.py','engine_io.py']
 js(HERE/'DESIGN_FREEZE.json',{'stage':'Before any new 14B training; after fixed allocation; never revise based on outcomes','files':{n:sha(HERE/n) for n in names},'training_label_policy_separate_from_eval_v3':True})
