"""One-shot pre-inference freeze; never update based on model outcomes."""
from common import HERE,sha,js
if __name__=='__main__':
 names=['PROTOCOL.txt','inference_config.json','source_manifest.json','upstream_commit.json','common.py','check_source.py','infer.py','scoring.py','score_model.py','analyze.py','run_pipeline.py','prepare_data.py','preflight.py','preflight_v1.py','submit.sh','run_exp13.sh','pack_review.py']
 names += [str(p.relative_to(HERE)) for p in (HERE/'data').glob('*.json*')]
 names += ['upstream/metrics/answer.py','upstream/metrics/metric.py','upstream/LICENSE']
 js(HERE/'FREEZE.json',{'stage':'Before first new14B inference; all data/selection/scoring/interpretation frozen','sha256':{n:sha(HERE/n) for n in sorted(names)}})
