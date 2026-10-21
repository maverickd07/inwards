"""Only frozen evaluation data/code; never upload raw large datasets or local smoke weights."""
import json,tarfile
from common import HERE,sha,js,verify_freeze
FILES=['common.py','prepare_data.py','check_source.py','preflight.py','preflight_v1.py','input_hashes.json','source_manifest.json','inference_config.json','infer.py','scoring.py','score_model.py','analyze.py','run_pipeline.py','run_exp13.sh','submit.sh','pack_review.py','freeze.py','FREEZE.json','PROTOCOL.txt','HPC_COMMANDS.txt','upstream_commit.json','LOCAL_VERIFICATION.json','test_integrity.py','validate_statistics.py','statistics_validation.json','build_package.py','upstream/metrics/answer.py','upstream/metrics/metric.py','upstream/LICENSE','upstream/README.md','upstream/evaluate_v1.0.py','upstream/download_data.sh','upstream/raw_data_to_official_format.py']
if __name__=='__main__':
 verify_freeze();verified=json.loads((HERE/'LOCAL_VERIFICATION.json').read_text())
 assert verified['thirteen_model_smoke_passed'] and verified['integrity_passed']
 for name,digest in verified['tested_source_hashes'].items():assert sha(HERE/name)==digest,name
 files=FILES+[str(p.relative_to(HERE)) for p in sorted((HERE/'data').glob('*.json*'))]
 assert len(files)==len(set(files))
 with (HERE/'SHA256SUMS').open('x') as f:
  for name in files:f.write(sha(HERE/name)+'  '+name+'\n')
 dest=HERE/'exp13_musique_upload.tgz'
 with tarfile.open(dest,'x:gz') as tar:
  for name in files+['SHA256SUMS']:tar.add(HERE/name,arcname=name,recursive=False)
 js(HERE/'PACKAGE_RECEIPT.json',{'archive':str(dest),'sha256':sha(dest),'bytes':dest.stat().st_size,'files':len(files)+1,'HPC_submitted':False})
 print(dest,dest.stat().st_size,'bytes')
