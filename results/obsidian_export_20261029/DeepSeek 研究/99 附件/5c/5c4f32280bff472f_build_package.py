"""Reproducible explicit upload allowlist; exclude local models, output and backups."""
import json,hashlib,tarfile
from pathlib import Path
from protocol import verify_freeze
from reuse import HERE,sha,js
FILES=['freeze_inputs.py','freeze_design.py','exp12_v1.py','engine_exp11_v1.py','exp1_gap_v2_v1.py','scoring_v2_v1.py','reuse.py','source_700454_manifest.json','input_hashes.json','preflight_v1.py','preflight.py','submit.sh','run_exp12.sh','train_seed.py','run_pipeline.py','protocol.py','engine_io.py','analyze.py','pack_review.py','audit_training_labels.py','training_label_scorer.py','training_label_policy.json','scorer_v3.py','aliases.json','reference_catalog.json','SCORER_FREEZE.json','ALLOCATION_FREEZE.json','DESIGN_FREEZE.json','PROTOCOL.txt','HPC_COMMANDS.txt','LOCAL_VERIFICATION.json','test_integrity.py','build_package.py','validate_statistics.py','statistics_validation.json']
if __name__=='__main__':
 verify_freeze()
 verified=json.loads((HERE/'LOCAL_VERIFICATION.json').read_text())
 assert verified['twelve_fit_smoke_passed'] and verified['integrity_checks_passed']
 for name,digest in verified['tested_source_hashes'].items():assert sha(HERE/name)==digest,name
 files=FILES+[str(p.relative_to(HERE)) for p in sorted((HERE/'frozen_inputs').rglob('*')) if p.is_file()]
 with (HERE/'SHA256SUMS').open('x') as f:
  for name in files:f.write(sha(HERE/name)+'  '+name+'\n')
 archive=HERE/'exp12_matched785_upload.tgz'
 with tarfile.open(archive,'x:gz') as tar:
  for name in files+['SHA256SUMS']:tar.add(HERE/name,arcname=name,recursive=False)
 js(HERE/'PACKAGE_RECEIPT.json',{'archive':str(archive),'sha256':sha(archive),'size_bytes':archive.stat().st_size,'file_count':len(files)+1,'HPC_submitted':False})
 print(archive,archive.stat().st_size,'bytes')
