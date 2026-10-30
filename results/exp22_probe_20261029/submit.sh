#!/bin/bash
set -euo pipefail
cd "$HOME/inward/results/exp22_probe_20261029"
exec 9>submission.lock
flock -n 9 || { echo 'Submission in progress'; exit 1; }
[ ! -e submitted_job.txt ] || { echo 'Receipt exists. Stop: do not resubmit.'; exit 1; }
[ ! -e ../exp21_latent_emergence_20261022/submitted_job.txt ] || { echo 'exp21 has a receipt. Inspect it before any merged rerun.'; exit 1; }
sha256sum -c SHA256SUMS
python3 verify_exp21.py
eval "$(conda shell.bash hook)"
conda activate inward
export PYTHONDONTWRITEBYTECODE=1
lquota
# Quota is account-specific: user supplies verified CURRENT available GiB from lquota.
# Never infer quota headroom from filesystem-wide df alone.
python - "${1:-}" <<'PY'
import sys,json,shutil
from pathlib import Path
import torch,transformers,peft,numpy,matplotlib,sklearn,scipy
try: free=float(sys.argv[1])
except ValueError:raise SystemExit('Usage: bash submit.sh AVAILABLE_QUOTA_GIB (read current lquota above). Need >=14 GiB; no submission made.')
assert free>=14,'Insufficient quota reserve: need >=14 GiB. No automatic deletion.'
assert shutil.disk_usage('.').free>=14*1024**3,'Filesystem free space <14 GiB'
v=json.loads(Path('LOCAL_VALIDATION.json').read_text());assert v['real_06b_smoke30_passed'] and v['synthetic_checks_passed'] and v['final_code_analysis_passed']
assert Path('../../models/Qwen3-14B/config.json').exists()
for name,info in json.loads(Path('ADAPTER_MANIFEST.json').read_text()).items():
 p=Path('../exp12_matched785_20261015/runs_701792')/info['relative_path'];assert (p/'adapter_model.safetensors').is_file() and (p/'adapter_config.json').is_file(),str(p)
Path('quota_attestation.json').write_text(json.dumps({'user_attested_free_gib':free,'required_gib':14})+'\n')
print('Smoke, dependencies, quota and file checks passed; weight hashes checked on GPU job.')
PY
jobs=$(squeue -h -u "$(id -un)" -o '%i')
count=$(printf '%s\n' "$jobs" | awk 'NF {n++} END {print n+0}')
if [ "$count" -ge 2 ]; then echo "Already $count queued/running jobs. Stop."; exit 1; fi
if [ "$count" -eq 1 ]; then
 job=$(sbatch --parsable --chdir="$HOME/inward" "--dependency=afterany:$jobs" "$PWD/run_combined.sh")
else
 job=$(sbatch --parsable --chdir="$HOME/inward" "$PWD/run_combined.sh")
fi
printf '%s\n' "$job" > submitted_job.txt
printf 'Actual submitted job ID: %s\n' "$job"
squeue -j "${job%%;*}"
