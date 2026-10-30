#!/bin/bash
set -euo pipefail
cd "$HOME/inward/results/exp21_latent_emergence_20261022"
exec 9>submission.lock
flock -n 9 || { echo 'Submission in progress'; exit 1; }
[ ! -e submitted_job.txt ] || { echo 'Receipt exists; inspect instead of resubmitting'; exit 1; }
sha256sum -c SHA256SUMS
eval "$(conda shell.bash hook)"
conda activate inward
export PYTHONDONTWRITEBYTECODE=1
# Lightweight checks only: never load a model or run inference on login node.
python - <<'PY'
import json
from pathlib import Path
import torch,transformers,peft,numpy,matplotlib
v=json.loads(Path('LOCAL_VALIDATION.json').read_text())
assert v['real_pretrained_qwen3_06b_10_items_passed'] and v['all20_final_top1_verified'] and v['full_analysis_and_plots_passed'] and v['unit_tests_passed']
for name,info in json.loads(Path('ADAPTER_MANIFEST.json').read_text()).items():
 path=Path('../exp12_matched785_20261015/runs_701792')/info['relative_path']
 assert (path/'adapter_model.safetensors').is_file(),str(path)
 assert (path/'adapter_config.json').is_file(),str(path)
assert Path('../../models/Qwen3-14B/config.json').is_file()
print('Local smoke gate and remote dependency/file checks passed')
PY
jobs=$(squeue -h -u "$(id -un)" -o '%i')
count=$(printf '%s\n' "$jobs" | awk 'NF {n++} END {print n+0}')
if [ "$count" -ge 2 ]; then echo "Already $count queued/running jobs; stop."; exit 1; fi
lquota > quota_login.txt
cat quota_login.txt
if [ "$count" -eq 1 ]; then
 job=$(sbatch --parsable "--dependency=afterany:$jobs" run_inference.sh)
else
 job=$(sbatch --parsable run_inference.sh)
fi
printf '%s\n' "$job" > submitted_job.txt
squeue -j "${job%%;*}"
