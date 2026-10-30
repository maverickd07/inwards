#!/bin/bash
set -euo pipefail
cd "$HOME/inward/results/exp19_pointer_20261022"
exec 9>submission.lock
flock -n 9 || { echo 'Submission in progress'; exit 1; }
[ ! -e submitted_job.txt ] || { echo 'Receipt exists; inspect instead of resubmitting'; exit 1; }
sha256sum -c SHA256SUMS
eval "$(conda shell.bash hook)"
conda activate inward
# Must have complete full local model flow validation before submission.
python - <<'PY'
import json
x=json.load(open('LOCAL_VALIDATION.json'))
assert x['tiny_model_full_pipeline_passed'] and x['unit_tests_passed'] and x['tokenizer_all402_lengths_match']
PY
jobs=$(squeue -h -u "$(id -un)" -o '%i')
count=$(printf '%s\n' "$jobs" | awk 'NF {n++} END {print n+0}')
if [ "$count" -ge 2 ]; then echo "Already $count queued/running jobs; stop."; exit 1; fi
lquota > quota_login.txt
if [ "$count" -eq 1 ]; then
 job=$(sbatch --parsable "--dependency=afterany:$jobs" run_inference.sh)
else
 job=$(sbatch --parsable run_inference.sh)
fi
printf '%s\n' "$job" > submitted_job.txt
squeue -j "${job%%;*}"
