#!/bin/bash
set -euo pipefail
cd "$HOME/inward"
experiment_dir=results/exp12_rescore_v3_20261015
exec 9>"$experiment_dir/submission.lock"
flock -n 9 || { echo "Another submission is in progress"; exit 1; }
if [ -e "$experiment_dir/submitted_job.txt" ] || [ -e "$experiment_dir/quota_ticket.json" ] || [ -e "$experiment_dir/submission_attempt.txt" ]; then
  echo "Previous submission/preflight exists. Inspect before retry; no automatic repeat."
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate inward
jobs=$(squeue -h -u "$(id -un)" -o '%i')
count=$(printf '%s\n' "$jobs" | awk 'NF {n++} END {print n+0}')
if [ "$count" -ge 2 ]; then
  echo "Already $count queued/running jobs; no submission."
  exit 1
fi
python "$experiment_dir/preflight.py" --quota-file "$experiment_dir/quota_login.txt"
date -u > "$experiment_dir/submission_attempt.txt"
if [ "$count" -eq 1 ]; then
  job=$(sbatch --parsable "--dependency=afterany:$jobs" "$experiment_dir/run_rescore.sh")
else
  job=$(sbatch --parsable "$experiment_dir/run_rescore.sh")
fi
printf '%s\n' "$job" > "$experiment_dir/submitted_job.txt"
echo "Submitted CPU-only v3 scoring: $job"
squeue -j "${job%%;*}" -o '%.18i %.12P %.20j %.8u %.2t %.10M %.6D %R'
