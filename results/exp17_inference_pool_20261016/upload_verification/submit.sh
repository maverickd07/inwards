#!/bin/bash
set -euo pipefail
cd "$HOME/inward"
experiment_dir=results/exp17_inference_pool_20261016
exec 9>"$experiment_dir/submission.lock"
flock -n 9 || { echo 'Another submission is in progress'; exit 1; }
if [ -e "$experiment_dir/submitted_job.txt" ] || [ -e "$experiment_dir/quota_ticket.json" ]; then
  echo 'Submission receipt/ticket already exists. Stop and inspect; no automatic retry.'; exit 1
fi
eval "$(conda shell.bash hook)"
conda activate inward
jobs=$(squeue -h -u "$(id -un)" -o '%i')
count=$(printf '%s\n' "$jobs" | awk 'NF {n++} END {print n+0}')
if [ "$count" -ge 2 ]; then echo "Already $count queued/running jobs. No submission."; exit 1; fi
python "$experiment_dir/preflight.py" --quota-file "$experiment_dir/quota_login.txt" --write-ticket "$experiment_dir/quota_ticket.json"
if [ "$count" -eq 1 ]; then
  echo "Queuing after existing job $jobs (afterany)."
  job=$(sbatch --parsable "--dependency=afterany:$jobs" "$experiment_dir/run_inference.sh")
else
  job=$(sbatch --parsable "$experiment_dir/run_inference.sh")
fi
printf '%s\n' "$job" > "$experiment_dir/submitted_job.txt"
echo "Submitted exp17 preparation ONLY: $job"
squeue -j "${job%%;*}" -o '%.18i %.12P %.20j %.8u %.2t %.10M %.6D %R'
