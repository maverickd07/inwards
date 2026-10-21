#!/bin/bash
# Run on the HPC login node after capturing fresh lquota output as quota_login.txt.
# One invocation/one job; a receipt prevents accidentally repeating this experiment.
set -euo pipefail
cd "$HOME/inward"
experiment_dir=results/exp12_seed345_20261014
exec 9>"$experiment_dir/submission.lock"
flock -n 9 || { echo "Another submission is in progress"; exit 1; }
if [ -e "$experiment_dir/submitted_job.txt" ] || [ -e "$experiment_dir/quota_ticket.json" ]; then
  echo "A previous submission or preflight ticket exists. Inspect it before any retry."
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
python "$experiment_dir/preflight.py" --quota-file "$experiment_dir/quota_login.txt" \
  --write-ticket "$experiment_dir/quota_ticket.json"
if [ "$count" -eq 1 ]; then
  echo "Serial dependency on existing job $jobs"
  job=$(sbatch --parsable "--dependency=afterany:$jobs" "$experiment_dir/run_exp12.sh")
else
  job=$(sbatch --parsable "$experiment_dir/run_exp12.sh")
fi
printf '%s\n' "$job" > "$experiment_dir/submitted_job.txt"
echo "Submitted exp12 seeds 3/4/5: $job"
squeue -j "${job%%;*}" -o '%.18i %.12P %.18j %.8u %.2t %.10M %.6D %R'
