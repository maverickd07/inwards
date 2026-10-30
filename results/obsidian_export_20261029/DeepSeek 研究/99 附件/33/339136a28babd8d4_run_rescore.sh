#!/bin/bash
#SBATCH -J inward-exp12-score3
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH -o results/exp12_rescore_v3_20261015/rescore-%j.out
set -euo pipefail
cd "$HOME/inward"
eval "$(conda shell.bash hook)"
conda activate inward
export PYTHONUNBUFFERED=1
experiment_dir=results/exp12_rescore_v3_20261015
python "$experiment_dir/preflight.py" --ticket "$experiment_dir/quota_ticket.json" && \
python -u "$experiment_dir/rescore.py" --root "$HOME/inward" \
  --out "$experiment_dir/runs_${SLURM_JOB_ID}" && \
python "$experiment_dir/pack_review.py" --run "$experiment_dir/runs_${SLURM_JOB_ID}"
