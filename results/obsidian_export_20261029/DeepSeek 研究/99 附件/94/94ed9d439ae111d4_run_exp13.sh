#!/bin/bash
#SBATCH -J inward-exp13-musi
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o results/exp13_musique_transfer_20261015/exp13-%j.out
set -euo pipefail
cd "$HOME/inward"
eval "$(conda shell.bash hook)"
conda activate inward
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
experiment_dir=results/exp13_musique_transfer_20261015
python "$experiment_dir/preflight.py" --ticket "$experiment_dir/quota_ticket.json" && \
python -u "$experiment_dir/run_pipeline.py" \
  --model models/Qwen3-14B \
  --source results/exp12_matched785_20261015/runs_701792 \
  --out "$experiment_dir/runs_${SLURM_JOB_ID}" && \
python "$experiment_dir/pack_review.py" --run "$experiment_dir/runs_${SLURM_JOB_ID}"
