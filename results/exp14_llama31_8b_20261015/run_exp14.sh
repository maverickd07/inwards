#!/bin/bash
#SBATCH -J inward-exp14-llama
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o results/exp14_llama31_8b_20261015/exp14-%j.out
set -euo pipefail
cd "$HOME/inward"
eval "$(conda shell.bash hook)"
conda activate inward
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
experiment_dir=results/exp14_llama31_8b_20261015
python "$experiment_dir/preflight.py" --ticket "$experiment_dir/quota_ticket.json"
python -u "$experiment_dir/run_pipeline.py" --model models/Llama-3.1-8B-Instruct --out "$experiment_dir/runs_${SLURM_JOB_ID}"
python "$experiment_dir/pack_review.py" --run "$experiment_dir/runs_${SLURM_JOB_ID}"
