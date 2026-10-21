#!/bin/bash
#SBATCH -J exp17-prep-only
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o results/exp17_inference_pool_20261016/exp17-%j.out
set -euo pipefail
cd "$HOME/inward"
eval "$(conda shell.bash hook)"
conda activate inward
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
experiment_dir=results/exp17_inference_pool_20261016
python "$experiment_dir/preflight.py" --ticket "$experiment_dir/quota_ticket.json" && \
python -c 'import torch; assert torch.cuda.is_available(); d=torch.cuda.get_device_properties(0); print(d); assert d.total_memory>=38*1024**3, "Need a >=40GB GPU slice"' && \
python -u "$experiment_dir/infer.py" --model models/Qwen3-14B --inputs "$experiment_dir/inputs" --out "$experiment_dir/runs_${SLURM_JOB_ID}" --gen-batch 8
# Deliberately no training, third-group run, resubmission or follow-on job.
