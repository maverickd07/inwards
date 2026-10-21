#!/bin/bash
#SBATCH -J inward-exp12-s345
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o results/exp12_seed345_20261014/exp12-%j.out
set -euo pipefail
cd "$HOME/inward"
eval "$(conda shell.bash hook)"
conda activate inward
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
experiment_dir=results/exp12_seed345_20261014
python "$experiment_dir/preflight.py" --ticket "$experiment_dir/quota_ticket.json" && \
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; d=torch.cuda.get_device_properties(0); print(d); assert d.total_memory >= 38*1024**3, "Need a >=40GB GPU slice"' && \
python -u "$experiment_dir/run_pipeline.py" \
  --model models/Qwen3-14B \
  --source results/exp12_sources_20261014_v2/runs_700454 \
  --out "$experiment_dir/runs_${SLURM_JOB_ID}" && \
python "$experiment_dir/pack_review.py" --run "$experiment_dir/runs_${SLURM_JOB_ID}"
