#!/bin/bash
#SBATCH -J exp21-latent
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o results/exp21_latent_emergence_20261022/exp21-%j.out
set -euo pipefail
cd "$HOME/inward"
eval "$(conda shell.bash hook)"
conda activate inward
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
experiment_dir=results/exp21_latent_emergence_20261022
export MPLCONFIGDIR="$HOME/inward/$experiment_dir/matplotlib_cache"
sha256sum -c "$experiment_dir/SMOKE_VERIFIED.sha256" && \
python -u "$experiment_dir/run.py" --model models/Qwen3-14B --adapter-root results/exp12_matched785_20261015/runs_701792 --out "$experiment_dir/runs_${SLURM_JOB_ID}" && \
python "$experiment_dir/package_results.py" --run "$experiment_dir/runs_${SLURM_JOB_ID}"
