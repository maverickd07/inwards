#!/bin/bash
#SBATCH -J exp21-22-probes
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o results/exp22_probe_20261029/combined-%j.out
set -euo pipefail
cd "$HOME/inward"
eval "$(conda shell.bash hook)"
conda activate inward
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
exp22_dir=results/exp22_probe_20261029
exp21_dir=results/exp21_latent_emergence_20261022
export MPLCONFIGDIR="$HOME/inward/$exp22_dir/matplotlib_cache"
mkdir -p "$exp21_dir"
(cd "$exp22_dir" && sha256sum -c SHA256SUMS) && \
(cd "$exp22_dir/frozen_exp21" && sha256sum -c SHA256SUMS && sha256sum -c SMOKE_VERIFIED.sha256) && \
timeout 6h python -u "$exp22_dir/frozen_exp21/run.py" --model models/Qwen3-14B --adapter-root results/exp12_matched785_20261015/runs_701792 --out "$exp21_dir/runs_${SLURM_JOB_ID}" && \
python "$exp22_dir/frozen_exp21/package_results.py" --run "$exp21_dir/runs_${SLURM_JOB_ID}" && \
python -u "$exp22_dir/run.py" --model models/Qwen3-14B --adapter-root results/exp12_matched785_20261015/runs_701792 --out "$exp22_dir/runs_${SLURM_JOB_ID}" && \
python "$exp22_dir/package_results.py" --run "$exp22_dir/runs_${SLURM_JOB_ID}"
