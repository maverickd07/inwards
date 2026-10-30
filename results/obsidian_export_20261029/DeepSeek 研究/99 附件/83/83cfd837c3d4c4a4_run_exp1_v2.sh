#!/bin/bash
#SBATCH -J exp1v2
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%j.out

# Usage: sbatch run_exp1_v2.sh Qwen3-0.6B Qwen3-1.7B
# Runs experiment 1 on each listed model, one after another, inside a single job.

eval "$(conda shell.bash hook)"
conda activate inward
export HF_HUB_OFFLINE=1
cd ~/inward

for m in "$@"; do
    echo "===== $m ====="
    python exp1_gap_v2.py --model models/$m --out results/exp1v2_$m
done
