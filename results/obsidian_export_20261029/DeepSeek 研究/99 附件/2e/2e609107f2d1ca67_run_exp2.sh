#!/bin/bash
#SBATCH -J lens14b
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-type=END,FAIL

eval "$(conda shell.bash hook)"
conda activate inward
export HF_HUB_OFFLINE=1
cd ~/inward
python exp2_lens.py --model models/Qwen3-14B \
    --items results/exp1v2_Qwen3-14B/items.jsonl \
    --out results/lens_Qwen3-14B
