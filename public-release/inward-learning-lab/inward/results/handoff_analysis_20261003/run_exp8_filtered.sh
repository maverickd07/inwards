#!/bin/bash
#SBATCH -J inward-filtered
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Prepared only. No submission has been performed.
# Template time retained from run_exp2.sh; fifteen runs may require a longer allowed wall time.
# Before manual submission: check squeue -u "$USER"; total submitted jobs including this must <= 2,
# and only one may run. This single GPU allocation executes every setting serially.
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate inward
export HF_HUB_OFFLINE=1
cd ~/inward
script=results/handoff_analysis_20261003/exp8_filtered.py
run_root="results/filtered_${SLURM_JOB_ID}"
test ! -e "$run_root"
mkdir "$run_root"
run_one() {
  python "$script" --model models/Qwen3-14B \
    --items results/exp1v2_Qwen3-14B/items.jsonl --target self \
    --keep-top "$1" --selection "${3:-confidence}" --seed 0 --train-seed "$2" \
    --out "$run_root/self_${3:-confidence}_top${1}_s${2}"
}
run_one 1.0 0 && run_one 0.5 0 && run_one 0.7 0 && run_one 0.5 0 random && run_one 0.7 0 random && \
run_one 1.0 1 && run_one 0.5 1 && run_one 0.7 1 && run_one 0.5 1 random && run_one 0.7 1 random && \
run_one 1.0 2 && run_one 0.5 2 && run_one 0.7 2 && run_one 0.5 2 random && run_one 0.7 2 random
