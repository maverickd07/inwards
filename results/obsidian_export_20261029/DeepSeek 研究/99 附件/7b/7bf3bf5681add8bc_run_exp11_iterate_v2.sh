#!/bin/bash
#SBATCH -J inward-exp11
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o results/handoff_analysis_20261003/exp11_iterate/exp11-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weiyu.chen2@studbocconi.it
set -euo pipefail
cd ~/inward
eval "$(conda shell.bash hook)"
conda activate inward
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
experiment_dir=results/handoff_analysis_20261003/exp11_iterate
# One job; all stages serialized. exp10 completed and must not be re-submitted.
python "$experiment_dir/preflight_v2.py" && \
python -u "$experiment_dir/run_pipeline.py" \
  --model models/Qwen3-14B \
  --items results/exp1v2_Qwen3-14B/items.jsonl \
  --data data/compositional_celebrities.json \
  --reference-evals "$experiment_dir/reference_exp8" \
  --gen-batch 8 --out "$experiment_dir/runs_${SLURM_JOB_ID}"
