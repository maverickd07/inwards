#!/bin/bash
#SBATCH -J exp10-resume
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o results/handoff_analysis_20261003/exp10_consistency/exp10-resume-%j.out
#SBATCH --mail-type=END,FAIL
# Submit one job only. Before submission, total queued+running including this <=2,
# and no other job running. Two complete runs reused; remaining 7 runs execute serially in one GPU allocation.
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate inward
export HF_HUB_OFFLINE=1
cd ~/inward
experiment_dir=results/handoff_analysis_20261003/exp10_consistency
run_root="$experiment_dir/resumed_681332_${SLURM_JOB_ID}"
test -f "$experiment_dir/frozen_targets.jsonl"
test -f "$experiment_dir/reference_exp8/eval_C_bothNew.jsonl"
test ! -e "$run_root"
python "$experiment_dir/prepare_resume_681332.py" --previous "$experiment_dir/runs_681332" --out "$run_root"
run_one() {
  python "$experiment_dir/exp10_consistency.py" \
    --model models/Qwen3-14B \
    --items results/exp1v2_Qwen3-14B/items.jsonl \
    --data data/compositional_celebrities.json \
    --target self --condition "$1" \
    --target-cache "$experiment_dir/frozen_targets.jsonl" \
    --reference-evals "$experiment_dir/reference_exp8" \
    --seed 0 --train-seed "$2" --selection-seed "$2" \
    --epochs 2 --lr 1e-4 --rank 16 --batch 8 --n-eval 500 --n-lens 200 \
    --out "$run_root/${1}_s${2}"
}
run_one self_correct 0 && \
run_one self_wrong 1 && run_one shuffled_wrong 1 && run_one self_correct 1 && \
run_one self_wrong 2 && run_one shuffled_wrong 2 && run_one self_correct 2 && \
python "$experiment_dir/analyze_exp10.py" --runs "$run_root" --data data/compositional_celebrities.json --out "$run_root/analysis"
