#!/bin/bash
set -euo pipefail
cd "$HOME/inward"
eval "$(conda shell.bash hook)"
conda activate inward
python results/exp14_llama31_8b_20261015/submit.py "$@"
