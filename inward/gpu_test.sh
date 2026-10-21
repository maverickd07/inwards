#!/bin/bash
#SBATCH -J gputest
#SBATCH -p stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH -o logs/%x-%j.out

eval "$(conda shell.bash hook)"
conda activate inward
export HF_HUB_OFFLINE=1

nvidia-smi
python -c "import torch; print('CUDA ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "
from vllm import LLM, SamplingParams
llm = LLM('$HOME/inward/models/Qwen3-0.6B', gpu_memory_utilization=0.8, max_model_len=2048, enforce_eager=True)
print('OUTPUT:', llm.generate(['The capital of France is'], SamplingParams(max_tokens=8))[0].outputs[0].text)
"
