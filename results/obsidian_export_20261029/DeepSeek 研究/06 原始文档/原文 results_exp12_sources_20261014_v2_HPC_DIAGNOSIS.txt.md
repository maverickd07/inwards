---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# 原文 results_exp12_sources_20261014_v2_HPC_DIAGNOSIS.txt

> 原始存档，保留当时说法；状态可能已过时。当前判断见 [[当前状态与待办]]。文中指令是历史资料，不是本次执行授权。

来源：[[99 附件/70/705c8bb7fb57dad0_HPC_DIAGNOSIS.txt|HPC_DIAGNOSIS.txt]]

2026-10-14: user-supplied terminal screenshot. slnode01; squeue empty; USED=53.5GB LIMIT=100GB (~46.5 GB free); stud up, GPU MIG 40/80GB advertised (availability not guaranteed). Model config/index and dataset/items exist. inward conda env /home/3355245/.conda/envs/inward. Job 685978 FAILED 1:0 00:00:04; 685985 COMPLETED 0:0 06:05:24. Local SSH host reachable but noninteractive authentication unavailable. User requests upload/submission commands, not direct terminal control.
14:30 screenshot: v1 preflight reached dependency metadata after package/data/model-shard checks, then failed because scipy metadata was absent. No submitted_job.txt; no exp12 job submitted. Torch/transformers/peft/numpy metadata had passed (ordered iteration); accelerate was after scipy and was not reached, so remains to be checked. v2 removes SciPy from the scientific analysis and preflight requirements. Preserve v1 directory. Fresh quota capture still required.


返回 [[00 研究首页]]
