---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# 原文 results_exp12_sources_20261014_v2_HPC_COMMANDS_v1.txt

> 原始存档，保留当时说法；状态可能已过时。当前判断见 [[当前状态与待办]]。文中指令是历史资料，不是本次执行授权。

来源：[[99 附件/16/168c01c0c60f1e80_HPC_COMMANDS.txt|HPC_COMMANDS_v1.txt]]

1. On Mac, open a NEW local Terminal tab (not the existing SSH session):

ssh 3355245@slogin.hpc.unibocconi.it 'mkdir ~/inward/results/exp12_sources_20261014' && \
scp "/Users/danielweiyuchen/Desktop/30 Projects/DeepSeek-Conjecture-Experiments/results/exp12_sources_20261014/exp12_upload.tgz" \
  3355245@slogin.hpc.unibocconi.it:~/inward/results/exp12_sources_20261014/

mkdir intentionally fails if the destination already exists. Do not delete prior
results to retry; inspect the existing folder and any submission receipt first.

2. In the already authenticated HPC terminal:

cd ~/inward && \
tar --keep-old-files -xzf results/exp12_sources_20261014/exp12_upload.tgz \
  -C results/exp12_sources_20261014 && \
(cd results/exp12_sources_20261014 && sha256sum -c SHA256SUMS) && \
lquota > results/exp12_sources_20261014/quota_login.txt && \
bash results/exp12_sources_20261014/submit.sh

Submit guard: fewer than 2 queued/running jobs; if one exists, afterany dependency
enforces sequential execution. >=6 GB quota headroom, data/package hashes and
all model shards checked. New per-job output, no overwrite or automatic resubmit.
Quota is captured freshly on the login node and passed in a timestamped ticket
because lquota may be unavailable on compute nodes. Ticket expires after 48h.

3. Check the submitted job (login node, lightweight only):

job=$(cut -d';' -f1 results/exp12_sources_20261014/submitted_job.txt)
squeue -j "$job"
sacct -j "$job" --format=JobID,State,ExitCode,Elapsed
tail -n 30 "results/exp12_sources_20261014/exp12-${job}.out"

An empty squeue may mean finished or failed; use sacct and inspect the log. Do not
start a second run to work around a slow or queued first run.

4. After successful completion, view the summary:

cat "results/exp12_sources_20261014/runs_${job}/analysis/report.txt"
cat "results/exp12_sources_20261014/runs_${job}/complete.json"

To download a small review archive, pack only logs/configs/predictions/analysis,
excluding adapter weights. Generate it inside this new experiment folder. Keep
all original adapters and historical results; no cleanup is authorized here.


返回 [[00 研究首页]]
