---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# 原文 results_exp12_seed345_20261014_HPC_COMMANDS.txt

> 原始存档，保留当时说法；状态可能已过时。当前判断见 [[当前状态与待办]]。文中指令是历史资料，不是本次执行授权。

来源：[[99 附件/4e/4ec859a7c537f909_HPC_COMMANDS.txt|HPC_COMMANDS.txt]]

1. 在 Mac 新开的本地终端执行（不是已登录的 HPC 窗口）：

ssh 3355245@slogin.hpc.unibocconi.it 'mkdir ~/inward/results/exp12_seed345_20261014' && \
scp "/Users/danielweiyuchen/Desktop/30 Projects/DeepSeek-Conjecture-Experiments/results/exp12_seed345_20261014/exp12_seed345_upload.tgz" \
  3355245@slogin.hpc.unibocconi.it:~/inward/results/exp12_seed345_20261014/

mkdir 会在目录已存在时停止。若上传中断，只需重传压缩包；不要删除原目录或重复提交。

2. 在已登录的 HPC 终端执行：

cd ~/inward && \
tar --keep-old-files -xzf results/exp12_seed345_20261014/exp12_seed345_upload.tgz \
  -C results/exp12_seed345_20261014 && \
(cd results/exp12_seed345_20261014 && sha256sum -c SHA256SUMS) && \
lquota > results/exp12_seed345_20261014/quota_login.txt && \
bash results/exp12_seed345_20261014/submit.sh

预检查需原目录 results/exp12_sources_20261014_v2/runs_700454 完整存在；校验原训练目标、
D 名单、预测和环境。已存在提交回执或 quota_ticket 会停止，避免重复提交。
只有一个旧任务时自动串行依赖；两个及以上时停止。新任务不会重新生成教师目标或基础预测。

3. 查看状态（HPC 终端）：

cd ~/inward
if [ -s results/exp12_seed345_20261014/submitted_job.txt ]; then
  job=$(cut -d';' -f1 results/exp12_seed345_20261014/submitted_job.txt)
  squeue -j "$job"
  sacct -j "$job" --format=JobID,JobName,State,ExitCode,Elapsed
  if [ -f "results/exp12_seed345_20261014/exp12-${job}.out" ]; then
    tail -n 30 "results/exp12_seed345_20261014/exp12-${job}.out"
  fi
else
  echo "尚无提交回执，请先查看提交时的错误。"
fi

队列为空不等于成功；检查 sacct 和日志，不要因此重复提交。

4. 完成后只查看状态与原评分器数字（HPC 终端）：

cd ~/inward
job=$(cut -d';' -f1 results/exp12_seed345_20261014/submitted_job.txt)
cat "results/exp12_seed345_20261014/runs_${job}/complete.json"
cat "results/exp12_seed345_20261014/runs_${job}/analysis_original_scorer/report.txt"

5. 下载（在 Mac 本地终端执行）：

scp '3355245@slogin.hpc.unibocconi.it:~/inward/results/exp12_seed345_20261014/runs_*_review.tgz*' \
  "/Users/danielweiyuchen/Desktop/30 Projects/DeepSeek-Conjecture-Experiments/results/exp12_seed345_20261014/received/"

下载包含原基础模型及全部十二次训练的 D 逐题预测、配置、原评分摘要和日志，便于以后统一重算。
adapter 权重保存在 HPC，本次不下载。不要删除或覆盖原始结果。


返回 [[00 研究首页]]
