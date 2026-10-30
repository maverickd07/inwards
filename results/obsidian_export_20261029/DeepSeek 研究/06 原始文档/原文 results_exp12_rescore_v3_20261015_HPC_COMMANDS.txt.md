---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# 原文 results_exp12_rescore_v3_20261015_HPC_COMMANDS.txt

> 原始存档，保留当时说法；状态可能已过时。当前判断见 [[当前状态与待办]]。文中指令是历史资料，不是本次执行授权。

来源：[[99 附件/9f/9fefcdff3f47cd43_HPC_COMMANDS.txt|HPC_COMMANDS.txt]]

1. Mac 本地终端上传（不是 HPC 窗口）：

ssh 3355245@slogin.hpc.unibocconi.it 'mkdir ~/inward/results/exp12_rescore_v3_20261015' && \
scp "/Users/danielweiyuchen/Desktop/30 Projects/DeepSeek-Conjecture-Experiments/results/exp12_rescore_v3_20261015/exp12_rescore_v3_upload.tgz" \
  3355245@slogin.hpc.unibocconi.it:~/inward/results/exp12_rescore_v3_20261015/

若目录已存在，不要删除。先核查是否已上传或提交。上传中断可单独重传同一个压缩包。

2. 已登录的 HPC 终端，解压、校验并提交一个 CPU 评分作业：

cd ~/inward && \
tar --keep-old-files -xzf results/exp12_rescore_v3_20261015/exp12_rescore_v3_upload.tgz \
  -C results/exp12_rescore_v3_20261015 && \
(cd results/exp12_rescore_v3_20261015 && sha256sum -c SHA256SUMS) && \
lquota > results/exp12_rescore_v3_20261015/quota_login.txt && \
bash results/exp12_rescore_v3_20261015/submit.sh

使用已知 stud 分区/qos，1CPU、4GB、不申请 GPU、20 分钟上限。集群若拒绝 CPU-only 请求，
把完整报错发回，不要自行重新申请 GPU 或重复提交。预检查只读哈希和配额，正式评分在计算节点。
当前只读 job700454/700972 的原预测；不生成新答案，不改标签，不训练，不覆盖旧结果。
若已有一个任务，自动 afterany 串行依赖；两个或更多则停止。提交回执及 quota_ticket 防止重复提交。

3. HPC 查看状态与报告：

cd ~/inward
if [ -s results/exp12_rescore_v3_20261015/submitted_job.txt ]; then
  job=$(cut -d';' -f1 results/exp12_rescore_v3_20261015/submitted_job.txt)
  squeue -j "$job"
  sacct -j "$job" --format=JobID,State,ExitCode,Elapsed
  if [ -f "results/exp12_rescore_v3_20261015/rescore-${job}.out" ]; then
    tail -n 20 "results/exp12_rescore_v3_20261015/rescore-${job}.out"
  fi
  if [ -f "results/exp12_rescore_v3_20261015/runs_${job}/complete.json" ]; then
    cat "results/exp12_rescore_v3_20261015/runs_${job}/complete.json"
    cat "results/exp12_rescore_v3_20261015/runs_${job}/report.txt"
  fi
else
  echo "尚无提交回执，请先查看提交报错。"
fi

4. 完成后在 Mac 本地终端下载：

scp '3355245@slogin.hpc.unibocconi.it:~/inward/results/exp12_rescore_v3_20261015/runs_*_review.tgz*' \
  "/Users/danielweiyuchen/Desktop/30 Projects/DeepSeek-Conjecture-Experiments/results/exp12_rescore_v3_20261015/received/"

报告含旧/新分数、每种子及六种子配对区间、逐题新旧标签、待复核清单。
v3 为事后修订的保守自动评分；出生地/货币年代等参考答案问题未修改。


返回 [[00 研究首页]]
