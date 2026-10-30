---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# 原文 results_exp13_musique_transfer_20261015_HPC_COMMANDS.txt

> 原始存档，保留当时说法；状态可能已过时。当前判断见 [[当前状态与待办]]。文中指令是历史资料，不是本次执行授权。

来源：[[99 附件/11/1171bd59c122c148_HPC_COMMANDS.txt|HPC_COMMANDS.txt]]

EXP13 MuSiQue跨数据集迁移 — 13模型，只推理，不训练

数据已冻结：2385题（两跳1229、三跳751、四跳405），每模型5176个唯一问题。
复用job701792的12份matched785适配器和现有Qwen3-14B基础模型。
此上传包包括处理好的问题/答案/评分规则，不需要在HPC下载数据、模型或安装新环境。
所有旧结果和权重保留；新结果单独写入runs_<job>。

1. Mac本地终端（不是HPC提示符）上传：
ssh 3355245@slogin.hpc.unibocconi.it 'cd ~/inward && mkdir results/exp13_musique_transfer_20261015'
scp '/Users/danielweiyuchen/Desktop/30 Projects/DeepSeek-Conjecture-Experiments/results/exp13_musique_transfer_20261015/exp13_musique_upload.tgz' 3355245@slogin.hpc.unibocconi.it:~/inward/results/exp13_musique_transfer_20261015/

如果mkdir提示目录已存在，先查看，不要覆盖或重复提交。

2. 已登录的HPC终端解压、校验、提交：
cd ~/inward && \
tar --keep-old-files -xzf results/exp13_musique_transfer_20261015/exp13_musique_upload.tgz -C results/exp13_musique_transfer_20261015 && \
(cd results/exp13_musique_transfer_20261015 && sha256sum -c SHA256SUMS) && \
lquota > results/exp13_musique_transfer_20261015/quota_login.txt && \
bash results/exp13_musique_transfer_20261015/submit.sh

出现Submitted exp13 MuSiQue transfer (13 models, no training): JOB即成功提交。
登录节点仅查小文件、模型分片存在性和配额。适配器权重哈希、推理、评分、打包都在计算节点执行。
需至少4GB配额余量；最多2个排队/运行作业，已有1个则afterany串行。申请1GPU、8CPU、64GB内存、24小时上限。
每次提交仅一个任务；有收据或预检票据时拒绝重复提交。失败或超时请保留输出诊断，不要自动重跑。

3. 查看状态（HPC）：
cd ~/inward
job=$(cut -d';' -f1 results/exp13_musique_transfer_20261015/submitted_job.txt)
squeue -j "$job"
sacct -j "$job" --format=JobID,JobName,State,ExitCode,Elapsed
 tail -n 40 "results/exp13_musique_transfer_20261015/exp13-${job}.out"

基础模型完成后，运行目录中的timing_estimate.json会给出剩余12个模型的粗略计时；不是保证耗时，也不含排队。
空队列不代表成功；应同时确认sacct COMPLETED/0:0、complete.json和review压缩包。

4. 完成后查看摘要（HPC）：
cd ~/inward
job=$(cut -d';' -f1 results/exp13_musique_transfer_20261015/submitted_job.txt)
cat "results/exp13_musique_transfer_20261015/runs_${job}/complete.json"
cat "results/exp13_musique_transfer_20261015/runs_${job}/analysis/report.txt"
ls -lh "results/exp13_musique_transfer_20261015/runs_${job}_review.tgz"*

5. 下载（Mac本地终端；本实验仅一次提交时用以下通配符）：
mkdir -p '/Users/danielweiyuchen/Desktop/30 Projects/DeepSeek-Conjecture-Experiments/results/exp13_musique_transfer_20261015/received'
scp '3355245@slogin.hpc.unibocconi.it:~/inward/results/exp13_musique_transfer_20261015/runs_*_review.tgz*' '/Users/danielweiyuchen/Desktop/30 Projects/DeepSeek-Conjecture-Experiments/results/exp13_musique_transfer_20261015/received/'

结果含全部逐题预测、单步诊断、评分、冻结规则、时长与源权重校验收据。没有新权重；原12份适配器留在原目录。
不按结果更换主集合、评分规则或增加运行。本轮只回答预定迁移问题。


返回 [[00 研究首页]]
