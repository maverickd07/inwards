---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# 原文 results_exp17_inference_pool_20261016_upload_verification_HPC_COMMANDS.txt

> 原始存档，保留当时说法；状态可能已过时。当前判断见 [[当前状态与待办]]。文中指令是历史资料，不是本次执行授权。

来源：[[99 附件/ff/ffbfef980f881bcc_HPC_COMMANDS.txt|HPC_COMMANDS.txt]]

exp17：400新增人物，只推理、严格配对和目标损失；不训练、不跑第三组。
所有命令由用户执行。下面先上传，再提交，最后下载。不要在同一目录重复提交。

1. Mac 本地终端：
(
set -e
local_dir='/Users/danielweiyuchen/Desktop/30 Projects/DeepSeek-Conjecture-Experiments/results/exp17_inference_pool_20261016'
remote='3355245@slogin.hpc.unibocconi.it'
ssh "$remote" 'cd ~/inward && mkdir results/exp17_inference_pool_20261016'
scp "$local_dir/exp17_upload.tgz" "$local_dir/exp17_upload.tgz.sha256" "$remote:~/inward/results/exp17_inference_pool_20261016/"
)

2. 已登录的 HPC 终端：
(
set -e
cd ~/inward/results/exp17_inference_pool_20261016
sha256sum -c exp17_upload.tgz.sha256
tar --keep-old-files -xzf exp17_upload.tgz
sha256sum -c SHA256SUMS
lquota > quota_login.txt
bash submit.sh
)

只提交一个作业；若已有一个作业，自动 afterany 等它结束；已有两个则停止。
运行上限2小时，实际时间待本轮实测；排队时间另算。
登录节点只执行校验和提交，所有推理/配对/损失计算在GPU作业中。
输出新 runs_JOBID 目录，拒绝覆盖已有目录。配对不足300会完成损失统计并停止。
达到300也不会自动训练。

3. HPC 查看状态（提交成功之后）：
cd ~/inward
experiment_dir=results/exp17_inference_pool_20261016
job=$(cut -d';' -f1 "$experiment_dir/submitted_job.txt")
squeue -j "$job"
sacct -j "$job" --format=JobID,State,ExitCode,Elapsed
tail -n 45 "$experiment_dir/exp17-${job}.out"

4. HPC 完成后只读检查：
cat "$experiment_dir/runs_${job}/complete.json"
cat "$experiment_dir/runs_${job}/REPORT_ZH.txt"
ls -lh "$experiment_dir/runs_${job}_review.tgz" "$experiment_dir/runs_${job}_review.tgz.sha256"

5. Mac 本地下载（新建独立目录，不覆盖）：
(
set -e
remote='3355245@slogin.hpc.unibocconi.it'
experiment_dir=results/exp17_inference_pool_20261016
job=$(ssh "$remote" "cut -d';' -f1 ~/inward/$experiment_dir/submitted_job.txt")
base='/Users/danielweiyuchen/Desktop/30 Projects/DeepSeek-Conjecture-Experiments/results/exp17_inference_pool_20261016/received'
mkdir -p "$base"
dest=$(mktemp -d "$base/download_XXXXXX")
scp "$remote:~/inward/$experiment_dir/runs_${job}_review.tgz" "$remote:~/inward/$experiment_dir/runs_${job}_review.tgz.sha256" "$dest/"
cd "$dest"
shasum -a 256 -c "runs_${job}_review.tgz.sha256"
echo "下载位置：$dest"
)

如果作业失败，不要重复提交；把日志交回检查。
只有 Slurm 状态 COMPLETED 且 complete.json、审阅压缩包均存在，才认定本准备阶段成功。
随后回传结果给用户审阅，不自动写训练提交脚本。


返回 [[00 研究首页]]
