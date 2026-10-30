---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# HPC与复现入口

本库用于知识与结果归档，不执行代码。原始包中的历史提交命令可能已经执行过；不得照抄后再次提交而覆盖结果。

正式流程：本地准备及smoke → 用户检查资源和现有队列 → 用户上传 → 用户sbatch提交 → 保存真实回执 → 下载review包 → 哈希与逐题统计核验 → 有限阶段决策。

| 阶段 | 冻结命令与协议 |
|---|---|
| exp12 matched785 | [[原文 results_exp12_matched785_20261015_HPC_COMMANDS.txt]] |
| exp13 | [[原文 results_exp13_musique_transfer_20261015_HPC_COMMANDS.txt]] |
| exp14 | [[原文 results_exp14_llama31_8b_20261015_HPC_COMMANDS.txt]] |
| exp19 | [[原文 results_exp19_pointer_20261022_HPC_COMMANDS.txt]] |
| exp21 | [[原文 results_exp21_latent_emergence_20261022_HPC_COMMANDS.txt]] |

exp22合并包仍在另一个聊天准备，不从当前不完整目录发明上传或提交命令。实际job号以原始receipt/结果包和用户回执为证，不能把准备成功当提交成功。

旧环境记载：`~/inward`、conda inward、离线模型。严格保持每个包自己的依赖版本和冻结哈希，不擅自升级现有HPC环境。

- [[99 附件/5e/5e0526b0598f815f_AGENTS.md|AGENTS.md]]
- [[99 附件/88/88298f0a566629a4_HPC_COMMANDS.txt|results/exp21_latent_emergence_20261022/HPC_COMMANDS.txt]]

返回 [[00 研究首页]]
