---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# 原文 results_exp21_latent_emergence_20261022_README_ZH.txt

> 原始存档，保留当时说法；状态可能已过时。当前判断见 [[当前状态与待办]]。文中指令是历史资料，不是本次执行授权。

来源：[[99 附件/ca/ca1c7fbe8fddf83b_README_ZH.txt|README_ZH.txt]]

exp21：exp12 matched785 隐空间初步实验，2026-10-22

状态：本地实现和真实0.6B冒烟验证；正式HPC提交由用户操作。
用户已说明没有HPC bridge。不要再要求用户给Codex配置SSH key/agent。
没有实际提交回执前，不能声称已有作业号。

协议：PREREGISTRATION.txt；第一阶段仅seed0、1，不训练、不自动补种子。
402唯一题；每模型按fixed303/known313分别统计。它们重叠214题。
base C0/C2 + context/direct各2种子 C0，共2412次生成。
首词元logit lens只取prompt末位置；层0是embedding，层1..40是block残差。
规范化后投影；每题逐层仅存目标排名、top1 ID、出现层及生成/评分。
从未进top10存null；统计用L+1删失标记并报告hit-only与never比例。
主机制分析排除桥/终答首词元重叠；另报全题、空格前缀、排除REVIEW敏感性。
冻结exp19格式无关评分；REVIEW/HISTORICAL未决项不按模型输出改参考答案。

运行后输出在 runs_JOB/：
 per_item.jsonl：逐题全部层数值、两个前缀的桥/终答token、输出、评分
 prompts.jsonl：原始提示及token IDs
 config.json、*_model_config.json：模型/环境/来源
 summary.json：每组统计、10000次配对题目bootstrap CI、分组正确率
 report.txt：可读报告
 distribution_*_MODEL_CONDITION.png：各集合、模型条件，两目标×全体/答对/答错分布
 complete.json：只在所有2412条通过检查并完成分析后写出
 runs_JOB_review.tgz及sha256：结果及冻结协议，保留原结果

LOCAL_VALIDATION.json和local_smoke_final/只说明工程流程通过，不作为14B研究结果。
本地下载的真实Qwen3-0.6B位于/private/tmp/exp21-Qwen3-0.6B，不打包到HPC。
本地python为/private/tmp/exp19-validation-venv/bin/python。
正式14B：bf16，分配的Slurm GPU，冻结adapter权重哈希；不合并/改写权重。

上传、提交、进度、下载完整命令见HPC_COMMANDS.txt。


返回 [[00 研究首页]]
