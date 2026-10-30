---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# exp11 迭代蒸馏与预算控制

## 白话解释

**想问什么：** 反复生成新目标再训练，收益能否持续？

**怎么做：** 比较从原模型重新训练、接着训练和不更新目标但增加训练量。

**目前怎么理解：** 各路线变化不同；保留所有轮次，不能挑最好轮或声称全部已经饱和。

下方数字是详细证据。遇到缩写可看[[研究术语白话解释]]；先理解比较对象，再看分数。

状态：**阶段已结束**。

C floor三种子均值：shared R1 52.53%；reset R2/R3 52.93/53.53%；continue R2/R3 53.67/53.67%；frozen-budget R2/R3 54.00/52.73%。基线50%。保留所有轮次，不选择最好轮；不能宣称所有路线两轮后饱和或不动点机制。

## 资料与复现

- [[99 附件/08/081cdeb85514c04f_report.txt|results/exp11_review_20261011/report.txt]]
- [[99 附件/44/4400303721e5521c_DECISIONS.txt|results/report_final_20261011/DECISIONS.txt]]

相关：[[实验总表]]、[[证据与机制边界]]。

返回 [[00 研究首页]]
