---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# exp06 向内学习训练

## 白话解释

**想问什么：** 不提供标准答案，只用模型自己的答案训练，组合能力会提高吗？

**怎么做：** 让模型先生成出生国，再据此答题，把得到的答案教给直接回答的模型。

**目前怎么理解：** 原任务上有改善；具体改善多少要看切分、种子和评分版本。

下方数字是详细证据。遇到缩写可看[[研究术语白话解释]]；先理解比较对象，再看分数。

状态：**已有结果**。

self教师目标来自模型自生成背景；hops只练单步作对照，gold使用参考答案。早期C500 self种子0为49.6→54.2%、种子1→52.2%，gold56.8%。数字属早期口径；不能概括成“self是gold的一半”。

## 资料与复现

- [[99 附件/84/848324885c826c5c_INWARD_HANDOFF.md|INWARD_HANDOFF.md]]
- [[99 附件/66/667d13b6d00cb7a4_exp6_inward.py|exp6_inward.py]]
- [[99 附件/44/4400303721e5521c_DECISIONS.txt|results/report_final_20261011/DECISIONS.txt]]

相关：[[实验总表]]、[[证据与机制边界]]。

返回 [[00 研究首页]]
