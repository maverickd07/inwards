---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# exp13 MuSiQue外部迁移

## 白话解释

**想问什么：** 原任务学到的改善，换成另一套多跳问题还能保留吗？

**怎么做：** 不再训练，在固定的MuSiQue闭卷题集上比较模型。

**目前怎么理解：** 整体迁移没有达到预设成功标准；少数题改善不能替代整体结果。

下方数字是详细证据。遇到缩写可看[[研究术语白话解释]]；先理解比较对象，再看分数。

状态：**已结束并固定benchmark**。

job701955：1229道两跳主EM，base6.428%、context6.564%、direct6.347%。context-direct+0.216979pp，t95[−0.086925,0.520882]，未达预设迁移成功标准。73道known有局部正信号，但修复合并仅3道不同题；不能替代主结果。固定已观察benchmark，不调参、不自动重跑。

## 资料与复现

- [[99 附件/cc/ccfd58c9d2abf340_RESULTS_ZH.txt|results/exp13_musique_transfer_20261015/analysis_701955_local/RESULTS_ZH.txt]]
- [[99 附件/46/46d5f3c21401e23d_DECISION_ZH.txt|results/research_decision_20261016/DECISION_ZH.txt]]

相关：[[实验总表]]、[[证据与机制边界]]。

返回 [[00 研究首页]]
