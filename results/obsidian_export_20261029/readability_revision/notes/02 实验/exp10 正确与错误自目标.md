---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# exp10 正确与错误自目标

## 白话解释

**想问什么：** 模型自己生成的错误答案，也可能提供有用训练信号吗？

**怎么做：** 比较正确自目标、错误自目标和打乱的错误目标。

**目前怎么理解：** 部分错误自目标也有收益，但还不能由此证明是哪种内部机制起作用。

下方数字是详细证据。遇到缩写可看[[研究术语白话解释]]；先理解比较对象，再看分数。

状态：**已有结果**。

比较self_wrong、self_correct及shuffled_wrong。错误自目标在部分切分也有正收益，不能因此证明自洽机制；C上错/对组未检出差异不等于等效。完整floor/旧评分、单步保留及配对统计见证据JSON。

## 资料与复现

- [[99 附件/7c/7c73af8ee5ba0fc5_evidence.json|results/report_final_20261011/evidence.json]]
- [[99 附件/2e/2effe3310b4692af_REPORT_ZH.txt|results/expxx_self_consistency_20261016/REPORT_ZH.txt]]

相关：[[实验总表]]、[[证据与机制边界]]。

返回 [[00 研究首页]]
