---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# exp14 Llama跨家族复现

## 白话解释

**想问什么：** 换成Llama模型家族，同一种方法还能有效吗？

**怎么做：** 按冻结协议做另一模型家族的比较。

**目前怎么理解：** 主要干净集合的结果不确定，不能拿更有利的其他口径替代主结论。

下方数字是详细证据。遇到缩写可看[[研究术语白话解释]]；先理解比较对象，再看分数。

状态：**结果包存在 主结论不确定**。

job703622报告主clean247：context-direct +0.404858pp，六种子t95[−1.091273,1.900990]。REVIEW全对敏感性+1.1471pp、共同排除+0.7141pp。主结果不支持明确正复现；不能拿known301或allD500更有利口径替代。报告原文已保存，本次未对包进行新的逐项独立复算；盲审评分CSV另存，不在本次归档中解盲。

## 资料与复现

- [[99 附件/df/dfe0ebc03b733ad8_report.txt|results/exp14_llama31_8b_20261015/received/download_703622_VcM3eA/extracted/runs_703622/analysis_v3/report.txt]]
- [[99 附件/40/403715a8c1e42c49_PROTOCOL.txt|results/exp14_llama31_8b_20261015/PROTOCOL.txt]]

相关：[[实验总表]]、[[证据与机制边界]]。

返回 [[00 研究首页]]
