---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# exp04 固定层回填

## 白话解释

**想问什么：** 不再逐题挑最好层，回填还有用吗？

**怎么做：** 先固定干预层，再用新题和随机对照衡量修复与损坏。

**目前怎么理解：** 能修好一部分，也会弄坏原本正确的题，整体净收益不理想。

下方数字是详细证据。遇到缩写可看[[研究术语白话解释]]；先理解比较对象，再看分数。

状态：**历史干预证据**。

旧交接记载修复14–16%对照3%，同时破坏5–9%原正确题，净收益为负。它支持某类内部状态干预能影响答案；本次未重新核算该批原始统计。

## 资料与复现

- [[99 附件/84/848324885c826c5c_INWARD_HANDOFF.md|INWARD_HANDOFF.md]]
- [[99 附件/cc/cc89c83f50923ec0_exp4_fixedpatch.py|exp4_fixedpatch.py]]
- [[99 附件/44/4400303721e5521c_DECISIONS.txt|results/report_final_20261011/DECISIONS.txt]]

相关：[[实验总表]]、[[证据与机制边界]]。

返回 [[00 研究首页]]
