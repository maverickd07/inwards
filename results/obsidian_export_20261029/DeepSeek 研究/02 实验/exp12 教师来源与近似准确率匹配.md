---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# exp12 教师来源与近似准确率匹配

状态：**已完成并有本地核验**。

matched785 job701792，context/direct各6种子，共12次训练。主303：base67.6568%，context71.0671%，direct68.5919%；差+2.475248pp，t95[1.827289,3.123206]。目标自动接受360/785与359/785；45个身份待核验、目标token差6.27%。削弱简单准确率解释，未证明机制。源系列、seed345、v3重评分、敏感性及匹配版本全部保留。

## 资料与复现

- [[99 附件/c1/c1a04b2906af4454_RESULTS_ZH.txt|results/exp12_matched785_20261015/analysis_701792_local/RESULTS_ZH.txt]]
- [[99 附件/49/49912ba74d4625df_RESULTS_V3_COMPARISON.txt|results/exp12_rescore_v3_20261015/analysis_701713_local/RESULTS_V3_COMPARISON.txt]]

相关：[[实验总表]]、[[证据与机制边界]]。

返回 [[00 研究首页]]
