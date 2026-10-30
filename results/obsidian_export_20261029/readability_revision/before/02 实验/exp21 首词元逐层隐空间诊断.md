---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# exp21 首词元逐层隐空间诊断

状态：**本地smoke已过 正式结果未确认**。

402题；base C0/C2和context/direct种子0、1的C0，共2412生成。prompt末位置的各层残差经final norm/unembedding读桥与终答首词元；排除重叠，never用null及L+1删失并报告never比例。0.6B10题20生成验证通过，仅是工程验证。用户上传提交，无正式回执不报告job号。

## 资料与复现

- [[99 附件/ca/ca1c7fbe8fddf83b_README_ZH.txt|results/exp21_latent_emergence_20261022/README_ZH.txt]]
- [[99 附件/fe/fe3d28226d7b7e0d_PREREGISTRATION.txt|results/exp21_latent_emergence_20261022/PREREGISTRATION.txt]]

相关：[[实验总表]]、[[证据与机制边界]]。

返回 [[00 研究首页]]
