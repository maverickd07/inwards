---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# exp22 主语与答案位置的线性探针

状态：**正在其他聊天实现 尚无科学结果**。

用户2026-10-29授权：原样exp21先执行，同一Slurm allocation接exp22。2402题=旧402+新400人×5类；base与context/direct种子0、1。缓存主语末token和prompt末token各层float16残差；按人物分割，探针只在base train拟合，再固定用于所有模型。国家和最终答案值各一组，含打乱标签与类别频率基线。预测信息在主语中层可读、失败题答案位置较弱或较晚、context修复时变强或变早；均为待检验。先30题0.6Bsmoke，阶段结束共同决定，不训练骨干或LoRA。

## 资料与复现

- [[99 附件/58/583a4936ddf5082a_PREREGISTRATION.txt|results/exp22_probe_20261029/PREREGISTRATION.txt]]
- [[99 附件/3c/3c3d02b1684e0872_INPUT_SUMMARY.json|results/exp22_probe_20261029/INPUT_SUMMARY.json]]
- [[99 附件/f4/f45ce10fd319648e_REFERENCE_ADDENDUM.txt|results/exp22_probe_20261029/REFERENCE_ADDENDUM.txt]]

相关：[[实验总表]]、[[证据与机制边界]]。

返回 [[00 研究首页]]
