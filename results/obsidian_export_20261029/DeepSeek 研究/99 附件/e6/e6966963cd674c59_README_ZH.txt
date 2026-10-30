exp19 pointer — 仅推理，五条件固定阶段

状态：协议、402题、评分、分析及提交包已准备；Qwen3-14B 正式结果尚未产生。
SSH 到 HPC 的免交互认证失败，未提交作业。请在已认证的终端按 HPC_COMMANDS.txt 运行。

数据：两个历史集合分别报告。D500-known313 按 exp1 原始 q1/q2 标记固定；
fixed303 原样保留，其中89题不在原始known313中。交集214，去重并集402。
fixed303 是此前结果选出的集合，不能称为全新未观察测试集。
五条件共2010次生成，单模型greedy/非思考/96 token上限，C1对照实际token长度匹配。

PREREGISTRATION.txt 保存用户预测、阈值、判定规则及证据边界。
inputs/ 保存冻结题目、原始exp1证据、国家属性和项目已归档ISO快照。
REFERENCE_AUDIT.jsonl 保存31条旧标签和ISO参考币种集合的差异；不删题，
货币按当前归档快照的身份集合评分，历史货币标HISTORICAL且错误。
这些参考差异包括增加合法币种/特殊ISO币种单位，不全部意味着旧标签错误。
本实验不以新评分重新选择 known313 或 fixed303。

C0/C1/C1-ctrl/C3 只输出最终答案，不能直接判断内部是否选错国家。
报告这些条件的真实内部错国比例为 NA；每条件均另报“答案与其他国家属性一致”
代理，保留候选国家，不能将代理当作直接机制证据。
C2 显式 Country 错误率分母是所有 C0 失败题，同时报告国家未知数及上下界。
≥50%和≤20%是用户的预注册判据，提示干预本身不能唯一识别选择/检索根因。

主指标按题配对重采样20000次，固定seed20261022；报告准确率、恢复率及95%区间，
另报C1-control配对差异。分母C3-C0<=0时恢复率未定义；bootstrap有非正分母时
不给稳定的无条件恢复率CI，只保留条件区间作诊断。恢复率不截断。
REVIEW主评分算错，另有全REVIEW算对敏感性；保持同题同分母。

本地验证：10项单元检查；真实Qwen3-14B tokenizer全402题长度检查；随机初始化
单层Qwen3约972万参数，3题x5条件完整生成/评分/20000次bootstrap。
本地微型模型输出不属于Qwen3-14B实验结果，不用于检验用户预测。
create_tiny_smoke.py 可复现该流程（需要torch/transformers/numpy及本地tokenizer）。
使用两个均不存在的新路径：
OMP_NUM_THREADS=2 python create_tiny_smoke.py --tokenizer LOCAL_QWEN_TOKENIZER --model-out NEW_TINY_MODEL --run-out NEW_LOCAL_RUN

HPC提交前核验SHA256SUMS；单个stud作业，最多2个已提交/排队，已有一个则依赖等待。
所有计算仅在GPU作业；登录节点只做轻量检查与提交。拒绝覆盖、拒绝重复提交。
完成输出见 runs_JOB/analysis/per_item.jsonl、summary.json、report.txt。
raw.jsonl、prompts.jsonl、config.json保留全部原始生成、token、完整提示和环境信息。
完成后只打包结果，不自动训练、不启动下一轮实验。
