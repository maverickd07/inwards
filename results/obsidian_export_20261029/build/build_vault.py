from pathlib import Path
import json,hashlib,shutil,csv,re,collections,urllib.parse,datetime
ROOT=Path.cwd(); B=ROOT/'results/obsidian_export_20261029'; V=B/'DeepSeek 研究';V.mkdir(exist_ok=False)
DATE='2026-10-29'
for d in ['01 项目','02 实验','03 证据与决策','04 数据与复现','05 对话记忆','06 原始文档','90 文件索引','99 附件']:(V/d).mkdir()
def write(rel,text):
 p=V/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.strip()+'\n')
def note(rel,title,body):write(rel,f'---\nproject: DeepSeek-Conjecture-Experiments\nupdated: {DATE}\ntags: [inward-research]\n---\n\n# {title}\n\n{body}\n\n返回 [[00 研究首页]]')
# Snapshot and deduplicate every inventoried file. Large caches and weights retain exact source paths.
inv=json.loads((B/'build/source_inventory.json').read_text());rows=[];objects={};changed=[]
ignored={'.pyc','.otf','.ttf','.aux','.fls','.fdb_latexmk','.toc','.DS_Store'}
for r in inv:
 p=ROOT/r['path'];s=p.stat()
 if s.st_size!=r['bytes'] or s.st_mtime_ns!=r['mtime_ns']:changed.append(r['path'])
 model=p.suffix in {'.safetensors','.bin','.pt','.pth','.npy','.npz'}
 exclude=model or s.st_size>32_000_000 or p.suffix in ignored
 h=hashlib.sha256()
 with p.open('rb') as f:
  while chunk:=f.read(4*1024*1024):h.update(chunk)
 sha=h.hexdigest();target=''
 if not exclude:
  if sha not in objects:
   name=re.sub(r'[^\w. -]','_',p.name)[:115]
   target=f'99 附件/{sha[:2]}/{sha[:16]}_{name}'
   q=V/target;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
   if hashlib.sha256(q.read_bytes()).hexdigest()!=sha:raise RuntimeError('copy hash mismatch')
   objects[sha]=target
  target=objects[sha]
 status='local_copy' if target else ('external_model_or_tensor' if model else 'external_large_source' if s.st_size>32_000_000 else 'external_build_artifact')
 rows.append(dict(source=r['path'],bytes=s.st_size,sha256=sha,status=status,snapshot=target,source_uri=p.as_uri(),changed_during_export=r['path'] in changed))
(V/'04 数据与复现/完整文件清单.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
with (V/'04 数据与复现/完整文件清单.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lookup={r['source']:r for r in rows}
def link(path,label=None):
 r=lookup.get(path);label=label or path
 if not r:return f'`{path}`（快照时未找到）'
 return f'[{label}](<{r["source_uri"]}>)'
def asset(path,label=None):
 r=lookup.get(path)
 if not r or not r['snapshot']:return link(path,label)
 # Absolute-within-vault wiki links work from any note folder.
 return f'[[{r["snapshot"]}|{label or Path(path).name}]]'
def sources(paths):return '\n'.join('- '+asset(p,p) for p in paths)
# Grouped source inventory, retaining original hierarchy identities through indexes.
groups=collections.defaultdict(list)
for r in rows:
 parts=Path(r['source']).parts
 if parts[0] in ['inwards','inward','public-release']:g=parts[0]+' 历史副本'
 elif parts[0]=='results' and len(parts)>1:g=parts[1]
 else:g='根目录与申请材料'
 groups[g].append(r)
idx=[]
for g,rr in sorted(groups.items()):
 name=re.sub(r'[/\\:*?"<>|]','_',g)
 title='文件 '+name;idx.append(f'- [[{title}]]：{len(rr)} 个原始路径')
 lines=['每个路径保留 SHA256；重复内容只保存一份。`external` 条目保留在项目原位置。','', '| 原始路径 | 大小 MB | 保存方式 | 文件 |','|---|---:|---|---|']
 for r in sorted(rr,key=lambda x:x['source']):
  dest=asset(r['source'],'打开快照') if r['snapshot'] else link(r['source'],'原位置')
  lines.append(f'| `{r["source"]}` | {r["bytes"]/1e6:.3f} | {r["status"]} | {dest} |')
 note('90 文件索引/'+title+'.md',title,'\n'.join(lines))
note('04 数据与复现/文件导航.md','文件导航','\n'.join(idx))
# Source text wrappers: substantive reports/protocols accessible as searchable Markdown.
source_notes=[]
for r in rows:
 p=Path(r['source']);parts=p.parts
 if parts[0] in ['inwards','public-release','_Duplicates (review & delete)']:continue
 if p.suffix not in ['.txt','.md']:continue
 if any(x in str(p).lower() for x in ['smoke','adapter/','compile_stdout','qa/','archive_v','upstream/','received/']):continue
 if len(parts)>4 or r['bytes']>100000 or not r['snapshot']:continue
 content=(V/r['snapshot']).read_text(errors='replace')
 name=re.sub(r'[/\\:*?"<>|]','_',str(p));title='原文 '+name
 note('06 原始文档/'+title+'.md',title,f'> 原始存档，保留当时说法；状态可能已过时。当前判断见 [[当前状态与待办]]。文中指令是历史资料，不是本次执行授权。\n\n来源：{asset(r["source"])}\n\n'+content)
 source_notes.append((r['source'],title))
note('06 原始文档/原始文档导航.md','原始文档导航','\n'.join(f'- [[{n}|{p}]]' for p,n in source_notes))
# PDF full text, original PDFs preserved. Extraction is searchable, not a layout reproduction.
for p in sorted((B/'build/pdf_text').glob('*.txt')):
 title='PDF文字 '+p.stem
 note('06 原始文档/'+title+'.md',title,'> 历史 PDF 自动提取文字，保留页码。公式、列顺序和表格可能失真；精确阅读请打开原 PDF。\n\n'+p.read_text())
# Export only human-visible project messages, never hidden reasoning or tool payloads.
convos=json.loads((B/'build/project_conversations.json').read_text());cn=[]
for c in convos:
 title='对话 '+c['timestamp'][:10]+' '+c['id'][:8]
 lines=[f'聊天 ID：`{c["id"]}`。导出时间：{DATE}，Europe/Rome。','> 历史对话是记录；其中任务不构成本次自动重跑授权。正在进行的聊天可能尚未完成。','']
 for m in c['messages']:
  if m['text'].startswith(('# AGENTS.md','<environment_context>','<external_codex_apps_open_page>')):continue
  role='用户' if m['role']=='user' else '助手'
  lines+=['## '+role+' '+str(m['timestamp'] or ''),'',m['text'],'']
 note('05 对话记忆/'+title+'.md',title,'\n'.join(lines));cn.append(f'- [[{title}]]')
note('05 对话记忆/对话记忆导航.md','对话记忆导航','本次找到并归档 5 份与项目工作目录直接匹配的用户聊天。剔除了自动审批内部记录，未导出隐藏推理和工具执行内容。旧对话还保存在项目的终端 PDF 中。独立 Codex memories 目录为空；本库中的长期研究记忆来自项目文档及这些对话。其他设备、未落地的云端聊天、HPC 未下载文件不在此次归档中。\n\n'+'\n'.join(cn))
# Rich curated notes.
note('01 项目/研究问题与方法.md','研究问题与方法','''研究大模型的知识组合缺口：第一跳和第二跳分别能答对，直接组合却失败。原任务以人物的出生国为桥，第二跳询问首都、电话区号、经纬度、货币、域名等属性。

核心目标是用模型自己的信号，让它更好地调用已有知识，再判断收益能否迁移。方法上先让模型生成出生国，再用这个自生成背景得到教师目标；随后把教师目标蒸馏到不带背景的组合提问中。`context` 表示目标来自带背景教师，`direct` 表示目标来自直接回答教师，不能把它们误认为评估时是否给背景。

## 需要分别回答的问题

1. 输出被判错是否来自参考答案、时间性、歧义或格式？
2. 单步知识是否可通过行为测到？两步都会只是一种行为代理。
3. 隐状态中知识是否可读？可读性不等于因果使用。
4. 哪个干预能改善最终回答，同时少破坏已有正确答案？
5. 收益是否跨人物、第二跳类别、桥接关系及模型家族？

LoRA 训练曾采用 rank16、2 epochs、学习率 1e-4；实际配置以对应冻结协议为准。早期864条、matched785以及后续只推理阶段不能合并成同一个训练设置。

相关：[[证据与机制边界]]、[[数据集与评分口径]]、[[实验总表]]。

'''+sources(['INWARD_HANDOFF.md','results/exp12_matched785_20261015/PROTOCOL.txt','results/research_decision_20261016/DECISION_ZH.txt']))
note('01 项目/研究协作记忆.md','研究协作记忆','''## 长期偏好

按有限阶段推进。达到约定实验数或足够有信息量的结果后，与用户明确选择继续、修订、拒绝具体机制或暂停；不自动无限补控制。综合正效应、负结果、不确定性、实用幅度、复现性及成本。

拒绝某个方法不等于拒绝整个研究方向。不能为了证实或证伪预想而挑结果。

## 结果保存与验证

新结果写新子目录，不覆盖历史结果；脚本修改前保留旧版。区分评估种子与训练种子。新实验有对照，并使用未参与选择的数据；历史已看过的集合明确标注。报告配对检验或区间，不把种子当独立题目。

机制性表述必须有直接证据。logit lens 排除桥答案和最终答案首词元重叠，并报告 never-hit。冒烟测试只证明软件流程，不能充当14B研究结果。

## HPC 约定

没有 agent 端 HPC bridge；用户上传并提交，助手本地准备、验证包并提供可复制命令。不得再次要求用户配置 SSH key/agent。有真实回执才报告作业号。

历史约束：Jupiter/stud、A100 40GB、最多2个排队或运行任务、最多1个运行；登录节点只做秒级轻量操作。硬件、配额和路径须以执行时检查为准，2026-10-03“只有Qwen模型、剩余配额”等记录属于旧快照。

## 已生效的阶段边界

exp11检查点后，用户已批准exp12/13；不能把旧的“禁止exp12”当现状。MuSiQue固定保留，不追着调参。exp19本轮已结束；exp21/22按已授权冻结范围准备，不自动补种子或进入修复训练。

'''+sources(['AGENTS.md','INWARD_HANDOFF.md','results/research_decision_20261016/DECISION_ZH.txt','results/exp22_probe_20261029/PREREGISTRATION.txt']))
note('01 项目/当前状态与待办.md','当前状态与待办','''截至2026-10-29的本地证据快照。当前研究主线是定位原任务知识组合失败，再选择有针对性的修复；隐空间诊断正在准备。

| 阶段 | 已确认状态 | 下一边界 |
|---|---|---|
| exp1–11 | 已有归档结果与汇总审计 | 保留全部轮次，不选C集最好轮 |
| exp12 matched785 | 已完成并有本地核验 | 保留同域正效应与目标质量限制 |
| exp13 MuSiQue | 已完成；主迁移标准未达到 | 固定已观察benchmark，不调参或自动重跑 |
| exp14 Llama | job703622结果包及报告存在 | 主clean247未给出明确正复现；盲审限制保留 |
| exp17 | 诊断设计、人物池、推理包存在 | 不能把设计和候选池当训练因果结果 |
| exp19 | job712816完整推理及本地重算已核验 | 中性指针预测未获支持，本轮结束 |
| exp21 | 0.6B冒烟通过；正式14B结果未找到 | 保持冻结协议；由用户提交 |
| exp22 | 另一聊天正在实现与准备 | 正式提交、30题smoke完成、科学结果均未在本快照核验 |

## 证据缺口

- matched785训练目标45个待复核身份及真实标签歧义仍限制机制归因。
- Qwen盲审配对已生成：主303有95对，D500有327对；没有在本次整理中重新裁决或读出盲审结论。
- exp14存在130行盲审评分CSV；存在文件不等于已完成条件解盲及内容/格式收益分解。本次只保存文件。
- exp19保留133条REVIEW输出与币种别名问题；不修改冻结主评分。
- exp21/22等待正式结果及本地核验，不能从预注册预测推导结论。

## 状态冲突的处理

exp19顶层旧report写PREPARED；后来的verification_712816报告和原始包证明已完成，应以后者为准。2026-10-26诊断报告“exp19待提交”也已过时。exp13报告末尾“exp14未启动”被用户后续报告和job703622结果覆盖。AGENTS中exp14运行中是历史状态，不能覆盖实际结果文件。

exp22目录仍由其他聊天写入，本库是快照；整理操作不控制该聊天，也不承诺自动同步后续变化。

'''+sources(['results/exp19_pointer_20261022/verification_712816/report.txt','results/exp14_llama31_8b_20261015/received/download_703622_VcM3eA/extracted/runs_703622/analysis_v3/report.txt','results/exp22_probe_20261029/PREREGISTRATION.txt']))
note('03 证据与决策/证据与机制边界.md','证据与机制边界','''| 命题 | 当前支持 | 限制 |
|---|---|---|
| 单步会答但组合失败 | 多批行为结果支持 | 单次生成、参考答案和选样影响分母 |
| 失败时桥知识可读 | 早期透镜中失败组约84%曾进top10 | 与成功组约91%的差异不证明瓶颈是“太晚” |
| context目标训练能改善原任务 | matched785主303较direct +2.475pp，六种子t95[1.827,3.123] | 303经过历史输出选择；目标质量、token和扰动强度未严格相同 |
| 作用集中于已有两步知识的gap层 | 2026-10-26缓存诊断支持 | 行为分层、非直接内部因果证据 |
| 错误目标的自洽解释训练收益 | 缓存目标兼容性有差异 | 覆盖不足、来源混淆，没有独立因果识别 |
| 简单提醒足以调用知识 | exp19未支持其冻结预测 | 不因此唯一认定“检索问题” |
| 显式中间步骤有帮助 | exp19 C2明显优于C0 | 外部计算与输出策略变化，不能倒推C0隐状态 |
| 跨数据集普遍有效 | exp13主终点未获支持 | 小known子集有局部信号，不能替代主集 |
| 跨家族稳定复现 | exp14主clean247证据不确定 | 评分敏感性明显，需保留冻结主终点 |
| 修复依靠更早的知识可读性 | 目前是exp21/22待检验预测 | 探针与透镜不是因果使用证明 |

## 已撤回或不能直接使用的说法

“规模解决不了缺口”“失败主因是想得太晚”“回填是原创方法”“self通过提前跳步起效”“self达到gold效果一半”“两轮后全部饱和”“未显著就是等效”。exp11新旧基线差也不能归因于换题或换评分：题身份顺序一致，本批评分标签一致，底层运行差异未定位。

## 2026年10月26日诊断

D303中gap41题，context平均修复15.3题（37.4%），direct2.8题（6.9%）；原本正确205题中context损失5.5题。D500 gap116题，context修复29.8、direct4.3。效果落点与“用上已有知识”相容，仍不能唯一识别机制。

MuSiQue gap题的桥接关系没有出生国；跨桥关系迁移是合理解释候选。direct目标在direct提示下NLL约0.20、context目标约3.28，训练扰动强度差异也是候选。二者均不是已证实根因。

'''+sources(['results/report_final_20261011/DECISIONS.txt','results/diag_20261026/REPORT_ZH.txt','results/expxx_self_consistency_20261016/REPORT_ZH.txt']))
note('03 证据与决策/研究决策时间线.md','研究决策时间线','''| 日期 | 决策与变化 |
|---|---|
| 2026-10-03 | 初始交接：组合缺口、透镜、回填、self训练；多项说法后来收紧 |
| 2026-10-05 | 约定做到exp11后停止扩展并共同选择路线 |
| 2026-10-11 | exp1–11最终审计；不挑最佳轮，不宣称不动点或普遍饱和 |
| 2026-10-14至15 | 用户授权exp12系列；比较教师来源、扩种子、修正评分、近似准确率匹配 |
| 2026-10-16 | exp13迁移主标准未达到；保留MuSiQue为已观察benchmark，研究转向根因诊断 |
| 2026-10-16 | 自洽缓存分析产生候选机制；exp17设计仍需处理来源与目标分布混淆 |
| 2026-10-22 | exp19完成；中性指针预测未获支持。保留更广研究方向，停止本轮提示扩展 |
| 2026-10-22 | 明确HPC由用户上传提交，不配置agent认证；exp21本地包及冒烟完成 |
| 2026-10-26 | 既有结果诊断：同域收益集中于gap；迁移覆盖与扰动强度仍是解释候选 |
| 2026-10-29 | 用户明确要求exp22探针诊断，与原样exp21放同一Slurm allocation；新增400人×5类共2000题已确认。先base与context/direct种子0、1；阶段结束共同决策 |

本次归档不启动新实验，不追加训练或自动提交。

'''+sources(['AGENTS.md','results/report_final_20261011/DECISIONS.txt','results/research_decision_20261016/DECISION_ZH.txt','results/exp22_probe_20261029/PREREGISTRATION.txt']))
note('04 数据与复现/数据集与评分口径.md','数据集与评分口径','''## 题集身份

- 原任务：Press等人的compositional celebrities；具体身份以冻结JSONL和计划清单为准。
- A：新人物；B：新第二跳类别；C：新人物加新类别。早期C500与后来的D_reservedC、D303不是同一个分母。
- matched785：785训练题，381题context/direct目标相同、404题不同；404是训练目标差异子集，不能当成测试集。
- D500：原任务保留评价500题；D303是根据历史多模型输出形成的固定子集，已经观察，非独立新测试集。
- known313按exp1历史两步输出标记；与fixed303交集214题、并集402题。两个集合不是独立复现。
- exp14自己的clean247和known301不能与Qwen的303/313混用。
- MuSiQue：固定1229道两跳题为主；三跳751、四跳405，闭卷不给文章，与阅读理解榜单不可直接比较。已观察，不再称未见测试集。
- exp17新400人物是冻结的资料筛选池，不是随机总体，也不能证明未见于预训练。exp22使用400×5类=2000题，加旧402共2402。

## 评分与不确定性

旧scorer、numeric_v2、floor和v3并存，必须把数字附到正确版本。floor尤其影响负数经纬度，不能静默混用截断或四舍五入。货币名称、代码和符号的等价关系由具体冻结规则决定；历史货币和REVIEW保留标记。

主评分通常待复核算错；全算对、共同排除为单独敏感性分析，不能把这些情景当所有真实裁决的严格上下界。不得为追求显著性改参考或重选题目。

跨训练种子的t95只反映固定集合上的种子波动，不含题目抽样、参考真值与预训练重叠不确定性。paired bootstrap按协议选择题或人物为单位；多个关系共享人物时不能增加独立人物数。

## 可复现入口

所有原始路径、字节大小、SHA256和快照位置见 [[文件导航]] 与附件清单。版本选择依冻结协议与完成标记，不以文件名最大或最新时间自动判定“正确版本”。

'''+sources(['results/exp12_matched785_20261015/analysis_701792_local/RESULTS_ZH.txt','results/exp13_musique_transfer_20261015/PROTOCOL.txt','results/exp19_pointer_20261022/verification_712816/report.txt','results/exp22_probe_20261029/PREREGISTRATION.txt']))
note('04 数据与复现/HPC与复现入口.md','HPC与复现入口','''本库用于知识与结果归档，不执行代码。原始包中的历史提交命令可能已经执行过；不得照抄后再次提交而覆盖结果。

正式流程：本地准备及smoke → 用户检查资源和现有队列 → 用户上传 → 用户sbatch提交 → 保存真实回执 → 下载review包 → 哈希与逐题统计核验 → 有限阶段决策。

| 阶段 | 冻结命令与协议 |
|---|---|
| exp12 matched785 | [[原文 results_exp12_matched785_20261015_HPC_COMMANDS.txt]] |
| exp13 | [[原文 results_exp13_musique_transfer_20261015_HPC_COMMANDS.txt]] |
| exp14 | [[原文 results_exp14_llama31_8b_20261015_HPC_COMMANDS.txt]] |
| exp19 | [[原文 results_exp19_pointer_20261022_HPC_COMMANDS.txt]] |
| exp21 | [[原文 results_exp21_latent_emergence_20261022_HPC_COMMANDS.txt]] |

exp22合并包仍在另一个聊天准备，不从当前不完整目录发明上传或提交命令。实际job号以原始receipt/结果包和用户回执为证，不能把准备成功当提交成功。

旧环境记载：`~/inward`、conda inward、离线模型。严格保持每个包自己的依赖版本和冻结哈希，不擅自升级现有HPC环境。

'''+sources(['AGENTS.md','results/exp21_latent_emergence_20261022/HPC_COMMANDS.txt']))
# Experiment notes with explicit source-backed summaries.
experiments=[
('exp01','组合缺口','已有结果','早期14B干净类别旧评分gap约34%；修正floor后804/2379=33.80%。严格共同known样本1613题中8B gap35.46%、14B23.19%，规模有帮助。不同筛选和评分下的分母不能混用。',['INWARD_HANDOFF.md','results/report_final_20261011/evidence.json']),
('exp02','逐层逻辑透镜','已有结果','排除桥和终答首词元重叠后，失败组243/288=84.38%、成功组254/278=91.37%曾进入top10。说明词元可读性普遍存在，不能由此认定模型已经因果使用或“主要想得太晚”。',['results/report_final_20261011/evidence.json','exp2_lens_v2.py']),
('exp03','逐题挑层回填','历史结果待谨慎使用','旧交接记载真回填37%对照24%；逐题挑最佳层有选择偏差。最终报告未核实回填数据不作图，数字只作历史记录，不升级为本次复核结果。',['INWARD_HANDOFF.md','exp3_backpatch.py','results/report_final_20261011/DECISIONS.txt']),
('exp04','固定层回填','历史干预证据','旧交接记载修复14–16%对照3%，同时破坏5–9%原正确题，净收益为负。它支持某类内部状态干预能影响答案；本次未重新核算该批原始统计。',['INWARD_HANDOFF.md','exp4_fixedpatch.py','results/report_final_20261011/DECISIONS.txt']),
('exp05','低置信度门控回填','历史结果','旧交接中错误识别AUROC约0.75，门控回填净收益约零。能预测错误不等于能有效修复。',['INWARD_HANDOFF.md','exp5_gated.py']),
('exp06','向内学习训练','已有结果','self教师目标来自模型自生成背景；hops只练单步作对照，gold使用参考答案。早期C500 self种子0为49.6→54.2%、种子1→52.2%，gold56.8%。数字属早期口径；不能概括成“self是gold的一半”。',['INWARD_HANDOFF.md','exp6_inward.py','results/report_final_20261011/DECISIONS.txt']),
('exp06b','训练后透镜','已有结果','旧交接记载self训练后桥出现略晚而gold提前；self提升不能解释成“学会更早跳步”。先看过滤与评分边界，后续exp21另有冻结设计。',['INWARD_HANDOFF.md','exp6b_lens.py']),
('exp07','评分与公平规模审计','已有审计','检查词元重叠、失败题、标签问题及共同样本规模比较。保留评分歧义与规模有帮助的修订，不再声称规模无法缓解缺口。',['exp7_audit.py','results/report_final_20261011/evidence.json']),
('exp08','置信度筛选','已有三种子结果','置信度能提高目标正确率，但没有在C上形成可靠优于等量随机筛选的结果。C原评分top50 confidence-random −0.60pp，t95[−4.04,2.84]；top70 −1.733pp，[−2.768,−0.699]。筛选、预算和类别构成一起看，不能只挑最佳一组。',['results/report_final_20261011/evidence.json']),
('exp09','打乱目标与类别匹配','已有结果','全量打乱目标使C原评分均值5.2%，self_full53.13%。说明输入目标对应关系重要；打乱强烈破坏不构成等强度良性扰动对照。类别匹配50% confidence52.53%、random52.47%，不能据接近均值宣布等效。',['results/report_final_20261011/evidence.json']),
('exp10','正确与错误自目标','已有结果','比较self_wrong、self_correct及shuffled_wrong。错误自目标在部分切分也有正收益，不能因此证明自洽机制；C上错/对组未检出差异不等于等效。完整floor/旧评分、单步保留及配对统计见证据JSON。',['results/report_final_20261011/evidence.json','results/expxx_self_consistency_20261016/REPORT_ZH.txt']),
('exp11','迭代蒸馏与预算控制','阶段已结束','C floor三种子均值：shared R1 52.53%；reset R2/R3 52.93/53.53%；continue R2/R3 53.67/53.67%；frozen-budget R2/R3 54.00/52.73%。基线50%。保留所有轮次，不选择最好轮；不能宣称所有路线两轮后饱和或不动点机制。',['results/exp11_review_20261011/report.txt','results/report_final_20261011/DECISIONS.txt']),
('exp12','教师来源与近似准确率匹配','已完成并有本地核验','matched785 job701792，context/direct各6种子，共12次训练。主303：base67.6568%，context71.0671%，direct68.5919%；差+2.475248pp，t95[1.827289,3.123206]。目标自动接受360/785与359/785；45个身份待核验、目标token差6.27%。削弱简单准确率解释，未证明机制。源系列、seed345、v3重评分、敏感性及匹配版本全部保留。',['results/exp12_matched785_20261015/analysis_701792_local/RESULTS_ZH.txt','results/exp12_rescore_v3_20261015/analysis_701713_local/RESULTS_V3_COMPARISON.txt']),
('exp13','MuSiQue外部迁移','已结束并固定benchmark','job701955：1229道两跳主EM，base6.428%、context6.564%、direct6.347%。context-direct+0.216979pp，t95[−0.086925,0.520882]，未达预设迁移成功标准。73道known有局部正信号，但修复合并仅3道不同题；不能替代主结果。固定已观察benchmark，不调参、不自动重跑。',['results/exp13_musique_transfer_20261015/analysis_701955_local/RESULTS_ZH.txt','results/research_decision_20261016/DECISION_ZH.txt']),
('exp14','Llama跨家族复现','结果包存在 主结论不确定','job703622报告主clean247：context-direct +0.404858pp，六种子t95[−1.091273,1.900990]。REVIEW全对敏感性+1.1471pp、共同排除+0.7141pp。主结果不支持明确正复现；不能拿known301或allD500更有利口径替代。报告原文已保存，本次未对包进行新的逐项独立复算；盲审评分CSV另存，不在本次归档中解盲。',['results/exp14_llama31_8b_20261015/received/download_703622_VcM3eA/extracted/runs_703622/analysis_v3/report.txt','results/exp14_llama31_8b_20261015/PROTOCOL.txt']),
('exp17','自洽候选与严格配对推理池','设计与包已存在','离线自洽分析中，404题错误目标按属性表一致率context75/316=23.73%、direct6/317=1.89%；覆盖不足，不能因果归因。后续exp17推理包准备新400人物、闭合置换控制目标边际分布，先封存配对再算NLL；不含训练阶段。N不足300停止，否则交用户审阅。未找到足以宣布训练因果结论的正式结果。',['results/expxx_self_consistency_20261016/REPORT_ZH.txt','results/exp17_inference_pool_20261016/package/README_ZH.txt','results/exp17_inference_pool_20261016/INTERPRETATION_CLARIFICATION_ZH.txt']),
('exp19','中性指针与显式两步','已完成并本地核验','job712816，402唯一题×5条件=2010输出，20000次配对bootstrap本地重算一致。C1恢复known313为15.9%[1.6,29.1]、fixed303为3.2%[−10.2,15.5]；对长度control准确率差均跨零。≥50%且高于control预测未获支持。C2显式两步恢复101.4%和53.2%，显著有帮助；不能唯一判定选择或检索。133条REVIEW保留。',['results/exp19_pointer_20261022/verification_712816/report.txt','results/exp19_pointer_20261022/PREREGISTRATION.txt']),
('exp21','首词元逐层隐空间诊断','本地smoke已过 正式结果未确认','402题；base C0/C2和context/direct种子0、1的C0，共2412生成。prompt末位置的各层残差经final norm/unembedding读桥与终答首词元；排除重叠，never用null及L+1删失并报告never比例。0.6B10题20生成验证通过，仅是工程验证。用户上传提交，无正式回执不报告job号。',['results/exp21_latent_emergence_20261022/README_ZH.txt','results/exp21_latent_emergence_20261022/PREREGISTRATION.txt']),
('exp22','主语与答案位置的线性探针','正在其他聊天实现 尚无科学结果','用户2026-10-29授权：原样exp21先执行，同一Slurm allocation接exp22。2402题=旧402+新400人×5类；base与context/direct种子0、1。缓存主语末token和prompt末token各层float16残差；按人物分割，探针只在base train拟合，再固定用于所有模型。国家和最终答案值各一组，含打乱标签与类别频率基线。预测信息在主语中层可读、失败题答案位置较弱或较晚、context修复时变强或变早；均为待检验。先30题0.6Bsmoke，阶段结束共同决定，不训练骨干或LoRA。',['results/exp22_probe_20261029/PREREGISTRATION.txt','results/exp22_probe_20261029/INPUT_SUMMARY.json','results/exp22_probe_20261029/REFERENCE_ADDENDUM.txt'])]
exp_index=['| 实验 | 主题 | 状态 |','|---|---|---|']
for eid,title,status,body,ss in experiments:
 nt=eid+' '+title
 note('02 实验/'+nt+'.md',nt,f'状态：**{status}**。\n\n{body}\n\n## 资料与复现\n\n'+sources(ss)+'\n\n相关：[[实验总表]]、[[证据与机制边界]]。')
 exp_index.append(f'| [[{nt}|{eid}]] | {title} | {status} |')
exp_index+=['','本地没有足够证据单独建立exp15、16、18、20的正式实验结果。编号缺口不补写。`expxx_self_consistency`为离线诊断，有单独资料而不冒充某个缺失编号。']
note('02 实验/实验总表.md','实验总表','\n'.join(exp_index))
note('03 证据与决策/待核验问题.md','待核验问题','''1. 评分层：matched785的45个待复核身份、exp19币种别名和历史标签。保留盲审来源隐藏，不因当前比较自动裁决。
2. 机制层：context相对于direct的目标条件似然、训练扰动、语义结构、自洽性尚未分离；训练标签自动匹配不是事实质量完全匹配。
3. 迁移层：出生国桥的训练能否迁移到其他桥关系仍未证明，MuSiQue保持固定。
4. 跨模型：exp14主247题区间跨零；未完成内容/答案惯例的解盲分解不能用表面全集增益代替。
5. 隐空间：exp21与exp22等待正式结果；探针线性可读不表示因果使用，跨adapter坐标变化也可能影响固定探针表现。
6. 复现层：若报告未附sacct或权重本地字节，维持原核验边界；程序complete不能等同于独立Slurm最终状态核验。

这些问题记录当前边界，不是自动追加实验清单。下一阶段完成后与用户选择一条路线。

相关：[[当前状态与待办]]、[[研究协作记忆]]。''')
note('03 证据与决策/相关工作线索.md','相关工作线索','''本页保留历史交接里的文献线索，未在本次归档中重新联网核查版本、新意或发表状态。

- Press等，2022：compositionality gap和原任务数据。
- Biran等，2024，Hopping Too Late：回填和Patchscopes；不可把回填当本项目原创。
- Snell等，2022，上下文蒸馏：self教师蒸馏的相关方法。
- Layer Order Inversion，历史记录arXiv 2601.03542：关于逐跳顺序解释的批评，题名、版本与论点需独立核查。

写论文或对外主张新意前再核对原论文；历史“预计截止日期”不是当前官方日程。

'''+sources(['INWARD_HANDOFF.md']))
# Exact, deterministic tables derived from already-audited evidence, without rerunning models.
evidence=json.loads((ROOT/'results/report_final_20261011/evidence.json').read_text());ls=['以下从既有evidence.json转成可读表，没有重新训练或推理。区间只反映固定题集上的训练种子波动。','', '## exp10组合表现的floor口径','','| 组别 | 切分 | 正确率均值 % | 相对基础增益 pp | 增益t95 |','|---|---|---:|---:|---|']
for g in evidence['exp10']['scoring']['floor']['groups']:
 if g['metric']!='composed':continue
 a=g['accuracy_pct'];z=g['gain_pp'];ls.append(f'| {g["condition"]} | {g["subset"]} | {a["mean"]:.3f} | {z["mean"]:.3f} | [{z["seed_mean_t95"][0]:.3f}, {z["seed_mean_t95"][1]:.3f}] |')
ls+=['','## exp8与exp9的C切分原评分','','| 阶段 | 条件 | 正确率均值 % |','|---|---|---:|']
for g in evidence['exp8']['groups']:
 if g['scoring']=='original' and g['subset']=='C_bothNew':ls.append(f'| exp8 | {g["selection"]} top{g["keep_top"]} | {g["mean_accuracy_pct"]:.3f} |')
for g in evidence['exp9']['groups']:
 if g['subset']=='C_bothNew':ls.append(f'| exp9 | {g["condition"]} | {g["accuracy_pct"]["mean"]:.3f} |')
ls+=['',sources(['results/report_final_20261011/evidence.json'])]
note('04 数据与复现/历史结果数值表.md','历史结果数值表','\n'.join(ls))
# Export coverage & maintenance policy.
counts=collections.Counter(r['status'] for r in rows);byte_total=sum((V/t).stat().st_size for t in objects.values())
coverage={'date':DATE,'source_root':str(ROOT),'original_file_paths':len(rows),'unique_copied_files':len(objects),'copied_bytes':byte_total,'status_counts':dict(counts),'project_conversations':len(convos),'source_text_notes':len(source_notes),'changed_during_export':changed,'cutoff':'initial inventory with per-file read timestamp; active exp22 may change later'}
(V/'04 数据与复现/归档覆盖统计.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2))
note('04 数据与复现/归档范围与更新方式.md','归档范围与更新方式',f'''本库是{DATE}本地快照，目标是保留研究知识、可复核数据和原始资料的身份，供Obsidian阅读与后续协作使用。源项目未覆盖、未删除，未启动模型或HPC作业。

## 本次覆盖

- 盘点原始文件路径：{len(rows)}。
- 按SHA256去重后实际复制：{len(objects)}份，约{byte_total/1e6:.1f}MB。
- 导出项目用户聊天：{len(convos)}份；含用户请求、确认、助手可见回复，不含隐藏推理或自动审批内部记录。
- 可搜索原始报告及协议笔记：{len(source_notes)}份；另有历史PDF自动提取文字。
- 模型权重/张量、超过32MB的大型抓取缓存和原始语料、字体/编译产物保留原位置，记录SHA256、字节数与本机链接。它们没有假装已搬入vault。
- 同内容的inwards等备份保留所有原路径映射，只存一份快照；不同字节版本分别保存。

完整映射：[[完整文件清单.csv]]、[[完整文件清单.json]]。SHA256用于找回准确文件，原位置附件要求项目目录继续存在。若把vault移到另一台机器，外部大文件也需另外迁移。

## 不完整或动态来源

独立Codex memories目录为空。仅覆盖本机找到的项目聊天、工作目录资料和终端PDF；不能声称覆盖全部历史云端对话或HPC未下载数据。exp22正在其他聊天写入，快照不能作为其最终包；源文件盘点后变化列表见[[归档覆盖统计.json]]。

本次只写本地Obsidian目录，没有配置Obsidian Sync或发布云端。整理为笔记不会自动给任何AI建立或隔离记忆权限。

## 后续更新

每次有正式结果，在新日期目录保留原包及哈希，再更新实验页、状态和决策页；不要改写旧原文。一个阶段结束先共同决策，再扩展实验。此次没有建立自动同步或后台监控。
''')
note('00 研究首页.md','DeepSeek 知识组合研究','''研究核心：模型分别知道两步事实，为什么组合时会失败；能否用自己的信号改善知识调用，以及这些改善能否迁移。

**当前判断：原任务中context教师目标有稳定但有限的收益；普遍迁移与唯一内部机制尚未证明。简单中性指针未达到预设预测；显式两步有效。下一步是已授权的隐空间诊断，exp21/22正式结果尚未确认。**

## 从这里开始

- [[当前状态与待办]]：最新进度、旧状态冲突和下一边界。
- [[研究问题与方法]]：研究问题、方法和基本概念。
- [[实验总表]]：exp1至exp22已有材料的逐项导航。
- [[证据与机制边界]]：哪些发现成立，哪些解释仍待检验。
- [[研究决策时间线]]：路线如何改变，以及每个停止点。
- [[研究协作记忆]]：用户偏好、结果保存、HPC与有限阶段约定。

## 数据与原始记录

- [[数据集与评分口径]]、[[历史结果数值表]]、[[HPC与复现入口]]。
- [[文件导航]]：逐题数据、代码、冻结协议、完整结果及历史副本。
- [[原始文档导航]]、[[对话记忆导航]]：可搜索的历史报告与对话。
- [[待核验问题]]、[[相关工作线索]]。
- [[归档范围与更新方式]]：实际复制范围、大文件原位置及快照限制。

整理日期：2026-10-29，Europe/Rome。最新判读以实验页及已核验结果为准；原始资料保留当时表述。
''')
print(json.dumps(coverage,ensure_ascii=False,indent=2))
