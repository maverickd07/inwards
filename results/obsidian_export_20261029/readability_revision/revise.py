from pathlib import Path
import re,json,collections,csv,hashlib
ROOT=Path.cwd();BASE=ROOT/'results/obsidian_export_20261029';R=BASE/'readability_revision';LIVE=Path('/Users/danielweiyuchen/Documents/Obsidian Vault/DeepSeek 研究');OUT=R/'notes';OUT.mkdir(exist_ok=True)
original={str(p.relative_to(LIVE)):p.read_text() for p in LIVE.rglob('*.md') if '99 附件' not in p.parts}
notes=original.copy();renames={};used=set();gloss='研究术语白话解释'
modules=[('exp12_matched785','实验12 目标匹配'),('exp12_rescore_v3','实验12 修正评分'),('exp12_seed345','实验12 追加种子'),('exp12_sources_20261014_v2','实验12 教师来源修订包'),('exp12_sources','实验12 教师来源'),('exp12_accuracy_match','实验12 匹配可行性'),('exp12_v3_review','实验12 评分敏感性'),('exp12_blind_review','实验12 答案盲审'),('exp13','实验13 外部迁移'),('exp14_blind','实验14 答案盲审'),('exp14','实验14 跨模型复现'),('exp17_design','实验17 设计检查'),('exp17','实验17 配对资料池'),('exp19','实验19 提醒与分步'),('exp21','实验21 逐层读出'),('exp22','实验22 线性探针'),('exp11','实验11 多轮学习'),('exp10','实验10 错误目标'),('expxx','错误答案自洽诊断'),('diag_20261026','十月二十六日诊断'),('report_final','第一阶段最终报告'),('report_20261005','第一阶段早期报告'),('research_decision','研究路线决定'),('handoff_analysis','早期实验与评分审计')]
def module(s):
 for k,v in modules:
  if k in s:return v
 if 'exp1v2_Qwen3' in s:return '实验1 缺口测量修订 '+s.split('Qwen3-')[-1].replace('.md','')
 if 'exp1_Qwen3' in s:return '实验1 缺口测量初版 '+s.split('Qwen3-')[-1].replace('.md','')
 return '项目资料'
kinds=[('INTERPRETATION_CLARIFICATION','结果解读补充'),('QUESTION_IDENTITY','题目身份说明'),('ADDITIONAL_DESIGN','设计补充检查'),('PREREGISTRATION','预先写好的检验规则'),('REFERENCE_ADDENDUM','参考答案补充'),('PRECOUNT_RULES','筛选前的计数规则'),('CURATION_POLICY','人物筛选规则'),('NEW400','新增四百人物说明'),('HPC_PRESERVE','服务器结果保留检查'),('AWAITING_NEW_SCORER','等待新评分器的历史状态'),('HPC_COMMANDS','服务器操作命令'),('HPC_DIAGNOSIS','服务器故障记录'),('PACKAGE_SHA256','上传包指纹'),('archive_sha256','归档包指纹'),('RESULTS_V3_COMPARISON','新旧评分结果比较'),('RESULTS_ORIGINAL_SCORER','旧评分下的结果'),('REVIEW_SUMMARY','结果核验摘要'),('RESULTS_ZH','研究结果'),('DECISIONS','报告表述的修订决定'),('DECISION_ZH','阶段路线决定'),('SOURCE_MAP','报告数据来源'),('BUILD_README','报告重建说明'),('FILE_TREE','历史文件目录'),('START_HERE','历史目录入口'),('ANALYSIS_STDOUT','分析程序运行记录'),('PROTOCOL','实验设置与规则'),('FEASIBILITY','是否可做的检查'),('CONDITIONAL_DESIGN','条件式实验设计'),('PLAN_ZH','实验设计'),('AUDIT_ZH','资料审计'),('README','使用说明'),('unit_tests_final','最终程序测试'),('unit_tests','程序测试'),('flow_stdout','完整流程测试'),('report','结果报告'),('REPORT','结果报告')]
def kind(s):
 for k,v in kinds:
  if k in s:return v
 return '历史原文'
def explanation(k):
 return {'预先写好的检验规则':'这份文件记录看结果之前约定的预测、题集、比较方式和停止条件。它告诉你以后怎样判断成败，不代表预测已经实现。','实验设置与规则':'这份文件说明用哪个模型、哪些题、怎么运行和怎么评分。看结果时用它核对比较是否公平；它本身不是结果。','服务器操作命令':'这份文件是上传、提交和下载服务器任务的操作步骤。只在实际运行时使用；历史命令不代表现在需要再提交一次。','上传包指纹':'这里是文件的校验码，用来确认上传或下载前后有没有改变。不是模型分数，也不需要背下来。','归档包指纹':'这里记录归档文件的校验码，用于确认找到的是同一份数据。','研究结果':'这是该阶段的原始结果记录。先看本页的实验解读，再按需要核对数字与局限；旧记录中的状态可能已更新。','结果报告':'这是一次分析或运行留下的报告。必须结合它来自正式实验还是软件测试来读，不能把测试数字当成研究发现。','服务器故障记录':'记录环境、连接或任务运行的问题，用来排查工程故障，不说明研究假设是否成立。','资料审计':'检查人物、参考资料、题目或来源是否满足事先规定的条件。它说明数据怎样选出，不直接证明模型机制。','人物筛选规则':'说明哪些人物可以进入候选池、哪些需要排除，以及理由和来源。筛选后的人物不能当作完全随机样本。','结果解读补充':'补充不同结果区间应该归到哪一类判断，避免同一结果被重复解释。它不是新增实验结果。','题目身份说明':'解释题目怎样去重、匹配和追踪。同一道题或同一个人重复出现，不能算作多份独立证据。','新增四百人物说明':'介绍后续诊断使用的新人物池和筛选边界。“新增”表示本项目此前未使用，不保证模型预训练没有见过。','程序测试':'这些是检查代码是否正常工作的测试记录，不是14B模型的正式研究表现。','最终程序测试':'这是最终程序测试记录，用来确认流程可以运行，不能代替正式实验。','完整流程测试':'这是从输入到输出的流程检查。小模型或合成数据跑通，只说明工程链路可用。','分析程序运行记录':'保存离线分析执行时的输出，便于排错。理解结论请先读对应实验页。','报告数据来源':'把报告中的表格、图和数字对应到原始数据，方便追查出处。','报告重建说明':'说明怎样从代码和数据重新生成报告，只在复现排版或报告时需要。','报告表述的修订决定':'记录哪些研究说法保留、收窄或撤回，避免以后重复使用已经不成立的表述。','阶段路线决定':'记录用户在该阶段选择继续、改方向或停止的决定，以及授权边界。它比更早的计划更适合判断后续路线。','使用说明':'介绍这组文件的目的、结构和使用方式。先确认它记录的是准备阶段还是正式结果。','是否可做的检查':'在正式运行前估算数据数量和配对条件是否够用。可行并不等于实验已经成功。','条件式实验设计':'说明满足哪些前提后才考虑继续。它是一份计划，不能当作已有证据。','实验设计':'解释准备检验的问题、对照和判断规则。没有正式输出时，它只是一份设计。','历史文件目录':'这是以前某一时刻的文件清单，用来找旧资料；最新归档入口见文件导航。','历史目录入口':'指向旧资料的入口说明，保留当时状态，不能自动当作最新进度。','历史原文':'保留当时的原始记录，适合核对出处。先读当前实验解读，再回来核查细节。'}.get(k,'说明这份资料的用途与阅读方式；它不自动构成正式研究结论。')
# Concise Chinese source titles; byte-exact original attachments are never modified.
for rel,text in original.items():
 p=Path(rel)
 if p.parent.name=='06 原始文档' and p.stem.startswith('原文 '):
  old=p.stem;s=old[3:];m=module(s);k=kind(s)
  specials={'AGENTS.md':'研究合作规则原文','INWARD_HANDOFF.md':'早期研究交接原文','commands_history.txt':'早期终端命令记录','inward_commands_history.txt':'早期工作副本命令记录'}
  title=specials.get(s,m+' '+k)
  if 'AGENTS_before' in s:title='十月十六日前的合作规则'
  if 'local_validation' in s:title+=' 本地测试'
  if 'synthetic' in s:title+=' 合成数据测试'
  if 'verification_712816' in s:title='实验19 正式结果核验'
  elif 'exp19' in s and s.endswith('_report.txt'):title='实验19 尚未运行时的旧记录'
  if '_package_' in s:title+=' 打包版本'
  if '_upload_verification_' in s:title+=' 上传校验副本'
  if '_v1.' in s:title+=' 旧版'
  if 'english_' in s:title+=' 英文版'
  if title in used:
   date=re.findall(r'20\d{6}',s);title+=' '+(date[-1] if date else '补充记录')
  if title in used:title+=' '+str(sum(x.startswith(title) for x in used)+1)
  used.add(title);new=str(p.with_name(title+'.md'));renames[old]=title
  text=text.replace('# '+old,'# '+title,1)
  intro=explanation(k)
  if 'exp19' in s and s.endswith('_report.txt') and 'verification' not in s:intro='这是实验19尚未运行时保存的旧状态。后来job712816已经完成并核验，当前结论请看实验19解读。保留它只是为了追踪历史。'
  if 'AGENTS' in s:intro='记录你与助手合作时的约定：按有限阶段推进、保留结果、对机制主张要求直接证据，以及由你上传和提交HPC任务。'
  if 'INWARD_HANDOFF' in s:intro='这是十月三日的早期项目交接，帮助理解研究起点。后续实验已经推进，里面的计划和部分结论不再是当前状态。'
  if 'synthetic' in s or 'local_validation' in s:intro+=' 本页属于工程验证，不能当作正式14B效果。'
  text=re.sub(r'(# [^\n]+\n)',lambda z:z[0]+'\n## 白话解释\n\n'+intro+'\n\n**什么时候看：** 需要核对这类原始记录时再读下面全文。看不懂的词见[['+gloss+']]。\n',text,count=1)
  notes.pop(rel);notes[new]=text
# Human titles for file directories.
for rel,text in original.items():
 p=Path(rel)
 if p.parent.name!='90 文件索引':continue
 old=p.stem;s=old[3:];title=module(s)+' 文件与数据'
 if s=='inwards 历史副本':title='完整历史副本 文件与数据'
 elif s=='inward 历史副本':title='早期工作副本 文件与数据'
 elif s=='public-release 历史副本':title='对外发布副本 文件与数据'
 elif s=='根目录与申请材料':title='项目脚本和申请材料 文件与数据'
 if title in used:title+=' '+s[-8:]
 used.add(title);renames[old]=title;notes.pop(rel);notes[str(p.with_name(title+'.md'))]=text.replace('# '+old,'# '+title,1)
# PDF extraction labels.
pdfnames={'Chen_Weiyu_CV_DeepSeek_CN':'研究申请简历','Chen_Weiyu_DeepSeek_Frontier':'研究申请简述','Chen_Weiyu_Research_Progress_DeepSeek_Frontier_CN':'研究进展旧报告','Chen_Weiyu_Research_Proposal_DeepSeek_CN':'研究申请提案','Terminal DeepSeek Conjecture Experiments':'早期完整终端记录','inward_report':'早期中文实验报告','inward_report_en':'早期英文实验报告'}
for rel,text in original.items():
 p=Path(rel)
 if not p.stem.startswith('PDF文字 '):continue
 title=pdfnames.get(p.stem[6:],p.stem[6:])+' PDF文字';renames[p.stem]=title;notes.pop(rel);notes[str(p.with_name(title+'.md'))]=text.replace('# '+p.stem,'# '+title,1)
# Replace conversation exports from saved complete visible messages, avoiding short-ID collisions.
for k in list(notes):
 if k.startswith('05 对话记忆/对话 20'):notes.pop(k)
cs=json.loads((BASE/'build/project_conversations.json').read_text());cl=[]
for c in cs:
 msgs=[m for m in c['messages'] if not m['text'].startswith(('# AGENTS.md','<environment_context>','<external_codex_apps_open_page>'))]
 cid=c['id'];title={'01a18742-7429-7c13-8725-9f4917de9193':'讨论隐空间诊断与实验22','01a16346-01df-7440-82fe-02cd995516e2':'准备实验12的答案盲审','01a16346-1e2b-72c1-80ce-c82a373ec1d4':'讨论实验19并核对结果','01a16315-efe1-7e63-968f-bbd220163dfe':'回顾实验14与盲审材料','01a1639e-76e3-7ca1-8a9b-f37f4d60e7df':'设计实验21的逐层诊断'}[cid]
 old='对话 '+c['timestamp'][:10]+' '+cid[:8];renames.setdefault(old,title)
 body=f'# {title}\n\n## 白话解释\n\n这是与本项目直接相关的一次历史讨论，保留你的原始要求和助手当时的可见回答。它用于回忆为什么这样做；最新进展仍看[[当前状态与待办]]。\n\n聊天ID：`{cid}`\n\n'
 for m in msgs:body+='## '+('你的消息' if m['role']=='user' else '助手当时的回答')+'\n\n'+m['text']+'\n\n'
 notes['05 对话记忆/'+title+'.md']=body;cl.append(f'- [[{title}]]：'+c['timestamp'][:10]+'的讨论。')
notes['05 对话记忆/对话记忆导航.md']='# 对话记忆导航\n\n这里按讨论主题找回你当时的要求和决定。五次聊天各自保存，不用记长编号。\n\n'+'\n'.join(cl)
# Plain-language explanation on every experiment.
exp={
'exp01':('到底有多少题是两步分开会答，合起来却答错？','分别问人物出生国、国家属性，再直接问组合问题。比较三种回答。','这种失败确实存在；模型变大有所缓解。评分规则和题目分母会改变比例。'),
'exp02':('模型答错时，内部有没有出现过出生国的信息？','把每层隐藏状态临时转换成词语排名，观察出生国是否进入前十。','许多失败题仍能读出相关词语，但读得到不等于模型真的用上了它。'),
'exp03':('把较后层的信息搬回较早层，能不能修好答案？','每题试多个回填层，再看最好一层的效果。','有修复现象，但逐题挑最好层会夸大效果，应结合下一实验。'),
'exp04':('不再逐题挑最好层，回填还有用吗？','先固定干预层，再用新题和随机对照衡量修复与损坏。','能修好一部分，也会弄坏原本正确的题，整体净收益不理想。'),
'exp05':('只在模型没把握的时候回填，能否少弄坏正确题？','先用输出置信度判断风险，再决定是否回填。','识别没把握的题有一定能力，但净收益仍接近零。'),
'exp06':('不提供标准答案，只用模型自己的答案训练，组合能力会提高吗？','让模型先生成出生国，再据此答题，把得到的答案教给直接回答的模型。','原任务上有改善；具体改善多少要看切分、种子和评分版本。'),
'exp06b':('自训练后的提高，是不是因为出生国更早出现？','比较训练前后内部词语的出现层。','早期结果不支持简单的“更早出现就更好”解释。'),
'exp07':('先前测到的缺口和内部信号，有多少受到评分和样本选择影响？','重查标签、词元重叠，并在同一批题上公平比较模型规模。','需要修正部分早期说法，不能宣称规模完全没有帮助。'),
'exp08':('只保留模型最有把握的训练答案，会不会比随机保留更好？','选取高置信度的一部分答案，与相同数量的随机答案比较。','答案标签更正确不一定带来更好的迁移，不能只看挑出来的最好组。'),
'exp09':('训练的帮助是不是只来自多看了一些答案文字？','把答案与题目的对应关系打乱，并做按类别匹配的筛选对照。','题目与答案的对应关系很重要；打乱造成的强破坏不能等同于普通训练扰动。'),
'exp10':('模型自己生成的错误答案，也可能提供有用训练信号吗？','比较正确自目标、错误自目标和打乱的错误目标。','部分错误自目标也有收益，但还不能由此证明是哪种内部机制起作用。'),
'exp11':('反复生成新目标再训练，收益能否持续？','比较从原模型重新训练、接着训练和不更新目标但增加训练量。','各路线变化不同；保留所有轮次，不能挑最好轮或声称全部已经饱和。'),
'exp12':('带出生国背景生成的教师答案更有效，会不会只是因为它更正确？','比较两种答案来源，并尽量匹配目标的自动评分正确率。','匹配后仍有同域优势，但目标真实质量和训练扰动差异没有完全排除。'),
'exp13':('原任务学到的改善，换成另一套多跳问题还能保留吗？','不再训练，在固定的MuSiQue闭卷题集上比较模型。','整体迁移没有达到预设成功标准；少数题改善不能替代整体结果。'),
'exp14':('换成Llama模型家族，同一种方法还能有效吗？','按冻结协议做另一模型家族的比较。','主要干净集合的结果不确定，不能拿更有利的其他口径替代主结论。'),
'exp17':('错误答案之间的“自洽”差异，能否成为可检验的解释？','整理人物和国家属性，设计严格配对，并准备先只推理的资料池。','已有线索和准备材料；不能把设计或数据池当作已证明的训练机制。'),
'exp19':('只提醒模型想一想出生国，能否像明确写出中间步骤那样有效？','同题比较直接答、提醒、等长无关提醒、显式两步和给正确国家五种条件。','简单提醒没有显示特异优势，明确写出两步明显有帮助。'),
'exp21':('修好组合题时，出生国词语是否更早能从隐藏层读出来？','比较基础模型与两种适配器在各层对桥实体和最终答案的词元排名。','本归档只有准备与小模型测试；正式14B结果尚未确认。'),
'exp22':('信息是否在人物位置存在，却没有有效传到答案生成位置？','在两个位置提取隐藏状态，用单独的线性分类器读国家和答案，并设打乱标签对照。','这是待检验的诊断设计；探针读得出仍不等于证明模型在因果使用。')}
for rel in list(notes):
 if not rel.startswith('02 实验/'):continue
 eid=Path(rel).stem.split(' ')[0]
 if eid not in exp:continue
 q,how,take=exp[eid];intro=f'## 白话解释\n\n**想问什么：** {q}\n\n**怎么做：** {how}\n\n**目前怎么理解：** {take}\n\n下方数字是详细证据。遇到缩写可看[[{gloss}]]；先理解比较对象，再看分数。\n\n'
 notes[rel]=re.sub(r'(# [^\n]+\n\n)',lambda z:z[0]+intro,notes[rel],count=1)
# Every archive entry gets a role explanation in its catalog.
def purpose(path):
 s=path.lower();name=Path(path).name.lower();ext=Path(path).suffix
 if 'smoke' in s or 'synthetic' in s or 'local_validation' in s:prefix='工程测试，不能当正式研究效果；'
 else:prefix=''
 if 'blinded_review' in name:return '盲审题目与匿名答案；不给出组别，裁决前不要看映射。'
 if 'flip_mapping' in name:return '盲审答案对应的真实组别与种子；供完成盲审后解盲。'
 if 'grading' in name:return '已保存的盲审标注表；本次不重新裁决或解盲。'
 if any(k in name for k in ['sha256','manifest','freeze','receipt','hash']):return prefix+'文件身份、冻结状态或收据，用来核对来源与完整性。'
 if ext in ['.safetensors','.bin','.pt','.pth','.npy','.npz']:return '模型权重或数值张量，供程序加载，不是日常阅读笔记。'
 if ext in ['.tgz','.zip','.gz']:return prefix+'压缩资料包，保存完整上传或下载内容；不是额外的一次实验。'
 if ext=='.py':return prefix+'Python程序；用于执行或重算，阅读研究结论不必先读代码。'
 if ext=='.sh':return prefix+'终端或服务器运行脚本；历史脚本不要未经核对重复执行。'
 if ext in ['.png','.pdf']:
  return prefix+('图像或图表，用来展示结果或记录界面。' if ext=='.png' else '排版后的报告或资料，适合按页阅读。')
 if any(k in name for k in ['prediction','raw.jsonl','eval_','per_item','items.jsonl']):return prefix+'逐题输入、模型输出或评分记录；核对具体题目时使用。'
 if 'summary' in name or 'report' in name or 'results' in name:return prefix+'汇总统计或报告；先确认模型、题集、评分版本和对照。'
 if 'config' in name or 'protocol' in name or 'preregistration' in name:return prefix+'实验配置或预先规定的比较规则；不代表已有正结果。'
 if ext in ['.json','.jsonl','.csv','.tsv']:return prefix+'结构化数据表；字段含义结合所属实验协议读取。'
 if ext in ['.log','.out']:return prefix+'程序运行日志；用于排错或核对完成过程。'
 if ext=='.tex':return '报告排版源代码；用于重建PDF。'
 if ext in ['.txt','.md']:return prefix+explanation(kind(Path(path).name))
 return prefix+'原始辅助文件，按所属实验及格式查阅。'
manifest=json.loads((LIVE/'04 数据与复现/完整文件清单.json').read_text())
for rel in list(notes):
 if not rel.startswith('90 文件索引/'):continue
 t=notes[rel];t=t.replace('| 保存方式 | 文件 |','| 保存方式 | 文件 | 这份文件做什么 |').replace('|---|---:|---|---|','|---|---:|---|---|---|')
 lines=[]
 for line in t.splitlines():
  if line.startswith('| `'):
   src=line.split('`')[1];line+=' '+purpose(src)+' |'
  lines.append(line)
 t='\n'.join(lines)
 intro='## 白话解释\n\n这是这一阶段的档案目录，查原始数据时再用。**先看最右边“这份文件做什么”**，不要靠英文文件名猜结论。\n\n“打开快照”是已复制的文件；“原位置”是留在原项目里的大文件。SHA256是文件指纹，不是实验指标。相同内容的备份可能有多个原路径，但只保存一份附件。\n\n'
 notes[rel]=re.sub(r'(# [^\n]+\n\n)',lambda z:z[0]+intro,t,count=1)
# Plain introductions for the remaining authored overview/docs.
for rel in list(notes):
 if rel.startswith(('02 实验/','90 文件索引/','05 对话记忆/')) or rel=='00 研究首页.md':continue
 if '## 白话解释' in notes[rel]:continue
 title=Path(rel).stem
 if rel.startswith('06 原始文档/'):
  desc='这是旧资料的文字版本，供搜索和核对来源；原文里的计划与说法保留当时状态。当前判断先看实验解读。' if title!='原始文档导航' else '这里按中文主题查原报告。先读解释，再决定是否打开原始全文。'
 else:
  desc={'当前状态与待办':'先看这页，知道哪些实验已完成、哪些仍在准备，以及哪些旧状态已被后续结果更新。','研究协作记忆':'这里是你希望研究怎样推进的约定，不是模型的实验结果。','研究问题与方法':'解释整个项目想解决什么，以及为什么要比较不同教师答案。','证据与机制边界':'把已经观察到的改善与尚未证明的原因分开，避免把“有效”直接解释成“机制已清楚”。','研究决策时间线':'按时间解释为什么继续、停止或换方向，帮助找回之前决定的理由。','数据集与评分口径':'解释每一组题到底是什么、为什么不同分母的百分比不能直接混在一起比较。','HPC与复现入口':'告诉你什么时候需要服务器、去哪里找命令，以及怎样避免重复提交。','历史结果数值表':'这是旧结果的数字对照表。先看实验的白话说明，再按需要回来核对数字。','待核验问题':'这些是影响结论的未解决问题；记录它们不代表自动授权一轮新实验。','相关工作线索':'帮助找回相关论文线索；写正式论文前仍需核对原论文。','归档范围与更新方式':'说明本库实际保存了什么、哪些大文件仍在原位置，以及以后怎样更新。','文件导航':'查数据时从这里进入。每个文件夹都有中文用途说明，原始文件也有“做什么”的解释。'}.get(title,'这是主题导航，帮助从中文说明进入详细资料。')
 notes[rel]=re.sub(r'(# [^\n]+\n\n)',lambda z:z[0]+'## 白话解释\n\n'+desc+'\n\n',notes[rel],count=1)
# Update authored links to renamed documents, preserving aliases and explicit paths.
for rel in list(notes):
 t=notes[rel]
 def repl(m):
  val=m[1];parts=val.split('|',1);target=parts[0];base=Path(target).name;key=base if base in renames else base[:-3] if base.endswith('.md') else base
  if key in renames:
   new=renames[key];target=(str(Path(target).parent)+'/' if '/' in target else '')+new+('.md' if base.endswith('.md') and base not in renames else '')
   alias=parts[1] if len(parts)>1 else None
   if alias and (alias.startswith(('results/','原文 ','文件 '))):alias=new
   return '[['+target+('|'+alias if alias else '')+']]'
  return m[0]
 notes[rel]=re.sub(r'\[\[([^\]]+)\]\]',repl,t)
# Glossary and a small, separately tagged reading map.
glossrows=[('组合缺口 gap','两步单独能答对，但合起来问就答错的那部分题。'),('桥实体','连接两步的中间对象。本项目通常是人物的出生国。'),('第一跳 Q1','先问人物出生在哪个国家。'),('第二跳 Q2','给定国家，问这个国家的首都、币种或其他属性。'),('base 基础模型','还没接受本轮额外训练的原模型。'),('context 带背景教师','先提供模型自己生成的出生国，再得到教师答案；不是说测试时一定给背景。'),('direct 直接教师','直接回答组合问题得到的教师答案。'),('self 自训练','把模型自己生成的目标拿来训练模型。'),('gold 参考答案训练','用数据集参考答案作训练目标。参考答案本身也可能有歧义。'),('target 训练目标','训练时希望模型输出的答案文字。'),('adapter 或 LoRA','附加在基础模型上的少量可训练参数，保存本轮学习变化。'),('seed 种子','控制训练或抽样随机性的编号，用不同种子观察结果是否稳定。'),('pp 百分点','从67%到70%是增加3个百分点。'),('t95或95%区间','描述估计的不确定范围；需看它反映的是种子波动还是题目抽样。'),('区间跨零','当前数据不能清楚区分正效应、零效应或负效应；不等于已经证明没有作用。'),('bootstrap','重复重抽同一批题或人物，估计结果会波动多少。'),('D500与fixed303','特定冻结题集的简称，不是实验分数。303已按历史输出选过，不能叫全新独立测试集。'),('known313','按历史单步回答固定的313题，与303题有重叠，不是独立复现。'),('matched785','用于匹配比较的785条训练题，不是785道全新测试题。'),('EM 精确匹配','答案按规定规范化后，是否与参考答案完全一致。'),('F1','按词语重合程度计分，可能与完整答对的趋势不同。'),('logit lens 逻辑透镜','把每层隐藏状态投影成词语排名，观察哪些词能被读出来。'),('probe 线性探针','额外训练一个简单分类器读隐藏状态中的信息；读得到不证明模型在使用。'),('residual 隐藏残差','模型某一层保存的数值表示，不是自然语言笔记。'),('token 词元','模型处理文字的片段；一个词或人名可能由多个词元组成。'),('NLL','模型给某段答案的负对数似然，越低表示在该提示下越容易生成这段文字。'),('REVIEW','自动规则无法可靠裁定、留待人工复核的答案。'),('HISTORICAL','识别为历史币种等过时答案，主分析按冻结规则处理。'),('smoke 冒烟测试','用小样本或小模型检查程序能跑通，不是正式研究证据。'),('HPC与Slurm','远程计算集群与它的作业排队系统。'),('job号','用户实际提交后产生的任务编号。'),('SHA256 哈希','文件指纹，检查文件是否改变，不是模型分数。'),('预注册','看结果前写好预测、比较规则与停止条件，减少事后挑选解释。'),('因果证据','通过受控干预判断改变某个因素是否导致结果变化；相关或可读性本身不够。')]
notes['01 项目/'+gloss+'.md']='# '+gloss+'\n\n不用一次背完。读笔记遇到词时回来查即可。\n\n| 术语 | 白话解释 |\n|---|---|\n'+'\n'.join('| '+a+' | '+b+' |' for a,b in glossrows)
def mapnote(title,body):notes['00 导览/'+title+'.md']='---\ntags: [研究地图]\n---\n\n# '+title+'\n\n'+body+'\n'
mapnote('研究地图','整个项目在问：**模型明明会两步，为什么合起来不会？怎样改善？**\n\n按下面顺序读，每个点都是一个研究问题。\n\n- [[问题是什么]]\n- [[怎样尝试修复]]\n- [[修复有没有用]]\n- [[能不能迁移]]\n- [[失败原因是什么]]\n- [[隐空间里看什么]]\n- [[下一步怎么决定]]\n- [[资料到哪里找]]')
mapnote('问题是什么','**白话：** 单独问“某人出生国”和“该国货币”，模型都对；直接问“这个人出生国的货币”却错了。\n\n这叫知识组合缺口。先区分模型缺知识、没有组合好、参考答案或评分有问题。\n\n详读：[[研究问题与方法]]、[[exp01 组合缺口]]。\n\n接着读：[[怎样尝试修复]]。')
mapnote('怎样尝试修复','**白话：** 试过改隐藏状态、只改没把握的题，以及用模型自己生成的答案训练。\n\n后来的关键比较：带自生成出生国背景得到的教师答案，与直接回答得到的教师答案，谁更能帮助模型？\n\n详读：[[exp04 固定层回填]]、[[exp06 向内学习训练]]、[[exp12 教师来源与近似准确率匹配]]。\n\n接着读：[[修复有没有用]]。')
mapnote('修复有没有用','**白话：** 原来的任务上有稳定但不大的改善；修好了多少题，也要同时看弄坏了多少题。\n\n实验12主集合中，带背景教师相对直接教师约多答对2.48个百分点。这个优势存在，但还不能说明内部原因已经清楚。\n\n详读：[[exp12 教师来源与近似准确率匹配]]、[[证据与机制边界]]。\n\n接着读：[[能不能迁移]]。')
mapnote('能不能迁移','**白话：** 换人物、题型、数据集或模型后，还有效吗？\n\nMuSiQue外部题集整体没有达到预设迁移成功标准。Llama主要集合的复现结果不确定。局部好结果不能替代主结果。\n\n详读：[[exp13 MuSiQue外部迁移]]、[[exp14 Llama跨家族复现]]。\n\n接着读：[[失败原因是什么]]。')
mapnote('失败原因是什么','**白话：** 需要区分“不知道”“没用上”“没组合好”和“评分出了问题”。不能先认定只有一个原因。\n\n实验19显示：只提醒想一想出生国，没有优于长度对照的明确优势；把两步写出来则明显有帮助。\n\n详读：[[exp19 中性指针与显式两步]]、[[待核验问题]]。\n\n接着读：[[隐空间里看什么]]。')
mapnote('隐空间里看什么','**白话：** 看模型内部能否读出出生国，以及这些信息在人物位置和答案位置有何不同。\n\n实验21看每层词语排名；实验22用简单分类器读隐藏状态。两者都不能只凭“读得出”证明模型真的使用了信息。归档时尚未确认正式结果。\n\n详读：[[exp21 首词元逐层隐空间诊断]]、[[exp22 主语与答案位置的线性探针]]。\n\n接着读：[[下一步怎么决定]]。')
mapnote('下一步怎么决定','**白话：** 完成约定的一小阶段，再一起决定继续、改方法或停止这条解释，不无限补实验。\n\n区分“某个办法没成功”和“整个研究方向没有价值”。先保存结果与不确定性，再选择下一条路线。\n\n详读：[[当前状态与待办]]、[[研究协作记忆]]、[[研究决策时间线]]。')
mapnote('资料到哪里找','**白话：** 平时读中文解释；要查某道题或核对数字时，再打开原始档案。\n\n- [[实验总表]]：逐个实验的白话解释与详细数字。\n- [[研究术语白话解释]]：缩写和术语是什么意思。\n- [[原始文档导航]]：报告和协议的中文说明。\n- [[文件导航]]：每份数据、代码和附件做什么。\n- [[对话记忆导航]]：你当时为什么提出或确认某项工作。\n\n原始附件长文件名用于追踪，不需要从关系图逐个阅读。')
notes['00 研究首页.md']='''# DeepSeek研究 从这里开始

先打开[[研究地图]]。它只保留九个中文主题，每个主题都说明“研究在问什么”和“目前怎么理解”。

## 怎么读

1. 想快速理解：从[[问题是什么]]顺着主题读。
2. 想看某个实验：打开[[实验总表]]，先读每页的白话解释。
3. 遇到缩写：查[[研究术语白话解释]]。
4. 想核对某道题或原始数字：再去[[文件导航]]，目录最右列解释每份文件的用途。

## 现在研究到哪里

原任务的训练改善仍保留；普遍迁移与唯一内部机制尚未证明。简单提醒未达到预设预测，显式两步有帮助。本归档时实验21与22正式结果尚未确认，详见[[当前状态与待办]]。

## 为什么之前的图看不懂

之前把原文、文件目录和研究结论混在同一张图里，长文件名互相覆盖。现在关系图筛选为“研究地图”主题；详细档案仍然保存，只有需要核对时才进入。

本页和关系图用于阅读，不代表所有原始文件都要逐个看。导入是本地快照，不是实时实验看板。
'''
# Replace technical labels in source navigation with new human titles.
for rel in ['06 原始文档/原始文档导航.md','04 数据与复现/文件导航.md']:
 t=notes[rel]
 t=re.sub(r'\[\[([^\]|]+)\|[^\]]+\]\]',r'[[\1]]',t)
 notes[rel]=t
# Ensure every overview carries a useful explanation, including experiment index.
notes['02 实验/实验总表.md']=notes['02 实验/实验总表.md'].replace('# 实验总表\n\n','# 实验总表\n\n先按主题选择一个实验，打开后先读“白话解释”。**已有结果、只有设计、仅完成程序测试，是三种不同状态。** 实验编号只是历史顺序，不是成绩。\n\n')
# Stage changes and preserve complete original authored notes in project backup.
backup=R/'before';backup.mkdir(exist_ok=True)
for rel,text in original.items():p=backup/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
for rel,text in notes.items():p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.rstrip()+'\n')
# Link validation includes unchanged attachment filenames. Ignore literal links inside historical conversation blocks.
existing={str(p.relative_to(LIVE)) for p in LIVE.rglob('*') if p.is_file() and ('99 附件' in p.parts or p.suffix!='.md')};paths=set(notes)|existing;names={Path(p).name for p in paths};stems={Path(p).stem for p in paths};bad=[]
for rel,t in notes.items():
 if rel.startswith('05 对话记忆/'):continue
 for target in re.findall(r'\[\[([^\]]+)\]\]',t):
  target=target.split('|')[0].split('#')[0]
  if target and target not in paths and target+'.md' not in paths and target not in names and target not in stems:bad.append((rel,target))
assert not bad,bad[:10]
plan={'original_authored_notes':len(original),'new_authored_notes':len(notes),'renamed_notes':renames,'remove_old_paths':sorted(set(original)-set(notes)),'staged_paths':sorted(notes),'source_attachment_files_unchanged':True,'file_purpose_descriptions':len(manifest),'topic_map_nodes':9,'all_authored_links_resolve':True}
(R/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in plan.items() if k not in ['renamed_notes','remove_old_paths','staged_paths']},ensure_ascii=False,indent=2))
