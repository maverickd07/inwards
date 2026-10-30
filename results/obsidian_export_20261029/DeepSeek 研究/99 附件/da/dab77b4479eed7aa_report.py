"""Render already-computed observations into a Chinese report; no classification changes."""
import sys
sys.dont_write_bytecode=True
from analyze import *
P=HERE
load=lambda name:json.loads((P/name).read_text())
rows=read(P/'ALL864_ENRICHED.jsonl');tables=load('TABLES.json');vds=load('VERDICTS.json');pairs=load('PAIRED_SENSITIVITY.json');cov=load('COVERAGE.json');t10=load('EXP10_TABLES.json');counts=load('EXP17_COUNTS.json');candidates_all=read(P/'EXP17_CANDIDATES.jsonl');examples=load('EXAMPLES_10.json');world=World()
CN={'attribute_floor':'属性表（floor主口径）','attribute_round':'属性表（四舍五入敏感性）','model_exp12':'模型单步（同job700454）','model_exp1':'模型单步（较早exp1）','model_extended':'模型单步（同job缺记录时补exp1）','model_exp12_round':'模型同job（四舍五入）','model_exp1_round':'模型exp1（四舍五入）','model_extended_round':'模型扩展（四舍五入）','intersection_exp12':'属性表∩模型同job且方向相同','intersection_extended':'属性表∩模型扩展且方向相同'}
CAT={CATS[0]:'电话区号',CATS[1]:'首都',CATS[2]:'纬度',CATS[3]:'经度','all':'全部类别'}
ARM={'context':'带背景','direct':'无背景'};ZS={'all':'全部z̃','correct':'z̃正确','wrong':'z̃错误（含待复核）','strict_wrong':'z̃错误且无待复核标记','review':'z̃待复核'};TS={'all':'全部目标','correct':'目标正确','wrong':'目标错误（含待复核）','strict_wrong':'目标错误且无待复核标记','review':'目标待复核'}
f=lambda x:'NA' if x is None else f'{x:.2f}'
def lookup(m,c,scope='different404',cat='all',z='all',target='all'):
 return next(r for r in tables if (r['method'],r['condition'],r['scope'],r['category'],r['z_stratum'],r['target_stratum'])==(m,c,scope,cat,z,target))
def cell(s):return ' / '.join(f"{s['counts'][v]} ({f(s['pct'][v])}%)" for v in STATUSES)
lines=[]
def say(s=''):lines.append(s)
def header(s):say();say(s);say('─'*65)
def triplet_table(rs,labels):
 say('分层 | 组别 | n | 一致 / 矛盾 / 无法判断（各为n及占本行比例） | 目标待复核n')
 for lab,c,r in zip(labels[0],labels[1],rs):say(f"{lab} | {c} | {r['n']} | {cell(r)} | {r['target_review_n']}")

say('expxx 自洽分析（仅已保存数据）');say('日期：2026-10-16；结果目录：'+str(P));say('未跑推理，未访问HPC，未修改历史预测、评分器、标签或实验结果。')
header('1. 按事先封存规则的判定')
say('判定为“自洽是候选机制”。主定义：国家属性表；主样本：固定404道目标不同的题；各组错误目标分别作分母。')
say('带背景75/316=23.73%，无背景6/317=1.89%，差+21.84个百分点，达到预设≥20pp门槛。')
say('这是一项机制线索，尚未证明之前约2.5pp的测试提升由自洽导致，更不能量化它解释了多少提升。')
say('主口径的无法判断比例为36.71%与41.01%；未知任意赋值时差距可在[-19.17,+58.55]pp内，覆盖不足使属性表结论不能视为稳健因果证据。')
say('同job模型定义差+29.13pp，也达到门槛，但同样受覆盖限制。较早exp1与扩展来源分别为+36.73、+37.05pp；扩展来源的未知最坏界仍为+27.27pp，但跨运行的回答不能视为完全同次知识测量。')
say('按类别，属性表证据集中在首都/区号；经纬度不能笼统说成“错误目标普遍自洽”。')
say('规则原文及先写时间、SHA256见PREREGISTRATION_ZH.txt、PREREGISTRATION_FREEZE.json；本报告没有回改规则。')
header('2. 数据、评分与定义')
say('864题已逐题整理；785同题匹配子集含381题两组目标字符串完全相同、404题不同。ALL864_ENRICHED.jsonl保存原始题目、z̃、两组目标、gold、类别和所有判定。')
say('训练标签直接复用job701792的revised标签。“错误”包括needs_review；strict_wrong排除待复核，仍不是人工确认的事实错误。z̃对错使用冻结v3国家答案匹配。')
say('404题：两组目标分别正确88、87题；错误316、317题；两组同时错误261题，同时strict_wrong236题。各组错误集合不同，故另报同题共同错误对照。')
say('属性表由已有数据集birth_country/A2重建并核对完全一致，非全球完整知识库。反查用国家集合，保留共享区号和数值桶歧义；“一致”只表示兼容，不能唯一识别国家。')
say('主数值口径为历史floor；固定另报ROUND_HALF_UP四舍五入。整段解析失败或合法属性值在表里反查不到，均是无法判断，不是矛盾。')
say('模型定义以国家+属性索引基础模型Q2；不是直接拿原题真实国家Q2充当z̃国家Q2。重复记录需规范值全一致，冲突不选票。所有原始Q2证据见SAVED_Q2_EVIDENCE.json。')
say('同job700454为主，exp1单独报告，扩展版只填补同job没有记录的国家，不覆盖同job冲突。这里是一份缓存目标，没有六份独立教师样本，所以不计算种子t95。')
header('3. 已保存单步知识覆盖率（已在主统计前告知）')
say('范围 | 单步来源 | 可用n/总n（%） | 没有对应记录 | z̃无法唯一解析 | 多记录冲突')
for scope in SCOPES:
 for source in ['exp12','exp1','extended']:
  x=cov[scope][source];c=x['counts'];say(f"{scope} | {source} | {c.get('available',0)}/{x['n']} ({f(x['available_pct'])}%) | {c.get('missing_saved_q2',0)} | {c.get('z_unresolved',0)} | {c.get('saved_q2_conflicting',0)}")
say('覆盖率只检查能否取得稳定单步回答；与具体目标比较时还可能因目标本身无法解析而少判一题。未补任何推理。')
header('4. 404题：全体、正确目标、错误目标与排除待复核')
for m in ['attribute_floor','model_exp12','model_exp1','model_extended']:
 say('\n'+CN[m]);rs=[];labs=[];arms=[]
 for ts in TARGETS:
  for c in CONDS:rs.append(lookup(m,c,target=ts));labs.append(TS[ts]);arms.append(ARM[c])
 triplet_table(rs,[labs,arms])
header('5. 错误目标差距、缺失界与数值敏感性')
say('定义 | 错误口径 | 带背景一致率 | 无背景一致率 | 差pp | 可判定子集差pp | 未知任意赋值差距界pp | 判定')
for v in vds:
 say(f"{CN[v['method']]} | {TS[v['target_stratum']]} | {f(v['context']['pct']['consistent'])}% | {f(v['direct']['pct']['consistent'])}% | {f(v['delta_pp'])} | {f(v['judgeable_delta_pp'])} | [{f(v['unknown_bounds_pp'][0])}, {f(v['unknown_bounds_pp'][1])}] | {v['verdict']}")
say('可判定子集的一致率排除了unknown，但两组可判定集合仍不同，不能替代同题配对分析。下表进一步固定双方同题错误集合。')
say('定义 | 两方错误口径 | 是否只留双方可判定 | 同题n | 带背景一致n | 无背景一致n | 差pp')
for x in pairs:
 if x['method'] in ['attribute_floor','model_exp12','model_extended']:
  say(f"{CN[x['method']]} | {'strict_wrong' if x['both_strict_wrong'] else 'wrong'} | {x['jointly_judgeable']} | {x['context']['n']} | {x['context']['counts']['consistent']} | {x['direct']['counts']['consistent']} | {f(x['delta_pp'])}")
say('尤其是236道同时strict_wrong题：属性表全分母差+15.68pp，落在“不确定”区间；双方均可判定的117题差+24.79pp。主结论没有因此换分母，须同时保留这项限制。')
header('6. 按第一步z̃对错拆分（404题）')
for m in ['attribute_floor','model_exp12','model_extended']:
 say('\n'+CN[m]);rs=[];labs=[];arms=[]
 for z in ['correct','wrong','strict_wrong','review']:
  for ts in ['all','wrong']:
   for c in CONDS:rs.append(lookup(m,c,z=z,target=ts));labs.append(ZS[z]+'；'+TS[ts]);arms.append(ARM[c])
 triplet_table(rs,[labs,arms])
say('z̃错误的167题中，属性表一致为带背景75题（44.91%）、无背景6题（3.59%）；其中包括待复核第一步答案，表内已另列严格层。')
say('同job模型定义为69/167（41.32%）对4/167（2.40%）；扩展定义为88/167（52.69%）对6/167（3.59%）。这说明“错得自洽”确实存在，但不等于所有错误目标都自洽。')
header('7. 按四类别拆分（404题）')
for m in ['attribute_floor','model_exp12','model_extended']:
 say('\n'+CN[m]);rs=[];labs=[];arms=[]
 for cat in CATS:
  for ts in ['all','correct','wrong']:
   for c in CONDS:rs.append(lookup(m,c,cat=cat,target=ts));labs.append(CAT[cat]+'；'+TS[ts]);arms.append(ARM[c])
 triplet_table(rs,[labs,arms])
say('\n错误目标一致率差（pp）：类别 | 属性表 | 模型同job | 模型扩展')
for cat in CATS:
 ds=[lookup(m,'context',cat=cat,target='wrong')['pct']['consistent']-lookup(m,'direct',cat=cat,target='wrong')['pct']['consistent'] for m in ['attribute_floor','model_exp12','model_extended']]
 say(CAT[cat]+' | '+' | '.join(f(d) for d in ds))
say('属性表首都/区号反查与真实属性对齐较强；经纬度的大量目标不落在表中，且模型可能稳定地记错该国家坐标。模型定义因此能发现“符合自己的错误单步知识”，属性表定义则未必认可。')
header('8. 全864与主785范围汇总')
for scope in ['all864','selected785']:
 say('\n'+scope);say('定义 | 目标口径 | 组别 | n | 一致 / 矛盾 / 无法判断')
 for m in ['attribute_floor','model_exp12','model_extended']:
  for ts in ['all','wrong']:
   for c in CONDS:
    x=lookup(m,c,scope=scope,target=ts);say(f"{CN[m]} | {TS[ts]} | {ARM[c]} | {x['n']} | {cell(x)}")
say('全部交叉表（范围×类别×z̃对错×目标对错×定义×组别）见TABLES.json；含785/864的分类别及严格错误层，未只保存有利分层。')
header('9. exp10：自己的错答案是否也自洽')
say('追溯exp8目标生成代码：第一步答案来自exp1v2_Qwen3-14B的pred_q1；本分析使用这一份z̃。逐条核对实际kept target_assignment与train_examples，各组228题。self_wrong/self_correct三种子名单和目标完全相同，只计一次。')
say('组别 | 种子 | 定义 | n | 一致 / 矛盾 / 无法判断')
for x in t10:
 if x['category']=='all' and x['z_stratum']=='all' and x['target_stratum']=='all' and x['method'] in ['attribute_floor','model_exp1']:
  say(f"{x['group']} | {x['seed']} | {CN[x['method']]} | {x['n']} | {cell(x)}")
say('exp10同批单步知识以exp1为主：自己的错目标138/228=60.53%一致，自己的对目标181/228=79.39%；打乱的错目标三种子仅0、3、1题一致。')
say('修正训练标签后，历史self_wrong里有1题被判正确；227题仍错，211题严格错、16题待复核。严格错211题中的模型自洽为128/211=60.66%，不是那1题重分类造成的现象。')
say('因此exp10的“自己的错答案也能有用”与自洽机制相容，且自己的错误目标远比打乱错误目标自洽；但对/错组并非同题，打乱还改变目标与问题的关联和条件概率，不能据此单独归因。')
say('\nself_wrong按类别：定义 | 类别 | n | 一致 / 矛盾 / 无法判断')
for x in t10:
 if x['group']=='self_wrong' and x['category']!='all' and x['z_stratum']=='all' and x['target_stratum']=='all' and x['method'] in ['attribute_floor','model_exp1']:
  say(f"{CN[x['method']]} | {CAT[x['category']]} | {x['n']} | {cell(x)}")
say('\nexp10既有floor结果（直接引用已有报告，不重评历史测试集）：子集 | 组别 | 基础准确率% | 三种子训练后准确率% | 平均提升pp')
old=json.loads(path('exp10_results').read_text())['scoring']['floor']['per_run']
for subset in ['A_newPerson','B_newCategory','C_bothNew']:
 for group in ['self_wrong','self_correct','shuffled_wrong']:
  rs=[r for r in old if r['subset']==subset and r['condition']==group and r['metric']=='composed']
  say(f"{subset} | {group} | {f(rs[0]['base_accuracy_pct'])} | {', '.join(f(r['trained_accuracy_pct']) for r in rs)} | {f(statistics.mean(r['difference_pp'] for r in rs))}")
say('“错答案和对答案效果相当”主要指C_bothNew：平均提升+1.93与+2.20pp；A、B并非一样，且近似数值不构成统计等效证明。')
say('逐题数据及全部分层见EXP10_ENRICHED.jsonl、EXP10_TABLES.json；覆盖和目标核对见EXP10_COVERAGE.json、EXP10_CHECKS.json。')
header('10. 十道代表例题：原文与可核对证据')
say('固定筛选：404题，两目标均strict_wrong优先，四类别轮转、类别内人物字典序。所列10题均满足双目标strict_wrong；同一定义下一组一致、一组矛盾。定义选择优先属性表，其次模型同job、再扩展。')
paired_raw={key(r):r for r in read(path('paired864'))};promptrows=[]
for r in rows:
 promptrows.append({'category':r['category'],'person':r['person'],'saved_context_prompt':paired_raw[key(r)]['context_prompt'],'training_question_clear':r['question'].replace(f"the birthplace of {r['person']}",f"the country where {r['person']} was born")})
write(P/'SOURCE_PROMPTS.jsonl',promptrows)
for n,r in enumerate(examples,1):
 m=r['definition'];say(f"\n例{n}：{r['person']}｜{CAT[r['category']]}｜{CN[m]}")
 for label,raw in [('原组合题',r['question']),('第一步题',r['Q1']),('z̃原文',r['z_tilde']),('gold第一步',r['gold_first']),('gold最终',r['gold_final']),('带背景目标原文',r['context_target']),('无背景目标原文',r['direct_target']),('原保存背景提示',paired_raw[key(r)]['context_prompt'])]:say(label+'：'+json.dumps(raw,ensure_ascii=False))
 say('z̃评分：'+json.dumps(r['z_label'],ensure_ascii=False))
 for c in CONDS:
  a=r['consistency'][m][c];say(ARM[c]+'判定：'+a['status']+'；错误标签：'+json.dumps(r['labels'][c],ensure_ascii=False))
  if m.startswith('attribute'):say('  隐含国家集合：'+json.dumps(a['implied_countries'],ensure_ascii=False))
  else:
   source=a['lookup_source'];l=world.lookup(r['category'],r['z_tilde'],source)
   distinct=sorted(set((x['question'],x['prediction']) for x in l['records']))
   say('  用于比较的已存Q2问答原文（去重展示，全部记录在证据文件）：'+json.dumps(distinct,ensure_ascii=False)+'；来源='+source)
header('11. exp17条件式设计与现成样本数（没有启动实验）')
say('候选限定：同题的两目标均strict_wrong，规范目标不同，一目标一致另一目标矛盾。每题可供两组各1个目标；n是每组题数，不是两组合计。')
say('定义 | 864/785/404各范围可用题数 | 电话区号/首都/纬度/经度 | 一致目标来自带背景/无背景')
for definition in ['attribute_floor','model_exp12','model_extended','intersection_exp12','intersection_extended']:
 x=next(x for x in counts if x['scope']=='different404' and x['definition']==definition)
 ns=[next(y['n'] for y in counts if y['scope']==s and y['definition']==definition) for s in SCOPES]
 say(f"{CN[definition]} | {'/'.join(map(str,ns))} | {'/'.join(str(x['by_category'][c]) for c in CATS)} | {x['consistent_source'].get('context',0)}/{x['consistent_source'].get('direct',0)}")
say('三个范围候选数相同：原匹配过程排除的79题没有新增满足“双错且一方一致另一方矛盾”的配对。')
source_balance=[]
for definition in ['attribute_floor','model_exp12','model_extended','intersection_exp12','intersection_extended']:
 rs=[r for r in candidates_all if r['scope']=='different404' and r['definition']==definition];bs={}
 for cat in CATS:
  c=sum(r['category']==cat and r['consistent_source']=='context' for r in rs);d=sum(r['category']==cat and r['consistent_source']=='direct' for r in rs)
  bs[cat]={'context_consistent':c,'direct_consistent':d,'max_balanced_source_pairs':2*min(c,d)}
 source_balance.append({'definition':definition,'by_category':bs,'max_balanced_source_pairs':sum(v['max_balanced_source_pairs'] for v in bs.values())})
js(P/'EXP17_SOURCE_BALANCE.json',source_balance)
say('\n额外可行性核算：如果还要求每个类别内一致目标来自两种教师来源的数量相同，现有缓存最多能保留：')
for x in source_balance:say(CN[x['definition']]+'：每组'+str(x['max_balanced_source_pairs'])+'题。')
say('这项数量核算不是修改候选判定；它揭示严重的来源混淆。两定义同job交集20题的一致目标全部来自带背景，直接做两臂训练仍可能把其他目标风格/置信度差异误叫自洽效应。')
say('\n建议的有限阶段设计：')
say('a. 若只做现成数据的机制探索，预先选择“模型自己的单步知识”为定义，可用同job70题/组；两组输入均为同一组合题，不在训练输入里加背景；一组用一致错误目标，另一组用矛盾错误目标。两组各类别数11/4/16/39、评分正确率0%，并保留人工核验后的事实正确率报告。')
say('b. 配对种子0–5、同基础模型和LoRA初始化、相同优化器与batch；若沿用两轮、有效batch8且每轮保留末尾不足batch，70题每轮9次更新、两轮18次/组。正式代码需核对累积梯度边界并封存准确步数。小样本不能靠重复更多轮数冒充更多独立训练题。')
say('c. 评价保留事先固定的D_reservedC及303题干净口径，另报三种待复核敏感性；基础模型预测复用。主对比一致错减矛盾错，六种子配对t95；预先规定正向下界>0作为这轮干预有效的标准，同时报告效应大小、逐题预测、训练损失和目标长度。不能把未显著直接当作机制不存在。')
say('d. 更严格的两定义共同支持版本只有20题/组（扩展来源28题），不宜据此马上做有说服力的因果结论。真正隔离自洽，需要在同类别错误目标中构造分配，使一致/矛盾组尽量复用相同目标词串分布并平衡来源、长度、格式；来源内配对或受约束置换是否可行需要另行预注册，不能拿现在70题就宣称这些混淆已排除。')
say('e. 当前决定：保留自洽为值得检验的候选机制；本轮只完成离线诊断，不提交exp17。先人工复核候选错误标签和目标分配可行性，再决定是否值得承担12次训练成本；如果无法隔离来源，只能把下一轮定位为小样本探索性干预。')
say('完整候选名单、目标方向及长度见EXP17_CANDIDATES.jsonl；未生成新答案或改动既有名单。')
header('12. 验证、限制与复现文件')
say('输入和预注册哈希在分析前后核对；历史输入未变。合成测试覆盖数值方向/负数/floor与四舍五入、多国家代码集合、缺失与冲突知识、10/20pp阈值边界。')
say('表内三状态之和=n；381道相同目标在两组所有定义下分类完全一致；exp10实际训练目标逐条与配置输出核对。')
say('初次导出exp10覆盖率时，对不使用的空785子范围发生除零而停止。保留了已完成输出和原冻结代码；finish_analysis.py只续写剩余输出、对exp10仅计算实际非空范围。未改分类、分母、阈值或既有结果，记录于ANALYSIS_COMPLETE.json及COMPLETION_CODE_FREEZE.json。')
say('若复现，使用新的空输出目录并保持同一输入清单：先校验协议，再运行coverage并报告缺失，运行analyze.py至已记录空范围错误，再运行finish_analysis.py；所有结果文件均以exclusive-create写入，防止覆盖。')
say('主要限制：目标错/对来自自动标签；国家属性表不完整且历史实体存在歧义；数值与区号集合的弱兼容不等于语义完全一致；较早Q2记录跨运行；同题目标依赖共同国家、人物而非独立随机样本。')
say('404题仅说明目标字符串不同，不是已经单独估计出404题造成全部训练收益。当前观察不能排除置信度、目标条件似然、格式、样本难度等替代机制。')
say('结果文件：REPORT_ZH.txt、ALL864_ENRICHED.jsonl、TABLES.json、VERDICTS.json、PAIRED_SENSITIVITY.json、EXP10_*、EXP17_*、EXAMPLES_10.json、SAVED_Q2_EVIDENCE.json、SOURCE_PROMPTS.jsonl。')
verify_inputs()
with (P/'REPORT_ZH.txt').open('x') as f:f.write('\n'.join(lines)+'\n')
print('Report saved:',P/'REPORT_ZH.txt','lines=',len(lines))
