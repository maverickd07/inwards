from pathlib import Path
import json,statistics,hashlib
O=Path(__file__).resolve().parent;R=O.parents[1];D=json.loads((O/'report_data.json').read_text());S=D['sources']
def g9(c,s='C_bothNew'):return next(x for x in D['exp9']['groups'] if x['condition']==c and x['subset']==s)
def g10(c,m='composed'):return next(x for x in D['exp10']['scoring']['floor']['groups'] if x['condition']==c and x['subset']=='C_bothNew' and x['metric']==m)
def g8(c,k):return next(x for x in D['exp8']['groups'] if x['selection']==c and x['keep_top']==k and x['subset']=='C_bothNew' and x['scoring']=='original')
def meanSD(g):return f"{g['accuracy_pct']['mean']:.1f} $\\pm$ {g['accuracy_pct']['sample_sd']:.1f}"
manifest=json.loads((O/'source_sha256.json').read_text());raw=[]
for seed in range(3):
 p=f'results/handoff_analysis_20261003/exp8_680959/results/filtered_680959/self_confidence_top1.0_s{seed}/inward_summary.json';raw.append(json.loads((R/p).read_text())['evals']['C_bothNew']);manifest[p]=hashlib.sha256((R/p).read_bytes()).hexdigest()
S['S13']='results/handoff_analysis_20261003/exp8_680959/results/filtered_680959/self_confidence_top1.0_s{0,1,2}/inward_summary.json'
S['S14']='results/handoff_analysis_20261003/exp9_shuffled/exp9_shuffled.py'
manifest[S['S14']]=hashlib.sha256((R/S['S14']).read_bytes()).hexdigest()
t={}
t['GAPS']='\n'.join(f"{x['size']} & {x['modes']['floor']['known_n']} & {100*x['modes']['floor']['gap_clear']:.1f} & {100*x['modes']['floor']['gap_context']:.1f} \\\\" for x in D['gap_clean'])
t['LABELS']='\n'.join(f"{k.replace('_',r'\_')} & {v} & {100*v/D['label_summary']['sample_n']:.1f} \\\\" for k,v in D['label_summary']['counts'].items())
t['LENS']='\n'.join(f"{'失败组' if x['group']=='fail' else '成功组'} & {x['raw_n']} & {x['excluded']} & {x['hits']}/{x['n']} & {100*x['share']:.1f} \\\\" for x in D['lens_filtered'])
t['ABC']='\n'.join(f"{name} & {100*raw[0][s]['acc_base']:.1f} & {meanSD(g9('self_full',s))} & {g9('self_full',s)['gain_pp']['mean']:+.1f} \\\\" for name,s in [('A：新人物','A_newPerson'),('B：新类别','B_newCategory'),('C：两者都新','C_bothNew')]) if False else '\n'.join(f"{name} & {100*json.loads((R/S['S13'].replace('{0,1,2}','0')).read_text())['evals'][s]['acc_base']:.1f} & {meanSD(g9('self_full',s))} & {g9('self_full',s)['gain_pp']['mean']:+.1f} \\\\" for name,s in [('A：新人物','A_newPerson'),('B：新类别','B_newCategory'),('C：两者都新','C_bothNew')])
quality=D['exp8']['training_quality']
t['FILTER']=''
for c,k,label in [('confidence',1.0,'全量'),('confidence',.5,'全局置信度 50\\%'),('random',.5,'全局随机 50\\%')]:
 q=next(x for x in quality if x['selection']==c and x['keep_top']==k and x['seed']==0);g=g8(c,k)
 # Random target quality differs by seed: explicitly report seed 0 only.
 t['FILTER']+=f"{label} & {q['n_kept']} & {100*q['target_correct_share']:.1f} & {g['mean_accuracy_pct']:.1f} $\\pm$ {g['sample_sd_pp']:.1f} \\\\\n"
t['BALANCED']='\n'.join(f"{label} & 432 & --- & {meanSD(g9(c))} \\\\" for c,label in [('category_confidence50','类别内置信度 50\\%'),('category_random50','类别内随机 50\\%')])
t['CONTROLS']='\n'.join(f"{label} & {meanSD(g10(c))} & {g10(c)['gain_pp']['mean']:+.1f} & {g10(c,'hop1')['accuracy_pct']['mean']:.1f} & {g10(c,'hop2')['accuracy_pct']['mean']:.1f} \\\\" for c,label in [('self_wrong','self 错答案'),('self_correct','self 对答案'),('shuffled_wrong','打乱错答案')])
t['Q1']=f"{100*raw[0]['q1_base']:.1f}\\% → {100*statistics.mean(x['q1_trained'] for x in raw):.1f}\\%";t['Q2']=f"{100*raw[0]['q2_base']:.1f}\\% → {100*statistics.mean(x['q2_trained'] for x in raw):.1f}\\%"
t['PAIR']=f"{D['scale_pair']['old']['n']}";t['PAIR8']=f"{100*D['scale_pair']['old']['8B']:.1f}";t['PAIR14']=f"{100*D['scale_pair']['old']['14B']:.1f}";t['PAIRF']=f"{D['scale_pair']['floor']['n']}";t['PAIR8F']=f"{100*D['scale_pair']['floor']['8B']:.1f}";t['PAIR14F']=f"{100*D['scale_pair']['floor']['14B']:.1f}"
t['COMMON']=str(D['scale_common3'][0]['n']);t['AUROC']=f"{quality[0]['auroc']:.3f}";t['SELF']=meanSD(g9('self_full'));t['SHUFF']=meanSD(g9('shuffled_full'));t['HOPS']=f"{D['hops_pending']:.1f}";t['GOLD']=f"{D['gold_pending']:.1f}"
pair=next(x for x in D['exp10']['scoring']['floor']['comparisons'] if x['condition']=='self_wrong' and x['control']=='self_correct' and x['subset']=='C_bothNew' and x['metric']=='composed')
t['DIFF']=f"{pair['difference_pp']['mean']:+.2f}";t['DIFFCI']=f"[{pair['difference_pp']['seed_mean_t95'][0]:.2f}, {pair['difference_pp']['seed_mean_t95'][1]:.2f}]"
tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=19mm,top=17mm,bottom=18mm]{geometry}
\usepackage{fontspec,xeCJK,graphicx,booktabs,amsmath,xcolor,hyperref,fancyhdr,enumitem,caption}
\setmainfont{Times New Roman}
\setCJKmainfont{Songti SC}
\setCJKmonofont{Songti SC}
\renewcommand{\tablename}{表}\renewcommand{\figurename}{图}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue!45!black,pdftitle={向内破研究报告},pdfauthor={陈威宇 Daniel Chen}}
\definecolor{ink}{HTML}{254C60}
\setlength{\parindent}{2em}\setlength{\parskip}{3pt}
\renewcommand{\baselinestretch}{1.22}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\setlist{nosep,leftmargin=1.7em}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small 向内破｜研究报告}\fancyhead[R]{\small 2026 年 10 月}\fancyfoot[C]{\thepage}
\newcommand{\src}[1]{{\scriptsize\color{ink}[#1]}}
\newcommand{\con}[1]{\par\noindent\textbf{结论：}#1\par}
\newcommand{\sub}[1]{\par\medskip\noindent{\large\bfseries\color{ink}#1}\par\smallskip}
\begin{document}
\thispagestyle{empty}
\begin{center}
\vspace*{8mm}
{\Huge\bfseries 向内破}\\[5mm]
{\LARGE 大语言模型能否靠自身信号\\[2mm]学会利用已有知识}\\[9mm]
{\large 陈威宇（Daniel Chen）}\\[3mm]
{\large Bocconi $\times$ HEC Paris}\\[4mm]
2026 年 10 月
\end{center}
\vspace{6mm}
\sub{摘要}
本研究考察语言模型单步事实可答、组合却失败的知识利用缺口。Qwen3 系列测量支持该缺口存在；自身生成目标的训练带来小幅迁移收益，打乱目标显著损害表现。置信度筛选未稳定优于随机筛选；错目标与对目标的收益尚难区分。结果支持继续研究内部知识利用，但不足以确认一般性的向内进化机制。
\sub{3\quad 研究问题与动机}
语言模型可能分别回答“某人物出生在哪个国家”和“该国的某个属性是什么”，却无法直接回答这两步组合而成的问题。这种现象将\textbf{拥有可提取的事实}与\textbf{能够在当前问题中调用并组合事实}区分开来。本文沿用“知识利用缺口”：只在两步均能独立答对的题上，统计组合回答失败的比例。\src{S0,S1}

“向外进化”通过外部事实、标准答案或任务反馈补充模型能力；“向内进化”尝试利用模型自身产生的答案或内部信号，改善已有知识的调用。二者并非互斥，本研究关注的是：\textbf{在训练目标不直接采用标准答案时，自身信号能否帮助组合泛化？}这里的“已有知识”是单步行为上的操作性定义，并非对模型内部知识存储的完整刻画。

研究分为三个层次：先确认缺口及评分误差；再考察桥接实体能否被内部读出、推理时回填是否有效；最后比较自身目标、置信度筛选及答案打乱等训练对照。已有结果主要来自单一事实数据集和 Qwen3，属于探索性证据。
\sub{报告的证据边界}
正文数值来自本地结果及重新计算，误差线为训练种子间样本标准差。“待核实”表示仅能在交接文档中找到记录，缺少相应原始结果。所有表图的来源编号对应随附 \texttt{SOURCE\_MAP.txt}；原始文件哈希记录于 \texttt{source\_sha256.json}。报告没有覆盖或改写任何旧结果。
\newpage
\sub{4\quad 实验设置与统计口径}
数据采用 Compositional Celebrities，围绕人物、出生国和国家属性构造两步事实问题。干净类别定义为 \texttt{category} 以 \texttt{birthplace} 开头、且不含 \texttt{common\_name}。缺口测量覆盖 Qwen3 的不同规模，训练主模型为 Qwen3-14B。\src{S1,S9}
\begin{table}[h]\centering\small
\caption{主要设置；参数来自保存配置及训练脚本。\src{S8,S10,S11,S14}}
\begin{tabular}{p{36mm}p{111mm}}\toprule
项目 & 设置\\\midrule
模型 & Qwen3：0.6B、1.7B、8B、14B；训练主模型 14B\\
训练问题 & 组合问题；全量池 864 条，包含首都、区号、经纬度类别\\
测试切分 & A：训练类别、新人物；B：新类别、训练人物；C：新类别、新人物\\
新类别族 & 货币、顶级域名、国家数字代码；C 为主要迁移指标\\
评估数量 & A/B/C 各 500 条，固定评估抽样种子 0\\
训练随机性 & 三个训练种子：0、1、2；每次从基础模型独立初始化适配器\\
LoRA & rank 16，alpha 32，dropout 0.05；学习率 $10^{-4}$，2 epochs，batch 8\\
目标生成 & 自身回答第一步，再将该回答放入上下文生成组合训练目标\\
对照数量 & exp8 半量 432 条；exp10 三条件均为 228 条\\\bottomrule
\end{tabular}\end{table}
\textbf{切分原则。}人物与类别族分别切分，C 同时排除训练人物和训练类别族，而不是仅把相同题型的题目随机拆分。exp8/exp9 冻结自身目标，改变筛选或目标分配；exp10 用标准答案划分对错，但标准答案不作为训练文本目标。因而 exp10 是有标准答案参与分组的机制诊断，不能称为完全无监督的样本选择。\src{S10,S11,S14}

\textbf{评分。}旧评分对字符串归一化，会误删数字的负号和小数点。修订评分对数字保留符号，按数学 floor 向下取整比较，同时检查截断取整的敏感性。缺口表采用修订评分；exp8/exp9 保留其历史评分，exp10 采用 floor。exp10 的 C 组组合结果在旧、floor、截断三种评分下相同，其他指标仍应区分版本。\src{S1,S7}
\[
K=\{i:q_{1,i}=1\land q_{2,i}=1\},\qquad
\mathrm{gap}_{\mathrm{clear}}=\frac{\sum_{i\in K}(1-\mathrm{clear}_i)}{|K|}.
\]
$\mathrm{gap}_{\mathrm{ctx}}$ 使用相同分母，将组合回答替换为提供中间事实后的回答。单步独立作答正确不等于组合过程中实际成功调用该事实。

\textbf{统计。}组合准确率均值及标准差来自三个训练种子；配对检验在同一种子、同一批评估题上进行。逐题差异使用 McNemar 检验，exp10 的多重比较采用分析文件中的 Holm 校正。种子均值区间采用配对种子的 $t$ 区间，仅反映种子波动；三个种子无法充分覆盖训练与数据不确定性。\src{S5,S6,S7}
\newpage
\sub{5\quad 实验结果}
\sub{5.1\quad 缺口测量与失败分类}
\begin{table}[h]\centering\small
\caption{干净类别、各模型自身的两步已知集合；floor 评分。百分比的分母为“已知题数”。\src{S1}}
\begin{tabular}{lrrr}\toprule
规模 & 已知题数 & gap\_clear（\%） & gap\_ctx（\%）\\\midrule
@@GAPS@@
\bottomrule\end{tabular}\end{table}
\begin{figure}[h]\centering\includegraphics[width=.90\linewidth]{fig1_scaling.pdf}
\caption{同一批 @@COMMON@@ 道题：三种规模均在 floor 评分下独立答对两步。此共同集合比表 2 更容易，不能混用分母。\src{S1；各规模 rescored 文件}}
\end{figure}
14B 的缺口约为三分之一，提供中间事实后明显缩小（表 2）。只比较 8B/14B 同时掌握的题，旧评分共有 @@PAIR@@ 题，缺口为 @@PAIR8@@\% → @@PAIR14@@\%；floor 评分共有 @@PAIRF@@ 题，为 @@PAIR8F@@\% → @@PAIR14F@@\%。因此“约 36\% → 24\%”有文件支持，但不是所有题上的整体规模曲线。\src{S1；rescored\_8B/14B}
\begin{table}[h]\centering\small
\caption{从旧评分的 818 道失败题中固定种子 7 抽取 120 题。\src{S2}}
\begin{tabular}{lrr}\toprule 标注 & 数量 & 占比（\%）\\\midrule
@@LABELS@@
\bottomrule\end{tabular}\end{table}
分类采用逐题模型辅助审阅、拿不准归真错的保守规则，\textbf{尚未经过独立人工复核}。历史货币、属性类型及评分程序造成部分争议，不能将样本比例直接当作整个缺口的精确真错率。\src{S2}
\con{行为缺口及同题规模差异支持“事实可答与组合可答并不等价”；分类提示该现象不全由格式和评分误差解释。}
\newpage
\sub{5.2\quad 内部激活：桥接实体可被读出}
\begin{table}[h]\centering\small
\caption{逻辑透镜：排除桥接实体与最终答案共享首词元的题；命中为任一层 rank $<10$。\src{S3,S4,S9；Qwen3-14B tokenizer}}
\begin{tabular}{lrrrr}\toprule
组别 & 原始样本 & 排除数 & 命中/保留 & 命中率（\%）\\\midrule
@@LENS@@
\bottomrule\end{tabular}\end{table}
过滤使用标准答案全部别名及带前导空格的词元编码，避免共同首词元把最终答案误当成桥接实体。约 84\% 与 91\% 的历史概括在这一口径下可复现。这里测量的是首词元可解码性，并非完整实体表示，也不直接代表该实体被后续计算因果地使用。\src{S3,S9；build\_figures.py}
\con{失败时桥接信号仍常可读出，提示问题可能涉及知识调用与组合；透镜本身不能确认具体失效机制。}
\sub{5.3\quad 推理时干预：局部修复与整体损害}
\begin{figure}[h]\centering\includegraphics[width=.86\linewidth]{fig3_patching.pdf}\setcounter{figure}{2}
\caption{交接文档中的固定层真回填与对照修复率。范围是文档记录，\textbf{并非误差条或置信区间}；缺原始回填文件，全部待核实。\src{S0}}
\end{figure}
\begin{table}[h]\centering\small
\caption{推理时干预的历史记录与核验状态。\src{S0}}
\begin{tabular}{p{36mm}p{82mm}l}\toprule
条件 & 交接记录 & 状态\\\midrule
固定层回填 & 修复 14--16\%，对照 3\%；$p<10^{-8}$ & 待核实\\
整体影响 & 原正确题被破坏，净效果为负 & 待核实\\
置信度门控 & 净效果约为 0 & 待核实\\\bottomrule
\end{tabular}\end{table}
修复率只描述原本答错的题；总体净变化还必须扣除原本答对、干预后变错的题。仅凭较高修复率不足以评价部署收益。由于本地缺少原始干预与配对预测文件，本节不作为已复核的因果证据。
\con{历史记录提示简单回填存在破坏性；当前可审计材料不足以确认其效果大小与显著性。}
\newpage
\sub{5.4\quad 向内训练：小幅组合泛化收益}
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{fig2_conditions.pdf}\setcounter{figure}{1}
\caption{C 组各条件正确率；虚线为基础模型，误差线为三种子样本标准差。斜线柱 $\dagger$ 是交接记录、待核实的单次结果。全量 864 条、半量 432 条、对错诊断 228 条，跨条件柱高不代表等量因果比较。\src{S0,S5,S6,S7}}
\end{figure}
\begin{table}[h]\centering\small
\caption{self 全量在固定切分上的组合正确率；均值 $\pm$ 标准差。\src{S5,S6}}
\begin{tabular}{lrrr}\toprule
测试集 & 基线（\%） & self（\%） & 增益（百分点）\\\midrule
@@ABC@@
\bottomrule\end{tabular}\end{table}
C 组 self 三种子均值为 @@SELF@@\%，高于基础模型（图 2）。历史仅单步训练 @@HOPS@@\%、标准答案训练 @@GOLD@@\% 均只能追溯至交接文档，\textbf{待核实}；标准答案条件为单次运行，不能与三种子均值作稳定性对等比较。\src{S0,S5,S6}

\textbf{单步保留需谨慎表述。}exp8 保存的 C 组单步检查只覆盖固定的前 150 题。第一步从 @@Q1@@，第二步从 @@Q2@@（训练后取三种子均值）。这支持第一步大致保持，但第二步略有回落，因此不能宣称所有单步能力均不下降。\src{S13}

\textbf{幅度与检验。}C 组收益为数个百分点，属于小幅改善。逐种子配对结果并非每次都显著，不能以三次运行方向相同替代全面稳定性验证。跨人物与类别族的切分使该收益比训练题记忆更有解释价值，但仍不能排除更一般的任务适应。\src{S5,S6}
\con{自身目标训练支持有限的组合迁移收益；尚不足以确认一般性、自主持续提升，也不能宣称完全不遗忘。}
\newpage
\sub{5.5\quad 置信度过滤：目标更准，迁移未稳定更好}
\begin{table}[h]\centering\small
\caption{目标质量与 C 组结果。质量列为冻结样本/随机种子 0；C 为三训练种子均值 $\pm$ 标准差。类别内两条件每类保留相同数量。\src{S5,S6}}
\begin{tabular}{lrrr}\toprule
条件 & 训练条数 & 目标正确率（\%） & C 正确率（\%）\\\midrule
@@FILTER@@
@@BALANCED@@
\bottomrule\end{tabular}\end{table}
目标整句平均对数概率区分正确与错误的 AUROC 为 @@AUROC@@（全量训练池）。它只衡量排序诊断，不表示概率校准；标准答案不参与 exp8/exp9 的训练目标或筛选。置信度筛选显著提高目标正确率，却未超过同数量随机筛选的 C 组均值；类别内比较也没有稳定优势。\src{S5,S6}
\con{置信度可预测部分目标正确性，但筛掉低置信度样本并未稳定改善组合迁移；“C 无提升”应指无额外筛选收益，而非低于基线。}
\sub{5.6\quad 对照分析：内容对应重要，正确性作用未分离}
\begin{table}[h]\centering\small
\caption{exp10：三条件各 228 条，类别数量匹配；C 组各 500 题，floor 评分。单步列为三种子均值。\src{S7,S10}}
\begin{tabular}{lrrrr}\toprule
条件 & C（\%） & 增益（百分点） & 第一步（\%） & 第二步（\%）\\\midrule
@@CONTROLS@@
\bottomrule\end{tabular}\end{table}
自身错目标与打乱错目标使用同一批问题，后者在类别内改变目标对应关系。全量打乱的 C 结果为 @@SHUFF@@\%（exp9），打乱错目标同样大幅下降（表 8）。这不支持“只熟悉题型就足够”的强解释；但打乱亦引入矛盾监督和分布扰动，不能据此唯一归因于知识内容。\src{S6,S7,S10,S11}

self 错答案相对 self 对答案的均值差为 @@DIFF@@ 个百分点，配对种子均值 95\% 区间 @@DIFFCI@@；两者未检出稳定差异，\textbf{不是等效性结论}。对基础模型的改善在逐题 Holm 校正后也未达到显著。正确组问题与错误组不同，尚残留难度及语义正确性混杂。\src{S7}
\con{结果提示问题与自身目标的对应关系重要；尚未分离“真实正确性”与“模型自身一致性”的独立作用。}
\newpage
\sub{6\quad 讨论：结论、撤回与局限}
\textbf{目前支持的四条结论：}
\begin{enumerate}
\item 单步可答并不保证组合可答；同题规模比较与中间事实上下文提示知识利用存在瓶颈。\src{S1}
\item 失败题中桥接实体常可读出，提示“无可读信号”不能解释全部失败；这仍是相关性观察。\src{S3,S9}
\item 自身目标训练在严格 C 切分上带来小幅平均收益；目标与问题的任意打乱会造成明显损害。\src{S5,S6,S7}
\item 目标正确率与下游迁移收益并非简单单调对应；置信度排序优势未转化为稳定筛选收益。\src{S5,S6}
\end{enumerate}
\textbf{已撤回或不再使用的四种强说法：}
\begin{enumerate}
\item “答案可从隐藏层读出，就说明模型实际使用了该知识。”缺少因果识别。
\item “提高目标正确率，组合能力必然更好。”筛选实验不支持该必然关系。
\item “错答案和对答案一样有效，所以一致性才是机制。”未检出差异不能确认等效，更不能唯一确定机制。
\item “向内训练完全不遗忘，并能持续自我进化。”单步有回落，多轮及跨模型家族尚未验证。
\end{enumerate}
\textbf{局限。}仅一个模型家族与一个事实组合基准；训练种子少、切分固定，存在探索性多重比较。对错标签受标准答案宽窄影响，120 题分类尚未独立人工复核。逻辑透镜仅看首词元，内部可解码不代表因果使用。旧回填、hops、gold 原始文件缺失，相关数字均待核实；不同实验训练量和评分版本亦限制横向比较。\src{S0,S1,S2,S5,S6,S7}
\sub{7\quad 下一步研究}
\begin{enumerate}
\item \textbf{迭代自我训练：}用新人物逐轮再生成自身目标，对照每轮从基础模型重新训练；监测错误题置信度、单步遗忘及累计训练预算。
\item \textbf{跨结构迁移：}改变关系链长度、关系类型及问题表达，固定数据量与优化步数，检验收益是否超出当前模板。
\item \textbf{隐空间探针：}与因果干预结合，排除词元重叠、读出器和位置混杂，检验桥接表示是否参与后续计算。
\item \textbf{第二个模型家族：}复现冻结目标、随机筛选、打乱目标和正确性诊断，检验现象适用范围。
\item \textbf{文献对比：}区分组合缺口、内部多跳时序、上下文蒸馏及自生成推理训练；明确本研究新增的对照与尚未解决的机制。
\end{enumerate}
\sub{参考文献}
{\footnotesize\raggedright\setlength{\parskip}{2pt}\noindent
[1] Press, O., et al. (2022). Measuring and Narrowing the Compositionality Gap in Language Models. \href{https://arxiv.org/abs/2210.03350}{arXiv:2210.03350}.\par\noindent
[2] Biran, E., et al. (2024). Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries. \href{https://arxiv.org/abs/2406.12775}{arXiv:2406.12775}.\par\noindent
[3] Snell, C., Klein, D., \& Zhong, R. (2022). Learning by Distilling Context. \href{https://arxiv.org/abs/2209.15189}{arXiv:2209.15189}.\par\noindent
[4] Zelikman, E., et al. (2022). STaR: Bootstrapping Reasoning With Reasoning. \href{https://arxiv.org/abs/2203.14465}{arXiv:2203.14465}.\par\noindent
[5] Liu, X., et al. (2026). Layer-Order Inversion: Rethinking Latent Multi-Hop Reasoning in Large Language Models. \href{https://arxiv.org/abs/2601.03542}{arXiv:2601.03542}.}
\end{document}
'''
for k,v in t.items():tex=tex.replace('@@'+k+'@@',v)
assert '@@' not in tex
(O/'inward_report.tex').write_text(tex)
(O/'source_sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
mp='数据来源与复现说明\n生成时间：2026-10-05\n所有路径相对于研究根目录 '+str(R)+'\n\n'
for k,v in S.items():mp+=k+'  '+v+'\n'
mp+='''\n表1：S8/S10/S11/S14：配置、协议、脚本中的参数与切分。\n表2：S1，category=CLEAN, modes.floor；known_n 分母。\n表3：S2，counts/sample_n；模型辅助分类，非独立人工验证。\n表4：S3/S4/S9 和 Qwen/Qwen3-14B tokenizer：剔除任何 gold 别名首词元重叠，min(bridge_ranks)<10。\n表5：S0：仅交接记录，原始回填及门控结果未找到，待核实。\n表6：S5/S6，self_full A/B/C 三种子 accuracy_pct mean/sample_sd。\n表7：S5 training_quality（种子0目标质量）/groups；S6 类别内均值。\n表8：S7 scoring.floor.groups，C/composed/q1/q2；S10 228条类别匹配协议。\n图1：tasks56_verified/rescored_{1.7B,8B,14B}.jsonl，在三者floor.q1 & floor.q2同时为真的题上取交集，category+person为键，同一738题分别计算失败率。\n图2：S5/S6/S7 C均值和种子样本SD；S0 hops和gold斜线柱待核实。训练量不同不作等量比较。\n图3：S0交接文档中的14–16%范围和3%对照，非置信区间，均待核实。\n单步保留：S13 C q1/q2，固定前150题；不能与exp10全部500题的单步率混用。\n配对差异：S7 scoring.floor.comparisons/self_wrong versus self_correct/C/composed。\n\n复现：先运行 build_figures.py（numpy/matplotlib/transformers，需Qwen3-14B tokenizer），再运行 build_report.py。然后在此目录运行 xelatex -interaction=nonstopmode -halt-on-error inward_report.tex 两次。字体为系统 Songti SC；图为matplotlib矢量PDF。最终PDF以本目录文件为输入，没有写入原始实验目录。\n'''
(O/'SOURCE_MAP.txt').write_text(mp)
print('TeX and source map written.')
