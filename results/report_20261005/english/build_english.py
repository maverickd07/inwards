from pathlib import Path
import json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;D=json.loads((O.parent/'report_data.json').read_text());CN=(O.parent/'inward_report.tex').read_text()
plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42,'font.size':10,'axes.spines.right':False,'axes.spines.top':False})
fig,ax=plt.subplots(figsize=(8,2.6));xs=np.arange(3)
for offset,key,label,color in [(-.17,'gap_clear','Direct composition','#254c60'),(.17,'gap_ctx','With intermediate fact','#809fa9')]:
 vals=[100*r[key] for r in D['scale_common3']];ax.bar(xs+offset,vals,.34,label=label,color=color)
 for x,v in zip(xs+offset,vals):ax.text(x,v+1.3,f'{v:.1f}',ha='center')
ax.set_xticks(xs,[r['size'] for r in D['scale_common3']]);ax.set_ylim(0,87);ax.set_ylabel('Gap (%)');ax.legend(frameon=False,ncols=2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(O/'fig1_scaling_en.pdf');plt.close(fig)
labels=['Base','Single-hop*','Self: full','Confidence 50%','Random 50%','Self: incorrect','Self: correct','Shuffled: incorrect','Shuffled: full','Gold*']
fig,ax=plt.subplots(figsize=(10,3.4));rs=D['fig2'];bars=ax.bar(range(len(rs)),[r['mean'] for r in rs],color=['#a64b3d' if 'Shuffled' in l else '#315f73' for l in labels],width=.7)
for i,r in enumerate(rs):
 if r['status']=='pending':bars[i].set_facecolor('white');bars[i].set_edgecolor('#888');bars[i].set_hatch('///')
 if r['sd']:ax.errorbar(i,r['mean'],yerr=r['sd'],fmt='none',ecolor='#182b36',capsize=3)
 ax.text(i,r['mean']+r['sd']+1.3,f"{r['mean']:.1f}",ha='center',fontsize=9)
ax.axhline(D['baseline'],ls='--',color='#a64b3d',lw=1);ax.set_xticks(range(len(rs)),labels,rotation=32,ha='right');ax.set_ylim(0,66);ax.set_ylabel('C accuracy (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(O/'fig2_conditions_en.pdf');plt.close(fig)
p=D['patch_pending'];fig,ax=plt.subplots(figsize=(7.7,2.3));ax.hlines(1,p['repair_low'],p['repair_high'],color='#315f73',lw=6);ax.plot(p['control'],0,'s',mfc='white',mec='#888',ms=8);ax.text(p['repair_high']+.6,1,f"{p['repair_low']:.0f}–{p['repair_high']:.0f}% (unverified)",va='center');ax.text(p['control']+.6,0,f"{p['control']:.0f}% (unverified)",va='center');ax.set_yticks([0,1],['Control','Fixed-layer patch']);ax.set_ylim(-.6,1.6);ax.set_xlim(0,29);ax.set_xlabel('Repair rate (%); reported range, not a confidence interval');ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(O/'fig3_patching_en.pdf');plt.close(fig)
# Reuse table bodies from the verified Chinese TeX: no numeric transcription.
bodies=re.findall(r'\\begin\{tabular\}\{.*?\}\\toprule\n(.*?)\\bottomrule\\end\{tabular\}',CN,re.S)
# Explicit extract instead: find content following top rule through bottom rule in each table.
bodies=[]
for block in re.findall(r'\\begin\{table\}.*?\\end\{table\}',CN,re.S):bodies.append(block.split(r'\toprule',1)[1].split(r'\bottomrule',1)[0].strip())
repl={'规模':'Size','已知题数':'Known items','gap\\_clear（\\%）':'gap\\_clear (\\%)','gap\\_ctx（\\%）':'gap\\_ctx (\\%)','标注':'Label','数量':'Count','占比（\\%）':'Share (\\%)','组别':'Group','原始样本':'Initial $n$','排除数':'Excluded','命中/保留':'Hits/retained','命中率（\\%）':'Hit rate (\\%)','失败组':'Failure','成功组':'Success','测试集':'Split','基线（\\%）':'Base (\\%)','self（\\%）':'Self (\\%)','增益（百分点）':'Gain (pp)','A：新人物':'A: new people','B：新类别':'B: new categories','C：两者都新':'C: both new','条件':'Condition','训练条数':'Training $n$','目标正确率（\\%）':'Target accuracy (\\%)','C 正确率（\\%）':'C accuracy (\\%)','全量':'Full','全局置信度':'Global confidence','全局随机':'Global random','类别内置信度':'Within-category confidence','类别内随机':'Within-category random','C（\\%）':'C (\\%)','第一步（\\%）':'Hop 1 (\\%)','第二步（\\%）':'Hop 2 (\\%)','self 错答案':'Self: incorrect','self 对答案':'Self: correct','打乱错答案':'Shuffled: incorrect'}
for i,b in enumerate(bodies):
 for a,v in repl.items():b=b.replace(a,v)
 bodies[i]=b
tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=19mm,top=17mm,bottom=18mm]{geometry}
\usepackage{fontspec,float,graphicx,booktabs,amsmath,xcolor,hyperref,fancyhdr,enumitem,caption}
\setmainfont{Times New Roman}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue!45!black,pdftitle={Inward Breakthrough: Learning to Use Existing Knowledge},pdfauthor={Daniel Chen}}
\definecolor{ink}{HTML}{254C60}
\setlength{\emergencystretch}{2em}\setlength{\tabcolsep}{5pt}\setlength{\parindent}{1em}\setlength{\parskip}{4pt}
\renewcommand{\baselinestretch}{1.12}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\setlist{nosep,leftmargin=1.6em}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Inward Breakthrough | Research Report}\fancyhead[R]{\small October 2026}\fancyfoot[C]{\thepage}
\newcommand{\src}[1]{{\scriptsize\color{ink}[#1]}}
\newcommand{\con}[1]{\par\noindent\textbf{Takeaway.} #1\par}
\newcommand{\sub}[1]{\par\medskip\noindent{\large\bfseries\color{ink}#1}\par\smallskip}
\begin{document}
\thispagestyle{empty}
\begin{center}\vspace*{9mm}
{\Huge\bfseries Inward Breakthrough}\\[5mm]
{\LARGE Can Large Language Models Learn to Use\\[2mm]Existing Knowledge Through Their Own Signals?}\\[9mm]
{\large Daniel Chen}\\[3mm]
{\large Bocconi $\times$ HEC Paris}\\[4mm]
October 2026
\end{center}\vspace{5mm}
\sub{Abstract}
This study examines a knowledge-use gap: language models can answer two constituent factual questions separately yet fail to answer their composition. Measurements on Qwen3 support the existence of this gap. Training on self-generated targets yields modest transfer gains, whereas shuffling targets severely damages performance. Confidence filtering increases target accuracy but does not consistently outperform random selection. Gains from correct and incorrect self-targets remain difficult to distinguish. These findings support further investigation of internal knowledge use, but do not establish a general mechanism of autonomous inward improvement.
\sub{1\quad Research question and motivation}
A model may identify a person's country of birth and separately answer a question about that country's attributes, yet fail when asked the composed question directly. This distinguishes \textbf{facts accessible through separate prompts} from \textbf{facts successfully retrieved and combined in the current task}. We operationalize the knowledge-use gap as the composition failure rate among items for which both individual hops are answered correctly.\src{S0,S1}

``Outward improvement'' adds external facts, gold answers, or task feedback. ``Inward improvement'' uses the model's own answers or internal signals to improve access to its existing knowledge. These approaches are compatible. The narrow question here is whether self-generated targets, without directly using gold answers as training targets, can improve compositional generalization. ``Existing knowledge'' is a behavioral definition, not a complete account of internal factual storage.
\sub{Scope of the evidence}
The report follows three stages: gap measurement and scoring checks; internal readout and inference-time patching; and training controls involving self-targets, confidence selection, and shuffling. All verified numbers are read from local files. Error bars represent sample standard deviations across training seeds. \textbf{Unverified} results are recorded only in the handoff document, with their original result files unavailable. Source IDs map to the accompanying source register; original files are not modified.
\newpage
\sub{2\quad Experimental setup and statistical conventions}
Compositional Celebrities supplies two-hop questions connecting people, countries of birth, and country attributes. ``Clean'' categories start with \texttt{birthplace} and exclude \texttt{common\_name}. Gap measurement covers several Qwen3 sizes; training experiments primarily use Qwen3-14B.\src{S1,S9}
\begin{table}[H]\centering\small
\caption{Main settings, from saved configurations, protocols, and scripts.\src{S8,S10,S11,S14}}
\begin{tabular}{p{32mm}p{115mm}}\toprule
Component & Setting\\\midrule
Models & Qwen3: 0.6B, 1.7B, 8B, and 14B; main training model: 14B\\
Training pool & 864 composed questions: capital, calling code, latitude/longitude\\
Evaluation splits & A: seen categories/new people; B: new categories/seen people; C: both new\\
Held-out families & Currency, top-level domain, and numeric country code; C is the primary transfer outcome\\
Evaluation size & 500 questions per split; fixed evaluation sampling seed 0\\
Training seeds & 0, 1, and 2; independently initialized adapters on the base model\\
LoRA & Rank 16, alpha 32, dropout 0.05; learning rate $10^{-4}$, 2 epochs, batch size 8\\
Self-target generation & Answer hop 1, then place the model's own answer in context to generate the composed target\\
Control sizes & exp8 half-pool: 432 items; each exp10 condition: 228 items\\\bottomrule
\end{tabular}\end{table}
\textbf{Splits and supervision.} People and category families are partitioned separately. C excludes both training people and training families, rather than randomly holding out questions of the same type. exp8/exp9 freeze self-targets and vary selection or assignment. exp10 uses gold answers to classify target correctness, but never uses gold answers as training text. It is therefore a gold-stratified diagnostic, not fully unsupervised sample selection.\src{S10,S11,S14}

\textbf{Scoring.} Legacy string normalization could delete numeric minus signs and decimal points. Revised numeric scoring preserves signs and compares values after mathematical floor rounding, with truncation as a sensitivity check. Gap tables use revised scoring; exp8/exp9 retain their historical scoring; exp10 uses floor. exp10 C composition outcomes are identical under legacy, floor, and truncation scoring, although other metrics must remain version-specific.\src{S1,S7}
\[
K=\{i:q_{1,i}=1\land q_{2,i}=1\},\qquad
\mathrm{gap}_{\mathrm{clear}}=\frac{\sum_{i\in K}(1-\mathrm{clear}_i)}{|K|}.
\]
$\mathrm{gap}_{\mathrm{ctx}}$ uses the same denominator, replacing direct composition with the answer obtained after supplying the intermediate fact.

\textbf{Statistics.} Means and sample standard deviations summarize three training seeds. Item-level comparisons pair the same evaluation questions within the same seed using McNemar tests; exp10 applies the analysis file's Holm correction. Paired-seed $t$ intervals describe seed variation only. Three seeds do not adequately capture uncertainty across training runs, datasets, and splits.\src{S5,S6,S7}
\newpage
\sub{3\quad Results}
\sub{3.1\quad Gap measurement and failure classification}
\begin{table}[H]\centering\small
\caption{Clean categories, each model's own two-hop-known subset; floor scoring. Percentages use the known-item denominator.\src{S1}}
\begin{tabular}{lrrr}\toprule
@@T2@@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=.90\linewidth]{fig1_scaling_en.pdf}
\caption{The same @@COMMON@@ questions, with both hops independently correct in all three models under floor scoring. This common subset is easier than the model-specific subsets in Table 2.\src{S1; rescored files}}
\end{figure}
For 14B, roughly one-third of known-hop questions fail in direct composition; supplying the intermediate fact sharply reduces the gap (Table 2). For the shared 8B/14B known subset, legacy scoring gives @@PAIR@@ questions and gaps of @@PAIR8@@\% versus @@PAIR14@@\%; floor scoring gives @@PAIRF@@ questions and @@PAIR8F@@\% versus @@PAIR14F@@\%. Thus, the shorthand ``about 36\% to 24\%'' is supported for that paired subset, not as an all-question scaling curve.\src{S1; rescored\_8B/14B}
\begin{table}[H]\centering\small
\caption{Seed-7 sample of 120 from 818 legacy-scored failures.\src{S2}}
\begin{tabular}{lrr}\toprule
@@T3@@
\bottomrule\end{tabular}\end{table}
Classification used item-by-item model-assisted review, assigning uncertain cases to GENUINE. It has \textbf{not undergone independent human validation}. Historical currencies, relation-type confusion, and scoring errors account for some disputed cases. The sample share is not a precise estimate of the genuine-error rate of the entire gap.\src{S2}
\con{Separate factual accessibility does not guarantee successful composition. The audit suggests that formatting and scoring errors do not explain all failures.}
\newpage
\sub{3.2\quad Internal activation: readable bridge signals}
\begin{table}[H]\centering\small
\caption{Logit lens after excluding first-token overlap between bridge entities and final gold answers. A hit means rank $<10$ at any layer.\src{S3,S4,S9; tokenizer}}
\begin{tabular}{lrrrr}\toprule
@@T4@@
\bottomrule\end{tabular}\end{table}
Overlap exclusion uses all gold aliases, including tokenizations with a leading space, to avoid mistaking a shared final-answer token for the bridge entity. The historical summary of roughly 84\% and 91\% is reproducible under this convention. However, this measures first-token decodability, not a complete entity representation or its causal use in subsequent computation.\src{S3,S9; figure code}
\con{Bridge signals are often readable even on failures, suggesting a possible retrieval or composition bottleneck. The lens alone does not identify its mechanism.}
\sub{3.3\quad Inference-time intervention: repairs and damage}
\begin{figure}[H]\centering\includegraphics[width=.90\linewidth]{fig3_patching_en.pdf}\setcounter{figure}{2}
\caption{Fixed-layer patching versus control, as recorded in the handoff. The range is \textbf{not a confidence interval}. Original patching files are unavailable; all values are unverified.\src{S0}}
\end{figure}
\begin{table}[H]\centering\small
\caption{Historical intervention claims and verification status.\src{S0}}
\begin{tabular}{p{29mm}p{86mm}p{25mm}}\toprule
Condition & Handoff record & Status\\\midrule
Fixed-layer patch & Repair 14--16\%; control 3\%; $p<10^{-8}$ & Unverified\\
Overall impact & Previously correct items damaged; negative net effect & Unverified\\
Confidence gating & Net effect approximately zero & Unverified\\\bottomrule
\end{tabular}\end{table}
Repair rate describes initially incorrect items only. Net accuracy change must also subtract initially correct items that become wrong. A higher repair rate is therefore insufficient to establish practical benefit. Without the original interventions and paired predictions, these claims are not treated as audited causal evidence.
\con{The historical record suggests destructive side effects from simple patching; available files do not allow independent verification of magnitude or significance.}
\newpage
\sub{3.4\quad Inward training: modest compositional transfer}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{fig2_conditions_en.pdf}\setcounter{figure}{1}
\caption{C composition accuracy. Dashed line: base model; error bars: sample SD across three seeds. Hatched (*) bars are unverified historical single-run results. Full, half, and correctness-diagnostic conditions use 864, 432, and 228 items respectively; cross-condition bars are not equal-budget causal comparisons.\src{S0,S5,S6,S7}}
\end{figure}
\begin{table}[H]\centering\small
\caption{Full self-target training on fixed splits; mean $\pm$ sample SD.\src{S5,S6}}
\begin{tabular}{lrrr}\toprule
@@T6@@
\bottomrule\end{tabular}\end{table}
Full self-target training reaches @@SELF@@\% on C, above the base model (Figure 2). Historical single-hop-only training (@@HOPS@@\%) and gold-target training (@@GOLD@@\%) can be traced only to the handoff and remain \textbf{unverified}. Gold training is a single run and cannot support a seed-stability comparison against three-seed means.\src{S0,S5,S6}

\textbf{Retention is imperfect.} The saved exp8 single-hop check covers only the fixed first 150 C items. Hop 1 changes from @@Q1@@, while hop 2 changes from @@Q2@@ (post-training seed means). Hop 1 is broadly retained, but hop 2 declines slightly. It would be inaccurate to claim that all single-hop capabilities remain unchanged.\src{S13}

\textbf{Magnitude and significance.} The C gain is a few percentage points, a modest effect. Item-level paired improvements are not significant in every seed. Consistent directions do not substitute for broader stability testing. Holding out both people and category families makes the gain more informative than training-item memorization, but general task adaptation remains a plausible contributor.\src{S5,S6}
\con{Self-target training supports limited compositional transfer; it does not establish sustained autonomous improvement or the absence of forgetting.}
\newpage
\sub{3.5\quad Confidence filtering: better targets, no stable extra gain}
\begin{table}[H]\centering\small
\caption{Target quality and C accuracy. Target quality is for the frozen pool/selection seed 0; C averages three training seeds. Within-category conditions preserve the same count per category.\src{S5,S6}}
\begin{tabular}{lrrr}\toprule
@@T7@@
\bottomrule\end{tabular}\end{table}
Mean whole-target log probability distinguishes correct from incorrect targets with AUROC @@AUROC@@ on the full pool. This is a ranking diagnostic, not probability calibration. Gold answers are not used to construct training targets or select samples in exp8/exp9. Confidence filtering substantially improves target accuracy but does not exceed the equally sized random-selection C mean; the within-category comparison also shows no stable advantage.\src{S5,S6}
\con{Confidence predicts some target correctness, but filtering does not consistently improve compositional transfer. ``No gain'' here means no added filtering benefit, not no improvement over the base model.}
\sub{3.6\quad Controls: correspondence matters, correctness unresolved}
\begin{table}[H]\centering\small
\caption{exp10: 228 items per condition with matched category counts; 500 C questions, floor scoring. Single-hop columns are seed means.\src{S7,S10}}
\begin{tabular}{lrrrr}\toprule
@@T8@@
\bottomrule\end{tabular}\end{table}
Incorrect self-targets and shuffled incorrect targets use the same questions; shuffling changes question--target correspondence within categories. Full shuffling reaches @@SHUFF@@\% on C (exp9), and shuffling incorrect targets likewise causes severe damage. This does not support the strong explanation that question-format familiarity alone is sufficient. However, shuffling also introduces contradictory supervision and distributional disruption, so the damage cannot be uniquely attributed to factual content.\src{S6,S7,S10,S11}

The incorrect-minus-correct self-target gain difference is @@DIFF@@ pp, with paired-seed 95\% interval @@DIFFCI@@. No stable difference is detected, but \textbf{this is not an equivalence result}. Neither improvement over the base survives the item-level Holm correction. Correct and incorrect groups contain different questions, leaving difficulty and semantic-correctness confounds.\src{S7}
\con{Question--self-target correspondence appears important, but the independent effects of factual correctness and consistency with the model's own beliefs remain unresolved.}
\newpage
\sub{4\quad Discussion: supported findings, withdrawn claims, limits}
\textbf{Four findings currently supported:}
\begin{enumerate}
\item Single-hop accessibility does not guarantee composition. Paired scaling and intermediate-fact context suggest a knowledge-use bottleneck.\src{S1}
\item Bridge signals are often readable on failures; absent readable signals cannot explain all failures. This remains correlational evidence.\src{S3,S9}
\item Self-target training yields modest average transfer on C; arbitrarily reassigning targets to questions severely damages performance.\src{S5,S6,S7}
\item Target accuracy and downstream transfer are not simply monotonic: confidence-ranking ability does not imply a stable selection advantage.\src{S5,S6}
\end{enumerate}
\textbf{Four strong claims withdrawn or no longer used:}
\begin{enumerate}
\item ``A hidden-layer readout shows that the model actually used that knowledge.'' Causal identification is missing.
\item ``More accurate training targets necessarily improve composition.'' Filtering results do not support that necessity.
\item ``Incorrect and correct targets work equally well, so consistency is the mechanism.'' Nonsignificance establishes neither equivalence nor a unique mechanism.
\item ``Inward training causes no forgetting and enables sustained self-improvement.'' Single-hop decline is observed; iteration and cross-family generality remain untested.
\end{enumerate}
\textbf{Limitations.} Evidence covers one model family and one factual-composition benchmark. Few seeds, one fixed split, and exploratory multiple comparisons limit inference. Gold-answer narrowness affects correctness labels; the 120-item audit lacks independent human validation. First-token readouts do not establish causal use. Original patching, single-hop-only, and gold-target files are unavailable, and those figures remain unverified. Different training sizes and scoring versions limit cross-experiment comparisons.\src{S0,S1,S2,S5,S6,S7}
\sub{5\quad Next steps}
\begin{enumerate}
\item \textbf{Iterative self-training:} regenerate targets using new people each round; compare with fresh base-model training. Monitor confidence on errors, forgetting, and cumulative optimization budgets.
\item \textbf{Cross-structure transfer:} change chain length, relation types, and phrasing while matching data and optimization steps.
\item \textbf{Latent probes:} combine readouts with causal interventions; control token overlap, decoder, and position confounds.
\item \textbf{A second model family:} reproduce frozen-target, random-selection, shuffling, and correctness diagnostics.
\item \textbf{Literature comparison:} distinguish composition gaps [1], internal multi-hop timing [2,5], context distillation [3], and self-generated reasoning training [4].
\end{enumerate}
\sub{References}
@@REFS@@
\end{document}
'''
for i in [2,3,4,6,7,8]:tex=tex.replace('@@T'+str(i)+'@@',bodies[i-1])
# Match verified numeric text from Chinese report/JSON, never fill from memory.
values={'COMMON':str(D['scale_common3'][0]['n']),'PAIR':str(D['scale_pair']['old']['n']),'PAIRF':str(D['scale_pair']['floor']['n']),'PAIR8':f"{100*D['scale_pair']['old']['8B']:.1f}",'PAIR14':f"{100*D['scale_pair']['old']['14B']:.1f}",'PAIR8F':f"{100*D['scale_pair']['floor']['8B']:.1f}",'PAIR14F':f"{100*D['scale_pair']['floor']['14B']:.1f}",'SELF':re.search(r'C 组 self 三种子均值为 (.*?)\\%，',CN)[1],'SHUFF':re.search(r'全量打乱的 C 结果为 (.*?)\\%',CN)[1],'HOPS':f"{D['hops_pending']:.1f}",'GOLD':f"{D['gold_pending']:.1f}",'Q1':re.search(r'第一步从 (.*?)，第二步',CN)[1].replace('→','to'),'Q2':re.search(r'第二步从 (.*?)（训练后',CN)[1].replace('→','to'),'AUROC':re.search(r'AUROC 为 ([\d.]+)',CN)[1],'DIFF':re.search(r'均值差为 ([-\d.]+)',CN)[1],'DIFFCI':re.search(r'区间 (\[[^\]]+\])',CN)[1],'REFS':CN.split(r'\sub{参考文献}',1)[1].split(r'\end{document}',1)[0]}
for k,v in values.items():tex=tex.replace('@@'+k+'@@',v)
assert '@@' not in tex;assert not re.search('[\u4e00-\u9fff]',tex)
(O/'inward_report_en.tex').write_text(tex)
(O/'SOURCE_MAP_EN.txt').write_text('English edition: identical data and source IDs to ../SOURCE_MAP.txt.\nAll 8 tables and 3 figures retain the Chinese edition mappings.\nEnglish table numeric bodies are extracted from the verified Chinese TeX; figure values are read from ../report_data.json.\nNo experiment files are modified. English plots are separately generated vector PDFs.\nReproduce: python build_english.py; xelatex inward_report_en.tex twice.\n\n'+(O.parent/'SOURCE_MAP.txt').read_text())
print('English TeX and vector figures generated.')
