"""Read verified source files, recompute statistics, and export vector PDF figures.
Run with Python + numpy + matplotlib + transformers. Existing results are read-only.
"""
from pathlib import Path
import json,re,hashlib,statistics,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
TOKENIZER=Path('/private/tmp/inward-report-tokenizer')
FONT='/System/Library/Fonts/Supplemental/Songti.ttc'
fp=FontProperties(fname=FONT)
plt.rcParams.update({'font.family':fp.get_name(),'pdf.fonttype':42,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'axes.labelsize':11})
S={
 'S0':'INWARD_HANDOFF.md',
 'S1':'results/handoff_analysis_20261003/tasks56_verified/gap_comparison.json',
 'S2':'inward/results/handoff_analysis_20261003/verified_analysis/label_summary.json',
 'S3':'inward/results/lens_Qwen3-14B/lens_items.jsonl',
 'S4':'inward/results/lens_Qwen3-14B/lens_summary.json',
 'S5':'results/handoff_analysis_20261003/exp8_680959/analysis/summary.json',
 'S6':'results/handoff_analysis_20261003/exp9_681194/verified_analysis_v2/report.json',
 'S7':'results/handoff_analysis_20261003/exp10_681386/verified_analysis/report.json',
 'S8':'results/handoff_analysis_20261003/exp8_680959/results/filtered_680959/self_confidence_top1.0_s0/run_config.json',
 'S9':'inward/data/compositional_celebrities.json',
 'S10':'results/handoff_analysis_20261003/exp10_consistency/protocol.json',
 'S11':'results/handoff_analysis_20261003/exp9_shuffled/protocol.json',
 'S12':'results/handoff_analysis_20261003/exp8_680959/analysis/training_categories.csv',
}
used=set()
def read(code):
 used.add(S[code]);return json.loads((ROOT/S[code]).read_text())
def jl(rel):
 used.add(str(rel));return [json.loads(s) for s in (ROOT/rel).read_text().splitlines() if s.strip()]
gaps=read('S1');labels=read('S2');lens_summary=read('S4');exp8=read('S5');exp9=read('S6');exp10=read('S7');cfg=read('S8');dataset=read('S9');protocol10=read('S10');protocol9=read('S11')
used.add(S['S0']);handoff=(ROOT/S['S0']).read_text()
# Read legacy-only claims verbatim from the handoff; no raw verification is implied.
row=next(x for x in handoff.splitlines() if 'hops（只练单步' in x);hops=float(re.findall(r'→\s*([\d.]+)',row)[-1])
row=next(x for x in handoff.splitlines() if 'gold（标准答案）' in x);gold=float(re.findall(r'→\s*([\d.]+)',row)[-1])
row=next(x for x in handoff.splitlines() if '| 4 | 回填' in x);lo,hi=re.search(r'(\d+)–(\d+)%',row).groups();control=float(re.search(r'对照 (\d+)%',row)[1]);patch={'repair_low':float(lo),'repair_high':float(hi),'control':control,'status':'待核实：仅交接文档记录，原始回填文件缺失','p_claim':re.search(r'p < ([^；]+)',row)[1]}
# Common-question scaling analysis: known in every compared model, corrected floor scoring.
rs={s:{(r['category'],r['person']):r for r in jl(f'results/handoff_analysis_20261003/tasks56_verified/rescored_{s}.jsonl') if r['category'].startswith('birthplace') and 'common_name' not in r['category']} for s in ['1.7B','8B','14B']}
shared=set.intersection(*[{k for k,r in v.items() if r['floor']['q1'] and r['floor']['q2']} for v in rs.values()])
scale=[{'size':s,'n':len(shared),'gap_clear':sum(not d[k]['floor']['clear'] for k in shared)/len(shared),'gap_ctx':sum(not d[k]['floor']['ctx'] for k in shared)/len(shared)} for s,d in rs.items()]
pair={}
for mode in ['old','floor']:
 k=set.intersection(*[{k for k,r in rs[s].items() if r[mode]['q1'] and r[mode]['q2']} for s in ['8B','14B']]);pair[mode]={'n':len(k),**{s:sum(not rs[s][x][mode]['clear'] for x in k)/len(k) for s in ['8B','14B']}}
# Token-overlap exclusion uses all gold aliases and leading-space variants, as in exp2_lens.py.
if not TOKENIZER.exists():
 tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-14B')
else:tok=AutoTokenizer.from_pretrained(str(TOKENIZER))
def ids(aliases):
 return {t[0] for a in aliases for s in [str(a).strip(),' '+str(a).strip()] if s.strip() and (t:=tok.encode(s,add_special_tokens=False))}
by={(r['category'],r['person']):r for r in dataset['data']};lr=jl(S['S3']);lens=[]
for group in ['fail','success']:
 raw=[r for r in lr if r['set']=='clean' and r['group']==group];kept=[r for r in raw if not ids(by[r['category'],r['person']]['A1']) & ids(by[r['category'],r['person']]['Answer'])];hit=sum(min(r['bridge_ranks'])<10 for r in kept)
 lens.append({'group':group,'raw_n':len(raw),'raw_hits':sum(min(r['bridge_ranks'])<10 for r in raw),'excluded':len(raw)-len(kept),'n':len(kept),'hits':hit,'share':hit/len(kept)})
# Verify all C accuracies against prediction files, not just the aggregate reports.
run9='results/handoff_analysis_20261003/exp9_681194/results/handoff_analysis_20261003/exp9_shuffled/runs_681194'
run10='results/handoff_analysis_20261003/exp10_681386/results/handoff_analysis_20261003/exp10_consistency/resumed_681332_681386'
base=None
for report,runroot,conditions in [(exp9,run9,['self_full','shuffled_full','category_confidence50','category_random50']),(exp10,run10,['self_wrong','shuffled_wrong','self_correct'])]:
 for c in conditions:
  vals=[]
  for seed in range(3):
   rows=jl(f'{runroot}/{c}_s{seed}/eval_C_bothNew.jsonl');assert len(rows)==500
   b=100*sum(r['base_ok'] for r in rows)/len(rows)
   if base is None:base=b
   assert b==base;vals.append(100*sum(r['trained_ok'] for r in rows)/len(rows))
  groups=report['groups'] if 'groups' in report else report['scoring']['floor']['groups']
  g=next(g for g in groups if g['condition']==c and g['subset']=='C_bothNew' and g.get('metric','composed')=='composed');assert np.allclose(vals,g['accuracy_pct']['values'])
def group9(c):return next(g for g in exp9['groups'] if g['condition']==c and g['subset']=='C_bothNew')
def group10(c,metric='composed'):return next(g for g in exp10['scoring']['floor']['groups'] if g['condition']==c and g['subset']=='C_bothNew' and g['metric']==metric)
def group8(selection,keep):return next(g for g in exp8['groups'] if g['selection']==selection and g['keep_top']==keep and g['subset']=='C_bothNew' and g['scoring']=='original')
fig2=[{'label':'基线','mean':base,'sd':0,'source':'S6,S7','status':'verified'}, {'label':'仅单步†','mean':hops,'sd':0,'source':'S0','status':'pending'}, {'label':'self 全量','mean':group9('self_full')['accuracy_pct']['mean'],'sd':group9('self_full')['accuracy_pct']['sample_sd'],'source':'S6','status':'verified'}, {'label':'全局置信度50%','mean':group8('confidence',.5)['mean_accuracy_pct'],'sd':group8('confidence',.5)['sample_sd_pp'],'source':'S5','status':'verified'}, {'label':'全局随机50%','mean':group8('random',.5)['mean_accuracy_pct'],'sd':group8('random',.5)['sample_sd_pp'],'source':'S5','status':'verified'}, *[{'label':label,'mean':group10(c)['accuracy_pct']['mean'],'sd':group10(c)['accuracy_pct']['sample_sd'],'source':'S7','status':'verified'} for c,label in [('self_wrong','self 错答案'),('self_correct','self 对答案'),('shuffled_wrong','打乱错答案')]],{'label':'打乱全量','mean':group9('shuffled_full')['accuracy_pct']['mean'],'sd':group9('shuffled_full')['accuracy_pct']['sample_sd'],'source':'S6','status':'verified'}, {'label':'标准答案†','mean':gold,'sd':0,'source':'S0','status':'pending'}]
# Figure 1.
fig,ax=plt.subplots(figsize=(8,3.1));x=np.arange(len(scale));w=.32
for off,field,label,color in [(-w/2,'gap_clear','组合缺口 gap_clear','#234d66'),(w/2,'gap_ctx','提供中间事实 gap_ctx','#7f9da8')]:
 vals=[100*r[field] for r in scale];ax.bar(x+off,vals,w,label=label,color=color)
 for xx,v in zip(x+off,vals):ax.text(xx,v+1.4,f'{v:.1f}',ha='center',fontsize=10)
ax.set_xticks(x,[r['size'] for r in scale]);ax.set_ylim(0,88);ax.set_ylabel('缺口（%）');ax.legend(frameon=False,ncols=2,loc='upper right');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(OUT/'fig1_scaling.pdf');plt.close(fig)
# Figure 2: bars + sample SD; historical single-run claims visibly hatched.
fig,ax=plt.subplots(figsize=(10,3.8));x=np.arange(len(fig2));bars=ax.bar(x,[r['mean'] for r in fig2],color=['#adb8bc' if r['status']=='pending' else ('#a64b3d' if '打乱' in r['label'] else '#315f73') for r in fig2],width=.7)
for i,r in enumerate(fig2):
 if r['status']=='pending':bars[i].set_hatch('///');bars[i].set_facecolor('white');bars[i].set_edgecolor('#888888')
 if r['sd']:ax.errorbar(i,r['mean'],yerr=r['sd'],fmt='none',ecolor='#172936',capsize=3,lw=1.2)
 ax.text(i,r['mean']+r['sd']+1.2,f"{r['mean']:.1f}",ha='center',fontsize=9)
ax.axhline(base,color='#a64b3d',ls='--',lw=1);ax.set_xticks(x,[r['label'] for r in fig2],rotation=32,ha='right');ax.set_ylim(0,66);ax.set_ylabel('C 组组合正确率（%）');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(OUT/'fig2_conditions.pdf');plt.close(fig)
# Figure 3: interval report, not invented point estimates or confidence intervals.
fig,ax=plt.subplots(figsize=(7.5,2.8));ax.hlines(1,patch['repair_low'],patch['repair_high'],color='#315f73',lw=7);ax.plot([patch['repair_low'],patch['repair_high']],[1,1],'|',color='#315f73',ms=16);ax.plot(patch['control'],0,'s',ms=9,mfc='white',mec='#888888');ax.text(patch['repair_high']+.8,1,f"{patch['repair_low']:.0f}–{patch['repair_high']:.0f}%（待核实）",va='center');ax.text(patch['control']+.8,0,f"{patch['control']:.0f}%（待核实）",va='center');ax.set_yticks([0,1],['对照','固定层真回填']);ax.set_ylim(-.6,1.6);ax.set_xlim(0,25);ax.set_xlabel('修复率（%）；交接文档报告范围，非置信区间');ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(OUT/'fig3_patching.pdf');plt.close(fig)
D={'sources':S,'gap_clean':[r for r in gaps if r['category']=='CLEAN'],'scale_common3':scale,'scale_pair':pair,'label_summary':labels,'lens_filtered':lens,'exp8':exp8,'exp9':exp9,'exp10':exp10,'config':cfg,'patch_pending':patch,'hops_pending':hops,'gold_pending':gold,'fig2':fig2,'baseline':base}
(OUT/'report_data.json').write_text(json.dumps(D,ensure_ascii=False,indent=2))
manifest={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in sorted(used)}
manifest.update({'tokenizer/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in TOKENIZER.glob('*') if p.is_file()})
(OUT/'source_sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('All figure values read/recomputed; source files hashed. Common pool',len(shared),'Lens',lens)
