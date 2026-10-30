"""Fixed base-trained probes; held-out-person curves and cluster uncertainty."""
import csv,json,warnings,math
from pathlib import Path
from collections import Counter
import numpy as np
from probe_utils import *

def fit_probe(X,y):
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 from sklearn.exceptions import ConvergenceWarning
 if len(set(y))<2:raise ValueError('Probe collapsed to one class; stop, do not call this decoding')
 p=make_pipeline(StandardScaler(),LogisticRegression(C=.01,solver='lbfgs',max_iter=1000,tol=.001))
 with warnings.catch_warnings(record=True) as w:
  warnings.simplefilter('always');p.fit(X,y)
 assert not any(issubclass(a.category,ConvergenceWarning) for a in w),'Probe not converged'
 return p

def bootstrap(values,groups,B=1000):
 if not len(values):return {'n':0,'people':0,'mean':None,'ci95':None,'insufficient':True}
 gs=sorted(set(groups));person=np.array([np.mean(values[groups==g]) for g in gs]);rng=np.random.default_rng(22)
 z=np.mean(person[rng.integers(0,len(gs),(B,len(gs)))],axis=1)
 return {'n':len(values),'people':len(gs),'mean':float(person.mean()),'ci95':list(map(float,np.quantile(z,[.025,.975]))),'insufficient':len(gs)<5}

def analyze(out,smoke=False):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=Path(out);rows=json.loads((out/'roster.json').read_text());raw=read(out/'per_item.jsonl');train,sp=split(rows);test=~train
 assert sp==json.loads((out/'split.json').read_text());assert not set(sp['train_groups'])&set(sp['test_groups'])
 variants=list(json.loads((out/'CACHE_MANIFEST.json').read_text()));base=[r for r in raw if r['variant']=='base'];assert [r['id'] for r in base]==[r['id'] for r in rows]
 by={v:[r for r in raw if r['variant']==v] for v in variants}
 for v in variants:assert [r['id'] for r in by[v]]==[r['id'] for r in rows]
 arr={v:np.load(out/(v+'.npy'),mmap_mode='r') for v in variants};N,K,_,D=arr['base'].shape;L=K-1
 if smoke:assert (N,K,D)==(30,29,1024)
 assert all(x.shape==(N,K,2,D) and x.dtype==np.float16 for x in arr.values())
 groups=np.array([r['group'] for r in rows]);idx=np.flatnonzero(test);gtest=groups[test]
 sets={'smoke30':np.ones(N,bool)} if smoke else {'fixed303':np.array([r['fixed303'] for r in rows]),'known313':np.array([r['known313'] for r in rows]),'new400':np.array([r['source']=='new400' for r in rows])}
 masks={'all':np.ones(N,bool),'correct':np.array([r['stratum']=='correct' for r in base]),'gap':np.array([r['stratum']=='gap' for r in base]),'missing_q1':np.array([r['stratum']=='missing_q1' for r in base]),'missing_q2':np.array([r['stratum']=='missing_q2' for r in base])}
 bands={'early':np.arange(1,L//3+1),'middle':np.arange(L//3+1,2*L//3+1),'late':np.arange(2*L//3+1,L+1)}
 table=[];bandtable=[];contrasts=[];metadata={};gate=[];onsets=[]
 (out/'figures').mkdir(exist_ok=False)
 with (out/'probe_predictions.jsonl').open('x') as pf:
  for target in ['country','answer']:
   y,meta=labels(rows,train,target);metadata[target]=meta;yp=permuted_labels(rows,y,train,target)
   classes,cnt=np.unique(y[train],return_counts=True);prior=dict(zip(classes,cnt/cnt.sum()));major=classes[cnt.argmax()]
   chance=np.array([prior.get(v,0.) for v in y[test]]);majority=(y[test]==major).astype(float)
   for pos,pname in enumerate(['subject','answer']):
    results={v:{'hit':np.zeros((len(idx),K)),'prob':np.zeros((len(idx),K)),'shuffled':np.zeros((len(idx),K)),'pred':[]} for v in variants}
    for layer in range(K):
     X=np.asarray(arr['base'][train,layer,pos,:],dtype=np.float64)
     p=fit_probe(X,y[train]);control=fit_probe(X,yp[train]);pc=p[-1].classes_
     for v in variants:
      Xt=np.asarray(arr[v][test,layer,pos,:],dtype=np.float64);pred=p.predict(Xt);probs=p.predict_proba(Xt);sh=control.predict(Xt)
      results[v]['hit'][:,layer]=pred==y[test];results[v]['shuffled'][:,layer]=sh==y[test]
      results[v]['prob'][:,layer]=[probs[i,np.flatnonzero(pc==truth)[0]] if truth in pc else 0. for i,truth in enumerate(y[test])]
      results[v]['pred'].append(pred.tolist())
     print(f'probe {target} {pname} layer{layer}/{L}',flush=True)
    for v,z in results.items():
     for j,i in enumerate(idx):pf.write(json.dumps({'id':rows[i]['id'],'variant':v,'target':target,'position':pname,'truth':y[i],'predictions':[z['pred'][k][j] for k in range(K)],'hit':z['hit'][j].tolist(),'true_probability':z['prob'][j].tolist(),'shuffled_hit':z['shuffled'][j].tolist()},ensure_ascii=False)+'\n')
     for sn,sm in sets.items():
      selections=dict(masks);selections['rescued']=masks['gap']&np.array([r['score']['correct'] for r in by[v]])
      selections['non_other']=np.array(['other' not in a for a in y])
      selections['non_overlap']=np.array([not r['first_token_overlap'] for r in base])
      selections['no_review']=np.array([not base[i]['reference_unresolved'] and r['score']['status'] not in ['REVIEW','HISTORICAL'] for i,r in enumerate(by[v])])
      for cat in sorted({r['e']['category'] for r in rows}):selections['category:'+cat]=np.array([r['e']['category']==cat for r in rows])
      for stratum,sel in selections.items():
       m=(sm&sel)[test];n=int(m.sum());ng=len(set(gtest[m]))
       for layer in range(K):
        truth=y[test][m];pred=np.array(z['pred'][layer])[m]
        macro=np.mean([np.mean(pred[truth==c]==c) for c in set(truth)]) if n else None
        table.append({'target':target,'position':pname,'variant':v,'set':sn,'stratum':stratum,'layer':layer,'n':n,'people':ng,'accuracy':float(z['hit'][m,layer].mean()) if n else None,'macro_recall':float(macro) if macro is not None else None,'true_probability':float(z['prob'][m,layer].mean()) if n else None,'shuffled_accuracy':float(z['shuffled'][m,layer].mean()) if n else None,'prior_chance':float(chance[m].mean()) if n else None,'majority':float(majority[m].mean()) if n else None})
       for bn,ls in bands.items():
        for metric in ['hit','prob']:
         vals=z[metric][:,ls].mean(axis=1);summ=bootstrap(vals[m],gtest[m]);bandtable.append(dict(target=target,position=pname,variant=v,set=sn,stratum=stratum,band=bn,metric=metric,**summ))
       onset=None
       if n:
        acc=z['hit'][m].mean(axis=0);control=z['shuffled'][m].mean(axis=0);threshold=chance[m].mean()+.05
        for k in range(1,K-2):
         if np.all(acc[k:k+3]>=control[k:k+3]+.05) and np.all(acc[k:k+3]>=threshold):onset=k;break
       onsets.append(dict(target=target,position=pname,variant=v,set=sn,stratum=stratum,onset=onset,n=n))
      # Paired comparisons on same held-out people/questions; middle band primary.
      comps=[('minus_base','base')]
      if v.startswith('context'):comps.append(('minus_direct','direct'+v[len('context'):]))
      for desc,ref in comps:
       for sub in ['all','correct','gap','rescued']:
        mask=(sm&selections[sub])[test]
        for bn,ls in bands.items():
         for metric in ['hit','prob']:
          delta=(z[metric][:,ls]-results[ref][metric][:,ls]).mean(axis=1)
          contrasts.append(dict(target=target,position=pname,variant=v,contrast=desc,set=sn,stratum=sub,band=bn,metric=metric,**bootstrap(delta[mask],gtest[mask])))
      # P1 and P2 are explicit contrasts, not separate significance searches.
      for bn,ls in bands.items():
       mask=(sm&masks['gap'])[test];delta=(z['hit'][:,ls]-z['shuffled'][:,ls]).mean(axis=1)
       contrasts.append(dict(target=target,position=pname,variant=v,contrast='gap_minus_shuffled',set=sn,stratum='gap',band=bn,metric='hit',**bootstrap(delta[mask],gtest[mask])))
       # Independent outcome groups resampled separately, with shared-person cluster IDs.
       ma=(sm&masks['gap'])[test];mb=(sm&masks['correct'])[test];gs=sorted(set(gtest[ma|mb]));vals=z['hit'][:,ls].mean(axis=1);rng=np.random.default_rng(22)
       a=np.array([vals[ma&(gtest==g)].mean() if np.any(ma&(gtest==g)) else np.nan for g in gs]);b=np.array([vals[mb&(gtest==g)].mean() if np.any(mb&(gtest==g)) else np.nan for g in gs]);dist=[]
       for _ in range(1000):
        ii=rng.integers(0,len(gs),len(gs)) if gs else [];aa=a[ii];bb=b[ii]
        if np.isfinite(aa).any() and np.isfinite(bb).any():dist.append(float(np.nanmean(aa)-np.nanmean(bb)))
       contrasts.append(dict(target=target,position=pname,variant=v,contrast='gap_minus_correct',set=sn,band=bn,metric='hit',gap_n=int(ma.sum()),correct_n=int(mb.sum()),people=len(gs),mean=float(np.nanmean(a)-np.nanmean(b)) if np.isfinite(a).any() and np.isfinite(b).any() else None,ci95:list(map(float,np.quantile(dist,[.025,.975]))) if dist else None,insufficient=len(gs)<5))
     if smoke and v=='base' and target=='country':
      from scipy.stats import binom
      nullp=max(float(chance.mean()),float(majority.mean()));bound=float(binom.ppf(.995,len(idx),nullp)/len(idx));obs=float(z['shuffled'][:,1:].mean());gate.append({'position':pname,'observed':obs,'bound':bound,'chance':float(chance.mean()),'n_people':len(idx),'passed':obs<=bound})
    # Full held-out subgroup curves. Solid true labels, dashed shuffled controls.
    for sn in sets:
     fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
     for ax,sub in zip(axes,['all','correct','gap']):
      for v in variants:
       rr=[r for r in table if r['target']==target and r['position']==pname and r['variant']==v and r['set']==sn and r['stratum']==sub]
       ax.plot([r['layer'] for r in rr],[r['accuracy'] for r in rr],label=v);ax.plot([r['layer'] for r in rr],[r['shuffled_accuracy'] for r in rr],ls='--',alpha=.4)
      ax.set_title(f'{sub} n={rr[0]["n"]}');ax.set_xlabel('layer (0=embedding)');ax.set_ylim(0,1)
     axes[0].set_ylabel('held-out accuracy');axes[-1].legend(fontsize=7);fig.suptitle(f'{sn} / {target} / {pname}; dashed=shuffled');fig.tight_layout();fig.savefig(out/'figures'/f'{sn}_{target}_{pname}.png',dpi=140);plt.close(fig)
 def csvwrite(name,rr):
  with (out/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 csvwrite('layer_accuracy.csv',table);js(out/'band_summary.json',bandtable);js(out/'contrasts.json',contrasts);js(out/'onsets.json',onsets);js(out/'LABEL_MAPPING.json',metadata)
 js(out/'SMOKE_GATE.json',{'checks':gate,'passed':all(g['passed'] for g in gate) if smoke else None,'dimension_passed':True,'person_split_passed':True,'finite_cache_passed':all(np.isfinite(x).all() for x in arr.values())})
 if smoke:assert gate and all(g['passed'] for g in gate),'Shuffled probe smoke gate failed; no seed retries'
 report=['exp22 '+('SMOKE ONLY — not scientific evidence' if smoke else 'held-out-person linear probe diagnostics'),'Train groups: '+str(len(sp['train_groups']))+'; test groups: '+str(len(sp['test_groups'])),'All classifier/scaler fits use base TRAIN only; model weights frozen.','Primary contrasts below: middle-third country readout. CIs cluster by person; two seeds are separate.','Positive probe readout is representation evidence, not causal knowledge use. Fixed-base probe transfer can reflect representation coordinate drift.','Outcome strata and rescues are descriptive; reference errors and teacher-training overlap remain limitations.','Report every set separately; overlapping old sets are not independent replications.']
 for c in contrasts:
  if c['target']=='country' and c['band']=='middle':report.append(json.dumps(c,ensure_ascii=False))
 report+=['Smoke shuffled gate: '+json.dumps(gate),'See layer_accuracy.csv, probe_predictions.jsonl, band_summary.json, contrasts.json, LABEL_MAPPING.json, CACHE_MANIFEST.json.','Stage ends here. No automatic added seeds/training/experiments.']
 (out/'report.txt').write_text('\n'.join(report)+'\n')
