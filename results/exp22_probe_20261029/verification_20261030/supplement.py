"""Post-result diagnostic only: OTHER coverage and gap non-other readout."""
import json,collections
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parents[1];p=H/'received/download_SaGWid/exp22_probe_20261029/runs_726630';o=Path(__file__).resolve().parent
rs=[json.loads(s) for s in (p/'per_item.jsonl').read_text().splitlines()];by={(r['variant'],r['id']):r for r in rs};groups=collections.defaultdict(list);coverage=collections.Counter();hist=collections.defaultdict(collections.Counter)
for line in (p/'probe_predictions.jsonl').open():
 x=json.loads(line);r=by[x['variant'],x['id']]
 if x['target']!='country':continue
 sets=['new400'] if r['source']=='new400' else [s for s in ['fixed303','known313'] if r[s]]
 for sn in sets:
  if x['position']=='subject' and x['variant']=='base':coverage[sn,'all']+=1;coverage[sn,'other']+=int(x['truth']=='other')
  if r['stratum']=='gap' and x['truth']!='other':
   h=np.mean(x['hit'][14:27]);s=np.mean(x['shuffled_hit'][14:27]);groups[sn,x['variant'],x['position']].append((r['group'],h,s))
  if r['stratum']=='gap':hist[sn,x['variant'],x['position']].update(x['predictions'][14:27])
ans=[]
for (sn,v,pos),a in groups.items():
 gs=sorted({x[0] for x in a});hits=np.array([np.mean([x[1] for x in a if x[0]==g]) for g in gs]);sh=np.array([np.mean([x[2] for x in a if x[0]==g]) for g in gs]);delta=hits-sh;rng=np.random.default_rng(22);samples=delta[rng.integers(len(gs),size=(1000,len(gs)))].mean(1)
 ans.append({'set':sn,'variant':v,'position':pos,'gap_nonother_items':len(a),'people':len(gs),'hit_person_mean':float(hits.mean()),'shuffled_person_mean':float(sh.mean()),'difference_ci95':list(map(float,np.quantile(samples,[.025,.975]))),'diagnostic_after_results':True})
out={'coverage':[{ 'set':sn,'heldout_items':coverage[sn,'all'],'other':coverage[sn,'other']} for sn in ['fixed303','known313','new400']],'gap_nonother':ans,'prediction_histograms':[{'set':k[0],'variant':k[1],'position':k[2],'counts':dict(v)} for k,v in hist.items()]}
(o/'SUPPLEMENT_OTHER.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['coverage']);print([x for x in ans if x['set']=='new400' and x['variant']=='base']);print([x for x in out['prediction_histograms'] if x['set']=='fixed303' and x['variant']=='context_s0' and x['position']=='answer'])
