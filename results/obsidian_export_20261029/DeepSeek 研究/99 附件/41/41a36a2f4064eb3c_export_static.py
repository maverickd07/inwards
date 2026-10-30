import sys,json,shutil
sys.dont_write_bytecode=True
from pathlib import Path
P=Path(__file__).resolve().parent;OLD=P.parent/'expxx_self_consistency_20261016';sys.path.insert(0,str(OLD))
from consistency import World,CATS,js
w=World();inputs=P/'package/inputs';inputs.mkdir(exist_ok=True)
qs={};q2={}
for cat in CATS:
 e=next(e for e in w.data if e['category']==cat)
 qs[cat]=e['Question'].replace('the birthplace of '+e['person'],'the country where {person} was born')
 q2[cat]=e['Q2'].replace(e['birth_country'],'{country}')
js(inputs/'catalog.json',{'attributes':w.table,'country_alias_map':{k:sorted(v) for k,v in w.country_map.items()},'question_templates':qs,'q2_templates':q2})
for name in ['PREREGISTRATION_ZH.txt','PREREGISTRATION_FREEZE.json']:shutil.copyfile(P/name,P/'package'/name)
mask=P.parents[1]/'results/exp12_matched785_20261015/frozen_inputs/primary_D303_keys.json';shutil.copyfile(mask,inputs/mask.name)
print('Exported frozen country/attribute/alias catalog and evaluation identities.')
