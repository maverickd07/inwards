import sys
sys.dont_write_bytecode=True
from consistency import *
from analyze import verdict,delta
assert parse(CATS[2],'-12.2')==frozenset(['-13'])
assert parse(CATS[2],'-12.2','round')==frozenset(['-12'])
assert parse(CATS[2],'-12.5','round')==frozenset(['-13'])
assert parse(CATS[2],'12 S')==frozenset(['-12'])
assert parse(CATS[2],'12 E') is None
assert parse(CATS[2],'91') is None
assert parse(CATS[3],'181') is None
assert parse(CATS[3],'roughly 25') is None
assert parse(CATS[3],'25, 26') is None
assert parse(CATS[1],'Kiev')==parse(CATS[1],'Kyiv')
assert parse(CATS[1],'The capital is Paris')==parse(CATS[1],'Paris')
assert parse(CATS[1],'Paris or London') is not None # no table match => unknown, not substring matching
assert parse(CATS[0],'+1 809 / +1 829')==frozenset(['1809','1829','1849'])
assert parse(CATS[0],'code: 44') is None
w=World.__new__(World);w.code_roots={'1','44','33','1809'}
assert w.compatible({'1'},{'1242'},CATS[0])
assert not w.compatible({'1242'},{'1441'},CATS[0])
assert not w.compatible({'33'},{'44'},CATS[0])
w.country=lambda z:z if z in ['A','B','C'] else None
w.attrs={'floor':{'A':{CATS[0]:{'1'}},'B':{CATS[0]:{'1242'}},'C':{CATS[0]:{'44'}}}}
a=w.attribute(CATS[0],'B','+1');assert a['status']=='consistent' and a['implied_countries']==['A','B']
assert w.attribute(CATS[0],'C','+1')['status']=='contradictory'
assert w.attribute(CATS[0],'C','+99')['status']=='unknown'
assert w.attribute(CATS[0],'D','+1')['status']=='unknown'
w.lookup_cache={};w.indexes={'exp12':{('A',CATS[1]):[{'prediction':'Paris'},{'prediction':'London'}]},'exp1':{('A',CATS[1]):[{'prediction':'Paris'}]}}
assert w.lookup(CATS[1],'A','exp12')['reason']=='saved_q2_conflicting'
assert w.model(CATS[1],'A','Paris','extended')['status']=='unknown'
assert w.model(CATS[1],'A','Paris','exp1')['status']=='consistent'
assert verdict(9.99)=='自洽解释不了' and verdict(10)=='不确定' and verdict(20)=='自洽是候选机制'
print('Synthetic checks passed (parsing, shared-country sets, missing/conflicting knowledge, threshold boundaries).')
