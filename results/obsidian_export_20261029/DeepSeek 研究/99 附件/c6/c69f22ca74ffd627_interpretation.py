"""Frozen future interpretation helper; does not run training or evaluation."""
import math,statistics
T95_DF5=2.570581835636305

def interval(xs):
 if len(xs)!=6:raise ValueError('Exactly six paired seeds required')
 mean=statistics.mean(xs);half=T95_DF5*statistics.stdev(xs)/math.sqrt(6)
 return {'mean_pp':mean,'low_pp':mean-half,'high_pp':mean+half}
def classify(c_minus_b,i_minus_b):
 if len(c_minus_b)!=len(i_minus_b):raise ValueError('Pair seeds')
 cb=interval(c_minus_b);ib=interval(i_minus_b);ci=interval([c-i for c,i in zip(c_minus_b,i_minus_b)])
 equivalent=cb['low_pp']>=-1 and cb['high_pp']<=1
 if cb['low_pp']>0 and ci['low_pp']>0:label='自洽带来收益'
 elif cb['high_pp']<0:label='本设计无法检验自洽的正面作用'
 elif equivalent and ib['high_pp']<0 and ci['low_pp']>0:label='自洽仅避免伤害，不足以解释exp12收益'
 else:label='证据不足或不满足三档条件'
 return {'label':label,'consistent_minus_baseline':cb,'inconsistent_minus_baseline':ib,'consistent_minus_inconsistent':ci,'practical_equivalence_within_1pp':equivalent}
if __name__=='__main__':
 assert classify([2]*6,[0]*6)['label']=='自洽带来收益'
 assert classify([0]*6,[-2]*6)['label'].startswith('自洽仅避免伤害')
 assert classify([-.5]*6,[-2]*6)['label']=='本设计无法检验自洽的正面作用'
 assert classify([.5]*6,[-2]*6)['label']=='自洽带来收益'
 assert classify([-4,4,-4,4,-4,4],[-5]*6)['label'].startswith('证据不足')
 print('Three-tier and equivalence interpretation checks passed.')
