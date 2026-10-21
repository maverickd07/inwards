import sys
sys.dont_write_bytecode=True
from shared import *
from matching import assignment,build_pairs
from infer import losses,summarize_loss,generate
import itertools,random,tempfile,math

def tests():
 rng=random.Random(17)
 for n in range(1,7):
  for _ in range(12):
   cost=[[1 if i==j else rng.choice([0,999999]) for j in range(n)] for i in range(n)]
   out=assignment(cost)
   assert sum(cost[i][j] for i,j in enumerate(out))==min(sum(cost[i][j] for i,j in enumerate(p)) for p in itertools.permutations(range(n)))
 k=Knowledge(HERE/'inputs');people=[{'person_id':'fixture_a','person':'Fixture Alpha','qid':'FIXA','birth_country':'France','source':'synthetic'},{'person_id':'fixture_b','person':'Fixture Beta','qid':'FIXB','birth_country':'Spain','source':'synthetic'}]
 q1=[{'person_id':'fixture_a','prediction':'Canada'},{'person_id':'fixture_b','prediction':'Germany'}]
 targets={'Canada':['+1','Ottawa','56','-106'],'Germany':['+49','Berlin','51','10']}
 q2=[{'country':co,'category':ca,'prediction':t} for co,ts in targets.items() for ca,t in zip(CATS,ts)]
 pairs,audit,stats=build_pairs(people,q1,q2,k,{'France':1,'Spain':1})
 assert len(pairs)==8 and len(audit)==8
 for r in pairs:
  assert r['donor_country'] not in [r['z_country'],r['birth_country']]
  assert not r['consistent_label']['correct'] and not r['inconsistent_label']['correct']
 # Shared calling-code aliases do not constitute an inconsistent target.
 q2shared=[{**r,'prediction':'+1'} if r['category']==CATS[0] else r for r in q2]
 p2,_,_=build_pairs(people,q1,q2shared,k,{'France':1,'Spain':1});assert len(p2)==6
 # Missing or correct Q1s cannot become eligible by changing target choice.
 assert build_pairs(people,[{'person_id':'fixture_a','prediction':'France'},{'person_id':'fixture_b','prediction':'Spain'}],q2,k,{})[0]==[]
 assert build_pairs(people,[{'person_id':'fixture_a','prediction':'Not a country'},{'person_id':'fixture_b','prediction':'Spain'}],q2,k,{})[0]==[]
 # Real tiny-model forward pass checks NLL indexing/EOS semantics against an independent teacher-forced gather.
 import torch
 from transformers import PreTrainedTokenizerFast,Qwen3Config,Qwen3ForCausalLM
 from tokenizers import Tokenizer,models,pre_tokenizers
 vocab={'[PAD]':0,'[EOS]':1,'[UNK]':2};words=['What','is','the','country','where','was','born','Fixture','Alpha','Beta','of','capital','calling','code','latitude','longitude','rounded','down','Canada','Germany','Ottawa','Berlin','France','Spain','+','1','49','56','51','-','106','10','Answer',':','?','.']
 for word in words:vocab.setdefault(word,len(vocab))
 backend=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'));backend.pre_tokenizer=pre_tokenizers.Whitespace()
 tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',eos_token='[EOS]',unk_token='[UNK]');tok.chat_template="{% for m in messages %}{{m['content']}} {% endfor %}Answer "
 torch.manual_seed(17);torch.set_num_threads(1)
 model=Qwen3ForCausalLM(Qwen3Config(vocab_size=len(vocab),hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=16,max_position_embeddings=1024,pad_token_id=0,eos_token_id=1)).float().eval();model.requires_grad_(False)
 versions={n:p._version for n,p in model.named_parameters()};lr=losses(tok,model,pairs);summary=summarize_loss(lr);assert len(lr)==16
 from exp1_gap_v2_v1 import to_prompt
 for pair in pairs:
  for group in ['consistent','inconsistent']:
   answer=tok(pair[group+'_target'],add_special_tokens=False)['input_ids'];prefix=tok(to_prompt(tok,pair['question']),add_special_tokens=False)['input_ids'];ids=torch.tensor([prefix+answer])
   with torch.inference_mode():lp=model(ids).logits[0].float().log_softmax(-1)
   expected=sum(-float(lp[len(prefix)+j-1,t]) for j,t in enumerate(answer))/len(answer)
   actual=next(r['mean_token_nll'] for r in lr if r['person_id']==pair['person_id'] and r['category']==pair['category'] and r['group']==group)
   assert math.isclose(expected,actual,abs_tol=1e-5)
 assert versions=={n:p._version for n,p in model.named_parameters()};assert all(p.grad is None for p in model.parameters())
 outputs=generate(tok,model,['What is the capital of Canada?'],batch=1,max_tokens=2);assert len(outputs)==1
 assert summarize_loss([])['all']['consistent']['mean_of_example_mean_token_nll'] is None
 print(json.dumps({'passed':True,'matching_fixture_pairs':8,'real_tiny_model_loss_records':16,'shared_code_and_wrong_q1_checks':True,'NLL_equals_independent_gather':True,'parameters_unchanged':True}))
 return tok,model,people,q1,q2
if __name__=='__main__':tests()
