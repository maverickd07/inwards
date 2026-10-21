"""Exp11: fixed-protocol target iteration and continued-training comparison.
Gold is diagnostic only. Stages run in independent processes; no original edits.
"""
import argparse,collections,hashlib,json,math,random,statistics,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model,PeftModel
from exp1_gap_v2_v1 import clear_question,to_prompt
from scoring_v2_v1 import is_correct,old_is_correct
FAMILIES={'capital':['birthplace_capital'],'callingcode':['birthplace_callingcode'],'latlng':['birthplace_rounded_lat','birthplace_rounded_lng'],'currency':['birthplace_currency','birthplace_currency_short','birthplace_currency_symbol'],'tld':['birthplace_tld'],'ccn3':['birthplace_ccn3']}
SPLITS=['A_newPerson','B_newCategory','C_bothNew'];TRAIN=['capital','callingcode','latlng']
def key(r):return r['category'],r['person']
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def write(p,rows):
 with Path(p).open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def js(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_unique(rows):
 if len({key(r) for r in rows})!=len(rows):raise ValueError('Duplicate item identity')
def plan(a):
 data=json.loads(Path(a.data).read_text())['data'];by={};fam={c:f for f,cs in FAMILIES.items() for c in cs}
 for e in data:by.setdefault(key(e),e)
 rows=[]
 for r in read(a.items):
  if r['category'] in fam and key(r) in by:
   rows.append({'category':r['category'],'person':r['person'],'e':by[key(r)],'family':fam[r['category']],'pset':'train' if int(hashlib.md5(r['person'].encode()).hexdigest(),16)%2==0 else 'test'})
 check_unique(rows);train=[r for r in rows if r['family'] in TRAIN and r['pset']=='train'];rng=random.Random(0)
 pools={'A_newPerson':[r for r in rows if r['family'] in TRAIN and r['pset']=='test'],'B_newCategory':[r for r in rows if r['family'] not in TRAIN and r['pset']=='train'],'C_bothNew':[r for r in rows if r['family'] not in TRAIN and r['pset']=='test']}
 evals={k:rng.sample(v,min(500,len(v))) for k,v in pools.items()}
 for s,rs in evals.items():
  ref=read(Path(a.reference_evals)/f'eval_{s}.jsonl')
  if list(map(key,rs))!=list(map(key,ref)):raise ValueError('Evaluation identity/order differs from exp8: '+s)
 people=sorted({r['person'] for r in train});random.Random(7).shuffle(people);counts=[]
 for j in range(3):
  group=set(people[j::3]);part=[r for r in train if r['person'] in group];counts.append({'round':j+1,'people':len(group),'items':len(part),'categories':dict(collections.Counter(r['category'] for r in part))})
 allocation={'original_partition':counts,'minimum_items':400,'use_all_each_round':min(r['items'] for r in counts)<400,'full_people':len(people),'full_items':len(train),'training_categories':dict(collections.Counter(r['category'] for r in train)),'evaluation_counts':{s:len(v) for s,v in evals.items()},'source_hashes':{a.items:sha(a.items),a.data:sha(a.data)}}
 # This agreed protocol uses the full pool only; fail rather than silently change allocation.
 if not allocation['use_all_each_round'] or len(train)!=864:raise ValueError('Pool differs from agreed 864-item all-person protocol')
 if a.smoke_test:
  train=[next(r for r in train if r['category']==c) for c in sorted({r['category'] for r in train})]
  evals={s:v[:2] for s,v in evals.items()}
 a.out.mkdir(parents=True,exist_ok=False);write(a.out/'train.jsonl',train)
 for s,rs in evals.items():write(a.out/f'items_{s}.jsonl',rs)
 js(a.out/'allocation.json',allocation);js(a.out/'manifest.json',{'smoke_test':a.smoke_test,'training':list(map(key,train)),'evaluation':{s:list(map(key,v)) for s,v in evals.items()},'full_allocation':allocation})
 if a.smoke_test:
  from tokenizers import Tokenizer,models,pre_tokenizers
  from transformers import PreTrainedTokenizerFast,Qwen3Config,Qwen3ForCausalLM
  voc={w:i for i,w in enumerate(['[PAD]','[EOS]','[UNK]','Answer','France','Nigeria','Paris','12','Euro','fact','country','born'])};backend=Tokenizer(models.WordLevel(voc,unk_token='[UNK]'));backend.pre_tokenizer=pre_tokenizers.Whitespace();tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',eos_token='[EOS]',unk_token='[UNK]');tok.chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}Answer "
  torch.manual_seed(0);model=Qwen3ForCausalLM(Qwen3Config(vocab_size=len(voc),hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=512,pad_token_id=0,eos_token_id=1)).float();model.save_pretrained(a.out/'tiny_model');tok.save_pretrained(a.out/'tiny_model')
 print(json.dumps(allocation,ensure_ascii=False),flush=True);js(a.out/'complete.json',{'stage':'plan'})
def load(a,train=False):
 torch.manual_seed(a.train_seed)
 tok=AutoTokenizer.from_pretrained(a.model);tok.padding_side='left'
 kwargs={'torch_dtype':torch.float32} if a.smoke_test else {'torch_dtype':torch.bfloat16,'device_map':'cuda'}
 model=AutoModelForCausalLM.from_pretrained(a.model,**kwargs)
 if a.adapter:model=PeftModel.from_pretrained(model,a.adapter,is_trainable=train)
 model.eval();return tok,model
@torch.no_grad()
def generate(tok,model,qs,a,scores=False):
 outputs=[];bs=min(a.gen_batch,2) if scores else a.gen_batch;dev=model.device;pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
 for i in range(0,len(qs),bs):
  enc=tok([to_prompt(tok,q) for q in qs[i:i+bs]],return_tensors='pt',padding=True,add_special_tokens=False).to(dev)
  g=model.generate(**enc,max_new_tokens=6 if a.smoke_test else 24,do_sample=False,pad_token_id=pad);width=enc['input_ids'].shape[1];cont=g[:,width:];lps=[None]*len(g)
  if scores and cont.shape[1]:
   mask=torch.ones_like(cont,dtype=torch.bool)
   for special in tok.all_special_ids:mask &= cont!=special
   if tok.eos_token_id is not None:
    eos=cont==tok.eos_token_id;mask &= eos.cumsum(1)==0;att=((eos.cumsum(1)-eos.long())==0).long()
   else:att=torch.ones_like(cont)
   logits=model(input_ids=g,attention_mask=torch.cat([enc['attention_mask'],att],1)).logits[:,width-1:-1].float();lp=logits.log_softmax(-1).gather(-1,cont.unsqueeze(-1)).squeeze(-1)
   lps=[float(lp[j][mask[j]].mean()) if mask[j].any() else None for j in range(len(g))]
  outputs.extend({'pred':tok.decode(c,skip_special_tokens=True).strip(),'mean_logprob':lp} for c,lp in zip(cont,lps))
 return outputs

def targets(a):
 rows=read(a.plan/'train.jsonl');tok,model=load(a);t0=time.time()
 first=generate(tok,model,[r['e']['Q1'] for r in rows],a)
 questions=[f"Background fact: {r['person']} was born in {p['pred']}.\nUsing this fact, answer the question: {clear_question(r['e'])}" for r,p in zip(rows,first)]
 outs=generate(tok,model,questions,a,scores=True);saved=[]
 for r,q,p in zip(rows,first,outs):saved.append({'category':r['category'],'person':r['person'],'pred_q1':q['pred'],'target':p['pred'],'mean_logprob':p['mean_logprob']})
 # Labels are diagnostic; neither target generation nor training allocation uses gold.
 checks=[is_correct(p['target'],r['e']['Answer']) for r,p in zip(rows,saved)]
 a.out.mkdir(parents=True,exist_ok=False);write(a.out/'targets.jsonl',saved);wrong=[p['mean_logprob'] for p,ok in zip(saved,checks) if not ok and p['mean_logprob'] is not None]
 js(a.out/'diagnostic.json',{'n':len(rows),'correct_n':sum(checks),'target_correct_share':sum(checks)/len(rows),'mean_logprob_wrong':statistics.mean(wrong) if wrong else None,'wrong_with_score_n':len(wrong),'null_score_n':sum(p['mean_logprob'] is None for p in saved),'q1_correct_share':sum(is_correct(q['pred'],r['e']['A1']) for r,q in zip(rows,first))/len(rows),'uses_gold_for_training':False,'generator_adapter':a.adapter,'generator_adapter_sha256':sha(Path(a.adapter)/'adapter_model.safetensors') if a.adapter else None,'seconds':time.time()-t0})
 js(a.out/'complete.json',{'stage':'targets','sha256':sha(a.out/'targets.jsonl')});print('targets complete',a.out,flush=True)
def evaluate(tok,model,a,out):
 for s in SPLITS:
  rows=read(a.plan/f'items_{s}.jsonl');comp=generate(tok,model,[clear_question(r['e']) for r in rows],a,scores=True);q1=generate(tok,model,[r['e']['Q1'] for r in rows],a);q2=generate(tok,model,[r['e']['Q2'] for r in rows],a)
  result=[]
  for r,p,x,y in zip(rows,comp,q1,q2):
   e=r['e'];result.append({'category':r['category'],'person':r['person'],'pred':p['pred'],'mean_logprob':p['mean_logprob'],'q1_pred':x['pred'],'q2_pred':y['pred'],'composed_ok':is_correct(p['pred'],e['Answer']),'q1_ok':is_correct(x['pred'],e['A1']),'q2_ok':is_correct(y['pred'],e['A2']),'legacy_composed_ok':old_is_correct(p['pred'],e['Answer'])})
  write(out/f'eval_{s}.jsonl',result);print(s,'accuracy',sum(r['composed_ok'] for r in result)/len(result),flush=True)
def train(a):
 rows=read(a.plan/'train.jsonl');cached=read(a.targets);check_unique(cached);by={key(r):r for r in cached}
 if set(by)!=set(map(key,rows)):raise ValueError('Target identities differ from full train pool')
 tok,model=load(a,train=True);torch.manual_seed(a.train_seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(a.train_seed)
 if not a.adapter:model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
 a.out.mkdir(parents=True,exist_ok=False);init=hashlib.sha256()
 for n,p in model.named_parameters():
  if p.requires_grad:init.update(n.encode());init.update(p.detach().float().cpu().numpy().tobytes())
 js(a.out/'initialization.json',{'train_seed':a.train_seed,'lora_initial_sha256':init.hexdigest(),'continued_from':a.adapter})
 examples=[(clear_question(r['e']),by[key(r)]['target']) for r in rows];write(a.out/'train_examples.jsonl',[{'category':r['category'],'person':r['person'],'question':q,'target':t} for r,(q,t) in zip(rows,examples)])
 steps=a.epochs*math.ceil(len(examples)/a.batch)
 config={'condition':a.condition,'round':a.round,'train_seed':a.train_seed,'epochs':a.epochs,'batch':a.batch,'lr':1e-4,'rank':16,'n_unique':len(rows),'sample_exposures':a.epochs*len(rows),'optimizer_steps':steps,'targets_sha256':sha(a.targets),'init_adapter_sha256':sha(Path(a.adapter)/'adapter_model.safetensors') if a.adapter else None,'optimizer_reset':True,'scheduler_reset':True,'smoke_test':a.smoke_test,'scoring':'floor','model':a.model,'eval_seed':0}
 js(a.out/'run_config.json',config);model.train();model.gradient_checkpointing_enable();model.enable_input_require_grads();model.config.use_cache=False
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.,(s+1)/20)*max(0.,1-s/steps));rng=random.Random(a.train_seed);pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;step=0;t0=time.time()
 with (a.out/'loss.jsonl').open('w') as f:
  for ep in range(a.epochs):
   rng.shuffle(examples)
   for i in range(0,len(examples),a.batch):
    seq=[];labs=[]
    for q,t in examples[i:i+a.batch]:
     p=tok(to_prompt(tok,q),add_special_tokens=False)['input_ids'];v=tok(str(t)+tok.eos_token,add_special_tokens=False)['input_ids'];seq.append(p+v);labs.append([-100]*len(p)+v)
    width=max(map(len,seq));ids=torch.full((len(seq),width),pad);att=torch.zeros_like(ids);lab=torch.full_like(ids,-100)
    for j,(s,l) in enumerate(zip(seq,labs)):ids[j,:len(s)]=torch.tensor(s);att[j,:len(s)]=1;lab[j,:len(l)]=torch.tensor(l)
    loss=model(input_ids=ids.to(model.device),attention_mask=att.to(model.device),labels=lab.to(model.device)).loss
    if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step();opt.zero_grad();step+=1;f.write(json.dumps({'step':step,'epoch':ep+1,'loss':loss.item()})+'\n')
    if step%50==0:print(a.condition,'round',a.round,'seed',a.train_seed,'step',step,'/',steps,'loss',loss.item(),flush=True)
 if step!=steps:raise AssertionError('Budget mismatch')
 model.eval();model.gradient_checkpointing_disable();model.config.use_cache=True;model.save_pretrained(a.out/'adapter');config['training_seconds']=time.time()-t0;eval_start=time.time();evaluate(tok,model,a,a.out);config['evaluation_seconds']=time.time()-eval_start;js(a.out/'run_config.json',config);js(a.out/'complete.json',{'stage':'train','steps':step,'adapter_sha256':sha(a.out/'adapter/adapter_model.safetensors')})
def baseline(a):
 tok,model=load(a);a.out.mkdir(parents=True,exist_ok=False);evaluate(tok,model,a,a.out)
 if a.reference_evals:
  checks=[]
  for split in SPLITS:
   ref=read(Path(a.reference_evals)/f'eval_{split}.jsonl');now=read(a.out/f'eval_{split}.jsonl');ref=ref[:len(now)]
   assert list(map(key,ref))==list(map(key,now))
   checks.append({'subset':split,'n':len(now),'identity_order_match':True,'raw_base_prediction_disagreements':sum(x['base_pred']!=y['pred'] for x,y in zip(ref,now)),'legacy_label_disagreements':sum(x['base_ok']!=y['legacy_composed_ok'] for x,y in zip(ref,now))})
  js(a.out/'reference_check.json',checks)
 js(a.out/'complete.json',{'stage':'baseline'})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['plan','targets','train','baseline']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model');ap.add_argument('--items');ap.add_argument('--data');ap.add_argument('--reference-evals');ap.add_argument('--plan',type=Path);ap.add_argument('--targets',type=Path);ap.add_argument('--adapter');ap.add_argument('--condition',default='');ap.add_argument('--round',type=int,default=0);ap.add_argument('--train-seed',type=int,default=0);ap.add_argument('--epochs',type=int,default=2);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--gen-batch',type=int,default=8);ap.add_argument('--smoke-test',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('No overwrite: '+str(a.out))
 if a.smoke_test:torch.set_num_threads(1)
 t0=time.time();{'plan':plan,'targets':targets,'train':train,'baseline':baseline}[a.stage](a)
 js(a.out/'timing.json',{'stage':a.stage,'total_seconds':time.time()-t0})
if __name__=='__main__':main()
