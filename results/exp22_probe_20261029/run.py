"""Frozen model inference: capture exactly two positions, fit separate linear probes."""
import argparse,gc,json,os,time,sys,importlib.metadata
from pathlib import Path
from probe_utils import *
from prompts import render,SYSTEM
from scoring import score_answer,country_score

def prompt(tok,q):return tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':q}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
def encoded(tok,r):
 text=render(tok,r['e'],'C0','');name=r['person'];start=text.index(name);assert text.count(name)==1
 enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
 span=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>start and a<start+len(name) and b>a];assert span
 pos=[span[-1],len(enc['input_ids'])-1];assert 0<=pos[0]<pos[1]
 return text,enc['input_ids'],pos

def generate(model,tok,text):
 import torch
 ids=tok(text,return_tensors='pt',add_special_tokens=False).to(model.device)
 with torch.inference_mode():gen=model.generate(**ids,do_sample=False,max_new_tokens=96,pad_token_id=tok.eos_token_id)
 tokens=gen[0,ids.input_ids.shape[1]:].tolist();end=model.generation_config.eos_token_id;end={end} if isinstance(end,int) else set(end or [tok.eos_token_id])
 return {'raw':tok.decode(tokens,skip_special_tokens=True).strip(),'generated_ids':tokens,'truncated':len(tokens)==96 and tokens[-1] not in end}

def capture(model,tok,r):
 import torch
 text,ids,pos=encoded(tok,r);core=model.get_base_model() if hasattr(model,'get_base_model') else model;trunk=core.model
 captured={};handles=[]
 def hook(layer):
  def f(module,args,out):
   h=out[0] if isinstance(out,tuple) else out
   x=h[0,pos,:].detach().float().cpu().numpy().astype(np.float16);assert np.isfinite(x).all();captured[layer]=x
  return f
 handles.append(trunk.embed_tokens.register_forward_hook(hook(0)))
 handles.extend(b.register_forward_hook(hook(i+1)) for i,b in enumerate(trunk.layers))
 ids_tensor=torch.tensor([ids],device=model.device)
 try:
  with torch.inference_mode():out=model(input_ids=ids_tensor,attention_mask=torch.ones_like(ids_tensor),use_cache=False,logits_to_keep=1)
  first=int(out.logits[0,-1].argmax());del out
 finally:
  for h in handles:h.remove()
 assert sorted(captured)==list(range(len(trunk.layers)+1))
 result=generate(model,tok,text);assert result['generated_ids'][0]==first
 return np.stack([captured[i] for i in sorted(captured)]),dict(result,prompt=text,prompt_ids=ids,positions=pos,subject_token=tok.convert_ids_to_tokens(ids[pos[0]]),first_top1_verified=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--adapter-root',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();verify()
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.manual_seed(22);torch.set_num_threads(4 if a.smoke else 8)
 if not a.smoke:assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and a.adapter_root
 rows=read(H/'inputs'/('smoke30.jsonl' if a.smoke else 'items.jsonl'));assert len(rows)==(30 if a.smoke else 2402)
 a.out.mkdir(exist_ok=False);train,sp=split(rows);js(a.out/'split.json',sp);js(a.out/'roster.json',rows)
 manifest=json.loads((H/'ADAPTER_MANIFEST.json').read_text());variants=['base'] if a.smoke else ['base',*manifest]
 if not a.smoke:
  for name,v in manifest.items():assert sha(a.adapter_root/v['relative_path']/'adapter_model.safetensors')==v['sha256'],name
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True);assert tok.is_fast
 js(a.out/'config.json',{'smoke':a.smoke,'model':a.model,'variants':variants,'dtype':'float32' if a.smoke else 'bfloat16','cache_dtype':'float16','versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','numpy','scikit-learn']},'protocol_sha256':sha(H/'PREREGISTRATION.txt'),'input_sha256':sha(H/'inputs'/('smoke30.jsonl' if a.smoke else 'items.jsonl')),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
 qcache={};base=[];cache_manifest={}
 with (a.out/'per_item.jsonl').open('x') as f:
  for variant in variants:
   model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32 if a.smoke else torch.bfloat16,attn_implementation='eager')
   if not a.smoke:assert model.config.hidden_size==5120 and model.config.num_hidden_layers==40 and model.config.model_type=='qwen3'
   if variant!='base':
    from peft import PeftModel
    model=PeftModel.from_pretrained(model,a.adapter_root/manifest[variant]['relative_path'],is_trainable=False)
   model.to('cpu' if a.smoke else 'cuda');model.eval();model.requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
   L=model.config.num_hidden_layers;D=model.config.hidden_size;js(a.out/(variant+'_model_config.json'),model.config.to_dict())
   arr=np.lib.format.open_memmap(a.out/(variant+'.npy'),mode='w+',dtype='float16',shape=(len(rows),L+1,2,D))
   for i,r in enumerate(rows):
    x,pred=capture(model,tok,r);assert x.shape==(L+1,2,D);arr[i]=x;arr.flush();s=score_answer(pred['raw'],r['e'])
    if pred['truncated']:s={'correct':False,'status':'REVIEW','reason':'truncated'}
    if variant=='base':
     hops=[]
     for hop in ['Q1','Q2']:
      q=r['e'][hop]
      if q not in qcache:qcache[q]=generate(model,tok,prompt(tok,q))
      hops.append(qcache[q])
     q1=country_score(hops[0]['raw'],r['e']) is True and not hops[0]['truncated']
     qe=dict(r['e'],Answer=r['e']['A2']);q2=score_answer(hops[1]['raw'],qe)['correct'] and not hops[1]['truncated']
     stratum='correct' if s['correct'] else 'missing_q1' if not q1 else 'missing_q2' if not q2 else 'gap'
     base.append({'stratum':stratum,'q1':q1,'q2':q2,'base_correct':s['correct'],'reference_unresolved':s['status'] in ['REVIEW','HISTORICAL'] or hops[0]['truncated'] or hops[1]['truncated'] or country_score(hops[0]['raw'],r['e']) is None or score_answer(hops[1]['raw'],qe)['status'] in ['REVIEW','HISTORICAL'],'first_token_overlap':tok.encode(r['e']['birth_country'],add_special_tokens=False)[0]==tok.encode(str(r['e']['Answer'][0]),add_special_tokens=False)[0]})
    record={'id':r['id'],'index':i,'variant':variant,'person':r['person'],'group':r['group'],'source':r['source'],'fixed303':r['fixed303'],'known313':r['known313'],'category':r['e']['category'],'birth_country':r['e']['birth_country'],'train':bool(train[i]),**base[i],**pred,'score':s}
    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();print(f'{variant} {i+1}/{len(rows)} stratum={base[i]["stratum"]}',flush=True)
   assert versions=={n:p._version for n,p in model.named_parameters()} and all(p.grad is None and not p.requires_grad for p in model.parameters())
   del arr;cache_manifest[variant]={'shape':[len(rows),L+1,2,D],'dtype':'float16','sha256':sha(a.out/(variant+'.npy')),'bytes':(a.out/(variant+'.npy')).stat().st_size}
   del model;gc.collect()
   if torch.cuda.is_available():torch.cuda.empty_cache()
 js(a.out/'hops.json',qcache);js(a.out/'CACHE_MANIFEST.json',cache_manifest)
 from analyze import analyze
 analyze(a.out,a.smoke)
 js(a.out/'complete.json',{'inference_complete':True,'analysis_complete':True,'smoke':a.smoke,'variants':variants,'rows_per_variant':len(rows),'all_weights_frozen':True})
if __name__=='__main__':main()
