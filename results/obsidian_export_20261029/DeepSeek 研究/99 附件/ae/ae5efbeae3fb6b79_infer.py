"""Read-only closed-book inference. This process receives questions, not reference answers."""
import argparse,json,time,sys,importlib.metadata
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from common import HERE,read,js,sha,verify_freeze
SYSTEM='Answer the question with only the answer itself: a name, a number, or a short phrase. No explanation.'

def prompt(tok,q):
 return tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':q}],tokenize=False,add_generation_prompt=True,enable_thinking=False)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--adapter',type=Path);ap.add_argument('--queries',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--name',required=True);ap.add_argument('--timing-queries',type=Path);ap.add_argument('--smoke-test',action='store_true');a=ap.parse_args()
 verify_freeze()
 if a.out.exists():raise FileExistsError('No overwrite: '+str(a.out))
 a.out.mkdir(parents=True);started=time.time();torch.manual_seed(0)
 if a.smoke_test:torch.set_num_threads(1)
 else:
  assert torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory>=38*1024**3
  expected=json.loads((HERE/'inference_config.json').read_text());assert torch.version.cuda==expected['cuda']
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False);tok.padding_side='left'
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,**({'torch_dtype':torch.float32} if a.smoke_test else {'torch_dtype':torch.bfloat16,'device_map':'cuda'}))
 if a.adapter:model=PeftModel.from_pretrained(model,str(a.adapter),is_trainable=False,local_files_only=True)
 model.eval();load_seconds=time.time()-started
 config=json.loads((HERE/'inference_config.json').read_text());batch=config['batch'];max_new=config['max_new_tokens'] if not a.smoke_test else 6
 def generate(rows,dest):
  start=time.time();count=0
  with dest.open('x') as f,torch.inference_mode():
   for offset in range(0,len(rows),batch):
    part=rows[offset:offset+batch]
    assert all(set(r)=={'query_id','question'} for r in part), 'Inference inputs must contain questions only'
    texts=[prompt(tok,r['question']) for r in part]
    enc=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
    max_prompt=config['max_prompt_tokens']
    if enc['input_ids'].shape[1]>max_prompt:raise ValueError('Prompt too long; no silent truncation')
    outputs=model.generate(**enc,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.pad_token_id)
    width=enc['input_ids'].shape[1]
    for row,tokens in zip(part,outputs[:,width:]):
     values=tokens.tolist();stop=next((i for i,x in enumerate(values) if x==tok.eos_token_id),len(values));used=values[:stop]
     pred=tok.decode(used,skip_special_tokens=True).strip()
     record={'query_id':row['query_id'],'question':row['question'],'prediction':pred,'answer_tokens':len(used),'hit_token_limit':stop==len(values) and len(values)>=max_new,'contains_think_markup':'<think>' in pred or '</think>' in pred}
     f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');count+=1
    f.flush()
    if offset//batch%25==0:print(a.name,count,'/',len(rows),'queries',flush=True)
  return {'queries':count,'seconds':time.time()-start}
 if a.timing_queries:
  timing=generate(read(a.timing_queries),a.out/'timing_predictions.jsonl');timing['purpose']='Throughput only; no reference scoring, selection or parameter tuning';js(a.out/'timing.json',timing)
 stats=generate(read(a.queries),a.out/'predictions.jsonl')
 js(a.out/'runtime.json',{'python':sys.version,'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','numpy','accelerate']},'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'smoke_test':a.smoke_test})
 js(a.out/'complete.json',{'name':a.name,'smoke_test':a.smoke_test,'queries_sha256':sha(a.queries),'predictions_sha256':sha(a.out/'predictions.jsonl'),'adapter_sha256':sha(a.adapter/'adapter_model.safetensors') if a.adapter else None,'config_sha256':sha(HERE/'inference_config.json'),'load_seconds':load_seconds,**stats})
 print(a.name,'COMPLETE',stats,flush=True)
if __name__=='__main__':main()
