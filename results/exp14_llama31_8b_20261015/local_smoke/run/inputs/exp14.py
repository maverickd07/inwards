"""Llama adapter of the unchanged exp12 target/optimizer implementation."""
import argparse,importlib.metadata,json,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
import exp12_source as source
import engine_exp11_v1 as engine
from common import HERE,CONFIG,SPLITS,read,write,js,sha,verify_seal,verify_freeze
from exp1_gap_v2_v1 import to_prompt


def load(a,train=False):
 torch.manual_seed(a.train_seed)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 tok.padding_side='left'
 if tok.eos_token_id is None:raise ValueError('Missing EOS')
 # The official Llama tokenizer has no pad token. EOS padding is masked out.
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
  torch_dtype=torch.float32 if a.smoke_test else torch.bfloat16,
  **({} if a.smoke_test else {'device_map':'cuda'}))
 if model.config.model_type!='llama':raise ValueError('Expected Llama architecture')
 stop=model.generation_config.eos_token_id
 stop=[stop] if isinstance(stop,int) else stop
 if not stop or tok.eos_token_id not in stop:raise ValueError('Tokenizer end-of-turn missing from generation stop ids')
 if not a.smoke_test and (tok.eos_token!='<|eot_id|>' or set(stop)!={128001,128008,128009}):
  raise ValueError('Unexpected official Llama stop tokens')
 if a.adapter:raise ValueError('Every fit must start from a fresh base model')
 model.eval();return tok,model

@torch.no_grad()
def generate(tok,model,questions,a,scores=False):
 result=[];limit=6 if a.smoke_test else 24
 stops=model.generation_config.eos_token_id
 stops=[stops] if isinstance(stops,int) else stops
 for start in range(0,len(questions),a.gen_batch):
  batch=questions[start:start+a.gen_batch]
  enc=tok([to_prompt(tok,q) for q in batch],padding=True,add_special_tokens=False,
          return_token_type_ids=False,return_tensors='pt').to(model.device)
  if enc['input_ids'].shape[1]+limit>2048:raise ValueError('Unexpected long prompt; no silent truncation')
  seq=model.generate(**enc,do_sample=False,max_new_tokens=limit,pad_token_id=tok.pad_token_id)
  continuation=seq[:,enc['input_ids'].shape[1]:].tolist();answers=[];traces=[]
  for q,tokens in zip(batch,continuation):
   end=next((i for i,t in enumerate(tokens) if t in stops),None)
   used=tokens[:end] if end is not None else tokens
   answer=tok.decode(used,skip_special_tokens=True).strip();answers.append(answer)
   traces.append({'question':q,'pred':answer,'generated_tokens':len(used),
                  'hit_token_cap':end is None and len(used)>=limit,'stop_token_id':tokens[end] if end is not None else None})
  stats=source.score_answers(tok,model,batch,answers) if scores else [{} for _ in answers]
  result.extend({'pred':ans,'mean_logprob':s.get('mean_logprob'),**s} for ans,s in zip(answers,stats))
  with (a.out/'generation_trace.jsonl').open('a') as f:
   for r in traces:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 return result


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['targets','baseline','train'],required=True)
 ap.add_argument('--model',required=True);ap.add_argument('--plan',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True);ap.add_argument('--targets',type=Path)
 ap.add_argument('--condition',choices=['context','direct']);ap.add_argument('--train-seed',type=int,default=0)
 ap.add_argument('--smoke-test',action='store_true');a=ap.parse_args()
 verify_freeze()
 a.gen_batch,a.epochs,a.batch,a.round,a.adapter=8,2,8,1,None
 if a.out.exists():raise FileExistsError(a.out)
 if a.smoke_test:torch.set_num_threads(1)
 else:
  if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<38*1024**3:
   raise RuntimeError('Require CUDA GPU with >=38 GiB')
 engine.load=load;source.generate=generate
 started=time.time()
 if a.stage=='train':
  verify_seal(a.out.parent)
  source.train(a)
  cfg=json.loads((a.out/'run_config.json').read_text())
  # Preserve original engine's metadata, then publish explicit experiment metadata.
  js(a.out/'exp14_config.json',{'experiment':CONFIG['experiment'],'training_engine':'unchanged_exp11',
    'original_run_config_sha256':sha(a.out/'run_config.json'),'primary':CONFIG['primary'],
    'scorer':'frozen_v3','model_revision':CONFIG['model_revision']})
 elif a.stage=='targets':source.paired_targets(a)
 else:
  tok,model=load(a);a.out.mkdir();source.evaluate(tok,model,a,a.out)
  js(a.out/'complete.json',{'stage':'baseline','model':'Llama-3.1-8B-Instruct','Qwen_predictions_reused':False})
 js(a.out/'timing.json',{'seconds':time.time()-started,'stage':a.stage})
 js(a.out/'runtime.json',{'python':sys.version,'versions':{p:importlib.metadata.version(p) for p in CONFIG['expected_versions']},
   'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'smoke_test':a.smoke_test})
if __name__=='__main__':main()
