"""Prompt-final residual logit lens. Inference only; hooks never persist tensors."""
import argparse, contextlib, gc, hashlib, importlib.metadata, json, os, platform, sys, time
from pathlib import Path
from prompts import render
from scoring import score_row
HERE=Path(__file__).resolve().parent

def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def verify():
 for line in (HERE/'SHA256SUMS').read_text().splitlines():
  digest,name=line.split('  ',1); assert sha(HERE/name)==digest,name

def target_spec(tok,e):
 strings={'bridge':e['birth_country'],'answer':next(x for x in e['Answer'] if str(x).strip())}
 out={}
 for key,s in strings.items():
  for suffix,prefix in [('', ''),('_space',' ')]:
   ids=tok.encode(prefix+s,add_special_tokens=False); assert ids
   out[key+suffix]={'text':prefix+s,'id':ids[0],'piece':tok.convert_ids_to_tokens(ids[0])}
 return out

def lens_generate(model,tok,entry):
 import torch
 core=model.get_base_model() if hasattr(model,'get_base_model') else model
 trunk=core.model; L=len(trunk.layers); captured={}; targets=entry['targets']
 ids=torch.tensor([entry['prompt_ids']],device=core.device); handles=[]
 def hook_at(layer):
  def hook(module,args,output):
   h=output[0] if isinstance(output,tuple) else output
   assert h.ndim==3 and h.shape[0]==1
   # Slice before normalization/projection; only one position ever projected.
   logits=core.lm_head(trunk.norm(h[:,-1:,:]))[0,0].float()
   assert torch.isfinite(logits).all().item()
   captured[layer]={'layer':layer,'top1_id':int(logits.argmax()),'ranks':{k:1+int((logits>logits[v['id']]).sum()) for k,v in targets.items()}}
  return hook
 handles.append(trunk.embed_tokens.register_forward_hook(hook_at(0)))
 handles.extend(block.register_forward_hook(hook_at(i+1)) for i,block in enumerate(trunk.layers))
 try:
  with torch.inference_mode():
   out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
   actual=int(out.logits[0,-1].argmax()); del out
 finally:
  for handle in handles: handle.remove()
 assert sorted(captured)==list(range(L+1)),sorted(captured)
 assert captured[L]['top1_id']==actual,'final lens vs forward mismatch'
 with torch.inference_mode():
  gen=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,num_beams=1,pad_token_id=tok.eos_token_id,use_cache=True)
 tokens=gen[0,ids.shape[1]:].tolist(); assert tokens and tokens[0]==actual,'final lens vs generation mismatch'
 end=core.generation_config.eos_token_id; end={end} if isinstance(end,int) else set(end or [tok.eos_token_id])
 layers=[captured[i] for i in range(L+1)]
 return {'layers':layers,'num_layers':L,'emergence':{k:next((i for i in range(1,L+1) if captured[i]['ranks'][k]<=10),None) for k in targets},'raw':tok.decode(tokens,skip_special_tokens=True).strip(),'generated_ids':tokens,'generated_tokens':len(tokens),'truncated':len(tokens)>=96 and tokens[-1] not in end,'final_top1_verified':True}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--adapter-root',type=Path);a=ap.parse_args();verify()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 torch.manual_seed(0);torch.set_num_threads(8 if not a.smoke else 4)
 if not a.smoke:
  assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(),'Allocated GPU required'
  assert a.adapter_root is not None
 rows=read(HERE/'inputs/items.jsonl'); assert len(rows)==402
 assert sum(r['fixed303'] for r in rows)==303 and sum(r['D500_known313'] for r in rows)==313
 if a.smoke:rows=rows[:10]
 manifest=json.loads((HERE/'ADAPTER_MANIFEST.json').read_text())
 if not a.smoke:
  for name,info in manifest.items():
   path=a.adapter_root/info['relative_path'];assert sha(path/'adapter_model.safetensors')==info['sha256'],name
   cfg=json.loads((path/'adapter_config.json').read_text());assert cfg['r']==16 and cfg['task_type']=='CAUSAL_LM'
 a.out.mkdir(exist_ok=False)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 entries=[]
 for r in rows:
  for condition in ['C0','C2']:
   text=render(tok,r['e'],condition,'');entries.append({**r,'condition':condition,'prompt':text,'prompt_ids':tok.encode(text,add_special_tokens=False),'targets':target_spec(tok,r['e'])})
 with (a.out/'prompts.jsonl').open('x') as f:
  for e in entries:f.write(json.dumps(e,ensure_ascii=False)+'\n')
 config={'model':str(Path(a.model).resolve()),'smoke':a.smoke,'dtype':'float32' if a.smoke else 'bfloat16','device':'cpu' if a.smoke else 'cuda','enable_thinking':False,'max_new_tokens':96,'do_sample':False,'roster_sha256':sha(HERE/'inputs/items.jsonl'),'preregistration_sha256':sha(HERE/'PREREGISTRATION.txt'),'adapters':manifest if not a.smoke else {},'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','numpy']},'python':sys.version,'platform':platform.platform(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 js(a.out/'config.json',config)
 variants=['base'] if a.smoke else ['base',*manifest]
 with (a.out/'per_item.jsonl').open('x') as f:
  for variant in variants:
   model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32 if a.smoke else torch.bfloat16,attn_implementation='eager')
   if not a.smoke:assert model.config.hidden_size==5120 and model.config.num_hidden_layers==40 and model.config.model_type=='qwen3'
   if variant!='base':
    from peft import PeftModel
    model=PeftModel.from_pretrained(model,a.adapter_root/manifest[variant]['relative_path'],is_trainable=False)
   model.to('cpu' if a.smoke else 'cuda');model.eval();model.requires_grad_(False)
   js(a.out/(variant+'_model_config.json'),model.config.to_dict())
   subset=[e for e in entries if variant=='base' or e['condition']=='C0']
   for n,e in enumerate(subset,1):
    start=time.monotonic(); result=lens_generate(model,tok,e)
    row=score_row({**{k:v for k,v in e.items() if k not in ['prompt','prompt_ids']},**result,'variant':variant,'smoke':a.smoke,'seconds':time.monotonic()-start})
    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    print(f'{variant} {n}/{len(subset)} id={e["id"]} {e["condition"]} top1=OK correct={row["score"]["correct"]}',flush=True)
   del model;gc.collect()
   if torch.cuda.is_available():torch.cuda.empty_cache()
 from analyze import analyze
 analyze(a.out/'per_item.jsonl',a.out,10000)
 actual=read(a.out/'per_item.jsonl');expected=len(rows)*(2+len(variants)-1)
 assert len(actual)==expected and len({(r['id'],r['variant'],r['condition']) for r in actual})==expected
 js(a.out/'complete.json',{'smoke':a.smoke,'items':len(rows),'rows':expected,'all_final_top1_verified':all(r['final_top1_verified'] for r in actual),'analysis_complete':True})
if __name__=='__main__':main()
