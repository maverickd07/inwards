"""Sequential greedy inference only, one model, all conditions on every item."""
import argparse,hashlib,importlib.metadata,json,os,platform,sys,time
from pathlib import Path
from prompts import CONDITIONS,choose_control,render,POINTER
from analyze import analyze
HERE=Path(__file__).resolve().parent

def verify():
 for line in (HERE/'SHA256SUMS').read_text().splitlines():
  digest,name=line.split('  ',1)
  assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest, 'Frozen file changed: '+name

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke-test',action='store_true');a=p.parse_args();verify()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 rows=[json.loads(x) for x in (HERE/'inputs/items.jsonl').read_text().splitlines()]
 if a.smoke_test:rows=rows[:3]
 assert a.smoke_test or len(rows)==402
 a.out.mkdir(exist_ok=False)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 ctrl=choose_control(tok,rows)
 torch.manual_seed(0)
 if not a.smoke_test:
  assert 'SLURM_JOB_ID' in os.environ and torch.cuda.is_available(),'Production inference requires allocated Slurm GPU'
  assert torch.cuda.get_device_properties(0).total_memory>=38*1024**3
  cfgpath=Path(a.model)/'config.json';mc=json.loads(cfgpath.read_text());assert mc['model_type']=='qwen3' and mc['hidden_size']==5120,'Expected base Qwen3-14B'
  assert not (Path(a.model)/'adapter_config.json').exists()
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32 if a.smoke_test else torch.bfloat16)
 model.to('cpu' if a.smoke_test else 'cuda');model.eval();torch.set_grad_enabled(False)
 eos=tok.eos_token_id;assert eos is not None
 config={'model':str(Path(a.model).resolve()),'model_config':model.config.to_dict(),'smoke_test':a.smoke_test,'conditions':CONDITIONS,'control_sentence':ctrl,'pointer_sentence':POINTER,'seed':0,'do_sample':False,'enable_thinking':False,'max_new_tokens':96,'batch_size':1,'backend':'transformers','python':sys.version,'platform':platform.platform(),'versions':{m:importlib.metadata.version(m) for m in ['torch','transformers','tokenizers','numpy']},'input_roster_sha256':hashlib.sha256((HERE/'inputs/items.jsonl').read_bytes()).hexdigest(),'model_files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(a.model).iterdir() if f.name in ['config.json','tokenizer_config.json','tokenizer.json','model.safetensors.index.json']},'preregistration_sha256':hashlib.sha256((HERE/'PREREGISTRATION.txt').read_bytes()).hexdigest(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 (a.out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
 # Freeze and verify every rendered length BEFORE first generation.
 prompts=[]
 for r in rows:
  for c in CONDITIONS:
   text=render(tok,r['e'],c,ctrl);ids=tok.encode(text,add_special_tokens=False);prompts.append(dict(id=r['id'],condition=c,prompt=text,prompt_ids=ids,prompt_tokens=len(ids)))
 for i in range(len(rows)):assert prompts[5*i+1]['prompt_tokens']==prompts[5*i+2]['prompt_tokens']
 with (a.out/'prompts.jsonl').open('x') as f:
  for entry in prompts:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
 with (a.out/'raw.jsonl').open('x') as f:
  for index,entry in enumerate(prompts):
   r=rows[index//5];inp=torch.tensor([entry['prompt_ids']],device=model.device);t=time.monotonic()
   gen=model.generate(input_ids=inp,attention_mask=torch.ones_like(inp),max_new_tokens=96,do_sample=False,num_beams=1,pad_token_id=eos,use_cache=True)
   tokens=gen[0,inp.shape[1]:].tolist();end=model.generation_config.eos_token_id;end={end} if isinstance(end,int) else set(end or [eos])
   row={**r,'condition':entry['condition'],'raw':tok.decode(tokens,skip_special_tokens=True).strip(),'generated_ids':tokens,'generated_tokens':len(tokens),'truncated':len(tokens)>=96 and tokens[-1] not in end,'ended_with_eos':bool(tokens and tokens[-1] in end),'seconds':time.monotonic()-t,'smoke_test':a.smoke_test}
   f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
   if (index+1)%25==0:print(f'{index+1}/{len(prompts)} generations',flush=True)
 summary=analyze(a.out/'raw.jsonl',a.out/'analysis')
 assert len(prompts)==len(rows)*5
 (a.out/'complete.json').write_text(json.dumps({'smoke_test':a.smoke_test,'items':len(rows),'generations':len(prompts),'conditions':CONDITIONS,'analysis_complete':True})+'\n')
 print((a.out/'analysis/report.txt').read_text(),flush=True)
if __name__=='__main__':main()
