"""Full twelve-fit CPU tiny-Llama integration; synthetic gold is software-only."""
import argparse,shutil,subprocess,sys,time
from common import *

def main():
 import torch
 from tokenizers import Tokenizer,models,pre_tokenizers
 from transformers import PreTrainedTokenizerFast,LlamaConfig,LlamaForCausalLM
 import exp14
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out=a.out.resolve();a.out.mkdir(exist_ok=False)
 torch.set_num_threads(1)
 vocab={w:i for i,w in enumerate(['[UNK]','<|eot_id|>','<|end_of_text|>','<|eom_id|>','France','Euro','Nigeria','Italy','Paris','12','Answer','country','born','fact','question'])}
 backend=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'));backend.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
 tok=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',eos_token='<|eot_id|>',additional_special_tokens=['<|end_of_text|>','<|eom_id|>'])
 tok.chat_template="{% for m in messages %}{{ m['content'] }} {{ eos_token }} {% endfor %}Answer "
 torch.manual_seed(14);model=LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab),hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=512,eos_token_id=[1,2,3])).float()
 model.generation_config.eos_token_id=[1,2,3];model.generation_config.suppress_tokens=[0,1,2,3]
 model.save_pretrained(a.out/'tiny_model');tok.save_pretrained(a.out/'tiny_model')
 plan=a.out/'plan';plan.mkdir();full=read(HERE/'plan/train.jsonl')
 train=[next(r for r in full if r['category']==c) for c in sorted({r['category'] for r in full})];write(plan/'train.jsonl',train)
 # Synthetic aliases match tiny-model predictions solely to exercise nonempty pretrain
 # selection in the complete pipeline. Real-data evaluation is tested separately.
 from types import SimpleNamespace
 from exp1_gap_v2_v1 import clear_question
 probe=a.out/'probe';probe.mkdir();args=SimpleNamespace(model=str(a.out/'tiny_model'),train_seed=0,adapter=None,smoke_test=True,gen_batch=8,out=probe)
 tok,model=exp14.load(args)
 assert tok.pad_token_id==tok.eos_token_id
 for split in SPLITS:
  rows=[r for r in read(HERE/'plan'/f'items_{split}.jsonl') if r['category']=='birthplace_currency'][:2]
  if split=='A_newPerson':rows=read(HERE/'plan'/f'items_{split}.jsonl')[:2]
  for metric,(field,gold) in METRICS.items():
   qs=[clear_question(r['e']) if metric=='composed' else r['e']['Q1' if metric=='q1' else 'Q2'] for r in rows]
   predictions=exp14.generate(tok,model,qs,args)
   if split!='A_newPerson':
    for r,p in zip(rows,predictions):
     assert p['pred'];r['e'][gold]=[p['pred']]
  write(plan/f'items_{split}.jsonl',rows)
 shutil.copy2(HERE/'plan/exp12_manifest.json',plan/'exp12_manifest.json')
 js(a.out/'SMOKE_ONLY.json',{'architecture':'LlamaForCausalLM','gold':'synthetic tiny predictions, software validation only','official_8B_or_GPU_tested':False})
 started=time.time()
 subprocess.run([sys.executable,str(HERE/'run_pipeline.py'),'--model',str(a.out/'tiny_model'),'--out',str(a.out/'run'),'--smoke-test','--smoke-plan',str(plan)],check=True)
 subprocess.run([sys.executable,str(HERE/'pack_review.py'),'--run',str(a.out/'run')],check=True)
 done=json.loads((a.out/'run/complete.json').read_text());assert done['training_runs']==12 and done['smoke_test']
 assert json.loads((a.out/'run/analysis_v3/report.json').read_text())['replication_status']=='software_smoke_only'
 js(a.out/'verification.json',{'passed':True,'fits':12,'seconds':time.time()-started,'official_model_tested':False})
if __name__=='__main__':main()
