"""Inference, frozen target pairing, and teacher-forced NLL only. No training API."""
import os,sys
sys.dont_write_bytecode=True
from shared import *
from matching import build_pairs
import argparse,time,statistics,tarfile,platform,importlib.metadata,shutil
from exp1_gap_v2_v1 import to_prompt

def generate(tok,model,questions,batch=8,max_tokens=24):
 import torch
 answers=[];pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
 with torch.inference_mode():
  for start in range(0,len(questions),batch):
   encoded=tok([to_prompt(tok,q) for q in questions[start:start+batch]],padding=True,add_special_tokens=False,return_token_type_ids=False,return_tensors='pt').to(model.device)
   out=model.generate(**encoded,do_sample=False,max_new_tokens=max_tokens,pad_token_id=pad)
   answers.extend(tok.decode(ids[encoded['input_ids'].shape[1]:],skip_special_tokens=True).strip() for ids in out)
   if start%128==0:print('generated',min(start+batch,len(questions)),'/',len(questions),flush=True)
 return answers

def losses(tok,model,pairs):
 import torch
 out=[]
 with torch.inference_mode():
  for i,r in enumerate(pairs):
   for group in ['consistent','inconsistent']:
    target=r[group+'_target'];prefix=tok(to_prompt(tok,r['question']),add_special_tokens=False)['input_ids'];answer=tok(target,add_special_tokens=False)['input_ids'];with_eos=tok(target+tok.eos_token,add_special_tokens=False)['input_ids']
    if not answer or not prefix or with_eos[:len(answer)]!=answer or len(with_eos)<=len(answer):raise ValueError('Empty target or EOS/tokenization mismatch')
    ids=torch.tensor([prefix+with_eos],device=model.device);output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    logits=output.logits[0,len(prefix)-1:-1].float();nll=torch.nn.functional.cross_entropy(logits,ids[0,len(prefix):],reduction='none')
    if not torch.isfinite(nll).all():raise ValueError('Nonfinite target NLL')
    out.append({'person_id':r['person_id'],'person':r['person'],'category':r['category'],'group':group,'target':target,'target_tokens':len(answer),'mean_token_nll':float(nll[:len(answer)].mean()),'sum_token_nll':float(nll[:len(answer)].sum()),'tokens_with_eos':len(with_eos),'mean_token_nll_with_eos':float(nll.mean()),'sum_token_nll_with_eos':float(nll.sum())})
    del output,logits,nll,ids
   if i%50==0:print('NLL pairs',i+1,'/',len(pairs),flush=True)
 return out

def summarize_loss(rows):
 result={}
 for cat in ['all']+CATS:
  arms={}
  for group in ['consistent','inconsistent']:
   rs=[r for r in rows if r['group']==group and(cat=='all' or r['category']==cat)]
   arms[group]={'n':len(rs),'mean_of_example_mean_token_nll':statistics.mean(r['mean_token_nll'] for r in rs) if rs else None,'token_weighted_nll':sum(r['sum_token_nll'] for r in rs)/sum(r['target_tokens'] for r in rs) if rs else None,'mean_of_example_mean_token_nll_with_eos':statistics.mean(r['mean_token_nll_with_eos'] for r in rs) if rs else None,'token_weighted_nll_with_eos':sum(r['sum_token_nll_with_eos'] for r in rs)/sum(r['tokens_with_eos'] for r in rs) if rs else None,'total_target_tokens':sum(r['target_tokens'] for r in rs)}
  c=arms['consistent']['mean_of_example_mean_token_nll'];d=arms['inconsistent']['mean_of_example_mean_token_nll'];arms['consistent_minus_inconsistent_nll']=c-d if c is not None and d is not None else None
  if arms['consistent']['n']:assert arms['consistent']['total_target_tokens']==arms['inconsistent']['total_target_tokens']
  result[cat]=arms
 return result

def run(a):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 t=time.time();manifest=verify_package();inputs=a.inputs;new=read(inputs/'new_people.jsonl');old=read(inputs/'old_people_eligible.jsonl');q1old=read(inputs/'old_q1.jsonl');allpeople=old+new
 if not a.smoke_test and len(new)!=400:raise ValueError('Expected exactly 400 new people')
 a.out.mkdir(parents=True,exist_ok=False);k=Knowledge(inputs)
 (a.out/'provenance').mkdir()
 for name in ['PREREGISTRATION_ZH.txt','PREREGISTRATION_FREEZE.json','INTERPRETATION_CLARIFICATION_ZH.txt','INTERPRETATION_CLARIFICATION_FREEZE.json','PACKAGE_FREEZE.json']:
  shutil.copyfile(HERE/name,a.out/'provenance'/name)
 for name in ['roster_summary.json','country_frequency.json','new_people.jsonl','old_people_eligible.jsonl']:
  shutil.copyfile(inputs/name,a.out/'provenance'/name)
 for name in ['curation_summary.json','evaluation_exclusion_index.json','primary_D303_keys.json']:
  if not a.smoke_test or (inputs/name).exists():shutil.copyfile(inputs/name,a.out/'provenance'/name)
 torch.manual_seed(17)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.padding_side='left'
 kwargs={'torch_dtype':torch.float32} if a.smoke_test else {'torch_dtype':torch.bfloat16,'device_map':'cuda'}
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,**kwargs);model.eval();model.requires_grad_(False)
 versions_before={n:p._version for n,p in model.named_parameters()}
 js(a.out/'runtime.json',{'model':str(a.model),'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),'device':str(model.device),'model_config_sha256':sha(a.model/'config.json'),'generation_batch':a.gen_batch,'max_new_tokens':24,'parameter_count':sum(p.numel() for p in model.parameters()),'all_parameters_frozen':not any(p.requires_grad for p in model.parameters()),'package_freeze_sha256':sha(HERE/'PACKAGE_FREEZE.json'),'smoke_test':a.smoke_test})
 predictions=generate(tok,model,[f"What is the birthplace (country only) of {p['person']}?" for p in new],a.gen_batch)
 q1=[*q1old,*[{'person_id':p['person_id'],'person':p['person'],'prediction':s,'source':'new_base_generation'} for p,s in zip(new,predictions)]]
 assert len(q1)==len(allpeople) and {r['person_id'] for r in q1}=={p['person_id'] for p in allpeople}
 write(a.out/'q1.jsonl',q1);by={p['person_id']:p for p in allpeople}
 countries=sorted({k.country(r['prediction']) for r in q1 if k.country(r['prediction']) and k.country(r['prediction'])!=by[r['person_id']]['birth_country']})
 keys=[(country,cat) for country in countries for cat in CATS];qs=[k.q2(country,cat) for country,cat in keys]
 preds=generate(tok,model,qs,a.gen_batch) if qs else []
 q2=[{'country':country,'category':cat,'question':q,'prediction':pred} for(country,cat),q,pred in zip(keys,qs,preds)];write(a.out/'q2.jsonl',q2)
 frequency=json.loads((inputs/'country_frequency.json').read_text())['frequency'];pairs,audit,stats=build_pairs(allpeople,q1,q2,k,frequency)
 write(a.out/'pairing_audit.jsonl',audit);write(a.out/'matched_pairs.jsonl',pairs)
 js(a.out/'PAIRING_FREEZE.json',{'before_target_loss':True,'matched_pairs_sha256':sha(a.out/'matched_pairs.jsonl'),'n':len(pairs),'country_frequency_sha256':sha(inputs/'country_frequency.json'),'no_target_selection_by_loss':True})
 loss=losses(tok,model,pairs);write(a.out/'target_losses.jsonl',loss);summ=summarize_loss(loss);js(a.out/'target_loss_summary.json',summ)
 assert sha(a.out/'matched_pairs.jsonl')==json.loads((a.out/'PAIRING_FREEZE.json').read_text())['matched_pairs_sha256']
 assert versions_before=={n:p._version for n,p in model.named_parameters()} and all(p.grad is None for p in model.parameters())
 state='STOP_INSUFFICIENT_AND_ASK_USER' if len(pairs)<300 else 'READY_FOR_USER_REVIEW'
 status={'status':state,'stage':'inference_pairing_loss_only','new_people':len(new),'eligible_old_people':len(old),'q1_generated':len(new),'q1_reused':len(old),'q2_generated':len(q2),'pairs_per_group':len(pairs),'matched_unique_people':len({r['person_id'] for r in pairs}),'counts':stats,'training_runs':0,'third_group_runs':0,'automatic_next_stage':False,'model_parameter_versions_unchanged':True,'seconds':time.time()-t,'smoke_test':a.smoke_test,'package_freeze_sha256':sha(HERE/'PACKAGE_FREEZE.json')}
 lines=['exp17 只推理准备阶段完成',state,f'新增人物 {len(new)}；清洁旧人物 {len(old)}；每组严格配对 {len(pairs)} 题。','训练次数=0；第三组=0；无自动后续作业。',f'配对题中的不同人物数：{status["matched_unique_people"]}；同一人物四类别相关。',json.dumps(stats,ensure_ascii=False,indent=2)]
 for cat,v in summ.items():lines.append(cat+' 基础模型平均token损失（题目等权）：一致='+str(v['consistent']['mean_of_example_mean_token_nll'])+'；不一致='+str(v['inconsistent']['mean_of_example_mean_token_nll'])+'；差='+str(v['consistent_minus_inconsistent_nll']))
 lines.append('不足300题时请将结果交回用户决定；即使达到300题，也必须经用户审阅后才能另行准备训练。')
 (a.out/'REPORT_ZH.txt').write_text('\n'.join(lines)+'\n');verify_package();js(a.out/'complete.json',status)
 archive=a.out.parent/(a.out.name+'_review.tgz')
 with tarfile.open(archive,'x:gz') as tar:
  for file in sorted(a.out.rglob('*')):
   if file.is_file():tar.add(file,arcname=str(Path(a.out.name)/file.relative_to(a.out)))
 (archive.parent/(archive.name+'.sha256')).write_text(sha(archive)+'  '+archive.name+'\n')
 print(json.dumps(status,ensure_ascii=False,indent=2),flush=True);print('Review archive:',archive,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--inputs',type=Path,default=HERE/'inputs');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--gen-batch',type=int,default=8);ap.add_argument('--smoke-test',action='store_true');run(ap.parse_args())
