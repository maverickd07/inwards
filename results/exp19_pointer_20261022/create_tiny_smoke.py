"""Create a local random 1-layer Qwen3 using the real Qwen tokenizer; no training."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True);p.add_argument('--model-out',type=Path,required=True);p.add_argument('--run-out',type=Path,required=True);a=p.parse_args()
import torch
from transformers import AutoTokenizer,Qwen3Config,Qwen3ForCausalLM
from prompts import choose_control
here=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (here/'inputs/items.jsonl').read_text().splitlines()]
tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True);print('Full 402-item matched control:',choose_control(tok,rows))
a.model_out.mkdir(exist_ok=False);torch.manual_seed(0)
config=Qwen3Config(vocab_size=len(tok),hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=1024,eos_token_id=tok.eos_token_id,bos_token_id=tok.bos_token_id,pad_token_id=tok.eos_token_id)
model=Qwen3ForCausalLM(config);model.save_pretrained(a.model_out);tok.save_pretrained(a.model_out)
a.run_out.parent.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(here/'run.py'),'--model',str(a.model_out),'--out',str(a.run_out),'--smoke-test'],check=True)
