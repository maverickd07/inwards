"""User-run download after private `hf auth login`; never print/read token files."""
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
from common import CONFIG,HERE,js,sha,verify_freeze

def main():
 verify_freeze();root=Path.home()/'inward';model=root/CONFIG['model_relative_path']
 receipt=HERE/'MODEL_DOWNLOAD.json'
 if receipt.exists():raise FileExistsError('Download receipt already exists; inspect it before retry')
 info=HfApi().model_info(CONFIG['model_repo'],revision=CONFIG['model_revision'],files_metadata=True)
 assert info.sha==CONFIG['model_revision']
 names=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json']
 selected=[s for s in info.siblings if s.rfilename in names or (s.rfilename.startswith('model-') and s.rfilename.endswith('.safetensors') and '/' not in s.rfilename)]
 assert set(names)<=set(s.rfilename for s in selected)
 # Uses the user's existing HF login through the standard library. No token argument or log.
 snapshot_download(CONFIG['model_repo'],revision=CONFIG['model_revision'],local_dir=model,
                   allow_patterns=[s.rfilename for s in selected],max_workers=2)
 expected={}
 for s in selected:
  p=model/s.rfilename;assert p.is_file() and p.stat().st_size==s.size
  expected[s.rfilename]={'size':s.size,'sha256':s.lfs.sha256 if s.lfs else sha(p)}
 js(receipt,{'repo':CONFIG['model_repo'],'revision':info.sha,'model_relative_path':CONFIG['model_relative_path'],
  'files':expected,'large_weight_hash_verification':'deferred to compute node before inference'})
 print('Pinned model downloaded. Receipt:',receipt,'; Qwen weights retained.')
if __name__=='__main__':main()
