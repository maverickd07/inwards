"""Local-only validation of the real pretrained smoke artifacts and freeze receipt."""
import hashlib,json,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H))
from run import read,sha
from scoring import score_row
S=H/'local_smoke_final';rows=read(S/'per_item.jsonl');prompts=read(S/'prompts.jsonl');cfg=json.loads((S/'base_model_config.json').read_text());L=cfg['num_hidden_layers']
assert cfg['model_type']=='qwen3' and cfg['hidden_size']==1024 and L==28
assert len(rows)==20 and len({r['id'] for r in rows})==10 and {(r['id'],r['condition']) for r in rows}=={(i,c) for i in range(10) for c in ['C0','C2']}
assert all(r['variant']=='base' and r['smoke'] and r['num_layers']==L for r in rows)
for r in rows:
 assert len(r['layers'])==L+1 and [x['layer'] for x in r['layers']]==list(range(L+1))
 assert r['final_top1_verified'] and r['layers'][-1]['top1_id']==r['generated_ids'][0]
 for layer in r['layers']:
  assert all(isinstance(v,int) and 1<=v<=cfg['vocab_size'] for v in layer['ranks'].values())
 for t,first in r['emergence'].items():
  assert first==next((layer['layer'] for layer in r['layers'][1:] if layer['ranks'][t]<=10),None)
 assert score_row(r)==r
assert len(prompts)==20 and len(list(S.glob('distribution_*.png')))==4
assert json.loads((S/'complete.json').read_text())['analysis_complete']
subprocess.run([sys.executable,str(H/'test_protocol.py')],check=True)
assert sha(H/'scoring.py')==sha(H.parent/'exp19_pointer_20261022/scoring.py')
assert sha(H/'inputs/items.jsonl')==sha(H.parent/'exp19_pointer_20261022/inputs/items.jsonl')
branch=json.loads((H/'synthetic_analysis_validation/VALIDATION.json').read_text());assert branch['six_conditions_and_rescued_subsets_passed']
receipt={'real_pretrained_qwen3_06b_10_items_passed':True,'smoke_model':'Qwen/Qwen3-0.6B pretrained official weights','smoke_dtype':'float32','smoke_device':'cpu','decoder_layers':L,'hook_positions_per_row':L+1,'items':10,'rows':20,'all20_final_top1_verified':True,'finite_ranks_and_first_top10_recomputed':True,'frozen_exp19_scoring_and_roster_identical':True,'full_analysis_and_plots_passed':True,'unit_tests_passed':True,'synthetic_statistical_branch_checks_passed':True,'formal_hpc_submitted':False,'hpc_upload_and_submission_by':'user; no bridge','source_sha256':{p.name:sha(p) for p in [H/'run.py',H/'analyze.py',H/'scoring.py',H/'PREREGISTRATION.txt']},'smoke_output_sha256':{p.name:sha(p) for p in S.iterdir() if p.is_file()},'note':'Initial full smoke local_smoke_01 passed. Final runner smoke local_smoke_final passed. Final plotting/report changes were validated by rerunning analysis on preserved final raw outputs; model/hook implementation unchanged.'}
(H/'LOCAL_VALIDATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
files=[H/'SHA256SUMS',H/'LOCAL_VALIDATION.json']+[H/line.split('  ',1)[1] for line in (H/'SHA256SUMS').read_text().splitlines()]
files+=sorted(p for p in S.iterdir() if p.is_file())
(H/'SMOKE_VERIFIED.sha256').write_text(''.join(sha(p)+'  results/'+H.name+'/'+str(p.relative_to(H))+'\n' for p in files))
print(json.dumps({k:receipt[k] for k in ['decoder_layers','hook_positions_per_row','rows','all20_final_top1_verified','formal_hpc_submitted']},indent=2))
