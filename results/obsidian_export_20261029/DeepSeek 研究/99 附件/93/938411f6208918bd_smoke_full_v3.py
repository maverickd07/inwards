"""Synthetic integration test only. Controlled public-fact fixture outputs; real tiny-model NLL."""
import sys,shutil,json,hashlib,argparse,tarfile
sys.dont_write_bytecode=True
from pathlib import Path
P=Path(__file__).resolve().parent;dest=P/'local_validation_v3';dest.mkdir(exist_ok=False);pkg=dest/'smoke_package';shutil.copytree(P/'package',pkg)
sys.path.insert(0,str(pkg))
from test_offline import tests
from shared import js,write,CATS,sha
import infer
tok,model,people,q1,q2=tests();modeldir=dest/'tiny_model';model.save_pretrained(modeldir);tok.save_pretrained(modeldir)
inputs=pkg/'inputs'
write(inputs/'new_people.jsonl',people);write(inputs/'old_people_eligible.jsonl',[]);write(inputs/'old_q1.jsonl',[]);js(inputs/'country_frequency.json',{'frequency':{'France':1,'Spain':1}});js(inputs/'roster_summary.json',{'new_people':2,'source_review_complete':True,'synthetic':True})
js(inputs/'curation_summary.json',{'synthetic':True});js(inputs/'evaluation_exclusion_index.json',{'synthetic':True})
js(pkg/'PACKAGE_FREEZE.json',{'synthetic':True,'files':{str(f.relative_to(pkg)):sha(f) for f in sorted(pkg.rglob('*')) if f.is_file()}})
k=infer.Knowledge(inputs);responses={k.q2(r['country'],r['category']):r['prediction'] for r in q2};responses.update({f"What is the birthplace (country only) of {p['person']}?":r['prediction'] for p,r in zip(people,q1)})
def fixture_generation(tok,model,questions,batch=8,max_tokens=24):return [responses[q] for q in questions]
infer.generate=fixture_generation
out=dest/'synthetic_run';infer.run(argparse.Namespace(model=modeldir,inputs=inputs,out=out,gen_batch=8,smoke_test=True))
complete=json.loads((out/'complete.json').read_text());assert complete['training_runs']==0 and complete['pairs_per_group']==8 and complete['status']=='STOP_INSUFFICIENT_AND_ASK_USER' and complete['smoke_test']
assert complete['model_parameter_versions_unchanged']
assert (out/'provenance/curation_summary.json').exists() and (out/'provenance/INTERPRETATION_CLARIFICATION_ZH.txt').exists()
assert len((out/'target_losses.jsonl').read_text().splitlines())==16
with tarfile.open(dest/'synthetic_run_review.tgz') as tar:assert 'synthetic_run/complete.json' in tar.getnames()
# Tampering is tested only in the disposable synthetic package; restore exact original bytes afterward.
f=inputs/'country_frequency.json';saved=f.read_bytes();f.write_bytes(saved+b' ')
try:
 infer.verify_package();raise AssertionError('Tamper was not rejected')
except ValueError:pass
finally:f.write_bytes(saved)
infer.verify_package()
js(dest/'VALIDATION.json',{'passed':True,'fixture_generation_is_synthetic':True,'real_tiny_model_NLL':True,'matching_pairs':8,'insufficient_stop':True,'archive_roundtrip':True,'tamper_detection':True,'no_training':True})
print('FULL SYNTHETIC INTEGRATION PASSED')
