from pathlib import Path
import json,hashlib,datetime,shutil,tarfile,subprocess,collections,sys
P=Path(__file__).resolve().parent;PKG=P/'package'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
summary=json.loads((P/'CURATION_SUMMARY.json').read_text());assert summary['selection_complete']
new=[json.loads(x) for x in (PKG/'inputs/new_people.jsonl').read_text().splitlines()]
assert len(new)==400 and len({r['qid'] for r in new})==400 and all(1985<=r['birth_year']<=1999 for r in new)
assert max(collections.Counter(r['birth_country'] for r in new).values())<=40
historical=json.loads((P/'HISTORICAL_INPUT_HASHES.json').read_text())
for name,d in historical['files'].items():assert sha(P.parents[1]/name)==d,name
for name in ['PREREGISTRATION_FREEZE.json','INTERPRETATION_CLARIFICATION_FREEZE.json']:
 for fn,d in json.loads((P/name).read_text())['files'].items():assert sha(P/fn)==d,fn
for f in P.glob('CURATION_IMPLEMENTATION_ADDENDUM*.json'):shutil.copyfile(f,PKG/f.name)
for name in ['ROSTER_FREEZE.json','HISTORICAL_INPUT_HASHES.json','NEW400_ZH.txt','AUDIT_ZH.txt']:shutil.copyfile(P/name,PKG/name)
shutil.copyfile(P/'local_validation_v2/VALIDATION.json',PKG/'LOCAL_VALIDATION.json')
for f in PKG.glob('*.py'):compile(f.read_text(),str(f),'exec')
for f in PKG.glob('*.sh'):subprocess.run(['bash','-n',str(f)],check=True)
files={str(f.relative_to(PKG)):sha(f) for f in sorted(PKG.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name not in ['PACKAGE_FREEZE.json','SHA256SUMS']}
with (PKG/'PACKAGE_FREEZE.json').open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_people':400,'stage':'inference_only','files':files},f,indent=2)
files['PACKAGE_FREEZE.json']=sha(PKG/'PACKAGE_FREEZE.json')
with (PKG/'SHA256SUMS').open('x') as f:f.write(''.join(d+'  '+name+'\n' for name,d in sorted(files.items())))
archive=P/'exp17_upload.tgz'
with tarfile.open(archive,'x:gz') as t:
 for name in [*sorted(files),'SHA256SUMS']:t.add(PKG/name,arcname=name)
with (P/'exp17_upload.tgz.sha256').open('x') as f:f.write(sha(archive)+'  '+archive.name+'\n')
verify=P/'upload_verification';verify.mkdir(exist_ok=False)
with tarfile.open(archive) as t:
 for m in t.getmembers():
  assert not m.name.startswith('/') and '..' not in Path(m.name).parts and m.isfile()
 t.extractall(verify)
for name,d in files.items():assert sha(verify/name)==d,name
subprocess.run(['shasum','-a','256','-c','SHA256SUMS'],cwd=verify,check=True,capture_output=True)
print('FINAL PACKAGE VERIFIED',archive.stat().st_size,'bytes',len(files),'files')
