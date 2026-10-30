import argparse,hashlib,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
c=json.loads((run/'complete.json').read_text());assert c['items']==402 and c['generations']==2010 and not c['smoke_test']
archive=run.with_name(run.name+'_review.tgz');assert not archive.exists()
with tarfile.open(archive,'x:gz') as tf:
 tf.add(run,arcname=run.name)
 for name in ['PREREGISTRATION.txt','SHA256SUMS','LOCAL_VALIDATION.json','TOKENIZER_PREFLIGHT.json','INPUT_MANIFEST.json']:
  tf.add(run.parent/name,arcname='frozen_protocol/'+name)
archive.with_suffix(archive.suffix+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
