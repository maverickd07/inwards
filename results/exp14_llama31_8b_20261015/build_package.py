from common import *
import tarfile
if __name__=='__main__':
 frozen=verify_freeze();names=list(frozen['files'])+['FREEZE.json','LOCAL_VERIFICATION.json']
 with (HERE/'SHA256SUMS').open('x') as f:
  for n in sorted(names):f.write(sha(HERE/n)+'  '+n+'\n')
 archive=HERE/'exp14_upload.tgz'
 with tarfile.open(archive,'x:gz') as tar:
  for n in sorted(names)+['SHA256SUMS']:tar.add(HERE/n,arcname=n,recursive=False)
 js(HERE/'PACKAGE_RECEIPT.json',{'archive':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(names)+1,'submitted':False})
 print(archive,archive.stat().st_size,'bytes')
