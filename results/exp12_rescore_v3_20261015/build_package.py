import json
import tarfile
from pathlib import Path
from rescore import HERE,sha,js

FILES=['scorer_v3.py','aliases.json','reference_catalog.json','scoring_v2_v1.py',
       'exp1_gap_v2_v1.py','SCORER_FREEZE.json','input_manifest.json','PROTOCOL.txt',
       'rescore.py','preflight.py','submit.sh','run_rescore.sh','pack_review.py',
       'HPC_COMMANDS.txt','LOCAL_VERIFICATION.json','build_package.py']


if __name__ == '__main__':
    validation=json.loads((HERE/'LOCAL_VERIFICATION.json').read_text())
    if not validation['full_pipeline_passed'] or not validation['synthetic_tests_passed']:
        raise ValueError('Validation incomplete')
    for n,digest in validation['tested_source_hashes'].items():
        if sha(HERE/n) != digest:
            raise ValueError('Source changed after validation: '+n)
    with (HERE/'SHA256SUMS').open('x') as f:
        for n in FILES:f.write(sha(HERE/n)+'  '+n+'\n')
    archive=HERE/'exp12_rescore_v3_upload.tgz'
    with tarfile.open(archive,'x:gz') as t:
        for n in FILES+['SHA256SUMS']:t.add(HERE/n,arcname=n,recursive=False)
    with (HERE/'PACKAGE_SHA256.txt').open('x') as f:f.write(sha(archive)+'  '+archive.name+'\n')
    print(archive,archive.stat().st_size,'bytes')
