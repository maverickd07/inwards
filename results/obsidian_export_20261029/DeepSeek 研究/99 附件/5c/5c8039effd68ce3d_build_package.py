"""Explicit code-only upload; no model, targets, prior outputs or smoke artifacts."""
import hashlib
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ['exp12_v1.py', 'engine_exp11_v1.py', 'exp1_gap_v2_v1.py', 'scoring_v2_v1.py',
         'reuse.py', 'train_seed.py', 'run_pipeline.py', 'summarize.py', 'pack_review.py',
         'preflight.py', 'preflight_v1.py', 'submit.sh', 'submit_v1.sh', 'run_exp12.sh',
         'run_exp12_v1.sh', 'source_700454_manifest.json', 'input_hashes.json',
         'PROTOCOL.txt', 'HPC_COMMANDS.txt', 'LOCAL_VERIFICATION.json', 'test_integrity.py',
         'build_package.py']


if __name__ == '__main__':
    verification = json.loads((HERE / 'LOCAL_VERIFICATION.json').read_text())
    if not verification['six_fit_smoke_passed'] or not verification['integrity_checks_passed']:
        raise ValueError('Validation incomplete')
    for name, digest in verification['tested_source_hashes'].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Changed after validation: ' + name)
    with (HERE / 'SHA256SUMS').open('x') as f:
        for name in FILES:
            f.write(hashlib.sha256((HERE / name).read_bytes()).hexdigest() + '  ' + name + '\n')
    archive = HERE / 'exp12_seed345_upload.tgz'
    with tarfile.open(archive, 'x:gz') as tar:
        for name in FILES + ['SHA256SUMS']:
            tar.add(HERE / name, arcname=name, recursive=False)
    with (HERE / 'PACKAGE_SHA256.txt').open('x') as f:
        f.write(hashlib.sha256(archive.read_bytes()).hexdigest() + '  ' + archive.name + '\n')
    print(archive, archive.stat().st_size, 'bytes')
