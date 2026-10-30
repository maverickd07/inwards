import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def pack(run):
    complete = json.loads((run/'complete.json').read_text())
    if complete['models'] != 13 or complete['metric_labels'] != 19500:
        raise ValueError('Incomplete scoring')
    dest = run.parent / (run.name+'_review.tgz')
    files = [p for p in sorted(run.iterdir()) if p.is_file() and p.suffix in ('.json','.jsonl','.txt')]
    log = run.parent / ('rescore-'+run.name.removeprefix('runs_')+'.out')
    if log.exists():
        files.append(log)
    with tarfile.open(dest,'x:gz') as archive:
        for p in files:
            archive.add(p,arcname=str(p.relative_to(run.parent)),recursive=False)
    with Path(str(dest)+'.sha256').open('x') as f:
        f.write(hashlib.sha256(dest.read_bytes()).hexdigest()+'  '+dest.name+'\n')
    print('Review archive:',dest,flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    pack(ap.parse_args().run)
