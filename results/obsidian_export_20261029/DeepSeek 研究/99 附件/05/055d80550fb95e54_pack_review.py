"""Export only small review files; leave every adapter and original result intact."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def pack(run):
    run = run.resolve()
    complete = json.loads((run / 'complete.json').read_text())
    if complete['training_runs'] != 6 or complete['added_seeds'] != [3, 4, 5]:
        raise ValueError('Not a completed six-fit extension')
    dest = run.parent / f'{run.name}_review.tgz'
    files = [p for p in sorted(run.rglob('*')) if p.is_file()
             and 'adapter' not in p.relative_to(run).parts
             and p.suffix in ('.json', '.jsonl', '.txt', '.csv')]
    if sum(p.stat().st_size for p in files) > 100 * 1024**2:
        raise ValueError('Review unexpectedly large; inspect before export')
    log = run.parent / f'exp12-{run.name.removeprefix("runs_")}.out'
    if log.is_file():
        files.append(log)
    with tarfile.open(dest, 'x:gz') as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(run.parent)), recursive=False)
    with dest.with_suffix(dest.suffix + '.sha256').open('x') as f:
        f.write(hashlib.sha256(dest.read_bytes()).hexdigest() + '  ' + dest.name + '\n')
    print('Review archive:', dest, dest.stat().st_size, 'bytes', flush=True)
    return dest


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    pack(ap.parse_args().run)
