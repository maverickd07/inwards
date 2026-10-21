"""Six sequential added fits. No baseline, allocation or teacher generation stage."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from reuse import HERE, NEW_SEEDS, CONDITIONS, js, sha, verify_source, verify_environment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke-test', action='store_true')
    a = ap.parse_args()
    a.source, a.out = a.source.resolve(), a.out.resolve()
    if a.out.exists():
        raise FileExistsError('No overwrite: ' + str(a.out))
    hashes = verify_source(a.source, a.smoke_test)
    if not a.smoke_test:
        verify_environment(a.source)
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    snapshot = a.out / 'reused_700454'
    for name, digest in hashes.items():
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.source / name, dest)
        if sha(dest) != digest:
            raise ValueError('Source changed while copying: ' + name)
    js(a.out / 'reuse_receipt.json', {'source_job': '700454' if not a.smoke_test else 'local_smoke_fixture',
        'source': str(a.source), 'copied_sha256': hashes, 'baseline_regenerated': False,
        'targets_regenerated': False, 'allocation_regenerated': False})
    js(a.out / 'source_hashes.json', {p.name: sha(p) for p in HERE.iterdir()
        if p.suffix in ('.py', '.txt', '.json', '.sh') and p.is_file()})
    commands, stages = [], []
    for seed in NEW_SEEDS:
        for c in CONDITIONS:
            dest = a.out / f'{c}_s{seed}'
            cmd = [sys.executable, '-u', str(HERE / 'train_seed.py'), '--source', str(snapshot),
                   '--model', a.model, '--out', str(dest), '--train-seed', str(seed), '--condition', c]
            if a.smoke_test:
                cmd.append('--smoke-test')
            commands.append(cmd)
            # Progress metadata is unique per stage; no overwrite or resume.
            js(a.out / f'command_{c}_s{seed}.json', cmd)
            print('RUN:', ' '.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            stages.append({'condition': c, 'seed': seed, 'path': dest.name})
    cmd = [sys.executable, str(HERE / 'summarize.py'), '--run', str(a.out)]
    if a.smoke_test:
        cmd.append('--smoke-test')
    subprocess.run(cmd, check=True)
    # Ensure the original run was not changed during this extension.
    if verify_source(a.source, a.smoke_test) != hashes:
        raise ValueError('Source run changed during execution')
    js(a.out / 'commands.json', commands)
    js(a.out / 'stages.json', stages)
    js(a.out / 'complete.json', {'training_runs': 6, 'added_seeds': list(NEW_SEEDS),
        'primary_split': 'D_reservedC', 'smoke_test': a.smoke_test, 'seconds': time.time() - start,
        'baseline_regenerated': False, 'targets_regenerated': False, 'new_scorer_applied': False,
        'next_experiment_authorized': False})
    print('SEED EXTENSION COMPLETE. Original scorer only; awaiting new scorer.', flush=True)


if __name__ == '__main__':
    main()
