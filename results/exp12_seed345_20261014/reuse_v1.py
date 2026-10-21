"""Read-only verification of the exact completed source run; no ML imports."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = ('context', 'direct')
NEW_SEEDS = (3, 4, 5)
METRICS = {'composed_ok': ('pred', 'Answer'), 'q1_ok': ('q1_pred', 'A1'),
           'q2_ok': ('q2_pred', 'A2')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def js(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def key(row):
    return row['category'], row['person']


def verify_source(source, smoke=False):
    source = Path(source).resolve()
    manifest = json.loads((HERE / 'source_700454_manifest.json').read_text())
    for name, digest in manifest['frozen_code'].items():
        if sha(HERE / name) != digest:
            raise ValueError('Frozen implementation changed: ' + name)
    completion = json.loads((source / 'complete.json').read_text())
    if completion['smoke_test'] != smoke or completion['training_runs'] != 6:
        raise ValueError('Source completion/smoke status mismatch')
    hashes = {}
    for name, digest in manifest['files'].items():
        actual = sha(source / name)
        if not smoke and actual != digest:
            raise ValueError('Source 700454 checksum mismatch: ' + name)
        hashes[name] = actual
    train = read(source / 'plan/train.jsonl')
    rows = read(source / 'plan/items_D_reservedC.jsonl')
    if len(train) != (4 if smoke else 864) or len(rows) != (2 if smoke else 500):
        raise ValueError('Unexpected train/evaluation count')
    if len({key(r) for r in train}) != len(train) or len({key(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate identities')
    if {r['person'] for r in train} & {r['person'] for r in rows}:
        raise ValueError('Train/evaluation person leakage')
    if {r['family'] for r in train} & {r['family'] for r in rows}:
        raise ValueError('Train/evaluation family leakage')
    targets_meta = json.loads((source / 'paired_targets/complete.json').read_text())
    for condition in CONDITIONS:
        target = source / f'paired_targets/{condition}_targets.jsonl'
        if sha(target) != targets_meta['target_sha256'][condition]:
            raise ValueError('Target hash mismatch')
        if list(map(key, read(target))) != list(map(key, train)):
            raise ValueError('Training target identity/order mismatch')
    expected_keys = list(map(key, rows))
    for folder in ['baseline'] + [f'{c}_s{s}' for s in range(3) for c in CONDITIONS]:
        if list(map(key, read(source / folder / 'eval_D_reservedC.jsonl'))) != expected_keys:
            raise ValueError('Cached prediction identity/order mismatch: ' + folder)
    return hashes


def verify_environment(source):
    prior = json.loads((Path(source) / 'context_s0/runtime.json').read_text())
    current = {p: importlib.metadata.version(p) for p in prior['versions']}
    if current != prior['versions']:
        raise ValueError(f'Environment drift: expected {prior["versions"]}, found {current}')
    if sys.version.split()[0] != prior['python'].split()[0]:
        raise ValueError('Python version differs from job 700454')
    return current
