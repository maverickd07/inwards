"""Only the seed and evaluation split differ from the frozen exp12 execution."""
import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import exp12_v1 as original
from reuse import NEW_SEEDS, CONDITIONS, js, verify_environment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--train-seed', type=int, choices=NEW_SEEDS, required=True)
    ap.add_argument('--condition', choices=CONDITIONS, required=True)
    ap.add_argument('--smoke-test', action='store_true')
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.smoke_test:
        original.torch.set_num_threads(1)
    else:
        verify_environment(a.source)
        prior = json.loads((a.source / 'context_s0/runtime.json').read_text())
        if original.torch.version.cuda != prior['cuda']:
            raise ValueError('CUDA runtime differs from job 700454')
    a.plan = a.source / 'plan'
    a.targets = a.source / f'paired_targets/{a.condition}_targets.jsonl'
    a.adapter, a.epochs, a.batch, a.round, a.gen_batch = None, 2, 8, 1, 8
    # Do not modify the optimizer, prompts, decoding, target processing or scorer.
    original.SPLITS = ['D_reservedC']
    started = time.time()
    original.train(a)
    js(a.out / 'timing.json', {'stage': 'train', 'seconds': time.time() - started})
    js(a.out / 'runtime.json', {'python': sys.version, 'platform': platform.platform(),
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft', 'numpy']},
        'cuda': original.torch.version.cuda,
        'gpu': original.torch.cuda.get_device_name(0) if original.torch.cuda.is_available() else None,
        'smoke_test': a.smoke_test})


if __name__ == '__main__':
    main()
