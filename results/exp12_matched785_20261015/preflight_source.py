"""Reuse original quota/model checks and verify the precise source run, lightly."""
import json
from pathlib import Path
import preflight_v1 as original
from reuse import HERE, verify_source, verify_environment

check_original = original.check_inputs


def check_inputs(root):
    versions = check_original(root)
    manifest = json.loads((HERE / 'source_700454_manifest.json').read_text())
    source = root / manifest['relative_hpc_source']
    hashes = verify_source(source)
    verify_environment(source)
    print(f'Exact job 700454 source verified: {len(hashes)} files; environment matches.')
    return versions


if __name__ == '__main__':
    original.check_inputs = check_inputs
    original.main()
