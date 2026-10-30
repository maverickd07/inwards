"""Light read-only checks; no ML imports or scoring on login node."""
import json
import preflight_v1 as original
from reuse import HERE,verify_source,verify_environment
from protocol import verify_freeze
check_original=original.check_inputs
original.MIN_FREE_GIB=12.0

def check_inputs(root):
 versions=check_original(root)
 source=root/'results/exp12_sources_20261014_v2/runs_700454'
 verify_source(source);verify_environment(source);verify_freeze()
 print('Matched785 allocation/design immutable; 12 fits; 198 steps each; cached baseline verified.')
 return versions
if __name__=='__main__':
 original.check_inputs=check_inputs
 original.main()
