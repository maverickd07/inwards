"""Login-node light checks; never hash or load multi-GB weights here."""
import preflight_v1 as base
from common import verify_freeze
from check_source import check,environment,MANIFEST
original=base.check_inputs
base.MIN_FREE_GIB=4.0

def inputs(root):
 versions=original(root);verify_freeze()
 source=root/MANIFEST['relative_hpc_source'];check(source,weights=False);environment(source)
 print('Matched785 source metadata/12 adapter files present; immutable MuSiQue plan verified.')
 return versions
if __name__=='__main__':
 base.check_inputs=inputs;base.main()
