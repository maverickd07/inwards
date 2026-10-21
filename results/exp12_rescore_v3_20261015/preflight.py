"""Light file-integrity and quota check; never load a model or score predictions."""
import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path
from rescore import HERE, sha, verify, js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path.home()/'inward')
    ap.add_argument('--quota-file',type=Path)
    ap.add_argument('--ticket',type=Path)
    a = ap.parse_args()
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name = line.split('  ',1)
        if Path(name).name != name or sha(HERE/name) != digest:
            raise ValueError('Package mismatch: '+name)
    verify(a.root)
    if a.quota_file:
        age = time.time()-a.quota_file.stat().st_mtime
        if not -60 <= age <= 300:
            raise ValueError('Refresh lquota immediately before submission')
        quota = {'checked_unix':time.time(),'quota_text':a.quota_file.read_text()}
    elif a.ticket:
        quota = json.loads(a.ticket.read_text())
        if not -60 <= time.time()-quota['checked_unix'] <= 48*3600:
            raise ValueError('Quota ticket expired; inspect before submitting again')
    else:
        raise ValueError('Quota capture or ticket required')
    m = re.search(r'USED\s*=\s*([\d.]+)\s*([KMGT])B\s+LIMIT\s*=\s*([\d.]+)\s*([KMGT])B',quota['quota_text'],re.I)
    if not m:
        raise ValueError('Unrecognized lquota output')
    factors = {'K':1/1024**2,'M':1/1024,'G':1,'T':1024}
    free = float(m[3])*factors[m[4].upper()] - float(m[1])*factors[m[2].upper()]
    if not math.isfinite(free) or free < .25 or shutil.disk_usage(a.root).free < .25*1024**3:
        raise ValueError('Require at least 0.25GB headroom')
    if a.quota_file:
        js(HERE/'quota_ticket.json',quota)
    print('Preflight passed: frozen v3, 14 matching input files, quota headroom',free,'GB. No scoring performed.')


if __name__ == '__main__':
    main()
