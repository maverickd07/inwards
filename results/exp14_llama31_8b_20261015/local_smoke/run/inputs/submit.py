"""Explicit exp13 afterany, user review acknowledgement, duplicate guards."""
import argparse,os,re,subprocess,time,fcntl
from common import *
from preflight import check

def dependencies(receipt,jobs):
 text=receipt.strip()
 if not re.fullmatch(r'[1-9][0-9]*(?:;[A-Za-z0-9_-]+)?',text):raise ValueError('Invalid MuSiQue receipt')
 musi=text.split(';')[0]
 if len(jobs)>=2:raise ValueError('Already two queued/running jobs; do not submit')
 if any(not re.fullmatch(r'[1-9][0-9]*',j) for j in jobs):raise ValueError('Unexpected queue IDs')
 return musi,':'.join(dict.fromkeys([musi]+jobs))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--configuration-reviewed',action='store_true');a=ap.parse_args()
 if not a.configuration_reviewed:raise SystemExit('Review CONFIG.json, PROTOCOL.txt and the 2–4h estimate first. Then pass --configuration-reviewed.')
 root=Path.home()/'inward';os.chdir(root)
 with (HERE/'submission.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for n in ['submitted_job.txt','submission.json','quota_ticket.json']:
   if (HERE/n).exists():raise FileExistsError('Prior submission/ticket exists; no automatic retry: '+n)
  jobs=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i'],text=True).split()
  receipt=root/CONFIG['dependency_receipt'];musi,deps=dependencies(receipt.read_text(),jobs)
  # Verify the receipt points to this user's actual exp13, even if completed.
  accounting=subprocess.check_output(['sacct','-n','-P','-X','-j',musi,'--format=JobIDRaw,User,JobName%80'],text=True)
  import getpass
  rows=[line.split('|') for line in accounting.splitlines() if line.strip()]
  if not any(r[0]==musi and r[1]==getpass.getuser() and r[2]=='inward-exp13-musi' for r in rows):
   raise ValueError('MuSiQue dependency not verified by sacct; no submission')
  ticket=check(root,quota_file=HERE/'quota_login.txt');js(HERE/'quota_ticket.json',ticket)
  request={'submitted_unix':time.time(),'configuration_reviewed':True,'exp13_job':musi,'dependency':'afterany:'+deps,'exp13_receipt_sha256':sha(receipt)}
  js(HERE/'submission.json',request)
  raw=subprocess.check_output(['sbatch','--parsable','--dependency=afterany:'+deps,str(HERE/'run_exp14.sh')],text=True).strip()
  if not re.fullmatch(r'[1-9][0-9]*(?:;[A-Za-z0-9_-]+)?',raw):raise ValueError('Unknown sbatch outcome: inspect squeue, do not retry')
  with (HERE/'submitted_job.txt').open('x') as f:f.write(raw+'\n')
  print('Submitted exp14:',raw,'dependency:',request['dependency'],flush=True)
if __name__=='__main__':main()
