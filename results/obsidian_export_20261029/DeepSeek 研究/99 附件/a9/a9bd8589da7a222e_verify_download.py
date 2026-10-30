"""Check downloaded HPC rescore against frozen inputs and validated local labels."""
import hashlib
import json
import math
import statistics
from pathlib import Path
from scipy.stats import t

OUT=Path(__file__).resolve().parent
PACKAGE=OUT.parent
RUN=PACKAGE/'received/extracted_701713/runs_701713'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert read(RUN/'scorer_freeze.json')==read(PACKAGE/'SCORER_FREEZE.json')
    for name,digest in read(PACKAGE/'SCORER_FREEZE.json')['sha256'].items():
        assert sha(PACKAGE/name)==digest
    code=read(RUN/'code_hashes.json')
    for line in (PACKAGE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert code[name]==digest==sha(PACKAGE/name),name
    receipt=read(RUN/'input_receipt.json')
    manifest=read(PACKAGE/'input_manifest.json')
    assert set(receipt)==set(manifest)
    for k,v in receipt.items():
        assert v['sha256']==manifest[k]['sha256']
    assert sha(RUN/'items_D_reservedC.jsonl')==manifest['plan']['sha256']
    models=['baseline']+[f'{c}_s{s}' for s in range(6) for c in ['context','direct']]
    files=[m+'.jsonl' for m in models]+['items_D_reservedC.jsonl','review_queue.jsonl']
    for name in files:
        assert sha(RUN/name)==sha(PACKAGE/'local_validation'/name),name
    report=read(RUN/'report.json')
    scalar_diffs=[]
    def equivalent(a,b,path=''):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equivalent(a[k],b[k],path+'/'+k)
        elif isinstance(a,list):
            assert len(a)==len(b)
            for i,(x,y) in enumerate(zip(a,b)):equivalent(x,y,path+'/'+str(i))
        elif isinstance(a,float):
            assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(path,a,b)
            if a!=b:scalar_diffs.append({'path':path,'absolute_difference':abs(a-b)})
        else:
            assert a==b,(path,a,b)
    equivalent(report,read(PACKAGE/'local_validation/report.json'))
    for metrics in report['groups'].values():
        for scores in metrics.values():
            for stats in scores.values():
                for mom in list(stats['accuracy_pct'].values())+list(stats['gain_over_base_pp'].values())+[stats['context_minus_direct_pp']]:
                    v=mom['values'];mean=statistics.mean(v);sd=statistics.stdev(v)
                    half=float(t.ppf(.975,len(v)-1))*sd/math.sqrt(len(v))
                    assert math.isclose(mean,mom['mean'],abs_tol=1e-9)
                    assert math.isclose(sd,mom['sd'],abs_tol=1e-9)
                    assert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip([mean-half,mean+half],mom['seed_t95']))
    complete=read(RUN/'complete.json')
    assert complete['models']==13 and complete['metric_labels']==19500
    assert complete['new_model_generations']==complete['new_training_runs']==0
    assert 'Traceback (most recent call last)' not in (RUN.parent/'rescore-701713.out').read_text()
    verification={'job_id':'701713','complete':complete,'archive_sha256_verified':True,
        'frozen_rules_match':True,'package_hashes_match':True,'all_input_hashes_match':True,
        'all_13_model_outputs_byte_identical_to_validated_local_results':True,
        'review_queue_byte_identical':True,'reports_numerically_equivalent':True,
        'floating_point_last_digit_differences':scalar_diffs,
        'all_intervals_independently_checked_with_scipy':True,
        'limitations':['No final sacct export; pipeline completion marker and log checked.',
                      'No manual factual adjudication or dataset reference relabeling.',
                      'Conservative v3 scores count flagged unsupported answers as unaccepted; they remain in the 500-item denominator.']}
    with (OUT/'verification.json').open('x') as f:json.dump(verification,f,indent=2)
    print('Passed. All 19500 labels and raw predictions match local validation exactly.')
    print('Largest floating-point summary difference:',max(x['absolute_difference'] for x in scalar_diffs))


if __name__=='__main__':
    main()
