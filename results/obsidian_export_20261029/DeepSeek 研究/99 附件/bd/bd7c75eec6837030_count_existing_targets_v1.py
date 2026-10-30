"""Stage 1 only: validate cached targets and count matchable training questions.

No sampling of training subsets, model loading, inference or HPC submission.
Frozen v3 has no capital scorer, so never silently fill that gap with v2.
"""
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parent
RESULTS=OUT.parent
SOURCE=RESULTS/'exp12_sources_20261014_v2/received_700454/extracted/runs_700454'
V3=RESULTS/'exp12_rescore_v3_20261015'
EVAL=V3/'received/extracted_701713/runs_701713'
sys.path.insert(0,str(V3))
from scoring_v2_v1 import is_correct
from scorer_v3 import score, SUPPORTED


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(r):
    return r['category'],r['person']


def classify(context,direct):
    return ('both_correct' if direct else 'context_only_correct') if context else ('direct_only_correct' if direct else 'both_wrong')


def counts_summary(rows,field):
    labels=['both_correct','both_wrong','context_only_correct','direct_only_correct']
    count={k:sum(r[field]==k for r in rows) for k in labels}
    smaller=min(count['context_only_correct'],count['direct_only_correct'])
    n=count['both_correct']+count['both_wrong']+2*smaller
    count.update({'n_original':len(rows),'matched_n_max':n,'each_one_sided_class_retained':smaller,
                  'matched_correct_each':count['both_correct']+smaller,
                  'matched_accuracy_pct':100*(count['both_correct']+smaller)/n if n else None})
    return count


def main():
    model_names=['baseline']+[f'{c}_s{s}' for s in range(6) for c in ['context','direct']]
    paths=[SOURCE/'plan/train.jsonl',SOURCE/'paired_targets/context_targets.jsonl',
           SOURCE/'paired_targets/direct_targets.jsonl',SOURCE/'paired_targets/paired_diagnostics.jsonl',
           SOURCE/'paired_targets/complete.json',SOURCE/'plan/items_D_reservedC.jsonl']
    paths += [V3/n for n in ['scorer_v3.py','scoring_v2_v1.py','aliases.json','reference_catalog.json','SCORER_FREEZE.json']]
    paths += [EVAL/(n+'.jsonl') for n in model_names]
    before={str(p):sha(p) for p in paths}
    freeze=json.loads((V3/'SCORER_FREEZE.json').read_text())
    for name,digest in freeze['sha256'].items():assert sha(V3/name)==digest
    train=read(SOURCE/'plan/train.jsonl')
    targets={c:read(SOURCE/f'paired_targets/{c}_targets.jsonl') for c in ['context','direct']}
    diagnostics=read(SOURCE/'paired_targets/paired_diagnostics.jsonl')
    meta=json.loads((SOURCE/'paired_targets/complete.json').read_text())
    assert len(train)==864 and len({key(r) for r in train})==864
    for c,rows in targets.items():
        assert list(map(key,rows))==list(map(key,train))
        assert sha(SOURCE/f'paired_targets/{c}_targets.jsonl')==meta['target_sha256'][c]
    assert list(map(key,diagnostics))==list(map(key,train))
    results=[]
    for i,row in enumerate(train):
        answers={c:targets[c][i]['target'] for c in targets}
        old={c:bool(is_correct(text,row['e']['Answer'])) for c,text in answers.items()}
        for c in targets:
            assert old[c]==diagnostics[i][c+'_ok']
            assert answers[c]==diagnostics[i][c+'_target']
        new=None
        if row['category'] in SUPPORTED:
            new={c:score(text,row['e']['Answer'],row['category']) for c,text in answers.items()}
        results.append({'category':row['category'],'person':row['person'],
            'context_target':answers['context'],'direct_target':answers['direct'],
            'gold':row['e']['Answer'],'old':old,'old_class':classify(old['context'],old['direct']),
            'targets_exactly_identical':answers['context']==answers['direct'],
            'v3':new,'v3_class':classify(new['context']['correct'],new['direct']['correct']) if new else None,
            'v3_status':'scored' if new else 'unsupported_category_no_fallback'})
    all_counts=counts_summary(results,'old_class')
    cats=sorted({r['category'] for r in results})
    by_category={c:counts_summary([r for r in results if r['category']==c],'old_class') for c in cats}
    assert sum(x['matched_n_max'] for x in by_category.values())==all_counts['matched_n_max']==785
    n=all_counts['matched_n_max']
    consensus=[r for r in results if r['old_class'] in ['both_correct','both_wrong']]
    supported=[r for r in results if r['v3'] is not None]
    unsupported=collections.Counter(r['category'] for r in results if r['v3'] is None)
    pending={c:sum(r['v3'][c]['needs_review'] for r in supported) for c in targets}
    plan=read(SOURCE/'plan/items_D_reservedC.jsonl')
    item_keys=list(map(key,plan))
    union=set()
    for name in model_names:
        rows=read(EVAL/(name+'.jsonl'))
        assert list(map(key,rows))==item_keys
        union|={key(r) for r in rows if r['scores']['composed']['needs_review']}
    fixed=[r for r in plan if key(r) not in union]
    assert len(fixed)==303
    assert not ({r['person'] for r in train}&{r['person'] for r in fixed})
    mask={'n':303,'identities':[list(key(r)) for r in fixed],
          'source':'Intersection of no-needs-review composed items across the 13 already-observed v3 model outputs.',
          'reference_full_D_sha256':sha(SOURCE/'plan/items_D_reservedC.jsonl'),
          'selection_models':model_names,'selection_prediction_hashes':{name:sha(EVAL/(name+'.jsonl')) for name in model_names},
          'policy':'Freeze this exact mask for future evaluation. Never drop new items based on new models flags. Not an untouched external test set.'}
    report={'stage':'local feasibility only; no training subset selected or HPC submitted',
        'original_scorer_counts':all_counts,'original_scorer_by_category':by_category,
        'old_teacher_accuracy_pct':{c:100*sum(r['old'][c] for r in results)/864 for c in targets},
        'retention_pct':100*n/864,'people_total':len({r['person'] for r in results}),
        'people_guaranteed_in_retained_consensus':len({r['person'] for r in consensus}),
        'consensus_items':len(consensus),'disagreement_items_before':864-len(consensus),
        'disagreement_items_after_matching':2*all_counts['each_one_sided_class_retained'],
        'identical_target_pairs_retained':sum(r['targets_exactly_identical'] for r in consensus),
        'v3_coverage':{'supported_n':len(supported),'unsupported_by_category':dict(unsupported),
                       'counts_supported_only':counts_summary(supported,'v3_class'),
                       'needs_review_supported_only':pending,
                       'full_864_v3_matched_accuracy_available':False},
        'budget_if_two_epochs_batch8':{'items_each':n,'epochs':2,'batch':8,
                                      'steps_each':2*math.ceil(n/8),'exposures_each':2*n,
                                      'prior_steps_each':216,'training_seeds':[0,1,2,3,4,5],'fits_if_later_run':12},
        'fixed_primary_eval_n':303,
        'limits':['Counts use the original saved diagnostic scoring, independently reproduced, not human-verified truth.',
                  'Only 55 direct-only-correct questions, but all 675 agreement questions can remain; pool is 785, not 110.',
                  'Frozen v3 does not support capital answers. No rule modification or silent old-score fallback was performed.',
                  'Matching by category changes category sizes; both conditions would use identical item identities.',
                  'The 303-item mask was selected from already observed outcomes; freeze it before new results, do not call it an untouched test set.',
                  'Sample count alone cannot establish power or guarantee that teacher factual accuracy is matched.']}
    assert {str(p):sha(p) for p in paths}==before
    for name,obj in [('counts.json',report),('primary_D303_keys.json',mask),('verification.json',{'source_hashes_before_after_unchanged':before,'targets_match_original_diagnostics':True,'original_labels_recomputed':1728,'no_inference':True,'no_HPC_submission':True,'no_scorer_changes':True})]:
        with (OUT/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
    with (OUT/'training_target_classes.jsonl').open('x') as f:
        for row in results:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
