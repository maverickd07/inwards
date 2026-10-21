"""Read-only exp12 v3 flag sensitivity and reconciliation; no scorer imports."""
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path
from scipy.stats import t

OUT=Path(__file__).resolve().parent
RESULTS=OUT.parent
PACKAGE=RESULTS/'exp12_rescore_v3_20261015'
RUN=PACKAGE/'received/extracted_701713/runs_701713'
AUDIT=RESULTS/'exp12_sources_20261014_v2/analysis_700454_local'
NAMES=['baseline']+[f'{c}_s{s}' for s in range(6) for c in ['context','direct']]
METRICS=['composed','q1','q2']


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def key(row):
    return row['category'],row['person']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(values):
    mean,sd=statistics.mean(values),statistics.stdev(values)
    half=float(t.ppf(.975,len(values)-1))*sd/math.sqrt(len(values))
    return {'values':values,'mean_pp':mean,'sd':sd,'df':len(values)-1,'t95':[mean-half,mean+half]}


def main():
    inputs=[RUN/(n+'.jsonl') for n in NAMES]+[RUN/'report.json',AUDIT/'primary_pair_review.jsonl',AUDIT/'primary_audit_summary.json']
    inputs += [PACKAGE/n for n in ['scorer_v3.py','aliases.json','reference_catalog.json','SCORER_FREEZE.json','input_manifest.json']]
    before={str(p):sha(p) for p in inputs}
    data={n:read(RUN/(n+'.jsonl')) for n in NAMES}
    keys=list(map(key,data['baseline']))
    assert len(keys)==len(set(keys))==500
    for rows in data.values():
        assert list(map(key,rows))==keys
        for row in rows:
            for m in METRICS:
                label=row['scores'][m]
                assert not (label['needs_review'] and label['v3'])
    cached=json.loads((RUN/'report.json').read_text())
    metrics={}
    for metric in METRICS:
        flags={n:{i for i,r in enumerate(rows) if r['scores'][metric]['needs_review']} for n,rows in data.items()}
        good={n:{i for i,r in enumerate(rows) if r['scores'][metric]['v3']} for n,rows in data.items()}
        counts={n:{'needs_review':len(flags[n]),'review_rate_pct':len(flags[n])/5,
                   'accepted':len(good[n]),'review_reasons':dict(collections.Counter(data[n][i]['scores'][metric]['reason'] for i in flags[n]))}
                for n in NAMES}
        assert all(counts[n]['needs_review']==cached['models'][n][metric]['needs_review'] for n in NAMES)
        overlaps=[]
        scenarios={name:[] for name in ['all_wrong','all_correct','exclude_pair_union',
                   'exclude_own_flags_different_denominators','exclude_union_all_12_fits','exclude_union_all_13_models']}
        all12=set.union(*(flags[n] for n in NAMES if n!='baseline'))
        all13=all12|flags['baseline']
        per_seed=[]
        for seed in range(6):
            c,d=f'context_s{seed}',f'direct_s{seed}'
            union=flags[c]|flags[d];both=flags[c]&flags[d]
            overlaps.append({'seed':seed,'context_n':len(flags[c]),'direct_n':len(flags[d]),
                'intersection_n':len(both),'union_n':len(union),
                'jaccard_pct':100*len(both)/len(union) if union else None,
                'intersection_over_context_pct':100*len(both)/len(flags[c]) if flags[c] else None,
                'intersection_over_direct_pct':100*len(both)/len(flags[d]) if flags[d] else None})
            values={'all_wrong':100*(len(good[c])-len(good[d]))/500,
                'all_correct':100*(len(good[c]|flags[c])-len(good[d]|flags[d]))/500,
                'exclude_own_flags_different_denominators':100*(len(good[c])/(500-len(flags[c]))-len(good[d])/(500-len(flags[d])))}
            for name,remove in [('exclude_pair_union',union),('exclude_union_all_12_fits',all12),('exclude_union_all_13_models',all13)]:
                keep=set(range(500))-remove
                values[name]=100*(len(good[c]&keep)-len(good[d]&keep))/len(keep)
            for name,value in values.items():scenarios[name].append(value)
            per_seed.append({'seed':seed,'pair_common_retained_n':500-len(union),
                'own_context_retained_n':500-len(flags[c]),'own_direct_retained_n':500-len(flags[d]),
                'differences_pp':values})
        metrics[metric]={'counts_13_models':counts,'paired_flag_overlaps':overlaps,
            'six_seed_sensitivity':{n:moments(v) for n,v in scenarios.items()},'per_seed':per_seed,
            'global_shared_retained_n_12':500-len(all12),'global_shared_retained_n_13':500-len(all13)}
        expected=cached['groups']['all_012345'][metric]['new']['context_minus_direct_pp']
        assert math.isclose(scenarios['all_wrong'][0],expected['values'][0],abs_tol=1e-9)
        assert math.isclose(statistics.mean(scenarios['all_wrong']),expected['mean'],abs_tol=1e-9)

    # Reconstruct yesterday's fixed item set directly from its saved pair labels.
    audit=read(AUDIT/'primary_pair_review.jsonl')
    remove_types={'representation_or_alias','historical_currency_or_time_convention','scoring_or_answer_ambiguity'}
    excluded_keys={key(r) for r in audit if r['linguistic_change_type'] in remove_types}
    excluded={i for i,k in enumerate(keys) if k in excluded_keys}
    kept=set(range(500))-excluded
    assert len(excluded)==28 and len(kept)==472 and {keys[i] for i in excluded}==excluded_keys
    matrix={}
    for setname,indices in [('full_500',set(range(500))),('yesterday_retained_472',kept),('yesterday_excluded_28',excluded)]:
        matrix[setname]={}
        for group,seeds in [('original_012',range(3)),('added_345',range(3,6)),('all_012345',range(6))]:
            matrix[setname][group]={}
            for scoring in ['old','v3']:
                values=[]
                for seed in seeds:
                    c,d=[data[f'{n}_s{seed}'] for n in ['context','direct']]
                    values.append(100*sum(int(c[i]['scores']['composed'][scoring])-int(d[i]['scores']['composed'][scoring]) for i in indices)/len(indices))
                matrix[setname][group][scoring]=moments(values)
    old_audit=json.loads((AUDIT/'primary_audit_summary.json').read_text())['posthoc_item_exclusion_sensitivity']['exclude_representation_temporal_ambiguous_items']['context_minus_direct_pp']
    measured=matrix['yesterday_retained_472']['original_012']['old']
    assert math.isclose(measured['mean_pp'],old_audit['mean'],abs_tol=1e-10)
    assert all(math.isclose(x,y,abs_tol=1e-9) for x,y in zip(measured['t95'],old_audit['seed_t95']))
    bridge=[{'stage':'yesterday: old scorer, seeds012, 472 items','value_pp':measured['mean_pp']},
            {'stage':'restore 28 excluded items, same old scorer and seeds012','value_pp':matrix['full_500']['original_012']['old']['mean_pp']},
            {'stage':'apply v3, same full500 and seeds012','value_pp':matrix['full_500']['original_012']['v3']['mean_pp']},
            {'stage':'add seeds345, same v3 and full500','value_pp':matrix['full_500']['all_012345']['v3']['mean_pp']}]
    for i in range(1,len(bridge)):bridge[i]['increment_pp']=bridge[i]['value_pp']-bridge[i-1]['value_pp']
    # Locate numeric score changes without judging their factual correctness.
    changes={}
    for group,seeds in [('original_012',range(3)),('all_012345',range(6))]:
        changes[group]={}
        for c in ['context','direct']:
            recs=[];by_reason=collections.Counter();by_category=collections.Counter()
            for seed in seeds:
                rows=data[f'{c}_s{seed}']
                down=up=0
                for row in rows:
                    label=row['scores']['composed'];delta=int(label['v3'])-int(label['old'])
                    down += delta==-1;up += delta==1
                    by_reason[label['reason']] += delta
                    by_category[row['category']] += delta
                recs.append({'seed':seed,'old_true_new_false':down,'old_false_new_true':up,'net_pp':(up-down)/5})
            changes[group][c]={'per_seed':recs,'mean_old_true_new_false':statistics.mean(r['old_true_new_false'] for r in recs),
                'mean_old_false_new_true':statistics.mean(r['old_false_new_true'] for r in recs),
                'mean_net_pp':statistics.mean(r['net_pp'] for r in recs),
                'net_pp_by_reason':{k:v/len(recs)/5 for k,v in by_reason.items()},
                'net_pp_by_category':{k:v/len(recs)/5 for k,v in by_category.items()}}
    output={'scope':'Primary D_reservedC composed answers; q1/q2 appended separately.',
        'overlap_definition':'intersection / union (Jaccard), with both directional overlaps also recorded.',
        'primary_exclusion_definition':'Within each seed remove the union of context/direct flags from BOTH conditions; same retained questions and denominator within each pair.',
        't95_definition':'mean of six seed-level differences +/- t(0.975,5)*sample_sd/sqrt(6); no pooling as independent questions.',
        'metrics':metrics,'yesterday_excluded_keys':[list(keys[i]) for i in sorted(excluded)],
        'reconciliation_matrix':matrix,'bridge':bridge,'original_to_v3_changes':changes,
        'limitations':['All-correct/all-wrong scenarios are common assumptions, not sharp bounds on the treatment contrast.',
            'Exclusion is selected by model-dependent output flags, so it changes the assessed population.',
            'Flagged unmatched answers are not independently adjudicated factual errors.',
            'Yesterday used post-hoc item exclusion based on model-assisted flip types; today uses a fixed post-hoc scorer on all items.',
            'Dataset birthplace/time reference conventions remain unchanged.']}
    after={str(p):sha(p) for p in inputs};assert before==after
    with (OUT/'sensitivity.json').open('x') as f:json.dump(output,f,ensure_ascii=False,indent=2)
    with (OUT/'verification.json').open('x') as f:json.dump({'read_only_inputs_unchanged':True,'input_sha256':before,'inference_rerun':False,'scorer_rules_modified':False,'old_099_reproduced':True,'v3_490_reproduced':True},f,indent=2)
    print(json.dumps({'primary':metrics['composed'],'bridge':bridge,'matrix':matrix,'changes':changes},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
