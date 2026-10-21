"""Model-assisted review of all 65 unique scorer-flip pairs in primary D.

Labels describe answer changes, NOT independently verified factual correctness.
Reviewer saw aggregate outcomes before reviewing randomized X/Y pairs. No human
inter-rater agreement; retain the original scoring and mark this post hoc.
"""
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
PACKAGE = OUT.parent
ROOT = PACKAGE / 'received_700454/extracted/runs_700454'
sys.path.insert(0, str(PACKAGE))
import analyze as stats

REPRESENTATION = {0,3,17,19,24,25,29,30,32,36,43,44,46,48,49,51,52,58,60}
TEMPORAL = {2,6,7,31,37,63}
CODE_SYSTEM = {1,4,8,12,23,27,53}
UNRESOLVED = {10,11,40,45,59}
CONTENT = {5,9,13,14,15,16,18,20,21,22,26,28,33,34,35,38,39,41,42,47,50,54,55,56,57,61,62,64}
EXTRA = {
 0: 'DIN vs RSD resembles dinar abbreviation normalization. Separately, the reference birthplace Serbia is inconsistent with Tesla museum evidence.',
 3: 'Tugrik name vs the currency symbol; answer-format compliance differs while the named currency is retained.',
 8: '370 is the Lithuania telephone code vs numeric-country code 440; reference birthplace Lithuania is separately inconsistent with NBA birth record Portland.',
 10: 'Y states .com while also including the expected .uy in an explanation. Substring scoring accepts it; unambiguous final-answer correctness is unresolved.',
 11: 'X contains both .com and the expected .uy; do not count substring acceptance as clear factual repair.',
 17: 'UAE Dirham and United Arab Emirates Dirham refer to the same currency; reference lacks the common abbreviation.',
 19: 'CFA vs ISO XAF: greater code specificity rather than clear new currency knowledge; question only asks abbreviation.',
 24: 'Currency code HUF vs symbol Ft; output type changes but currency identity is retained.',
 25: 'Currency code ARS vs dollar symbol; output type changes, not clearly currency identity.',
 29: 'Yen vs Japanese yen: added qualifier to the currency name.',
 30: 'Dkr vs DKK: symbol vs code; same Danish currency interpretation, subject to task-specific answer format.',
 35: 'Country-specific .sn vs generic .com. Akon birthplace reference Senegal is a separate unverified concern, not counted as independently validated here.',
 36: 'Both answers explicitly say Taka; one adds the symbol.',
 39: 'Albanian lekë vs Leu: distinct currency terms; reference substring accepts lek in lekë.',
 40: 'Y contains contradictory/multiple TLD tokens including expected .uy; no unambiguous semantic gain adjudicated.',
 43: 'Ruble vs Russian ruble: added qualifier; possible broad alias rather than knowledge change.',
 44: 'Both answers name manat; one adds the symbol.',
 45: 'Braille pattern dots-123456 is accepted because normalized reference br is a substring of braille. This is a demonstrable scoring false positive. Ethiopian-script answer correctness not adjudicated.',
 46: 'Both answers explicitly identify Cuban Peso/CUP; one adds the dollar symbol.',
 48: 'EUR vs euro symbol: currency identity retained, answer rendering differs.',
 49: 'LD vs LYD: dinar abbreviation vs ISO code; no independent factual correction established.',
 51: 'EGP vs pound symbol: code/symbol distinction; not proof of newly inferred country.',
 52: 'Pound sterling and British pound are names of the same currency.',
 58: 'Tugrik vs T followed by the tugrik symbol; symbol-format difference.',
 59: 'Reference B/. normalizes to single b and accepts B$; ambiguity of symbol/currency usage and overly permissive substring scoring remain.',
 60: 'Dirham vs United Arab Emirates Dirham: added country specificity; generic name is under-specified but not necessarily a distinct currency.'
}
FACT_SOURCES = [
 {'person':'Nikola Tesla','dataset_birth_country':'Serbia',
  'verified_statement':'Nikola Tesla Museum identifies his birth village as Smiljan in the then Austrian Empire; Tesla Science Center identifies the site as modern-day Croatia.',
  'urls':['https://tesla-museum.org/en/extraordinary-world/birthplace/',
          'https://teslasciencecenter.org/staging/pivotalmoments/birth/'],
  'status':'reference birthplace contradicted; no relabeling of headline benchmark'},
 {'person':'Domantas Sabonis','dataset_birth_country':'Lithuania',
  'verified_statement':'Sacramento Kings official NBA media guide lists Portland, Oregon as birthplace.',
  'urls':['https://cdn.nba.com/teams/uploads/sites/1610612758/2022/10/22-23SacramentoKingsMediaGuide.pdf'],
  'status':'reference birthplace contradicted; no relabeling of headline benchmark'}]


def main():
    mapping = list(csv.DictReader((ROOT / 'analysis/flip_mapping.csv').open()))
    ids = {r['audit_id'] for r in mapping if r['split'] == 'D_reservedC'}
    blinded = [r for r in csv.DictReader((ROOT / 'analysis/blinded_review.csv').open()) if r['audit_id'] in ids]
    assert len(blinded) == 65
    assert blinded[0]['audit_id'] == '1063cadc814944681837'
    assert blinded[45]['audit_id'] == '835c4df5381e83d7006a'
    groups = [REPRESENTATION, TEMPORAL, CODE_SYSTEM, UNRESOLVED, CONTENT]
    assert set.union(*groups) == set(range(65)) and sum(map(len, groups)) == 65
    labels = {}
    for index, row in enumerate(blinded):
        if index in REPRESENTATION:
            cls = 'representation_or_alias'
            note = 'Likely same currency with name/code/symbol/abbreviation changes; task-format compliance can still differ.'
        elif index in TEMPORAL:
            cls = 'historical_currency_or_time_convention'
            note = 'Historical vs modern currency or historical geography. Timing convention not specified; not treated as clean content repair.'
        elif index in CODE_SYSTEM:
            cls = 'phone_vs_numeric_code_system'
            note = 'Telephone-code vs numeric-country-code answer. Attribute selection differs; not merely typography and not proof of new country knowledge.'
        elif index in UNRESOLVED:
            cls = 'scoring_or_answer_ambiguity'
            note = 'Scoring acceptance does not establish an unambiguous correct answer.'
        else:
            cls = 'different_country_currency_or_domain_content'
            note = 'Answer content changes beyond an apparent alias. Factual correctness and causal knowledge-use mechanism not independently adjudicated.'
        labels[row['audit_id']] = {**row, 'linguistic_change_type': cls,
             'review_note': EXTRA.get(index, note), 'reviewer': 'Codex model-assisted',
             'factual_correctness_status': 'not independently adjudicated',
             'reference_birthplace_conflict_verified': row['person'] in ['Nikola Tesla','Domantas Sabonis']}
    with (OUT / 'primary_pair_review.jsonl').open('x') as f:
        for row in labels.values():
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    types = sorted({r['linguistic_change_type'] for r in labels.values()})
    contributions = []
    for condition in ['context','direct']:
        for typ in types:
            counts = []
            for seed in range(3):
                selected = [r for r in mapping if r['split']=='D_reservedC' and r['condition']==condition
                            and int(r['seed'])==seed and labels[r['audit_id']]['linguistic_change_type']==typ]
                repairs = sum(r['direction']=='repair' for r in selected)
                damages = sum(r['direction']=='damage' for r in selected)
                counts.append({'seed':seed,'repairs':repairs,'damages':damages,'net_pp':(repairs-damages)/5})
            contributions.append({'condition':condition,'type':typ,'by_seed':counts,
                                  'mean_net_scorer_pp':statistics.mean(x['net_pp'] for x in counts)})
    net_difference = {typ:next(c['mean_net_scorer_pp'] for c in contributions if c['condition']=='context' and c['type']==typ)
                          -next(c['mean_net_scorer_pp'] for c in contributions if c['condition']=='direct' and c['type']==typ)
                      for typ in types}
    # Fixed-denominator decomposition: contributions sum to the original 2.8667 pp.
    assert abs(sum(net_difference.values()) - 2.8666666666666667) < 1e-10
    items = stats.read(ROOT / 'plan/items_D_reservedC.jsonl')
    baseline = stats.read(ROOT / 'baseline/eval_D_reservedC.jsonl')
    predictions = {(c,k):stats.read(ROOT/f'{c}_s{k}/eval_D_reservedC.jsonl') for c in ['context','direct'] for k in range(3)}
    flagged_items = {stats.key(r) for r in items if r['person'] in ['Nikola Tesla','Domantas Sabonis']}
    flags = [r['person'] for r in items if stats.key(r) in flagged_items]
    scenarios = {}
    # Exclusions are POST HOC sensitivity analyses, not replacement primary tests.
    excluded_types = {
      'exclude_representation_items': {'representation_or_alias'},
      'exclude_representation_temporal_ambiguous_items': {'representation_or_alias','historical_currency_or_time_convention','scoring_or_answer_ambiguity'},
      'exclude_only_verified_birthplace_conflicts': set(),
    }
    for name, omit in excluded_types.items():
        excluded = { (r['category'],r['person']) for r in labels.values() if r['linguistic_change_type'] in omit }
        if name == 'exclude_only_verified_birthplace_conflicts':
            excluded = flagged_items
        idx = [i for i,r in enumerate(items) if stats.key(r) not in excluded]
        gains = {c:[] for c in ['context','direct']}
        paired_gains = []
        for seed in range(3):
            b = [baseline[i]['composed_ok'] for i in idx]
            v = {c:[predictions[c,seed][i]['composed_ok'] for i in idx] for c in ['context','direct']}
            for c in v:gains[c].append(stats.paired(v[c],b)['gain_pp'])
            paired_gains.append(stats.paired(v['context'],v['direct'])['gain_pp'])
        scenarios[name] = {'n_kept':len(idx),'condition_gain_pp':{c:stats.moments(v) for c,v in gains.items()},
                           'context_minus_direct_pp':stats.moments(paired_gains)}
    result = {'scope':'all 65 primary D scorer-flip pairs; 149 flip records across six fits',
       'review_status':'model-assisted answer-change audit; independent human semantic labels pending',
       'blinding':'X/Y identities randomized; aggregate group outcomes already known; not fully blinded independent adjudication',
       'type_counts':dict(collections.Counter(r['linguistic_change_type'] for r in labels.values())),
       'net_scorer_contributions':contributions,'context_minus_direct_contribution_pp':net_difference,
       'posthoc_item_exclusion_sensitivity':scenarios,'verified_reference_sources':FACT_SOURCES,
       'verified_reference_conflict_D_items':len(flags),
       'limits':['Types describe semantic changes in answer strings, not independently verified factual repairs.',
         'Decomposition uses signed scorer repairs and damages with fixed denominator 500; no cross-seed pooling as independent questions.',
         'Item exclusions were chosen after seeing flips; estimates are descriptive sensitivity only, not new confirmatory p-values.',
         'Non-flipping errors and A/B/C answer pairs have not received the same per-pair review.',
         'Different content can be more correct, less correct, or equally wrong; mechanisms remain unidentified.']}
    stats.js(OUT/'primary_audit_summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
