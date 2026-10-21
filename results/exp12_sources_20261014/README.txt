exp12_sources_20261014

Read PROTOCOL.txt for the fixed scientific design, HPC_COMMANDS.txt for execution.
User approved this bounded post-exp11 stage on 2026-10-14.

Production: two target sources x three seeds = six fresh LoRA fits, 2 epochs each.
Same 864 training questions; matching paired initialization and optimizer budgets.
Fresh baseline; historical A/B/C plus prespecified D_reservedC (500 each).
No automatic continuation to exp13. No gold in targets or sample selection.
Perplexity is a paired likelihood diagnostic, not another training condition.

Code:
exp12.py                plan / paired generation / baseline / training stages
engine_exp11_v1.py       frozen optimizer/allocation implementation from exp11
exp1_gap_v2_v1.py        frozen prompt and legacy scoring implementation
scoring_v2_v1.py         signed numeric scorer
run_pipeline.py         fresh-process serial orchestration of the six fits
analyze.py              paired tests, seed/cluster intervals and flip audit export
preflight.py            package/data/model shard/installed metadata/quota checks
submit.sh               queue cap and serial dependency, duplicate-submit guard
run_exp12.sh            single-GPU SLURM job, no computation on login node
test_integrity.py       local tiny-model scoring and failure-safety tests
input_hashes.json       original data/items hashes fixed during local preparation
historical_C_exclusion.json  prior C identities and source hash inventory
reference_exp8/         fixed historical A/B/C identity and baseline references
LOCAL_VERIFICATION.json tested local scope; not evidence of 14B execution

Results under runs_JOBID:
plan/                   fixed train and four evaluation manifests
paired_targets/         context/direct target files without gold;
                        paired diagnostic correctness and four NLL evaluations
baseline/               fresh composed/hop1/hop2 predictions on every split
context_s0..s2/         adapter, examples, losses, predictions, config and timing
direct_s0..s2/          same artifacts for the no-context control
analysis/report.txt     headline outcomes; all seeds and uncertainty in report.json
analysis/metrics.csv    all split/condition/metric means and seed intervals
analysis/blinded_review.csv  randomized X/Y answers, human semantic fields blank
analysis/flip_mapping.csv    model/direction/scorer mapping; keep from first reviewer
complete.json           computation completion, manual audit still marked pending

Audit instructions:
Review blinded_review.csv without flip_mapping.csv. For each answer independently,
judge semantic correctness; flag ambiguity or overly narrow reference aliases.
Use change_class values content_change, format_or_alias, scoring_issue, ambiguous,
unclear. Do not force uncertain answers into content improvement. Record confidence
and evidence/notes. Join labels to the mapping only after review; count both repairs
and damages, keep each seed separate, report unresolved pairs and conservative
sensitivity bounds. Never interpret string changes alone as knowledge improvement.

Historical audit prepared locally from exp11 (not uploaded): 4887 scorer flips,
3522 repair and 1365 damage records across 21 fits/3 splits; 522 distinct item-answer
pairs after deduplication. These are repeated experimental records, not independent
test samples and not a completed semantic classification.

Local small-model checks use a RANDOM tiny Qwen3. Zero accuracy is expected and is
not research evidence. Production must run with existing HPC inward environment;
do not upgrade it merely to match local validation versions. Runtime versions and
GPU details are logged per stage. All original experiment files remain unchanged.
