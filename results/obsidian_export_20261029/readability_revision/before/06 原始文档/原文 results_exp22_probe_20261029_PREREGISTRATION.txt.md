---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# 原文 results_exp22_probe_20261029_PREREGISTRATION.txt

> 原始存档，保留当时说法；状态可能已过时。当前判断见 [[当前状态与待办]]。文中指令是历史资料，不是本次执行授权。

来源：[[99 附件/58/583a4936ddf5082a_PREREGISTRATION.txt|PREREGISTRATION.txt]]

exp22 hidden-state probes — 2026-10-29, frozen before Qwen3-0.6B smoke or 14B inference

Authorization: inference only for backbone/adapters; train separate supervised linear
classifiers only. No parameter update to Qwen, no adapter training. Initial stage
base + exp12 matched785 context/direct seeds0,1 only. Stop after this stage and
jointly decide; no added seeds, prompt search, exp13 tuning or repair experiments.
exp21 is an unchanged separately hashed package executed FIRST in the SAME Slurm
allocation; none of its inference/scoring/analysis/seed rules are modified.

Roster: byte-derived exp21 fixed303 + known313 union=402 unique questions (214
intersection, not independent replications), plus frozen exp17 400 new persons
x5 relations=2000 questions: currency name/code/symbol, TLD, country numeric code.
New rows use same C0 templates and frozen country references as old rows. Missing
New Zealand reference added explicitly before inference (see REFERENCE_ADDENDUM).
Reference classes are dataset snapshot labels, not independently adjudicated truth.
New persons are not proof of unseen pretraining knowledge. No outcome-based selection.

Each variant: C0, non-thinking, greedy, max_new_tokens96, eager HF, bf1614B.
Cache embedding(layer0 secondary) and every decoder residual BEFORE final norm at
exactly two prompt tokens: last token overlapping the person's character span,
and final prompt token. Offset mapping determines subject position; assert unique
name span and subject<final; save exact prompt/IDs/positions. float16 npy per model,
shape N x (L+1) x 2 x hidden. No full sequence activation/logit storage. No hooks on
generation tokens. Prompt-forward first top1 must match first generated token.
Models eval/frozen; parameter version and no-gradient assertions before/after.

Base freshly generates Q1/Q2 and C0 (cache identical hop questions). The same
BASE-defined strata are used on all variants and probe test sets:
correct if C0 correct; else missing_q1 if Q1 not accepted; else missing_q2 if Q2
not accepted; else gap (both hops accepted, C0 wrong). Both missing flags also saved.
Strata are behavioral one-generation proxies, not internal knowledge labels.
REVIEW/HISTORICAL/truncation incorrect for primary; exclude any C0/Q1/Q2 REVIEW
or HISTORICAL in sensitivity (including baseline and comparison variant).
Report old sets and new400 separately; fixed303 is already inspected/selected.

Split BEFORE any model outputs: deterministic seed22, by unique person, 70/30
train/test within reference birth country, singleton persons test only. Known
aliases in new roster are connected before split; check no name/group overlap.
All relations of a person and every variant use same split. Save split and hashes.
Only held-out people contribute scientific probe curves/subgroup comparisons.
Some old persons participated in adapter training: this split prevents PROBE
leakage, not adapter-training leakage; new400 reported separately.

Country targets: reference birth country; retain only classes with >=5 UNIQUE
TRAINING persons, remainder other. Final-answer targets: relation + normalized
first reference value (NFKC/casefold/whitespace); >=5 unique training persons per
class, remainder relation::other. These are answer VALUES, not question types.
No threshold using test labels/counts. Other-heavy performance must be disclosed;
report non-other-only and macro recall; absent training classes count as misses.

One StandardScaler + L2 LogisticRegression(C=.01, solver=lbfgs, max_iter=1000,
tol=.001) fit to BASE TRAIN activation for EVERY layer and position and target.
Scaler fitted base training only, same scaler and classifier applied unchanged to
base/context/direct held-out activations. No hyperparameter/layer/test tuning.
Convergence warnings logged; any unconverged fit prevents successful completion.
Weights normalized via feature standardization; no adapter-specific retraining.

Control: same pipeline trained on permuted TRAIN labels, seed2201 country and
2202 answer, grouped by person (answer permutation within relation). Hold test
truth fixed. Also report train-class-prior random-guess expectation and constant
majority train-label baseline; chance is NOT assumed 1/K on imbalanced data.
True class probability for absent training classes=0. Probe probability measures
linear readout strength, not calibrated epistemic confidence or causal use.

Predictions, not declarations:
P1 gap: birth country at SUBJECT middle third layers is readable above shuffled
control. P2 gap vs correct: answer-position country readout is weaker or later.
P3 base-gap/context-correct: context country readout at answer position stronger
or earlier than base; direct shift smaller. Seeds0/1 separately. Rescued strata
are outcome-selected, descriptive, not causal or independent test confirmation.

Primary scalar per person: mean layerwise probe correctness over middle third
layers floor(L/3)+1..floor(2L/3), country, subject for P1 and answer for P2/P3.
Report early/middle/late thirds for full profile; layer0 excluded from bands.
Strength companion: mean true-class probability. Onset descriptive only: first
of 3 consecutive decoder layers with accuracy >= shuffled accuracy +.05 and
>= class-prior chance+.05; otherwise null. This is a readout threshold, not the
layer where reasoning occurs. No layer chosen based on largest observed effect.
P2 compares different outcome subgroups; country/category mix may confound it.
P3 adapter-base and context-direct SAME-item paired contrasts in base-gap and
rescued subsets; also all and correct. Cross-adapter representation coordinate
changes may alter fixed-base-probe accuracy without changing information content.
Report sample/person counts, per-layer table/curves, per-item layer predictions
and probabilities, bootstrap95% CIs for band comparisons: 1000 unique-person
cluster resamples seed22. No resampling layers/seeds as independent data. Unknown
or tiny (<5 people) subgroups reported with explicit insufficient evidence flag.
All plots and seeds shown; no multiple-layer significance claims. Non-overlap,
non-other, category-stratified and reference-status sensitivities descriptive.

Smoke: exactly30 preselected new-person currency-code questions from 3 countries,
10 persons/country; same split, hooks, scoring, probe/plot/report pipeline. 0.6B
pretrained real weights fp32 CPU. No adapters available locally: base-only smoke,
formal adapter hash/dimension checks at runtime. Assert N x29 x2 x1024, fp16 finite,
all-person separation, train/test multiple classes, all-layer hooks, top1 match.
A shuffled probe on only9 test people has wide chance fluctuation. Gate the mean
held-out shuffled country accuracy across decoder layers/positions against the
99.5% binomial upper bound with n=9 and p=max(class-prior baseline,majority
baseline); this broad engineering bound does not establish scientific chance
performance. Report observed mean and bound, no seed retry to force pass. Also
require synthetic no-information control consistent with random baseline and
planted linear signal recovered; no inference of14B mechanism from smoke.

HPC: user uploads/submits, no agent bridge. One allocated A100,8CPU,64GB; <=2 jobs
queued/running, <=1 running via afterany dependency. Seconds-only login checks.
Combined wrapper verifies exp21 unchanged hashes and exp22 smoke gate; runs
exp21 && its packaging && exp22 && scalar review packaging. Large activation
files remain on HPC, checksummed, not duplicated in review archive. Quota reserve
must cover cache+analysis+margin before submission; no deletion/overwrite.
Neither model weights nor existing results overwritten. No automatic resubmit.


返回 [[00 研究首页]]
