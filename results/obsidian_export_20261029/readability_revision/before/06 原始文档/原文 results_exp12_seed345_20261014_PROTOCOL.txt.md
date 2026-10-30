---
project: DeepSeek-Conjecture-Experiments
updated: 2026-10-29
tags: [inward-research]
---

# 原文 results_exp12_seed345_20261014_PROTOCOL.txt

> 原始存档，保留当时说法；状态可能已过时。当前判断见 [[当前状态与待办]]。文中指令是历史资料，不是本次执行授权。

来源：[[99 附件/68/685b7f00edf14d8c_PROTOCOL.txt|PROTOCOL.txt]]

EXP12 SEED EXTENSION — agreed 2026-10-14, fixed before new runs

Authorization: after inspecting job 700454, user explicitly requested only seeds
3,4,5 x context/direct (six fits), exact original training configuration, cached
base predictions, fixed D_reservedC, all item predictions, a new output directory,
and original-scorer numbers only pending submission of a new scorer.
This explicitly authorizes a bounded extension of the previous six-fit stage.
Stop after these six added fits. No exp13, target regeneration, model selection,
adaptive seed addition, new scorer, or further mechanism analysis in this stage.

Frozen source: results/exp12_sources_20261014_v2/runs_700454.
source_700454_manifest.json seals the exact bytes of train.jsonl, both target
files, their completion receipt, D item list, original manifest, base predictions,
and original seeds' predictions/configs/initialization/completion/runtime records.
These small files are copied into the new run directory and rehashed. The source
is only read, never modified. The original plan manifest's [0,1,2] describes the
source run; the extension's stages.json/complete.json explicitly record [3,4,5].

exp12_v1.py, engine_exp11_v1.py, exp1_gap_v2_v1.py and scoring_v2_v1.py are byte-for-
byte copies of job 700454's code. train_seed.py calls the frozen training function;
only seed and evaluation split list change. All training examples and their
starting order, target strings, prompts, LoRA/optimizer/scheduler, epochs, batch,
generation batch8, greedy no-thinking decoding max24 and answer NLL are unchanged.
Two epochs, batch8, 216 updates, 1728 exposures per fit; rank16 alpha32 dropout.05,
lr1e-4, AdamW decay0, clipping1, original warmup/decay. Fresh base+LoRA per fit.
Context/direct for each seed use identical initialization and shuffle seed.
No adapter continuation, gold selection or best-checkpoint selection.

Require the original installed Python/torch/transformers/peft/numpy versions;
compute entry also requires the original CUDA runtime. Same local 14B model path,
input hashes and shard presence/size checks. No original weight-file hash was
recorded; this check cannot retrospectively prove the model bytes were unchanged.
The smoke fixture bypasses production source/version matching ONLY with an
explicit --smoke-test flag and requires a source marked smoke_test=true.

Evaluation is D_reservedC only: exactly the original 500 items in original order.
Save pred, q1_pred, q2_pred, existing scorer labels, legacy composition labels,
answer NLL and token counts for every item, plus all six adapters and losses.
Base predictions are copied verbatim; no base-model evaluation stage is invoked.
Targets and own-Q1 answers are not regenerated; teacher/PPL diagnostics are not
repeated. A/B/C are not reevaluated in this explicitly D-only extension.

Reporting: original scorer only, per-seed composition/hop1/hop2 accuracies;
separately original0/1/2, added3/4/5 and all0..5 mean/SD, gain over cached baseline,
paired context-minus-direct seed differences and 95% t intervals (df2 or df5).
No pooling six fits as independent questions; no new confirmatory testing family
or significance-based stopping. D has already been observed: this assesses
training stability, not independent untouched-dataset generalization.
Only completion status and these original-scorer numbers are to be reported
after the run. Await user-submitted scorer before uniformly rescoring base plus
all twelve trained models. Preserve both original labels and future new labels.

One serial Slurm job; stud/qos stud; one >=40GB GPU; 8CPU/64GB/24h. Fewer than two
existing queued/running jobs; with one existing job use afterany dependency.
Require 6GB quota headroom and fresh login-node lquota, with timestamped fallback
ticket for compute nodes. No automatic retry or overwrites. Each job uses a fresh
runs_JOBID. A lightweight review archive is generated at the end, excluding
adapters; all adapters remain on HPC. No deletion or cleanup is authorized.

Software validation: six tiny CPU fits in the existing SciPy-free environment,
source/prediction/target immutability, configuration and pairing checks, hash
tampering rejection, no-overwrite rejection, and scheduler guard checks. Local
tiny smoke is a software check, not a 14B result or performance estimate.


返回 [[00 研究首页]]
