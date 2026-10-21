# Inward Learning Lab

Experiments on compositional reasoning, internal knowledge access, and self-generated training signals in language models.

本项目研究模型“单步知道、组合用不上”的现象，并检验只用模型自身生成的信号进行训练，能否改善对新人物和新问题类别的组合推理。

## Experiment index

| Stage | Code | Question |
| --- | --- | --- |
| exp1 | `exp1_gap_v2.py` | Measure the compositionality gap |
| exp2 | `exp2_lens.py`, `exp2_lens_v2.py` | Inspect intermediate answers with a logit lens |
| exp3 | `exp3_backpatch.py` | Explore activation backpatching |
| exp4 | `exp4_fixedpatch.py` | Evaluate fixed-layer patches and controls |
| exp5 | `exp5_gated.py` | Gate interventions by confidence |
| exp6 | `exp6_inward.py`, `exp6b_lens.py` | Train from self-generated targets and inspect changes |
| exp7 | `exp7_audit.py` | Audit scoring, overlap, and scale comparisons |
| exp8 | `inward/results/handoff_analysis_20261003/exp8_filtered.py` | Compare confidence filtering with random selection |
| exp9 | `results/handoff_analysis_20261003/exp9_shuffled/` | Evaluate shuffled-target controls |
| exp10 | `results/handoff_analysis_20261003/exp10_consistency/` | Analyze correct and incorrect training targets |
| exp11 | `results/handoff_analysis_20261003/exp11_iterate/` | Compare iterative reset, continuation, and frozen-target budgets |

## Evidence and limitations

Historical and revised results are retained with their original relative paths. Start with `results/report_final_20261011/evidence.json`, its `SOURCE_MAP.txt`, and `raw_exp11/analysis/report.json`. Some earlier numbers remain unverified; they must not be presented as confirmed findings. Smoke runs use tiny random models and are pipeline checks, not research evidence.

The project ends this stage at exp11. Further experiments require a separate research decision. The evidence does not establish that delayed knowledge activation explains the gap or that self-training advances intermediate reasoning to earlier layers. Scoring ambiguity, seed variation, resource budgets, and baseline regeneration differences matter when interpreting improvements.

## Running the code

Use a Python environment with PyTorch, Transformers, vLLM, PEFT, NumPy, SciPy, scikit-learn, and Matplotlib as required by individual scripts. Dependency versions are not fully locked in the original experiment archive. Read script arguments and each experiment's protocol before running. GPU experiments were developed for Qwen3-14B on an A100 40 GB; this release has not rerun training.

Obtain Qwen3 model weights from their original publisher and the compositional celebrities dataset from its original source (Press et al., 2022, compositionality-gap research). Place the dataset at `inward/data/compositional_celebrities.json` or pass the relevant script's data argument. Third-party dataset redistribution rights have not been verified, so the dataset is excluded from this release. Models, adapters, and bundled fonts are also excluded and retain their publishers' licensing requirements.

Preserve prior results: write new runs to new output directories. On a managed HPC system, use its scheduler and comply with local resource limits.

## Release contents

This is a curated publication copy of the research folder. It includes experiment scripts, protocols, analysis code, configurations, and text-based research results. Original paths are kept to preserve cross-references; duplicate historical copies are intentional.

Personal application documents, resumes, shell histories, runtime logs, credentials, caches, compressed backups, large binary artifacts, and third-party source datasets are excluded. Email addresses and local user home paths in published text have been replaced with placeholders. `PUBLICATION_MANIFEST.json` records hashes and whether each copied file was sanitized; hashes refer to this publication copy, not the original archive.

## License

Original project code and documentation are released under the MIT License. Third-party works, model outputs, and externally sourced data are not relicensed by this repository; their original terms apply.
