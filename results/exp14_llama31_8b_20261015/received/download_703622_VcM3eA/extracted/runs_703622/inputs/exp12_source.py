"""Bounded source-ablation experiment. Six fits, fixed targets, no gold training.

The original exp11 implementation is frozen in engine_exp11_v1.py. This module
reuses its optimizer and allocation, and replaces generation/evaluation explicitly.
"""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import torch
import engine_exp11_v1 as engine
from exp1_gap_v2_v1 import clear_question, to_prompt
from scoring_v2_v1 import is_correct, old_is_correct

HERE = Path(__file__).resolve().parent
SPLITS = [*engine.SPLITS, "D_reservedC"]
read, write, js, key, sha = engine.read, engine.write, engine.js, engine.key, engine.sha


def make_plan(a):
    engine.plan(a)
    entries = json.loads(Path(a.data).read_text())["data"]
    by = {}
    for e in entries:
        by.setdefault(key(e), e)
    families = {c: f for f, cats in engine.FAMILIES.items() for c in cats}
    excluded = {tuple(k) for k in json.loads((HERE / "historical_C_exclusion.json").read_text())["keys"]}
    historical = {key(r) for r in read(Path(a.reference_evals) / "eval_C_bothNew.jsonl")}
    if not historical <= excluded:
        raise ValueError("Historical exclusion does not cover reference C")
    candidates = []
    for r in read(a.items):
        fam = families.get(r["category"])
        if fam is None or fam in engine.TRAIN or key(r) not in by:
            continue
        if int(hashlib.md5(r["person"].encode()).hexdigest(), 16) % 2 != 1:
            continue
        if key(r) not in excluded:
            candidates.append({"category": r["category"], "person": r["person"],
                               "e": by[key(r)], "family": fam, "pset": "test"})
    engine.check_unique(candidates)
    if len(candidates) != 756:
        raise ValueError(f"Unexpected reserved candidate count: {len(candidates)}")
    reserved = random.Random(12).sample(candidates, 500)
    full_keys = list(map(key, reserved))
    if a.smoke_test:
        reserved = reserved[:2]
    train = read(a.out / "train.jsonl")
    if {r["person"] for r in reserved} & {r["person"] for r in train}:
        raise ValueError("Train/reserved person leakage")
    write(a.out / "items_D_reservedC.jsonl", reserved)
    js(a.out / "exp12_manifest.json", {
        "primary_split": "D_reservedC", "historical_eval_seed": 0,
        "reserved_eval_seed": 12, "train_seeds": [0, 1, 2],
        "reserved_candidate_n": len(candidates), "reserved_full_keys": full_keys,
        "actual_counts": {s: len(read(a.out / f"items_{s}.jsonl")) for s in SPLITS},
        "historical_exclusion_sha256": sha(HERE / "historical_C_exclusion.json"),
        "smoke_test": a.smoke_test,
        "caveat": "New to the archived training comparisons, not a new dataset or new country/person pool relative to all measurement work.",
    })


@torch.no_grad()
def score_answers(tok, model, questions, answers):
    """Score exactly the retokenized answer strings used by training, no EOS.

    One unpadded sequence per forward avoids left-padding position ambiguity and
    bounds logits memory on 40 GB GPUs. Prefix tokens are never included in NLL.
    """
    if len(questions) != len(answers):
        raise ValueError("Scoring length mismatch")
    results = []
    for q, answer in zip(questions, answers):
        prefix = tok(to_prompt(tok, q), add_special_tokens=False)["input_ids"]
        target = tok(answer, add_special_tokens=False)["input_ids"]
        if not prefix:
            raise ValueError("Empty prompt")
        if not target:
            results.append({"n_tokens": 0, "sum_nll": None, "mean_nll": None,
                            "perplexity": None, "mean_logprob": None})
            continue
        ids = torch.tensor([prefix + target], device=model.device)
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        logits = output.logits[0, len(prefix) - 1:len(prefix) + len(target) - 1].float()
        nll = torch.nn.functional.cross_entropy(logits, ids[0, len(prefix):], reduction="none")
        if not torch.isfinite(nll).all():
            raise RuntimeError("Nonfinite answer NLL")
        avg = float(nll.mean())
        results.append({"n_tokens": len(target), "sum_nll": float(nll.sum()),
                        "mean_nll": avg, "perplexity": math.exp(avg) if avg < 700 else None,
                        "mean_logprob": -avg})
        del output, logits, nll
    return results


@torch.no_grad()
def generate(tok, model, questions, a, scores=False):
    results = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for start in range(0, len(questions), a.gen_batch):
        batch = questions[start:start + a.gen_batch]
        encoded = tok([to_prompt(tok, q) for q in batch], padding=True,
                      add_special_tokens=False, return_token_type_ids=False,
                      return_tensors="pt").to(model.device)
        sequences = model.generate(**encoded, do_sample=False,
                                   max_new_tokens=6 if a.smoke_test else 24,
                                   pad_token_id=pad)
        answers = [tok.decode(row[encoded["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip() for row in sequences]
        stats = score_answers(tok, model, batch, answers) if scores else [{} for _ in answers]
        results.extend({"pred": ans, "mean_logprob": stat.get("mean_logprob"),
                        **stat} for ans, stat in zip(answers, stats))
    return results


def paired_targets(a):
    rows = read(a.plan / "train.jsonl")
    tok, model = engine.load(a)
    a.out.mkdir(parents=True, exist_ok=False)
    questions = [clear_question(r["e"]) for r in rows]
    print("Generating direct targets", flush=True)
    direct = generate(tok, model, questions, a)
    print("Generating own first-hop answers", flush=True)
    first = generate(tok, model, [r["e"]["Q1"] for r in rows], a)
    contextual = [f"Background fact: {r['person']} was born in {p['pred']}.\n"
                  f"Using this fact, answer the question: {q}"
                  for r, p, q in zip(rows, first, questions)]
    print("Generating contextual targets", flush=True)
    context = generate(tok, model, contextual, a)
    # Save the only two files that training is allowed to consume BEFORE labels.
    for name, predictions in [("context", context), ("direct", direct)]:
        write(a.out / f"{name}_targets.jsonl", [
            {"category": r["category"], "person": r["person"], "target": p["pred"]}
            for r, p in zip(rows, predictions)])
    all_scores = {}
    for source, predictions in [("context", context), ("direct", direct)]:
        for prompt_name, prompts in [("direct", questions), ("context", contextual)]:
            print(f"Answer NLL: {source} targets under {prompt_name} prompts", flush=True)
            all_scores[source, prompt_name] = score_answers(tok, model, prompts,
                                                           [p["pred"] for p in predictions])
    diagnostics = []
    for i, r in enumerate(rows):
        rec = {"category": r["category"], "person": r["person"],
               "direct_target": direct[i]["pred"], "context_target": context[i]["pred"],
               "own_q1": first[i]["pred"], "context_prompt": contextual[i],
               "direct_ok": is_correct(direct[i]["pred"], r["e"]["Answer"]),
               "context_ok": is_correct(context[i]["pred"], r["e"]["Answer"]),
               "q1_ok": is_correct(first[i]["pred"], r["e"]["A1"])}
        for (source, prompt_name), scores in all_scores.items():
            rec[f"{source}_under_{prompt_name}"] = scores[i]
        diagnostics.append(rec)
    write(a.out / "paired_diagnostics.jsonl", diagnostics)
    js(a.out / "complete.json", {"stage": "targets", "n": len(rows),
        "gold_used_for_training": False,
        "target_sha256": {c: sha(a.out / f"{c}_targets.jsonl") for c in ["context", "direct"]}})


def evaluate(tok, model, a, out):
    for split in SPLITS:
        rows = read(a.plan / f"items_{split}.jsonl")
        composed = generate(tok, model, [clear_question(r["e"]) for r in rows], a, scores=True)
        q1 = generate(tok, model, [r["e"]["Q1"] for r in rows], a)
        q2 = generate(tok, model, [r["e"]["Q2"] for r in rows], a)
        saved = []
        for r, p, first, second in zip(rows, composed, q1, q2):
            e = r["e"]
            saved.append({"category": r["category"], "person": r["person"],
                "pred": p["pred"], "mean_logprob": p["mean_logprob"],
                "answer_nll": p["mean_nll"], "answer_tokens": p["n_tokens"],
                "q1_pred": first["pred"], "q2_pred": second["pred"],
                "composed_ok": is_correct(p["pred"], e["Answer"]),
                "q1_ok": is_correct(first["pred"], e["A1"]),
                "q2_ok": is_correct(second["pred"], e["A2"]),
                "legacy_composed_ok": old_is_correct(p["pred"], e["Answer"])})
        write(out / f"eval_{split}.jsonl", saved)
        print(split, sum(r["composed_ok"] for r in saved), "/", len(saved), flush=True)


def baseline(a):
    tok, model = engine.load(a)
    a.out.mkdir(parents=True, exist_ok=False)
    evaluate(tok, model, a, a.out)
    checks = []
    for split in engine.SPLITS:
        now = read(a.out / f"eval_{split}.jsonl")
        old = read(Path(a.reference_evals) / f"eval_{split}.jsonl")[:len(now)]
        if list(map(key, now)) != list(map(key, old)):
            raise ValueError("Historical evaluation identity mismatch")
        checks.append({"split": split, "prediction_disagreements": sum(
            x["pred"] != y["base_pred"] for x, y in zip(now, old))})
    js(a.out / "reference_check.json", checks)
    js(a.out / "complete.json", {"stage": "baseline"})


def train(a):
    if a.adapter or a.condition not in ["context", "direct"] or a.epochs != 2:
        raise ValueError("Fixed design: fresh LoRA, two conditions, two epochs")
    expected = a.targets.parent / "complete.json"
    meta = json.loads(expected.read_text())
    if sha(a.targets) != meta["target_sha256"][a.condition]:
        raise ValueError("Target hash mismatch")
    # Match the old optimizer exactly; only evaluation and prompt scoring change.
    engine.evaluate = evaluate
    engine.train(a)
    cfg = json.loads((a.out / "run_config.json").read_text())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    examples = read(a.out / "train_examples.jsonl")
    target_lengths = [len(tok(r["target"] + tok.eos_token, add_special_tokens=False)["input_ids"])
                      for r in examples]
    prompt_lengths = [len(tok(to_prompt(tok, r["question"]), add_special_tokens=False)["input_ids"])
                      for r in examples]
    cfg.update({"experiment": "exp12_sources", "primary_split": "D_reservedC",
                "reserved_eval_seed": 12, "plan_sha256": sha(a.plan / "exp12_manifest.json"),
                "trained_target_tokens_including_eos": 2 * sum(target_lengths),
                "presented_prompt_tokens": 2 * sum(prompt_lengths),
                "max_sequence_tokens": max(x + y for x, y in zip(target_lengths, prompt_lengths))})
    js(a.out / "run_config.json", cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["plan", "targets", "baseline", "train"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--items")
    parser.add_argument("--data")
    parser.add_argument("--reference-evals", default=str(HERE / "reference_exp8"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--condition", default="")
    parser.add_argument("--gen-batch", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    a = parser.parse_args()
    a.adapter, a.epochs, a.batch, a.round = None, 2, 8, 1
    if a.out.exists():
        raise FileExistsError(f"No overwrite: {a.out}")
    if a.smoke_test:
        torch.set_num_threads(1)
    started = time.time()
    {"plan": make_plan, "targets": paired_targets, "baseline": baseline, "train": train}[a.stage](a)
    js(a.out / "timing.json", {"stage": a.stage, "seconds": time.time() - started})
    js(a.out / "runtime.json", {"python": sys.version, "platform": platform.platform(),
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "numpy"]},
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "smoke_test": a.smoke_test})


if __name__ == "__main__":
    main()
