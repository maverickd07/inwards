"""Paired comparisons, seed uncertainty, cluster sensitivity and blinded audit.

Human semantic classifications are deliberately left blank. String normalization
cannot establish whether knowledge use or answer formatting caused a benefit.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import random
import statistics
import unicodedata
from pathlib import Path

import numpy as np
from exp1_gap_v2_v1 import clear_question
from scoring_v2_v1 import is_correct, old_is_correct

SPLITS = ["A_newPerson", "B_newCategory", "C_bothNew", "D_reservedC"]
FIELDS = {"composed": "composed_ok", "hop1": "q1_ok", "hop2": "q2_ok"}


def read(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def key(r):
    return r["category"], r["person"]


def js(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mcnemar_exact(wins, losses):
    """Two-sided exact binomial p at p=1/2, using integer arithmetic.

    Equivalent to scipy.stats.binomtest(wins, wins+losses).pvalue;
    equal-tail symmetry holds specifically for the McNemar null p=1/2.
    """
    if wins < 0 or losses < 0:
        raise ValueError("Counts must be nonnegative")
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / (1 << n))


def paired(x, y):
    if len(x) != len(y) or not x:
        raise ValueError("Unpaired data")
    wins = sum(bool(a) and not bool(b) for a, b in zip(x, y))
    losses = sum(bool(b) and not bool(a) for a, b in zip(x, y))
    return {"n": len(x), "wins": wins, "losses": losses,
            "gain_pp": 100 * (wins - losses) / len(x),
            "mcnemar_p": mcnemar_exact(wins, losses)}


def moments(values):
    if len(values) != 3:
        raise ValueError("This fixed protocol requires exactly three training seeds")
    m = statistics.mean(values)
    sd = statistics.stdev(values)
    # Student t(df=2): F(x)=1/2+x/(2*sqrt(x*x+2)). Invert at .975.
    q = 2 * .975 - 1
    t975_df2 = math.sqrt(2 * q * q / (1 - q * q))
    half = t975_df2 * sd / math.sqrt(3)
    return {"values": values, "mean": m, "sample_sd": sd, "seed_t95": [m - half, m + half]}


def holm(records):
    previous = 0.0
    for i, rec in enumerate(sorted(records, key=lambda r: r["mcnemar_p"])):
        previous = max(previous, min(1.0, (len(records) - i) * rec["mcnemar_p"]))
        rec["mcnemar_p_holm"] = previous


def cluster_interval(differences, labels):
    # First average paired seed differences per item; seeds are never pooled as
    # additional test questions. Bootstrap whole clusters, preserving item weights.
    means = np.mean(np.asarray(differences, dtype=float), axis=0)
    groups = collections.defaultdict(list)
    for value, label in zip(means, labels):
        groups[label].append(value)
    totals = np.asarray([sum(v) for v in groups.values()])
    sizes = np.asarray([len(v) for v in groups.values()])
    rng = np.random.default_rng(12014)
    draws = rng.integers(len(groups), size=(2000, len(groups)))
    estimates = 100 * totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {"clusters": len(groups), "bootstrap_replicates": 2000,
            "percentile95_pp": [float(v) for v in np.quantile(estimates, [.025, .975])],
            "scope": "Conditional on fitted seeds; no simultaneous crossed-country/person uncertainty model."}


def write_csv(path, records, columns=None):
    columns = columns or (list(records[0]) if records else [])
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(records)


def surface(s):
    # Deliberately preserves signs, punctuation, symbols, numbers and words.
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def export_audit(out, items, baselines, runs):
    raw = []
    pairs = {}
    for (condition, seed, split), predictions in runs.items():
        for item, base, pred in zip(items[split], baselines[split], predictions):
            if base["composed_ok"] == pred["composed_ok"]:
                continue
            answers = sorted([base["pred"], pred["pred"]])
            pairkey = (key(item), tuple(answers))
            if pairkey not in pairs:
                digest = hashlib.sha256(json.dumps(pairkey, ensure_ascii=False).encode()).hexdigest()[:20]
                shuffled = answers.copy()
                random.Random(digest).shuffle(shuffled)
                pairs[pairkey] = {"audit_id": digest, "category": item["category"],
                    "person": item["person"], "question": clear_question(item["e"]),
                    "reference_aliases": json.dumps(item["e"]["Answer"], ensure_ascii=False),
                    "answer_X": shuffled[0], "answer_Y": shuffled[1],
                    "X_semantically_correct": "", "Y_semantically_correct": "",
                    "change_class": "", "confidence": "", "notes": ""}
            raw.append({"audit_id": pairs[pairkey]["audit_id"], "condition": condition,
                "seed": seed, "split": split, "category": item["category"], "person": item["person"],
                "direction": "repair" if pred["composed_ok"] else "damage",
                "before": base["pred"], "after": pred["pred"],
                "surface_only_candidate": surface(base["pred"]) == surface(pred["pred"]),
                "manual_semantics_status": "pending"})
    write_csv(out / "flip_mapping.csv", raw, ["audit_id", "condition", "seed", "split", "category", "person",
              "direction", "before", "after", "surface_only_candidate", "manual_semantics_status"])
    blinded = list(pairs.values())
    random.Random(12014).shuffle(blinded)
    write_csv(out / "blinded_review.csv", blinded, ["audit_id", "category", "person", "question",
              "reference_aliases", "answer_X", "answer_Y", "X_semantically_correct",
              "Y_semantically_correct", "change_class", "confidence", "notes"])
    return {"flip_records": len(raw), "unique_answer_pairs": len(blinded),
            "repair_records": sum(r["direction"] == "repair" for r in raw),
            "damage_records": sum(r["direction"] == "damage" for r in raw),
            "human_review": "pending; no automatic content/format attribution"}


def analyze(root, out):
    stages = json.loads((root / "stages.json").read_text())
    if {(s["condition"], s["seed"]) for s in stages} != {(c, k) for c in ["context", "direct"] for k in range(3)} or len(stages) != 6:
        raise ValueError("Expected exactly six completed fits")
    manifest = json.loads((root / "plan/exp12_manifest.json").read_text())
    items = {s: read(root / "plan" / f"items_{s}.jsonl") for s in SPLITS}
    train = read(root / "plan/train.jsonl")
    baselines = {s: read(root / "baseline" / f"eval_{s}.jsonl") for s in SPLITS}
    data, configs, init = {}, {}, {}

    def validate_predictions(rows, split):
        if list(map(key, rows)) != list(map(key, items[split])):
            raise ValueError("Prediction identity/order mismatch")
        for r, item in zip(rows, items[split]):
            for field, pred, gold in [("composed_ok", "pred", "Answer"),
                                     ("q1_ok", "q1_pred", "A1"), ("q2_ok", "q2_pred", "A2")]:
                if r[field] != is_correct(r[pred], item["e"][gold]):
                    raise ValueError("Saved scoring mismatch")

    for s in SPLITS:
        validate_predictions(baselines[s], s)
    for st in stages:
        c, k = st["condition"], st["seed"]
        p = root / st["path"]
        completed = json.loads((p / "complete.json").read_text())
        if completed["adapter_sha256"] != sha(p / "adapter/adapter_model.safetensors"):
            raise ValueError("Adapter integrity mismatch")
        cfg = json.loads((p / "run_config.json").read_text())
        expected_steps = 2 * math.ceil(len(train) / 8)
        if (cfg["condition"], cfg["train_seed"], cfg["epochs"], cfg["batch"], cfg["optimizer_steps"],
            cfg["sample_exposures"]) != (c, k, 2, 8, expected_steps, 2 * len(train)):
            raise ValueError("Training budget differs from fixed protocol")
        target_path = root / "paired_targets" / f"{c}_targets.jsonl"
        if cfg["targets_sha256"] != sha(target_path):
            raise ValueError("Training target hash mismatch")
        examples = read(p / "train_examples.jsonl")
        targets = read(target_path)
        if list(map(key, examples)) != list(map(key, train)) or list(map(key, targets)) != list(map(key, train)):
            raise ValueError("Training identity mismatch")
        if [r["target"] for r in examples] != [r["target"] for r in targets]:
            raise ValueError("Actual training answers differ from targets")
        init[c, k] = json.loads((p / "initialization.json").read_text())["lora_initial_sha256"]
        configs[f"{c}_s{k}"] = cfg
        for s in SPLITS:
            data[c, k, s] = read(p / f"eval_{s}.jsonl")
            validate_predictions(data[c, k, s], s)
    for seed in range(3):
        if init["context", seed] != init["direct", seed]:
            raise ValueError("Paired initialization differs")

    records, groups, primary, retention = [], [], [], []
    for split in SPLITS:
        for metric, field in FIELDS.items():
            base = [r[field] for r in baselines[split]]
            for c in ["context", "direct"]:
                gains, accuracies = [], []
                for seed in range(3):
                    values = [r[field] for r in data[c, seed, split]]
                    rec = {"split": split, "metric": metric, "condition": c, "control": "baseline",
                           "seed": seed, "accuracy_pct": 100 * statistics.mean(values), **paired(values, base)}
                    records.append(rec)
                    gains.append(rec["gain_pp"])
                    accuracies.append(rec["accuracy_pct"])
                    if split == "D_reservedC":
                        (primary if metric == "composed" else retention).append(rec)
                groups.append({"split": split, "metric": metric, "condition": c,
                    "baseline_pct": 100 * statistics.mean(base), "accuracy_pct": moments(accuracies),
                    "gain_pp": moments(gains)})
    comparisons = []
    for split in SPLITS:
        for metric, field in FIELDS.items():
            comparisons_seed, diffs = [], []
            for seed in range(3):
                x = [r[field] for r in data["context", seed, split]]
                y = [r[field] for r in data["direct", seed, split]]
                rec = {"split": split, "metric": metric, "condition": "context", "control": "direct",
                       "seed": seed, **paired(x, y)}
                comparisons_seed.append(rec)
                diffs.append([int(a) - int(b) for a, b in zip(x, y)])
                if split == "D_reservedC":
                    (primary if metric == "composed" else retention).append(rec)
            comparison = {"split": split, "metric": metric, "paired_seeds": comparisons_seed,
                          "gain_pp": moments([r["gain_pp"] for r in comparisons_seed])}
            if split == "D_reservedC":
                comparison["person_cluster"] = cluster_interval(diffs, [r["person"] for r in items[split]])
                comparison["bridge_country_cluster"] = cluster_interval(diffs, [r["e"]["A1"][0] for r in items[split]])
            comparisons.append(comparison)
    if len(primary) != 9 or len(retention) != 18:
        raise ValueError("Multiplicity family changed")
    holm(primary)
    holm(retention)

    target_rows = read(root / "paired_targets/paired_diagnostics.jsonl")
    if list(map(key, target_rows)) != list(map(key, train)):
        raise ValueError("Unpaired teacher diagnostics")
    for r, item in zip(target_rows, train):
        for c in ["context", "direct"]:
            if r[f"{c}_ok"] != is_correct(r[f"{c}_target"], item["e"]["Answer"]):
                raise ValueError("Teacher scoring mismatch")
    teacher = {**paired([r["context_ok"] for r in target_rows], [r["direct_ok"] for r in target_rows]),
               "context_pct": 100 * statistics.mean(r["context_ok"] for r in target_rows),
               "direct_pct": 100 * statistics.mean(r["direct_ok"] for r in target_rows),
               "same_target_n": sum(r["context_target"] == r["direct_target"] for r in target_rows),
               "scope": "Paired training questions, same frozen base model; diagnostic, not held-out transfer."}
    teacher_diff = [[int(r["context_ok"]) - int(r["direct_ok"]) for r in target_rows]]
    teacher["person_cluster"] = cluster_interval(teacher_diff, [r["person"] for r in target_rows])
    perplexities = {}
    for source in ["context", "direct"]:
        for prompt in ["context", "direct"]:
            field = f"{source}_under_{prompt}"
            valid = [r[field] for r in target_rows if r[field]["n_tokens"]]
            total_tokens = sum(v["n_tokens"] for v in valid)
            corpus_nll = sum(v["sum_nll"] for v in valid) / total_tokens if total_tokens else None
            perplexities[field] = {"n": len(valid), "empty_n": len(target_rows) - len(valid),
                "token_weighted_nll": corpus_nll,
                "token_weighted_perplexity": math.exp(corpus_nll) if corpus_nll is not None and corpus_nll < 700 else None,
                "mean_per_item_nll": statistics.mean(v["mean_nll"] for v in valid) if valid else None}
    paired_nll = {}
    for source in ["context", "direct"]:
        rows = [r for r in target_rows if r[f"{source}_under_context"]["n_tokens"] and r[f"{source}_under_direct"]["n_tokens"]]
        ds = [r[f"{source}_under_context"]["mean_nll"] - r[f"{source}_under_direct"]["mean_nll"] for r in rows]
        paired_nll[source] = {"n": len(ds), "same_answer_context_minus_direct_nll": statistics.mean(ds) if ds else None}
    sensitivity = []
    for (c, k, split), rows in data.items():
        sensitivity.append({"condition": c, "seed": k, "split": split,
            "legacy_pct": 100 * statistics.mean(old_is_correct(r["pred"], i["e"]["Answer"]) for r, i in zip(rows, items[split])),
            "truncate_pct": 100 * statistics.mean(is_correct(r["pred"], i["e"]["Answer"], "truncate") for r, i in zip(rows, items[split]))})
    out.mkdir(parents=True, exist_ok=False)
    audit = export_audit(out, items, baselines, data)
    report = {"manifest": manifest, "groups": groups, "per_run": records,
        "context_minus_direct": comparisons, "teacher_advantage": teacher,
        "perplexity_diagnostics": perplexities, "same_answer_prompt_effect": paired_nll,
        "primary_family_n": len(primary), "retention_family_n": len(retention),
        "configurations": configs, "scoring_sensitivity": sensitivity, "flip_audit": audit,
        "limitations": ["Only one task domain/model family; country/person overlap and correlated questions remain.",
            "D excludes 76 archived C evaluation files (500 unique identities), not all historical exp1 measurement exposure.",
            "Six fits, one cached target pair per question; seed intervals reflect training variability only.",
            "Context teacher costs extra first-hop generation and longer prompts; gradient budgets match, total compute does not.",
            "Perplexity is conditional answer-string likelihood, not a factuality estimate or a separate training arm.",
            "Manual semantic adjudication remains pending; scorer flips are not automatically factual repairs.",
            "Failure to detect context/direct differences does not establish equivalence; retention is not a formal noninferiority claim."]}
    js(out / "report.json", report)
    write_csv(out / "metrics.csv", [{"split": g["split"], "metric": g["metric"], "condition": g["condition"],
        "baseline_pct": g["baseline_pct"], "accuracy_pct": g["accuracy_pct"]["mean"],
        "seed_sd": g["accuracy_pct"]["sample_sd"], "gain_pp": g["gain_pp"]["mean"],
        "seed_ci_low": g["gain_pp"]["seed_t95"][0], "seed_ci_high": g["gain_pp"]["seed_t95"][1]} for g in groups])
    lines = ["EXP12: six fits; primary D_reservedC; no seed pooling.",
             f"Teacher context-minus-direct: {teacher['gain_pp']:+.2f} pp (paired training questions)"]
    for g in groups:
        if g["split"] == "D_reservedC":
            lines.append(f"{g['condition']} {g['metric']}: {g['accuracy_pct']['mean']:.2f}% +/- {g['accuracy_pct']['sample_sd']:.2f} SD; gain {g['gain_pp']['mean']:+.2f} pp")
    comp = next(c for c in comparisons if c["split"] == "D_reservedC" and c["metric"] == "composed")
    lines += [f"Context-minus-direct D composed: {comp['gain_pp']}",
              "Manual content/format review pending: blinded_review.csv. Stop and decide next direction."]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    analyze(a.runs, a.out)
