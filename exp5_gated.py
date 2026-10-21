"""
Experiment 5: confidence-gated back-patching.

Experiment 4 showed fixed-layer back-patching fixes ~15% of failures (vs ~3% control) but also
breaks 5-9% of correct answers, so the net effect of patching everything is slightly negative.
Idea: patch ONLY when the model is unsure of its own unpatched answer.

Step 1 (GPU): for every item from experiment 4, regenerate the unpatched answer greedily and record
          the model's confidence in it: first-token probability, mean token log-prob, min token prob.
Step 2 (CPU): for each confidence measure and each fixed layer pair, choose a threshold on a random
          half of the items and evaluate the gated policy on the other half (200 random splits).
          Net accuracy change is weighted by the real share of failures, as in experiment 4.

Usage:
  python exp5_gated.py --model models/Qwen3-14B --fixed results/fixed_Qwen3-14B --out results/gated_Qwen3-14B
"""
import argparse
import json
import math
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp1_gap_v2 import clear_question, is_correct, to_prompt

MEASURES = ["p_first", "mean_logp", "p_min"]


def auroc(scores, labels):
    """P(score of a wrong item < score of a right item): how well LOW confidence flags wrong answers."""
    pos = [s for s, y in zip(scores, labels) if y]        # y = baseline wrong
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p < n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def net_effect(rows, key, measure, thr, p_fail):
    """Gated policy: patch if confidence < thr. Returns net accuracy change on the population."""
    wrong = [r for r in rows if not r["base_ok"]]
    right = [r for r in rows if r["base_ok"]]
    if not wrong or not right:
        return 0.0, 0.0, 0.0
    fix = sum(r["conf"][measure] < thr and r["pairs"][key]["real_ok"] for r in wrong) / len(wrong)
    brk = sum(r["conf"][measure] < thr and not r["pairs"][key]["real_ok"] for r in right) / len(right)
    return p_fail * fix - (1 - p_fail) * brk, fix, brk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixed", required=True, help="output dir of experiment 4")
    ap.add_argument("--data", default="data/compositional_celebrities.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    data = json.load(open(args.data))["data"]
    by_cp = {}
    for e in data:
        by_cp.setdefault((e["category"], e["person"]), e)
    rows = [json.loads(line) for line in open(os.path.join(args.fixed, "fixed_items.jsonl"))]
    p_fail = json.load(open(os.path.join(args.fixed, "fixed_summary.json")))["p_fail"]
    keys = list(rows[0]["pairs"].keys())

    # ------------------------------------------------------------ step 1: confidence
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    t0, mismatch = time.time(), 0
    with torch.no_grad():
        for i, r in enumerate(rows):
            e = by_cp[(r["category"], r["person"])]
            enc = tok(to_prompt(tok, clear_question(e)), return_tensors="pt", add_special_tokens=False).to(model.device)
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = out.sequences[0, enc["input_ids"].shape[1]:]
            probs = []
            for step, tid in enumerate(gen.tolist()):
                if tid == tok.eos_token_id or tid == tok.pad_token_id:
                    break
                probs.append(float(out.scores[step][0].float().softmax(-1)[tid]))
            probs = probs or [1e-9]
            pred = tok.decode(gen, skip_special_tokens=True).strip()
            mismatch += pred != r["base_pred"]
            r["conf"] = {"p_first": probs[0],
                         "mean_logp": sum(math.log(max(p, 1e-12)) for p in probs) / len(probs),
                         "p_min": min(probs)}
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"confidence done in {time.time() - t0:.0f}s; baseline answer differs from exp 4 on "
          f"{mismatch}/{len(rows)} items (should be ~0)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gated_items.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ step 2: gating with held-out thresholds
    labels = [not r["base_ok"] for r in rows]
    print("\n" + "=" * 100)
    print(f"Items: {sum(labels)} baseline-wrong, {len(rows) - sum(labels)} baseline-right; population failure share {p_fail:.1%}")
    print("\nHow well does LOW confidence flag a wrong answer? (AUROC: 0.5 = useless, 1.0 = perfect)")
    for m in MEASURES:
        print(f"  {m:10s} {auroc([r['conf'][m] for r in rows], labels):.3f}")

    rng = random.Random(0)
    summary = {"p_fail": p_fail, "auroc": {m: auroc([r["conf"][m] for r in rows], labels) for m in MEASURES},
               "results": {}}
    print(f"\nGated policy, threshold picked on a random half, evaluated on the other half ({args.splits} splits).")
    print(f"Ungated = patch everything (experiment 4).  Net = change in overall accuracy on the clean population.")
    print(f"  {'pair':8s} {'measure':10s} {'ungated net':>12s} {'gated net':>10s} {'95% range':>18s} "
          f"{'fix kept':>9s} {'breaks':>8s} {'% splits >0':>12s}")
    for key in keys:
        ungated = net_effect(rows, key, "p_first", float("inf"), p_fail)[0]
        for m in MEASURES:
            nets, fixes, brks = [], [], []
            for _ in range(args.splits):
                idx = list(range(len(rows)))
                rng.shuffle(idx)
                half = len(idx) // 2
                tr, te = [rows[i] for i in idx[:half]], [rows[i] for i in idx[half:]]
                cands = sorted({r["conf"][m] for r in tr}) + [float("inf")]
                best_thr = max(cands, key=lambda th: net_effect(tr, key, m, th, p_fail)[0])
                n, fx, bk = net_effect(te, key, m, best_thr, p_fail)
                nets.append(n), fixes.append(fx), brks.append(bk)
            nets_s = sorted(nets)
            lo, hi = nets_s[int(0.025 * len(nets))], nets_s[int(0.975 * len(nets)) - 1]
            mean = sum(nets) / len(nets)
            pos = sum(n > 0 for n in nets) / len(nets)
            print(f"  {key:8s} {m:10s} {ungated:+12.2%} {mean:+10.2%} {lo:+8.2%}..{hi:+.2%} "
                  f"{sum(fixes) / len(fixes):9.1%} {sum(brks) / len(brks):8.1%} {pos:12.0%}")
            summary["results"][f"{key}|{m}"] = {"ungated_net": ungated, "gated_net_mean": mean,
                                                "gated_net_95": [lo, hi], "fix_rate": sum(fixes) / len(fixes),
                                                "break_rate": sum(brks) / len(brks), "share_splits_positive": pos}
    json.dump(summary, open(os.path.join(args.out, "gated_summary.json"), "w"), indent=2)
    print("\nHow to read: 'gated net' clearly above 0 with most splits > 0 means confidence-gated back-patching is a"
          "\nusable improvement. 'fix kept' = share of failures still fixed; 'breaks' = share of correct answers broken.")


if __name__ == "__main__":
    main()
