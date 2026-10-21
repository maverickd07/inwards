"""
Experiment 4: FIXED-layer back-patching on held-out items, with significance tests.

Experiment 3 picked the best (source, target) layer pair per item after the fact, so it was an
upper bound. Here the layer pairs are fixed in advance (chosen from exp 3's grid) and evaluated
on items NOT used in exp 3, on BOTH failed and successful items, so we can measure:
  - fix rate   : composed answer wrong without patch -> right with patch
  - break rate : composed answer right without patch -> wrong with patch
  - net effect : overall accuracy change, weighted by how common failures/successes really are
and compare every number against a random-donor control (same layers, another item's state).

Correctness is always judged on the full generated answer (same scorer as experiment 1), and the
unpatched baseline is regenerated here with the same code path, so labels are self-consistent.

Usage:
  python exp4_fixedpatch.py --model models/Qwen3-14B \
      --items results/exp1v2_Qwen3-14B/items.jsonl \
      --exclude results/patch_Qwen3-14B/patch_items.jsonl --out results/fixed_Qwen3-14B
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

PAIRS = [(36, 8), (38, 4), (34, 12)]      # (source layer, target layer), fixed before seeing this data


def is_clean(c):
    return c.startswith("birthplace") and "common_name" not in c


def mcnemar_p(b, c):
    """Exact two-sided McNemar test: b, c = discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--exclude", default=None, help="patch_items.jsonl from exp 3 (items to leave out)")
    ap.add_argument("--data", default="data/compositional_celebrities.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-fail", type=int, default=400)
    ap.add_argument("--n-success", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    data = json.load(open(args.data))["data"]
    by_key = {(e["category"], e["person"], e["Question"]): e for e in data}
    rows = [json.loads(line) for line in open(args.items)]
    used = set()
    if args.exclude and os.path.exists(args.exclude):
        used = {(j["category"], j["person"]) for j in map(json.loads, open(args.exclude))}
    both = [r for r in rows if is_clean(r["category"]) and r["q1"] and r["q2"]
            and (r["category"], r["person"]) not in used and (r["category"], r["person"], r["question"]) in by_key]
    fail_pool = [r for r in both if not r["clear"]]
    succ_pool = [r for r in both if r["clear"]]
    rng = random.Random(args.seed)
    picked = rng.sample(fail_pool, min(args.n_fail, len(fail_pool))) + \
        rng.sample(succ_pool, min(args.n_success, len(succ_pool)))
    items = [by_key[(r["category"], r["person"], r["question"])] for r in picked]
    print(f"held-out clean items: {len(fail_pool)} failed / {len(succ_pool)} succeeded available; "
          f"using {len(items)} (excluded {len(used)} exp-3 items)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.layers
    L = model.config.num_hidden_layers
    pairs = [(s, t) for s, t in PAIRS if t < s < L]
    sources = sorted({s for s, _ in pairs})
    patch = {"vecs": None}

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if patch["vecs"] is not None and h.shape[1] > 1:          # prefill only
            h[:, -1, :] = patch["vecs"].to(h.dtype)
        return out

    def encode(e):
        return tok(to_prompt(tok, clear_question(e)), return_tensors="pt", add_special_tokens=False).to(model.device)

    def generate(enc, rows=1, vecs=None, target=None):
        handle = None
        if vecs is not None:
            handle = layers[target - 1].register_forward_hook(hook)
            patch["vecs"] = vecs
        try:
            out = model.generate(input_ids=enc["input_ids"].expand(rows, -1),
                                 attention_mask=enc["attention_mask"].expand(rows, -1),
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        finally:
            patch["vecs"] = None
            if handle is not None:
                handle.remove()
        n = enc["input_ids"].shape[1]
        return [tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]

    t0 = time.time()
    with torch.no_grad():
        states = []
        for e in items:
            hs = model(**encode(e), output_hidden_states=True).hidden_states
            states.append({s: hs[s][0, -1].clone() for s in sources})
        print(f"source states in {time.time() - t0:.0f}s", flush=True)

        donors = list(range(len(items)))
        rng.shuffle(donors)
        donors = [d if d != i else (i + 1) % len(items) for i, d in enumerate(donors)]

        os.makedirs(args.out, exist_ok=True)
        recs = []
        with open(os.path.join(args.out, "fixed_items.jsonl"), "w") as f:
            for i, e in enumerate(items):
                enc = encode(e)
                base_pred = generate(enc)[0]
                rec = {"category": e["category"], "person": e["person"], "gold": e["Answer"][0],
                       "base_pred": base_pred, "base_ok": is_correct(base_pred, e["Answer"]), "pairs": {}}
                for s, t in pairs:
                    vecs = torch.stack([states[i][s], states[donors[i]][s]])
                    real, ctrl = generate(enc, rows=2, vecs=vecs, target=t)
                    rec["pairs"][f"{s}->{t}"] = {"real_pred": real, "real_ok": is_correct(real, e["Answer"]),
                                                 "ctrl_pred": ctrl, "ctrl_ok": is_correct(ctrl, e["Answer"])}
                recs.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(items)}  ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------ report
    wrong = [r for r in recs if not r["base_ok"]]
    right = [r for r in recs if r["base_ok"]]
    p_fail = len(fail_pool) / (len(fail_pool) + len(succ_pool))   # real-world share of failures (held-out pool)
    print(f"\ndone in {time.time() - t0:.0f}s")
    print("=" * 100)
    print(f"Baseline regenerated: {len(wrong)} wrong, {len(right)} right "
          f"(population failure share used for net effect: {p_fail:.1%})")
    summary = {"model": args.model, "n_wrong": len(wrong), "n_right": len(right), "p_fail": p_fail, "pairs": {}}

    for s, t in pairs:
        key = f"{s}->{t}"
        rf = sum(r["pairs"][key]["real_ok"] for r in wrong)
        cf = sum(r["pairs"][key]["ctrl_ok"] for r in wrong)
        rb = sum(not r["pairs"][key]["real_ok"] for r in right)
        cb = sum(not r["pairs"][key]["ctrl_ok"] for r in right)
        nw, nr = max(len(wrong), 1), max(len(right), 1)
        # paired test on the wrong items: real fixes vs control fixes
        b = sum(r["pairs"][key]["real_ok"] and not r["pairs"][key]["ctrl_ok"] for r in wrong)
        c = sum(r["pairs"][key]["ctrl_ok"] and not r["pairs"][key]["real_ok"] for r in wrong)
        p_rc = mcnemar_p(b, c)
        net_real = p_fail * rf / nw - (1 - p_fail) * rb / nr
        net_ctrl = p_fail * cf / nw - (1 - p_fail) * cb / nr
        lo, hi = wilson(rf, nw)
        print(f"\n[{key}]")
        print(f"  fix rate   REAL {rf:4d}/{nw} = {rf / nw:6.1%}  (95% CI {lo:.1%}-{hi:.1%})"
              f"   CONTROL {cf:4d}/{nw} = {cf / nw:6.1%}   REAL vs CONTROL p = {p_rc:.2g}")
        print(f"  break rate REAL {rb:4d}/{nr} = {rb / nr:6.1%}   CONTROL {cb:4d}/{nr} = {cb / nr:6.1%}")
        print(f"  net accuracy change on the clean population:  REAL {net_real:+.1%}   CONTROL {net_ctrl:+.1%}")
        summary["pairs"][key] = {"fix_real": rf / nw, "fix_ctrl": cf / nw, "fix_real_ci": [lo, hi],
                                 "break_real": rb / nr, "break_ctrl": cb / nr, "p_real_vs_ctrl": p_rc,
                                 "net_real": net_real, "net_ctrl": net_ctrl}

    # oracle across the fixed pairs (still no per-item tuning beyond these 3)
    any_fix = sum(any(r["pairs"][f"{s}->{t}"]["real_ok"] for s, t in pairs) for r in wrong)
    any_ctrl = sum(any(r["pairs"][f"{s}->{t}"]["ctrl_ok"] for s, t in pairs) for r in wrong)
    print(f"\nAny of the {len(pairs)} fixed pairs fixes: REAL {any_fix / max(len(wrong), 1):.1%}   "
          f"CONTROL {any_ctrl / max(len(wrong), 1):.1%}")
    summary["any_fix_real"] = any_fix / max(len(wrong), 1)
    summary["any_fix_ctrl"] = any_ctrl / max(len(wrong), 1)
    json.dump(summary, open(os.path.join(args.out, "fixed_summary.json"), "w"), indent=2)
    print("\nHow to read: a real effect = REAL fix rate clearly above CONTROL with small p (< 0.01).\n"
          "A usable fix = positive REAL net accuracy change (fixes outweigh breaks).")


if __name__ == "__main__":
    main()
