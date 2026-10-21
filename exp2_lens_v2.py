"""
Experiment 2: logit lens on the knowledge-utilization gap.

Question: when the model knows both hops but fails the composed question,
does the bridge entity (the hop-1 answer, e.g. the birth country) still show up
inside the model? If it does but is not used, knowledge is present and
utilization breaks. If it never shows up, hop 1 is not resolved internally.

For each sampled item (both hops correct on their own) we read the residual
stream at the last prompt position, project every layer onto the vocabulary
(final norm + unembedding), and record the rank of:
  - the bridge entity    on the composed (clear-wording) prompt
  - the final answer     on the composed prompt
  - the bridge entity    on the hop-1-only prompt (positive control)
Rank 0 = top token. Only the first token of each alias is checked.

Usage:
  python exp2_lens.py --model models/Qwen3-14B \
      --items results/exp1v2_Qwen3-14B/items.jsonl --out results/lens_Qwen3-14B
"""
import argparse
import json
import os
import random
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp1_gap_v2 import clear_question, to_prompt


def cat_type(c):
    """clean: birth country -> ordinary second hop (main result).
    translation: birth country -> its name in another language; failures there are often
    the right country in English, so the bridge is trivially present. Reported separately."""
    if c.startswith("birthplace"):
        return "translation" if "common_name" in c else "clean"
    return "other"


def first_ids(tok, aliases):
    ids = set()
    for a in aliases:
        a = str(a).strip()
        if not a:
            continue
        for s in (a, " " + a):
            t = tok.encode(s, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--data", default="data/compositional_celebrities.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300, help="clean items per group (fail / success)")
    ap.add_argument("--n-trans", type=int, default=100, help="translation items per group")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.load(open(args.data))["data"]
    by_key = {(e["category"], e["person"], e["Question"]): e for e in data}

    rows = [json.loads(line) for line in open(args.items)]
    both = [r for r in rows if r["q1"] and r["q2"]]
    rng = random.Random(args.seed)
    # (set, group) -> items. Date/year categories are left out: Qwen splits numbers into
    # single digits, so a first-token rank cannot tell whether "1936" is represented.
    groups = {}
    for set_name, n in (("clean", args.n), ("translation", args.n_trans)):
        pool = [r for r in both if cat_type(r["category"]) == set_name]
        for g, sub in (("fail", [r for r in pool if not r["clear"]]),
                       ("success", [r for r in pool if r["clear"]])):
            groups[(set_name, g)] = rng.sample(sub, min(n, len(sub)))
    print({f"{s}/{g}": len(v) for (s, g), v in groups.items()}, "items sampled", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    norm, head = model.model.norm, model.lm_head
    L = model.config.num_hidden_layers

    @torch.no_grad()
    def lens(prompt, id_groups):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states   # L+1 entries, last one already normed
        res = {k: [] for k in id_groups}
        for l in range(1, L + 1):
            h = hs[l][0, -1]
            if l < L:
                h = norm(h)
            logits = head(h).float()
            for k, ids in id_groups.items():
                res[k].append(min(int((logits > logits[i]).sum()) for i in ids))
        return res

    os.makedirs(args.out, exist_ok=True)
    recs, skipped, t0 = [], 0, time.time()
    with open(os.path.join(args.out, "lens_items.jsonl"), "w") as f:
        for (set_name, gname), rs in groups.items():
            for r in rs:
                e = by_key.get((r["category"], r["person"], r["question"]))
                if e is None:
                    skipped += 1
                    continue
                ids = {"bridge": first_ids(tok, e["A1"]), "answer": first_ids(tok, e["Answer"])}
                comp = lens(to_prompt(tok, clear_question(e)), ids)
                ctrl = lens(to_prompt(tok, e["Q1"]), {"bridge": ids["bridge"]})
                rec = {"set": set_name, "group": gname, "category": r["category"], "person": r["person"],
                       "bridge": e["A1"][0], "answer": e["Answer"][0], "pred_clear": r["pred_clear"],
                       "bridge_ranks": comp["bridge"], "answer_ranks": comp["answer"],
                       "q1_bridge_ranks": ctrl["bridge"]}
                recs.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"lens done on {len(recs)} items in {time.time() - t0:.0f}s (skipped {skipped})", flush=True)

    # ------------------------------------------------------------ summary
    step = max(1, L // 10)
    show = sorted(set(list(range(step, L + 1, step)) + [L]))
    summary = {"model": args.model, "layers": L, "sets": {}}

    def report(title, rs, key):
        if not rs:
            return
        per_layer = list(zip(*[x[key] for x in rs]))          # per_layer[l-1] = ranks at layer l
        ever10 = sum(min(x[key]) < 10 for x in rs) / len(rs)
        peak = statistics.median([x[key].index(min(x[key])) + 1 for x in rs])
        print(f"\n{title}  (n={len(rs)})   bridge ever in top-10: {ever10:.0%}   median best layer: {peak:.0f}/{L}")
        print("  layer      " + " ".join(f"{l:>6d}" for l in show))
        print("  median rk  " + " ".join(f"{statistics.median(per_layer[l-1]):>6.0f}" for l in show))
        print("  % top-10   " + " ".join(f"{sum(v < 10 for v in per_layer[l-1]) / len(rs):>6.0%}" for l in show))
        summary["sets"][title] = {"n": len(rs), "ever_top10": ever10, "median_best_layer": peak,
                                  "median_rank_by_layer": [statistics.median(v) for v in per_layer],
                                  "top10_by_layer": [sum(v < 10 for v in col) / len(rs) for col in per_layer]}

    for set_name, label in (("clean", "CLEAN categories: birth country -> ordinary 2nd hop (MAIN RESULT)"),
                            ("translation", "TRANSLATION categories: country name in another language "
                                            "(failures often = right country in English; bridge trivially present)")):
        sub = [x for x in recs if x["set"] == set_name]
        if not sub:
            continue
        tag = set_name[:5]
        print("\n" + "=" * 100 + f"\n{label}")
        fail = [x for x in sub if x["group"] == "fail"]
        succ = [x for x in sub if x["group"] == "success"]
        report(f"[{tag}] composed, model FAILS -> bridge rank", fail, "bridge_ranks")
        report(f"[{tag}] composed, model SUCCEEDS -> bridge rank", succ, "bridge_ranks")
        report(f"[{tag}] hop-1 alone (control) -> bridge rank", sub, "q1_bridge_ranks")
        report(f"[{tag}] composed, model FAILS -> final answer rank", fail, "answer_ranks")
        report(f"[{tag}] composed, model SUCCEEDS -> final answer rank", succ, "answer_ranks")

    json.dump(summary, open(os.path.join(args.out, "lens_summary.json"), "w"), indent=2)
    print("\nHow to read: if the FAIL group's bridge still reaches the top-10 in middle layers (close to the"
          "\nSUCCESS group), the hop-1 fact is retrieved but not used. If it never surfaces in the FAIL"
          "\ngroup, hop 1 is not resolved internally on the composed prompt.")


if __name__ == "__main__":
    main()
