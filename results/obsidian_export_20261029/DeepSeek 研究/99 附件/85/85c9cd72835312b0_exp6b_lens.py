"""
Experiment 6b: corrected logit-lens check of the inward-learning adapters.

Fixes in this version compared with the lens inside exp6:
  - items whose bridge entity and final answer share the first token are excluded
    (otherwise "bridge appears" could just mean "answer appears");
  - the bridge layer is compared with the hop-1-alone prompt as a per-item reference;
  - base, self-trained and gold-trained models are compared on the SAME items in one process.

Usage:
  python exp6b_lens.py --model models/Qwen3-14B --eval results/inward_self_Qwen3-14B/eval_C_bothNew.jsonl \
      --adapters self=results/inward_self_Qwen3-14B/adapter gold=results/inward_gold_Qwen3-14B/adapter \
      --out results/lens6b_Qwen3-14B
"""
import argparse
import json
import math
import os
import statistics

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp1_gap_v2 import clear_question, to_prompt


def sign_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True, help="eval_C_bothNew.jsonl from exp6 (defines the items)")
    ap.add_argument("--adapters", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--data", default="data/compositional_celebrities.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(args.data))["data"]
    by_cp = {}
    for e in data:
        by_cp.setdefault((e["category"], e["person"]), e)
    ev = [json.loads(l) for l in open(args.eval)]

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    adapters = dict(a.split("=", 1) for a in args.adapters)
    names = list(adapters)
    model = PeftModel.from_pretrained(base, adapters[names[0]], adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(adapters[n], adapter_name=n)
    model.eval()
    norm, head = base.model.norm, base.lm_head
    L = base.config.num_hidden_layers

    def fid(aliases):
        ids = set()
        for a in aliases:
            for s in (str(a).strip(), " " + str(a).strip()):
                t = tok.encode(s, add_special_tokens=False)
                if t:
                    ids.add(t[0])
        return ids

    items, n_overlap = [], 0
    for r in ev:
        e = by_cp[(r["category"], r["person"])]
        b, a = fid(e["A1"]), fid(e["Answer"])
        if b & a:
            n_overlap += 1
            continue
        items.append((r, e, sorted(b)))
    print(f"{len(items)} items kept, {n_overlap} excluded for bridge/answer token overlap", flush=True)

    @torch.no_grad()
    def first_layer(question, ids):
        enc = tok(to_prompt(tok, question), return_tensors="pt", add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        for l in range(1, L + 1):
            h = hs[l][0, -1] if l == L else norm(hs[l][0, -1])
            logits = head(h).float()
            if min(int((logits > logits[i]).sum()) for i in ids) < 10:
                return l
        return None

    res = {"base": [], **{n: [] for n in names}}
    ref = []
    for r, e, ids in items:
        with model.disable_adapter():
            res["base"].append(first_layer(clear_question(e), ids))
            ref.append(first_layer(e["Q1"], ids))
        for n in names:
            model.set_adapter(n)
            res[n].append(first_layer(clear_question(e), ids))

    print("\n" + "=" * 90)
    print(f"First layer (of {L}) at which the bridge enters the top-10, composed question, n={len(items)}")
    print(f"  reference: hop-1 asked directly (base model): hit {sum(x is not None for x in ref) / len(ref):.0%}, "
          f"median {statistics.median([x for x in ref if x]) if any(ref) else 'nan'}")
    summary = {"n": len(items), "excluded_overlap": n_overlap, "conditions": {}}
    for c, v in res.items():
        hit = [x for x in v if x is not None]
        med = statistics.median(hit) if hit else float("nan")
        line = f"  {c:6s} hit {len(hit) / len(v):5.0%}  median first layer {med}"
        entry = {"hit": len(hit) / len(v), "median_layer": med}
        if c != "base":
            pairs = [(b, t) for b, t in zip(res["base"], v)]
            earlier = sum((t is not None) and (b is None or t < b) for b, t in pairs)
            later = sum((b is not None) and (t is None or t > b) for b, t in pairs)
            line += f"   vs base: earlier {earlier}, later {later}, sign test p = {sign_p(earlier, later):.2g}"
            entry.update({"earlier": earlier, "later": later, "p": sign_p(earlier, later)})
            fixed = [i for i, (r, _, _) in enumerate(items) if r["trained_ok"] and not r["base_ok"]] if c == names[0] else []
            if fixed:
                fe = sum((v[i] is not None) and (res["base"][i] is None or v[i] < res["base"][i]) for i in fixed)
                line += f"\n         among the {len(fixed)} items this adapter FIXED: bridge earlier in {fe}"
        print(line)
        summary["conditions"][c] = entry
    os.makedirs(args.out, exist_ok=True)
    json.dump(summary, open(os.path.join(args.out, "lens6b_summary.json"), "w"), indent=2)
    print("\nHow to read: a real 'learned to hop earlier' effect = trained median layer clearly below base,"
          "\nmany more 'earlier' than 'later' items with small p. If the layer does not move but accuracy rose,"
          "\nthe adapter improved composition by some other route, and the paper must not claim 'earlier hop'.")


if __name__ == "__main__":
    main()
