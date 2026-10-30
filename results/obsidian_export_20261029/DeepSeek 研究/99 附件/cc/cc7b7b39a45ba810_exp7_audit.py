"""
Audit of experiments 1-2 (CPU only, tokenizer + json, runs in seconds).

(a) Token overlap: when the bridge entity and the final answer share their FIRST token
    (e.g. "United States" / "United States dollar"), the logit lens cannot tell them apart.
    Recompute the experiment-2 table without those items.
(b) Bridge in the composed prompt vs the hop-1-alone prompt: how many layers LATER does it appear
    in the composed question than when asked directly? (Logit lens is blind before ~layer 30 here,
    so this paired comparison is the fair way to read "too late".)
(c) Scorer spot check: print raw model outputs so a human can confirm the automatic scoring.
(d) Fair scaling: composition gap of 1.7B / 8B / 14B on the SAME items (clean, both hops known by all).

Usage: python exp7_audit.py --model models/Qwen3-14B > results/audit.txt
"""
import argparse
import json
import random
import statistics

from transformers import AutoTokenizer

CLEAN = lambda c: c.startswith("birthplace") and "common_name" not in c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/compositional_celebrities.json")
    ap.add_argument("--lens", default="results/lens_Qwen3-14B/lens_items.jsonl")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    data = json.load(open(args.data))["data"]
    by_cp = {}
    for e in data:
        by_cp.setdefault((e["category"], e["person"]), e)

    def fid(aliases):
        ids = set()
        for a in aliases:
            for s in (str(a).strip(), " " + str(a).strip()):
                t = tok.encode(s, add_special_tokens=False)
                if t:
                    ids.add(t[0])
        return ids

    # ---------------------------------------------------------------- (a)
    lens = [json.loads(l) for l in open(args.lens)]
    lens = [x for x in lens if x.get("set") == "clean"]
    for x in lens:
        e = by_cp[(x["category"], x["person"])]
        x["overlap"] = bool(fid(e["A1"]) & fid(e["Answer"]))
    print("=" * 90 + "\n(a) bridge/answer first-token overlap in experiment-2 clean items")
    cats = sorted({x["category"] for x in lens})
    for c in cats:
        xs = [x for x in lens if x["category"] == c]
        print(f"  {c:30s} {sum(x['overlap'] for x in xs):4d}/{len(xs)} overlap")

    def top10_first(ranks):
        for i, r in enumerate(ranks):
            if r < 10:
                return i + 1
        return None

    def stats(xs, key):
        firsts = [top10_first(x[key]) for x in xs]
        hit = [f for f in firsts if f]
        at36 = sum(x[key][35] < 10 for x in xs) / max(len(xs), 1) if xs and len(xs[0][key]) >= 36 else float("nan")
        return len(hit) / max(len(xs), 1), (statistics.median(hit) if hit else float("nan")), at36

    for label, keep in (("ALL clean items (as reported before)", lambda x: True),
                        ("WITHOUT overlapping items (corrected)", lambda x: not x["overlap"])):
        print(f"\n  {label}")
        for g in ("fail", "success"):
            xs = [x for x in lens if x["group"] == g and keep(x)]
            ever, med, at36 = stats(xs, "bridge_ranks")
            print(f"    composed, {g:8s} n={len(xs):4d}  bridge ever top-10 {ever:5.0%}  "
                  f"median first top-10 layer {med}  top-10 at layer 36 {at36:5.0%}")

    # ---------------------------------------------------------------- (b)
    print("\n" + "=" * 90 + "\n(b) composed prompt vs hop-1-alone prompt (non-overlapping items): "
          "first layer where the bridge enters the top-10")
    for g in ("fail", "success"):
        diffs = []
        for x in lens:
            if x["group"] != g or x["overlap"]:
                continue
            a, b = top10_first(x["bridge_ranks"]), top10_first(x["q1_bridge_ranks"])
            if a and b:
                diffs.append(a - b)
        if diffs:
            later = sum(d > 0 for d in diffs) / len(diffs)
            print(f"  {g:8s} n={len(diffs):4d}  median delay {statistics.median(diffs):+.1f} layers  "
                  f"mean {statistics.mean(diffs):+.2f}  later in {later:.0%} of items")

    # ---------------------------------------------------------------- (c)
    rows = [json.loads(l) for l in open("results/exp1v2_Qwen3-14B/items.jsonl")]
    clean_known = [r for r in rows if CLEAN(r["category"]) and r["q1"] and r["q2"]]
    rng = random.Random(0)
    print("\n" + "=" * 90 + "\n(c) scorer spot check, Qwen3-14B. Check by eye that the 'ok' labels are right.")
    for title, pool, k in (("both hops known, composed judged WRONG", [r for r in clean_known if not r["clear"]], 15),
                           ("both hops known, composed judged RIGHT", [r for r in clean_known if r["clear"]], 8)):
        print(f"\n  -- {title}")
        for r in rng.sample(pool, min(k, len(pool))):
            print(f"  [{r['category'][11:]}] q1='{r['pred_q1']}' | q2='{r['pred_q2']}' | composed='{r['pred_clear']}'"
                  f" | gold={r['gold'][:2]}")

    # ---------------------------------------------------------------- (d)
    print("\n" + "=" * 90 + "\n(d) fair scaling: same items for every model (clean, both hops known by ALL models)")
    load = lambda m: {(r["category"], r["person"], r["question"]): r
                      for r in map(json.loads, open(f"results/exp1v2_Qwen3-{m}/items.jsonl"))}
    models = ["1.7B", "8B", "14B"]
    tables = {m: load(m) for m in models}
    common = [k for k in tables["14B"] if CLEAN(k[0]) and all(
        k in tables[m] and tables[m][k]["q1"] and tables[m][k]["q2"] for m in models)]
    print(f"  common items: {len(common)}")
    for m in models:
        rs = [tables[m][k] for k in common]
        n = max(len(rs), 1)
        print(f"  {m:5s} gap_clear {1 - sum(r['clear'] for r in rs) / n:6.1%}   gap_ctx {1 - sum(r['ctx'] for r in rs) / n:6.1%}")
    print("\n  same comparison, 8B vs 14B only (more common items):")
    common2 = [k for k in tables["14B"] if CLEAN(k[0]) and all(
        k in tables[m] and tables[m][k]["q1"] and tables[m][k]["q2"] for m in ("8B", "14B"))]
    print(f"  common items: {len(common2)}")
    for m in ("8B", "14B"):
        rs = [tables[m][k] for k in common2]
        n = max(len(rs), 1)
        print(f"  {m:5s} gap_clear {1 - sum(r['clear'] for r in rs) / n:6.1%}   gap_ctx {1 - sum(r['ctx'] for r in rs) / n:6.1%}")


if __name__ == "__main__":
    main()
