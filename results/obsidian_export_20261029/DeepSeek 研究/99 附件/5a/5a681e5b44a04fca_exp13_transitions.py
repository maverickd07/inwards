"""Per-item transitions base -> adapter on 1229 two-hop items: format tightening vs genuine fix vs loss."""
import json, sys, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from exp13_format_check_lib import norm, contains  # noqa

RUN = Path(sys.argv[1])
items = [json.loads(l) for l in open(RUN / "inputs/items.jsonl")]
two = [it for it in items if it["shape"] == "2hop"]
def load(m):
    P = {json.loads(l)["query_id"]: json.loads(l) for l in open(RUN / m / "predictions.jsonl")}
    S = {json.loads(l)["id"]: json.loads(l) for l in open(RUN / m / "scored_items.jsonl")}
    return P, S
bP, bS = load("baseline")
rows = {}
examples = []
for cond in ["context", "direct"]:
    agg = {k: [] for k in ["tighten", "genuine_fix", "em_loss", "em_loss_but_lenient", "lenient_gain_new", "lenient_loss"]}
    for s in range(6):
        m = f"{cond}_s{s}"; P, S = load(m)
        c = dict.fromkeys(agg, 0)
        for it in two:
            be, ae = bS[it["id"]]["scores"]["em"], S[it["id"]]["scores"]["em"]
            bl = contains(bP[it["query_id"]]["prediction"], it["answers"])
            al = contains(P[it["query_id"]]["prediction"], it["answers"])
            if not be and ae:
                if bl: c["tighten"] += 1
                else:
                    c["genuine_fix"] += 1
                    examples.append((m, it["question"], it["answers"][0], bP[it["query_id"]]["prediction"], P[it["query_id"]]["prediction"]))
            if be and not ae:
                c["em_loss"] += 1
                if al: c["em_loss_but_lenient"] += 1
            if not bl and al: c["lenient_gain_new"] += 1
            if bl and not al: c["lenient_loss"] += 1
        for k in agg: agg[k].append(c[k])
    rows[cond] = agg
    print(cond)
    for k, v in agg.items(): print(f"  {k:22s} per-seed={v}  mean={st.mean(v):.2f}")
print("\ngenuine fixes (base not containing gold -> adapter EM):")
seen = {}
for m, q, g, b, a in examples:
    seen.setdefault((q, g, b), []).append((m, a))
for (q, g, b), ms in seen.items():
    print(f"- [{len(ms)} runs: {','.join(x[0] for x in ms)}] Q: {q} | gold: {g} | base: {b!r} -> {ms[0][1]!r}")
json.dump(rows, open(Path(sys.argv[2]) / "exp13_transitions.json", "w"), indent=1)
