"""Error taxonomy on 1229 two-hop items: does the model stop at the bridge (answer = hop-1 gold)? Losses listed."""
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
def etype(pred, it, P):
    if contains(pred, it["answers"]): return "correct"
    if contains(pred, it["hops"][0]["answers"]): return "bridge_stop"          # answered hop-1 entity
    if contains(P[it["hops"][0]["query_id"]]["prediction"], [pred]) and norm(pred): return "own_hop1_answer"  # answered its own hop-1 guess
    return "other_wrong"
res = {}
for m in ["baseline"] + [f"{c}_s{s}" for c in ["context", "direct"] for s in range(6)]:
    P, S = load(m)
    cnt = {}
    for it in two:
        t = etype(P[it["query_id"]]["prediction"], it, P); cnt[t] = cnt.get(t, 0) + 1
    res[m] = cnt
for m, c in res.items(): print(m.ljust(12), c)
for cond in ["context", "direct"]:
    print(cond, "mean", {k: round(st.mean(res[f"{cond}_s{s}"].get(k, 0) for s in range(6)), 2) for k in res["baseline"]})

# what happens to base bridge_stop items under context
bP, bS = load("baseline")
bs_items = [it for it in two if etype(bP[it["query_id"]]["prediction"], it, bP) == "bridge_stop"]
print(f"\nbase bridge_stop items: {len(bs_items)}")
for cond in ["context", "direct"]:
    trans = {}
    for s in range(6):
        P, _ = load(f"{cond}_s{s}")
        for it in bs_items:
            t = etype(P[it["query_id"]]["prediction"], it, P); trans[t] = trans.get(t, 0) + 1
    print(f"  under {cond} (summed over 6 seeds):", trans)

# EM losses under context: base answer vs adapter answer
print("\ncontext EM losses (base EM=1 -> adapter EM=0), by item:")
loss = {}
for s in range(6):
    P, S = load(f"context_s{s}")
    for it in two:
        if bS[it["id"]]["scores"]["em"] and not S[it["id"]]["scores"]["em"]:
            loss.setdefault(it["id"], []).append((s, P[it["query_id"]]["prediction"], etype(P[it["query_id"]]["prediction"], it, P)))
for iid, v in sorted(loss.items(), key=lambda kv: -len(kv[1])):
    it = next(x for x in two if x["id"] == iid)
    print(f"- [{len(v)}/6] {it['question']} | gold {it['answers'][0]} | hop1 {it['hops'][0]['answers'][0]} | -> {[(a, t) for _, a, t in v][:2]}")
json.dump(res, open(Path(sys.argv[2]) / "exp13_error_types.json", "w"), indent=1)
