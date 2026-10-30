"""Diagnostic (read-only) on exp13 job701955: answer format drift, lenient containment, single-hop known rate."""
import json, re, string, statistics as st, sys
from pathlib import Path

RUN = Path(sys.argv[1])
ITEMS = RUN / "inputs/items.jsonl"
MODELS = ["baseline"] + [f"context_s{i}" for i in range(6)] + [f"direct_s{i}" for i in range(6)]

def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def contains(pred, golds):
    p = norm(pred)
    return any(norm(g) and re.search(r"(^| )" + re.escape(norm(g)) + r"( |$)", p) for g in golds)

SENT = re.compile(r"\b(is|was|are|were|founded|located|by)\b", re.I)

items = [json.loads(l) for l in open(ITEMS)]
two = [it for it in items if it["shape"] == "2hop"]
hop_gold = {}
for it in items:
    for h in it["hops"]:
        hop_gold[h["query_id"]] = h["answers"]

preds = {}
for m in MODELS:
    preds[m] = {}
    for l in open(RUN / m / "predictions.jsonl"):
        r = json.loads(l)
        preds[m][r["query_id"]] = r
scored = {m: {json.loads(l)["id"]: json.loads(l) for l in open(RUN / m / "scored_items.jsonl")} for m in MODELS}

out = {}
for m in MODELS:
    P = preds[m]
    comp = [P[it["query_id"]] for it in two]
    hops = [P[h["query_id"]] for it in two for h in it["hops"]]
    em = [scored[m][it["id"]]["scores"]["em"] for it in two]
    f1 = [scored[m][it["id"]]["scores"]["f1"] for it in two]
    lenient = [contains(P[it["query_id"]]["prediction"], it["answers"]) for it in two]
    hop_em_all = [all(scored[m][it["id"]]["hops"][k]["scores"]["em"] for k in range(len(it["hops"]))) for it in two]
    hop_len_all = [all(contains(P[h["query_id"]]["prediction"], h["answers"]) for h in it["hops"]) for it in two]
    out[m] = dict(
        comp_mean_tokens=st.mean(r["answer_tokens"] for r in comp),
        comp_sentence_like=sum(bool(SENT.search(r["prediction"])) and r["answer_tokens"] > 5 for r in comp),
        comp_hit_limit=sum(r["hit_token_limit"] for r in comp),
        hop_mean_tokens=st.mean(r["answer_tokens"] for r in hops),
        hop_sentence_like=sum(bool(SENT.search(r["prediction"])) and r["answer_tokens"] > 5 for r in hops),
        comp_em=sum(em), comp_f1pos_em0=sum(1 for e, f in zip(em, f1) if f > 0 and not e),
        comp_lenient=sum(lenient), comp_lenient_not_em=sum(1 for e, c in zip(em, lenient) if c and not e),
        hops_known_em=sum(hop_em_all), hops_known_lenient=sum(hop_len_all),
        gap_lenient=sum(1 for k, c in zip(hop_len_all, lenient) if k and not c),
    )

json.dump(out, open(Path(sys.argv[2]) / "exp13_format_summary.json", "w"), indent=1)
keys = list(out["baseline"].keys())
print("model".ljust(12), *[k[:14].rjust(14) for k in keys])
for m in MODELS:
    print(m.ljust(12), *[(f"{out[m][k]:.2f}" if isinstance(out[m][k], float) else str(out[m][k])).rjust(14) for k in keys])

def avg(prefix, k):
    return st.mean(out[f"{prefix}_s{i}"][k] for i in range(6))
print("\nmeans over 6 seeds (n=1229 two-hop):")
for k in keys:
    print(f"  {k:22s} base={out['baseline'][k]:8.2f}  context={avg('context',k):8.2f}  direct={avg('direct',k):8.2f}")
