"""exp12 matched785 (job701792): where do context/direct adapters fix and break, by baseline stratum (v3 labels)."""
import json, sys, re, statistics as st
from pathlib import Path

RUN = Path(sys.argv[1]); KEYS = Path(sys.argv[2]); OUT = Path(sys.argv[3])
d303 = {tuple(x) for x in json.load(open(KEYS))["identities"]}
def load(m):
    return {(r["category"], r["person"]): r for r in map(json.loads, open(RUN / "analysis_v3" / f"{m}.jsonl"))}
def ok(r, k): return r["scores"][k]["correct"]
def n(s): return re.sub(r"[\s.,]", "", s.lower())
base = load("baseline")
def stratum(r):
    if ok(r, "composed"): return "base_correct"
    if ok(r, "q1") and ok(r, "q2"): return "gap(q1&q2 ok)"
    if not ok(r, "q1"): return "q1_wrong"
    return "q2_wrong(q1 ok)"
report = {}
for subset_name, keys in [("D303", d303), ("D500", set(base))]:
    strata = {}
    for k in keys: strata.setdefault(stratum(base[k]), []).append(k)
    print(f"\n=== {subset_name}: baseline strata sizes", {s: len(v) for s, v in strata.items()})
    for cond in ["context", "direct"]:
        per = {s: [] for s in strata}
        cons = []
        for seed in range(6):
            A = load(f"{cond}_s{seed}")
            for s, ks in strata.items():
                if s == "base_correct": per[s].append(sum(not ok(A[k], "composed") for k in ks))   # losses
                else: per[s].append(sum(ok(A[k], "composed") for k in ks))                          # fixes
            q1ok = [k for k in keys if ok(A[k], "q1")]
            cons.append(sum(n(A[k]["original_prediction"]["pred"]) == n(A[k]["original_prediction"]["q2_pred"]) for k in q1ok) / len(q1ok))
        print(f"  {cond}:")
        for s, v in per.items():
            lab = "LOSSES" if s == "base_correct" else "fixes"
            print(f"    {s:18s} {lab:6s} per-seed={v} mean={st.mean(v):.2f} ({100*st.mean(v)/len(strata[s]):.1f}% of {len(strata[s])})")
        print(f"    composed==own q2 answer (q1 ok items): {[round(c*100,1) for c in cons]}")
        report[f"{subset_name}_{cond}"] = {"per_stratum": per, "consistency": cons}
    bq1 = [k for k in keys if ok(base[k], "q1")]
    bc = sum(n(base[k]["original_prediction"]["pred"]) == n(base[k]["original_prediction"]["q2_pred"]) for k in bq1) / len(bq1)
    print(f"  baseline composed==own q2 answer: {bc*100:.1f}%")
json.dump(report, open(OUT / "exp12_strata.json", "w"), indent=1)
