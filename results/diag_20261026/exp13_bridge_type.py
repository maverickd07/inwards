"""exp13: gap-closure by hop-1 relation type (country/birthplace-like bridge vs other). Lenient containment for 'known'."""
import json, sys, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from exp13_format_check_lib import contains  # noqa
RUN = Path(sys.argv[1])
items = [json.loads(l) for l in open(RUN / "inputs/items.jsonl")]
two = [it for it in items if it["shape"] == "2hop"]
def preds(m): return {json.loads(l)["query_id"]: json.loads(l)["prediction"] for l in open(RUN / m / "predictions.jsonl")}
B = preds("baseline")
def rel(it): return it["hops"][0]["source_question"].split(">>")[-1].strip().lower() if ">>" in it["hops"][0]["source_question"] else "natural-language"
COUNTRYISH = ("country", "place of birth", "citizenship", "born")
def group(it): return "country/birth bridge" if any(w in rel(it) for w in COUNTRYISH) else "other bridge"
gap = [it for it in two if all(contains(B[h["query_id"]], h["answers"]) for h in it["hops"]) and not contains(B[it["query_id"]], it["answers"])]
print("gap items (lenient):", len(gap), {g: sum(group(it) == g for it in gap) for g in ["country/birth bridge", "other bridge"]})
print("hop-1 relation counts in gap:", sorted({rel(it) for it in gap})[:40])
for cond in ["context", "direct"]:
    by = {"country/birth bridge": [], "other bridge": []}
    for s in range(6):
        P = preds(f"{cond}_s{s}")
        for g in by:
            sub = [it for it in gap if group(it) == g]
            by[g].append(sum(contains(P[it["query_id"]], it["answers"]) for it in sub))
    print(cond, {g: (v, f"{100*st.mean(v)/max(1,sum(group(it)==g for it in gap)):.1f}%") for g, v in by.items()})
