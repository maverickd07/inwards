"""
Experiment 1: measure the knowledge-utilization (compositionality) gap.

For every 2-hop question in Compositional Celebrities we ask the model:
  q1   - first hop alone            ("What is the birthplace of Rumi?")
  q2   - second hop alone           ("What is the capital of Afghanistan?")
  comp - the composed question      ("What is the capital of the birthplace of Rumi?")
  ctx  - composed question with the first-hop fact given in context

Gap = share of items where the model gets BOTH hops right on their own
      but still gets the composed question wrong.

Usage:
  python exp1_gap.py --model models/Qwen3-0.6B --out results/exp1_Qwen3-0.6B
  python exp1_gap.py --dry-run          # logic test without a GPU
"""
import argparse
import json
import os
import random
import re
import time
import unicodedata
from collections import defaultdict

SYSTEM = "Answer the question with only the answer itself: a name, a number, or a short phrase. No explanation."


# ---------------------------------------------------------------- scoring
def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"<think>.*?</think>", " ", s, flags=re.S)
    s = s.replace("+", " ")
    s = re.sub(r"[^\w\s]", " ", s)          # drop punctuation, keep all scripts
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\b0+(\d)", r"\1", s)        # 004 -> 4
    return re.sub(r"\s+", " ", s).strip()


def is_correct(pred, aliases):
    p = norm(pred)
    for a in aliases:
        raw = unicodedata.normalize("NFKC", str(a)).strip()
        a = norm(a)
        if not a:                                         # pure symbols, e.g. currency sign "؋" or "$"
            if raw and raw in unicodedata.normalize("NFKC", str(pred)):
                return True
            continue
        if re.fullmatch(r"\d+", a):                       # numbers: whole-token match
            if re.search(rf"(?<!\d){a}(?!\d)", p):
                return True
            continue
        if a in p:                                        # exact phrase inside answer
            return True
        toks = [t for t in a.split() if len(t) > 1]       # "Franklin D. Roosevelt" vs "Franklin Roosevelt"
        if len(toks) >= 2 and all(re.search(rf"\b{re.escape(t)}\b", p) for t in toks):
            return True
    return False


# ---------------------------------------------------------------- prompts
def build_items(data_path, per_category, seed):
    data = json.load(open(data_path))["data"]
    by_cat = defaultdict(list)
    for e in data:
        if any(str(e[k][0]).strip() == "" for k in ("A1", "A2", "Answer")):
            continue                      # a few items in the dataset have an empty gold answer
        by_cat[e["category"]].append(e)
    rng = random.Random(seed)
    items = []
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        if per_category and len(pool) > per_category:
            pool = rng.sample(pool, per_category)
        items.extend(pool)
    return items


def clear_question(e):
    """'the birthplace of X' usually means a city in English, but the gold answer is about the country.
    Make the composed question say 'country' explicitly, like the single-hop Q1 does."""
    p = e["person"]
    return e["Question"].replace(f"the birthplace of {p}", f"the country where {p} was born")


def ctx_question(e):
    """Composed question with the first-hop fact stated up front."""
    verb = "on" if e["category"] == "birthdate_uspresident" else "in"
    return (f"Background fact: {e['person']} was born {verb} {e['A1'][0]}.\n"
            f"Using this fact, answer the question: {clear_question(e)}")


def to_prompt(tok, question):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    if tok is None:  # dry run
        return question
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default="data/compositional_celebrities.json")
    ap.add_argument("--out", default="results/exp1_dry")
    ap.add_argument("--per-category", type=int, default=0, help="0 = use all items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = build_items(args.data, args.per_category, args.seed)
    questions = {}
    for e in items:
        for q in (e["Q1"], e["Q2"], e["Question"], clear_question(e), ctx_question(e)):
            questions.setdefault(q, None)
    uniq = list(questions)
    print(f"{len(items)} items, {len(uniq)} unique prompts", flush=True)

    t0 = time.time()
    if args.dry_run:
        # fake model: answers each hop correctly, composed questions correctly 60% of the time
        rng = random.Random(1)
        gold = {}
        for e in items:
            gold[e["Q1"]] = e["A1"][0]
            gold[e["Q2"]] = e["A2"][0]
            gold[e["Question"]] = e["Answer"][0] if rng.random() < 0.6 else "Unknown"
            gold[clear_question(e)] = e["Answer"][0] if rng.random() < 0.7 else e["A1"][0]
            gold[ctx_question(e)] = e["Answer"][0] if rng.random() < 0.9 else "Unknown"
        outs = [str(gold[q]) for q in uniq]
    else:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        llm = LLM(args.model, gpu_memory_utilization=0.85, max_model_len=1024, seed=args.seed)
        prompts = [to_prompt(tok, q) for q in uniq]
        res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
        outs = [r.outputs[0].text.strip() for r in res]
    answers = dict(zip(uniq, outs))
    print(f"generation took {time.time() - t0:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    with open(os.path.join(args.out, "items.jsonl"), "w") as f:
        for e in items:
            r = {
                "category": e["category"], "person": e["person"], "question": e["Question"],
                "gold": e["Answer"],
                "pred_q1": answers[e["Q1"]], "pred_q2": answers[e["Q2"]],
                "pred_comp": answers[e["Question"]], "pred_clear": answers[clear_question(e)],
                "pred_ctx": answers[ctx_question(e)],
            }
            r["q1"] = is_correct(r["pred_q1"], e["A1"])
            r["q2"] = is_correct(r["pred_q2"], e["A2"])
            r["comp"] = is_correct(r["pred_comp"], e["Answer"])        # original dataset wording
            r["clear"] = is_correct(r["pred_clear"], e["Answer"])      # unambiguous wording
            r["ctx"] = is_correct(r["pred_ctx"], e["Answer"])
            # "stopped halfway": gave the intermediate answer (hop 1) instead of the final one
            r["stop_hop1"] = (not r["clear"]) and is_correct(r["pred_clear"], e["A1"])
            r["ctx_echo"] = (not r["ctx"]) and is_correct(r["pred_ctx"], e["A1"])
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def summarize(rs):
        n = len(rs)
        both = [r for r in rs if r["q1"] and r["q2"]]
        s = {
            "n": n,
            "acc_q1": sum(r["q1"] for r in rs) / n,
            "acc_q2": sum(r["q2"] for r in rs) / n,
            "acc_comp": sum(r["comp"] for r in rs) / n,
            "acc_ctx": sum(r["ctx"] for r in rs) / n,
            "n_both_hops_correct": len(both),
        }
        if both:
            k = len(both)
            s["gap_original"] = 1 - sum(r["comp"] for r in both) / k   # knows both hops, fails to combine
            s["gap_clear"] = 1 - sum(r["clear"] for r in both) / k     # same, unambiguous wording (headline)
            s["gap_context"] = 1 - sum(r["ctx"] for r in both) / k     # still fails with hop-1 fact given
            s["stopped_at_hop1"] = sum(r["stop_hop1"] for r in both) / k
            s["context_echo"] = sum(r["ctx_echo"] for r in both) / k
        return s

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    cats = {c: summarize(rs) for c, rs in sorted(by_cat.items())}
    # headline: only categories where enough items have both hops right (small cells are too noisy)
    MIN_BOTH = 30
    kept = [c for c, s in cats.items() if s["n_both_hops_correct"] >= MIN_BOTH]
    headline = summarize([r for c in kept for r in by_cat[c]]) if kept else {}
    summary = {"model": args.model or "dry-run", "min_both_per_category": MIN_BOTH,
               "categories_in_headline": kept, "headline": headline,
               "all_items": summarize(rows), "by_category": cats}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2, ensure_ascii=False)

    print(f"\n{'category':30s} {'both':>5s} {'gap_orig':>9s} {'gap_clear':>9s} {'gap_ctx':>8s} {'stop@1':>7s}")
    for c, s in cats.items():
        if s["n_both_hops_correct"]:
            print(f"{c:30s} {s['n_both_hops_correct']:5d} {s['gap_original']:9.2f} {s['gap_clear']:9.2f} "
                  f"{s['gap_context']:8.2f} {s['stopped_at_hop1']:7.2f}")
    h = headline
    if h:
        print(f"\nGAP ({len(kept)} categories with >= {MIN_BOTH} usable items): "
              f"of {h['n_both_hops_correct']} items where both hops are known, "
              f"{h['gap_clear']:.1%} fail when composed (original wording: {h['gap_original']:.1%}); "
              f"{h['stopped_at_hop1']:.1%} stop at the intermediate answer; "
              f"{h['gap_context']:.1%} still fail with hop 1 given in context.")


if __name__ == "__main__":
    main()
