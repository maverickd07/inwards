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


def ctx_question(e):
    return f"Given that the answer to \"{e['Q1']}\" is {e['A1'][0]}.\n{e['Question']}"


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
        for q in (e["Q1"], e["Q2"], e["Question"], ctx_question(e)):
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
                "pred_comp": answers[e["Question"]], "pred_ctx": answers[ctx_question(e)],
            }
            r["q1"] = is_correct(r["pred_q1"], e["A1"])
            r["q2"] = is_correct(r["pred_q2"], e["A2"])
            r["comp"] = is_correct(r["pred_comp"], e["Answer"])
            r["ctx"] = is_correct(r["pred_ctx"], e["Answer"])
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
            s["gap"] = 1 - sum(r["comp"] for r in both) / len(both)        # knows both, fails to combine
            s["gap_with_context"] = 1 - sum(r["ctx"] for r in both) / len(both)
        return s

    summary = {"model": args.model or "dry-run", "overall": summarize(rows), "by_category": {}}
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for c, rs in sorted(by_cat.items()):
        summary["by_category"][c] = summarize(rs)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2, ensure_ascii=False)

    o = summary["overall"]
    print(json.dumps(o, indent=2))
    print(f"\nGAP: of {o['n_both_hops_correct']} items where both hops are known, "
          f"{o.get('gap', 0):.1%} fail when composed; "
          f"{o.get('gap_with_context', 0):.1%} still fail with hop 1 given in context.")


if __name__ == "__main__":
    main()
