"""
Experiment 6: INWARD learning. Can the model teach itself to use its own knowledge on composed questions?

Training signal comes from the model itself (target=self):
  1. the model's own answer to hop 1 ("Which country was X born in?")        -> pred_q1 from experiment 1
  2. that answer is stated as a background fact in the composed question; the model's answer there
     becomes the training target for the SAME composed question WITHOUT the fact.
  No gold answers are used for training. (Context distillation, applied to the composition gap.)
Control (target=gold): same pipeline but the target is the dataset's gold answer (external teacher).

Generalisation is tested on a strict split by category FAMILY and by PERSON:
  train : TRAIN families x TRAIN persons
  eval A: TRAIN families x TEST persons   (new people)
  eval B: TEST families  x TRAIN persons  (new kinds of question)
  eval C: TEST families  x TEST persons   (both new -> main result)
For every eval set: composed accuracy before vs after (same process, adapter disabled vs enabled),
also restricted to items whose two hops the base model knows (= the composition gap), plus
single-hop accuracy (does training damage what the model already knew?) and a logit-lens check of
WHERE the bridge entity appears before vs after training.

Usage:
  python exp6_inward.py --model models/Qwen3-14B --items results/exp1v2_Qwen3-14B/items.jsonl \
      --target self --out results/inward_self_Qwen3-14B
  python exp6_inward.py ... --target gold --out results/inward_gold_Qwen3-14B
"""
import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp1_gap_v2 import clear_question, is_correct, to_prompt

FAMILIES = {
    "capital": ["birthplace_capital"],
    "callingcode": ["birthplace_callingcode"],
    "latlng": ["birthplace_rounded_lat", "birthplace_rounded_lng"],
    "currency": ["birthplace_currency", "birthplace_currency_short", "birthplace_currency_symbol"],
    "tld": ["birthplace_tld"],
    "ccn3": ["birthplace_ccn3"],
}
TRAIN_FAMILIES = ["capital", "callingcode", "latlng"]
TEST_FAMILIES = ["currency", "tld", "ccn3"]


def person_split(name):
    return "train" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 2 == 0 else "test"


def self_ctx_question(e, fact):
    return (f"Background fact: {e['person']} was born in {fact}.\n"
            f"Using this fact, answer the question: {clear_question(e)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--data", default="data/compositional_celebrities.json")
    ap.add_argument("--target", choices=["self", "gold", "hops"], default="self")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--n-lens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0, help="fixes the eval/lens item samples")
    ap.add_argument("--train-seed", type=int, default=0, help="change this to re-run training with a new seed")
    args = ap.parse_args()
    torch.manual_seed(args.train_seed)
    rng = random.Random(args.seed)
    rng_train = random.Random(args.train_seed)
    os.makedirs(args.out, exist_ok=True)

    fam_of = {c: f for f, cs in FAMILIES.items() for c in cs}
    data = json.load(open(args.data))["data"]
    by_cp = {}
    for e in data:
        by_cp.setdefault((e["category"], e["person"]), e)
    rows = [r for r in map(json.loads, open(args.items))
            if r["category"] in fam_of and (r["category"], r["person"]) in by_cp]
    for r in rows:
        r["family"] = fam_of[r["category"]]
        r["pset"] = person_split(r["person"])
        r["e"] = by_cp[(r["category"], r["person"])]
    train = [r for r in rows if r["family"] in TRAIN_FAMILIES and r["pset"] == "train"]
    evals = {
        "A_newPerson": [r for r in rows if r["family"] in TRAIN_FAMILIES and r["pset"] == "test"],
        "B_newCategory": [r for r in rows if r["family"] in TEST_FAMILIES and r["pset"] == "train"],
        "C_bothNew": [r for r in rows if r["family"] in TEST_FAMILIES and r["pset"] == "test"],
    }
    evals = {k: rng.sample(v, min(args.n_eval, len(v))) for k, v in evals.items()}
    print(f"train items {len(train)}; eval " + ", ".join(f"{k}={len(v)}" for k, v in evals.items()), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    dev = model.device

    @torch.no_grad()
    def generate(questions, bs=32):
        outs = []
        for i in range(0, len(questions), bs):
            prompts = [to_prompt(tok, q) for q in questions[i:i + bs]]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=pad_id)
            outs += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in g]
        return outs

    # ------------------------------------------------------------ build training targets
    t0 = time.time()
    if args.target == "self":
        targets = generate([self_ctx_question(r["e"], r["pred_q1"]) for r in train])
        src = "model's own answer with its own hop-1 answer stated in context (no gold used)"
        train_ex = [(clear_question(r["e"]), t) for r, t in zip(train, targets) if t]
    elif args.target == "gold":
        targets = [str(r["e"]["Answer"][0]) for r in train]
        src = "gold answer from the dataset (external teacher)"
        train_ex = [(clear_question(r["e"]), t) for r, t in zip(train, targets) if t]
    else:
        # control: same number of examples, same self-generated labels, but ONLY single-hop questions
        # (half hop-1, half hop-2), never a composed question. Tests whether gains come from answer format.
        qs = [r["e"]["Q1"] if i % 2 == 0 else r["e"]["Q2"] for i, r in enumerate(train)]
        targets = generate(qs)
        src = "CONTROL: model's own answers to SINGLE-HOP questions only (no composed questions, no gold)"
        train_ex = [(q, t) for q, t in zip(qs, targets) if t]
    agree = (sum(is_correct(t, r["e"]["Answer"]) for r, t in zip(train, targets)) / max(len(train), 1)
             if args.target != "hops" else float("nan"))
    print(f"targets: {src}; {len(train_ex)} examples in {time.time() - t0:.0f}s; "
          f"share of targets that happen to be correct: {agree:.1%}", flush=True)
    with open(os.path.join(args.out, "train_examples.jsonl"), "w") as f:
        for q, t in train_ex:
            f.write(json.dumps({"question": q, "target": t}, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ LoRA training
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_total = args.epochs * math.ceil(len(train_ex) / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps_total))

    def collate(batch):
        ids, labels = [], []
        for q, t in batch:
            p = tok(to_prompt(tok, q), add_special_tokens=False)["input_ids"]
            a = tok(str(t) + tok.eos_token, add_special_tokens=False)["input_ids"]
            ids.append(p + a)
            labels.append([-100] * len(p) + a)
        m = max(map(len, ids))
        input_ids = torch.full((len(ids), m), pad_id)
        lab = torch.full((len(ids), m), -100)
        att = torch.zeros((len(ids), m), dtype=torch.long)
        for i, (x, y) in enumerate(zip(ids, labels)):                 # right padding for training
            input_ids[i, :len(x)] = torch.tensor(x)
            lab[i, :len(y)] = torch.tensor(y)
            att[i, :len(x)] = 1
        return input_ids.to(dev), att.to(dev), lab.to(dev)

    model.train()
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        rng_train.shuffle(train_ex)
        losses = []
        for i in range(0, len(train_ex), args.batch):
            ids, att, lab = collate(train_ex[i:i + args.batch])
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                print(f"  epoch {ep + 1} step {step}/{steps_total} loss {statistics.mean(losses[-50:]):.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.save_pretrained(os.path.join(args.out, "adapter"))
    print(f"training done in {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------ evaluation: base (adapter off) vs trained
    def evaluate(rs):
        comp = generate([clear_question(r["e"]) for r in rs])
        sub = rs[:150]
        q1 = generate([r["e"]["Q1"] for r in sub])
        q2 = generate([r["e"]["Q2"] for r in sub])
        return {"comp": [is_correct(p, r["e"]["Answer"]) for p, r in zip(comp, rs)],
                "q1": [is_correct(p, r["e"]["A1"]) for p, r in zip(q1, sub)],
                "q2": [is_correct(p, r["e"]["A2"]) for p, r in zip(q2, sub)],
                "pred": comp}

    results = {}
    for name, rs in evals.items():
        with model.disable_adapter():
            base = evaluate(rs)
        trained = evaluate(rs)
        results[name] = (rs, base, trained)

    # ------------------------------------------------------------ logit lens: where does the bridge appear?
    base_lm = model.get_base_model()
    norm, head = base_lm.model.norm, base_lm.lm_head
    L = base_lm.config.num_hidden_layers

    def first_ids(aliases):
        ids = set()
        for a in aliases:
            for s in (str(a).strip(), " " + str(a).strip()):
                t = tok.encode(s, add_special_tokens=False)
                if t:
                    ids.add(t[0])
        return sorted(ids)

    @torch.no_grad()
    def bridge_layer(e):
        enc = tok(to_prompt(tok, clear_question(e)), return_tensors="pt", add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        ids = first_ids(e["A1"])
        for l in range(1, L + 1):
            h = hs[l][0, -1] if l == L else norm(hs[l][0, -1])
            logits = head(h).float()
            if min(int((logits > logits[i]).sum()) for i in ids) < 10:
                return l
        return None

    lens_items = rng.sample(evals["C_bothNew"], min(args.n_lens, len(evals["C_bothNew"])))
    with model.disable_adapter():
        lb = [bridge_layer(r["e"]) for r in lens_items]
    la = [bridge_layer(r["e"]) for r in lens_items]

    # ------------------------------------------------------------ report
    def mcnemar_p(b, c):
        n = b + c
        if n == 0:
            return 1.0
        return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

    print("\n" + "=" * 100)
    print(f"TARGET = {args.target.upper()}  ({src})")
    summary = {"target": args.target, "n_train": len(train_ex), "target_correct_share": agree, "evals": {}}
    for name, (rs, b, t) in results.items():
        n = len(rs)
        known = [i for i, r in enumerate(rs) if r["q1"] and r["q2"]]
        acc = lambda v, idx=None: sum(v[i] for i in (idx if idx is not None else range(len(v)))) / max(len(idx if idx is not None else v), 1)
        gap_b = 1 - acc(b["comp"], known)
        gap_t = 1 - acc(t["comp"], known)
        fixed = sum(t["comp"][i] and not b["comp"][i] for i in range(n))
        broke = sum(b["comp"][i] and not t["comp"][i] for i in range(n))
        p = mcnemar_p(fixed, broke)
        print(f"\n[{name}]  n={n}  (both hops known by base model: {len(known)})")
        print(f"  composed accuracy      base {acc(b['comp']):6.1%}  ->  trained {acc(t['comp']):6.1%}   "
              f"(fixed {fixed}, broke {broke}, McNemar p = {p:.2g})")
        print(f"  composition gap        base {gap_b:6.1%}  ->  trained {gap_t:6.1%}   (lower is better)")
        print(f"  hop-1 alone accuracy   base {acc(b['q1']):6.1%}  ->  trained {acc(t['q1']):6.1%}   (should not drop)")
        print(f"  hop-2 alone accuracy   base {acc(b['q2']):6.1%}  ->  trained {acc(t['q2']):6.1%}   (should not drop)")
        summary["evals"][name] = {"n": n, "n_known": len(known), "acc_base": acc(b["comp"]), "acc_trained": acc(t["comp"]),
                                  "gap_base": gap_b, "gap_trained": gap_t, "fixed": fixed, "broke": broke, "p": p,
                                  "q1_base": acc(b["q1"]), "q1_trained": acc(t["q1"]),
                                  "q2_base": acc(b["q2"]), "q2_trained": acc(t["q2"])}
        with open(os.path.join(args.out, f"eval_{name}.jsonl"), "w") as f:
            for i, r in enumerate(rs):
                f.write(json.dumps({"category": r["category"], "person": r["person"], "gold": r["e"]["Answer"][0],
                                    "base_pred": b["pred"][i], "trained_pred": t["pred"][i],
                                    "base_ok": b["comp"][i], "trained_ok": t["comp"][i]}, ensure_ascii=False) + "\n")

    def lens_stats(v):
        hit = [x for x in v if x is not None]
        return len(hit) / len(v), (statistics.median(hit) if hit else float("nan"))
    hb, mb = lens_stats(lb)
    ha, ma = lens_stats(la)
    both = [(x, y) for x, y in zip(lb, la) if x is not None and y is not None]
    earlier = sum(y < x for x, y in both)
    later = sum(y > x for x, y in both)
    print(f"\n[logit lens, C_bothNew, n={len(lens_items)}] bridge entity enters top-10 at layer (of {L}):")
    print(f"  base    : reaches top-10 in {hb:.0%} of items, median first layer {mb}")
    print(f"  trained : reaches top-10 in {ha:.0%} of items, median first layer {ma}")
    print(f"  per item: earlier after training {earlier}, later {later}, same {len(both) - earlier - later} "
          f"(sign test p = {mcnemar_p(earlier, later):.2g})")
    summary["lens"] = {"base_hit": hb, "base_median_layer": mb, "trained_hit": ha, "trained_median_layer": ma,
                       "earlier": earlier, "later": later, "p": mcnemar_p(earlier, later)}
    json.dump(summary, open(os.path.join(args.out, "inward_summary.json"), "w"), indent=2)
    print("\nHow to read: C_bothNew is the main result. Success = composed accuracy up with small p,"
          "\ngap down, single-hop accuracy NOT down, and the bridge entity appearing at an EARLIER layer.")


if __name__ == "__main__":
    main()
