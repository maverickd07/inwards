"""
Experiment 9: gold-free within-category target derangement and balanced filtering.
Uses frozen exp8 self targets for all arms. Resets training RNG before LoRA creation.
Production uses frozen exp8 targets and evaluations. Gold stratifies correctness and constrains shuffled labels; this is a supervised diagnostic.
Conditions: self_full, shuffled_full, category_confidence50, category_random50.
All outputs require a fresh directory; original exp8 code/results are untouched.
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

import sys
from pathlib import Path
for ancestor in Path(__file__).resolve().parents:
    candidates = [ancestor / "exp1_gap_v2.py", ancestor / "inward" / "exp1_gap_v2.py"]
    found = next((f for f in candidates if f.is_file()), None)
    if found is not None:
        sys.path.insert(0, str(found.parent))
        break
else:
    raise FileNotFoundError("Cannot locate original exp1_gap_v2.py")
from exp10_plan import make_plan, identity
from exp1_gap_v2 import clear_question, to_prompt
from scoring_v2 import is_correct

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


def diagnostic_auc(scores, labels):
    """Pairwise AUROC with half credit for ties; gold never selects targets."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


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
    ap.add_argument("--condition", required=True, choices=["self_wrong", "shuffled_wrong", "self_correct"])
    ap.add_argument("--selection-seed", type=int, default=None, help="independent Python RNG stream; defaults to train seed")
    ap.add_argument("--target-cache", help="frozen exp8 target_scores.jsonl; gold used only for controlled allocation; never as production training targets")
    ap.add_argument("--reference-evals", help="exp8 full-self run directory, required in production")
    ap.add_argument("--smoke-test", action="store_true", help="random tiny Qwen3 on CPU; reduced fixture only")
    args = ap.parse_args()
    if args.target != "self":
        ap.error("exp10 supports self-generated targets only")
    if args.selection_seed is None:
        args.selection_seed = args.train_seed
    if not args.smoke_test and (not args.target_cache or not args.reference_evals):
        ap.error("Production requires frozen --target-cache and --reference-evals from exp8")
    torch.manual_seed(args.train_seed)
    rng = random.Random(args.seed)
    rng_train = random.Random(args.train_seed)
    os.makedirs(args.out, exist_ok=False)
    json.dump(vars(args), open(os.path.join(args.out, "run_config.json"), "w"), indent=2)

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
        r["q1"] = is_correct(r["pred_q1"], r["e"]["A1"])
        r["q2"] = is_correct(r["pred_q2"], r["e"]["A2"])
    train = [r for r in rows if r["family"] in TRAIN_FAMILIES and r["pset"] == "train"]
    evals = {
        "A_newPerson": [r for r in rows if r["family"] in TRAIN_FAMILIES and r["pset"] == "test"],
        "B_newCategory": [r for r in rows if r["family"] in TEST_FAMILIES and r["pset"] == "train"],
        "C_bothNew": [r for r in rows if r["family"] in TEST_FAMILIES and r["pset"] == "test"],
    }
    evals = {k: rng.sample(v, min(args.n_eval, len(v))) for k, v in evals.items()}
    if args.reference_evals:
        reference_manifest = {}
        for name, rs in evals.items():
            path = Path(args.reference_evals) / f"eval_{name}.jsonl"
            reference = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if [identity(r) for r in reference] != [identity(r) for r in rs]:
                raise ValueError("Evaluation identity/order differs from exp8: " + name)
            reference_manifest[name] = reference
    json.dump({name: [identity(r) for r in rs] for name, rs in evals.items()}, open(os.path.join(args.out, "evaluation_manifest.json"), "w"), ensure_ascii=False, indent=2)
    print(f"train items {len(train)}; eval " + ", ".join(f"{k}={len(v)}" for k, v in evals.items()), flush=True)

    if args.smoke_test:
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        vocab = {w: i for i, w in enumerate(["[PAD]", "[EOS]", "[UNK]", "Answer", "France", "Nigeria", "Paris", "12", "Euro", "fact", "country", "born"])}
        backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
        tok.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}Answer "
        # Keep the synthetic base identical across training seeds.
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=512, pad_token_id=0, eos_token_id=1)).float()
        # Four examples per relation allow derangements and equal per-category half selection.
        smoke_counts = {}
        smoke_train = []
        for r in train:
            cat = r["category"]
            if smoke_counts.get(cat, 0) < 4:
                smoke_train.append(r)
                smoke_counts[cat] = smoke_counts.get(cat, 0) + 1
        train = smoke_train
        # Synthetic fixture has both correctness classes; not scientific model outputs.
        evals = {k: v[:2] for k, v in evals.items()}
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if not args.smoke_test:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    dev = model.device

    @torch.no_grad()
    def generate(questions, bs=32, with_scores=False):
        outs, scores = [], []
        for i in range(0, len(questions), bs):
            prompts = [to_prompt(tok, q) for q in questions[i:i + bs]]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=pad_id)
            if with_scores:
                # Teacher-forced raw model log probabilities on the exact greedy continuation.
                # Ignore EOS and other special/padding tokens, including padding after early EOS.
                width = enc["input_ids"].shape[1]
                continuation = g[:, width:]
                mask = torch.ones_like(continuation, dtype=torch.bool)
                for special in tok.all_special_ids:
                    mask &= continuation != special
                if tok.eos_token_id is not None:
                    mask &= (continuation == tok.eos_token_id).cumsum(dim=1) == 0
                valid_context = torch.ones_like(continuation, dtype=torch.long)
                if tok.eos_token_id is not None:
                    eos = continuation == tok.eos_token_id
                    valid_context = ((eos.cumsum(dim=1) - eos.long()) == 0).long()
                full_att = torch.cat([enc["attention_mask"], valid_context], dim=1)
                logits = model(input_ids=g, attention_mask=full_att).logits[:, width - 1:-1].float()
                lp = logits.log_softmax(-1).gather(-1, continuation.unsqueeze(-1)).squeeze(-1)
                scores += [(float(lp[j][mask[j]].mean()) if mask[j].any() else None) for j in range(len(g))]
            outs += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in g]
        return (outs, scores) if with_scores else outs

    # ------------------------------------------------------------ build training targets
    t0 = time.time()
    if args.target_cache:
        cache_path = Path(args.target_cache)
        cached = [json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
        cache_by_key = {identity(r): r for r in cached}
        if len(cache_by_key) != len(cached):
            raise ValueError("Duplicate cached identity")
        if not args.smoke_test and set(cache_by_key) != {identity(r) for r in train}:
            raise ValueError("Frozen target pool differs from training pool")
        targets = [cache_by_key[identity(r)]["target"] for r in train]
        confidence = [cache_by_key[identity(r)]["mean_logprob"] for r in train]
        cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    else:
        targets, confidence = generate([self_ctx_question(r["e"], r["pred_q1"]) for r in train], with_scores=True)
        cache_hash = None
    if args.smoke_test:
        targets = [r["e"]["Answer"][0] if i % 4 < 2 else "WRONG_FIXTURE_" + str(i) for i,r in enumerate(train)]
        confidence = [-1.0] * len(train)
        cache_hash = "synthetic-smoke-only"
    donors, kept, category_counts = make_plan(train, targets, confidence, args.condition, args.selection_seed, is_correct)
    original_targets = targets.copy()
    targets = [original_targets[j] for j in donors]
    train_ex = [(clear_question(r["e"]), targets[i]) for i, r in enumerate(train) if i in kept]
    src = args.condition + ": frozen self outputs; gold-stratified mechanism diagnostic"
    # Gold also determines allocation; no gold answer becomes a production target.
    labels = [is_correct(t, r["e"]["Answer"]) for r, t in zip(train, targets)]
    original_labels = [is_correct(t, r["e"]["Answer"]) for r, t in zip(train, original_targets)]
    changed = sum(targets[i] != original_targets[i] for i in kept)
    per_category = {}
    for cat, counts in category_counts.items():
        ids = [i for i, r in enumerate(train) if r["category"] == cat and i in kept]
        per_category[cat] = {**counts, "target_changed_n": sum(targets[i] != original_targets[i] for i in ids), "assigned_target_correct_n": sum(labels[i] for i in ids), "original_target_correct_n": sum(original_labels[i] for i in ids)}
    diagnostic = {"condition": args.condition, "target_cache_sha256": cache_hash, "n_pool": len(train), "n_kept": len(kept), "target_changed_n": changed, "target_changed_share": changed / len(kept), "assigned_target_correct_n": sum(labels[i] for i in kept), "assigned_target_correct_share": sum(labels[i] for i in kept) / len(kept), "original_pool_target_correct_share": sum(original_labels) / len(train), "original_target_auroc": diagnostic_auc(confidence, original_labels), "uses_gold_for_allocation": True, "allocation_scorer": "scoring_v2_floor", "subset_seed": 7, "category_counts": per_category}
    json.dump(diagnostic, open(os.path.join(args.out, "target_diagnostic.json"), "w"), indent=2)
    print("Allocation diagnostic BEFORE training:", diagnostic, flush=True)
    with open(os.path.join(args.out, "target_assignment.jsonl"), "w") as f:
        for i, r in enumerate(train):
            j = donors[i]
            f.write(json.dumps({"category": r["category"], "person": r["person"], "donor_person": train[j]["person"], "original_target": original_targets[i], "assigned_target": targets[i], "original_mean_logprob": confidence[i], "kept": i in kept, "diagnostic_assigned_target_ok": labels[i]}, ensure_ascii=False) + "\n")
    agree = sum(labels[i] for i in kept) / len(kept)
    with open(os.path.join(args.out, "train_examples.jsonl"), "w") as f:
        for q, t in train_ex:
            f.write(json.dumps({"question": q, "target": t}, ensure_ascii=False) + "\n")
    # Allocation and target loading must not affect LoRA initialization/dropout RNG.
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)

    # ------------------------------------------------------------ LoRA training
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    init_hash = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            init_hash.update(name.encode())
            init_hash.update(parameter.detach().float().cpu().numpy().tobytes())
    json.dump({"train_seed": args.train_seed, "lora_initial_sha256": init_hash.hexdigest()}, open(os.path.join(args.out, "initialization.json"), "w"), indent=2)
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
        sub = rs  # exp10 evaluates both single hops on every sampled evaluation item
        q1 = generate([r["e"]["Q1"] for r in sub])
        q2 = generate([r["e"]["Q2"] for r in sub])
        return {"pred_q1": q1, "pred_q2": q2, "comp": [is_correct(p, r["e"]["Answer"]) for p, r in zip(comp, rs)],
                "q1": [is_correct(p, r["e"]["A1"]) for p, r in zip(q1, sub)],
                "q2": [is_correct(p, r["e"]["A2"]) for p, r in zip(q2, sub)],
                "pred": comp}

    results = {}
    for name, rs in evals.items():
        with model.disable_adapter():
            base = evaluate(rs)
        trained = evaluate(rs)
        results[name] = (rs, base, trained)
        if args.reference_evals:
            ref = reference_manifest[name]
            reference_check = {"subset": name, "identity_order_match": True, "base_prediction_disagreements": sum(r["base_pred"] != p for r, p in zip(ref, base["pred"])), "base_correctness_disagreements": sum(r["base_ok"] != ok for r, ok in zip(ref, base["comp"]))}
            with open(os.path.join(args.out, "reference_base_check.jsonl"), "a") as f:
                f.write(json.dumps(reference_check) + "\n")

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

    # Exclude shared bridge/final first tokens BEFORE selecting the lens sample.
    lens_pool = [r for r in evals["C_bothNew"] if not set(first_ids(r["e"]["A1"])) & set(first_ids(r["e"]["Answer"]))]
    if not lens_pool:
        # Tiny synthetic tokenizer may collapse all unseen entities to UNK.
        lens_items = []
    else:
        lens_items = rng.sample(lens_pool, min(args.n_lens, len(lens_pool)))
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
    summary = {"condition": args.condition, "selection_seed": args.selection_seed, "smoke_test": args.smoke_test, "target": args.target, "n_train": len(train_ex), "target_correct_share": agree, "evals": {}}
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
                                    "base_q1_pred": b["pred_q1"][i] if i < len(b["pred_q1"]) else None,
                                    "trained_q1_pred": t["pred_q1"][i] if i < len(t["pred_q1"]) else None,
                                    "base_q2_pred": b["pred_q2"][i] if i < len(b["pred_q2"]) else None,
                                    "trained_q2_pred": t["pred_q2"][i] if i < len(t["pred_q2"]) else None,
                                    "base_ok": b["comp"][i], "trained_ok": t["comp"][i]}, ensure_ascii=False) + "\n")

    def lens_stats(v):
        hit = [x for x in v if x is not None]
        return (len(hit) / len(v) if v else None), (statistics.median(hit) if hit else float("nan"))
    hb, mb = lens_stats(lb)
    ha, ma = lens_stats(la)
    both = [(x, y) for x, y in zip(lb, la) if x is not None and y is not None]
    earlier = sum(y < x for x, y in both)
    later = sum(y > x for x, y in both)
    print(f"\n[logit lens, C_bothNew, n={len(lens_items)}] bridge entity enters top-10 at layer (of {L}):")
    print(f"  base    : reaches top-10 in {hb} of items, median first layer {mb}")
    print(f"  trained : reaches top-10 in {ha} of items, median first layer {ma}")
    print(f"  per item: earlier after training {earlier}, later {later}, same {len(both) - earlier - later} "
          f"(sign test p = {mcnemar_p(earlier, later):.2g})")
    summary["lens"] = {"n": len(lens_items), "excluded_token_overlap_n": len(evals["C_bothNew"]) - len(lens_pool), "base_hit": hb, "base_median_layer": mb, "trained_hit": ha, "trained_median_layer": ma,
                       "earlier": earlier, "later": later, "p": mcnemar_p(earlier, later)}
    json.dump(summary, open(os.path.join(args.out, "inward_summary.json"), "w"), indent=2)
    print("\nHow to read: C_bothNew is the main result. Success = composed accuracy up with small p,"
          "\ngap down, single-hop accuracy NOT down, mechanism requires separate direct evidence; earlier bridge activation is not required.")


if __name__ == "__main__":
    main()
