"""
Experiment 3: back-patching. Can we FIX composition failures by giving hop 2 more layers?

Experiment 2 showed that on failed composed questions the bridge entity (birth country) is
usually resolved internally, but only around layer 34-35 of 40, leaving few layers for hop 2.
Here we take the residual stream at the last prompt position from a LATER source layer s and
write it into an EARLIER target layer t (t < s) of a fresh forward pass on the same prompt,
so the later computation gets to run again with more layers left.

Control: the same patch but with the hidden state taken from a DIFFERENT item (random donor).
If the control "fixes" as many items as the real patch, the effect is noise, not a fix.
Because we try many (s, t) pairs per item, both real and control are scored best-of-grid,
so they get exactly the same number of chances.

Steps per item (clean categories only, model fails the composed question but knows both hops):
  1. one forward pass, store the last-position hidden state at every source layer s
  2. for every target layer t: one batched forward pass whose rows are the real patch (each s>t)
     and the control patch (donor's state at the same s); record whether the top-1 next token
     is the first token of the correct answer
  3. for the best real pair and the best control pair: greedy-generate the full answer with the
     patch applied and check it with the same scorer as experiment 1 (verified fix)

Usage:
  python exp3_backpatch.py --model models/Qwen3-14B \
      --items results/exp1v2_Qwen3-14B/items.jsonl --out results/patch_Qwen3-14B
"""
import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp1_gap_v2 import clear_question, is_correct, to_prompt

SOURCES = [24, 28, 32, 34, 36, 38]
TARGETS = [4, 8, 12, 16, 20, 24, 28, 32]


def is_clean(c):
    return c.startswith("birthplace") and "common_name" not in c


def first_ids(tok, aliases):
    ids = set()
    for a in aliases:
        a = str(a).strip()
        if not a:
            continue
        for s in (a, " " + a):
            t = tok.encode(s, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--data", default="data/compositional_celebrities.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.load(open(args.data))["data"]
    by_key = {(e["category"], e["person"], e["Question"]): e for e in data}
    rows = [json.loads(line) for line in open(args.items)]
    pool = [r for r in rows if is_clean(r["category"]) and r["q1"] and r["q2"] and not r["clear"]]
    rng = random.Random(args.seed)
    picked = rng.sample(pool, min(args.n, len(pool)))
    items = [(r, by_key[(r["category"], r["person"], r["question"])]) for r in picked
             if (r["category"], r["person"], r["question"]) in by_key]
    print(f"{len(items)} failed clean items", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.layers
    L = model.config.num_hidden_layers
    pairs = [(s, t) for s in SOURCES for t in TARGETS if t < s and s < L]

    patch = {"vecs": None}           # rows x hidden, applied at the last prompt position

    def make_hook():
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if patch["vecs"] is not None and h.shape[1] > 1:          # prefill only, not decode steps
                h[:, -1, :] = patch["vecs"].to(h.dtype)
            return out
        return hook

    def encode(e):
        return tok(to_prompt(tok, clear_question(e)), return_tensors="pt",
                   add_special_tokens=False).to(model.device)

    # ---- pass 1: source states for every item
    t0 = time.time()
    states = []
    with torch.no_grad():
        for _, e in items:
            hs = model(**encode(e), output_hidden_states=True).hidden_states
            states.append({s: hs[s][0, -1].clone() for s in SOURCES if s < L})
    donors = list(range(len(items)))
    rng.shuffle(donors)
    donors = [d if d != i else (i + 1) % len(items) for i, d in enumerate(donors)]   # never self
    print(f"source states in {time.time() - t0:.0f}s", flush=True)

    # ---- pass 2: first-token test over the (s, t) grid, real vs control, batched by target layer
    grid_real = {p: 0 for p in pairs}
    grid_ctrl = {p: 0 for p in pairs}
    recs = []
    with torch.no_grad():
        for i, (r, e) in enumerate(items):
            enc = encode(e)
            ans = first_ids(tok, e["Answer"])
            base_top = int(model(**enc).logits[0, -1].argmax())
            real_hits, ctrl_hits = [], []
            for t in TARGETS:
                srcs = [s for s in SOURCES if t < s < L]
                if not srcs:
                    continue
                vecs = torch.stack([states[i][s] for s in srcs] + [states[donors[i]][s] for s in srcs])
                n = vecs.shape[0]
                h = layers[t - 1].register_forward_hook(make_hook())     # output of layer t = hidden_states[t]
                patch["vecs"] = vecs
                try:
                    logits = model(input_ids=enc["input_ids"].expand(n, -1),
                                   attention_mask=enc["attention_mask"].expand(n, -1)).logits[:, -1].float()
                finally:
                    patch["vecs"] = None
                    h.remove()
                top = logits.argmax(-1).tolist()
                for j, s in enumerate(srcs):
                    if top[j] in ans:
                        real_hits.append((s, t, float(logits[j].softmax(-1)[top[j]])))
                        grid_real[(s, t)] += 1
                    if top[len(srcs) + j] in ans:
                        ctrl_hits.append((s, t, float(logits[len(srcs) + j].softmax(-1)[top[len(srcs) + j]])))
                        grid_ctrl[(s, t)] += 1
            recs.append({"r": r, "e": e, "i": i, "base_first_ok": base_top in ans,
                         "real_hits": real_hits, "ctrl_hits": ctrl_hits})
            if (i + 1) % 50 == 0:
                print(f"  grid {i + 1}/{len(items)}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- pass 3: verify the best pair by generating the full answer with the patch on
    def generate_patched(e, vec, t):
        enc = encode(e)
        h = layers[t - 1].register_forward_hook(make_hook())
        patch["vecs"] = vec.unsqueeze(0)
        try:
            out = model.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        finally:
            patch["vecs"] = None
            h.remove()
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    os.makedirs(args.out, exist_ok=True)
    n_base = n_real_first = n_ctrl_first = n_real_ver = n_ctrl_ver = 0
    with torch.no_grad(), open(os.path.join(args.out, "patch_items.jsonl"), "w") as f:
        for rec in recs:
            r, e, i = rec["r"], rec["e"], rec["i"]
            n_base += rec["base_first_ok"]
            out = {"category": r["category"], "person": r["person"], "bridge": e["A1"][0],
                   "gold": e["Answer"][0], "pred_unpatched": r["pred_clear"],
                   "base_first_ok": rec["base_first_ok"], "n_real_pairs": len(rec["real_hits"]),
                   "n_ctrl_pairs": len(rec["ctrl_hits"])}
            if rec["base_first_ok"]:            # already starts right without patching: not a fixable failure
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue
            for kind, hits, src in (("real", rec["real_hits"], i), ("ctrl", rec["ctrl_hits"], donors[i])):
                if not hits:
                    continue
                s, t, _ = max(hits, key=lambda x: x[2])
                pred = generate_patched(e, states[src][s], t)
                ok = is_correct(pred, e["Answer"])
                out[f"{kind}_pair"], out[f"{kind}_pred"], out[f"{kind}_ok"] = [s, t], pred, ok
                if kind == "real":
                    n_real_first += 1
                    n_real_ver += ok
                else:
                    n_ctrl_first += 1
                    n_ctrl_ver += ok
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    n = len(recs)
    m = n - n_base
    print(f"\ndone in {time.time() - t0:.0f}s")
    print("=" * 90)
    print(f"Failed clean items: {n}   already correct first token without patch: {n_base} -> fixable set: {m}")
    print(f"                              first-token fix (best of grid)   verified full-answer fix")
    print(f"  REAL back-patch            {n_real_first:5d}  ({n_real_first / m:6.1%})           "
          f"{n_real_ver:5d}  ({n_real_ver / m:6.1%})")
    print(f"  CONTROL (other item's state){n_ctrl_first:5d}  ({n_ctrl_first / m:6.1%})           "
          f"{n_ctrl_ver:5d}  ({n_ctrl_ver / m:6.1%})")

    print("\nFirst-token fix rate per (source s -> target t), REAL / CONTROL, over all items:")
    print("  s\\t  " + " ".join(f"{t:>11d}" for t in TARGETS))
    for s in SOURCES:
        if s >= L:
            continue
        cells = []
        for t in TARGETS:
            cells.append(f"{grid_real[(s, t)] / n:5.0%}/{grid_ctrl[(s, t)] / n:<5.0%}" if (s, t) in grid_real
                         else " " * 11)
        print(f"  {s:>3d}  " + " ".join(cells))

    summary = {"model": args.model, "n_items": n, "n_base_first_ok": n_base, "fixable": m,
               "real_first_token_fix": n_real_first / m, "ctrl_first_token_fix": n_ctrl_first / m,
               "real_verified_fix": n_real_ver / m, "ctrl_verified_fix": n_ctrl_ver / m,
               "grid_real": {f"{s}->{t}": v / n for (s, t), v in grid_real.items()},
               "grid_ctrl": {f"{s}->{t}": v / n for (s, t), v in grid_ctrl.items()}}
    json.dump(summary, open(os.path.join(args.out, "patch_summary.json"), "w"), indent=2)
    print("\nHow to read: the headline is the VERIFIED fix rate, REAL vs CONTROL. A real effect means"
          "\nREAL is clearly above CONTROL. Grid cells show where (which layers) the fix comes from.")


if __name__ == "__main__":
    main()
