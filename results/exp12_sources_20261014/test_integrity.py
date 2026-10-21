"""Meaningful local checks for pairing, answer NLL and failure safety."""
import argparse
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import analyze
import exp12
from exp1_gap_v2_v1 import to_prompt
from preflight import parse_quota


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(1)
    model_dir = a.smoke / "plan/tiny_model"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).eval()
    question, answer = "France country", "Paris Euro"
    prefix = tok(to_prompt(tok, question), add_special_tokens=False)["input_ids"]
    target = tok(answer, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prefix + target])
    labels = ids.clone()
    labels[:, :len(prefix)] = -100
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss.item()
    actual = exp12.score_answers(tok, model, [question], [answer])[0]
    if abs(expected - actual["mean_nll"]) > 1e-6 or actual["n_tokens"] != 2:
        raise AssertionError("Answer-only score differs from independent masked causal LM loss")
    if exp12.score_answers(tok, model, [question], [""])[0]["mean_nll"] is not None:
        raise AssertionError("Empty answer silently assigned likelihood")
    generated = exp12.generate(tok, model, [question, question], SimpleNamespace(gen_batch=2, smoke_test=True), scores=True)
    if generated[0] != generated[1]:
        raise AssertionError("Identical prompts differ within deterministic batch")
    if abs(analyze.paired([True] * 10, [False] * 10)["mcnemar_p"] - 2 / 1024) > 1e-12:
        raise AssertionError("Exact paired test incorrect")
    if parse_quota("USED=53.5GB LIMIT=100GB") != 46.5:
        raise AssertionError("Quota parser incorrect")
    for invalid in ["unknown", "USED=99GB LIMIT=100GB"]:
        try:
            parse_quota(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe quota accepted")
    if analyze.surface("-12") == analyze.surface("12") or analyze.surface("$") == analyze.surface("€"):
        raise AssertionError("Audit normalization erased semantic symbols")
    with tempfile.TemporaryDirectory(prefix="exp12-integrity-") as directory:
        copied = Path(directory) / "copy"
        shutil.copytree(a.smoke, copied)
        bad = copied / "context_s0/eval_D_reservedC.jsonl"
        rows = analyze.read(bad)
        rows[0]["composed_ok"] = not rows[0]["composed_ok"]
        bad.write_text("".join(json.dumps(r) + "\n" for r in rows))
        try:
            analyze.analyze(copied, Path(directory) / "bad_analysis")
        except ValueError as error:
            if "scoring mismatch" not in str(error):
                raise
        else:
            raise AssertionError("Corrupted saved label accepted")
    before = analyze.sha(a.smoke / "complete.json")
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("run_pipeline.py")),
        "--model", "unused", "--items", "unused", "--data", "unused", "--out", str(a.smoke)],
        text=True, capture_output=True)
    if result.returncode == 0 or "FileExistsError" not in result.stderr or analyze.sha(a.smoke / "complete.json") != before:
        raise AssertionError("Overwrite prevention failed")
    report = {"answer_nll_matches_masked_causal_loss": True, "empty_answer_handling": True,
              "deterministic_duplicate_prompts": True, "exact_mcnemar": True,
              "quota_fail_closed": True, "numeric_symbol_audit_safety": True,
              "corrupted_scoring_rejected": True, "existing_run_overwrite_rejected": True,
              "scope": "CPU tiny model and data integrity only; no 14B GPU execution"}
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
