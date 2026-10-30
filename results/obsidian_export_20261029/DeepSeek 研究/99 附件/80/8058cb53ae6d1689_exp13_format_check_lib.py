"""Shared helpers copied from exp13_format_check.py."""
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

