"""Seconds-only file/quota checks. Safe on login node; no torch import."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_FREE_GIB = 6.0


def parse_quota(text):
    match = re.search(r"USED\s*=\s*([\d.]+)\s*([KMGT])B\s+LIMIT\s*=\s*([\d.]+)\s*([KMGT])B", text, re.I)
    if not match:
        raise ValueError("Unrecognized quota output; inspect lquota before submitting")
    factor = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}
    free = float(match[3]) * factor[match[4].upper()] - float(match[1]) * factor[match[2].upper()]
    if not math.isfinite(free) or free < MIN_FREE_GIB:
        raise ValueError(f"Quota free {free:.2f} GB; require {MIN_FREE_GIB} GB")
    return free


def verify_package():
    for line in (HERE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe manifest entry")
        p = HERE / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Package checksum mismatch: {name}")


def check_inputs(root):
    model = root / "models/Qwen3-14B"
    for rel in ["models/Qwen3-14B/config.json", "models/Qwen3-14B/tokenizer_config.json",
                "results/exp1v2_Qwen3-14B/items.jsonl", "data/compositional_celebrities.json"]:
        if not (root / rel).is_file():
            raise FileNotFoundError(root / rel)
    index = model / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(index)
    idx = json.loads(index.read_text())
    shards = sorted(set(idx["weight_map"].values()))
    for name in shards:
        if Path(name).name != name or not (model / name).is_file() or (model / name).stat().st_size == 0:
            raise ValueError(f"Missing or empty model shard: {name}")
    weight_bytes = sum((model / name).stat().st_size for name in shards)
    if weight_bytes < idx.get("metadata", {}).get("total_size", 0):
        raise ValueError("Model shards smaller than indexed tensor sizes")
    expected = json.loads((HERE / "input_hashes.json").read_text())
    for rel, digest in expected.items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Data changed since local validation: {rel}")
    versions = {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "numpy", "scipy", "accelerate"]}
    print("Input/model shards checked:", len(shards), "shards;", round(weight_bytes / 1024**3, 2), "GiB")
    print("Installed versions:", versions)
    return versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "inward")
    ap.add_argument("--quota-file", type=Path)
    ap.add_argument("--ticket", type=Path)
    ap.add_argument("--write-ticket", type=Path)
    a = ap.parse_args()
    verify_package()
    versions = check_inputs(a.root)
    quota_source = ""
    if a.quota_file:
        age = time.time() - a.quota_file.stat().st_mtime
        if age < -60 or age > 300:
            raise ValueError("Refresh quota capture immediately before submission (<5 minutes)")
        quota_text = a.quota_file.read_text()
        quota_source = str(a.quota_file)
    elif shutil.which("lquota"):
        result = subprocess.run(["lquota"], capture_output=True, text=True, check=True, timeout=15)
        quota_text = result.stdout + result.stderr
        quota_source = "live lquota"
    elif a.ticket:
        ticket = json.loads(a.ticket.read_text())
        if time.time() - ticket["checked_unix"] > 48 * 3600:
            raise ValueError("Submission quota ticket older than 48h; recheck and submit a new run")
        quota_text = ticket["quota_text"]
        quota_source = "login-node quota ticket; compute-node quota command unavailable"
    else:
        raise ValueError("No quota information available")
    free = parse_quota(quota_text)
    if shutil.disk_usage(a.root).free < MIN_FREE_GIB * 1024**3:
        raise ValueError("Filesystem has insufficient free space")
    print(f"Quota headroom: {free:.2f} GB; source: {quota_source}")
    if a.write_ticket:
        with a.write_ticket.open("x") as f:
            json.dump({"checked_unix": time.time(), "quota_text": quota_text,
                       "free_gb": free, "versions": versions}, f, indent=2)
    print("Preflight passed. No inference performed.")


if __name__ == "__main__":
    main()
