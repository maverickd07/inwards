"""One serial job; never resumes into or overwrites an old result directory."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--smoke-test", action="store_true")
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    commands = []
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in HERE.iterdir() if p.suffix in [".py", ".txt", ".sh", ".json"]}
    (a.out / "source_hashes.json").write_text(json.dumps(hashes, indent=2))

    def run(stage, dest, **kwargs):
        cmd = [sys.executable, "-u", str(HERE / "exp12.py"), "--stage", stage, "--out", str(dest)]
        for name, val in kwargs.items():
            cmd += ["--" + name.replace("_", "-"), str(val)]
        if a.smoke_test:
            cmd.append("--smoke-test")
        commands.append(cmd)
        (a.out / "commands.json").write_text(json.dumps(commands, indent=2))
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    plan = a.out / "plan"
    run("plan", plan, items=a.items, data=a.data)
    model = plan / "tiny_model" if a.smoke_test else a.model
    common = {"model": model, "plan": plan, "gen_batch": a.gen_batch}
    run("targets", a.out / "paired_targets", **common)
    run("baseline", a.out / "baseline", **common)
    stages = []
    for seed in [0, 1, 2]:
        for condition in ["context", "direct"]:
            out = a.out / f"{condition}_s{seed}"
            run("train", out, **common, condition=condition, train_seed=seed,
                targets=a.out / "paired_targets" / f"{condition}_targets.jsonl")
            stages.append({"condition": condition, "seed": seed, "path": out.name})
            (a.out / "stages.json").write_text(json.dumps(stages, indent=2))
    subprocess.run([sys.executable, str(HERE / "analyze.py"), "--runs", str(a.out),
                    "--out", str(a.out / "analysis")], check=True)
    (a.out / "complete.json").write_text(json.dumps({"training_runs": 6,
        "smoke_test": a.smoke_test, "primary_split": "D_reservedC",
        "seconds": time.time() - started, "manual_flip_review": "pending",
        "next_experiment_authorized": False}, indent=2))
    print("EXP12 COMPUTATION COMPLETE. Manual semantic audit remains pending.", flush=True)


if __name__ == "__main__":
    main()
