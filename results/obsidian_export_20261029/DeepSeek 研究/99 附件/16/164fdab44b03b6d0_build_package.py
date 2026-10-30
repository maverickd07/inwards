"""Create the explicit upload allowlist, checksum manifest and archive once."""
import hashlib
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["exp12.py", "engine_exp11_v1.py", "exp1_gap_v2_v1.py", "scoring_v2_v1.py",
         "analyze.py", "run_pipeline.py", "preflight.py", "submit.sh", "run_exp12.sh",
         "test_integrity.py", "PROTOCOL.txt", "README.txt", "HPC_COMMANDS.txt",
         "HPC_DIAGNOSIS.txt", "LOCAL_VERIFICATION.json", "input_hashes.json",
         "historical_C_exclusion.json", "build_package.py"]
FILES += ["reference_exp8/" + f"eval_{s}.jsonl" for s in ["A_newPerson", "B_newCategory", "C_bothNew"]]


def main():
    verification = json.loads((HERE / "LOCAL_VERIFICATION.json").read_text())
    if not verification["full_tiny_pipeline_passed"] or not verification["integrity_checks_passed"]:
        raise ValueError("Local validation incomplete")
    for name in FILES:
        if not (HERE / name).is_file():
            raise FileNotFoundError(name)
    with (HERE / "SHA256SUMS").open("x") as f:
        for name in FILES:
            f.write(hashlib.sha256((HERE / name).read_bytes()).hexdigest() + "  " + name + "\n")
    with tarfile.open(HERE / "exp12_upload.tgz", "x:gz") as tar:
        for name in FILES + ["SHA256SUMS"]:
            tar.add(HERE / name, arcname=name, recursive=False)
    archive = HERE / "exp12_upload.tgz"
    with (HERE / "PACKAGE_SHA256.txt").open("x") as f:
        f.write(hashlib.sha256(archive.read_bytes()).hexdigest() + "  exp12_upload.tgz\n")
    print(archive, archive.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
