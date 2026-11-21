"""Run bounded analytical and real-data acceptance of replacement features.

Uses one fixed cell and first three previously aligned published rows per
dataset. This is an implementation check, not a whole-dataset rerun or training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    if output.exists():
        raise ValueError("Use a new acceptance directory")
    output.mkdir(parents=True)
    feature = "outputs/full_reproduction/feature_audit/"
    commands = [
        ("XJTU", ["--kind", "xjtu-mat", "--input", feature + "raw_sample/2C_battery-1.mat",
                   "--expected-sha256", "336a0c25c9d6e4c18a231c0b4004edfb44fbd7808533490cddf4d073ce854dbb",
                   "--cycles", "10", "12", "13", "--reference-csv", "PINN4SOH/data/XJTU data/2C_battery-1.csv",
                   "--alignment-csv", feature + "boundary_extraction/XJTU/2C_battery-1_a973e3fa7df8/cycle_alignment.csv"]),
        ("TJU", ["--kind", "official-archive", "--input", "outputs/full_reproduction/raw_data/TJU/Dataset_1_NCA_battery.zip",
                  "--expected-sha256", "0d58f6f09f4ada76c2c3ab038ce98926ebdba7d18aaa31a9d7bebdccc9a834ac",
                  "--member", "Dataset_1_NCA_battery/CY25-025_1-#1.csv", "--cycles", "1", "2", "4",
                  "--reference-csv", "PINN4SOH/data/TJU data/Dataset_1_NCA_battery/CY25-025_1-#1.csv",
                  "--alignment-csv", feature + "raw_extraction/TJU/CY25-025_1-#1_8cc6959e7744/cycle_alignment.csv"]),
        ("HUST", ["--kind", "official-archive", "--input", "outputs/full_reproduction/raw_data/HUST/our_data.zip",
                   "--expected-sha256", "071d24617153693b0d29059568525e620f6af6512acc9d00c98c7adcf15125db",
                   "--member", "our_data/1-1.pkl", "--cycles", "12", "14", "15",
                   "--reference-csv", "PINN4SOH/data/HUST data/1-1.csv",
                   "--alignment-csv", feature + "boundary_extraction/HUST/1-1_4cb8cac358ed/cycle_alignment.csv"]),
        ("MIT", ["--kind", "mit-mirror", "--input", feature + "MIT_mirror_audit/within_cycle_2017-05-12.h5",
                  "--expected-sha256", "8a789f4a83409ad0c1d6c214c836f2de94b63ef2e73a50722fe057c3d8a89d77",
                  "--csv-name", "2017-05-12_battery-1.csv", "--current-unit", "native_unit_unresolved",
                  "--cycles", "8", "15", "16", "--reference-csv", "PINN4SOH/data/MIT data/2017-05-12/2017-05-12_battery-1.csv",
                  "--alignment-csv", str(output / "mit_alignment.csv")]),
    ]
    (output / "mit_alignment.csv").write_text("processed_row,raw_cycle\n0,8\n1,15\n2,16\n", encoding="utf-8")
    tests = subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "reproduction",
                            "-p", "test_replacement_features.py", "-v"], cwd=ROOT, capture_output=True, text=True)
    (output / "analytical_tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    if tests.returncode:
        raise RuntimeError("Analytical tests failed")
    real = []
    for dataset, arguments in commands:
        target = output / (dataset.lower() + "_real3")
        command = [sys.executable, "-X", "utf8", "reproduction/export_replacement_features.py", "--dataset", dataset,
                   "--limit", "3", "--output", str(target), *arguments]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (output / (dataset.lower() + "_command.json")).write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / (dataset.lower() + "_console.txt")).write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(dataset + " export failed; see saved log")
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if manifest["code_sha256"]["export_replacement_features.py"] != sha(ROOT / "reproduction/export_replacement_features.py"):
            raise ValueError("Implementation changed during acceptance")
        if any(sha(target / name) != digest for name, digest in manifest["artifacts_sha256"].items()):
            raise ValueError("Export artifact SHA mismatch")
        summary = json.loads((target / "released_comparison_summary.json").read_text(encoding="utf-8"))
        groups = {}
        for status in ("prior_audited_convention", "replacement"):
            values = [row for row in summary["features"] if row["status"] == status]
            groups[status] = {"feature_count": len(values), "finite_pairs": sum(row["finite_pairs"] for row in values),
                              "matches_at_1e_8": sum(row["matches_at_1e_8"] for row in values),
                              "maximum_absolute_error": max(row["maximum_absolute_error"] for row in values)}
        real.append({"dataset": dataset, "kind": "real_raw_sample", "manifest": str(target / "manifest.json"),
                     "manifest_sha256": sha(target / "manifest.json"), "cycles": manifest["selected_cycles"], "groups": groups})
    state = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "implementation_acceptance": "passed_bounded_samples", "all_released_features_reconstructed": False,
             "analytical_fixture": {"kind": "analytic_and_failure_contract", "tests": 5, "returncode": tests.returncode,
                                    "log_sha256": sha(output / "analytical_tests.txt")},
             "real_samples": real, "training_performed": False, "frozen_inputs_changed": False,
             "code_sha256": {name: sha(ROOT / "reproduction" / name) for name in
                             ["export_replacement_features.py", "validate_replacement_features.py", "test_replacement_features.py"]}}
    (output / "VALIDATION.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
