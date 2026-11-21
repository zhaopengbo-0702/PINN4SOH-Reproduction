"""Verify required original-raw cells, frozen implementations, and saved hashes."""

import csv
import hashlib
import json
from pathlib import Path
import time


ROOT = Path("outputs/full_reproduction/feature_audit")
SCRIPTS = ("extract_raw_feature_candidates.py", "audit_xjtu_feature_sample.py")
IMPLEMENTATION = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SCRIPTS}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


required = list(csv.DictReader((ROOT / "required_raw_files.csv").open(encoding="utf-8", newline="")))
expected = {row["processed_csv"] for row in required if row["dataset"] in {"XJTU", "TJU", "HUST"}}
verified, errors = [], []
for path in sorted((ROOT / "raw_extraction").rglob("manifest.json")):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = dict(processed_csv=manifest["processed_csv"], dataset=manifest["dataset"], raw_cycles=manifest.get("raw_cycles"),
                  status_complete=manifest["status"] == "complete", frozen_implementation=manifest.get("implementation_sha256") == IMPLEMENTATION,
                  selected_without_csv_residual=manifest.get("selected_by_csv_residual") is False)
    candidate = path.parent / "candidate_features.csv"
    result["saved_candidate_sha256_matches"] = candidate.is_file() and digest(candidate) == manifest.get("candidate_csv_sha256")
    result["passed"] = all(result[name] for name in ("status_complete", "frozen_implementation", "selected_without_csv_residual", "saved_candidate_sha256_matches"))
    verified.append(result)
    if not result["passed"]:
        errors.append(result)
boundary_verified = []
boundary_implementation = dict(IMPLEMENTATION)
boundary_implementation["supplement_voltage_boundaries.py"] = hashlib.sha256(Path(__file__).with_name("supplement_voltage_boundaries.py").read_bytes()).hexdigest()
for path in sorted((ROOT / "boundary_extraction").rglob("manifest.json")):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    primary = next(row for row in verified if row["processed_csv"] == manifest["processed_csv"])
    result = dict(processed_csv=manifest["processed_csv"], status_complete=manifest["status"] == "complete",
                  frozen_implementation=manifest.get("implementation_sha256") == boundary_implementation,
                  primary_file_verified=primary["passed"],
                  saved_candidate_sha256_matches=digest(path.parent / "candidate_features.csv") == manifest.get("candidate_sha256"))
    result["passed"] = all(result[key] for key in ("status_complete", "frozen_implementation", "primary_file_verified", "saved_candidate_sha256_matches"))
    boundary_verified.append(result)
    if not result["passed"]:
        errors.append(result)
observed = [row["processed_csv"] for row in verified]
state = dict(verified_at_unix=time.time(), expected_original_raw_cells=262, observed_cells=len(observed),
             unique_observed_cells=len(set(observed)), missing_cells=sorted(expected - set(observed)),
             unexpected_cells=sorted(set(observed) - expected), raw_physical_cycles=sum(row["raw_cycles"] or 0 for row in verified),
             files_passed=sum(row["passed"] for row in verified), errors=errors,
             boundary_expected_cells=132, boundary_observed_cells=len(boundary_verified),
             boundary_files_passed=sum(row["passed"] for row in boundary_verified), boundary_cells=boundary_verified,
             frozen_implementation_sha256=IMPLEMENTATION, cells=verified,
             scope="XJTU/TJU/HUST only; this does not establish the original MIT MAT chain")
state["all_checks_passed"] = len(observed) == len(set(observed)) == len(expected) == 262 and len(boundary_verified) == 132 and all(row["passed"] for row in boundary_verified) and not errors and not state["missing_cells"] and not state["unexpected_cells"]
(ROOT / "RAW_FEATURE_INTEGRITY.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in state.items() if key not in {"cells", "boundary_cells"}}, indent=2))
if not state["all_checks_passed"]:
    raise SystemExit(1)
