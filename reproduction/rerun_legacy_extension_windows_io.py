"""Rerun the frozen selected-extension audit with Windows long-path I/O.

Only process-local Path.resolve behavior is adapted. Task identities, arrays,
parameters, comparability predicates and metric arithmetic stay unchanged.
Requires the separately collected extension archive to have been safely restored
and checked against its external collection inventory before invocation.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


TABLES = (
    "task_matching.csv", "per_repeat_metric_differences.csv",
    "per_battery_differences.csv", "per_repeat_method_effects.csv",
    "ten_repeat_summaries.csv", "ten_repeat_effect_summaries.csv",
    "comparison_evidence.json",
)
SUMMARY_FIELDS = (
    "format", "task_counts", "expected_tasks", "expected_repeats",
    "reused_repeat1", "new_training_repeats", "assessment", "selection",
    "no_significance_or_equivalence_claim", "summaries",
    "method_effect_summaries", "analysis_manifest_sha256",
    "analysis_scripts_sha256", "training_performed", "limitation",
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def extended_path_resolution():
    original = Path.resolve

    def resolved(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        if os.name == "nt" and not str(value).startswith("\\\\?\\"):
            value = Path("\\\\?\\" + str(value))
        return value

    Path.resolve = resolved
    return original


def suite_hashes(document):
    result = {}
    for path, checksum in document["input_suite_manifest_sha256"].items():
        name = path.replace("\\", "/").rstrip("/").split("/")[-1]
        if name in result:
            raise ValueError("Duplicate suite name in summary")
        result[name] = checksum
    return result


def summary_rounding_equivalent(left, right, path="", differences=None):
    """Exact values except a single ULP in named SD statistics.

    Python3.9/3.10 statistics variance square-root implementations can differ
    by one representable binary64 step. This is a numerical reporting check,
    never a scientific tie, effect-size or method-comparability tolerance.
    """
    differences = [] if differences is None else differences
    if type(left) != type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(summary_rounding_equivalent(left[key], right[key], path + "/" + key, differences) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(summary_rounding_equivalent(a, b, path + "/" + str(i), differences) for i, (a, b) in enumerate(zip(left, right)))
    if left == right:
        return True
    if isinstance(left, float) and path.endswith(("/sd_ddof0", "/sd_ddof1")):
        allowed = max(math.ulp(left), math.ulp(right))
        accepted = math.isfinite(left) and math.isfinite(right) and abs(left - right) <= allowed
        differences.append(dict(path=path, cloud=left, local=right, absolute_difference=abs(left - right),
                                difference_in_ulps=abs(left - right) / allowed, accepted=accepted))
        return accepted
    return False


def parsed_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [{key: json.loads(value) if value.startswith(("{", "[")) else value for key, value in row.items()}
                for row in csv.DictReader(stream)]


def rerun(extracted, output, cloud_comparison=None):
    extended_path_resolution()
    root, output = Path(extracted).resolve(), Path(output).resolve()
    cloud = Path(cloud_comparison).resolve() if cloud_comparison else root / "legacy_selected_comparison_20261110"
    if output.exists() or root == output or root in output.parents or output in root.parents:
        raise ValueError("Use a fresh output outside the restored input directory")
    cloud_summary = json.loads((cloud / "summary.json").read_text(encoding="utf-8-sig"))
    if cloud_summary.get("assessment") != "complete_selected_ten_repeat_sensitivity" or cloud_summary.get("task_counts") != {"matched": 110}:
        raise ValueError("Requires the complete original 110-matched cloud comparison")
    tools = Path(__file__).resolve().parent
    expected_scripts = cloud_summary["analysis_scripts_sha256"]
    for name, expected in expected_scripts.items():
        if digest(tools / name) != expected:
            raise ValueError("Local frozen comparator dependency differs from the cloud: " + name)
    input_hashes = {name: digest(cloud / name) for name in [*TABLES, "summary.json"]}
    import compare_legacy_extension as frozen
    result = frozen.compare(
        root / "legacy_selected_support_20261110",
        root / "pinn4soh_legacy_formal_control_20261110",
        root / "pinn4soh_legacy_selected_extension_20261110",
        [root / "pinn4soh_full_paper_stage1", root / "pinn4soh_full_paper_stage2"],
        output,
    )
    checks = {name: {"cloud_sha256": input_hashes[name], "local_sha256": digest(output / name)} for name in TABLES}
    for check in checks.values():
        check["byte_identical"] = check["cloud_sha256"] == check["local_sha256"]
    differences = []
    summary_checks = {name: summary_rounding_equivalent(cloud_summary.get(name), result.get(name), "/" + name, differences)
                      for name in SUMMARY_FIELDS}
    summary_checks["input_suite_manifest_hashes_by_suite_name"] = suite_hashes(result) == suite_hashes(cloud_summary)
    parsed_checks = {name: summary_rounding_equivalent(parsed_csv(cloud / name), parsed_csv(output / name), "/" + name)
                     for name in ["ten_repeat_summaries.csv", "ten_repeat_effect_summaries.csv"]}
    parsed_checks["comparison_evidence.json"] = json.loads((cloud / "comparison_evidence.json").read_text(encoding="utf-8")) == json.loads((output / "comparison_evidence.json").read_text(encoding="utf-8"))
    cloud_unchanged = all(digest(cloud / name) == checksum for name, checksum in input_hashes.items())
    passed = (result["task_counts"] == {"matched": 110} and
              all(checks[name]["byte_identical"] for name in TABLES[:4]) and all(parsed_checks.values()) and
              all(summary_checks.values()) and cloud_unchanged)
    audit = {
        "format": "pinn4soh-extension-local-windows-io-rerun-v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_restored_root": str(root), "cloud_comparison": str(cloud),
        "local_comparison": str(output), "task_counts": result["task_counts"],
        "byte_comparisons": checks, "summary_field_comparisons": summary_checks,
        "parsed_non_byte_identical_comparisons": parsed_checks,
        "sd_one_ulp_differences": differences,
        "numeric_check_policy": "All per-repeat values and scientific identities exact; only named sd_ddof0/sd_ddof1 may differ by <=1 binary64 ULP; mean, directions and counts remain exact. JSON line endings/order do not alter parsed evidence.",
        "cloud_comparison_bytes_unchanged": cloud_unchanged,
        "scope": "Frozen comparator recomputes metrics from restored arrays and checks planned identities, completion-time hashes and source lineage; no training or checkpoint pickle load",
        "path_adapter": "Process-local Path.resolve preserves Windows extended-length prefix only",
        "expected_summary_difference": "input_suite_manifest_sha256 keys contain local restored suite paths instead of cloud paths; values checked by unique suite name",
        "external_inventory_audit_required_separately": True,
        "wrapper_sha256": digest(__file__), "passed": passed,
    }
    (output / "LOCAL_RERUN_IO_VERIFIED.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError("Local independent rerun differs; inspect LOCAL_RERUN_IO_VERIFIED.json")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud-comparison", type=Path)
    args = parser.parse_args()
    document = rerun(**vars(args))
    print(json.dumps({key: document[key] for key in ["task_counts", "cloud_comparison_bytes_unchanged", "passed"]}))
