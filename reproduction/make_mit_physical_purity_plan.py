"""Freeze a separate MIT physical-identity sensitivity plan without training."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

from audit_validation_suite import evidence_inputs
from make_paper_plan import extended_task
from run_extended import normalize_task
from run_full_suite import prepare_plan


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = "mit_physical_purity_v1"
INDEX_SHA256 = "93176963bbaaaea60a46591a3e7d1e5c3f1a04a250107eec1b754a3311b207dc"
CRITICAL_SOURCE_SHA256 = {
    "2017-05-12": "8a789f4a83409ad0c1d6c214c836f2de94b63ef2e73a50722fe057c3d8a89d77",
    "2017-06-30": "5f3f752d9d9a0276442fbb853941d9b212e7fffd152995b207ae6fa4a4503fae",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def canonical_csv(path):
    value = str(path).replace("\\", "/")
    if value.startswith("PINN4SOH/"):
        value = value[len("PINN4SOH/"):]
    if not value.startswith("data/MIT data/") or "/../" in value:
        raise ValueError("Unexpected MIT relative path: " + value)
    return value


def overlaps(records, uid_by_csv):
    by_uid = defaultdict(lambda: defaultdict(list))
    for record in records:
        for path in record["paths"]:
            for uid in uid_by_csv[path]:
                by_uid[uid][record["role"]].append(path)
    return [{"physical_cell_uid": uid, "roles": dict(roles)} for uid, roles in sorted(by_uid.items())
            if len(roles) > 1]


def cross_physical_pairs(trace, cycle_mapping, provenance):
    with cycle_mapping.open(encoding="utf-8-sig", newline="") as stream:
        identities = {(canonical_csv(row["csv"]), int(row["csv_row_zero_based"])): row["cell_uid"]
                      for row in csv.DictReader(stream)}
    pairs = {}
    for frame in trace["loader_trace"]:
        for battery in frame["batteries"]:
            path, rows = canonical_csv(battery["battery_id"]), battery["retained_original_rows"]
            for left, right in zip(rows[:-1], rows[1:]):
                if identities[path, left] != identities[path, right]:
                    pairs[path, left, right] = {"path": path, "x1_original_row_zero_based": left,
                        "x2_original_row_zero_based": right, "x1_physical_uid": identities[path, left],
                        "x2_physical_uid": identities[path, right]}
    endpoint_counts = defaultdict(Counter)
    with provenance.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (canonical_csv(row["battery_id"]), int(row["x1_original_row_0based"]), int(row["x2_original_row_0based"]))
            endpoint_counts[row["role"], "x1"][identities[key[0], key[1]]] += 1
            endpoint_counts[row["role"], "x2"][identities[key[0], key[2]]] += 1
            if key in pairs:
                if "role" in pairs[key]:
                    raise ValueError("Cross-physical pair appears in multiple original roles")
                pairs[key].update(role=row["role"], original_dataset_position=int(row["dataset_position"]))
    if any("role" not in row for row in pairs.values()):
        raise ValueError("A retained cross-physical pair has no original loader role")
    for role in ["train", "valid", "test"]:
        if sum(endpoint_counts[role, "x1"].values()) != trace["counts"]["n_" + role + "_pairs"]:
            raise ValueError("Physical endpoint counts disagree with the verified trace")
    test = endpoint_counts["test", "x1"]
    exposures = {}
    for model_use, seen in [("PINN_train_x1_and_x2", set(endpoint_counts["train", "x1"]) | set(endpoint_counts["train", "x2"])),
                            ("MLP_CNN_train_x1_only", set(endpoint_counts["train", "x1"]))]:
        shared = sorted(set(test) & seen)
        n_shared, n_test = sum(test[uid] for uid in shared), sum(test.values())
        exposures[model_use] = {"n_test_x1": n_test, "n_test_x1_with_physical_identity_seen_in_training": n_shared,
            "fraction": n_shared / n_test, "by_physical_uid": [{"physical_uid": uid, "test_x1": test[uid],
                "train_x1": endpoint_counts["train", "x1"][uid], "train_x2": endpoint_counts["train", "x2"][uid]} for uid in shared]}
    return list(pairs.values()), exposures


def build(options):
    repo, output = options.repo.resolve(), options.output.resolve()
    evidence, inventory, manifest = evidence_inputs(options.evidence)
    trace, audit, replay_summary = map(read_json, [options.trace, options.mapping_audit, options.replay_summary])
    signatures = read_json(options.signatures)
    if (signatures["selection_uses_feature_error"] or signatures["selected_rows"] != 15
            or signatures["selected_CSVs"] != 3 or len(signatures["records"]) != 30):
        raise ValueError("Expected the unchanged prespecified 15-cycle critical source-signature audit")
    for date, checksum in CRITICAL_SOURCE_SHA256.items():
        if signatures["source_checksum"][date].get("verified_full_sha256") != checksum:
            raise ValueError("Critical source signatures require published full-file SHA validation")
    if digest(options.evidence) != replay_summary["evidence_sha256"]:
        raise ValueError("Replay evidence package changed")
    if replay_summary.get("status") != "complete" or not replay_summary.get("full_sample_counts_match_recorded_cloud_runs"):
        raise ValueError("The recorded-order replay is not verified")
    if inventory["fingerprint"] != replay_summary["source_fingerprint"]:
        raise ValueError("Replay/source fingerprint differs")
    for name in ["dataloader/dataloader.py", "Model/Model.py", "main_MIT.py", "main_comparision.py"]:
        if digest(repo / name) != inventory["sha256"][name]:
            raise ValueError("Current author source differs from verified old110: " + name)
    if trace["dataset"] != "MIT" or trace["sample_size"] is not None or trace["parameters"]["normalization_method"] != "min-max":
        raise ValueError("Expected full MIT author normalization replay")
    if trace["identity_label_verification"].get("all_bitwise_equal") is not True:
        raise ValueError("Replay identity/label check did not pass")
    expected = manifest["runs"]["MIT/all/Experiment01"]["sample_counts"]
    if {role: trace["counts"]["n_" + role + "_pairs"] for role in ["train", "valid", "test"]} != expected:
        raise ValueError("Replay counts differ from old110")
    if digest(options.mapping_audit.parent / "within_cycle_index.csv") != INDEX_SHA256:
        raise ValueError("Third-party index differs from its published hash")
    if audit["mapped_cycle_rows"] != audit["csv_cycle_rows"] or audit["unresolved_capacity_rows"]:
        raise ValueError("A physical-identity mapping is incomplete or unresolved")
    uid_by_csv, by_csv = {}, {}
    for item in audit["mapping"]:
        relative = canonical_csv(item["csv"])
        if not item["all_rows_capacity_sourced"] or sum(item["physical_cell_uid_counts"].values()) != item["csv_rows"]:
            raise ValueError("Not every CSV row has an exact physical source: " + relative)
        actual = digest(repo / relative)
        if actual != item["csv_sha256"] or actual != inventory["sha256"][relative]:
            raise ValueError("Local CSV, MIT mapping and old110 inventory disagree: " + relative)
        uid_by_csv[relative] = set(item["physical_cell_uid_counts"])
        by_csv[relative] = item
    if len(by_csv) != 125:
        raise ValueError("Expected all 125 released MIT CSVs")
    original, retained_rows = [], {}
    if len(trace["loader_trace"]) != 2:
        raise ValueError("Expected author train_validation then test loader calls")
    if (len(trace["loader_trace"][0]["pairs"]) != expected["train"] + expected["valid"]
            or len(trace["loader_trace"][1]["pairs"]) != expected["test"]):
        raise ValueError("Recorded-order trace calls do not match old110 train_validation/test roles")
    for role, frame in zip(["train_validation", "test"], trace["loader_trace"]):
        paths = [canonical_csv(path) for path in frame["paths"]]
        if paths != [canonical_csv(battery["battery_id"]) for battery in frame["batteries"]]:
            raise ValueError("Trace file paths and per-battery metadata have different order")
        for battery in frame["batteries"]:
            relative = canonical_csv(battery["battery_id"])
            if battery["sha256"] != inventory["sha256"][relative]:
                raise ValueError("Trace CSV hash differs from old110")
            retained_rows[relative] = len(battery["retained_original_rows"])
        original.append({"role": role, "paths": paths, "nominal_capacity": frame["nominal_capacity"]})
    flat = [path for record in original for path in record["paths"]]
    if len(flat) != len(set(flat)) or set(flat) != set(by_csv):
        raise ValueError("Old110 replay roles do not partition the audited MIT CSVs")
    original_cross_pairs, original_exposures = cross_physical_pairs(trace, options.cycle_mapping, options.provenance)
    mixed = {path for path, uids in uid_by_csv.items() if len(uids) > 1}
    if mixed != {"data/MIT data/2017-05-12/2017-05-12_battery-5.csv",
                 "data/MIT data/2017-05-12/2017-05-12_battery-6.csv"}:
        raise ValueError("Mixed physical-source exclusions changed; review a new diagnostic version")
    excluded = []
    filtered = []
    for record in original:
        filtered.append({**record, "paths": [path for path in record["paths"] if path not in mixed]})
        excluded.extend({"path": path, "original_role": record["role"], "sha256": by_csv[path]["csv_sha256"],
                         "reason": "Released CSV contains rows from more than one physical barcode",
                         "physical_cell_uid_counts": by_csv[path]["physical_cell_uid_counts"],
                         "csv_rows": by_csv[path]["csv_rows"]}
                        for path in record["paths"] if path in mixed)
    additional_overlap = overlaps(filtered, uid_by_csv)
    conflicting_train = {path for issue in additional_overlap for path in issue["roles"]["train_validation"]}
    if conflicting_train:
        filtered[0]["paths"] = [path for path in filtered[0]["paths"] if path not in conflicting_train]
        excluded.extend({"path": path, "original_role": "train_validation", "sha256": by_csv[path]["csv_sha256"],
                         "reason": "Pure file shares its physical barcode with a retained test file",
                         "physical_cell_uid_counts": by_csv[path]["physical_cell_uid_counts"],
                         "csv_rows": by_csv[path]["csv_rows"]} for path in sorted(conflicting_train))
    if overlaps(filtered, uid_by_csv) or any(not record["paths"] for record in filtered):
        raise ValueError("Physical train/test identity disjointness check failed")
    output.mkdir(parents=True, exist_ok=True)
    split_path = output / "mit_physical_purity_split.json"
    write_json(split_path, filtered)
    cloud_split = options.cloud_plan_dir.rstrip("/") + "/" + split_path.name
    tasks = []
    for model, lane in [("PINN", "published_code"), ("PINN", "paper_equation"),
                        ("MLP", "shared_baseline"), ("CNN", "shared_baseline")]:
        for repeat in range(1, 11):
            task = extended_task("MIT", "all", model, lane, repeat, "full", options.reference)
            task["task_id"] = f"sensitivity/{DIAGNOSTIC}/{lane}/MIT/all/{model}/Experiment{repeat:02d}"
            task["seed"] = manifest["runs"][f"MIT/all/Experiment{repeat:02d}"]["seed"]
            task.update(split_json=cloud_split, diagnostic_protocol=DIAGNOSTIC,
                        paper_coverage=[], comparison_to_published_values="Extra diagnostic; not the original paper protocol",
                        exclusion_policy="Exclude entire mixed-barcode CSVs, then conflicting train files if needed; preserve remaining order")
            task["protocol_evidence"].extend([
                "Verified old110 exact recorded-directory-order MIT replay, with all 125 CSV hashes matched",
                "Zenodo 10.5281/zenodo.23984188 index provenance and prespecified within-cycle feature signatures",
                "Original author sample validation and per-battery min-max normalization retained after whole-file exclusions"])
            normalize_task(task)
            tasks.append(task)
    plan = {"schema": "pinn4soh-full-paper-plan-v1", "stage": "separate_mit_physical_purity_sensitivity",
            "diagnostic_protocol": DIAGNOSTIC, "training_launched": False,
            "interpretation": "Additional sensitivity analysis; never replace, relabel, or combine with the original formal runs",
            "tasks": tasks}
    if len(prepare_plan(plan, 2)) != 40:
        raise ValueError("Expected exactly 40 independent tasks")
    write_json(output / "mit_physical_purity_plan.json", plan)
    smoke = {**plan, "stage": "separate_mit_physical_purity_smoke",
             "tasks": [task for task in tasks if task["repeat"] == 1]}
    if len(prepare_plan(smoke, 2)) != 4:
        raise ValueError("Expected one smoke candidate for each PINN lane and baseline model")
    write_json(output / "mit_physical_purity_smoke_plan.json", smoke)
    roles = {}
    for record in filtered:
        paths = record["paths"]
        roles[record["role"]] = {"csv_count": len(paths),
            "physical_cell_count": len(set().union(*(uid_by_csv[path] for path in paths))),
            "released_csv_rows": sum(by_csv[path]["csv_rows"] for path in paths),
            "retained_adjacent_pairs_expected_from_verified_trace": sum(retained_rows[path] - 1 for path in paths)}
    train_valid = roles["train_validation"]["retained_adjacent_pairs_expected_from_verified_trace"]
    validation_pairs = (train_valid + 4) // 5
    summary = {"diagnostic_protocol": DIAGNOSTIC, "training_launched": False,
        "tasks": 40, "by_model": dict(Counter(task["model"] for task in tasks)),
        "by_loss_variant": dict(Counter(task["loss_variant"] for task in tasks)),
        "seeds": sorted({task["seed"] for task in tasks}),
        "original_roles": {item["role"]: len(item["paths"]) for item in original},
        "retained_roles": roles, "expected_author_pair_counts": {"train": train_valid - validation_pairs,
            "valid": validation_pairs, "test": roles["test"]["retained_adjacent_pairs_expected_from_verified_trace"]},
        "excluded_files": excluded, "original_physical_train_test_overlap": overlaps(original, uid_by_csv),
        "original_cross_physical_adjacent_retained_pairs": original_cross_pairs,
        "original_test_physical_identity_exposure": original_exposures,
        "cross_physical_pair_interpretation": "The verified old110 trace assigns the two cross-physical boundary pairs to validation and test, with zero in train. Author Valid/Test read x1/y1 only, so this is not evidence that the training physical penalty directly spans those two boundaries.",
        "retained_physical_train_test_overlap": overlaps(filtered, uid_by_csv),
        "validation_protocol": "author_sample_split: original sklearn train_test_split(test_size=0.2, random_state=420)",
        "normalization": "Unmodified author DF: per-file cleaning and min-max; capacity divided by 1.1 Ah",
        "physical_identity_evidence_scope": "Every released row's exact capacity source in the published third-party index; independently checked prespecified within-cycle signatures. This is not full16-feature per-row reconstruction.",
        "comparison_target": "Main full-paper stage1 MIT tasks matched by model, loss_variant, seed and effective parameters; old110 supplies verified input order and seeds, not a substitute loss/parameter control.",
        "source_fingerprint": inventory["fingerprint"], "cloud_plan_dir": options.cloud_plan_dir,
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
            [options.evidence, options.trace, options.replay_summary, options.mapping_audit, options.signatures,
             options.cycle_mapping, options.provenance]},
        "published_index_sha256": INDEX_SHA256,
        "plan_sha256": digest(output / "mit_physical_purity_plan.json"),
        "split_sha256": digest(split_path),
        "limitations": ["Physical identities come from a third-party derived mirror, not newly obtained original MAT files",
            "Removing a mixed test file changes the test population; effect sizes cannot isolate contamination bias",
            "Author sample validation still reuses adjacent endpoints; this diagnostic does not correct that separate issue",
            "Remaining pure CSVs are used unchanged; tails are not reassigned or reconstructed",
            "Pair counts are derived from the verified original per-file cleaning trace; confirm on the cloud before full launch"]}
    if options.verify_loader:
        from run_extended import load_data
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0, str(repo))
        previous = Path.cwd()
        try:
            os.chdir(repo)
            args = argparse.Namespace(batch_size=512, normalization_method="min-max", log_dir=None, save_folder=None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                loaders = load_data(normalize_task(tasks[0]), args, None, None, None, repo, split_path)
            counts = {role: len(loader.dataset) for role, loader in loaders.items()}
            if counts != summary["expected_author_pair_counts"]:
                raise ValueError("Actual unchanged author loader counts disagree with the verified trace")
            summary["actual_loader_check"] = {"passed": True, "training_performed": False,
                "counts": counts, "entry_point": "Unmodified run_extended.load_data with author_sample_split and the generated split_json",
                "tensor_sha256": {role: [hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                    for tensor in loader.dataset.tensors] for role, loader in loaders.items()}}
        finally:
            os.chdir(previous)
    write_json(output / "mit_physical_purity_exclusion_evidence.json", summary)
    print(json.dumps({key: summary[key] for key in ["tasks", "original_roles", "retained_roles", "expected_author_pair_counts",
        "original_cross_physical_adjacent_retained_pairs", "retained_physical_train_test_overlap", "plan_sha256", "split_sha256"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT / "PINN4SOH")
    parser.add_argument("--evidence", type=Path, default=ROOT / "outputs/final_results/pinn4soh_final_evidence.json")
    parser.add_argument("--trace", type=Path, default=ROOT / "outputs/validation_overlap_43_20261110/MIT/all/nall/trace.json")
    parser.add_argument("--replay-summary", type=Path, default=ROOT / "outputs/validation_overlap_43_20261110/summary.json")
    parser.add_argument("--mapping-audit", type=Path, default=ROOT / "outputs/full_reproduction/feature_audit/MIT_mirror_audit/MIT_mirror_mapping_audit.json")
    parser.add_argument("--signatures", type=Path, default=ROOT / "outputs/full_reproduction/feature_audit/MIT_mirror_audit/signatures/MIT_mirror_feature_signatures.json")
    parser.add_argument("--cycle-mapping", type=Path, default=ROOT / "outputs/full_reproduction/feature_audit/MIT_mirror_audit/PINN_CSV_to_mirror_cycles.csv")
    parser.add_argument("--provenance", type=Path, default=ROOT / "outputs/validation_overlap_43_20261110/MIT/all/nall/cycle_provenance.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/full_reproduction/MIT_physical_purity_sensitivity")
    parser.add_argument("--cloud-plan-dir", default="/workspace/full_supplement_tools_v1/MIT_physical_purity_sensitivity")
    parser.add_argument("--reference", default="/workspace/pinn4soh_main_20261109")
    parser.add_argument("--verify-loader", action="store_true", help="Check actual unchanged author loader counts locally, without training")
    build(parser.parse_args())
