"""Freeze Stage A9 improvement splits/specifications without training or scores.

This is exploratory improvement after historical test results were observed.
Only identity, filenames and checksums determine the new validation split.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SELECT_SEEDS = [2026111101, 2026111102, 2026111103]
CONFIRM_SEEDS = list(range(2026111111, 2026111121))
HUST_TEST = {"1-4", "1-8", "2-4", "2-8", "3-4", "3-8", "4-4", "4-8", "5-4", "5-7",
             "6-4", "6-8", "7-4", "7-8", "8-4", "8-8", "9-4", "9-8", "10-4", "10-8"}
FIXED = {"epochs": 200, "min_epochs": 30, "patience": 30, "warmup_epochs": 5,
         "batch_size": 512, "threads": 2, "normalization_method": "min-max",
         "diagnostic_batch_size": 64, "grad_clip_norm": 0, "warmup_ratio": .1,
         "final_lr_ratio": .01, "deterministic": True, "allow_tf32": False}
STRUCTURE = {"u_layers": 3, "u_hidden": 60, "F_layers_num": 3, "F_hidden_dim": 60}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_name(value):
    value = value.replace("\\", "/")
    if value.startswith("PINN4SOH/"):
        value = value[len("PINN4SOH/"):]
    if not value.startswith("data/MIT data/") or ".." in Path(value).parts:
        raise ValueError("Unexpected MIT path")
    return value


def number(path):
    return int(Path(path).stem.rsplit("-", 1)[-1])


def validate_partition(splits, identities):
    uid_sets, csv_sets = {}, {}
    for role, record in splits.items():
        uid_sets[role], csv_sets[role] = set(), set()
        if not record["paths"]:
            raise ValueError("Empty split")
        for path in record["paths"]:
            if len(identities[path]["physical_ids"]) != 1:
                raise ValueError("Mixed physical identity retained")
            uid_sets[role].update(identities[path]["physical_ids"])
            csv_sets[role].add(path)
        if len(csv_sets[role]) != len(record["paths"]):
            raise ValueError("Duplicate CSV within role")
        record["physical_ids"] = [identities[path]["physical_ids"][0] for path in record["paths"]]
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        if uid_sets[a] & uid_sets[b] or csv_sets[a] & csv_sets[b]:
            raise ValueError("Cross-role physical/CSV identity overlap")
    return {role: {"csv_count": len(csv_sets[role]), "physical_id_count": len(uid_sets[role])} for role in splits}


def mit(repo, identity_root):
    audit_path = identity_root / "MIT_mirror_mapping_audit.json"
    mapping_path = identity_root / "PINN_CSV_to_mirror_cycles.csv"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["mapped_cycle_rows"] != 85577 or audit["csv_cycle_rows"] != 85577 or audit["unresolved_capacity_rows"]:
        raise ValueError("MIT physical row mapping incomplete")
    expected = {csv_name(row["csv"]): row for row in audit["mapping"]}
    counts, row_numbers = defaultdict(Counter), defaultdict(set)
    with mapping_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            path = csv_name(row["csv"])
            index = int(row["csv_row_zero_based"])
            if index in row_numbers[path]:
                raise ValueError("Duplicate MIT mapping row")
            row_numbers[path].add(index)
            counts[path][row["cell_uid"]] += 1
    identities, excluded = {}, []
    for path, record in expected.items():
        if counts[path] != Counter(record["physical_cell_uid_counts"]) or row_numbers[path] != set(range(record["csv_rows"])):
            raise ValueError("MIT per-row mapping does not match independent source audit")
        actual = sha(repo / path)
        if actual != record["csv_sha256"] or not record["all_rows_capacity_sourced"]:
            raise ValueError("MIT source checksum/mapping mismatch")
        identities[path] = dict(physical_ids=sorted(counts[path]), csv_sha256=actual,
                                published_rows=record["csv_rows"], physical_uid_row_counts=dict(counts[path]),
                                identity_basis="Exact capacity source in full third-party index plus previously audited raw signatures")
        if len(counts[path]) != 1:
            excluded.append(path)
    if set(excluded) != {"data/MIT data/2017-05-12/2017-05-12_battery-5.csv", "data/MIT data/2017-05-12/2017-05-12_battery-6.csv"}:
        raise ValueError("MIT mixed identity exclusion changed")
    retained = set(identities) - set(excluded)
    test = sorted([path for path in retained if number(path) % 5 == 0], key=lambda p: (Path(p).parent.name, number(p)))
    remaining = retained - set(test)
    test_uids = {identities[path]["physical_ids"][0] for path in test}
    conflicting = [path for path in remaining if identities[path]["physical_ids"][0] in test_uids]
    if conflicting:
        raise ValueError("Pure MIT train CSV shares historical test physical UID; explicit review required")
    valid = []
    by_batch = defaultdict(list)
    for path in remaining:
        by_batch[Path(path).parent.name].append(path)
    for batch, paths in sorted(by_batch.items()):
        # Every fifth item in ascending battery number, after removing the
        # historical test and mixed files. No capacity/error statistic is used.
        ordered = sorted(paths, key=number)
        selected_uids = {identities[path]["physical_ids"][0] for path in ordered[4::5]}
        valid.extend(path for path in ordered if identities[path]["physical_ids"][0] in selected_uids)
    valid = sorted(valid, key=lambda p: (Path(p).parent.name, number(p)))
    train = sorted(remaining - set(valid), key=lambda p: (Path(p).parent.name, number(p)))
    splits = {role: {"paths": paths, "nominal_capacity": 1.1} for role, paths in [("train", train), ("valid", valid), ("test", test)]}
    summary = validate_partition(splits, identities)
    if len(test) != 22 or len(train) + len(valid) != 101:
        raise ValueError("MIT retained historical population changed")
    return splits, identities, dict(excluded_files=excluded, split_counts=summary,
        selection_rule="Exclude both mixed May CSV5/6; retain 22 historical numeric-id%5 test CSVs; for remaining pool by original batch ascending numeric battery-id select every fifth (positions 5,10,...) UID for validation; others train",
        batch_validation_counts=dict(Counter(Path(path).parent.name for path in valid)),
        input_sha256={str(audit_path.relative_to(ROOT)): sha(audit_path), str(mapping_path.relative_to(ROOT)): sha(mapping_path)},
        identity_caveat="Third-party derived MIT identity/capacity index, not original MAT byte provenance or proof of all 16 features.")


def hust(repo):
    paths = sorted((repo / "data/HUST data").glob("*.csv"), key=lambda p: tuple(map(int, p.stem.split("-"))))
    if len(paths) != 77 or not HUST_TEST.issubset({p.stem for p in paths}):
        raise ValueError("HUST published cell inventory changed")
    identities = {p.relative_to(repo).as_posix(): dict(physical_ids=["HUST:" + p.stem], csv_sha256=sha(p),
                    identity_basis="Distinct official cell key and published CSV filename; no independently measured barcode available") for p in paths}
    by_group = defaultdict(list)
    test = []
    for p in paths:
        name = p.relative_to(repo).as_posix()
        if p.stem in HUST_TEST:
            test.append(name)
        else:
            by_group[int(p.stem.split("-")[0])].append(name)
    valid = [paths[0] for group, paths in sorted(by_group.items())]
    train = [path for group, paths in sorted(by_group.items()) for path in paths[1:]]
    splits = {role: {"paths": names, "nominal_capacity": 1.1} for role, names in [("train", train), ("valid", valid), ("test", test)]}
    summary = validate_partition(splits, identities)
    if [summary[role]["csv_count"] for role in ["train", "valid", "test"]] != [47, 10, 20]:
        raise ValueError("HUST fixed split counts changed")
    return splits, identities, dict(excluded_files=[], split_counts=summary,
        selection_rule="Keep all 20 original paper test cell keys. Per group1..10 sort remaining cells by numeric cell-id, choose lowest available ID for validation; all other non-test cells train",
        group_validation={str(group): paths[0] for group, paths in sorted(by_group.items())},
        identity_caveat="HUST cell-key identity assumes official distinct raw keys represent distinct cells; physical barcode was not published.")


def candidates(model):
    values = []
    for lr in [.0003, .001, .003]:
        if model == "PINN":
            for alpha, beta in [(0., 0.), (.1, 0.), (.1, .01)]:
                values.append(dict(lr_u=lr, lr_F=lr if alpha else 0., alpha=alpha, beta_mean=beta, weight_decay=0.))
        else:
            for wd in [0., .00001, .0001]:
                values.append(dict(lr_u=lr, lr_F=0., alpha=0., beta_mean=0., weight_decay=wd))
    return values


def main(output):
    if output.exists():
        raise ValueError("Use a new preregistration directory")
    repo = ROOT / "PINN4SOH"
    source = ROOT / "outputs/full_reproduction/feature_audit/MIT_mirror_audit"
    output.mkdir(parents=True)
    components = {}
    for dataset, builder in [("MIT", lambda: mit(repo, source)), ("HUST", lambda: hust(repo))]:
        splits, identities, evidence = builder()
        identity_path = output / (dataset.lower() + "_physical_identity.json")
        write(identity_path, dict(dataset=dataset, identities=identities, evidence=evidence,
                                  cross_role_physical_id_overlap=[], checksum_verified=True))
        split_path = output / (dataset.lower() + "_split.json")
        write(split_path, splits)
        specs = []
        for model in ["PINN", "MLP", "CNN"]:
            spec = dict(label="exploratory_stage_a9_20261111_" + dataset + "_" + model, dataset=dataset, model=model,
                        seeds=SELECT_SEEDS, splits=splits, fixed=FIXED, structure=STRUCTURE, candidates=candidates(model),
                        protocol_evidence=["Exploratory improvement, separate from paper reproduction; historical test results were already observed",
                            "Stage A9 validation-only selection; workers must never load test split during candidate selection",
                            "identity_manifest=" + identity_path.name + ";sha256=" + sha(identity_path),
                            "split_manifest=" + split_path.name + ";sha256=" + sha(split_path),
                            "Selection metric: per-physical-cell RMSE equal-weight mean across validation cells, then mean across all 3 seeds; no best-seed selection",
                            "Monotonic penalty: mean relu(eval-mode prediction2-prediction1), gradient enabled; beta is beta_mean, not paper beta_sum",
                            "Author full-battery cleaning incl capacity and full-life per-file minmax retained; not deployment-causal preprocessing",
                            "10 future confirmation seeds frozen in registration; not evaluated by candidate workers"], smoke=False)
            name = dataset.lower() + "_" + model.lower() + "_stage_a9.json"
            write(output / name, spec)
            specs.append(dict(model=model, path=name, sha256=sha(output / name), candidates=9, seeds=len(SELECT_SEEDS), runs=27))
        components[dataset] = dict(split_path=split_path.name, split_sha256=sha(split_path),
                                   identity_path=identity_path.name, identity_sha256=sha(identity_path), evidence=evidence, specs=specs)
    write(output / "REGISTRATION.json", dict(schema="pinn4soh-exploratory-stage-a9-scientific-protocol-v1",
        status="frozen_before_new_candidate_training", created_at_utc=datetime.now(timezone.utc).isoformat(),
        implementation_sha256=sha(Path(__file__)), historical_test_observed=True, label="exploratory_improvement_not_paper_reproduction",
        selection_seeds=SELECT_SEEDS, future_confirmation_seeds=CONFIRM_SEEDS, datasets=components,
        total_selection_runs=162, total_candidates=54, training_launched=False,
        score_access_for_split_selection="None: filenames, row identity mappings and checksums only",
        stop_rule="Run all prespecified candidates and seeds; choose validation macro-RMSE only; do not expand grid after viewing test to guarantee a win",
        confirmation_policy="Freeze selected configuration per model/dataset before all 10 confirmation seeds; evaluate historical test only after selection and mark exploratory; cross-runtime confirmation preserves paired source/input SHA",
        preprocessing_limits=["3-sigma union removes rows using all columns including capacity", "Per-battery full-lifetime feature/cycle minmax uses held-out covariates; transductive", "Historical test is not pristine confirmatory holdout", "Dataset grouping not equivalent to unseen laboratory/chemistry generalization"],
        artifacts_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps({dataset: value["evidence"]["split_counts"] for dataset, value in components.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
