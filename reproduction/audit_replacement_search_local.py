"""Read-only portable audit of an extracted replacement-search archive.

No checkpoint deserialization, training, imports of archived Python or mutations
of restored evidence. Selection and CSV preprocessing are independently rebuilt.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics

DIMENSIONS = ("u_layers", "u_hidden", "F_layers_num", "F_hidden_dim", "alpha", "beta")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def relative_source(name, prefix):
    # Cloud paths are POSIX; local smoke may contain Windows separators.
    name, prefix = name.replace("\\", "/"), prefix.replace("\\", "/").rstrip("/")
    require(name.startswith(prefix + "/"), "Reference path is outside planned root")
    value = name[len(prefix) + 1:]
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == value, "Unsafe relative reference")
    return value


def validate_worker(worker, plan, folder, mode, candidate, seed):
    roles = ("train", "valid") if mode == "train" else ("test",)
    expect_splits = {role: plan["resolved_splits"][role] for role in roles}
    expect_input = {name: plan["input_sha256"][name] for role in roles for name in expect_splits[role]["paths"]}
    require(worker["mode"] == mode and worker["candidate_id"] == candidate["candidate_id"] and worker["seed"] == seed,
            "Worker mode/candidate/seed mismatch")
    require(worker["parameters"] == candidate["parameters"] and worker["fixed"] == plan["specification"]["fixed"] and
            worker["loss_variant"] == plan["specification"]["loss_variant"] and worker["device"] == plan["device"] and
            worker["repo"] == plan["repo"] and worker["source_sha256"] == plan["source_sha256"], "Worker frozen configuration mismatch")
    require(worker["splits"] == expect_splits and worker["input_sha256"] == expect_input, "Worker split or input SHA differs from plan")
    normalized = worker["output"].replace("\\", "/")
    require(normalized.endswith("/" + folder.parent.name + "/" + folder.name), "Worker output directory mismatch")


def preprocessing(plan, repo):
    import numpy as np
    import pandas as pd
    result, input_checks = {}, 0
    for role, record in plan["resolved_splits"].items():
        batteries, tensors, removed = [], [[], [], [], []], []
        for cloud_name in record["paths"]:
            local = repo / relative_source(cloud_name, plan["repo"])
            require(sha(local) == plan["input_sha256"][cloud_name], "Local input CSV SHA mismatch")
            input_checks += 1
            frame = pd.read_csv(local)
            require(frame.shape[1] == 17 and frame.columns[-1] == "capacity", "Expected sixteen features and capacity")
            frame.insert(frame.shape[1] - 1, "cycle index", np.arange(len(frame)))
            original_rows = len(frame)
            frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
            bad = set()
            for column in frame:
                series = frame[column]
                invalid = (series.mean() - 3 * series.std() > series) | (series.mean() + 3 * series.std() < series)
                bad.update(np.arange(len(series))[invalid].tolist())
            frame = frame.drop(list(bad)).reset_index(drop=True)
            frame["capacity"] = frame["capacity"] / record["nominal_capacity"]
            features = frame.iloc[:, :-1]
            if plan["specification"]["fixed"]["normalization_method"] == "min-max":
                features = 2 * (features - features.min()) / (features.max() - features.min()) - 1
            else:
                features = (features - features.mean()) / features.std()
            # Avoid pandas incompatible-dtype assignment while retaining float64
            # arithmetic and final float32 cast of the original worker.
            x = features.to_numpy()
            y = frame.iloc[:, -1].to_numpy()
            arrays = [x[:-1], x[1:], y[:-1, None], y[1:, None]]
            for target, values in zip(tensors, arrays):
                target.append(np.ascontiguousarray(values.astype(np.float32)))
            batteries.append(dict(path=cloud_name, n_pairs=len(x) - 1))
            removed.append(dict(path=cloud_name, original_rows=original_rows, retained_rows=len(frame)))
        combined = [np.concatenate(values, axis=0) for values in tensors]
        require(all(np.isfinite(values).all() and len(values) for values in combined), "Invalid rebuilt tensors")
        result[role] = dict(sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in combined], batteries=batteries,
                            labels=combined[2], row_counts=removed)
    return result, input_checks


def audit(extracted, repo, collection=None, expected_runs=192):
    import numpy as np
    extracted, repo = Path(extracted).resolve(), Path(repo).resolve()
    search, tools = extracted / "search", extracted / "tools"
    plan, state, selection = [read(search / name) for name in ("plan.json", "status.json", "selection.json")]
    require(state["status"] == "complete" and state["plan_sha256"] == stable(plan), "Search not complete or plan hash differs")
    seeds = plan["specification"]["seeds"]
    candidates = [dict(candidate_id="candidate_%04d" % (i + 1), parameters=dict(zip(DIMENSIONS, values))) for i, values in
                  enumerate(itertools.product(*(plan["specification"]["grid"][name] for name in DIMENSIONS)))]
    require(plan["candidates"] == candidates and len(seeds) == len(set(seeds)), "Cartesian grid or seed identity differs")
    require(len(candidates) * len(seeds) == plan["candidate_runs"] == expected_runs == state["total_runs"] == state["completed_runs"], "Formal count differs")
    inventory_checks = 0
    if collection:
        external = read(collection)
        require(external["format"] == "pinn4soh-replacement-search-collection-v1" and external["status"] == "complete" and
                external["all_archived_member_hashes_verified"] is True, "Invalid external collection evidence")
        inventory = {}
        for row in external["inventory"]:
            value = PurePosixPath(row["path"])
            require(not value.is_absolute() and ".." not in value.parts and value.as_posix() == row["path"] and
                    row["path"].casefold() not in {key.casefold() for key in inventory}, "Unsafe/duplicate inventory path")
            inventory[row["path"]] = row
        actual = {p.relative_to(extracted).as_posix() for p in extracted.rglob("*") if p.is_file()}
        require(actual == set(inventory), "Extracted member set differs from external inventory")
        for name, record in inventory.items():
            path = extracted / name
            require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "External member SHA/size mismatch")
        inventory_checks = len(inventory)
    tool_manifest = read(tools / "TOOLS_SHA256.json")
    for name, record in tool_manifest["files"].items():
        require(sha(tools / name) == record["sha256"] and (tools / name).stat().st_size == record["bytes"], "Archived tool SHA differs")
    local_py = {p.relative_to(repo).as_posix(): sha(p) for p in repo.rglob("*.py")}
    source_checks = 0
    for name, checksum in plan["source_sha256"].items():
        normalized = name.replace("\\", "/")
        if normalized.startswith(plan["repo"].replace("\\", "/").rstrip("/") + "/"):
            require(local_py[relative_source(name, plan["repo"])] == checksum, "Local author source differs")
        else:
            require(sha(tools / normalized.rsplit("/", 1)[-1]) == checksum, "Archived search source differs")
        source_checks += 1
    rebuilt, input_checks = preprocessing(plan, repo)
    artifact_checks, rows, environments = 0, [], set()
    expected_dirs = set()
    for candidate in candidates:
        for seed in seeds:
            folder = search / candidate["candidate_id"] / ("seed_%d" % seed)
            expected_dirs.add(folder.relative_to(search).as_posix())
            worker, result = read(folder / "worker.json"), read(folder / "result.json")
            validate_worker(worker, plan, folder, "train", candidate, seed)
            require(result["status"] == "complete" and result["worker_identity"] == stable(worker) and result["test_loaded"] is False,
                    "Candidate completion/isolation differs")
            require(set(result["artifact_sha256"]) == {"best_model.pth", "history.json", "valid_truth.npy", "valid_prediction.npy", "data_identity.json"}, "Missing candidate artifacts")
            for name, checksum in result["artifact_sha256"].items():
                require(sha(folder / name) == checksum, "Candidate completion-time artifact SHA differs")
                artifact_checks += 1
            data = read(folder / "data_identity.json")
            require(data == {role: {k: rebuilt[role][k] for k in ("sha256", "batteries")} for role in ("train", "valid")}, "Rebuilt candidate tensors differ")
            truth = np.load(folder / "valid_truth.npy", allow_pickle=False)
            pred = np.load(folder / "valid_prediction.npy", allow_pickle=False)
            require(np.array_equal(truth, rebuilt["valid"]["labels"]) and truth.shape == pred.shape and np.isfinite(pred).all(), "Validation labels/order/prediction invalid")
            mse = float(np.mean((truth.astype(np.float64) - pred.astype(np.float64)) ** 2))
            history = read(folder / "history.json")
            require(history and [r["epoch"] for r in history] == list(range(1, len(history) + 1)), "History epoch sequence invalid")
            require(all(math.isfinite(row["validation_mse"]) and len(row["train_losses"]) == 3 and
                        all(math.isfinite(v) for v in row["train_losses"]) for row in history), "Nonfinite history")
            best = min(history, key=lambda row: (row["validation_mse"], row["epoch"]))
            require(result["validation_mse"] == mse == best["validation_mse"] and result["best_epoch"] == best["epoch"] and
                    result["epochs_completed"] == len(history) <= plan["specification"]["fixed"]["epochs"], "Best validation history differs from arrays")
            rows.append(dict(candidate_id=candidate["candidate_id"], seed=seed, validation_mse=mse, result_sha256=sha(folder / "result.json")))
            environments.add(stable(result["environment"]))
    actual_dirs = {p.parent.relative_to(search).as_posix() for p in search.glob("candidate_*/seed_*/result.json")}
    require(actual_dirs == expected_dirs, "Unexpected candidate result set")
    ranking = []
    for candidate in candidates:
        scores = [r["validation_mse"] for r in rows if r["candidate_id"] == candidate["candidate_id"]]
        ranking.append(dict(candidate_id=candidate["candidate_id"], mean_validation_mse=sum(scores) / len(scores),
                            per_seed_validation_mse=scores, parameters=candidate["parameters"]))
    ranking.sort(key=lambda row: (row["mean_validation_mse"], row["candidate_id"]))
    require(selection["ranking"] == ranking and selection["selected"] == ranking[0] and selection["candidate_results"] == rows and
            selection["test_metrics_used"] is False and selection["plan_sha256"] == stable(plan), "Independent validation selection differs")
    selected = ranking[0]
    test_rows = []
    expected_test_dirs = {"seed_%d" % seed for seed in seeds}
    require({p.parent.name for p in search.glob("selected_test/seed_*/result.json")} == expected_test_dirs and state["test_evaluated"] is True,
            "Selected test directory/seed set must match exactly")
    for seed in seeds:
        folder = search / "selected_test" / ("seed_%d" % seed)
        worker, result = read(folder / "worker.json"), read(folder / "result.json")
        validate_worker(worker, plan, folder, "test", selected, seed)
        checkpoint = search / selected["candidate_id"] / ("seed_%d" % seed) / "best_model.pth"
        worker_root = read(checkpoint.parent / "worker.json")["output"].replace("\\", "/")
        require(worker["checkpoint"].replace("\\", "/") == worker_root + "/best_model.pth" and
                worker["checkpoint_sha256"] == sha(checkpoint), "Test checkpoint is not selected same-seed best checkpoint")
        require(worker["selection"].replace("\\", "/").endswith("/selection.json") and
                worker["selection_sha256"] == result["selection_sha256"] == sha(search / "selection.json"), "Test selection hash mismatch")
        require(result["status"] == "complete" and result["worker_identity"] == stable(worker) and result["test_used_for_selection"] is False,
                "Test completion identity differs")
        require(set(result["artifact_sha256"]) == {"test_truth.npy", "test_prediction.npy", "data_identity.json"}, "Missing selected test artifacts")
        for name, checksum in result["artifact_sha256"].items():
            require(sha(folder / name) == checksum, "Selected test completion-time artifact changed")
            artifact_checks += 1
        data = read(folder / "data_identity.json")
        require(data == {"test": {k: rebuilt["test"][k] for k in ("sha256", "batteries")}}, "Rebuilt test tensors differ")
        truth = np.load(folder / "test_truth.npy", allow_pickle=False)
        pred = np.load(folder / "test_prediction.npy", allow_pickle=False)
        require(np.array_equal(truth, rebuilt["test"]["labels"]) and truth.shape == pred.shape and np.isfinite(pred).all(), "Test labels/order/predictions invalid")
        diff = pred.astype(np.float64) - truth.astype(np.float64)
        mse, mae = float(np.mean(diff ** 2)), float(np.mean(np.abs(diff)))
        require(result["test_mse"] == mse and result["test_mae"] == mae, "Test array metric mismatch")
        require(np.all(truth > 0), "Nonpositive SOH cannot define this MAPE")
        test_rows.append(dict(seed=seed, MSE=mse, RMSE=math.sqrt(mse), MAE=mae,
                              MAPE=float(np.mean(np.abs(diff) / truth.astype(np.float64))) * 100, n=len(truth)))
        environments.add(stable(result["environment"]))
    require(len(environments) == 1, "Worker recorded environments differ")
    summary = {name: dict(mean=statistics.mean(row[name] for row in test_rows),
                         sample_sd=statistics.stdev(row[name] for row in test_rows) if len(test_rows) > 1 else None)
               for name in ("MSE", "RMSE", "MAE", "MAPE")}
    return dict(format="pinn4soh-replacement-search-portable-local-audit-v1", status="passed", expected_runs=expected_runs,
                candidate_runs=len(rows), candidate_count=len(candidates), seeds=seeds, selected_test_runs=len(test_rows),
                inventory_sha256_checks=inventory_checks, source_sha256_checks=source_checks, original_csv_sha256_checks=input_checks,
                artifact_sha256_checks=artifact_checks, independent_csv_preprocessing_and_tensor_rebuild=True,
                selection_independently_recomputed_without_importing_search_code=True, selected_same_seed_checkpoints_verified=True,
                plan_sha256=stable(plan), selection_sha256=sha(search / "selection.json"), selected=selected,
                selected_test_per_seed=test_rows, selected_test_summary=summary,
                preprocessing_row_counts={role: record["row_counts"] for role, record in rebuilt.items()},
                limits=["Only explicit replacement XJTU/2C search, not recovered historical grid or all-dataset tuning",
                        "Full-trajectory per-battery normalization and capacity-based 3sigma cleaning use future/test-label distribution",
                        "Cloud GPU smoke evaluated the same test batteries before formal search; its metrics were not inputs to formal selection",
                        "This audit verifies checkpoint byte lineage; it does not execute archived pickle or rerun GPU inference",
                        "Three seeds are repeated optimization runs on one fixed validation/test split, not independent battery-level sampling"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--expected-runs", type=int, default=192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a fresh audit output file")
    result = audit(args.extracted, args.repo, args.collection, args.expected_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "candidate_runs", "candidate_count", "selected_test_runs", "artifact_sha256_checks")}))
