"""Trace exact battery/original-row identities through unmodified author loaders."""

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys


LOADERS = {"XJTU": "load_data", "MIT": "load_MIT_data", "HUST": "load_HUST_data", "TJU": "load_TJU_data"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def rng_snapshot(np, torch):
    state = np.random.get_state()
    return {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            "numpy": hashlib.sha256(state[0].encode() + state[1].tobytes() + repr(state[2:]).encode()).hexdigest(),
            "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}


class ProvenanceTrace:
    def __init__(self, module, repo):
        self.module, self.repo = module, repo
        self.frames, self.active, self.datasets, self.current_read = [], [], {}, []

    def battery_id(self, path):
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self.repo).as_posix()
        except ValueError:
            return str(resolved)

    @contextlib.contextmanager
    def capture(self):
        import numpy as np
        import torch
        df = self.module.DF
        original = {key: getattr(df, key) for key in ["delete_3_sigma", "read_one_csv", "load_one_battery", "load_all_battery"]}
        original_split = self.module.train_test_split
        owner = self

        def delete_sigma(instance, frame):
            result = original["delete_3_sigma"](instance, frame)
            if owner.current_read:
                cycles = result["cycle index"].to_numpy(copy=True)
                converted = cycles.astype(np.int64)
                if not np.array_equal(cycles, converted):
                    raise ValueError("Original generated cycle index is not integral")
                owner.current_read[-1]["retained_original_rows"] = converted.tolist()
            return result

        def read_csv(instance, path, nominal_capacity=None):
            record = {"battery_id": owner.battery_id(path), "path": str(Path(path).resolve()), "sha256": digest(path)}
            owner.current_read.append(record)
            try:
                result = original["read_one_csv"](instance, path, nominal_capacity)
                if len(record["retained_original_rows"]) != len(result):
                    raise ValueError("Original-row trace does not match preprocessed data")
                if owner.active:
                    owner.active[-1]["batteries"].append(record)
                return result
            finally:
                owner.current_read.pop()

        def load_one(instance, path, nominal_capacity=None):
            result = original["load_one_battery"](instance, path, nominal_capacity)
            if owner.active:
                frame = owner.active[-1]
                battery = frame["batteries"][-1]
                rows = battery["retained_original_rows"]
                if len(result[0][0]) != len(rows) - 1:
                    raise ValueError("Adjacent-pair provenance length mismatch")
                frame["pairs"].extend((battery["battery_id"], first, second) for first, second in zip(rows[:-1], rows[1:]))
                frame["arrays"].append((result[0][0], result[1][0], result[0][1], result[1][1]))
            return result

        def traced_split(*arrays, **kwargs):
            if not owner.active or len(arrays) != 4:
                raise ValueError("Unexpected author split call")
            frame = owner.active[-1]
            size = len(arrays[0])
            # The added index array is split by the same single sklearn permutation.
            before = rng_snapshot(np, torch)
            outputs = original_split(*arrays, np.arange(size, dtype=np.int64), **kwargs)
            after = rng_snapshot(np, torch)
            train_indices, valid_indices = outputs[-2].tolist(), outputs[-1].tolist()
            for column, values in enumerate(arrays):
                expected = torch.from_numpy(np.concatenate([part[column] for part in frame["arrays"]], axis=0)).float()
                if column >= 2:
                    expected = expected.view(-1, 1)
                if not torch.equal(values, expected[:size]):
                    raise ValueError("Split inputs are not the traced concatenated prefix")
                if not torch.equal(outputs[2 * column], values[train_indices]) or not torch.equal(outputs[2 * column + 1], values[valid_indices]):
                    raise ValueError("Identity indices do not reproduce actual split tensors")
            frame["splits"].append({"n_input_pairs": size, "test_size": kwargs.get("test_size"),
                                    "random_state": kwargs.get("random_state"),
                                    "train_indices": train_indices, "valid_indices": valid_indices,
                                    "global_rng_unchanged": before == after})
            return outputs[:-2]

        def load_all(instance, path_list, nominal_capacity):
            frame = {"paths": list(path_list), "nominal_capacity": nominal_capacity, "batteries": [],
                     "pairs": [], "arrays": [], "splits": []}
            owner.frames.append(frame)
            owner.active.append(frame)
            try:
                result = original["load_all_battery"](instance, path_list, nominal_capacity)
            finally:
                owner.active.pop()
            if len(frame["splits"]) != 2:
                raise ValueError("Author loader no longer makes its two expected split calls")
            pairs = frame["pairs"]
            first, second = frame["splits"]
            mappings = {"train": [pairs[index] for index in first["train_indices"]],
                        "valid": [pairs[index] for index in first["valid_indices"]],
                        "test": pairs[first["n_input_pairs"]:],
                        "train_2": [pairs[index] for index in second["train_indices"]],
                        "valid_2": [pairs[index] for index in second["valid_indices"]], "test_3": pairs}
            for name, loader in result.items():
                if len(loader.dataset) != len(mappings[name]):
                    raise ValueError("Dataset identity length mismatch: " + name)
                owner.datasets[id(loader.dataset)] = mappings[name]
            return result

        df.delete_3_sigma, df.read_one_csv = delete_sigma, read_csv
        df.load_one_battery, df.load_all_battery = load_one, load_all
        self.module.train_test_split = traced_split
        try:
            yield self
        finally:
            for key, value in original.items():
                setattr(df, key, value)
            self.module.train_test_split = original_split

    def public_frames(self):
        return [{key: value for key, value in frame.items() if key != "arrays"} for frame in self.frames]


def endpoint_counts(train, valid, test):
    train_x1 = {(battery, first) for battery, first, _ in train}
    train_x2 = {(battery, second) for battery, _, second in train}
    all_training = train_x1 | train_x2

    def aggregate(pairs):
        first = [(battery, row) for battery, row, _ in pairs]
        n = len(first)
        counts = {"n_evaluated_x1": n,
                  "x1_in_train_x1": sum(point in train_x1 for point in first),
                  "x1_in_train_x2": sum(point in train_x2 for point in first),
                  "x1_in_train_either_endpoint": sum(point in all_training for point in first)}
        counts.update({key + "_fraction": value / n if n else None for key, value in list(counts.items()) if key != "n_evaluated_x1"})
        return counts

    batteries = sorted({battery for battery, _, _ in valid})
    return {"n_train_pairs": len(train), "n_valid_pairs": len(valid), "n_test_pairs": len(test),
            "train_unique_x1_endpoints": len(train_x1), "train_unique_x2_endpoints": len(train_x2),
            "train_unique_either_endpoints": len(all_training),
            "valid": aggregate(valid), "test": aggregate(test),
            "valid_per_battery": [{"battery_id": battery, **aggregate([pair for pair in valid if pair[0] == battery])} for battery in batteries],
            "shared_train_valid_battery_ids": sorted({pair[0] for pair in train} & {pair[0] for pair in valid}),
            "shared_train_test_battery_ids": sorted({pair[0] for pair in train} & {pair[0] for pair in test})}


def verify_identity_labels(loaders, provenance):
    import torch
    labels = {}
    comparisons = 0
    for role, loader in loaders.items():
        pairs = provenance[role]
        tensors = loader.dataset.tensors
        for index, (battery, first, second) in enumerate(pairs):
            for endpoint, row, values in [("x1", first, tensors[2]), ("x2", second, tensors[3])]:
                key = battery, row
                value = values[index]
                if key in labels:
                    if not torch.equal(labels[key], value):
                        raise ValueError("The same battery/original row has inconsistent labels")
                    comparisons += 1
                else:
                    labels[key] = value.clone()
    return {"same_identity_label_comparisons": comparisons, "all_bitwise_equal": True}


def map_path(value, mappings, repo):
    text = str(value)
    for old, new in mappings:
        if text == old or text.startswith(old.rstrip("/\\") + "/") or text.startswith(old.rstrip("/\\") + "\\"):
            text = new.rstrip("/\\") + text[len(old):]
            break
    path = Path(text)
    return str(path.resolve() if path.is_absolute() else (repo / path).resolve())


def load_author_arguments(module):
    previous = sys.argv
    try:
        sys.argv = [module.__name__ + ".py"]
        return module.get_args()
    finally:
        sys.argv = previous


def replay(options, sample_size, config, split_records, perturb_rng=False, instrument=True):
    import numpy as np
    import torch
    dataset = config.get("task", {}).get("dataset", config.get("parameters", {}).get("data", options.dataset))
    module = importlib.import_module("main_" + dataset)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = load_author_arguments(module)
    for key, value in config.get("parameters", {}).items():
        setattr(args, key, value)
    args.save_folder, args.log_dir = None, None
    args.normalization_method = getattr(args, "normalization_method", options.normalization_method)
    if not config:
        args.normalization_method = options.normalization_method
        args.batch = int(options.batch) if dataset == "TJU" else options.batch
        if sample_size is not None:
            args.batch_size = 128
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    if perturb_rng:
        random.random()
        np.random.standard_normal(13)
        torch.rand(13)
        from Model.Model import Solution_u
        Solution_u()
    before = rng_snapshot(np, torch)
    data_module = importlib.import_module("dataloader.dataloader")
    trace = ProvenanceTrace(data_module, options.repo)
    with trace.capture() if instrument else contextlib.nullcontext():
        if split_records:
            from dataloader.dataloader import DF
            data = DF(args)
            by_role = {item["role"]: item for item in split_records}
            if "train_validation" in by_role:
                training = by_role["train_validation"]
                testing = by_role["test"]
                train = data.load_all_battery(training["paths"], training["nominal_capacity"])
                test = data.load_all_battery(testing["paths"], testing["nominal_capacity"])
                loaders = {"train": train["train_2"], "valid": train["valid_2"], "test": test["test_3"]}
            elif set(by_role) == {"train", "valid", "test"}:
                loaders = {role: data.load_all_battery(by_role[role]["paths"], by_role[role]["nominal_capacity"])["test_3"]
                           for role in ["train", "valid", "test"]}
            else:
                raise ValueError("Unsupported recorded split roles")
        else:
            loaders = getattr(module, LOADERS[dataset])(args, small_sample=sample_size)
    after = rng_snapshot(np, torch)
    hashes = {role: [hashlib.sha256(tensor.numpy().tobytes()).hexdigest() for tensor in loader.dataset.tensors]
              for role, loader in loaders.items()}
    if not instrument:
        return {"tensor_sha256": hashes, "global_rng_unchanged_by_loading": before == after}
    provenance = {role: trace.datasets[id(loader.dataset)] for role, loader in loaders.items()}
    labels = verify_identity_labels(loaders, provenance)
    return {"sample_size": sample_size, "seed": options.seed, "dataset": dataset, "parameters": vars(args),
            "counts": endpoint_counts(provenance["train"], provenance["valid"], provenance["test"]),
            "provenance": provenance, "loader_trace": trace.public_frames(), "tensor_sha256": hashes,
            "rng_before_loading": before, "rng_after_loading": after, "global_rng_unchanged_by_loading": before == after,
            "identity_label_verification": labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, help="Replay config.json/splits.json; verify data_identity.json when present")
    parser.add_argument("--dataset", choices=list(LOADERS), default="XJTU")
    parser.add_argument("--batch", default="2C")
    parser.add_argument("--sample-sizes", nargs="+", default=None, help="Native loader audit, default: all 1")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--normalization-method", choices=["min-max", "z-score"], default="min-max")
    parser.add_argument("--path-map", action="append", default=[], help="Explicit OLD_PREFIX=NEW_PREFIX for recorded paths")
    options = parser.parse_args()
    options.repo, options.output = options.repo.resolve(), options.output.resolve()
    if options.run:
        options.run = options.run.resolve()
    if (options.output == options.repo or options.repo in options.output.parents or options.output in options.repo.parents
            or options.run and (options.output == options.run.resolve() or options.run.resolve() in options.output.parents)):
        raise ValueError("Audit output must be separate from the author repository and recorded run")
    if options.output.exists() and any(options.output.iterdir()):
        raise ValueError("Audit output must be new or empty")
    config, split_records, mappings = {}, None, []
    for item in options.path_map:
        old, separator, new = item.partition("=")
        if not separator or not old or not new:
            raise ValueError("Expected --path-map OLD_PREFIX=NEW_PREFIX")
        mappings.append((old, new))
    if options.run:
        config = read_json(options.run / "config.json")
        options.dataset = config.get("task", {}).get("dataset", config.get("parameters", {}).get("data", options.dataset))
        split_records = read_json(options.run / "splits.json")
        for item in split_records:
            item["paths"] = [map_path(path, mappings, options.repo) for path in item["paths"]]
        recorded_seed = config.get("seed", config.get("task", {}).get("seed"))
        if recorded_seed is None:
            raise ValueError("Recorded run does not declare its seed")
        if options.seed is not None and options.seed != recorded_seed:
            raise ValueError("--seed differs from recorded run")
        options.seed = recorded_seed
        sample_sizes = [config.get("task", {}).get("sample_size")]
        if options.sample_sizes is not None:
            raise ValueError("--sample-sizes applies only to native loader diagnostics")
    else:
        options.seed = 20261109 if options.seed is None else options.seed
        sample_sizes = [None if item == "all" else int(item) for item in (options.sample_sizes or ["all", "1"])]
        if any(item is not None and not 1 <= item <= 4 for item in sample_sizes):
            raise ValueError("Sample sizes must be all or 1..4")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import pandas as pd
    import sklearn
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(options.repo))
    previous_directory = Path.cwd()
    source_names = ["dataloader/dataloader.py", "Model/Model.py", "main_" + options.dataset + ".py", "main_comparision.py"]
    source_hashes = {name: digest(options.repo / name) for name in source_names}
    options.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    recorded_inventory = read_json(options.run / "source_inventory.json").get("sha256", {}) if options.run and (options.run / "source_inventory.json").is_file() else {}
    recorded_input_hashes = read_json(options.run / "data_identity.json").get("input_file_sha256", {}) if options.run and (options.run / "data_identity.json").is_file() else {}
    expected_inputs = {map_path(path, mappings, options.repo): checksum for path, checksum in recorded_input_hashes.items()}
    for name, actual in source_hashes.items():
        if name in recorded_inventory and recorded_inventory[name] != actual:
            raise ValueError("Current author source differs from recorded run: " + name)
    try:
        os.chdir(options.repo)
        for size in sample_sizes:
            result = replay(options, size, config, split_records)
            unmodified = replay(options, size, config, split_records, instrument=False)
            preserved = result["tensor_sha256"] == unmodified["tensor_sha256"]
            if not preserved:
                raise ValueError("Instrumentation changed actual loader tensors")
            perturbed = replay(options, size, config, split_records, perturb_rng=True)
            invariant = result["tensor_sha256"] == perturbed["tensor_sha256"] and result["provenance"] == perturbed["provenance"]
            if not invariant:
                raise ValueError("Model/global RNG perturbation changed validation membership; exact initialization replay is required")
            source_checks = 0
            for frame in result["loader_trace"]:
                for battery in frame["batteries"]:
                    expected = expected_inputs.get(battery["path"], recorded_inventory.get(battery["battery_id"]))
                    if expected is not None:
                        if expected != battery["sha256"]:
                            raise ValueError("Loaded CSV differs from recorded run: " + battery["path"])
                        source_checks += 1
            original_match = None
            if options.run and (options.run / "data_identity.json").is_file():
                original_match = read_json(options.run / "data_identity.json")["tensor_sha256"] == result["tensor_sha256"]
                if not original_match:
                    raise ValueError("Replayed tensors do not match the recorded run's data_identity.json")
            name = "recorded_run" if options.run else "nall" if size is None else "n" + str(size)
            destination = options.output / name
            destination.mkdir()
            write_json(destination / "trace.json", {key: value for key, value in result.items() if key != "provenance"})
            with (destination / "cycle_provenance.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["role", "dataset_position", "battery_id", "x1_original_row_0based", "x2_original_row_0based"])
                for role, pairs in result["provenance"].items():
                    writer.writerows((role, index, *pair) for index, pair in enumerate(pairs))
            counts = result["counts"]
            summary = {"name": name, "dataset": result["dataset"], "sample_size": size, "seed": options.seed,
                       **counts, "global_rng_unchanged_by_loading": result["global_rng_unchanged_by_loading"],
                       "membership_invariant_to_model_and_global_rng_perturbation": invariant,
                       "instrumentation_preserved_unmodified_loader_tensors": preserved,
                       "recorded_input_hash_checks": source_checks,
                       "original_run_tensor_hash_match": original_match,
                       "split_random_states": sorted({split["random_state"] for frame in result["loader_trace"] for split in frame["splits"]}),
                       "PINN_validation_label_reuse_fraction": counts["valid"]["x1_in_train_either_endpoint_fraction"],
                       "baseline_validation_label_reuse_fraction": counts["valid"]["x1_in_train_x1_fraction"]}
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    finally:
        os.chdir(previous_directory)
    if source_hashes != {name: digest(options.repo / name) for name in source_names}:
        raise ValueError("Author source files changed during audit")
    report = {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
              "audit_sha256": digest(__file__), "source_sha256": source_hashes,
              "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                              "pandas": pd.__version__, "sklearn": sklearn.__version__},
              "recorded_run": str(options.run.resolve()) if options.run else None,
              "method": "Trace generated cycle index before normalization, augment actual sklearn call with one integer identity array, retain exact tensors",
              "identity": "Canonical battery CSV path plus original zero-based CSV data-row index, before dropna/outlier removal",
              "normalization_scope": "Original per-battery full-lifetime preprocessing retained; this audit quantifies endpoint membership only",
              "interpretation": "PINN supervises y1 and y2; original CNN/MLP supervise y1 only. Endpoint overlap is not held-out test-battery leakage or a quantified accuracy bias.",
              "rng": "Author split fixes random_state=420; run seed is replayed, no model initialization is required for membership. Separate perturbation consumes all global RNGs and initializes Solution_u to verify invariance.",
              "training_performed": False, "formal_run_modified": False, "rows": summaries}
    write_json(options.output / "summary.json", report)
    write_json(options.output / "artifact_sha256.json", {str(path.relative_to(options.output)): digest(path)
               for path in options.output.rglob("*") if path.is_file() and path.name != "artifact_sha256.json"})


if __name__ == "__main__":
    main()
