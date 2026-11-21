"""Read-only CPU reconstruction of saved control loaders; never load a model or train.

Run separately with each stack's isolated Python. The original DF loader is used,
with logging disabled, on the task's saved ordered splits and effective parameters.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def locate(suites, task_id):
    found = [suite / "tasks" / task_id for suite in suites
             if (suite / "tasks" / task_id / "manifest.json").is_file()]
    if len(found) > 1:
        raise ValueError("Ambiguous task identity: " + task_id)
    return found[0] if found else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--suite", type=Path, action="append", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--stack", choices=["legacy", "current"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modern-site", type=Path,
                        help="Explicit persistent modern scientific dependencies only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan, repo, suites = read(args.plan), args.repo.resolve(), [p.resolve() for p in args.suite]
    if args.output.exists():
        raise ValueError("Use a fresh diagnostic output file")
    if any(args.output.resolve() == p or p in args.output.resolve().parents for p in [repo] + suites):
        raise ValueError("Diagnostic output must be outside source and suites")
    if args.modern_site is not None and args.stack != "current":
        raise ValueError("Modern dependencies must never enter the legacy stack")
    if args.dry_run:
        print(json.dumps(dict(training_performed=False, checkpoint_loaded=False,
            tasks=len(plan["tasks"]), stack=args.stack, output=str(args.output),
            diagnostic="CPU only; original loader; no model instantiation")))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = "1"
    if not sys.flags.isolated:
        raise RuntimeError("Run reconstruction with python -I")
    if args.modern_site is not None:
        site = args.modern_site.resolve()
        if not site.is_dir() or site.is_symlink():
            raise ValueError("Explicit modern site must be an ordinary directory")
        sys.path.insert(0, str(site))
    import numpy as np
    import pandas
    import scipy
    import sklearn
    import torch
    if args.stack == "legacy":
        if (sys.version_info[:3] != (3, 7, 10) or torch.__version__ != "1.7.1+cu110"
                or np.__version__ != "1.20.3" or pandas.__version__ != "1.3.5"
                or sklearn.__version__ != "0.24.2" or scipy.__version__ != "1.7.3"):
            raise RuntimeError("Legacy package versions differ from frozen target")
        if any(Path(sys.prefix).resolve() not in Path(m.__file__).resolve().parents
               for m in [torch, np, pandas, scipy, sklearn]):
            raise RuntimeError("Legacy scientific import escaped its prefix")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(repo))
    DF = importlib.import_module("dataloader.dataloader").DF
    document = dict(format="pinn4soh-control-data-reconstruction-v1", stack=args.stack,
        created_at=datetime.now(timezone.utc).isoformat(), plan_sha256=digest(args.plan),
        script_sha256=digest(__file__), loader_sha256=digest(repo / "dataloader/dataloader.py"),
        training_performed=False, checkpoint_loaded=False, model_instantiated=False,
        protocol="Saved ordered splits; original DF.load_all_battery; train_2/valid_2/test_3; logging disabled",
        environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
            pandas=pandas.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
            imports={m.__name__:str(Path(m.__file__).resolve()) for m in [torch,np,pandas,scipy,sklearn]},
            isolated=bool(sys.flags.isolated), threads=torch.get_num_threads(),
            cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"]), tasks=[])
    for task in plan["tasks"]:
        identity = task["task_id"] if args.stack == "legacy" else task["original_task_id"]
        folder = locate(suites, identity)
        record = dict(task_id=identity, legacy_task_id=task["task_id"])
        if folder is None or read(folder / "manifest.json").get("status") != "complete":
            document["tasks"].append(dict(record, status="pending"))
            continue
        try:
            config, splits = read(folder / "config.json"), read(folder / "splits.json")
            parameters = dict(config.get("parameters") or read(folder / "parameters.json"))
            parameters.update(log_dir=None, save_folder=None)
            if task["validation_protocol"] != "author_sample_split":
                raise ValueError("Only frozen sample-split controls are supported")
            loader = DF(SimpleNamespace(**parameters))
            roles, input_hashes = {}, {}
            for split in splits:
                paths = []
                for name in split["paths"]:
                    # Author split paths are repository-relative; reject escaping the repo.
                    path = (repo / name).resolve()
                    if repo not in path.parents:
                        raise ValueError("Input path escaped repository")
                    paths.append(str(path))
                    input_hashes[name.replace("\\", "/")] = digest(path)
                values = loader.load_all_battery(paths, split.get("nominal_capacity"))
                if split["role"] in {"train_validation", "train"}:
                    roles.update(train=values["train_2"], valid=values["valid_2"])
                elif split["role"] == "test":
                    roles["test"] = values["test_3"]
            tensors = {role:[hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                            for t in value.dataset.tensors] for role, value in roles.items()}
            labels = roles["test"].dataset.tensors[2].numpy()
            retained = np.load(folder / "true_label.npy", allow_pickle=False)
            if not np.array_equal(labels, retained):
                raise ValueError("Reconstructed test labels differ from retained training result")
            stored = read(folder / "data_identity.json") if (folder / "data_identity.json").is_file() else None
            if stored is not None and (stored["tensor_sha256"] != tensors or stored["sample_counts"] !=
                                      {role:len(value.dataset) for role,value in roles.items()}):
                raise ValueError("Reconstruction differs from tensor hashes recorded during training")
            record.update(status="verified", config_sha256=digest(folder / "config.json"),
                splits_sha256=digest(folder / "splits.json"), parameters_sha256=digest(folder / "parameters.json")
                if (folder / "parameters.json").is_file() else None,
                tensor_sha256=tensors, input_file_sha256=input_hashes,
                sample_counts={role:len(value.dataset) for role,value in roles.items()},
                retained_labels_bitwise_equal=True, matched_training_time_tensor_record=stored is not None)
        except Exception as error:
            record.update(status="failed", error=repr(error))
        document["tasks"].append(record)
        print(json.dumps({key:record[key] for key in ["task_id", "status"]}), flush=True)
    document["assessment"] = "complete" if all(r["status"] == "verified" for r in document["tasks"]) else "pending_or_failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(assessment=document["assessment"], output=str(args.output))))


if __name__ == "__main__":
    main()
