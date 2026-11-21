"""Measure MIT/HUST training-only label/time scales from already saved splits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    values = np.asarray(values)
    return dict(n=int(values.size), minimum=float(values.min()), maximum=float(values.max()),
                mean=float(values.mean()), median=float(np.median(values)),
                q01=float(np.quantile(values, .01)), q99=float(np.quantile(values, .99)))


def main(output):
    if output.exists():
        raise ValueError("New output file required")
    sys.path.insert(0, str(ROOT / "PINN4SOH"))
    from dataloader.dataloader import DF
    data = DF(SimpleNamespace(normalization_method="min-max", log_dir=None, save_folder=None))
    recovered = ROOT / "outputs/full_reproduction/cloud_emergency_recovered/workspace/pinn4soh_full_paper_stage1/tasks/full/paper_equation"
    results = {}
    for dataset in ["MIT", "HUST"]:
        path = recovered / dataset / "all/PINN/Experiment01/splits.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        train = next(r for r in records if r["role"] == "train_validation")
        labels, changes, dt, ratio, cells = [], [], [], [], []
        for name in train["paths"]:
            source = ROOT / "PINN4SOH" / name
            (x1, y1), (x2, y2) = data.load_one_battery(str(source), train["nominal_capacity"])
            delta = y2 - y1
            spacing = x2[:, -1] - x1[:, -1]
            labels.extend(y1)
            changes.extend(delta)
            dt.extend(spacing)
            ratio.extend(delta / spacing)
            cells.append(dict(path=name, sha256=hashlib.sha256(source.read_bytes()).hexdigest(), pairs=len(y1)))
        changes = np.asarray(changes)
        results[dataset] = dict(scope="train_validation batteries only; before author random sample split; no held-out test file loaded",
                                nominal_capacity=train["nominal_capacity"], files=len(cells), source_files=cells,
                                normalized_soh=stats(labels), adjacent_label_delta=stats(changes),
                                positive_delta_fraction=float(np.mean(changes > 0)),
                                normalized_cycle_delta=stats(dt), numerical_soh_derivative_per_normalized_cycle=stats(ratio),
                                mean_positive_label_delta=float(np.maximum(changes, 0).mean()),
                                note="Consecutive cleaned rows can skip raw cycles; cycle coordinate is independently minmax scaled per battery.")
    output.write_text(json.dumps(dict(status="complete", datasets=results, test_files_loaded=False,
                                     implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                                 indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key:{name:value for name,value in record.items() if name != "source_files"} for key,record in results.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
