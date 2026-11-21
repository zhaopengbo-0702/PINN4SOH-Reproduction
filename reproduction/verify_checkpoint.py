"""Reload one completed run and compare predictions with its saved best-epoch arrays."""

import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True,
                        help="One ExperimentNN directory containing config/splits/model/npy files")
    parser.add_argument("--runner", type=Path,
                        default=Path(__file__).resolve().with_name("run_reproduction.py"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--report", type=Path)
    options = parser.parse_args()
    repo, run = options.repo.resolve(), options.run.resolve()
    report_path = options.report.resolve() if options.report else run / "verification_checkpoint.json"
    if options.atol < 0 or options.rtol < 0:
        parser.error("Tolerances must be non-negative")
    spec = importlib.util.spec_from_file_location("reproduction_helpers", options.runner.resolve())
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    splits = json.loads((run / "splits.json").read_text(encoding="utf-8"))
    test_split = next(item for item in splits if item["role"] == "test")
    inventory = helpers.source_inventory(repo)
    if inventory["fingerprint"] != config["source_fingerprint"]:
        raise RuntimeError("Source/data hashes differ from this run; prediction comparison is invalid")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if options.device == "cuda" else ""
    os.environ["OMP_NUM_THREADS"] = str(config.get("threads", 2))
    os.environ["MKL_NUM_THREADS"] = str(config.get("threads", 2))
    import numpy as np
    import torch
    torch.set_num_threads(config.get("threads", 2))
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu only for an explicit CPU comparison")
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from dataloader.dataloader import DF
    from Model.Model import PINN
    args = argparse.Namespace(**config["parameters"])
    args.save_folder, args.log_dir = None, None
    data = DF(args)
    loaders = data.load_all_battery(test_split["paths"], test_split["nominal_capacity"])
    testloader = loaders["test_3"]
    expected_count = sum(battery["n_pairs"] for battery in test_split["batteries"])
    if len(testloader.dataset) != expected_count:
        raise RuntimeError("Recorded test boundaries disagree with reloaded sample count")
    model = PINN(args)
    logger = logging.getLogger("mylogger")
    checkpoint = torch.load(run / "model.pth", map_location=options.device)
    if not isinstance(checkpoint, dict) or not {"solution_u", "dynamical_F"} <= set(checkpoint):
        raise ValueError("Checkpoint does not contain both PINN networks")
    model.solution_u.load_state_dict(checkpoint["solution_u"], strict=True)
    model.dynamical_F.load_state_dict(checkpoint["dynamical_F"], strict=True)
    try:
        truth, prediction = model.Test(testloader)
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    saved_truth = np.load(run / "true_label.npy", allow_pickle=False)
    saved_prediction = np.load(run / "pred_label.npy", allow_pickle=False)
    if truth.shape != saved_truth.shape or prediction.shape != saved_prediction.shape:
        raise ValueError("Reloaded and saved array shapes differ")
    finite = all(np.isfinite(array).all() for array in
                 [truth, prediction, saved_truth, saved_prediction])
    labels_equal = bool(np.array_equal(truth, saved_truth))
    predictions_equal = bool(np.array_equal(prediction, saved_prediction))
    predictions_close = bool(np.allclose(prediction, saved_prediction,
                                        rtol=options.rtol, atol=options.atol))
    errors = np.abs(prediction.astype(np.float64) - saved_prediction.astype(np.float64))
    report = {"verified_at": helpers.utc_now(), "run": str(run),
              "device": options.device, "gpu": torch.cuda.get_device_name(0)
              if options.device == "cuda" else None,
              "source_fingerprint_verified": True, "n_test": int(len(truth)),
              "all_arrays_finite": bool(finite), "labels_bitwise_equal": labels_equal,
              "predictions_bitwise_equal": predictions_equal,
              "predictions_allclose": predictions_close,
              "max_absolute_prediction_difference": float(errors.max()),
              "mean_absolute_prediction_difference": float(errors.mean()),
              "atol": options.atol, "rtol": options.rtol,
              "saved_metrics": helpers.calculate_metrics(saved_truth, saved_prediction),
              "reloaded_metrics": helpers.calculate_metrics(truth, prediction),
              "checkpoint_sha256": helpers.digest(run / "model.pth"),
              "saved_prediction_sha256": helpers.digest(run / "pred_label.npy"),
              "passed": bool(finite and labels_equal and predictions_close)}
    helpers.write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("Checkpoint verification FAILED; inspect " + str(report_path))


if __name__ == "__main__":
    main()
