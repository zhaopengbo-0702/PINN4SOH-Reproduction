"""Read existing MIT/HUST logs; do not load or rank held-out test predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import median


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, output):
    if output.exists():
        raise ValueError("Use new diagnostic output directory")
    rows, evidence = [], []
    for variant in ["published_code", "paper_equation"]:
        for dataset in ["MIT", "HUST"]:
            base = root / "tasks/full" / variant / dataset / "all/PINN"
            for path in sorted(base.glob("Experiment*/config.json")):
                config = json.loads(path.read_text(encoding="utf-8"))
                parameters = config["parameters"]
                log = path.with_name("logging.txt")
                content = log.read_text(encoding="utf-8")
                valid = [(int(e), float(value)) for e, value in re.findall(r"\[Valid\] epoch:(\d+), MSE: ([\deE.+-]+)", content)]
                if not valid:
                    raise ValueError("No validation trace " + str(path))
                best_epoch, best_mse = min(valid, key=lambda value: value[1])
                physical = path.with_name("physical_loss_diagnostic.json")
                diagnostic = json.loads(physical.read_text(encoding="utf-8"))
                grad = diagnostic["first_batch_output_requires_grad"]
                raw_sum = diagnostic["first_batch_physical_penalty_sum"]
                batch = parameters["batch_size"]
                rows.append(dict(dataset=dataset, variant=variant, repeat=config["task"]["repeat"], seed=config["task"]["seed"],
                                 epochs_completed=valid[-1][0], best_validation_epoch=best_epoch, best_validation_mse=best_mse,
                                 stopped_before_warmup_end=valid[-1][0] < parameters["warmup_epochs"],
                                 best_before_warmup_end=best_epoch < parameters["warmup_epochs"],
                                 batch_size=batch, alpha=parameters["alpha"], beta_sum=parameters["beta"],
                                 equivalent_full_batch_beta_mean=parameters["beta"] * batch,
                                 first_batch_physical_sum=raw_sum, first_batch_weighted_physical=raw_sum * parameters["beta"],
                                 physical_has_parameter_gradient=grad,
                                 lr=parameters["lr"], warmup_lr=parameters["warmup_lr"], lr_F=parameters["lr_F"],
                                 patience=parameters["early_stop"], warmup_epochs=parameters["warmup_epochs"]))
                for source in [path, log, physical]:
                    evidence.append(dict(path=str(source.resolve()), sha256=sha(source)))
    output.mkdir(parents=True)
    with (output / "log_diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    for dataset in ["MIT", "HUST"]:
        for variant in ["published_code", "paper_equation"]:
            values = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant]
            groups.append(dict(dataset=dataset, variant=variant, runs=len(values),
                               stopped_before_warmup_end=sum(r["stopped_before_warmup_end"] for r in values),
                               best_before_warmup_end=sum(r["best_before_warmup_end"] for r in values),
                               completed_epochs_range=[min(r["epochs_completed"] for r in values), max(r["epochs_completed"] for r in values)],
                               best_epoch_range=[min(r["best_validation_epoch"] for r in values), max(r["best_validation_epoch"] for r in values)],
                               median_best_validation_mse=median(r["best_validation_mse"] for r in values),
                               weighted_physical_initial_range=[min(r["first_batch_weighted_physical"] for r in values), max(r["first_batch_weighted_physical"] for r in values)],
                               parameters={k: values[0][k] for k in ["alpha", "beta_sum", "batch_size", "equivalent_full_batch_beta_mean", "lr", "warmup_lr", "lr_F", "patience", "warmup_epochs"]}))
    (output / "summary.json").write_text(json.dumps(dict(status="complete", groups=groups, evidence=evidence,
        test_arrays_loaded=False, training_performed=False, frozen_results_changed=False,
        implementation_sha256=sha(Path(__file__))), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/full_reproduction/cloud_emergency_recovered/workspace/pinn4soh_full_paper_stage1"))
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    main(options.root, options.output)
