# PINN4SOH Cloud Run

Prepared on 2026-11-09 (Asia/Shanghai).

## Cloud configuration

- Platform: http://ali.ustb-ai3d.cn:10213
- Account volume: zhaopengbo, mounted at `/workspace`
- Job: `pinn4soh-reproduce-20261109`
- Pool: `node01-3090`; one NVIDIA GeForce RTX 3090 (24576 MiB)
- CPU allocation: 9; RAM allocation: 60 GB
- Image: `public/base:pytorch1131.jupyterlab.tensorboard`
- Observed Python: 3.10.13; torch: 1.13.1+cu116; CUDA available
- Jupyter is disabled; use the platform Web terminal.

## Source provenance

Repository: https://github.com/wang-fujin/PINN4SOH

Verified reference commit: `cc3cc5053caf38f04e0665f7f88cb109144d035e`.
The official source archive was verified locally and after upload:

`9672ecfc35b68c6afb6e596bbc8d5bab33f70eac4f4816d775870078643322e5`

Cloud source directory: `/workspace/PINN4SOH-main`.
The author's `fix.md` describes September 2026 fixes and changed alpha/beta
relative to the paper's supplement. This run targets that current corrected
implementation, with a modern CUDA environment. It is not a claim of exact
recovery of the 2024 paper's numerical results.

## Restore Persistent Dependencies

The container could not validate the HTTPS certificate chain when downloading
from GitHub/PyPI. Official artifacts were downloaded locally with certificate
verification, checked against published hashes, and uploaded through the
platform. The offline package contains all required Linux CPython 3.10 wheels;
the platform's CUDA-enabled torch is retained.

Offline package SHA256:
`da5e2e86bb16aff35db9fc20d8db415c49fdb34c7bb3825b7548480fe7a170f0`

```bash
python -m zipfile -e /workspace/offline-wheels-cp310-linux.zip /workspace/offline
python -m pip install --no-index --find-links /workspace/offline/wheels -r /workspace/offline/requirements-cloud.txt --target /workspace/site-packages
```

## Run And Verify

Smoke test: four datasets, two epochs each, one batch each. This only checks
the complete execution path; its accuracy is not a convergence result.

```bash
PYTHONPATH=/workspace/site-packages python -u /workspace/run_reproduction.py --repo /workspace/PINN4SOH-main --output /workspace/pinn4soh_smoke_20261109 --dataset all --smoke --plots
```

Main experiments: XJTU 60, MIT 10, HUST 10, TJU 30 = 110 runs.
Each uses the author's default hyperparameters, at most 200 epochs and the
original validation-based early stopping. CNN/MLP, small-sample experiments,
and transfer experiments are separate experiments and are not included here.

```bash
PYTHONPATH=/workspace/site-packages python -u /workspace/run_reproduction.py --repo /workspace/PINN4SOH-main --output /workspace/pinn4soh_main_20261109 --dataset all --repeats 10 --epochs 200 --threads 2 --resume --plots
```

The remaining independent runs can share the allocated GPU using the tested
parallel scheduler. The coordinator alone writes the common manifest and
summary. Each worker has its own process, random seed, model, run directory,
and state file. Three workers use two CPU threads each. This changes scheduling,
not the per-run model, training parameters, or existing completed results.

Extract `parallel-tools-20261110.zip` to `/workspace/parallel_tools`. Stop the
old serial training process before launching the scheduler; never run both
writers against the same output directory. `--previous-pid` is a guard and
does not stop that process. A read-only plan is available with `--dry-run`.

```bash
PYTHONPATH=/workspace/site-packages python -u /workspace/parallel_tools/run_parallel.py --repo /workspace/PINN4SOH-main --output /workspace/pinn4soh_main_20261109 --workers 3 --threads 2 --previous-pid 243 --plots
```

Per-run training times under concurrent scheduling include resource contention
and are not directly comparable with the earlier serial timings. The final
manifest records each run's actual start/end and training duration.

```bash
PYTHONPATH=/workspace/site-packages python -u /workspace/verify_checkpoint.py --repo /workspace/PINN4SOH-main --run /workspace/pinn4soh_main_20261109/XJTU/2C/Experiment01
```

Outputs include `manifest.json`, `environment.json`, `source_inventory.json`,
`metrics_per_run.csv`, `metrics_summary.csv`, and a Chinese result report.
Each run contains parameters, seed, actual battery paths/sample boundaries,
training log, prediction arrays, best checkpoint, and per-battery metrics.
The independent runner corrects the best-checkpoint reference-copy problem
without changing training, and recomputes MAPE with true labels as denominator.

Stop the cloud job through the platform after training and verification are
finished. Files in `/workspace` remain on the account's persistent volume.

## Execution Status

See the final execution report for measured training status and results. The
commands above are the reproducible procedure, not evidence of completion.
