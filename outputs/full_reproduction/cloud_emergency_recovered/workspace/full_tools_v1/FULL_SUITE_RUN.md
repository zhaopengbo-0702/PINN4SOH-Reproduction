# Full Suite Coordinator

`run_full_suite.py` uses only the Python standard library. It schedules existing
`run_extended.py` and `run_transfer.py` workers without importing their training
modules. A separate, once-per-GPU Python process probes the actual training
environment before execution. Dry runs and summaries do not probe CUDA or import
training dependencies.

```bash
python -u /workspace/run_full_suite.py \
  --plan /workspace/full_plan.json \
  --repo /workspace/PINN4SOH-main \
  --output /workspace/full_suite \
  --gpus 0 1 --workers-per-gpu 3 --threads 2 --resume
```

The output must be separate from the author repository. Six workers is the
maximum for the command above; dependencies can reduce the active count.
Each worker receives one assigned `CUDA_VISIBLE_DEVICES` value and the declared
CPU thread limits. GPU assignment can change on resume when the probed hardware
and software signatures remain identical.

## Plan

```json
{
  "tasks": [
    {
      "task_id": "full/XJTU/2C/CNN/01",
      "runner": "extended",
      "family": "full",
      "dataset": "XJTU",
      "batch": "2C",
      "model": "CNN",
      "repeat": 1,
      "seed": 20261109,
      "epochs": 200,
      "threads": 2,
      "protocol_id": "author_current",
      "validation_protocol": "author_sample_split",
      "reference_run": "/workspace/pinn4soh_main_20261109/XJTU/2C/Experiment01"
    }
  ]
}
```

Every task declares `runner: "extended"` or `runner: "transfer"`. Other fields
are passed to the selected worker unchanged, except a missing `threads` field
is filled from `--threads`. An explicitly different thread count is rejected.
`depends_on: ["another/task/id"]` declares prerequisites; `dependencies` is an
accepted alias. Unknown IDs, duplicate IDs, cycles, absolute paths, parent
traversal, noncanonical paths, and nested task output directories are rejected.
IDs are ASCII path segments containing letters, numbers, dots, underscores, or
hyphens, with platform-reserved names excluded.

The coordinator always passes `--resume` to each worker. Workers retain ownership
of their own manifest and archive their own incomplete output when supported.
The shared coordinator manifest never acts as a worker manifest. A failed task
does not prevent independent tasks from running. Its dependents are marked
`blocked`; a later identical `--resume` retries failed tasks and then their
dependents.

## Integrity And Ownership

- `manifest.json`: coordinator-owned atomic state, plan/source/code/environment
  fingerprints, per-task status, attempts, GPU, PID creation identity, process
  log, input fingerprints, and verified output hashes.
- `tasks/<task_id>/manifest.json`: worker-owned task manifest.
- `.coordinator/`: immutable submitted task JSON and per-attempt process logs,
  outside worker-owned output directories.
- `.full_suite.lock`: OS advisory lock held for the coordinator's lifetime.
  The lock file is intentionally retained after exit; its presence alone does
  not mean a coordinator is active.
- `plan.json` and `source_inventory.json`: initial audited inputs.

Resume freezes the canonical plan hash, source/data and directory-order hashes,
coordinator/runner/helper code, thread/smoke settings, and normalized runtime
environment. Worker completion is accepted only after checking its raw task
hash, source fingerprint, expected implementation hashes, core artifact hashes,
actual artifact bytes, smoke mode, environment, and external reference inputs.
All files in an accepted completed task directory are then hashed by the
coordinator, including the worker manifest. Resume checks that entire file tree
again before skipping. Updating both an artifact and its worker manifest cannot
bypass the coordinator's stored fingerprints.

Reference splits, referenced battery CSVs, source checkpoints, source training
configuration/splits/logging/manifest, and optional parameter files are recorded
when each task becomes runnable. This permits a dependency to produce a source
checkpoint before the downstream input snapshot is taken.

A different writer's manifest is refused. A prior worker with a still-matching
PID and creation time is refused rather than duplicated. Ctrl+C terminates only
the subprocess handles created by this coordinator; unfinished tasks remain
eligible for a later resume. `kill -9` cannot run cleanup, so a surviving prior
worker must finish before a new coordinator can reuse its task output. A durable
`launching` intent is saved before `Popen`: a hard crash before worker PID
publication leaves an ambiguous launch, and resume refuses it pending process
inspection. This prevents silently duplicating even a worker without its own
task lock.

## Inspection And Verification

```bash
python /workspace/run_full_suite.py --plan /workspace/full_plan.json \
  --repo /workspace/PINN4SOH-main --output /workspace/full_suite --dry-run
python /workspace/run_full_suite.py --output /workspace/full_suite --summary
python reproduction/test_full_suite.py
```

`--smoke` forwards two-epoch smoke mode to each worker and is frozen separately
from formal execution. Smoke and formal suites must use different output roots.
`--allow-cpu` is available only for explicitly requested CPU verification.
Custom worker/helper locations can be supplied with `--extended-runner`,
`--transfer-runner`, and `--helpers`; transfer resolves its common helper next
to its own script, and that actual file is separately fingerprinted.

The tests use fake subprocess workers and temporary repositories. They cover
concurrency limits, dependencies, failure/retry, immutable completed outputs,
input and artifact changes, configuration/code/environment freezes, invalid
paths, locks, prior-worker identity, interruption, smoke separation, and
training-free dry-run/summary execution. They are scheduling tests, not model
convergence evidence.
