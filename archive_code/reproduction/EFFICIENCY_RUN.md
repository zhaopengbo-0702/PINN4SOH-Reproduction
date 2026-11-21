# CPU Inference Efficiency

`benchmark_efficiency.py` implements the iteration protocol described under
Supplementary Table S.8: a batch of 1000 samples, 1000 timed forward passes,
and average seconds per batch. All models run on CPU in `eval()` mode inside
`torch.no_grad()`. It does not train or load accuracy checkpoints.

```bash
python -u /workspace/benchmark_efficiency.py \
  --repo /workspace/PINN4SOH-main \
  --output /workspace/efficiency_formal_20261110 \
  --threads 1 2 --batch-size 1000 --forwards 1000 --warmup 50 --repeats 5
```

The output directory must be new or empty. The script hides all GPUs in its
measurement processes and pins OpenMP/MKL/OpenBLAS/NumExpr and PyTorch intra-op
threads to the selected count. PyTorch inter-op threads are fixed to one.
Each repeat/thread pair runs in a fresh sequential subprocess, which measures
all three models. Model order rotates by repeat and is paired across thread
counts. Allocation, model initialization, imports, warmup and output validation
are outside the timer. Timers use `perf_counter_ns`; process CPU time is also
recorded.

The three inference architectures are reused directly from the author source:

| Label | Author Class | Required Parameter Count |
| --- | --- | ---: |
| F | `Model.Model.Solution_u` | 7781 |
| MLP | `Model.Compare_Models.MLP` | 7781 |
| CNN | `Model.Compare_Models.CNN` | 8465 |

F is the inference network, not the training-only dynamical network G and not
the full PINN forward path that computes automatic derivatives. Parameter
counts, finite CPU outputs, shapes, disabled gradients, evaluation state and
unchanged BatchNorm buffers are checked. Inputs are 1000 by 17 float32 tensors;
the seed and input hash permit pairing across models and thread counts.

The parent records the exact source and benchmark hashes. Raw subprocess files
include the actual CPU model/topology, available CPU affinity and cgroup limits
on Linux, Python and PyTorch versions, backend configuration, warmup, iterations,
thread settings, model order and every timing. `summary.json` and `summary.csv`
contain means, sample standard deviations (`ddof=1`), medians, ranges and raw
values across independent process repeats. All produced artifacts are hashed
in `manifest.json`.

The paper reports Intel Core i5-10400F at 2.90 GHz and PyTorch 1.7.1. It does not
state warmup, thread count, eval/no_grad flags or independent process repeats.
This benchmark declares those choices. Descriptive ratios to the published
times do not establish equivalent hardware, software or numerically identical
timing conditions.

## Smoke Check

```bash
python -u /workspace/benchmark_efficiency.py \
  --repo /workspace/PINN4SOH-main \
  --output /workspace/efficiency_smoke --smoke
```

Smoke mode retains batch size 1000 and both thread counts, but uses two warmup
passes, five timed passes and two independent process repeats. It validates the
benchmark implementation and records `smoke: true`; these short measurements
are not the formal 1000-forward result.

## Dependency Check

`verify_full_suite_dag.py` is a separate standard-library fake-worker check.
It does not modify the frozen coordinator or perform training. It verifies
that a future `source_run` and `source_checkpoint` can be absent during plan
construction, that an upstream failure blocks the child without reading those
missing files, and that resume retries the upstream task before capturing the
child's input fingerprints. A second resume must preserve the source model.

```bash
python reproduction/verify_full_suite_dag.py \
  --output outputs/full_reproduction/coordinator_dag_check.json
```

The verifier imports the existing fake-worker test fixture from
`test_full_suite.py`; both files must be present alongside `run_full_suite.py`.
