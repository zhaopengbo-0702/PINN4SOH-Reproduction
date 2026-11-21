"""Meaningful functional checks for the separate Stage A improvement protocol."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from search_replacement_grid import read, write, sha
from search_stage_a import battery_metrics, mono_eval_with_grad, should_stop, lr_ratio, build_plan


def main():
    import numpy as np
    import torch
    root = Path(__file__).resolve().parents[1]
    # Macro means each physical battery's RMSE, not pooled RMSE; repeated CSV
    # segments of the same UID must be pooled within that physical battery.
    truth = np.array([1.] * 101)
    prediction = truth + np.array([.1] * 100 + [.9])
    metrics = battery_metrics(truth, prediction, [dict(physical_id="one", n_pairs=50), dict(physical_id="one", n_pairs=50), dict(physical_id="two", n_pairs=1)])
    assert abs(metrics["battery_macro_RMSE"] - .5) < 1e-15
    assert abs(metrics["pooled_MSE"] ** .5 - .5) > .1
    solution = torch.nn.Sequential(torch.nn.Dropout(.8), torch.nn.Linear(2, 1))
    solution.train()
    x = torch.ones(8, 2)
    with mono_eval_with_grad(solution):
        first, second = solution(x), solution(x)
        assert torch.equal(first, second) and first.requires_grad and not solution.training
        first.sum().backward()
    assert solution.training and solution[1].weight.grad.abs().sum() > 0
    schedule = dict(min_epochs=30, patience=30, warmup_epochs=5, warmup_ratio=.1, final_lr_ratio=.01, epochs=200)
    assert not should_stop(29, 100, schedule) and should_stop(30, 30, schedule) and not should_stop(30, 29, schedule)
    assert lr_ratio(1, schedule) == .1 and lr_ratio(5, schedule) == 1 and abs(lr_ratio(200, schedule) - .01) < 1e-15
    print("PASS physical-battery macro objective, eval branch retains gradients, minimum epochs/patience and schedule", flush=True)
    with tempfile.TemporaryDirectory(prefix="stage_a_fixture_") as tmp:
        directory = Path(tmp)
        paths = []
        for battery in range(4):
            path = directory / ("battery%d.csv" % battery)
            n = 41 + battery * 3
            cycle = np.arange(n)
            features = np.stack([np.sin((cycle + battery + 1) * (i + 1) / 50) + cycle / 100 for i in range(16)], axis=1)
            np.savetxt(path, np.column_stack([features, 2 * (1 - cycle * .001)]), delimiter=",", comments="",
                       header=",".join(["feature%d" % i for i in range(16)] + ["capacity"]))
            paths.append(str(path))
        base = dict(label="STAGE_A_SYNTHETIC_FUNCTIONAL_ONLY", dataset="MIT", model="PINN", smoke=True,
            seeds=[2026111101], splits=dict(train=dict(paths=paths[:1], physical_ids=["train"], nominal_capacity=2),
                valid=dict(paths=paths[1:3], physical_ids=["valid1", "valid2"], nominal_capacity=2),
                test=dict(paths=paths[3:], physical_ids=["test"], nominal_capacity=2)),
            fixed=dict(epochs=2, min_epochs=2, patience=1, warmup_epochs=1, batch_size=64, threads=1,
                normalization_method="min-max", diagnostic_batch_size=8, grad_clip_norm=0, warmup_ratio=.1, final_lr_ratio=.01,
                deterministic=True, allow_tf32=False), structure=dict(u_layers=3,u_hidden=60,F_layers_num=3,F_hidden_dim=60),
            candidates=[dict(lr_u=.001,lr_F=0,alpha=0,beta_mean=0,weight_decay=0),
                        dict(lr_u=.001,lr_F=.001,alpha=.1,beta_mean=.01,weight_decay=0)],
            protocol_evidence=["Synthetic disjoint physical IDs; functional test only"])
        invalid = copy.deepcopy(base)
        invalid["splits"]["test"]["physical_ids"] = ["train"]
        try:
            build_plan(invalid, root / "PINN4SOH")
            raise AssertionError("Physical overlap accepted")
        except ValueError:
            pass
        records = []
        for model in ("PINN", "MLP", "CNN"):
            config = copy.deepcopy(base)
            config["model"] = model
            if model != "PINN":
                config["candidates"][1] = dict(lr_u=.001,lr_F=0,alpha=0,beta_mean=0,weight_decay=1e-4)
            spec, output = directory / (model + ".json"), directory / model
            write(spec, config)
            command = [sys.executable,str(root/'reproduction/search_stage_a.py'),'--spec',str(spec),'--repo',str(root/'PINN4SOH'),
                '--output',str(output),'--device','cpu','--execute','--allow-cpu','--evaluate-test']
            p = subprocess.run(command, text=True, capture_output=True)
            if p.returncode:
                print(p.stderr)
                for path in output.rglob("console.log"):
                    print(path.read_text(encoding="utf-8"))
            assert p.returncode == 0
            assert read(output / "status.json")["completed_runs"] == 2
            selection = read(output / "selection.json")
            assert selection["test_metrics_used"] is False
            for path in output.glob("candidate_*/seed_*/result.json"):
                result, worker = read(path), read(path.with_name("worker.json"))
                assert result["checkpoint_reload_bitwise_prediction_equal"] and set(worker["splits"]) == {"train","valid"}
                assert paths[3] not in json.dumps(worker)
                history = read(path.with_name("history.json"))
                diagnostic = read(path.with_name("gradient_diagnostics.json"))
                assert all(row["mono_requires_grad"] for row in diagnostic)
                assert all(row["gradients_finite"] and row["mono_branch_mode"] == "eval_with_grad" for row in history)
            before = {str(path): (sha(path),path.stat().st_mtime_ns) for path in output.rglob('*') if path.name in {'result.json','selection.json'}}
            resumed = subprocess.run(command+['--resume'],text=True,capture_output=True)
            assert resumed.returncode == 0,resumed.stderr
            assert all((sha(path),Path(path).stat().st_mtime_ns)==value for path,value in before.items())
            records.append(dict(model=model,candidates=2,epochs=2,status="passed"))
            print("PASS " + model + ": mean/eval branch, gradients, battery macro, best reload, test isolation, resume",flush=True)
        # Alpha=beta=0 PINN must be an exact controlled supervised counterpart to
        # MLP under identical initialization, paired exposure, RNG and optimizer.
        prefix = "candidate_01/seed_2026111101/"
        pin, mlp = directory/'PINN'/prefix, directory/'MLP'/prefix
        assert read(pin/'result.json')['initial_solution_sha256']==read(mlp/'result.json')['initial_solution_sha256']
        assert read(pin/'data_identity.json')==read(mlp/'data_identity.json')
        assert np.array_equal(np.load(pin/'valid_prediction.npy'),np.load(mlp/'valid_prediction.npy'))
        print("PASS supervised PINN alpha=beta=0 exactly matches MLP initialization/tensors/prediction",flush=True)
    print(json.dumps(dict(status="passed",records=records)))
    print("ALL_STAGE_A_FUNCTIONAL_TESTS_PASSED")


if __name__ == '__main__':
    main()
