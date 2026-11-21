"""Export source-checkpoint inventory and the two explicit transfer designs."""

import json
from pathlib import Path

import run_transfer as transfer


def main():
    import torch
    root = Path(__file__).resolve().parents[1]
    repo = root / "PINN4SOH"
    checkpoints = []
    reference_shapes = None
    for path in sorted((repo / "pretrained model").glob("*.pth")):
        content = torch.load(path, map_location="cpu", weights_only=True)
        if set(content) != {"solution_u", "dynamical_F"}:
            raise ValueError("Unexpected source checkpoint keys: " + path.name)
        shapes = {module: {name: list(tensor.shape) for name, tensor in state.items()}
                  for module, state in content.items()}
        if reference_shapes is None:
            reference_shapes = shapes
        if shapes != reference_shapes or not all(torch.isfinite(tensor).all()
                                                  for state in content.values() for tensor in state.values()):
            raise ValueError("Incompatible/nonfinite supplied checkpoint: " + path.name)
        checkpoints.append({"path": path.relative_to(repo).as_posix(), "sha256": transfer.common.digest(path),
                            "bytes": path.stat().st_size,
                            "parameter_counts": {name: sum(tensor.numel() for tensor in state.values())
                                                 for name, state in content.items()},
                            "state_sha256": {name: transfer.state_hash(state) for name, state in content.items()},
                            "training_seed": None, "training_split": None, "training_history_verified": False})
    plan = {
        "reference_current_commit": transfer.common.UPSTREAM_COMMIT,
        "source_checkpoint_modes": transfer.SOURCE_MODES,
        "script_native19": {"configurations": transfer.transfer_configurations("script_native19"),
                             "tasks": transfer.build_transfer_tasks("script_native19")},
        "paper_fixed12": {"configurations": transfer.transfer_configurations("paper_fixed12"),
                            "tasks": transfer.build_transfer_tasks("paper_fixed12")},
        "supplied_checkpoints": checkpoints,
        "checkpoint_tensor_shapes": reference_shapes,
        "source_pretraining_policy": "Supplied checkpoint training provenance is unknown. Reproduced mode requires an explicit complete source run with best-validation checkpoint and recorded splits. Source-run selection, when used, minimizes source validation MSE only.",
        "paper_design_conflict": "Table 4 fixes XJTU source/target batch 1 and TJU source/target batch 3. Native XJTU-to-TJU target batch 3 selects XJTU source batch 3. paper_fixed12 explicitly changes that source to batch 1; target-specific script learning rates remain an implementation choice not disclosed by Note 3.",
        "implementation_limit": "The provided checkpoints predate current corrected source. Architecture compatibility does not establish identical original training, identifiable electrochemical dynamics, or paper numerical equivalence.",
    }
    output = root / "reproduction" / "transfer_task_map.json"
    transfer.common.write_json(output, plan)
    print(json.dumps({"output": str(output), "checkpoints": len(checkpoints),
                      "native_configurations": len(plan["script_native19"]["configurations"]),
                      "native_tasks": len(plan["script_native19"]["tasks"]),
                      "paper_configurations": len(plan["paper_fixed12"]["configurations"]),
                      "paper_tasks": len(plan["paper_fixed12"]["tasks"])}))


if __name__ == "__main__":
    main()
