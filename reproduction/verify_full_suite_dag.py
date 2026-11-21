"""Exercise missing-yet-dependent source checkpoints without changing the coordinator."""

import argparse
import json
from pathlib import Path

import run_full_suite as suite
from test_full_suite import FAKE_WORKER, FullSuiteTests


def verify():
    fixture = FullSuiteTests()
    fixture.setUp()
    try:
        # Add valid provenance files after the fake model has finished; no training is performed.
        fixture.engine.write_text(FAKE_WORKER + "\n" + r'''
j(o/'splits.json', [])
(o/'logging.txt').write_text('[Valid] epoch:1, MSE: 0.01\n')
record['artifact_sha256']['splits.json']=h(o/'splits.json')
j(o/'manifest.json',record)
''')
        source_id = "pretrain/source"
        target_id = "transfer/downstream"
        source_run = fixture.output / "tasks" / source_id
        checkpoint = source_run / "model.pth"
        source = fixture.task(source_id, family="full", model="PINN", fail_once=True)
        target = fixture.task(target_id, family="transfer", source_mode="reproduced_checkpoint",
                              source_run=str(source_run), source_checkpoint=str(checkpoint),
                              depends_on=[source_id])
        target["runner"] = "transfer"
        tasks = [target, source]
        if checkpoint.exists() or source_run.exists():
            raise AssertionError("Fixture source output should not exist")
        first = fixture.coordinator(tasks)
        if checkpoint.exists():
            raise AssertionError("Preparing a plan must not create a checkpoint")
        if fixture.run_quietly(first) != 1:
            raise AssertionError("Injected upstream failure was not reported")
        initial = fixture.state()
        if initial["tasks"][source_id]["status"] != "failed" or initial["tasks"][target_id]["status"] != "blocked":
            raise AssertionError("Dependency failure was not propagated")
        if "input_sha256" in initial["tasks"][target_id]:
            raise AssertionError("Blocked child tried to fingerprint its nonexistent source")
        if checkpoint.exists():
            raise AssertionError("Failed fixture must not have produced a checkpoint")
        if fixture.run_quietly(fixture.coordinator(tasks, resume=True)) != 0:
            raise AssertionError("Dependency chain did not resume")
        resumed = fixture.state()
        fingerprints = resumed["tasks"][target_id]["input_sha256"]
        required = {str((source_run / name).resolve()) for name in ["model.pth", "config.json", "splits.json", "logging.txt", "manifest.json"]}
        if not required.issubset(fingerprints):
            raise AssertionError("Downstream provenance snapshot is incomplete")
        if resumed["tasks"][source_id]["attempts"] != 2 or resumed["tasks"][target_id]["attempts"] != 1:
            raise AssertionError("Unexpected retry counts")
        modified = checkpoint.stat().st_mtime_ns
        if fixture.run_quietly(fixture.coordinator(tasks, resume=True)) != 0:
            raise AssertionError("Completed dependency chain did not resume")
        if checkpoint.stat().st_mtime_ns != modified:
            raise AssertionError("Completed source checkpoint was rewritten")
        return {"status": "passed", "coordinator_sha256": suite.digest(Path(suite.__file__)),
                "initial_source_outputs_missing": True,
                "construction_did_not_require_future_inputs": True,
                "failed_parent_blocked_child_without_input_fingerprinting": True,
                "resume_retried_parent_then_child": True,
                "source_checkpoint_and_provenance_frozen_after_parent_completion": True,
                "second_resume_preserved_completed_checkpoint": True,
                "source_attempts": 2, "downstream_attempts": 1,
                "final_counts": resumed["counts"], "training_performed": False}
    finally:
        fixture.tearDown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    result = verify()
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        suite.atomic_json(options.output, result)
    print(json.dumps(result, indent=2))
