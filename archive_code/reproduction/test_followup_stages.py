import json
from pathlib import Path
import tempfile
import unittest

from execute_followup_stages import main_ready


class PipelineGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def pipeline(self, statuses):
        self.write("pinn4soh_full_paper_pipeline.json", {"stages": [
            dict(plan=plan, status=status, returncode=0 if status == "complete" else 1)
            for plan, status in zip(["smoke_plan.json", "stage1_plan.json",
                                    "stage2_reproduced_transfer_plan.json"], statuses)]})

    def suites(self, count1=1880, count2=240):
        for name, count in [("stage1", count1), ("stage2", count2)]:
            self.write("pinn4soh_full_paper_" + name + "/manifest.json", {
                "status": "complete", "tasks": {str(index): {"status": "complete"} for index in range(count)}})

    def test_waits_for_all_stages_and_rejects_failure(self):
        self.assertFalse(main_ready(self.root))
        self.pipeline(["complete", "running"])
        self.assertFalse(main_ready(self.root))
        self.pipeline(["complete", "failed"])
        with self.assertRaises(RuntimeError):
            main_ready(self.root)

    def test_exact_completed_counts_are_required(self):
        self.pipeline(["complete"] * 3)
        self.suites(count2=239)
        with self.assertRaises(RuntimeError):
            main_ready(self.root)
        self.suites()
        self.assertTrue(main_ready(self.root))

    def test_a_failed_worker_blocks_followups(self):
        self.pipeline(["complete"] * 3)
        self.suites()
        path = self.root / "pinn4soh_full_paper_stage2/manifest.json"
        value = json.loads(path.read_text())
        value["tasks"]["0"]["status"] = "failed"
        self.write(path.relative_to(self.root), value)
        with self.assertRaises(RuntimeError):
            main_ready(self.root)


if __name__ == "__main__":
    unittest.main()
