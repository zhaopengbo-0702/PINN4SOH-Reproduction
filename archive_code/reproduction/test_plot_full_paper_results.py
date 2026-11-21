"""Isolated semantic tests and explicitly synthetic layout fixtures; no training."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import plot_full_paper_results as plot


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "outputs/full_reproduction/paper_protocol.json"
REFERENCES = PROTOCOL.parent / "source_data"


def options(report, evidence, output):
    return argparse.Namespace(report=report, evidence=evidence, output=output, protocol=PROTOCOL,
        references=REFERENCES, efficiency=None, lanes=list(plot.LANES), protocol_id="author_current",
        validation_protocol="author_sample_split", transfer_design="paper_fixed12")


def fixture(directory):
    directory = Path(directory)
    report, evidence = directory / "report", directory / "evidence"
    report.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    summaries, repeats, batteries, curves, truths = [], [], [], [], []
    paper_batteries = plot.read_csv(REFERENCES / "paper_per_battery_tables.csv")

    def battery_path(published):
        group, battery = published["group"], str(published["battery"])
        if group.startswith("XJTU/"):
            return "/fixture/" + group + "/2C_" + battery + ".csv"
        if group == "MIT":
            date = "-".join(part.zfill(2) for part in str(published["date"]).split("/"))
            return "/fixture/MIT/" + date + "/" + battery + ".csv"
        return "/fixture/" + group + "/" + battery + ".csv"

    def add(family, group, model, loss, size=None, evaluation="final", **extra):
        key = "|".join(str(value) for value in [family, group, model, loss, size, evaluation, *extra.values()])
        count = 9 if (family, group, model, size) == ("regular", "HUST", "CNN", None) else 10
        mean = .008 + (plot.GROUPS.index(group) if group in plot.GROUPS else 0) * .0004
        mean += {"PINN": 0, "MLP": .009, "CNN": .005}[model] + (.004 if size else 0)
        mean += .002 if loss == plot.LANES["paperloss"] else 0
        identity = {"comparison_group": key, "family": family, "paper_group": group,
            "dataset": group.split("/")[0], "model": model, "loss_variant": loss, "sample_size": size,
            "evaluation": evaluation, "aggregation": "battery_macro", "convention": "standard",
            "protocol_id": "author_current", "validation_protocol": "author_sample_split",
            "test_split_sha256": hashlib.sha256(group.encode()).hexdigest(), **extra}
        summary = {**identity, "n_repeats": count, "expected_repeats": 10,
            "complete_repeat_set": count == 10, "repeat_ids": list(range(1, count + 1))}
        for metric, scale in [("MAE", .8), ("MAPE", 1.1), ("RMSE", 1), ("MSE", .02)]:
            summary.update({metric + "_mean": mean * scale, metric + "_sd_ddof1": .0006 * scale,
                            metric + "_n": count})
        summaries.append(summary)
        for repeat in range(1, count + 1):
            repeats.append({**identity, "task_id": key + "|r" + str(repeat), "repeat": repeat,
                            "seed": 4000 + repeat})
        if family == "regular":
            for published in [row for row in paper_batteries if row["group"] == group]:
                path = battery_path(published)
                batteries.append({**summary, "battery": path})
                truth_id = hashlib.sha256(path.encode()).hexdigest()
                truth = {"truth_id": truth_id, "battery": path, "n_full_retained_points": 300,
                    "relative_retained_point_indices": list(range(0, 300, 5)) + [299],
                    "measured_SOH": [.99 - index * .0006 for index in list(range(0, 300, 5)) + [299]]}
                if not any(row["truth_id"] == truth_id for row in truths):
                    truths.append(truth)
                curves.append({"task_id": key + "|r1", "family": family, "evaluation": evaluation,
                    "truth_id": truth_id, "predicted_SOH": [value + mean * .35 for value in truth["measured_SOH"]],
                    "selection": "prespecified repeat1; no test-error selection"})
        return summary

    arms = [("PINN", loss) for loss in plot.LANES.values()] + [(model, "author_current") for model in ["MLP", "CNN"]]
    for group in plot.GROUPS:
        for model, loss in arms:
            add("regular", group, model, loss)
    for group in ["XJTU/" + str(index) for index in range(1, 7)] + ["HUST", "TJU/3", "MIT"]:
        for size in range(1, 5):
            for model, loss in arms:
                if (group, size, model) != ("HUST", 4, "CNN"):
                    added = add("small_sample", group, model, loss, size)
                    if (group, size, model) == ("XJTU/6", 4, "CNN"):
                        duplicate = deepcopy(added)
                        duplicate["comparison_group"] += "|different-configuration"
                        summaries.append(duplicate)
    for target in plot.DOMAINS:
        for source in [item for item in plot.DOMAINS if item != target]:
            dataset = source.split("/")[0]
            for mode in plot.SOURCE_MODES:
                for loss in plot.LANES.values():
                    for evaluation in ["final", "source_only"]:
                        add("transfer", target, "PINN", loss, evaluation=evaluation,
                            source=dataset, source_batch={"XJTU": 0, "TJU": 2}.get(dataset, -1),
                            source_mode=mode, transfer_design="paper_fixed12")
    tables = {"repeat_summaries": summaries, "per_repeat": repeats,
        "per_battery_summaries": batteries, "published_comparisons": [], "comparison_effects": [], "task_audit": []}
    plot.write_json(report / "suite_report.json", {"synthetic_layout_fixture": True})
    parts = []
    for name, rows in tables.items():
        path = report / (name + ".csv")
        plot.write_csv(path, rows)
        destination = evidence / path.name
        shutil.copyfile(path, destination)
        parts.append({"path": destination.name, "kind": "csv_" + name, "sha256": plot.digest(destination)})
    for name, kind, values in [("curves.json", "curve_predictions", curves), ("truth.json", "curve_truth", truths)]:
        path = evidence / name
        plot.write_json(path, values)
        parts.append({"path": name, "kind": kind, "sha256": plot.digest(path)})
    plot.write_json(evidence / "suite_evidence_index.json", {
        "format": "pinn4soh-suite-evidence-export-v1", "synthetic_layout_fixture": True, "files": parts})
    return report, evidence


class PlotSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pinn-plot-tests-")
        cls.directory = Path(cls.temp.name)
        cls.report, cls.evidence = fixture(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def renderer(self):
        return plot.Renderer(options(self.report, self.evidence, self.directory / "render"))

    def test_lanes_and_shared_baselines(self):
        renderer = self.renderer()
        first = renderer.cell("published", "regular", "XJTU/1", "PINN", "RMSE")
        second = renderer.cell("paperloss", "regular", "XJTU/1", "PINN", "RMSE")
        self.assertNotEqual(first["observed"], second["observed"])
        self.assertEqual(renderer.cell("published", "regular", "XJTU/1", "MLP", "RMSE")["observed"],
                         renderer.cell("paperloss", "regular", "XJTU/1", "MLP", "RMSE")["observed"])

    def test_incomplete_remains_pending(self):
        cell = self.renderer().cell("published", "regular", "HUST", "CNN", "RMSE")
        self.assertEqual(cell["status"], "pending")
        self.assertIsNone(cell["observed"])
        self.assertGreater(cell["partial_mean"], 0)

    def test_fewer_than_ten_expected_is_not_complete(self):
        renderer = self.renderer()
        row = next(item for item in renderer.data.summaries if item["family"] == "regular"
                   and item["paper_group"] == "XJTU/1" and item["model"] == "PINN"
                   and item["loss_variant"] == plot.LANES["published"])
        row.update(n_repeats=2, expected_repeats=2, complete_repeat_set=True)
        self.assertIsNone(renderer.cell("published", "regular", "XJTU/1", "PINN", "RMSE")["observed"])

    def test_missing_metric_repeats_are_pending(self):
        renderer = self.renderer()
        row = next(item for item in renderer.data.summaries if item["family"] == "regular"
                   and item["paper_group"] == "XJTU/1" and item["model"] == "PINN"
                   and item["loss_variant"] == plot.LANES["published"])
        row["RMSE_n"] = 9
        cell = renderer.cell("published", "regular", "XJTU/1", "PINN", "RMSE")
        self.assertEqual(cell["status"], "metric_pending")
        self.assertIsNone(cell["observed"])

    def test_ambiguous_is_not_pooled(self):
        renderer = self.renderer()
        cell = renderer.cell("published", "small_sample", "XJTU/6", "CNN", "RMSE", 4)
        self.assertEqual(cell["status"], "ambiguous")
        self.assertIsNone(cell["observed"])
        self.assertEqual(len(renderer.ambiguities), 1)

    def test_test_battery_mismatch_is_pending(self):
        renderer = self.renderer()
        row = next(item for item in renderer.data.summaries if item["family"] == "regular"
                   and item["paper_group"] == "XJTU/1" and item["model"] == "MLP")
        row["test_split_sha256"] = "mismatched"
        cell = renderer.cell("published", "regular", "XJTU/1", "MLP", "RMSE")
        self.assertEqual(cell["status"], "test_split_unmatched")
        self.assertIsNone(cell["observed"])

    def test_source_modes_remain_separate(self):
        renderer = self.renderer()
        selected = [renderer.select("published", "transfer", "HUST", source="XJTU", source_batch=0,
            source_mode=mode, transfer_design="paper_fixed12")[0] for mode in plot.SOURCE_MODES]
        self.assertNotEqual(selected[0]["comparison_group"], selected[1]["comparison_group"])

    def test_only_prespecified_first_repeat_is_selected(self):
        renderer = self.renderer()
        curves, status = renderer.selected_curve("published", "XJTU/1", "PINN")
        self.assertEqual(status, "complete")
        self.assertTrue(curves)
        self.assertTrue(all(row["task_id"].endswith("|r1") for row in curves))

    def test_export_only_matches_csv_input(self):
        data = plot.Inputs(None, self.evidence)
        self.assertEqual(len(data.summaries), len(plot.Inputs(self.report, self.evidence).summaries))

    def test_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as directory:
            copy = Path(directory) / "evidence"
            shutil.copytree(self.evidence, copy)
            (copy / "curves.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash-inconsistent"):
                plot.Inputs(None, copy)

    def test_exact_battery_and_date_identity(self):
        published = {"group": "MIT", "battery": "battery-5", "date": "2017/5/12"}
        self.assertTrue(plot.battery_matches("/MIT/2017-05-12/battery-5.csv", published))
        self.assertFalse(plot.battery_matches("/MIT/2017-06-30/battery-5.csv", published))
        self.assertFalse(plot.battery_matches("/MIT/2017-05-12/battery-15.csv", published))

    def test_wrong_convention_and_pooled_rows_are_excluded(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as directory:
            report = Path(directory)
            original = plot.read_csv(self.report / "repeat_summaries.csv")
            sensitivity = [{**original[0], "aggregation": "pooled", "RMSE_mean": 999},
                           {**original[0], "convention": "reversed", "RMSE_mean": 999}]
            plot.write_csv(report / "repeat_summaries.csv", original + sensitivity)
            data = plot.Inputs(report, None)
            self.assertEqual(len(data.summaries), len(original))
            self.assertFalse(any(row["RMSE_mean"] == 999 for row in data.summaries))

    def test_summary_json_fallback_preserves_audited_values(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as directory:
            report = Path(directory)
            summaries = plot.read_csv(self.report / "repeat_summaries.csv")
            plot.write_json(report / "suite_report.json", {"summaries": summaries, "metadata_only": False})
            data = plot.Inputs(report, None)
            self.assertEqual(data.summaries, summaries)

    def test_smoke_efficiency_is_never_formal(self):
        renderer = self.renderer()
        renderer.options.efficiency = ROOT / "outputs/efficiency_smoke_local_20261110/summary.json"
        plot.apply_figure_style()
        renderer.efficiency()
        rows = plot.read_csv(renderer.output / "tables/Table_S8.csv")
        self.assertTrue(all(row["observed_seconds_per_1000"] is None for row in rows))
        self.assertTrue(all(row["status"] == "pending" for row in rows))
        self.assertTrue(renderer.qa[-1]["geometry_passed"])

    def test_entirely_pending_small_panels_have_clear_geometry(self):
        renderer = self.renderer()
        renderer.data.summaries = []
        plot.apply_figure_style()
        renderer.small("published", ["XJTU/1", "HUST"], "all_pending_test", "Pending fixture")
        self.assertTrue(renderer.qa[-1]["geometry_passed"], renderer.qa[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--render", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.fixture:
        report, evidence = fixture(args.fixture)
        if args.render:
            plot.apply_figure_style()
            result = plot.Renderer(options(report, evidence, args.fixture / "rendered")).render()
            print(json.dumps({key: result[key] for key in ["synthetic_layout_fixture", "metric_cell_counts",
                "all_text_geometry_passed"]}))
    else:
        unittest.main(argv=[__file__] + remaining)
