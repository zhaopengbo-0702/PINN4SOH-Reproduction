"""Read-only acceptance of immutable results and declared replacement tools.

Checks saved evidence and independently recomputes bounded feature diagnostics
and replacement-search validation objectives. It never trains, opens a cloud
session, changes frozen results, or declares historical protocols recovered.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
FULL = Path("outputs/full_reproduction")
FROZEN = {
    "final_results_2120/suite_report.json": "ebdc9f8a7ddeef608ee93a4b4aa6215b8ad06ac4b26e75d744233419f76d9738",
    "final_results_2120/independent_review_complete/claim_decisions_21.csv": "01e16e72ddcf8c28f55d7c57773cc8c42a412e8cb92c26744de2fae05ee63428",
    "final_results_2120/independent_review_complete/figure_table_evidence_map_25.csv": "106ada4af838f80b533c2e7a99bc2cb6874142786110ae6276cede6f550dcf96",
}


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as reader:
        return list(csv.DictReader(reader))


def require(value, message):
    if not value:
        raise ValueError(message)


def close(left, right):
    return math.isclose(left, right, rel_tol=2e-12, abs_tol=1e-15)


def feature_check(directory, source_root):
    manifest = read(directory / "manifest.json")
    require(manifest["status"] == "complete" and manifest["feature_count"] == 16 and
            manifest["replacement_feature_count"] == 2, "Incomplete 16-feature export")
    require(manifest["all_released_slopes_reconstructed"] is False and
            manifest["training_inputs_changed"] is False and
            manifest["preset_protocol"].get("replacement_slope_selected_by_released_feature_error",
                                            manifest["preset_protocol"].get("selected_by_released_feature_error")) is False,
            "Replacement boundary or frozen-input statement changed")
    replacements = {row["feature"] for row in manifest["feature_schema"] if row["status"] == "replacement"}
    require(replacements == {"voltage slope", "current slope"}, "Exactly two explicitly declared replacement slopes required")
    checked = []
    for name, checksum in manifest["artifacts_sha256"].items():
        path = directory / name
        require(path.is_file() and sha(path) == checksum, "Feature artifact SHA changed: " + str(path))
        checked.append(str(path))
    for name, checksum in manifest["code_sha256"].items():
        require(sha(source_root / "reproduction" / name) == checksum, "Feature implementation changed: " + name)
    records = csv_rows(directory / "features_16.csv")
    diagnostics = {row["raw_cycle"]: row["curves"] for row in read(directory / "window_diagnostics.json")}
    slope_checks = 0
    for row in records:
        for prefix in ("voltage", "current"):
            report = diagnostics[int(row["raw_cycle"])][prefix]
            if report["status"] == "invalid_window" or not row[prefix + " slope"]:
                continue
            expected = (report["observed_last"] - report["observed_first"]) / (report["time_end_seconds"] - report["time_start_seconds"])
            require(close(float(row[prefix + " slope"]), expected), "Endpoint-per-second slope differs")
            slope_checks += 1
    comparison = None
    if manifest["comparison_available"]:
        saved = read(directory / "released_comparison_summary.json")
        details = csv_rows(directory / "released_comparison.csv")
        recomputed = []
        for item in saved["features"]:
            rows = [row for row in details if row["feature"] == item["feature"]]
            errors = []
            for row in rows:
                if row["exported"] and row["released"]:
                    error = abs(float(row["exported"]) - float(row["released"]))
                    require(close(error, float(row["absolute_error"])), "Feature detail error differs")
                    errors.append(error)
            maximum = max(errors) if errors else None
            require(len(rows) == item["rows"] and len(errors) == item["finite_pairs"] and
                    sum(value <= 1e-8 for value in errors) == item["matches_at_1e_8"] and
                    (maximum is None and item["maximum_absolute_error"] is None or
                     maximum is not None and close(maximum, item["maximum_absolute_error"])), "Feature residual summary differs")
            recomputed.append(dict(feature=item["feature"], status=item["status"], finite_pairs=len(errors),
                                   matches_at_1e_8=sum(value <= 1e-8 for value in errors), maximum_absolute_error=maximum))
        comparison = recomputed
    return dict(directory=str(directory), manifest_sha256=sha(directory / "manifest.json"), dataset=manifest["dataset"],
                selected_cycles=manifest["selected_cycles"], profile=manifest["profile"],
                artifact_sha_checks=len(checked), implementation_sha_checks=len(manifest["code_sha256"]),
                endpoint_slope_recomputations=slope_checks, comparison_recomputed=comparison,
                all_released_slopes_reconstructed=False, passed=True)


def search_check(directory):
    import numpy as np
    plan, state, selection = [read(directory / name) for name in ("plan.json", "status.json", "selection.json")]
    require(state["status"] == "complete" and state["workers"] == 1 and
            state["completed_runs"] == state["total_runs"] == plan["candidate_runs"] and
            selection["test_metrics_used"] is False, "Incomplete or non-validation search")
    expected = {(candidate["candidate_id"], seed) for candidate in plan["candidates"] for seed in plan["specification"]["seeds"]}
    seen, per_candidate, artifact_checks = set(), {}, 0
    for path in sorted(directory.glob("candidate_*/seed_*/result.json")):
        result, worker = read(path), read(path.with_name("worker.json"))
        key = (worker["candidate_id"], worker["seed"])
        require(key not in seen and key in expected and result["status"] == "complete" and result["test_loaded"] is False and
                set(worker["splits"]) == {"train", "valid"}, "Candidate identity or test isolation differs")
        seen.add(key)
        for name, checksum in result["artifact_sha256"].items():
            require(sha(path.parent / name) == checksum, "Search artifact SHA changed: " + name)
            artifact_checks += 1
        truth, pred = [np.load(path.parent / name, allow_pickle=False).astype(np.float64) for name in ("valid_truth.npy", "valid_prediction.npy")]
        require(truth.shape == pred.shape and truth.size > 0 and np.isfinite(truth).all() and np.isfinite(pred).all(), "Validation arrays invalid")
        score = float(np.mean((truth - pred) ** 2))
        require(score == result["validation_mse"], "Validation objective differs from saved arrays")
        history = read(path.parent / "history.json")
        require(min(row["validation_mse"] for row in history) == score and
                min(history, key=lambda row: row["validation_mse"])["epoch"] == result["best_epoch"], "Best epoch differs from validation history")
        per_candidate.setdefault(worker["candidate_id"], []).append(score)
    require(seen == expected, "Candidate/seed set incomplete")
    means = {name: sum(values) / len(values) for name, values in per_candidate.items()}
    chosen = min(means, key=lambda name: (means[name], name))
    require(chosen == selection["selected"]["candidate_id"] and close(means[chosen], selection["selected"]["mean_validation_mse"]), "Validation-only selection differs")
    tests = []
    for path in sorted(directory.glob("selected_test/seed_*/result.json")):
        result, worker = read(path), read(path.with_name("worker.json"))
        require(result["status"] == "complete" and result["test_used_for_selection"] is False and
                worker["candidate_id"] == chosen and set(worker["splits"]) == {"test"} and
                worker["selection_sha256"] == sha(directory / "selection.json"), "Selected-test binding differs")
        for name, checksum in result["artifact_sha256"].items():
            require(sha(path.parent / name) == checksum, "Selected-test artifact changed")
            artifact_checks += 1
        truth, pred = [np.load(path.parent / name, allow_pickle=False).astype(np.float64) for name in ("test_truth.npy", "test_prediction.npy")]
        require(float(np.mean((truth - pred) ** 2)) == result["test_mse"] and
                float(np.mean(np.abs(truth - pred))) == result["test_mae"], "Selected-test metrics differ")
        tests.append(worker["seed"])
    require(not state["test_evaluated"] or sorted(tests) == sorted(plan["specification"]["seeds"]), "Selected-test seed set incomplete")
    return dict(directory=str(directory), candidate_runs=len(seen), selected_candidate=chosen,
                selected_test_workers=len(tests), artifact_sha_checks=artifact_checks,
                validation_array_objectives_recomputed=True, test_used_for_selection=False,
                historical_author_grid_recovered=False, passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BASE)
    parser.add_argument("--feature-validation", type=Path, required=True)
    parser.add_argument("--search-validation", type=Path, required=True)
    parser.add_argument("--extension-comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    require(not output.exists(), "Use a fresh acceptance output directory")
    frozen = []
    for relative, expected in FROZEN.items():
        path = root / FULL / relative
        checksum = sha(path)
        require(checksum == expected, "Frozen evidence changed: " + relative)
        frozen.append(dict(path=str(path), sha256=checksum, unchanged=True))
    feature_acceptance = read(args.feature_validation.resolve() / "VALIDATION.json")
    require(feature_acceptance["status"] == "complete" and
            feature_acceptance["implementation_acceptance"] == "passed_bounded_samples" and
            feature_acceptance["all_released_features_reconstructed"] is False and
            feature_acceptance["analytical_fixture"]["returncode"] == 0 and
            feature_acceptance["training_performed"] is False and
            feature_acceptance["frozen_inputs_changed"] is False,
            "Feature replacement acceptance incomplete or boundary changed")
    require(sha(args.feature_validation.resolve() / "analytical_tests.txt") ==
            feature_acceptance["analytical_fixture"]["log_sha256"], "Analytical feature validation log changed")
    for name, checksum in feature_acceptance["code_sha256"].items():
        require(sha(root / "reproduction" / name) == checksum, "Feature validation code changed: " + name)
    features = []
    for sample in feature_acceptance["real_samples"]:
        path = Path(sample["manifest"])
        require(sha(path) == sample["manifest_sha256"], "Feature validation manifest binding changed")
        features.append(feature_check(path.parent, root))
    require(features, "No bounded real feature exports found")
    search_acceptance = read(args.search_validation.resolve() / "ACCEPTANCE.json")
    require(search_acceptance["status"] == "passed_functional_and_real_data_smoke" and
            search_acceptance["search_test_passed"] is True and
            search_acceptance["real_data_plan"]["executed"] is False,
            "Requires completed functional validation and an accurately labelled unexecuted example grid")
    search_hash_checks = 0
    for relative, checksum in search_acceptance["artifact_sha256"].items():
        require(sha(root / relative) == checksum, "Replacement search acceptance artifact changed: " + relative)
        search_hash_checks += 1
    searches = [search_check(Path(search_acceptance["real_data_smoke"]["path"]))]
    implementation_evidence = {}
    for name, directory, filename in (("feature", args.feature_validation, "VALIDATION.json"), ("search", args.search_validation, "ACCEPTANCE.json")):
        path = directory.resolve() / filename
        require(path.is_file(), "Await completed implementation validation: " + str(path))
        implementation_evidence[name] = dict(path=str(path), sha256=sha(path), document=read(path))
    extension = None
    if args.extension_comparison:
        path = args.extension_comparison.resolve() / "LOCAL_RERUN_IO_VERIFIED.json"
        document = read(path)
        require(document["passed"] and document["task_counts"] == {"matched": 110}, "Extension rerun incomplete")
        extension = dict(path=str(path), sha256=sha(path), task_counts=document["task_counts"], passed=True)
    document = dict(format="pinn4soh-delivery-acceptance-v1", checked_at_utc=datetime.now(timezone.utc).isoformat(),
                    status="passed_bounded_delivery_checks", frozen_artifacts=frozen, feature_exports=features,
                    replacement_searches=searches, search_acceptance_artifact_sha_checks=search_hash_checks,
                    implementation_validation=implementation_evidence,
                    extension=extension, extension_not_provided_is_not_completion=extension is None,
                    all_historical_author_protocols_recovered=False, training_performed=False,
                    limitation="This bounded acceptance rereads saved evidence and arrays. It does not rerun the full 2120, restore unknown historical seeds/grid/checkpoint provenance, establish original MIT MAT units, align satellite schedule phase, or reproduce physical ageing hardware.",
                    verifier_sha256=sha(__file__))
    output.mkdir(parents=True)
    (output / "ACCEPTANCE.json").write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = ("# 交付文件独立验收\n\n"
            + "冻结主报告、21结论与25图表CSV的SHA保持不变。此次仅重新检查保存证据，不重新训练主2120。\n\n"
            + "16特征替代入口：核代码和产物SHA，独立重算每个已测真实样本的端点/秒斜率及发布列残差汇总；两个slope始终标为替代定义。真实导出集合数量：" + str(len(features)) + "。\n\n"
            + "替代搜索：重新读取小型已完成验证的validation数组、history和checkpoint选择分数，核完整候选/seed、仅train/valid worker payload、选定后test绑定和产物SHA。验收集合数量：" + str(len(searches)) + "；完整示例大grid并未因此成为已跑完的正式搜索。\n\n"
            + ("选定110项扩展本地数组重算验收记录已包含。\n\n" if extension else "本次未传扩展本地比较目录，不能据此验收110项扩展完成。\n\n")
            + "验收通过表示这些已落盘工具和有限执行通过检查，不表示作者未知历史协议、原MAT单位、卫星映射或物理装置被恢复。详见ACCEPTANCE.json完整精度及来源。\n")
    (output / "README.md").write_text(text, encoding="utf-8")
    print(json.dumps(dict(status=document["status"], feature_exports=len(features), search_smokes=len(searches), extension_included=extension is not None)))


if __name__ == "__main__":
    main()
