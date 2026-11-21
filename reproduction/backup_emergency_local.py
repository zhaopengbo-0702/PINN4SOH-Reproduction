"""Preserve selected local code, audited text results, manifests and final figures."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
LIMIT = 32 * 1024 * 1024
SUFFIXES = {".py", ".ps1", ".md", ".json", ".csv", ".png", ".pdf", ".mmd", ".txt", ".patch"}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024 * 1024),b""):
            result.update(block)
    return result.hexdigest()


def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = ROOT / "outputs/emergency_local_backup" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    selected, excluded, missing = set(), [], []

    def add(path):
        if not path.is_file():
            return
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts or path.suffix.lower() not in SUFFIXES:
            return
        if path.stat().st_size > LIMIT:
            excluded.append({"path":relative.as_posix(),"bytes":path.stat().st_size,"reason":"per-file bounded backup"})
            return
        selected.add(path)

    for path in (ROOT / "reproduction").rglob("*"):
        if "wheels" not in path.parts:
            add(path)
    for path in (ROOT / "outputs/full_reproduction").rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT / "outputs/full_reproduction")
        if any(part.startswith("plot_layout_fixture") for part in relative.parts):
            continue
        if relative.parts[0] in {"legacy_runtime", "legacy_upload_parts"}:
            if path.name.endswith(".json"):
                add(path)
            continue
        if "feature_audit" in relative.parts and path.suffix.lower() not in {".json", ".md"}:
            if "raw_curve_figures" not in relative.parts and path.name != "raw_source_audit_summary.csv":
                continue
        add(path)
    for name in ["final_results", "validation_overlap_43_20261110", "efficiency_smoke_local_20261110"]:
        for path in (ROOT / "outputs" / name).rglob("*"):
            if name == "validation_overlap_43_20261110" and path.suffix.lower() not in {".json", ".md"}:
                continue
            add(path)
    for path in (ROOT / "PINN4SOH").rglob("*"):
        if path.suffix == ".py" or path.name in {"README.md", "fix.md"}:
            add(path)
    report_path = ROOT / "outputs/full_reproduction/final_results_2120/suite_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("format") != "pinn4soh-independent-suite-report-v1":
        raise ValueError("Unexpected final report format")
    report_check = {key:report.get(key) for key in ["format", "generated_at", "metadata_only", "task_count", "task_status_counts", "repeat_summary_count"]}
    report_check.update(actual_tasks=len(report["tasks"]), actual_summaries=len(report["summaries"]),
        report_sha256=digest(report_path), all_tasks_included=all(row.get("included") is True for row in report["tasks"]),
        report_bytes=report_path.stat().st_size)
    if report_check["actual_tasks"] != 2120 or report["task_status_counts"] != {"included":2120}:
        raise ValueError("Final local report does not confirm all 2120 tasks")
    rows=[]
    for path in sorted(selected):
        rows.append({"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":digest(path)})
    index = dict(format="pinn4soh-emergency-local-evidence-index-v1", created_at=datetime.now(timezone.utc).isoformat(),
        final_report_check=report_check, file_count=len(rows), total_uncompressed_bytes=sum(row["bytes"] for row in rows),
        files=rows, excluded_large_files=excluded,
        scope="Critical local code, prepared plans/manifests, available report/evidence JSON and final figures; no raw H5/MAT/ZIP or runtime payloads",
        cloud_checkpoint_or_full_array_backup=False,
        note="This local backup preserves currently downloaded evidence, not untransferred cloud checkpoints/arrays")
    index_path=destination/"emergency_evidence_index.json"
    index_path.write_text(json.dumps(index,indent=2,ensure_ascii=False),encoding="utf-8")
    archive=destination/"PINN4SOH_critical_local_evidence.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED,compresslevel=3) as stream:
        stream.write(index_path,"emergency_evidence_index.json")
        for row in rows:
            stream.write(ROOT/row["path"],row["path"])
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError("Emergency backup ZIP CRC failed")
        for row in rows:
            if hashlib.sha256(stream.read(row["path"])).hexdigest()!=row["sha256"]:
                raise ValueError("Emergency backup member SHA mismatch: "+row["path"])
    summary=dict(status="verified", backup_directory=str(destination), archive=str(archive),
        archive_bytes=archive.stat().st_size, archive_sha256=digest(archive), index=str(index_path),
        index_sha256=digest(index_path), file_count=len(rows), final_report=report_check,
        zip_crc_passed=True, all_member_sha256_passed=True)
    (destination/"BACKUP_VERIFIED.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
