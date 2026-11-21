"""Write a compact evidence snapshot without treating pending raw work as complete."""

import json
from pathlib import Path
import time


root = Path("outputs/full_reproduction/feature_audit").resolve()
raw_root = root.parent / "raw_data"
tasks = []
for path in sorted((root / "raw_extraction").rglob("manifest.json")):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tasks.append(manifest)
datasets = {}
for dataset, expected in [("XJTU", 55), ("TJU", 130), ("HUST", 77)]:
    selected = [item for item in tasks if item["dataset"] == dataset]
    datasets[dataset] = dict(expected_batteries=expected,
                             complete=sum(item["status"] == "complete" for item in selected),
                             failed=sum(item["status"] == "failed" for item in selected),
                             running=sum(item["status"] == "running" for item in selected),
                             completed_raw_cycles=sum(item.get("raw_cycles", 0) for item in selected if item["status"] == "complete"),
                             measured_completed_seconds=sum(item.get("seconds", 0) for item in selected if item["status"] == "complete"))
archives = []
for path in sorted(raw_root.rglob("*.status.json")):
    archives.append(json.loads(path.read_text(encoding="utf-8")))
boundaries = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "boundary_extraction").rglob("manifest.json"))]
actual_bytes = sum(path.stat().st_size for path in raw_root.rglob("*") if path.is_file())
feature_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
processed_path = root / "processed_figures" / "processed_feature_audit.json"
processed = json.loads(processed_path.read_text(encoding="utf-8")) if processed_path.exists() else None
snapshot = dict(updated_at_unix=time.time(), raw_datasets=datasets,
                total_raw_extraction_complete=sum(item["complete"] for item in datasets.values()),
                total_raw_extraction_expected=262, official_archives=archives,
                all_262_raw_extractions_complete=all(item["complete"] == item["expected_batteries"] and not item["failed"] for item in datasets.values()),
                supplemental_boundary_expected=132,
                supplemental_boundary_complete=sum(item["status"] == "complete" for item in boundaries),
                supplemental_boundary_failed=sum(item["status"] == "failed" for item in boundaries),
                local_raw_download_storage_bytes=actual_bytes, local_feature_audit_storage_bytes=feature_bytes,
                combined_raw_and_feature_storage_bytes=actual_bytes + feature_bytes,
                processed_csv_audit=processed,
                limitations=["Raw feature candidates are not silently substituted for released training CSVs",
                             "All slope candidates and entropy definitions remain separately labelled",
                             "MIT derived mirror is audited separately; it is not original-MAT byte identity"])
(root / "FEATURE_AUDIT_PROGRESS.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in snapshot.items() if key not in {"official_archives", "processed_csv_audit"}}, indent=2))
