"""Build small single-GPU deployment tools and optional exact source/reference ZIP.

Neither package launches training. Existing original offline archives are never
rewritten. Packaging hashes the recovered, frozen source inventory first.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    reproduction, output = root / "reproduction", root / "outputs/full_reproduction"
    recovered = output / "cloud_emergency_recovered/workspace"
    tools_zip = output / "legacy-single-gpu-tools-20261110-v3.zip"
    source_zip = output / "legacy-source-reference-20261110-v1.zip"
    manifest_path = output / "legacy_single_gpu_package_manifest_v3.json"
    for path in [tools_zip, manifest_path]:
        if path.exists():
            raise ValueError("Refusing to replace deployment artifact: " + str(path))
    tool_files = {"tools/" + name: reproduction / name for name in [
        "reassemble_legacy_upload.py", "restore_legacy_source_reference.py", "install_legacy_runtime.py", "probe_legacy_runtime.py",
        "launch_legacy_smoke.py", "launch_legacy_control.py", "compare_legacy_control.py",
        "build_suite_report.py", "test_reassemble_legacy_upload.py", "test_legacy_runtime_tools.py",
        "test_compare_legacy_control.py"]}
    tool_files.update({"docs/" + name: reproduction / name for name in
                       ["LEGACY_SINGLE_GPU_RUN.md", "LEGACY_RUNTIME_RUN.md"]})
    for name in ["legacy_control_plan.json", "legacy_control_design.json"]:
        tool_files["control/" + name] = output / "legacy_control_support" / name
    tool_hashes = {name: {"bytes": path.stat().st_size, "sha256": sha(path)} for name, path in tool_files.items()}
    with zipfile.ZipFile(tools_zip, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in tool_files.items():
            archive.write(path, name)
        archive.writestr("TOOLS_SHA256.json", json.dumps(tool_hashes, indent=2))
    inventory = json.loads((recovered / "pinn4soh_full_paper_stage1/source_inventory.json").read_text(encoding="utf-8"))
    source_files = {}
    for name, checksum in inventory["sha256"].items():
        path = recovered / "PINN4SOH-main" / name
        if sha(path) != checksum:
            raise ValueError("Recovered scientific input changed: " + name)
        source_files["PINN4SOH-main/" + name] = path
    # The smoke's two fixed XJTU-to-HUST paths use this one supplied checkpoint.
    source_files["PINN4SOH-main/pretrained model/model_XJTU_0.pth"] = recovered / "PINN4SOH-main/pretrained model/model_XJTU_0.pth"
    # Every matching completed formal task must agree on its observed source.
    state = json.loads((recovered / "pinn4soh_full_paper_stage1/manifest.json").read_text(encoding="utf-8"))
    input_hashes = {}
    for record in state["tasks"].values():
        for path, checksum in record.get("input_sha256", {}).items():
            input_hashes.setdefault(path, set()).add(checksum)
    checkpoint = source_files["PINN4SOH-main/pretrained model/model_XJTU_0.pth"]
    if input_hashes.get("/workspace/PINN4SOH-main/pretrained model/model_XJTU_0.pth") != {sha(checkpoint)}:
        raise ValueError("Author checkpoint identity not uniquely corroborated")
    refs = ["XJTU/2C/Experiment01", "TJU/2/Experiment01", "MIT/all/Experiment01", "HUST/all/Experiment01"]
    for folder in refs:
        for name in ["config.json", "splits.json"]:
            relative = "pinn4soh_main_20261109/" + folder + "/" + name
            source_files[relative] = recovered / relative
            if input_hashes.get("/workspace/" + relative) != {sha(source_files[relative])}:
                raise ValueError("Reference identity not uniquely corroborated: " + relative)
    source_hashes = {name: {"bytes": path.stat().st_size, "sha256": sha(path)} for name, path in source_files.items()}
    restore = {"format": "pinn4soh-legacy-source-reference-support-v1", "files": source_hashes,
        "source_fingerprint": inventory["fingerprint"], "inventory": inventory,
        "scope": "Exact 417 scientific fingerprint files; one supplied smoke checkpoint; four repeat1 config/splits",
        "destination": "/workspace", "no_overwrite": True,
        "training_launched": False, "modern_model_or_array_results_included": False}
    if not source_zip.exists():
        with zipfile.ZipFile(source_zip, "x", zipfile.ZIP_DEFLATED) as archive:
            for name, path in source_files.items():
                archive.write(path, name)
            archive.writestr("LEGACY_SOURCE_REFERENCE_SHA256.json", json.dumps(restore, indent=2))
    elif sha(source_zip) != "28ded058f73c3c74acda043a94d555e3c9cda34ddee9e84d22588e30caf2f504":
        raise ValueError("Existing immutable source/reference package changed")
    for path in [tools_zip, source_zip]:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Package ZIP CRC failed: " + path.name)
    manifest = {"format": "pinn4soh-single-gpu-deployment-package-v3",
        "runtime_archive_unchanged": {"path": "legacy-offline-20261110-v1.zip", "bytes": 1338319780,
            "sha256": "e422cc82b6b95062809bf72cea8727beb4cb2a360ab09244dcd00f633948e87f"},
        "parts_index_sha256": sha(output / "legacy_upload_parts/legacy_archive_parts.json"),
        "deployment": {"gpus": ["0"], "workers_per_gpu": 1, "threads": 2, "concurrent_platform_jobs": 1},
        "tools": {"filename": tools_zip.name, "bytes": tools_zip.stat().st_size, "sha256": sha(tools_zip), "files": tool_hashes},
        "source_reference_optional": {"filename": source_zip.name, "bytes": source_zip.stat().st_size,
            "sha256": sha(source_zip), "files": source_hashes}, "training_launched": False,
        "packager_sha256": sha(__file__)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"tools": {k: v for k, v in manifest["tools"].items() if k != "files"},
                      "source_reference_optional": {k: v for k, v in manifest["source_reference_optional"].items() if k != "files"}}, indent=2))


if __name__ == "__main__":
    main()
