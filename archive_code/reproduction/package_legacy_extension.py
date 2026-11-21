"""Package independent selected-extension tools without creating a selection."""

import argparse
import json
from pathlib import Path
import zipfile

from prepare_legacy_extension import digest, require, CONTROL_SHA, STAGE1_SHA, STAGE2_SHA, MODERN_CODE_SHA


def package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists() and output.parent.is_dir(), "Use a fresh extension tool ZIP")
    source, base = root / "reproduction", root / "outputs/full_reproduction"
    files = {"tools/" + name: source / name for name in [
        "prepare_legacy_extension.py", "launch_legacy_extension.py", "compare_legacy_extension.py",
        "collect_legacy_extension_results.py", "package_legacy_extension.py", "test_legacy_extension.py",
        "compare_legacy_control.py", "build_suite_report.py", "install_legacy_runtime.py", "probe_legacy_runtime.py",
        "restore_backup_json.py"]}
    files["docs/LEGACY_EXTENSION_RUN.md"] = source / "LEGACY_EXTENSION_RUN.md"
    for name, checksum, path in [
        ("legacy_control_plan.json", CONTROL_SHA, base / "legacy_control_support/legacy_control_plan.json"),
        ("stage1_plan.json", STAGE1_SHA, base / "plans/stage1_plan.json"),
        ("stage2_reproduced_transfer_plan.json", STAGE2_SHA, base / "plans/stage2_reproduced_transfer_plan.json")]:
        require(digest(path) == checksum, "Frozen input plan changed")
        files["plans/" + name] = path
    for name, checksum in MODERN_CODE_SHA.items():
        require(digest(source / name) == checksum, "Modern runner changed")
    selection = base / "legacy_extension_selection_20261110.json"
    commands = source / "LEGACY_EXTENSION_SELECTED_20261110.md"
    if output.stem.endswith("v2"):
        require(selection.is_file() and commands.is_file(), "Reviewed selection and commands required for v2")
        files["selection/legacy_extension_selection_20261110.json"] = selection
        files["docs/LEGACY_EXTENSION_SELECTED_20261110.md"] = commands
    hashes = {name: {"bytes": path.stat().st_size, "sha256": digest(path)} for name, path in files.items()}
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("TOOLS_SHA256.json", json.dumps(hashes, indent=2))
    with zipfile.ZipFile(output) as archive:
        require(archive.testzip() is None and set(archive.namelist()) == set(files) | {"TOOLS_SHA256.json"}, "ZIP validation failed")
        import hashlib
        for name, record in hashes.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == record["sha256"], "ZIP member SHA mismatch")
    evidence = {"format": "pinn4soh-selected-legacy-extension-tool-package-v2" if output.stem.endswith("v2") else "pinn4soh-selected-legacy-extension-tool-package-v1", "filename": output.name,
        "bytes": output.stat().st_size, "sha256": digest(output), "files": hashes,
        "real_selection_created": output.stem.endswith("v2"), "training_started": False, "frozen40_plan_changed": False,
        "modern_runner_hashes_unchanged": MODERN_CODE_SHA,
        "scope": "Prepare explicitly selected groups after complete40; old repeat1 stays original, new2-10 only; max360 new/400 analysis"}
    evidence_path = output.with_suffix(".manifest.json")
    require(not evidence_path.exists(), "Package evidence exists")
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    result = package(**vars(parser.parse_args()))
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
