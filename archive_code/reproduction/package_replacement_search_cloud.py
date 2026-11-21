"""Build a minimal checksummed cloud bundle without changing accepted search code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Use a fresh ZIP in an existing directory")
    source, repo = root / "reproduction", root / "PINN4SOH"
    names = ["search_replacement_grid.py", "run_transfer.py", "run_reproduction.py", "run_extended.py",
             "collect_replacement_search.py", "run_replacement_search_cloud.py", "replacement_search_example.json",
             "REPLACEMENT_SEARCH_RUN.md", "REPLACEMENT_SEARCH_CLOUD.md", "package_replacement_search_cloud.py",
             "restore_backup_json.py"]
    files = {name: (source / name).read_bytes() for name in names}
    accepted = json.loads((root / "outputs/full_reproduction/replacement_search_validation_20261111/ACCEPTANCE.json").read_text(encoding="utf-8"))
    accepted_hashes = {name.replace("\\", "/"): value for name, value in accepted["artifact_sha256"].items()}
    for name in ("search_replacement_grid.py", "replacement_search_example.json", "REPLACEMENT_SEARCH_RUN.md"):
        if sha(source / name) != accepted_hashes["reproduction/" + name]:
            raise ValueError("Previously accepted search artifact changed: " + name)
    for name, data in files.items():
        if name.endswith(".py"):
            ast.parse(data, filename=name, feature_version=(3, 10))
    grid = json.loads(files["replacement_search_example.json"])
    data = {path: sha(repo / path) for item in grid["splits"].values() for path in item["paths"]}
    expected = dict(format="pinn4soh-replacement-cloud-expected-inputs-v1", author_python_sha256={
        path.relative_to(repo).as_posix(): sha(path) for path in sorted(repo.rglob("*.py"))}, data_sha256=data,
        accepted_search_validation_sha256=sha(root / "outputs/full_reproduction/replacement_search_validation_20261111/ACCEPTANCE.json"),
        target="modern Python3.10 torch1.13.1+cu116; one visible GPU; PYTHONPATH=/workspace/site-packages",
        dependency_minimum=["Python>=3.8", "torch>=1.13 with CUDA and torch.load weights_only", "numpy", "pandas", "scikit-learn", "scipy"],
        scope="Replacement XJTU/2C explicit grid; not recovered author grid; immutable original2120")
    files["EXPECTED_AUTHOR_INPUTS.json"] = json.dumps(expected, indent=2).encode()
    hashes = {name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in files.items()}
    files["TOOLS_SHA256.json"] = json.dumps(dict(format="pinn4soh-replacement-cloud-package-v1", files=hashes), indent=2).encode()
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(files):
            raise ValueError("ZIP structure verification failed")
        for name, expected_data in files.items():
            if archive.read(name) != expected_data:
                raise ValueError("ZIP member bytes differ")
    evidence = dict(format="pinn4soh-replacement-cloud-package-evidence-v1", filename=output.name,
        bytes=output.stat().st_size, sha256=sha(output), members=len(files), files=hashes,
        accepted_search_artifacts_unchanged=True, python310_ast_all_passed=True,
        training_started=False, cloud_accessed=False)
    output.with_suffix(".manifest.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    result = package(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
