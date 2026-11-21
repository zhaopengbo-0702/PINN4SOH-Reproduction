"""Check official local archive metadata for the Linux CPython 3.7.10 runtime."""

import argparse
from collections import Counter
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import tarfile
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import install_legacy_runtime as installer


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def safe_member(name):
    if (not name or "\\" in name or ":" in name or name.startswith("/")
            or ".." in PurePosixPath(name).parts):
        raise ValueError("Unsafe archive member: " + name)


def audit(assets):
    assets = Path(assets).resolve()
    manifests = [assets / "legacy_asset_manifest.json", assets / "legacy_torch_manifest.json"]
    entries = json.loads(manifests[0].read_text(encoding="utf-8-sig"))
    entries.append(json.loads(manifests[1].read_text(encoding="utf-8-sig")))
    if len(entries) != 31 or len({row["filename"] for row in entries}) != 31:
        raise ValueError("Expected the 30 runtime archives plus one official CUDA torch wheel")
    rows, wheels, conda = [], [], []
    for entry in entries:
        if PurePosixPath(entry["filename"]).name != entry["filename"]:
            raise ValueError("Manifest filename must be a basename")
        path = installer.inside(assets, entry["filename"])
        if path.stat().st_size != entry["bytes"] or installer.sha256(path) != entry["sha256"]:
            raise ValueError("Asset hash/size differs: " + path.name)
        row = {**entry}
        if entry["kind"] == "wheel":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    safe_member(name)
                metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
                metadata = BytesParser().parsebytes(archive.read(metadata_name))
                row.update(metadata_name=metadata["Name"], metadata_version=metadata["Version"],
                           requires_python=metadata["Requires-Python"], requires_dist=metadata.get_all("Requires-Dist", []),
                           wheel_data_schemes=sorted({name.split("/")[0] for name in archive.namelist() if ".data/" in name}))
                if canonical(metadata["Name"]) != canonical(entry["package"]) or metadata["Version"] != entry["version"]:
                    raise ValueError("Wheel identity differs from manifest")
                if row["wheel_data_schemes"]:
                    raise ValueError("Unexpected .data scheme requires a full wheel installer")
                if row["requires_python"] and not SpecifierSet(row["requires_python"]).contains("3.7.10"):
                    raise ValueError("Wheel does not support Python 3.7.10")
                wheels.append(row)
        elif entry["kind"] == "conda":
            with tarfile.open(path, "r:bz2") as archive:
                for member in archive.getmembers():
                    safe_member(member.name)
                    if member.issym() or member.islnk():
                        target = (posixpath.join(posixpath.dirname(member.name), member.linkname)
                                  if member.issym() else member.linkname)
                        normalized = posixpath.normpath(target)
                        if member.linkname.startswith("/") or normalized == ".." or normalized.startswith("../"):
                            raise ValueError("Archive link leaves prefix: " + member.name)
                    elif not (member.isfile() or member.isdir()):
                        raise ValueError("Unsupported conda archive node")
                index = json.loads(archive.extractfile("info/index.json").read().decode())
                if index["name"] != entry["package"] or index["version"] != entry["version"]:
                    raise ValueError("Conda identity differs from manifest")
                row.update(depends=index.get("depends", []), build=index["build"],
                           relocation=installer.prefix_records(archive))
                conda.append(row)
        else:
            raise ValueError("Unknown asset kind")
        rows.append(row)
    versions = {canonical(row["package"]): row["version"] for row in rows}
    environment = dict(python_version="3.7", python_full_version="3.7.10", implementation_name="cpython",
        implementation_version="3.7.10", platform_python_implementation="CPython", os_name="posix",
        sys_platform="linux", platform_machine="x86_64", platform_system="Linux", platform_release="",
        platform_version="", extra="")
    dependencies = []
    for wheel in wheels:
        for raw in wheel["requires_dist"]:
            requirement = Requirement(raw)
            needed = not requirement.marker or requirement.marker.evaluate(environment)
            installed = versions.get(canonical(requirement.name))
            passed = not needed or installed is not None and requirement.specifier.contains(Version(installed))
            dependencies.append(dict(package=wheel["package"], requirement=raw, required_for_target=needed,
                                     installed_version=installed, passed=passed))
    exceptions = {"_libgcc_mutex": "Conda solver mutex metadata; no runtime payload",
                  "_openmp_mutex": "Conda solver mutex metadata; libgomp payload is present",
                  "ca-certificates": "TLS trust store is external; offline training does not require package TLS",
                  "__glibc": "Host Linux virtual package; >=2.17 must be checked on cloud"}
    conda_dependencies = []
    for package in conda:
        for raw in package["depends"]:
            name = raw.split()[0]
            conda_dependencies.append(dict(package=package["package"], requirement=raw,
                installed_version=versions.get(canonical(name)), external_or_metadata_reason=exceptions.get(name),
                available_or_explained=canonical(name) in versions or name in exceptions))
    passed = all(row["passed"] for row in dependencies) and all(row["available_or_explained"] for row in conda_dependencies)
    return dict(format="pinn4soh-legacy-static-audit-v1", static_runtime_dependency_check_passed=passed,
        executed_original_runtime=False, gpu_smoke_status="not_executed_by_static_audit", assets=rows,
        manifest_sha256={path.name: installer.sha256(path) for path in manifests},
        wheel_dependencies=dependencies, conda_dependencies=conda_dependencies,
        conda_manager_environment_complete=False,
        relocation_record_count=sum(len(row.get("relocation", [])) for row in rows),
        count_by_kind=dict(Counter(row["kind"] for row in rows)),
        explicit_limits=["Runtime execution requires Linux x86_64; local Windows has no installed WSL distribution",
            "SciPy 1.7.3 is compatible and explicitly selected; README does not pin its version",
            "SciencePlots is unpinned and absent; author training entry points do not import it",
            "No claim that dependent OpenSSL/GCC/BLAS/OS/CUDA driver match the publication machine",
            "Conda mutex packages are metadata; system glibc and TLS trust are external requirements"],
        audit_script_sha256=installer.sha256(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["static_runtime_dependency_check_passed", "count_by_kind", "relocation_record_count"]}))
    raise SystemExit(0 if result["static_runtime_dependency_check_passed"] else 1)


if __name__ == "__main__":
    main()
