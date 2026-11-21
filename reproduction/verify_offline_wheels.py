import json
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

from pip._vendor.packaging.markers import default_environment
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import canonicalize_name


root = Path(__file__).resolve().parent
environment = default_environment()
environment.update(
    implementation_name="cpython",
    implementation_version="3.10.13",
    os_name="posix",
    platform_machine="x86_64",
    platform_python_implementation="CPython",
    platform_release="",
    platform_system="Linux",
    platform_version="",
    python_full_version="3.10.13",
    python_version="3.10",
    sys_platform="linux",
    extra="",
)
metadata = {}
for wheel in sorted((root / "wheels").glob("*.whl")):
    with ZipFile(wheel) as archive:
        entry = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        info = Parser().parsestr(archive.read(entry).decode("utf-8"))
    if "3.10.13" not in SpecifierSet(info.get("Requires-Python", "")):
        raise RuntimeError("Incompatible Requires-Python: " + wheel.name)
    metadata[canonicalize_name(info["Name"])] = info

checked = []
for name, info in metadata.items():
    for declaration in info.get_all("Requires-Dist", []):
        requirement = Requirement(declaration)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        dependency = canonicalize_name(requirement.name)
        if dependency not in metadata:
            raise RuntimeError(f"Missing dependency of {name}: {requirement}")
        if metadata[dependency]["Version"] not in requirement.specifier:
            raise RuntimeError(f"Unsatisfied dependency of {name}: {requirement}")
        checked.append({"package": name, "requirement": str(requirement)})

report = {
    "environment": environment,
    "wheel_count": len(metadata),
    "active_dependency_count": len(checked),
    "active_dependencies": checked,
    "status": "passed",
}
(root / "offline-wheels-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "active_dependencies"}, indent=2))
