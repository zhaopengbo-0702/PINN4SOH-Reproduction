"""Assemble checksum-pinned official Python 3.7.10 packages into an isolated prefix."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def inside(root, name):
    if (not name or "\\" in name or ":" in name or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts):
        raise ValueError("Unsafe archive path: " + name)
    result = (root / name).resolve()
    if result != root and root not in result.parents:
        raise ValueError("Archive path escapes prefix: " + name)
    return result


def load_prefixer():
    path = Path(__file__).with_name("legacy_vendor") / "conda_pack/prefixes.py"
    if sha256(path) != "88b58df57aa4f6772f8d6aeeb85fe1f91bfd31646ced75560fbe91267a60dc8f":
        raise ValueError("Vendored conda-pack 0.8.1 prefix helper differs from pinned official source")
    spec = importlib.util.spec_from_file_location("pinn_legacy_conda_pack_prefixes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prefix_records(archive):
    if "info/has_prefix" not in archive.getnames():
        return []
    rows = []
    for line in archive.extractfile("info/has_prefix").read().decode("utf-8").splitlines():
        fields = shlex.split(line)
        if len(fields) != 3 or fields[1] not in {"text", "binary"}:
            raise ValueError("Unsupported has_prefix record: " + line)
        rows.append(dict(placeholder=fields[0], mode=fields[1], path=fields[2]))
    return rows


def relocate(prefix, records, prefixer):
    evidence = []
    for row in records:
        path = inside(prefix, row["path"])
        if not path.is_file() or path.is_symlink():
            raise ValueError("Relocation target must be a regular installed file: " + row["path"])
        if row["mode"] == "binary" and len(str(prefix).encode()) > len(row["placeholder"].encode()):
            raise ValueError("Prefix exceeds binary placeholder length")
        before = path.read_bytes()
        occurrences = before.count(row["placeholder"].encode())
        prefixer.update_prefix(str(path), str(prefix), row["placeholder"], row["mode"])
        after = path.read_bytes()
        if row["placeholder"].encode() in after:
            raise ValueError("Unreplaced prefix in " + row["path"])
        if row["mode"] == "binary" and len(before) != len(after):
            raise ValueError("Binary relocation changed file size")
        if after.startswith(b"#!") and len(after.splitlines()[0]) > 127:
            raise ValueError("Relocated shebang too long; use a shorter prefix")
        evidence.append({**row, "occurrences": occurrences, "before_sha256": hashlib.sha256(before).hexdigest(),
                         "after_sha256": hashlib.sha256(after).hexdigest(), "bytes_before": len(before), "bytes_after": len(after)})
    return evidence


def isolated_environment(prefix):
    environment = dict(os.environ)
    for key in ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL"]:
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PATH"] = str(prefix / "bin") + ":/usr/bin:/bin"
    libraries = [prefix / "lib", prefix / "lib/python3.7/site-packages/torch/lib",
                 Path("/usr/local/nvidia/lib"), Path("/usr/local/nvidia/lib64")]
    environment["LD_LIBRARY_PATH"] = ":".join(str(path) for path in libraries if path.is_dir())
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(prefix / "matplotlib-cache")
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    assets, prefix = args.assets.resolve(), args.prefix.resolve()
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("These official assets require Linux x86_64; do not extract/execute them on Windows")
    if prefix == assets or prefix in assets.parents or assets in prefix.parents:
        raise ValueError("Asset storage and isolated prefix must be separate")
    if any(character.isspace() for character in str(prefix)) or len(str(prefix).encode()) > 80:
        raise ValueError("Use a short, whitespace-free Linux prefix (at most 80 bytes)")
    if prefix.exists() and any(prefix.iterdir()):
        raise ValueError("Legacy prefix must be new or empty")
    entries = json.loads((assets / "legacy_asset_manifest.json").read_text(encoding="utf-8-sig"))
    entries.append(json.loads((assets / "legacy_torch_manifest.json").read_text(encoding="utf-8-sig")))
    if len({entry["filename"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate asset filenames")
    for entry in entries:
        if PurePosixPath(entry["filename"]).name != entry["filename"] or entry["kind"] not in {"conda", "wheel"}:
            raise ValueError("Invalid asset name/kind")
        path = inside(assets, entry["filename"])
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError("Invalid asset: " + path.name)
    prefix.mkdir(parents=True, exist_ok=True)
    site = prefix / "lib/python3.7/site-packages"
    site.mkdir(parents=True)
    prefixer, relocation_evidence = load_prefixer(), []
    for entry in entries:
        path = assets / entry["filename"]
        if entry["kind"] == "conda":
            with tarfile.open(path, "r:bz2") as archive:
                relocations = prefix_records(archive)
                for member in archive.getmembers():
                    if member.name.startswith("info/"):
                        continue
                    destination = inside(prefix, member.name)
                    if member.issym():
                        link = (destination.parent / member.linkname).resolve()
                        if link != prefix and prefix not in link.parents:
                            raise ValueError("Unsafe archive symlink: " + member.name)
                    elif member.islnk():
                        inside(prefix, member.linkname)
                    elif not (member.isfile() or member.isdir()):
                        raise ValueError("Unsupported archive member: " + member.name)
                    archive.extract(member, prefix)
                relocation_evidence.extend({"asset": path.name, **row} for row in relocate(prefix, relocations, prefixer))
        else:
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    destination = inside(site, member.filename)
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode) or ".data" in PurePosixPath(member.filename).parts[0]:
                        raise ValueError("Unsupported wheel link/data scheme: " + member.filename)
                    if member.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target, 1024 * 1024)
                    if mode & 0o777:
                        destination.chmod(mode & 0o777)
        print("INSTALLED " + path.name, flush=True)
    env = isolated_environment(prefix)
    command = [str(prefix / "bin/python"), "-I", "-c", "import sys,json,numpy,pandas,sklearn,scipy,torch,matplotlib; "
               "from pathlib import Path; expected=Path(sys.argv[1]).resolve(); "
               "assert sys.version_info[:3]==(3,7,10); assert Path(sys.prefix).resolve()==expected; "
               "modules=[numpy,pandas,sklearn,scipy,torch,matplotlib]; "
               "assert all(expected in Path(m.__file__).resolve().parents for m in modules); "
               "assert [m.__version__ for m in modules]==['1.20.3','1.3.5','0.24.2','1.7.3','1.7.1+cu110','3.3.4']; "
               + ("assert torch.cuda.is_available(); " if args.require_cuda else "") +
               "print(json.dumps({'python':sys.version,'numpy':numpy.__version__,'pandas':pandas.__version__,"
               "'sklearn':sklearn.__version__,'scipy':scipy.__version__,'torch':torch.__version__,"
               "'matplotlib':matplotlib.__version__,'prefix':sys.prefix,'module_paths':{m.__name__:m.__file__ for m in modules},"
               "'sys_path':sys.path,'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()}))", str(prefix)]
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    evidence = dict(assets=entries, command=command, returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr, prefix=str(prefix),
                    relocation=relocation_evidence, installer_sha256=sha256(Path(__file__)),
                    relocation_library="conda-pack 0.8.1, official wheel SHA256 4ac43dbb5b5a1ee6f10df27f83fd227bfdd8a99cb2acbe6d05f1e5a0a681d307",
                    isolation="python -I; no inherited PYTHONPATH/PYTHONHOME/Conda/virtualenv; explicit native library path",
                    scipy_version_policy="1.7.3 is a compatible explicit dependency, not a version specified in the paper README")
    (prefix / "installation_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(result.stdout)
    print(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
