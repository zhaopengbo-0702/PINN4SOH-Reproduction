"""Split the audited offline ZIP into ordinary ZIP parts for the platform size cap."""

import hashlib
import json
from pathlib import Path
import zipfile


root = Path(__file__).resolve().parents[1] / "outputs/full_reproduction"
source = root / "legacy-offline-20261110-v1.zip"
destination = root / "legacy_upload_parts"
destination.mkdir(exist_ok=False)
parts, total = [], hashlib.sha256()
with source.open("rb") as stream:
    for index in range(1, 100):
        block = stream.read(96 * 1024 * 1024)
        if not block:
            break
        total.update(block)
        name = "legacy_archive.part%03d" % index
        path = destination / (name + ".zip")
        with zipfile.ZipFile(path, "x", zipfile.ZIP_STORED) as archive:
            archive.writestr(name, block)
        parts.append(dict(archive=path.name, member=name, bytes=len(block),
            sha256=hashlib.sha256(block).hexdigest(), archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert total.hexdigest() == "e422cc82b6b95062809bf72cea8727beb4cb2a360ab09244dcd00f633948e87f"
record = dict(format="pinn4soh-offline-archive-parts-v1", original=source.name,
    bytes=source.stat().st_size, sha256=total.hexdigest(), parts=parts)
(destination / "legacy_archive_parts.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(dict(parts=len(parts), bytes=record["bytes"], sha256=record["sha256"])))
