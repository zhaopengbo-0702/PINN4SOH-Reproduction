"""Independently audit replacement search and create a verified immutable archive."""
import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

from search_replacement_grid import read, write, sha, identity, select


def audit_search(search, require_test=False):
    import numpy as np
    search = Path(search).resolve()
    plan, state, selection = [read(search / name) for name in ("plan.json", "status.json", "selection.json")]
    if state.get("status") != "complete" or state.get("plan_sha256") != identity(plan):
        raise ValueError("Search is incomplete or plan identity differs")
    for path, checksum in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        if sha(path) != checksum:
            raise ValueError("Source/input changed since plan: " + path)
    expected, rows, artifact_checks, tensor_cohorts = set(), [], 0, {}
    seeds = plan["specification"]["seeds"]
    if state["total_runs"] != len(plan["candidates"]) * len(seeds) or state["completed_runs"] != state["total_runs"]:
        raise ValueError("Candidate counts are incomplete")
    for candidate in plan["candidates"]:
        for seed in seeds:
            folder = search / candidate["candidate_id"] / ("seed_%d" % seed)
            expected.add(folder.relative_to(search).as_posix())
            worker, result = read(folder / "worker.json"), read(folder / "result.json")
            if (worker["mode"] != "train" or set(worker["splits"]) != {"train", "valid"}
                    or worker["parameters"] != candidate["parameters"] or worker["seed"] != seed
                    or worker["candidate_id"] != candidate["candidate_id"] or worker["source_sha256"] != plan["source_sha256"]
                    or worker["fixed"] != plan["specification"]["fixed"] or worker["loss_variant"] != plan["specification"]["loss_variant"]
                    or result.get("status") != "complete" or result.get("worker_identity") != identity(worker)
                    or result.get("test_loaded") is not False):
                raise ValueError("Candidate worker identity/isolation differs")
            if worker["splits"] != {role: plan["resolved_splits"][role] for role in ("train", "valid")}:
                raise ValueError("Candidate split differs from plan")
            required = {"best_model.pth", "history.json", "valid_truth.npy", "valid_prediction.npy", "data_identity.json"}
            if set(result["artifact_sha256"]) != required:
                raise ValueError("Candidate required artifacts missing")
            for name, checksum in result["artifact_sha256"].items():
                if sha(folder / name) != checksum:
                    raise ValueError("Candidate artifact changed")
                artifact_checks += 1
            truth = np.load(folder / "valid_truth.npy", allow_pickle=False).astype(np.float64)
            pred = np.load(folder / "valid_prediction.npy", allow_pickle=False).astype(np.float64)
            if not truth.size or truth.shape != pred.shape or not (np.isfinite(truth).all() and np.isfinite(pred).all()):
                raise ValueError("Invalid validation arrays")
            mse = float(np.mean((truth - pred) ** 2))
            history = read(folder / "history.json")
            if [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
                raise ValueError("Epoch sequence invalid")
            best = min(history, key=lambda r: (r["validation_mse"], r["epoch"]))
            if mse != result["validation_mse"] or best["validation_mse"] != mse or best["epoch"] != result["best_epoch"]:
                raise ValueError("Saved arrays disagree with best validation epoch")
            data = read(folder / "data_identity.json")
            if set(data) != {"train", "valid"}:
                raise ValueError("Candidate data contains test evidence")
            cohort = identity(data)
            if seed in tensor_cohorts and tensor_cohorts[seed] != cohort:
                raise ValueError("Candidate normalized tensors differ within seed")
            tensor_cohorts[seed] = cohort
            rows.append(dict(candidate_id=candidate["candidate_id"], seed=seed, validation_mse=mse,
                             result_sha256=sha(folder / "result.json")))
    actual = {path.parent.relative_to(search).as_posix() for path in search.glob("candidate_*/seed_*/result.json")}
    if actual != expected:
        raise ValueError("Unexpected/missing candidate result directory")
    computed = select(rows, plan["candidates"], seeds)
    computed.update(plan_sha256=identity(plan), candidate_results=rows)
    if computed != selection or state["selected_candidate"] != computed["selected"]["candidate_id"]:
        raise ValueError("Validation selection differs from independent recomputation")
    tests = list(search.glob("selected_test/seed_*/result.json"))
    if require_test and (len(tests) != len(seeds) or not state.get("test_evaluated")):
        raise ValueError("All selected test evaluations required")
    if not state.get("test_evaluated") and tests:
        raise ValueError("Test outputs disagree with state")
    for path in tests:
        worker, result = read(path.with_name("worker.json")), read(path)
        if (worker["mode"] != "test" or set(worker["splits"]) != {"test"} or result.get("test_used_for_selection") is not False
                or worker["candidate_id"] != computed["selected"]["candidate_id"] or worker["seed"] not in seeds
                or worker["parameters"] != computed["selected"]["parameters"] or worker["source_sha256"] != plan["source_sha256"]
                or worker["selection_sha256"] != sha(search / "selection.json") or result.get("status") != "complete"
                or worker["checkpoint_sha256"] != sha(worker["checkpoint"]) or result["worker_identity"] != identity(worker)):
            raise ValueError("Selected test identity differs")
        if worker["splits"] != {"test": plan["resolved_splits"]["test"]}:
            raise ValueError("Test split differs")
        if set(result["artifact_sha256"]) != {"test_truth.npy", "test_prediction.npy", "data_identity.json"}:
            raise ValueError("Test artifacts missing")
        for name, checksum in result["artifact_sha256"].items():
            if sha(path.parent / name) != checksum:
                raise ValueError("Test artifact changed")
            artifact_checks += 1
        truth = np.load(path.with_name("test_truth.npy"), allow_pickle=False).astype(np.float64)
        pred = np.load(path.with_name("test_prediction.npy"), allow_pickle=False).astype(np.float64)
        if truth.shape != pred.shape or not truth.size or not (np.isfinite(truth).all() and np.isfinite(pred).all()):
            raise ValueError("Invalid test arrays")
        if float(np.mean((truth - pred) ** 2)) != result["test_mse"] or float(np.mean(np.abs(truth - pred))) != result["test_mae"]:
            raise ValueError("Test metrics differ from saved arrays")
    return dict(format="pinn4soh-replacement-search-independent-audit-v1", status="passed", candidate_runs=len(rows),
                candidates=len(plan["candidates"]), seeds=seeds, test_evaluations=len(tests), artifact_sha256_checks=artifact_checks,
                validation_selection_independently_recomputed=True, test_metrics_independently_recomputed=True,
                tensor_cohort_sha256=tensor_cohorts, selected=computed["selected"], plan_sha256=identity(plan),
                selection_sha256=sha(search / "selection.json"), replacement_not_author_historical_search=True)


def collect(search, bundle, smoke, smoke_spec, pipeline_status, output, require_test=False, chunk_mib=48):
    search, bundle, smoke, smoke_spec, pipeline_status, output = [Path(p).resolve() for p in
        (search, bundle, smoke, smoke_spec, pipeline_status, output)]
    if output.exists() or not re.fullmatch(r"[A-Za-z0-9_-]+", output.name) or not 1 <= chunk_mib <= 64:
        raise ValueError("Use fresh simple output name and chunk size 1..64MiB")
    audit = audit_search(search, require_test)
    smoke_audit = audit_search(smoke, True)
    paths = {}

    def add(root, label):
        for path in sorted(root.rglob("*")) if root.is_dir() else [root]:
            if path.is_symlink():
                raise ValueError("Symlinks forbidden in evidence")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                name = label + "/" + (path.relative_to(root).as_posix() if root.is_dir() else path.name)
                paths[name] = path

    add(search, "search")
    add(bundle, "tools")
    add(smoke, "gpu_smoke")
    add(smoke_spec, "pipeline")
    # Live pipeline status changes after collector returns, so freeze its current
    # bytes as explicitly observed state instead of pretending it was final.
    observed_state = read(pipeline_status)
    output.mkdir(parents=True)
    write(output / "PIPELINE_STATUS_AT_COLLECTION.json", observed_state)
    write(output / "INDEPENDENT_SEARCH_AUDIT.json", dict(formal=audit, smoke=smoke_audit))
    add(output / "PIPELINE_STATUS_AT_COLLECTION.json", "audit")
    add(output / "INDEPENDENT_SEARCH_AUDIT.json", "audit")
    frozen = {name: dict(bytes=path.stat().st_size, sha256=sha(path)) for name, path in paths.items()}
    archive = output / ("PINN4SOH_LEGACY_SEARCH_" + output.name + ".tar.gz")
    with archive.open("wb") as compressed:
        with gzip.GzipFile(fileobj=compressed, mode="wb", compresslevel=1, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w|") as tar:
                for name, path in sorted(paths.items()):
                    before = path.stat()
                    h = hashlib.sha256()
                    with path.open("rb") as stream:
                        class Reader:
                            def read(self, size):
                                block = stream.read(size)
                                h.update(block)
                                return block
                        tar.addfile(tar.gettarinfo(str(path), arcname=name), Reader())
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or h.hexdigest() != frozen[name]["sha256"]:
                        raise ValueError("Evidence changed while streaming archive")
    seen = set()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile() or member.name not in frozen or member.name in seen:
                raise ValueError("Archive member set invalid")
            h, size = hashlib.sha256(), 0
            stream = tar.extractfile(member)
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
                size += len(block)
            if size != frozen[member.name]["bytes"] or h.hexdigest() != frozen[member.name]["sha256"]:
                raise ValueError("Archive member hash failed")
            seen.add(member.name)
    if seen != set(frozen):
        raise ValueError("Missing archived evidence")
    size, checksum = archive.stat().st_size, sha(archive)
    chunk_size = chunk_mib * 1024 * 1024
    count = max(1, (size + chunk_size - 1) // chunk_size)
    records = []
    with archive.open("rb") as stream:
        for i in range(count):
            data = stream.read(chunk_size)
            metadata = dict(format="pinn4soh-backup-json-base64-v1", archive_type="tar.gz", original_zip_name=archive.name,
                original_zip_bytes=size, original_zip_sha256=checksum, chunk_index=i, chunk_count=count,
                chunk_offset=i * chunk_size, chunk_bytes=len(data), chunk_sha256=hashlib.sha256(data).hexdigest(), encoding="base64")
            path = output / ("results.search." + output.name + ".chunk%03d.json" % (i + 1))
            path.write_text(json.dumps(dict(metadata, data_base64=base64.b64encode(data).decode("ascii")), separators=(",", ":")) + "\n", encoding="ascii")
            if base64.b64decode(read(path)["data_base64"], validate=True) != data:
                raise ValueError("Transfer wrapper roundtrip failed")
            records.append(dict(metadata, wrapper_name=path.name, wrapper_bytes=path.stat().st_size,
                                wrapper_sha256=sha(path), roundtrip_verified=True))
    evidence = dict(format="pinn4soh-replacement-search-collection-v1", status="complete", created_at=datetime.now(timezone.utc).isoformat(),
        archive_bytes=size, archive_sha256=checksum, all_archived_member_hashes_verified=True, formal_audit=audit,
        smoke_audit=smoke_audit, inventory=[dict(path=name, **record) for name, record in sorted(frozen.items())],
        original_author_source_and_data_copied=False, restore_tool_compatible_archive_name=True,
        name_note="LEGACY_SEARCH prefix retained solely for existing restore-name grammar; training runtime is recorded in each result and is modern")
    write(output / "SEARCH_COLLECTION_VERIFIED.json", evidence)
    write(output / "JSON_TRANSFER_MANIFEST.json", dict(format="pinn4soh-backup-json-base64-v1", archive_type="tar.gz",
        verified_without_errors=True, original_archives=1, wrapper_files=len(records), records=records))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("search", "bundle", "smoke", "smoke-spec", "pipeline-status", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--require-test", action="store_true")
    parser.add_argument("--chunk-mib", type=int, default=48)
    result = collect(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k != "inventory"}, indent=2))
