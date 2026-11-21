"""Read-only binding of Stage A CSV bytes, physical identity and explicit specs."""
import argparse
from pathlib import Path

from search_replacement_grid import read, sha


def require(value, text):
    if not value:
        raise ValueError(text)


def verify(registration, repo):
    registration, repo = Path(registration).resolve(), Path(repo).resolve()
    document = read(registration / "REGISTRATION.json")
    require(document["status"] == "frozen_before_new_candidate_training" and document["total_selection_runs"] == 162,
            "Registration not frozen StageA162")
    for name, checksum in document["artifacts_sha256"].items():
        require(sha(registration / name.replace("\\", "/")) == checksum, "Registered artifact SHA changed: " + name)
    groups, csv_checks, identities = [], 0, {}
    for dataset in ("MIT", "HUST"):
        record = document["datasets"][dataset]
        split = read(registration / record["split_path"])
        identity = read(registration / record["identity_path"])
        require(sha(registration / record["split_path"]) == record["split_sha256"] and
                sha(registration / record["identity_path"]) == record["identity_sha256"], "Dataset registration binding differs")
        seen_paths, seen_uids = set(), {}
        for role, data in split.items():
            require(role in {"train", "valid", "test"} and len(data["paths"]) == len(data["physical_ids"]), "Bad split role/count")
            for path, uid in zip(data["paths"], data["physical_ids"]):
                require(path not in seen_paths and (uid not in seen_uids or seen_uids[uid] == role), "Cross-role physical/path overlap")
                source = identity["identities"][path]
                require(source["physical_ids"] == [uid] and sha(repo / path) == source["csv_sha256"], "CSV/UID manifest differs")
                seen_paths.add(path)
                seen_uids[uid] = role
                csv_checks += 1
        excluded = set(record["evidence"].get("excluded_files", []))
        require(seen_paths == set(identity["identities"]) - excluded and not (seen_paths & excluded), "Identity set differs from registered exclusions")
        identities[dataset] = dict(csv_count=len(seen_paths), physical_ids=len(seen_uids))
        for model in ("pinn", "mlp", "cnn"):
            name = dataset.lower() + "_" + model + "_stage_a9.json"
            spec = read(registration / name)
            require(spec["dataset"] == dataset and spec["model"] == model.upper() and spec["splits"] == split and
                    spec["seeds"] == document["selection_seeds"] and len(spec["candidates"]) == 9 and spec["smoke"] is False,
                    "Group spec not bound to registered split/seeds")
            groups.append(dict(dataset=dataset, model=model.upper(), spec=name, spec_sha256=sha(registration / name), candidate_runs=27))
    require(sum(g["candidate_runs"] for g in groups) == document["total_selection_runs"], "Registered budget differs")
    return dict(status="passed", registration_sha256=sha(registration / "REGISTRATION.json"), csv_sha256_checks=csv_checks,
                identities=identities, groups=groups, total_candidate_runs=162, historical_test_observed=document["historical_test_observed"])


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(verify(options.registration, options.repo), indent=2))
