"""Guard against falsely treating different scientific cohorts as matched."""
import copy
import unittest

from diagnose_transfer_environment_schema import (
    canonical_environment, corroborate_environment, compare_split_evidence, stable, validate_task_identity)


class EnvironmentSchemaTests(unittest.TestCase):
    def setUp(self):
        self.env = dict(python="3.10.13", torch="1.13.1+cu116", numpy="1.24.4",
                        pandas="1.5.3", scipy="1.10.1", sklearn="1.2.2",
                        cuda="11.6", cuda_available=True, gpu="RTX 3090")
        self.task = {"threads": 2}
        self.frozen = {"threads": 2, "environment_signatures": {stable(self.env): self.env}}
        self.coordinator = {"assigned_gpu": "1"}
        self.manifest = {"assigned_cuda_visible_devices": "1"}

    def test_aliases_resolve_only_after_probe_corroboration(self):
        raw = dict(self.env, threads=2)
        raw["scikit_learn"] = raw.pop("sklearn")
        raw["cuda_runtime"] = raw.pop("cuda")
        raw.pop("scipy")
        result = corroborate_environment(raw, self.task, self.frozen,
                                         self.coordinator, self.manifest)
        self.assertEqual(result["filled_from_completed_coordinator_probe"], ["scipy"])
        self.assertEqual(result["normalized"], dict(self.env, threads=2))

    def test_conflicting_alias_is_rejected(self):
        with self.assertRaises(ValueError):
            canonical_environment(dict(self.env, scikit_learn="1.5.0"))

    def test_actual_version_difference_cannot_be_filled(self):
        raw = dict(self.env, numpy="1.25.0")
        with self.assertRaises(ValueError):
            corroborate_environment(raw, self.task, self.frozen,
                                     self.coordinator, self.manifest)

    def test_thread_difference_is_rejected(self):
        with self.assertRaises(ValueError):
            corroborate_environment(dict(self.env, threads=4), self.task, self.frozen,
                                     self.coordinator, self.manifest)

    def test_actual_gpu_visibility_must_match_launch_record(self):
        with self.assertRaises(ValueError):
            corroborate_environment(dict(self.env, CUDA_VISIBLE_DEVICES="0"), self.task,
                                     self.frozen, self.coordinator, self.manifest)

    def test_no_probe_cannot_substitute_missing_scipy(self):
        raw = dict(self.env)
        raw.pop("scipy")
        with self.assertRaises(ValueError):
            corroborate_environment(raw, self.task, {"threads": 2, "environment_signatures": {}},
                                     self.coordinator, self.manifest)

    def test_n2_is_ordered_prefix_comparison(self):
        train = {"role": "train_validation", "paths": ["a"], "nominal_capacity": 2,
                 "batteries": [{"path": "a", "n_pairs": 100}]}
        test = {"role": "test", "paths": ["c"], "nominal_capacity": 2,
                "batteries": [{"path": "c", "n_pairs": 50}]}
        left = {"splits": [train, test], "sample_counts": {"train": 80, "valid": 20, "test": 50}}
        right = copy.deepcopy(left)
        right["splits"][0]["paths"].append("b")
        right["splits"][0]["batteries"].append({"path": "b", "n_pairs": 100})
        self.assertTrue(all(compare_split_evidence(left, right, 2)[0].values()))
        right["splits"][0]["paths"].reverse()
        self.assertFalse(compare_split_evidence(left, right, 2)[0]["adaptation_paths_equal_target_prefix"])

    def test_identical_paths_with_different_battery_boundaries_fail(self):
        split = {"role": "train_validation", "paths": ["a"], "nominal_capacity": 2,
                 "batteries": [{"path": "a", "n_pairs": 100}]}
        test = {"role": "test", "paths": ["b"], "batteries": [{"path": "b", "n_pairs": 50}]}
        left = {"splits": [split, test], "sample_counts": {"train": 80}}
        right = copy.deepcopy(left)
        right["splits"][1]["batteries"][0]["n_pairs"] = 49
        self.assertFalse(compare_split_evidence(left, right, 1)[0]["ordered_test_split_and_boundaries_equal"])

    def test_raw_extended_task_hash_preserves_absent_smoke_default(self):
        raw = {"runner": "extended", "seed": 1}
        effective = dict(raw, smoke=False)
        validate_task_identity(effective, raw, effective, raw, stable(raw), stable(raw))

    def test_effective_task_cannot_silently_change_seed(self):
        raw = {"runner": "extended", "seed": 1}
        effective = dict(raw, smoke=False, seed=2)
        with self.assertRaises(ValueError):
            validate_task_identity(effective, raw, effective, raw, stable(raw), stable(raw))


if __name__ == "__main__":
    unittest.main()
