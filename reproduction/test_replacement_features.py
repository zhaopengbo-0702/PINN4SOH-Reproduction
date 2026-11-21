"""Analytical and failure-contract checks for the transparent replacement API."""
import unittest
import numpy as np

from export_replacement_features import extract_window, extract_cycle, FEATURES


class FeatureTests(unittest.TestCase):
    def test_linear_voltage_physical_seconds(self):
        t = np.array([0., 2., 4., 6.])
        x = 4 + t / 100
        q = .1 + t / 3600
        f, r = extract_window(x, t, q, np.ones(4), "voltage")
        self.assertAlmostEqual(f["voltage slope"], .01)
        self.assertAlmostEqual(f["CC Q"], 6 / 3600)
        self.assertEqual(f["CC charge time"], 6)
        self.assertAlmostEqual(f["voltage std"], np.sqrt(.0005))
        self.assertEqual(r["status"], "computed")
        # Known probabilities after min-max normalization: [0,1/6,1/3,1/2].
        self.assertAlmostEqual(f["voltage entropy"], -sum(p * np.log(p) for p in [1/6, 1/3, 1/2]))

    def test_negative_current_slope_and_paper_integral(self):
        t = np.array([0., 1., 3., 4.])
        x = .5 - .1 * t
        f, _ = extract_window(x, t, np.zeros(4), x, "current", "paper_note2")
        self.assertAlmostEqual(f["current slope"], -.1)
        self.assertAlmostEqual(f["CV Q"], 1.2 / 3600)
        self.assertAlmostEqual(f["current std"], np.sqrt(.1 / 3))

    def test_invalid_window_is_preserved_not_repaired(self):
        for t, cause in [([0, 1, 1, 3], "time_not_strictly_increasing"), ([0, 1, 3, 2], "time_not_strictly_increasing")]:
            f, r = extract_window(np.arange(4), t, np.arange(4), np.ones(4), "voltage")
            self.assertTrue(all(value is None for value in f.values()))
            self.assertIn(cause, r["issues"])
        f, r = extract_window([1, 2, 3], [0, 1, 2], [0, 1, 2], [1, 1, 1], "current")
        self.assertIn("fewer_than_four_points", r["issues"])

    def test_constant_moments_are_missing(self):
        f, r = extract_window(np.ones(4), np.arange(4), np.arange(4), np.ones(4), "voltage")
        self.assertEqual(f["voltage slope"], 0)
        self.assertEqual(f["voltage std"], 0)
        self.assertIsNone(f["voltage kurtosis"])
        self.assertIsNone(f["voltage entropy"])
        self.assertIn("voltage entropy", r["undefined_features"])

    def test_explicit_stage_masks_and_complete_schema(self):
        cycle = dict(V=np.array([4.01, 4.04, 4.08, 4.12, 4.2, 4.2, 4.2, 4.2]),
                     I=np.array([2, 2, 2, 2, .48, .4, .3, .15]), t_seconds=np.arange(8.),
                     Q_Ah=np.arange(8.) / 1000, cc_mask=np.array([1,1,1,1,0,0,0,0], dtype=bool),
                     cv_mask=np.array([0,0,0,0,1,1,1,1], dtype=bool))
        f, r = extract_cycle(cycle, "TJU")
        self.assertEqual(set(f), set(FEATURES))
        self.assertEqual(r["voltage"]["source_indices"], [0,1,2,3])
        self.assertAlmostEqual(f["current slope"], -.11)


if __name__ == "__main__":
    unittest.main()
