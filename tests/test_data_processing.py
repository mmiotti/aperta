"""Tests for `aperta.data_processing.weighted_group_mean`.

Other data_processing helpers are exercised indirectly through
`test_workflow` and higher-level module tests; this file focuses on the
weighted-mean aggregator's edge cases (NaN handling, empty groups, etc.).
"""

import unittest

import numpy as np
import pandas as pd

from aperta.data_processing import weighted_group_mean


class WeightedGroupMeanTestCase(unittest.TestCase):
    def test_basic_uniform_weights(self):
        v = pd.Series([1.0, 2.0, 3.0, 4.0])
        w = pd.Series([1.0, 1.0, 1.0, 1.0])
        g = pd.Series(["A", "A", "B", "B"])
        out = weighted_group_mean(v, w, g)
        self.assertAlmostEqual(out["A"], 1.5)
        self.assertAlmostEqual(out["B"], 3.5)

    def test_weighted_shifts_toward_heavier_row(self):
        v = pd.Series([0.0, 10.0])
        w = pd.Series([1.0, 3.0])  # second row 3× heavier
        g = pd.Series(["A", "A"])
        # (0 * 1 + 10 * 3) / (1 + 3) = 30 / 4 = 7.5
        out = weighted_group_mean(v, w, g)
        self.assertAlmostEqual(out["A"], 7.5)

    def test_nan_value_drops_row(self):
        v = pd.Series([1.0, float("nan"), 3.0])
        w = pd.Series([1.0, 5.0, 1.0])  # NaN row's weight is ignored
        g = pd.Series(["A", "A", "A"])
        # (1 + 3) / (1 + 1) = 2.0
        out = weighted_group_mean(v, w, g)
        self.assertAlmostEqual(out["A"], 2.0)

    def test_nan_weight_drops_row(self):
        v = pd.Series([1.0, 100.0, 3.0])
        w = pd.Series([1.0, float("nan"), 1.0])
        g = pd.Series(["A", "A", "A"])
        # NaN weight → row dropped, same as before
        out = weighted_group_mean(v, w, g)
        self.assertAlmostEqual(out["A"], 2.0)

    def test_all_nan_group_yields_nan(self):
        v = pd.Series([float("nan"), float("nan")])
        w = pd.Series([1.0, 1.0])
        g = pd.Series(["A", "A"])
        out = weighted_group_mean(v, w, g)
        self.assertTrue(np.isnan(out["A"]))

    def test_all_zero_weight_group_yields_nan(self):
        v = pd.Series([1.0, 2.0])
        w = pd.Series([0.0, 0.0])
        g = pd.Series(["A", "A"])
        out = weighted_group_mean(v, w, g)
        self.assertTrue(np.isnan(out["A"]))

    def test_result_indexed_by_group_ids(self):
        v = pd.Series([1.0, 2.0, 3.0])
        w = pd.Series([1.0, 1.0, 1.0])
        g = pd.Series(["X", "Y", "Z"])
        out = weighted_group_mean(v, w, g)
        self.assertEqual(sorted(out.index.tolist()), ["X", "Y", "Z"])

    def test_shared_row_index_across_inputs(self):
        # Non-trivial index is preserved through the groupby
        idx = pd.Index(["c1", "c2", "c3", "c4"], name="cell_id")
        v = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        w = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
        g = pd.Series(["Z1", "Z1", "Z2", "Z2"], index=idx)
        out = weighted_group_mean(v, w, g)
        self.assertAlmostEqual(out["Z1"], 1.5)
        self.assertAlmostEqual(out["Z2"], 3.5)


if __name__ == "__main__":
    unittest.main()
