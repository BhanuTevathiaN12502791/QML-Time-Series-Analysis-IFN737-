import unittest
import numpy as np
import pandas as pd
from analyse_event_tradeoffs import paired_weekly_interval


class PairedErrorTests(unittest.TestCase):
    def frame(self):
        dates = pd.to_datetime(['2025-03-12', '2025-03-12', '2025-03-20', '2025-03-27'])
        return pd.DataFrame({'timestamp_utc': ['a', 'b', 'c', 'd'], 'date': dates,
                             'actual': np.zeros(4), 'prediction': [1., 3., 8., 4.]})

    def test_paired_identical_models_have_zero_difference(self):
        result = paired_weekly_interval(self.frame(), self.frame(), draws=500)
        self.assertEqual(result['delta_mae'], 0)
        self.assertEqual(result['ci95_low'], 0)
        self.assertEqual(result['ci95_high'], 0)
        self.assertEqual(result['n_week_clusters'], 3)

    def test_unequal_cluster_sizes_keep_interval_weighted_estimand(self):
        left = self.frame()
        right = left.assign(prediction=0.)
        result = paired_weekly_interval(left, right, draws=1000)
        self.assertEqual(result['delta_mae'], 4.)
        self.assertLess(result['ci95_low'], result['delta_mae'])
        self.assertGreater(result['ci95_high'], result['delta_mae'])

    def test_missing_target_rejected(self):
        with self.assertRaises(ValueError):
            paired_weekly_interval(self.frame(), self.frame().iloc[:3])


if __name__ == '__main__':
    unittest.main()
