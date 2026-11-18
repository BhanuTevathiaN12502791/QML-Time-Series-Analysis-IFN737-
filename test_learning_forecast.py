import importlib.util
import unittest
import numpy as np
import pandas as pd
from learning_forecast import forecast_arrays, run_forecast


class ForecastLearningTests(unittest.TestCase):
    def data(self, n=1000):
        return pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=n, freq='15min'),
                             'Volume': 20 + np.arange(n) % 96})

    def test_every_training_input_and_target_inside_lookback(self):
        d = self.data()
        start = d.Date.iloc[200]
        a = forecast_arrays(d, start, 700, 850)
        targets = a['indices'][a['fit']]
        self.assertTrue((d.Date.iloc[targets - 20] > start).all())
        self.assertTrue((targets < 700).all())
        self.assertEqual(targets[0], 221)

    def test_validation_and_test_targets_fixed_across_training_sizes(self):
        d = self.data()
        small = forecast_arrays(d, d.Date.iloc[300], 700, 850)
        large = forecast_arrays(d, d.Date.iloc[0], 700, 850)
        for split in ['validation', 'test']:
            np.testing.assert_array_equal(small['indices'][small[split]], large['indices'][large[split]])

    def test_future_values_and_labels_do_not_affect_scaling(self):
        d = self.data()
        a = forecast_arrays(d, d.Date.iloc[200], 700, 850)
        d.loc[700:, 'Volume'] = 999999
        d['label_set_1'] = 1
        b = forecast_arrays(d, d.Date.iloc[200], 700, 850)
        self.assertEqual(a['mean'], b['mean'])
        self.assertEqual(a['std'], b['std'])
        np.testing.assert_array_equal(a['x'][a['fit']], b['x'][b['fit']])

    def test_forecast_window_excludes_current_target(self):
        d = self.data()
        a = forecast_arrays(d, d.Date.iloc[0] - pd.Timedelta(minutes=15), 700, 850)
        i = np.flatnonzero(a['indices'] == 800)[0]
        reconstructed = a['x'][i, :, 0] * a['std'] + a['mean']
        np.testing.assert_allclose(reconstructed, d.Volume.iloc[780:800], atol=1e-5)
        self.assertEqual(a['actual'][i], d.Volume.iloc[800])

    def test_gap_invalidates_all_crossing_inputs(self):
        d = self.data().drop(index=400).reset_index(drop=True)
        a = forecast_arrays(d, d.Date.iloc[0] - pd.Timedelta(minutes=15), 700, 850)
        self.assertFalse(((a['indices'] >= 400) & (a['indices'] <= 419)).any())

    def test_actual_zero_counts_remain_valid(self):
        d = self.data()
        d['Volume'] = 0
        a = forecast_arrays(d, d.Date.iloc[0] - pd.Timedelta(minutes=15), 700, 850)
        self.assertEqual(a['training_windows'], 680)
        self.assertEqual(a['std'], 1.)
        self.assertTrue((a['actual'] == 0).all())

    def test_invalid_split_or_empty_training_history_rejected(self):
        d = self.data()
        with self.assertRaises(ValueError): forecast_arrays(d, d.Date.iloc[0], 850, 700)
        with self.assertRaises(ValueError): forecast_arrays(d, d.Date.iloc[695], 700, 850)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'optional torch not installed')
    def test_one_epoch_forecast_and_training_callback(self):
        d = self.data(200)
        calls = []
        r = run_forecast(d, d.Date.iloc[0] - pd.Timedelta(minutes=15), 100, 150, 42,
                         max_epochs=1, patience=1, on_training_complete=lambda: calls.append('trained'))
        self.assertEqual(calls, ['trained'])
        self.assertEqual(r['epochs_trained'], 1)
        self.assertEqual(r['parameter_count'], 4513)
        self.assertTrue(np.isfinite(r['mae']))
        self.assertEqual(len(r['test_indices']), 50)
        self.assertGreaterEqual(r['training_seconds'], r['fit_seconds'])


if __name__ == '__main__': unittest.main()
