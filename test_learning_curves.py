import unittest
import numpy as np,pandas as pd
from learning_curves import masks_for_days,run_detector,RAMMonitor

class HistoryCurveTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame(dict(Date=pd.date_range('2020-01-01',periods=1000,freq='15min'),Volume=30+np.arange(1000)%96,label_set_1=0,label_set_2=0))
    def test_every_training_input_is_inside_lookback(self):
        d=self.data();a=masks_for_days(d,2)
        first=d.Date.to_numpy()[a['ends'][a['fit']]-7]
        self.assertTrue((first>a['start'].to_datetime64()).all())
        self.assertTrue((a['ends'][a['fit']]<a['nfit']).all())
    def test_validation_and_test_targets_are_fixed(self):
        d=self.data();a=masks_for_days(d,2);b=masks_for_days(d,4)
        for split in ['val','test']:np.testing.assert_array_equal(a['ends'][a[split]],b['ends'][b[split]])
    def test_calendar_gaps_reduce_observed_coverage(self):
        d=self.data().drop(index=500).reset_index(drop=True);a=masks_for_days(d,3)
        self.assertLess(int(a['rows'].sum()),3*96)
        self.assertFalse(((a['ends'][a['fit']]>=500)&(a['ends'][a['fit']]<=506)).any())
    def test_invalid_or_unavailable_history_rejected(self):
        for days in [0,-1,30]:
            with self.assertRaises(ValueError):masks_for_days(self.data(),days)
    def test_training_callback_and_sampled_ram(self):
        calls=[]
        with RAMMonitor() as monitor:
            result=run_detector(self.data(),3,'sitdd_adapted',42,{'sitdd_k':3},lambda:calls.append(monitor.snapshot()))
        self.assertEqual(len(calls),1);self.assertGreaterEqual(calls[0],monitor.baseline/1024**2)
        self.assertEqual(len(result['test_indices']),150);self.assertGreater(result['training_seconds'],0)
        self.assertEqual(result['f1_set1'],0)
if __name__=='__main__':unittest.main()
