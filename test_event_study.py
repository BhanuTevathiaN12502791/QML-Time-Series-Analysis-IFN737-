import unittest
import numpy as np,pandas as pd
from event_study import expected_intervals,calendar_rows,known_calendar_features,load_series,prepare_design,aggregate_daily,daily_demand_screen
from pathlib import Path
import tempfile

CAL=dict(public_holidays=[{'date':'2025-12-25','name':'Christmas'}],school_terms=[{'term':3,'start':'2025-07-21','end':'2025-09-19'}],weather_events=[{'name':'Storm','start':'2025-10-22','end':'2025-10-22'}])
class EventStudyTests(unittest.TestCase):
    def test_dst_local_days(self):
        self.assertEqual(expected_intervals('2025-04-06'),100)
        self.assertEqual(expected_intervals('2025-10-05'),92)
        self.assertEqual(expected_intervals('2025-12-25'),96)
    def test_fixed_aest_end_becomes_correct_local_start(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.csv';pd.DataFrame(dict(lane_id=['100:2'],timestamp=['2025-12-24T23:15:00+10:00'],volume=[25])).to_csv(p,index=False)
            raw,d=load_series(p)
            self.assertEqual(str(d.local_time.iloc[0]),'2025-12-25 00:00:00+11:00')
            self.assertEqual(d.Date.iloc[0],pd.Timestamp('2025-12-24 13:00:00'))
    def test_weather_is_not_a_predictor(self):
        t=pd.date_range('2025-10-22',periods=10,freq='15min',tz='Australia/Melbourne')
        a=known_calendar_features(t,CAL);other={**CAL,'weather_events':[]}
        np.testing.assert_array_equal(a,known_calendar_features(t,other))
    def test_calendar_labels_christmas_and_term(self):
        t=pd.DatetimeIndex(['2025-12-25 12:00','2025-08-01 12:00']).tz_localize('Australia/Melbourne');c=calendar_rows(t,CAL)
        self.assertTrue(c.is_public_holiday.iloc[0]);self.assertEqual(c.school_term.iloc[1],3)
    def test_daily_mae_and_bias_do_not_cancel(self):
        t=pd.date_range('2025-12-25',periods=96,freq='15min',tz='Australia/Melbourne')
        c=calendar_rows(t,CAL);c['model']='x';c['split']='test';c['actual']=10.;c['prediction']=np.tile([9.,11.],48)
        source=pd.DataFrame(dict(local_time=t,Volume=np.full(96,10.)))
        row=aggregate_daily(c,source,CAL).iloc[0]
        self.assertEqual(row.mae,1.);self.assertEqual(row.bias,0.);self.assertTrue(row.complete_24h)
    def test_partial_day_is_not_complete(self):
        t=pd.date_range('2025-12-25',periods=95,freq='15min',tz='Australia/Melbourne');c=calendar_rows(t,CAL)
        c['model']='x';c['split']='test';c['actual']=10.;c['prediction']=10.
        source=pd.DataFrame(dict(local_time=t,Volume=np.full(95,10.)));self.assertFalse(aggregate_daily(c,source,CAL).complete_24h.iloc[0])
    def test_demand_screen_uses_only_prior_days(self):
        t=pd.date_range('2025-01-01',periods=96*100,freq='15min',tz='Australia/Melbourne');source=pd.DataFrame(dict(local_time=t,Volume=np.full(len(t),10.)))
        a=daily_demand_screen(source,CAL);cut='2025-03-01';source.loc[source.local_time.dt.strftime('%Y-%m-%d')>=cut,'Volume']=10000.
        b=daily_demand_screen(source,CAL)
        np.testing.assert_array_equal(a.loc[a.date<cut,'reference_mean_median'],b.loc[b.date<cut,'reference_mean_median'])
    def test_missing_days_remain_explicit_in_demand_table(self):
        t=pd.date_range('2025-01-01',periods=96,freq='15min',tz='Australia/Melbourne');source=pd.DataFrame(dict(local_time=t,Volume=10.))
        table=daily_demand_screen(source,CAL);self.assertEqual(len(table),365);self.assertEqual(table.loc[table.date=='2025-05-27','available_observations'].iloc[0],0)
if __name__=='__main__':unittest.main()
