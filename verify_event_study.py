"""Independent source/metric/chronology and saved-model checks for the 2025 study."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from event_study import load_series,prepare_design,expected_intervals,new_hgb,ZONE
ROOT=Path(__file__).resolve().parent

def main():
    threadpool_limits(1);out=ROOT/'results/events2025';summary=json.loads((out/'summary.json').read_text());calendar=json.loads((ROOT/'event_calendar_2025.json').read_text())
    source,d=load_series(ROOT/'data_events/lane_series.csv');a=prepare_design(d,calendar);arrays=a['a'];checks={}
    checks['source_hash']=hashlib.sha256((ROOT/'data_events/lane_series.csv').read_bytes()).hexdigest()==summary['source_sha256']
    path=out/'predictions.csv';p=pd.read_csv(path if path.exists() else path.with_suffix('.csv.gz'));daily=pd.read_csv(out/'daily_metrics.csv');overall=pd.read_csv(out/'overall_metrics.csv').set_index('model');models=set(overall.index)
    checks['model_grid']=models=={'persistence','lstm_frozen','hgb_lags_frozen','hgb_calendar_frozen','hgb_calendar_rolling56'}
    valid=(np.lib.stride_tricks.sliding_window_view(np.diff(d.Date.to_numpy())==np.timedelta64(15,'m'),20).all(axis=1));ends=np.arange(20,len(d))[valid]
    checks['gap_windows_match']=np.array_equal(ends,a['indices'])
    points=True;calibration=True
    for model in models:
        for split,key in [('validation','validation'),('test','test')]:
            g=p[(p.model==model)&(p.split==split)].sort_values('source_clean_row');expected=a['indices'][arrays[key]]
            points &= np.array_equal(g.source_clean_row.to_numpy(),expected) and np.array_equal(g.actual.to_numpy(),d.Volume.to_numpy()[expected])
            points &= np.array_equal(pd.to_datetime(g.timestamp_utc).to_numpy(),d.Date.to_numpy()[expected])
            points &= np.array_equal(pd.to_datetime(g.local_time,utc=True).to_numpy(),d.interval_start_utc.to_numpy()[expected])
        test=p[(p.model==model)&(p.split=='test')];e=test.prediction.to_numpy()-test.actual.to_numpy()
        calc=dict(n_intervals=len(e),mae=abs(e).mean(),rmse=np.sqrt((e*e).mean()),bias=e.mean(),wape_pct=100*abs(e).sum()/test.actual.sum())
        calibration &= all(np.isclose(v,overall.loc[model,k],atol=1e-10,rtol=1e-10) for k,v in calc.items())
    checks['source_targets_dates_and_all_models_match']=bool(points);checks['overall_metrics_recompute']=bool(calibration)
    daily_ok=True
    for row in daily.itertuples():
        g=p[(p.model==row.model)&(p.date==row.date)&(p.split==row.split)];e=g.prediction.to_numpy()-g.actual.to_numpy()
        expected=expected_intervals(row.date)
        vals=dict(n=len(g),expected_intervals=expected,actual_mean=g.actual.mean(),predicted_mean=g.prediction.mean(),mae=abs(e).mean(),rmse=np.sqrt((e*e).mean()),bias=e.mean())
        daily_ok &= all(np.isclose(v,getattr(row,k),rtol=1e-10,atol=1e-10) for k,v in vals.items()) and bool(row.complete_24h)==(len(g)==expected)
    checks['all_daily_means_errors_and_coverage_recompute']=bool(daily_ok)
    checks['dst_day_lengths']=expected_intervals('2025-04-06')==100 and expected_intervals('2025-10-05')==92 and expected_intervals('2025-12-25')==96
    coverage=pd.read_csv(out/'source_daily_coverage.csv')
    checks['all_365_days_including_absent_date']=len(coverage)==365 and int(coverage.set_index('date').loc['2025-05-27','count'])==0
    checks['christmas_full_24h']=len(daily[daily.date=='2025-12-25'])==5 and daily.loc[daily.date=='2025-12-25','complete_24h'].all()
    cal_ok=True
    for model in models:
        reference='hgb_calendar_frozen' if model=='hgb_calendar_rolling56' else model
        v=daily[(daily.model==reference)&(daily.split=='validation')&daily.complete_24h].mae.quantile(.95)
        target=daily[daily.model==model]
        cal_ok &= np.allclose(target.error_flag_threshold,v) and np.array_equal(target.high_forecast_error_flag.to_numpy(),(target.complete_24h&(target.mae>v)).to_numpy())
    checks['daily_error_flags_use_validation_only']=bool(cal_ok)
    predictions_ok=True
    for model,key in [('hgb_lags_frozen','x_lags'),('hgb_calendar_frozen','x_calendar')]:
        fitted=joblib.load(out/'models'/f'{model}.joblib');values=np.maximum(0,fitted.predict(a[key][arrays['test']]))
        saved=p[(p.model==model)&(p.split=='test')].sort_values('source_clean_row').prediction.to_numpy();predictions_ok &= np.allclose(saved,values,rtol=1e-12,atol=1e-12)
    checks['both_saved_hgb_models_match_predictions']=bool(predictions_ok)
    refits=pd.read_csv(out/'rolling_refits.csv');local=pd.DatetimeIndex(d.local_time.iloc[a['indices']]);first=pd.DatetimeIndex(d.local_time.iloc[a['indices']-20]);rolling_ok=True
    for row in refits.itertuples():
        start,end,history=pd.Timestamp(row.block_start),pd.Timestamp(row.block_end_exclusive),pd.Timestamp(row.training_start)
        train=(local<start)&(first>=history);test=arrays['test']&(local>=start)&(local<end)
        rolling_ok &= local[train].max()<start and int(train.sum())==row.fit_windows and int(test.sum())==row.test_windows
        model=new_hgb().fit(a['x_calendar'][train],a['target'][train]);pred=np.maximum(0,model.predict(a['x_calendar'][test]))
        saved=p[(p.model=='hgb_calendar_rolling56')&(p.split=='test')&p.source_clean_row.isin(a['indices'][test])].sort_values('source_clean_row').prediction.to_numpy()
        rolling_ok &= np.allclose(saved,pred,rtol=1e-12,atol=1e-12)
    checks['all_rolling_refits_use_past_only_and_reproduce']=bool(rolling_ok)
    import torch
    from torch import nn
    class Net(nn.Module):
        def __init__(self):super().__init__();self.rnn=nn.LSTM(1,32,batch_first=True);self.out=nn.Linear(32,1)
        def forward(self,x):return self.out(self.rnn(x)[1][0][-1]).squeeze(-1)
    torch.set_num_threads(1);checkpoint=torch.load(out/'models/event_lstm.pt',weights_only=True,map_location='cpu');model=Net();model.load_state_dict(checkpoint['state']);model.eval()
    with torch.no_grad():values=torch.cat([model(batch) for batch in torch.tensor(arrays['x'][arrays['test']]).split(512)]).numpy()
    values=np.maximum(0,values*arrays['std']+arrays['mean']);saved=p[(p.model=='lstm_frozen')&(p.split=='test')].sort_values('source_clean_row').prediction.to_numpy()
    checks['saved_lstm_weights_and_training_only_scaling_match']=np.isclose(checkpoint['mean'],arrays['mean']) and np.isclose(checkpoint['std'],arrays['std']) and np.allclose(values,saved,rtol=1e-12,atol=1e-12)
    result=dict(verified_on='2026-11-19',prediction_rows=len(p),daily_metric_rows=len(daily),rolling_models_refitted=len(refits),checks={k:bool(v) for k,v in checks.items()},passed=all(checks.values()))
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['passed']
if __name__=='__main__':main()
