"""Melbourne 2025 holiday, school-term and weather-window forecast stress study.

Rolling one-step (15-minute) forecasts are aggregated over civil days. Calendar
markers are contextual exposures, not ground-truth incident/anomaly labels.
"""
import argparse,copy,hashlib,json,pickle,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from learning_forecast import forecast_arrays,configure_torch

ROOT=Path(__file__).resolve().parent
ZONE='Australia/Melbourne'
MODEL_NAMES=['persistence','hgb_lags_frozen','hgb_calendar_frozen','hgb_calendar_rolling56','lstm_frozen']


def expected_intervals(date):
    start=pd.Timestamp(date).tz_localize(ZONE)
    end=(start.tz_localize(None)+pd.Timedelta(days=1)).tz_localize(ZONE)
    return int((end.tz_convert('UTC')-start.tz_convert('UTC')).total_seconds()/900)


def calendar_rows(local_times,calendar):
    times=pd.Series(pd.DatetimeIndex(local_times)).reset_index(drop=True)
    dates=times.dt.strftime('%Y-%m-%d')
    holiday_names={x['date']:x['name'] for x in calendar['public_holidays']}
    term=np.zeros(len(times),dtype=int)
    for item in calendar['school_terms']:
        term[(dates>=item['start'])&(dates<=item['end'])]=int(item['term'])
    weather=np.full(len(times),'',dtype=object)
    for item in calendar.get('weather_events',[]):
        mask=((dates>=item['start'])&(dates<=item['end'])).to_numpy()
        weather[mask]=np.where(weather[mask]=='',item['name'],weather[mask]+'; '+item['name'])
    month=times.dt.month.to_numpy()
    season=np.select([np.isin(month,[12,1,2]),np.isin(month,[3,4,5]),np.isin(month,[6,7,8])],['summer','autumn','winter'],default='spring')
    return pd.DataFrame(dict(date=dates,weekday=times.dt.day_of_week.to_numpy(),month=times.dt.strftime('%Y-%m'),season=season,
        school_term=term,school_status=np.where(term>0,'term','school_break'),is_public_holiday=dates.isin(holiday_names).to_numpy(),
        public_holiday_name=dates.map(holiday_names).fillna('').to_numpy(),weather_event=weather,
        is_weekend=(times.dt.day_of_week>=5).to_numpy()))


def known_calendar_features(local_times,calendar):
    times=pd.Series(pd.DatetimeIndex(local_times)).reset_index(drop=True);c=calendar_rows(times,calendar)
    minute=times.dt.hour.to_numpy()*60+times.dt.minute.to_numpy();weekday=times.dt.day_of_week.to_numpy();day=times.dt.day_of_year.to_numpy()
    year_days=np.where(times.dt.is_leap_year.to_numpy(),366,365)
    # Weather is deliberately absent: retrospective observed storm flags are not future-known predictors.
    return np.column_stack([np.sin(2*np.pi*minute/1440),np.cos(2*np.pi*minute/1440),
        np.sin(2*np.pi*weekday/7),np.cos(2*np.pi*weekday/7),np.sin(2*np.pi*day/year_days),np.cos(2*np.pi*day/year_days),
        c.is_public_holiday.astype(float),c.is_weekend.astype(float),(c.school_term>0).astype(float),(c.school_term==0).astype(float)])


def load_series(path,lane='100:2',year=2025):
    source=pd.read_csv(path,dtype={'lane_id':str});source=source[source.lane_id==lane].copy()
    if source.empty:raise ValueError(f'No records for {lane}')
    source['timestamp_end']=pd.to_datetime(source.timestamp,utc=True)
    source['interval_start_utc']=source.timestamp_end-pd.Timedelta(minutes=15)
    source['local_time']=source.interval_start_utc.dt.tz_convert(ZONE)
    source=source[source.local_time.dt.year==year].sort_values('interval_start_utc').reset_index(drop=True)
    if source.interval_start_utc.duplicated().any():raise ValueError('Duplicate detector timestamps')
    source['Volume']=pd.to_numeric(source.volume,errors='coerce')
    bad=source.Volume.notna()&((source.Volume<0)|~np.isfinite(source.Volume))
    if bad.any():raise ValueError('Negative/nonfinite unmasked count')
    source['Date']=source.interval_start_utc.dt.tz_localize(None)
    clean=source[source.Volume.notna()].copy().reset_index(drop=True)
    return source,clean


def prepare_design(d,calendar):
    fit_cut=pd.Timestamp('2025-02-26',tz=ZONE).tz_convert('UTC').tz_localize(None)
    val_cut=pd.Timestamp('2025-03-12',tz=ZONE).tz_convert('UTC').tz_localize(None)
    nf=int((d.Date<fit_cut).sum());nv=int((d.Date<val_cut).sum())
    arrays=forecast_arrays(d,d.Date.iloc[0]-pd.Timedelta(minutes=15),nf,nv)
    indices=arrays['indices'];raw_windows=np.lib.stride_tricks.sliding_window_view(d.Volume.to_numpy(float),21)
    lags=raw_windows[indices-20,:20].copy()
    calendar_x=known_calendar_features(d.local_time.iloc[indices],calendar)
    return dict(a=arrays,nfit=nf,nval=nv,indices=indices,x_lags=lags,x_calendar=np.column_stack([lags,calendar_x]),
                target=d.Volume.to_numpy()[indices],fit_cut=fit_cut,val_cut=val_cut)


def new_hgb():
    return HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=42)


def fit_lstm(arrays,max_epochs=40,patience=7):
    torch=configure_torch(42);x=torch.from_numpy(arrays['x']);y=torch.from_numpy(arrays['y'])
    fit=torch.tensor(arrays['fit'],dtype=torch.bool);val=torch.tensor(arrays['validation'],dtype=torch.bool);test=torch.tensor(arrays['test'],dtype=torch.bool)
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__();self.rnn=torch.nn.LSTM(1,32,batch_first=True);self.out=torch.nn.Linear(32,1)
        def forward(self,b):return self.out(self.rnn(b)[1][0][-1]).squeeze(-1)
    started=time.perf_counter();model=Net();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[fit],y[fit]),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(42))
    best,state,stale,best_epoch=float('inf'),None,0,0;history=[]
    for epoch in range(1,max_epochs+1):
        model.train();total=0.
        for bx,by in loader:
            optimizer.zero_grad();loss=torch.nn.functional.mse_loss(model(bx),by);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())*len(bx)
        model.eval()
        with torch.no_grad():vl=float(torch.nn.functional.mse_loss(model(x[val]),y[val]))
        history.append(dict(epoch=epoch,train_mse=total/int(fit.sum()),validation_mse=vl))
        if vl<best:best=vl;state=copy.deepcopy(model.state_dict());stale=0;best_epoch=epoch
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(state);model.eval();fit_seconds=time.perf_counter()-started
    start=time.perf_counter()
    with torch.no_grad():
        def predict(mask):return np.maximum(0,torch.cat([model(batch) for batch in x[mask].split(512)]).numpy()*arrays['std']+arrays['mean'])
        pv,pt=predict(val),predict(test)
    return pv,pt,dict(model='lstm_frozen',fit_seconds=fit_seconds,inference_seconds=time.perf_counter()-start,
        fit_windows=int(fit.sum()),epochs=len(history),best_epoch=best_epoch,model_bytes=len(pickle.dumps(state,protocol=5))),history,state


def forecast_metrics(actual,prediction):
    actual=np.asarray(actual,float);prediction=np.asarray(prediction,float);ok=np.isfinite(actual)&np.isfinite(prediction);a=actual[ok];p=prediction[ok]
    if not len(a):return dict(n_intervals=0,mae=np.nan,rmse=np.nan,bias=np.nan,wape_pct=np.nan)
    e=p-a
    return dict(n_intervals=len(a),mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),bias=float(e.mean()),wape_pct=float(100*abs(e).sum()/a.sum()) if a.sum() else np.nan)


def fit_predict_all(d,design,out,skip_lstm=False,max_epochs=40):
    a=design['a'];fit,val,test=a['fit'],a['validation'],a['test'];y=design['target'];pred={};timings=[];rolling=[]
    pred['persistence']=(a['persistence'][val],a['persistence'][test]);timings.append(dict(model='persistence',fit_seconds=0.,inference_seconds=0.,fit_windows=0,model_bytes=0))
    for name,key in [('hgb_lags_frozen','x_lags'),('hgb_calendar_frozen','x_calendar')]:
        x=design[key];start=time.perf_counter();model=new_hgb().fit(x[fit],y[fit]);seconds=time.perf_counter()-start
        start=time.perf_counter();pv=np.maximum(0,model.predict(x[val]));pt=np.maximum(0,model.predict(x[test]));predict_seconds=time.perf_counter()-start
        pred[name]=(pv,pt);timings.append(dict(model=name,fit_seconds=seconds,inference_seconds=predict_seconds,fit_windows=int(fit.sum()),model_bytes=len(pickle.dumps(model,protocol=5))))
        import joblib
        (out/'models').mkdir(exist_ok=True);joblib.dump(model,out/'models'/f'{name}.joblib',compress=3)
    # The rolling model can use observations that have become available before each block.
    local=pd.DatetimeIndex(d.local_time.iloc[design['indices']]);first=pd.Timestamp('2025-03-12',tz=ZONE);finish=pd.Timestamp('2026-01-01',tz=ZONE)
    rolling_pred=np.full(int(test.sum()),np.nan);test_positions=np.flatnonzero(test);total_fit=total_infer=0.;last_model=None
    block_start=first
    while block_start<finish:
        block_end=min(block_start+pd.DateOffset(days=28),finish);history_start=block_start-pd.DateOffset(days=56)
        first_inputs=pd.DatetimeIndex(d.local_time.iloc[design['indices']-20])
        train=(local<block_start)&(first_inputs>=history_start);block=test&(local>=block_start)&(local<block_end)
        if block.any():
            if train.sum()<100:raise ValueError('Insufficient rolling training windows')
            start=time.perf_counter();last_model=new_hgb().fit(design['x_calendar'][train],y[train]);ft=time.perf_counter()-start
            start=time.perf_counter();values=np.maximum(0,last_model.predict(design['x_calendar'][block]));it=time.perf_counter()-start
            rolling_pred[np.searchsorted(test_positions,np.flatnonzero(block))]=values;total_fit+=ft;total_infer+=it
            rolling.append(dict(block_start=str(block_start),block_end_exclusive=str(block_end),training_start=str(history_start),
                latest_training_target=str(local[train].max()),fit_windows=int(train.sum()),test_windows=int(block.sum()),fit_seconds=ft,inference_seconds=it))
        block_start=block_end
    if not np.isfinite(rolling_pred).all():raise ValueError('Unscored rolling targets')
    pred['hgb_calendar_rolling56']=(pred['hgb_calendar_frozen'][0].copy(),rolling_pred)
    timings.append(dict(model='hgb_calendar_rolling56',fit_seconds=total_fit,inference_seconds=total_infer,fit_windows=sum(r['fit_windows'] for r in rolling),n_refits=len(rolling),model_bytes=len(pickle.dumps(last_model,protocol=5))))
    if not skip_lstm:
        pv,pt,timing,history,state=fit_lstm(a,max_epochs);pred['lstm_frozen']=(pv,pt);timings.append(timing)
        pd.DataFrame(history).to_csv(out/'lstm_training_history.csv',index=False)
        import torch
        torch.save(dict(state=state,mean=a['mean'],std=a['std'],window=20),out/'models/event_lstm.pt')
    pd.DataFrame(rolling).to_csv(out/'rolling_refits.csv',index=False);pd.DataFrame(timings).to_csv(out/'model_timing.csv',index=False)
    return pred,timings


def long_predictions(d,design,predictions,calendar):
    frames=[];a=design['a']
    for name,(pv,pt) in predictions.items():
        for split,mask,pred in [('validation',a['validation'],pv),('test',a['test'],pt)]:
            indices=design['indices'][mask];frame=calendar_rows(d.local_time.iloc[indices],calendar)
            frame['timestamp_utc']=d.Date.iloc[indices].to_numpy();frame['local_time']=d.local_time.iloc[indices].astype(str).to_numpy()
            frame['actual']=d.Volume.iloc[indices].to_numpy();frame['prediction']=pred;frame['model']=name;frame['split']=split;frame['source_clean_row']=indices
            frames.append(frame)
    return pd.concat(frames,ignore_index=True)


def aggregate_daily(predictions,source,calendar):
    available=source.assign(date=source.local_time.dt.strftime('%Y-%m-%d')).groupby('date').Volume.agg(['count','mean','sum']).reindex(pd.date_range('2025-01-01','2025-12-31').strftime('%Y-%m-%d'))
    available['count']=available['count'].fillna(0)
    rows=[]
    for (name,date,split),g in predictions.groupby(['model','date','split']):
        e=g.prediction.to_numpy()-g.actual.to_numpy();expected=expected_intervals(date)
        row=dict(model=name,date=date,split=split,n=len(g),expected_intervals=expected,complete_24h=len(g)==expected,
            actual_mean=float(g.actual.mean()),predicted_mean=float(g.prediction.mean()),mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),
            wape_pct=float(100*abs(e).sum()/g.actual.sum()) if g.actual.sum() else np.nan,
            sum_abs_error=float(abs(e).sum()),sum_squared_error=float((e**2).sum()),sum_error=float(e.sum()),sum_actual=float(g.actual.sum()),
            available_observations=int(available.loc[date,'count']),source_observed_mean=float(available.loc[date,'mean']),source_observed_total=float(available.loc[date,'sum']))
        for col in ['weekday','month','season','school_term','school_status','is_public_holiday','public_holiday_name','weather_event','is_weekend']:row[col]=g.iloc[0][col]
        rows.append(row)
    result=pd.DataFrame(rows);thresholds={}
    for name,g in result[(result.split=='validation')&result.complete_24h].groupby('model'):
        thresholds[name]=float(g.mae.quantile(.95)) if len(g)>=5 else np.nan
    # Validation predictions for rolling are the frozen-calendar proxy; explicitly retain that provenance.
    for name in result.model.unique():
        threshold_model='hgb_calendar_frozen' if name=='hgb_calendar_rolling56' else name
        mask=result.model==name;result.loc[mask,'error_flag_threshold']=thresholds.get(threshold_model,np.nan)
        result.loc[mask,'threshold_calibration_model']=threshold_model
    result['high_forecast_error_flag']=result.complete_24h&(result.mae>result.error_flag_threshold)
    return result


def aggregate_groups(daily,groups):
    rows=[]
    for group_name,mask in groups.items():
        selected=daily[mask & (daily.split=='test') & daily.complete_24h]
        for model,g in selected.groupby('model'):
            n=g.n.sum();actual=g.sum_actual.sum()
            weekday=g.groupby('weekday').mae.mean()
            rows.append(dict(group=group_name,model=model,n_complete_days=len(g),n_intervals=int(n),mean_daily_mae=float(g.mae.mean()),
                pooled_mae=float(g.sum_abs_error.sum()/n),pooled_rmse=float(np.sqrt(g.sum_squared_error.sum()/n)),pooled_bias=float(g.sum_error.sum()/n),
                pooled_wape_pct=float(100*g.sum_abs_error.sum()/actual) if actual else np.nan,mean_observed_volume=float(g.actual_mean.mean()),
                weekday_balanced_mae=float(weekday.mean()),represented_weekdays=len(weekday),forecast_error_flagged_days=int(g.high_forecast_error_flag.sum())))
    return pd.DataFrame(rows)


def daily_demand_screen(source,calendar):
    raw=source.assign(date=source.local_time.dt.strftime('%Y-%m-%d')).groupby('date').Volume.agg(['count','mean']).reindex(pd.date_range('2025-01-01','2025-12-31').strftime('%Y-%m-%d')).rename_axis('date').reset_index()
    raw['count']=raw['count'].fillna(0)
    raw['expected_intervals']=raw.date.map(expected_intervals);raw['complete_24h']=raw['count']==raw.expected_intervals
    c=calendar_rows(pd.to_datetime(raw.date).dt.tz_localize(ZONE),calendar)
    for col in c:raw[col]=c[col]
    raw['reference_days']=0;raw['reference_mean_median']=np.nan;raw['robust_z']=np.nan;raw['demand_screen_flag']=False
    dates=pd.to_datetime(raw.date)
    for i,row in raw.iterrows():
        lookback=(dates<dates.iloc[i])&(dates>=dates.iloc[i]-pd.Timedelta(days=56))&(raw.weekday==row.weekday)&raw.complete_24h&~raw.is_public_holiday
        reference=raw.loc[lookback,'mean'].to_numpy()
        raw.loc[i,'reference_days']=len(reference)
        if row.complete_24h and len(reference)>=4:
            median=float(np.median(reference));mad=max(float(np.median(abs(reference-median))),1.)
            z=abs(float(row['mean'])-median)/(1.4826*mad)
            raw.loc[i,['reference_mean_median','robust_z','demand_screen_flag']]=[median,z,z>3.5]
    return raw.rename(columns={'count':'available_observations','mean':'observed_daily_mean'})


def make_outputs(source,d,predictions,daily,calendar,out):
    test=predictions[predictions.split=='test'];overall=[]
    for model,g in test.groupby('model'):overall.append(dict(model=model,**forecast_metrics(g.actual,g.prediction)))
    pd.DataFrame(overall).to_csv(out/'overall_metrics.csv',index=False)
    groups={'ordinary_nonholiday':~daily.is_public_holiday,'public_holiday':daily.is_public_holiday,'school_term':daily.school_status=='term','school_break':daily.school_status=='school_break',
        'Christmas Day':daily.date=='2025-12-25','Boxing Day':daily.date=='2025-12-26'}
    for event in calendar.get('weather_events',[]):groups[event['name']]=(daily.date>=event['start'])&(daily.date<=event['end'])
    for holiday in calendar['public_holidays']:groups['holiday: '+holiday['name']]=daily.date==holiday['date']
    event=aggregate_groups(daily,groups).rename(columns={'group':'event'});event.to_csv(out/'event_metrics.csv',index=False)
    monthly=aggregate_groups(daily,{month:daily.month==month for month in sorted(daily.month.unique())}).rename(columns={'group':'month'});monthly.to_csv(out/'monthly_metrics.csv',index=False)
    seasons=aggregate_groups(daily,{season:daily.season==season for season in ['summer','autumn','winter','spring']}).rename(columns={'group':'season'});seasons.to_csv(out/'seasonal_metrics.csv',index=False)
    periods={'term3_first8weeks':(daily.date>='2025-07-21')&(daily.date<='2025-09-14'),
        'following_spring_break':(daily.date>='2025-09-20')&(daily.date<='2025-10-05')}
    periods.update({name+'_weekdays_no_public':mask & (daily.weekday<5)&~daily.is_public_holiday for name,mask in list(periods.items())})
    school=aggregate_groups(daily,periods).rename(columns={'group':'period'});school.to_csv(out/'school_period_metrics.csv',index=False)
    demand=daily_demand_screen(source,calendar);demand.to_csv(out/'demand_anomaly_days.csv',index=False)
    return overall,event,monthly,seasons,school,demand


def plots(predictions,daily,monthly,school,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colours={'persistence':'#99a5af','lstm_frozen':'#ba572e','hgb_lags_frozen':'#6a7888','hgb_calendar_frozen':'#1a7188','hgb_calendar_rolling56':'#668237'}
    christmas=predictions[(predictions.date=='2025-12-25')&(predictions.split=='test')]
    fig,ax=plt.subplots(figsize=(11,4.5))
    first=True
    for model,g in christmas.groupby('model'):
        local=pd.to_datetime(g.local_time,utc=True).dt.tz_convert(ZONE);hours=local.dt.hour+local.dt.minute/60
        if first:ax.plot(hours,g.actual,color='#172f41',linewidth=2,label='Observed');first=False
        ax.plot(hours,g.prediction,label=model,alpha=.8,color=colours.get(model))
    if christmas.empty:ax.text(.5,.5,'No eligible Christmas forecasts',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Melbourne local interval-start hour',ylabel='Detections per 15 minutes',title='Christmas Day 2025 - rolling one-step forecasts');ax.set_xticks([0,4,8,12,16,20,24]);ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(out/'christmas_2025.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4.5))
    for model,g in daily[(daily.split=='test')&daily.complete_24h].groupby('model'):
        ax.plot(pd.to_datetime(g.date),g.mae,label=model,color=colours.get(model),linewidth=.8,alpha=.8)
    ax.set(xlabel='Melbourne civil date',ylabel='Daily mean absolute error',title='Complete local-day forecast error');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(out/'daily_mae_2025.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.5))
    for model,g in monthly.groupby('model'):ax.plot(g.month,g.mean_daily_mae,marker='o',label=model,color=colours.get(model))
    ax.set(xlabel='Month in test period',ylabel='Mean complete-day MAE',title='Seasonal drift and rolling adaptation');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15);fig.autofmt_xdate();fig.tight_layout();fig.savefig(out/'seasonal_metrics.png',dpi=180);plt.close(fig)
    selected=school[school.period.str.endswith('weekdays_no_public')]
    fig,ax=plt.subplots(figsize=(11,4.5));models=list(selected.model.unique());positions=np.arange(len(models));width=.36
    for j,period in enumerate(['term3_first8weeks_weekdays_no_public','following_spring_break_weekdays_no_public']):
        values=selected[selected.period==period].set_index('model').reindex(models).weekday_balanced_mae
        ax.bar(positions+(j-.5)*width,values,width,label='Term 3 first eight weeks' if j==0 else 'Following school break')
    ax.set_xticks(positions,models,rotation=15,ha='right');ax.set(ylabel='Weekday-balanced daily MAE',title='School-term and break comparison - weekdays excluding public holidays');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(out/'school_term_break.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',default=str(ROOT/'data_events/lane_series.csv'));parser.add_argument('--calendar',default=str(ROOT/'event_calendar_2025.json'))
    parser.add_argument('--lane',default='100:2');parser.add_argument('--out',default=str(ROOT/'results/events2025'));parser.add_argument('--skip-lstm',action='store_true');parser.add_argument('--epochs',type=int,default=40);args=parser.parse_args()
    if not Path(args.data).exists():parser.error('Full-year event data missing; run the project event-data acquisition step first')
    if not Path(args.calendar).exists():parser.error('Verified event_calendar_2025.json is required')
    threadpool_limits(1);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);calendar=json.loads(Path(args.calendar).read_text())
    source,d=load_series(args.data,args.lane);start=time.perf_counter();design=prepare_design(d,calendar);preprocessing_seconds=time.perf_counter()-start
    predictions,timing=fit_predict_all(d,design,out,args.skip_lstm,args.epochs);long=long_predictions(d,design,predictions,calendar)
    long.to_csv(out/'predictions.csv',index=False);daily=aggregate_daily(long,source,calendar);daily.to_csv(out/'daily_metrics.csv',index=False)
    overall,event,monthly,seasons,school,demand=make_outputs(source,d,long,daily,calendar,out);plots(long,daily,monthly,school,out)
    quality=source.assign(date=source.local_time.dt.strftime('%Y-%m-%d')).groupby('date').Volume.agg(['size','count']).reindex(pd.date_range('2025-01-01','2025-12-31').strftime('%Y-%m-%d')).rename_axis('date').reset_index();quality[['size','count']]=quality[['size','count']].fillna(0);quality['expected_intervals']=quality.date.map(expected_intervals);quality['complete_source_day']=quality['count']==quality.expected_intervals;quality.to_csv(out/'source_daily_coverage.csv',index=False)
    summary=dict(completed_on='2026-11-19',lane=args.lane,year=2025,source_rows=len(source),usable_rows=len(d),masked_rows=int(source.Volume.isna().sum()),source_missing_dates=quality.loc[quality['size']==0,'date'].tolist(),source_incomplete_dates=quality.loc[~quality.complete_source_day,'date'].tolist(),
        train_end_local='2025-02-25 23:45',validation_start_local='2025-02-26 00:00',validation_end_local='2025-03-11 23:45',test_start_local='2025-03-12 00:00',test_end_local='2025-12-31 23:45',
        train_rows=design['nfit'],validation_rows=design['nval']-design['nfit'],common_test_targets=int(design['a']['test'].sum()),preprocessing_seconds_shared=preprocessing_seconds,
        overall_metrics=overall,timing=timing,christmas_rows=daily[daily.date=='2025-12-25'].replace({np.nan:None}).to_dict('records'),
        complete_test_days_by_model=daily[(daily.split=='test')&daily.complete_24h].groupby('model').size().to_dict(),
        public_holidays_with_complete_test_days=sorted(daily[(daily.split=='test')&daily.complete_24h&daily.is_public_holiday].date.unique().tolist()),
        flagged_demand_days=int(demand.demand_screen_flag.sum()),source_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),calendar_sha256=hashlib.sha256(Path(args.calendar).read_bytes()).hexdigest(),
        hgb_parameters=dict(max_iter=100,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,early_stopping=False,seed=42),
        notes=['One Melbourne detector and one year; exploratory geographic/seasonal stress test, not external-city generalisation.',
            'Daily averages aggregate rolling 15-minute forecasts; these are not 24-hour-ahead forecasts.',
            'Calendar-day grouping converts interval start from fixed AEST source to Australia/Melbourne; DST days can contain92/100 intervals.',
            'Weather windows and holidays are context/exposure indicators, not ground-truth incident labels or proof of causal weather effects.',
            'Cyclone robustness is unassessed because documented Alfred impacts were in Queensland/NSW, not this Melbourne site.',
            'The rolling model uses prior56days and refits every28days; its error-screen threshold is the frozen-calendar validation proxy.',
            'Full-day event comparisons use only complete predicted local days; partial-day coverage remains in daily_metrics.csv.',
            'Summer test coverage is December only; January/February are training/validation data. School periods have unequal lengths, so weekdays are balanced and public holidays excluded in the controlled view.',
            'Model timing is serial CPU fitting; LSTM includes validation-based early stopping, rolling fit time is the sum of all refits; training duration is cost, not a predictive feature.'],
        platform=platform.platform())
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    print(pd.DataFrame(overall).to_string(index=False));print('Complete Christmas target counts:',daily[daily.date=='2025-12-25'][['model','n','expected_intervals','complete_24h']].to_dict('records'));print('Saved',out,flush=True)

if __name__=='__main__':main()
