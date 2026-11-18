"""Verify measured learning curves and export MAE, time, RAM and detection figures."""
import argparse,hashlib,json,platform,shutil
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from core import read_data,metrics
from learning_curves import MODELS,masks_for_days
from learning_forecast import forecast_arrays
ROOT=Path(__file__).resolve().parent
LABEL={'lstm_forecast':'LSTM forecast (companion task)','if_context_full':'Contextual IF · full','if_context_pca2':'Contextual IF · PCA 2','sitdd_adapted':'Adapted Si-TDD'}
COLOUR={'lstm_forecast':'#b84e27','if_context_full':'#176882','if_context_pca2':'#697a31','sitdd_adapted':'#7e5aa6'}

def verify(folder):
    config=json.loads((folder/'protocol.json').read_text());d=read_data(ROOT/'data'/f'{config["site"]}.csv');rows=pd.read_csv(folder/'runs.csv');checks={}
    expected={(m,t,s) for m in config['models'] for t in config['days'] for s in config['seeds']}
    checks['complete_trial_grid']=set(zip(rows.model,rows.training_days,rows.seed))==expected and len(rows)==len(expected)
    checks['source_hash']=hashlib.sha256((ROOT/'data'/f'{config["site"]}.csv').read_bytes()).hexdigest()==config['source_sha256']
    checks['frozen_fitting_code']=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in config['code_sha256'].items())
    checks['trial_fingerprints_match']=rows.config_sha256.eq(hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest()).all()
    reference={};checks['all_values_finite']=all(np.isfinite(rows[k]).all() for k in ['training_seconds','training_peak_rss_mib','training_extra_rss_mib','train_rows','train_windows'])
    checks['memory_ordering']=bool((rows.training_peak_rss_mib>=rows.baseline_rss_mib).all() and (rows.pipeline_peak_rss_mib>=rows.training_peak_rss_mib).all())
    checks['time_ordering']=bool((rows.training_seconds>0).all() and (rows.pipeline_seconds>=rows.training_seconds).all())
    counts=[];scores_good=True;split_good=True;identical=True
    for row in rows.itertuples():
        stem=f'{row.model}_{row.training_days}d_seed{row.seed}'
        with np.load(folder/'trials'/f'{stem}.npz') as z:
            if row.model=='lstm_forecast':
                nfit,nval=int(len(d)*.7),int(len(d)*.85);start=d.Date.iloc[nfit-1]-pd.Timedelta(days=row.training_days)
                a=forecast_arrays(d,start,nfit,nval);idx=a['indices'][a['test']];actual=d.Volume.to_numpy()[idx];pred=z['test_prediction']
                calc=dict(mae=mean_absolute_error(actual,pred),rmse=np.sqrt(mean_squared_error(actual,pred)),r2=r2_score(actual,pred),
                          wape_pct=100*abs(actual-pred).sum()/actual.sum(),persistence_mae=np.mean(abs(actual-a['persistence'][a['test']])))
                scores_good &= all(np.isclose(getattr(row,k),v,atol=1e-9,rtol=1e-9) for k,v in calc.items())
                split_good &= int(row.train_rows)==a['training_observations'] and int(row.train_windows)==a['training_windows']
            else:
                a=masks_for_days(d,row.training_days);idx=a['ends'][a['test']];actual=d.Volume.to_numpy()[idx]
                for label in [1,2]:
                    calc=metrics(d[f'label_set_{label}'].to_numpy()[idx],z['test_prediction'],row.threshold)
                    scores_good &= all(np.isclose(getattr(row,f'{k}_set{label}'),v,atol=1e-9,rtol=1e-9) for k,v in calc.items())
                split_good &= int(row.train_rows)==int(a['rows'].sum()) and int(row.train_windows)==int(a['fit'].sum())
            scores_good &= np.array_equal(z['test_indices'],idx) and np.array_equal(z['test_actual'],actual)
            key='forecast' if row.model=='lstm_forecast' else 'detection'
            if key not in reference:reference[key]=idx.copy()
            identical &= np.array_equal(reference[key],idx)
    checks['saved_predictions_reproduce_metrics']=bool(scores_good);checks['training_coverage_matches_calendar_days']=bool(split_good);checks['fixed_test_targets_within_each_task']=bool(identical)
    result=dict(checked_on='2026-11-19',trials=len(rows),checks={k:bool(v) for k,v in checks.items()},passed=all(checks.values()))
    (folder/'verification.json').write_text(json.dumps(result,indent=2));assert result['passed'],result
    return rows,config,result

def curve(ax,rows,model,metric,**kwargs):
    grouped=rows[rows.model==model].groupby('training_days')[metric].agg(['mean','min','max']).sort_index()
    x=grouped.index.to_numpy();line=ax.plot(x,grouped['mean'].to_numpy(),marker='o',markersize=4,label=LABEL[model],color=COLOUR[model],**kwargs)
    ax.fill_between(x,grouped['min'].to_numpy(),grouped['max'].to_numpy(),color=COLOUR[model],alpha=.13)

def style(ax,days):
    ax.set_xlabel('Training history T (calendar days)');ax.set_xticks(days);ax.grid(alpha=.2)

def save(fig,folder,name):
    fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(folder/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    svg=folder/(name+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')

def figures(rows,config,folder):
    days=config['days'];forecast=rows[rows.model=='lstm_forecast'];persistence=forecast.persistence_mae.iloc[0]
    fig,axes=plt.subplots(3,1,figsize=(11,12))
    curve(axes[0],rows,'lstm_forecast','mae');axes[0].axhline(persistence,linestyle='--',color='#5c6872',label=f'Repeat latest count · MAE {persistence:.2f}')
    axes[0].set_ylabel('Forecast MAE (volume counts)');axes[0].set_title('Forecast error versus training history');axes[0].legend(fontsize=9)
    for model in MODELS:curve(axes[1],rows,model,'training_seconds')
    axes[1].set_yscale('log');axes[1].set_ylabel('Training wall time (seconds, log scale)');axes[1].set_title('Preprocessing + fitting + validation stopping/calibration');axes[1].legend(fontsize=9,ncol=2)
    for model in MODELS:curve(axes[2],rows,model,'training_peak_rss_mib')
    axes[2].set_ylabel('Sampled peak process RAM (MiB)');axes[2].set_title('Peak RAM during training phase · includes loaded libraries');axes[2].legend(fontsize=9,ncol=2)
    for ax in axes:style(ax,days)
    fig.suptitle(f'History-size learning curves · AnoLT {config["site"]}\nFixed validation/test periods · mean and min–max across three seeds',fontsize=14,y=1.015)
    save(fig,folder,'mae_time_ram_vs_days')
    fig,ax=plt.subplots(figsize=(9,5));curve(ax,rows,'lstm_forecast','mae');ax.axhline(persistence,linestyle='--',color='#5c6872',label='Persistence reference');ax.set(ylabel='Forecast MAE (volume counts)',title='MAE versus training history · companion next-interval LSTM');style(ax,days);ax.legend();save(fig,folder,'mae_vs_days')
    fig,ax=plt.subplots(figsize=(10,5))
    for model in MODELS:curve(ax,rows,model,'training_seconds')
    ax.set_yscale('log');ax.set(ylabel='Training wall time (seconds, log scale)',title='Training time versus history · fixed procedures');style(ax,days);ax.legend(fontsize=9);save(fig,folder,'training_time_vs_days')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for model in MODELS:
        curve(axes[0],rows,model,'training_peak_rss_mib');curve(axes[1],rows,model,'training_extra_rss_mib')
    axes[0].set(ylabel='Peak process RAM (MiB)',title='Absolute sampled training peak');axes[1].set(ylabel='Peak minus loaded baseline (MiB)',title='Additional RAM above baseline')
    for ax in axes:style(ax,days);ax.legend(fontsize=8)
    save(fig,folder,'ram_vs_days')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    fields=[('f1_set1','F1'),('precision_set1','Precision'),('recall_set1','Recall'),('false_alerts_per_observed_day_set1','False alerts / equivalent observed day'),('average_precision_set1','Average precision'),('false_positive_rate_set1','False-positive rate')]
    for ax,(field,label) in zip(axes.flat,fields):
        for model in MODELS[1:]:curve(ax,rows,model,field)
        ax.set_ylabel(label);style(ax,days)
    axes[0,0].legend(fontsize=8);fig.suptitle('Anomaly detection metrics · inspector label set 1',fontsize=14,y=1.02);save(fig,folder,'detection_metrics_vs_days')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    fields=[('mae','MAE (volume counts)'),('rmse','RMSE (volume counts)'),('wape_pct','WAPE (%)'),('r2','R²'),('mae_skill_vs_persistence','MAE skill versus persistence'),('epochs','Epochs executed')]
    for ax,(field,label) in zip(axes.flat,fields):curve(ax,rows,'lstm_forecast',field);ax.set_ylabel(label);style(ax,days)
    axes[1,1].axhline(0,color='#5c6872',linestyle='--');fig.suptitle('Companion forecasting metrics · same held-out targets at every T',fontsize=14,y=1.02);save(fig,folder,'forecast_metrics_vs_days')

def docs(rows,config,folder):
    scalar=[c for c in rows.select_dtypes('number').columns if c not in ['seed','training_days']]
    summary=rows.groupby(['model','training_days'])[scalar].agg(['mean','min','max']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(folder/'summary.csv',index=False)
    coverage=rows[rows.model=='if_context_full'].groupby('training_days')[['train_rows','observed_training_days','train_windows','validation_windows','test_windows']].first().reset_index();coverage['forecast_train_windows']=rows[rows.model=='lstm_forecast'].groupby('training_days').train_windows.first().reindex(coverage.training_days).to_numpy();coverage.to_csv(folder/'coverage.csv',index=False)
    compact=rows[rows.model=='lstm_forecast'].groupby('training_days')[['mae','rmse','training_seconds','training_peak_rss_mib','training_extra_rss_mib','epochs']].mean().reset_index()
    header='| '+' | '.join(compact.columns)+' |\n|'+'|'.join(['---']*len(compact.columns))+'|\n'
    table=header+'\n'.join('| '+' | '.join(f'{value:.3f}' for value in row)+' |' for row in compact.to_numpy())
    text=f'''# MAE, training time and RAM versus training history

All three requested Y-axes are provided separately, with **T (calendar training days)** on X. MAE measures next-interval prediction error in original volume-count units. Training time is wall-clock seconds. RAM is process resident memory in MiB (1 MiB = 1,048,576 bytes).

![MAE, training time and RAM](mae_time_ram_vs_days.png)

## Protocol

- Section: **{config['site']}**, an already-used development/pilot section. This is exploratory single-section analysis, not a new 117-site generalisation study.
- T = 7, 14, 28, 56, 84, 112 and 140 trailing calendar days, all ending **{config['training_end']}**.
- Fixed validation: **{config['validation_first']}** through **{config['validation_last']}**; fixed test: **{config['test_first']}** through **{config['test_last']}**.
- Three seeds/repetitions: 17, 42, 91. Lines are means; shading is observed min–max, **not a confidence interval**. Si-TDD is deterministic; its repetitions measure runtime variation.
- Every training input and target lies inside its lookback. Valid zeros remain; gap-crossing windows are excluded. `coverage.csv` records actual observations and equivalent observed days, which differ from calendar T when dates are absent.
- A companion **LSTM(32)** predicts the next volume from 20 preceding readings. Adam 0.001, batch 64, at most {config['forecast']['max_epochs']} epochs, validation-MSE early stopping with patience 7, training-only scaling, nonnegative predictions. Test targets stay fixed across T. Persistence repeats the latest observed count.
- The established contextual IF full/PCA-2 and adapted Si-TDD procedures retain their frozen global settings and validation-only anomaly thresholds. Detection uses the current observation plus seven prior readings. MAE and F1 address separate tasks and are not compared numerically.

## Measurements

Trials run **serially in fresh CPU processes**, in a seeded shuffled order. Training time includes feature/context/scaling preparation, fitting and validation-based early stopping or threshold calibration. It excludes final test inference and file I/O. Initial top-level framework imports occur before timing; lazy optimiser setup inside fitting remains part of the measured pipeline.

RAM is sampled every 2 ms. The training peak is captured before final test prediction. Absolute process RSS includes Python and loaded libraries, so the optional PyTorch forecaster has a larger loaded baseline. The second RAM panel shows additional peak RSS above that baseline. These are sampled peaks, not exact allocation totals; very short transient peaks may be missed. Small timing differences can reflect system load, scheduling and sampling overhead. The CSV inference_seconds field covers validation plus test for the forecaster and test only for detectors; it is diagnostic and is not plotted as a matched latency comparison.

Each procedure is fixed: IF uses 100 trees and max_samples 256, so fitting cost need not grow linearly with T. LSTM epochs vary with validation stopping and are plotted. At seven days the weekday/time-slot profile has only one reference per slot, which can make contextual residuals degenerate; this short-history limitation is retained in the results.

## Companion forecasting means

{table}

## Other metrics and files

- `detection_metrics_vs_days.png`: F1, precision, recall, average precision, false-positive rate and false alerts/day (inspector set 1). Set-2 values are also retained in the CSV.
- `forecast_metrics_vs_days.png`: MAE, RMSE, WAPE, R², persistence skill and epochs.
- `runs.csv`: all measured trials; `summary.csv`: means and observed ranges; `coverage.csv`: data coverage; `verification.json`: source/target/metric checks.
- Every figure has PNG and SVG versions. Detailed predictions and epoch histories are generated locally in `trials/`.

The curves describe performance under these fixed settings. More history need not monotonically improve error. Increasing T also introduces older traffic regimes and seasonal composition, so this is not a pure causal experiment on sample count alone. Do not choose a preferred history size using test curves and then present the same test as untouched validation. The original frozen final-site anomaly results remain unchanged.
'''
    (folder/'README.md').write_text(text)
    style='<style>body{font:16px/1.6 system-ui;margin:30px auto;max-width:1100px;padding:20px;color:#183248}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}pre{overflow:auto}</style>'
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Learning curves</title>'+style+'<h1>MAE, training time and RAM versus T</h1><p>Calendar training history; mean and min–max across three seeds. Companion forecasting MAE and anomaly metrics are separate tasks.</p>'
    for name in ['mae_time_ram_vs_days','ram_vs_days','detection_metrics_vs_days','forecast_metrics_vs_days']:page+=f'<img src="{name}.png" alt="{name.replace("_"," ")}">'
    page+='<h2>Forecasting means</h2>'+compact.to_html(index=False,float_format=lambda x:f'{x:.3f}')+'<p>See <a href="README.md">the protocol and measurement notes</a>, <a href="runs.csv">raw measured runs</a> and <a href="verification.json">verification</a>.</p></html>'
    (folder/'report.html').write_text(page)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',default=str(ROOT/'results/learning_curves'));a=p.parse_args();folder=Path(a.results)
    rows,config,verification=verify(folder);figures(rows,config,folder);docs(rows,config,folder)
    print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
