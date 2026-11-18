"""Measure error, training time, RAM and detection metrics versus calendar training days."""
import argparse,hashlib,json,os,platform,random,subprocess,sys,threading,time
from pathlib import Path
import numpy as np,pandas as pd,psutil
from numpy.lib.stride_tricks import sliding_window_view
from threadpoolctl import threadpool_limits
from core import read_data,windows,slot_numbers,Context,Detector,ObservedBaseline,threshold,metrics
ROOT=Path(__file__).resolve().parent
MODELS=['lstm_forecast','if_context_full','if_context_pca2','sitdd_adapted']

class RAMMonitor:
    """Sample process RSS during one isolated trial; units are MiB, not Python allocations."""
    def __init__(self):
        self.process=psutil.Process();self.baseline=self.process.memory_info().rss;self.peak=self.baseline
        self.stop=threading.Event();self.thread=threading.Thread(target=self.sample,daemon=True)
    def sample(self):
        while not self.stop.wait(.002):self.snapshot()
    def snapshot(self):
        self.peak=max(self.peak,self.process.memory_info().rss);return self.peak/1024**2
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.snapshot();self.stop.set();self.thread.join()

def masks_for_days(d,days,width=8):
    if days<=0:raise ValueError('Training days must be positive')
    nf,nv=int(len(d)*.7),int(len(d)*.85);end=d.Date.iloc[nf-1];start=end-pd.Timedelta(days=days)
    if start<d.Date.iloc[0]-pd.Timedelta(minutes=15):raise ValueError('Requested history exceeds available training chronology')
    w,ends=windows(d,width)
    rows=((d.Date>start)&(d.Date<=end)).to_numpy()
    fit=(ends<nf)&(d.Date.to_numpy()[ends-width+1]>start.to_datetime64())
    val=(ends>=nf)&(ends<nv);test=ends>=nv
    if min(fit.sum(),val.sum(),test.sum())<8:raise ValueError('Insufficient gap-valid windows')
    return dict(nfit=nf,nval=nv,start=start,end=end,rows=rows,w=w,ends=ends,fit=fit,val=val,test=test)

def run_detector(d,days,kind,seed,selected,on_training_complete):
    start=time.perf_counter();a=masks_for_days(d,days);slots=slot_numbers(d);x=d.Volume.to_numpy();ctx=Context().fit(x[a['rows']],slots[a['rows']])
    if kind.startswith('if_context'):
        residual=ctx.transform(x,slots);w=sliding_window_view(residual,8)[a['ends']-7].copy()
        model=Detector(kind,selected['max_samples'],seed,selected['n_estimators']).fit(w[a['fit']])
        score=lambda mask:model.score(w[mask])
    else:
        w=a['w'];model=ObservedBaseline(kind,ctx,selected['sitdd_k']).fit(w[a['fit']])
        score=lambda mask:model.score(w[mask],slots[a['ends'][mask]])
    validation_score=score(a['val']);y=d.label_set_1.to_numpy()[a['ends']]
    operating_threshold=threshold(validation_score,y[a['val']]);training_seconds=time.perf_counter()-start
    on_training_complete();start=time.perf_counter();test_score=score(a['test']);inference_seconds=time.perf_counter()-start
    result=dict(training_seconds=training_seconds,inference_seconds=inference_seconds,train_rows=int(a['rows'].sum()),train_windows=int(a['fit'].sum()),
        validation_windows=int(a['val'].sum()),test_windows=int(a['test'].sum()),epochs=0,threshold=operating_threshold)
    for label in [1,2]:
        target=d[f'label_set_{label}'].to_numpy()[a['ends'][a['test']]]
        result.update({f'{k}_set{label}':v for k,v in metrics(target,test_score,operating_threshold).items()})
    result.update(test_indices=a['ends'][a['test']],test_actual=x[a['ends'][a['test']]],test_prediction=test_score)
    return result

def code_hash():
    names=['core.py','learning_curves.py','learning_forecast.py','selected_parameters.json']
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}

def worker(args):
    threadpool_limits(1);config=json.loads(Path(args.config).read_text());d=read_data(ROOT/'data'/f'{config["site"]}.csv')
    if args.model=='lstm_forecast':
        from learning_forecast import configure_torch,run_forecast
        configure_torch(args.seed)  # Library import cost is excluded from training wall time.
    selected=json.loads((ROOT/'selected_parameters.json').read_text());peak_training=[]
    with RAMMonitor() as monitor:
        callback=lambda:peak_training.append(monitor.snapshot())
        pipeline_start=time.perf_counter()
        if args.model=='lstm_forecast':
            nf,nv=int(len(d)*.7),int(len(d)*.85);start=d.Date.iloc[nf-1]-pd.Timedelta(days=args.days)
            result=run_forecast(d,start,nf,nv,args.seed,max_epochs=config['forecast']['max_epochs'],patience=config['forecast']['patience'],on_training_complete=callback)
        else:result=run_detector(d,args.days,args.model,args.seed,selected,callback)
        total=time.perf_counter()-pipeline_start
    arrays={k:result.pop(k) for k in ['test_indices','test_actual','test_prediction']}
    history=result.pop('history',[])
    if args.model=='lstm_forecast':
        result.update(train_rows=result['training_observations'],train_windows=result['training_windows'],epochs=result['epochs_trained'])
    result['observed_training_days']=result['train_rows']/96
    result.update(model=args.model,training_days=args.days,seed=args.seed,site=config['site'],baseline_rss_mib=monitor.baseline/1024**2,
        training_peak_rss_mib=peak_training[-1],training_extra_rss_mib=max(0,peak_training[-1]-monitor.baseline/1024**2),
        pipeline_peak_rss_mib=monitor.peak/1024**2,pipeline_seconds=total,config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest())
    prefix=Path(args.out)/'trials'/f'{args.model}_{args.days}d_seed{args.seed}';prefix.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(prefix.with_suffix('.npz'),**arrays)
    if history:pd.DataFrame(history).to_csv(prefix.with_name(prefix.name+'_history.csv'),index=False)
    prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    print(json.dumps({k:v for k,v in result.items() if k in ['model','training_days','seed','training_seconds','mae','training_peak_rss_mib']}),flush=True)

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);d=read_data(ROOT/'data'/f'{args.site}.csv');nf,nv=int(len(d)*.7),int(len(d)*.85)
    config=dict(created_on='2026-11-19',site=args.site,days=args.days,seeds=[17,42,91],models=MODELS,
        source_sha256=hashlib.sha256((ROOT/'data'/f'{args.site}.csv').read_bytes()).hexdigest(),code_sha256=code_hash(),
        training_end=str(d.Date.iloc[nf-1]),validation_first=str(d.Date.iloc[nf]),validation_last=str(d.Date.iloc[nv-1]),test_first=str(d.Date.iloc[nv]),test_last=str(d.Date.iloc[-1]),
        definition='T is a trailing calendar-day lookback ending at the fixed 70% raw-row boundary. All inputs and targets of each training window must fall in that lookback. No imputation.',
        timing='Training wall seconds include feature/context/scaling preparation, model fitting and validation-based stopping/calibration. Exclude library imports, final test inference and artifact I/O.',
        memory='2ms sampled process RSS during preprocessing/training/calibration, baseline after data/framework load; absolute peak includes libraries, incremental peak is peak minus baseline. Each trial is a fresh serial CPU process.',
        forecast=dict(window=20,hidden_units=32,max_epochs=args.epochs,patience=7,batch_size=64,optimizer='Adam',learning_rate=.001),
        evaluation='Fixed gap-valid validation/test targets for every T within each task. Forecast task uses 20 past readings; detection task uses current + seven prior readings. Tasks have different eligible targets and metrics.',
        selected_parameters=json.loads((ROOT/'selected_parameters.json').read_text()),machine=platform.platform(),logical_cpus=os.cpu_count(),physical_ram_bytes=psutil.virtual_memory().total,
        interpretation='Exploratory single development-section learning/resource curves. Original final-site research protocol and results are unchanged. Do not choose an optimal T from the test curves without separate validation.')
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Output belongs to a different curve protocol; choose a new output folder')
    path.write_text(json.dumps(config,indent=2));jobs=[(model,day,seed) for model in MODELS for day in args.days for seed in config['seeds']]
    random.Random(2026).shuffle(jobs)
    (out/'execution_order.json').write_text(json.dumps(jobs,indent=2))
    digest=hashlib.sha256(path.read_bytes()).hexdigest();done=0
    for model,day,seed in jobs:
        existing=out/'trials'/f'{model}_{day}d_seed{seed}.json'
        if existing.exists():
            if json.loads(existing.read_text())['config_sha256']!=digest:raise ValueError('Trial fingerprint mismatch')
            done+=1;continue
        command=[sys.executable,__file__,'--worker','--model',model,'--days',str(day),'--seed',str(seed),'--config',str(path),'--out',str(out)]
        result=subprocess.run(command,check=True,capture_output=True,text=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPLBACKEND':'Agg'})
        done+=1;print(f'{done}/{len(jobs)} '+result.stdout.strip(),flush=True)
    rows=[json.loads(p.read_text()) for p in sorted((out/'trials').glob('*.json'))]
    pd.DataFrame(rows).sort_values(['model','training_days','seed']).to_csv(out/'runs.csv',index=False)
    print('Completed learning curves:',out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--model',choices=MODELS);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--site',default='0185-E');p.add_argument('--days',nargs='+',type=int,default=[7,14,28,56,84,112,140]);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--config');p.add_argument('--out',default=str(ROOT/'results/learning_curves'))
    a=p.parse_args()
    if a.worker:
        a.days=a.days[0];worker(a)
    else:run(a)
if __name__=='__main__':main()
