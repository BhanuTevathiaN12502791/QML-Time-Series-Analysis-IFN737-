"""Frozen development tuning and expanded site/fold evaluation; resumable per site."""
import argparse,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from core import PRIMARY,ALL_IF,BASELINES,Detector,Representation,read_data,prepare_fold,threshold,metrics
from cache import fingerprint,reusable,stamp
ROOT=Path(__file__).resolve().parent

def preserve_frozen_selection(target,selected):
    """Keep frozen choices byte-stable across harmless numeric round-off."""
    target=Path(target)
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(selected,indent=2))
        return selected
    frozen=json.loads(target.read_text())
    choices=['max_samples','sitdd_k','n_estimators','demo_model','protocol_sha256']
    changed=[key for key in choices if frozen.get(key)!=selected.get(key)]
    if changed:raise ValueError('Frozen selection changed: '+', '.join(changed)+'. Use a separate protocol/results copy for a new study.')
    for name in ['max_samples_validation_f1','knn_validation_f1']:
        old,new=frozen.get(name,{}),selected.get(name,{})
        if set(old)!=set(new) or not all(np.isclose(old[key],new[key],rtol=1e-8,atol=1e-10) for key in old):
            raise ValueError('Frozen validation diagnostics changed materially: '+name+'. Investigate before proceeding.')
    # Preserve the original file and its digest; only candidate diagnostics may differ by round-off.
    return frozen

def protocol():
    raw=(ROOT/'protocol.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==(ROOT/'protocol.sha256').read_text().strip(),'Protocol changed'
    return json.loads(raw)

def dev_site(site):
    threadpool_limits(1);binding=fingerprint(site,'development');p=protocol();d=read_data(ROOT/'data'/f'{site}.csv');rows=[]
    for fi,fold in enumerate(p['folds']):
        a=prepare_fold(d,fold);v=a['val'];tr=a['fit'];y=d.label_set_1.to_numpy()[a['ends'][v]]
        for size in p['max_samples_candidates']:
            for kind in PRIMARY:
                w=a['rw'] if 'context' in kind else a['w']
                for seed in p['forest_seeds']:
                    model=Detector(kind,size,seed,p['n_estimators']).fit(w[tr]);s=model.score(w[v]);th=threshold(s,y)
                    rows.append(dict(site=site,fold=fi,model=kind,max_samples=size,seed=seed,**metrics(y,s,th)))
        for k in p['knn_candidates']:
            s=a['context'].knn(d.Volume.to_numpy()[a['ends'][v]],a['slots'][a['ends'][v]],k)
            th=threshold(s,y);rows.append(dict(site=site,fold=fi,model='sitdd_adapted',k=k,seed=0,**metrics(y,s,th)))
    target=ROOT/'results/development'/f'{site}.csv';target.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(target,index=False);stamp(site,'development',expected=binding);return site

def tuning(workers):
    p=protocol();sites=[r['site'] for r in p['sites'] if r['role']=='development']
    todo=[s for s in sites if not reusable(s,'development')]
    execute(dev_site,todo,workers,'development')
    df=pd.concat([pd.read_csv(ROOT/'results/development'/f'{s}.csv') for s in sites],ignore_index=True)
    df.to_csv(ROOT/'results/development_validation.csv',index=False)
    rank=df[df.model.isin(PRIMARY)].groupby('max_samples').f1.mean().sort_values(ascending=False,kind='stable')
    kr=df[df.model=='sitdd_adapted'].groupby('k').f1.mean().sort_values(ascending=False,kind='stable')
    selected=dict(max_samples=int(rank.index[0]),sitdd_k=int(kr.index[0]),n_estimators=p['n_estimators'],
        selection_data='development validation label_set_1 only',protocol_sha256=(ROOT/'protocol.sha256').read_text().strip(),
        max_samples_validation_f1={str(int(k)):float(v) for k,v in rank.items()},
        knn_validation_f1={str(int(k)):float(v) for k,v in kr.items()})
    # One shared hyperparameter choice; isolate representation effects in all later comparisons.
    by=df[(df.model.isin(PRIMARY))&(df.max_samples==selected['max_samples'])].groupby('model').f1.mean()
    selected['demo_model']=str(by.sort_values(ascending=False,kind='stable').index[0])
    target=ROOT/'selected_parameters.json'
    selected=preserve_frozen_selection(target,selected)
    print('Frozen selection:',selected,flush=True)

def evaluate_site(args):
    threadpool_limits(1);site,normal=args;phase='normal_only' if normal else 'evaluation';binding=fingerprint(site,phase);p=protocol();sel=json.loads((ROOT/'selected_parameters.json').read_text())
    d=read_data(ROOT/'data'/f'{site}.csv');rows=[];splitrows=[];path=ROOT/'results'/('normal_only' if normal else 'evaluation');path.mkdir(parents=True,exist_ok=True)
    role=next(r['role'] for r in p['sites'] if r['site']==site)
    for fi,fold in enumerate(p['folds']):
        t0=time.perf_counter();a=prepare_fold(d,fold,normal);prep=time.perf_counter()-t0
        tr,v,ts=a['fit'],a['val'],a['test'];ends=a['ends'];x=d.Volume.to_numpy();val_y=d.label_set_1.to_numpy()[ends[v]]
        cache=dict(indices=ends[ts],timestamps=d.Date.iloc[ends[ts]].astype(str).to_numpy(dtype='U'),
                   volume=x[ends[ts]].astype('float32'),label1=d.label_set_1.to_numpy()[ends[ts]],label2=d.label_set_2.to_numpy()[ends[ts]])
        splits=dict(site=site,role=role,fold=fi,normal_only=normal,fit_end_row=a['nfit'],validation_end_row=a['nval'],test_end_row=a['ntest'],
            fit_windows=int(tr.sum()),validation_windows=int(v.sum()),test_windows=int(ts.sum()),
            test_first=str(d.Date.iloc[ends[ts][0]]),test_last=str(d.Date.iloc[ends[ts][-1]]),preprocess_seconds=prep,
            unseen_test_slots=int((~a['context'].seen[a['slots'][ends[ts]]]).sum()))
        splitrows.append(splits)
        def save(kind,seed,sv,st,fitsec,scoresec,size=0):
            th=threshold(sv,val_y);cache[f'{kind}__{seed}']=np.asarray(st,dtype='float64')
            cache[f'threshold__{kind}__{seed}']=np.array(th)
            validation=metrics(val_y,sv,th)
            for label in ['label_set_1','label_set_2']:
                rows.append(dict(site=site,role=role,fold=fi,normal_only=normal,model=kind,seed=seed,label_set=label,threshold=th,
                    validation_f1_set1=validation['f1'],fit_seconds=fitsec,score_seconds=scoresec,
                    serialized_model_bytes=size,**metrics(d[label].to_numpy()[ends[ts]],st,th)))
        # Comparable simple baselines, all evaluated at the same valid window endpoints.
        ctx=a['context'];t=time.perf_counter();zs=Representation('if_raw_pca2').fit(a['w'][tr]);reconfit=time.perf_counter()-t
        for kind in BASELINES:
            t=time.perf_counter()
            if kind=='global_mad':scores=abs(x[ends]-ctx.median)/ctx.scale
            elif kind=='time_of_week_mad':scores=abs(ctx.transform(x[ends],a['slots'][ends]))
            elif kind=='sitdd_adapted':scores=ctx.knn(x[ends],a['slots'][ends],sel['sitdd_k'])
            else:
                z=zs.scaler.transform(a['w']);reconstructed=zs.pca.inverse_transform(zs.pca.transform(z));scores=np.mean((z-reconstructed)**2,axis=1)
            elapsed=time.perf_counter()-t
            save(kind,0,scores[v],scores[ts],reconfit if kind=='pca_reconstruction' else prep,elapsed)
        for kind in (PRIMARY if normal else ALL_IF):
            w=a['rw'] if 'context' in kind else a['w']
            for seed in ([42] if normal else p['forest_seeds']):
                model=Detector(kind,sel['max_samples'],seed,sel['n_estimators']).fit(w[tr])
                t=time.perf_counter();sv=model.score(w[v]);st=model.score(w[ts]);scoresec=time.perf_counter()-t
                import pickle
                size=len(pickle.dumps(model,protocol=5));save(kind,seed,sv,st,model.fit_seconds,scoresec,size)
        np.savez_compressed(path/f'{site}_fold{fi}.npz',**cache)
    pd.DataFrame(rows).to_csv(path/f'{site}_metrics.csv',index=False)
    pd.DataFrame(splitrows).to_csv(path/f'{site}_splits.csv',index=False)
    stamp(site,phase,expected=binding)
    return site

def execute(fn,todo,workers,tag):
    if not todo:return
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fn,s):s for s in todo}
        for i,future in enumerate(as_completed(futures),1):
            result=future.result()
            if i==1 or i%5==0 or i==len(todo):print(f'{tag}: {i}/{len(todo)} sites completed ({result}), elapsed {(time.perf_counter()-start)/60:.1f} min',flush=True)

def evaluate(workers,normal=False):
    p=protocol();sites=[r['site'] for r in p['sites'] if not normal or r['role']=='final'];folder='normal_only' if normal else 'evaluation'
    todo=[(s,normal) for s in sites if not reusable(s,folder)]
    execute(evaluate_site,todo,workers,folder)
    for suffix in ['metrics','splits']:
        df=pd.concat([pd.read_csv(ROOT/'results'/folder/f'{s}_{suffix}.csv') for s in sites],ignore_index=True)
        df.to_csv(ROOT/'results'/f'{folder}_{suffix}.csv',index=False)
    print(folder,'complete',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['tune','evaluate','normal']);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.phase=='tune':tuning(args.workers)
    else:evaluate(args.workers,args.phase=='normal')
if __name__=='__main__':main()
