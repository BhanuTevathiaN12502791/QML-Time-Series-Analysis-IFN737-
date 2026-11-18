"""Aggregate per-site results, paired site-cluster intervals, curves and error groups."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import precision_recall_curve
from core import PRIMARY,ALL_IF,BASELINES,read_data
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
VALUE=['precision','recall','f1','false_positive_rate','false_alerts_per_observed_day','average_precision','fit_seconds','score_seconds','serialized_model_bytes']
PAIRS=[('context_full','if_context_full','if_raw_full'),('context_pca2','if_context_pca2','if_raw_pca2'),
       ('pca2_raw','if_raw_pca2','if_raw_full'),('pca2_context','if_context_pca2','if_context_full'),
       ('pca4_raw','if_raw_pca4','if_raw_full'),('pca4_context','if_context_pca4','if_context_full'),
       ('pca6_raw','if_raw_pca6','if_raw_full'),('pca6_context','if_context_pca6','if_context_full'),
       ('context_mad','time_of_week_mad','global_mad')]

def paired(site_values,pairs=PAIRS,category='primary'):
    rng=np.random.default_rng(2026);rows=[]
    for label,g in site_values.groupby('label_set'):
        pivot=g.pivot(index='site',columns='model',values='f1')
        for name,a,b in pairs:
            if a not in pivot or b not in pivot:continue
            delta=(pivot[a]-pivot[b]).dropna();v=delta.to_numpy();boot=v[rng.integers(len(v),size=(1000,len(v)))].mean(axis=1)
            grouped=delta.groupby(delta.index.str.split('-').str[0]).mean().to_numpy()
            cb=grouped[rng.integers(len(grouped),size=(1000,len(grouped)))].mean(axis=1)
            rows.append(dict(category=category,label_set=label,comparison=name,method_a=a,method_b=b,n_sites=len(v),
                mean_f1_difference=float(v.mean()),ci95_low=float(np.percentile(boot,2.5)),ci95_high=float(np.percentile(boot,97.5)),
                fraction_sites_positive=float((v>0).mean()),prefix_cluster_n=len(grouped),
                prefix_cluster_equal_weight_mean=float(grouped.mean()),prefix_ci95_low=float(np.percentile(cb,2.5)),prefix_ci95_high=float(np.percentile(cb,97.5))))
    return pd.DataFrame(rows)

def aggregate():
    df=pd.read_csv(OUT/'evaluation_metrics.csv');fold=df.groupby(['role','site','fold','model','label_set'],as_index=False)[VALUE].mean()
    site=fold.groupby(['role','site','model','label_set'],as_index=False)[VALUE].mean();site.to_csv(OUT/'site_metrics.csv',index=False)
    summary=site.groupby(['role','model','label_set'],as_index=False)[VALUE].mean()
    ns=site.groupby(['role','model','label_set']).site.nunique().rename('n_sites').reset_index();summary=summary.merge(ns)
    summary.to_csv(OUT/'summary.csv',index=False)
    final=site[site.role=='final'];paired(final).to_csv(OUT/'paired_effects.csv',index=False)
    df[df.role=='final'].groupby(['model','label_set','seed'],as_index=False)[['f1','precision','recall']].mean().to_csv(OUT/'seed_sensitivity.csv',index=False)
    df[df.role=='final'].groupby(['fold','model','label_set'],as_index=False)[VALUE].mean().to_csv(OUT/'fold_metrics.csv',index=False)
    normal=pd.read_csv(OUT/'normal_only_metrics.csv');primary=df[(df.role=='final')&(df.seed.isin([0,42]))]
    ns=normal.groupby(['site','model','label_set'],as_index=False)[VALUE].mean();ps=primary.groupby(['site','model','label_set'],as_index=False)[VALUE].mean()
    both=ns.merge(ps,on=['site','model','label_set'],suffixes=('_normal','_primary'))
    diff=[]
    rng=np.random.default_rng(2026)
    for (kind,label),g in both.groupby(['model','label_set']):
        v=(g.f1_normal-g.f1_primary).to_numpy();boot=v[rng.integers(len(v),size=(1000,len(v)))].mean(axis=1)
        diff.append(dict(model=kind,label_set=label,n_sites=len(g),primary_f1=g.f1_primary.mean(),normal_only_f1=g.f1_normal.mean(),
            difference=v.mean(),ci95_low=np.percentile(boot,2.5),ci95_high=np.percentile(boot,97.5)))
    pd.DataFrame(diff).to_csv(OUT/'normal_only_sensitivity.csv',index=False)
    return df,site,summary

def error_and_curves():
    p=json.loads((ROOT/'protocol.json').read_text());final=[r['site'] for r in p['sites'] if r['role']=='final'];errors=[];curve_rows=[];episode_rows=[];grid=np.linspace(0,1,51)
    for number,site in enumerate(final,1):
        d=read_data(ROOT/'data'/f'{site}.csv');dif=d.Date.diff().dt.total_seconds().to_numpy();gaps=np.flatnonzero(np.r_[False,dif[1:]!=900])
        for fold,fr in enumerate(p['folds']):
            z=np.load(OUT/'evaluation'/f'{site}_fold{fold}.npz');idx=z['indices'];y=z['label1'].astype(bool);y2=z['label2'].astype(bool)
            volume=d.Volume.to_numpy()[idx];train=d.Volume.to_numpy()[:int(len(d)*fr[0])];low,high=np.quantile(train,[.1,.9])
            lastgap=np.searchsorted(gaps,idx,side='right')-1;near=np.zeros(len(idx),bool);ok=lastgap>=0;near[ok]=(idx[ok]-gaps[lastgap[ok]])<=15
            groups={'all':np.ones(len(idx),bool),'zero_count':volume==0,'low_count':volume<=low,'high_count':volume>=high,
                    'night_00_06':d.Date.dt.hour.to_numpy()[idx]<6,'annotation_disagreement':y!=y2,'near_gap_first_16_rows':near}
            timestamps=d.Date.to_numpy()[idx];episode_start=y & np.r_[True,(~y[:-1]) | (np.diff(timestamps)!=np.timedelta64(15,'m'))]
            starts=np.flatnonzero(episode_start)
            for key in z.files:
                if '__' not in key or key.startswith('threshold__'):continue
                kind,seed=key.rsplit('__',1)
                if kind not in ALL_IF+BASELINES:continue
                s=z[key]
                for label,target in [('label_set_1',y),('label_set_2',y2)]:
                    if target.any():
                        pr,re,_=precision_recall_curve(target,s);values=np.interp(grid,re[::-1],pr[::-1])
                        curve_rows.extend(dict(site=site,fold=fold,model=kind,seed=int(seed),label_set=label,recall=float(r),precision=float(v)) for r,v in zip(grid,values))
                if (kind in ALL_IF and int(seed)!=42):continue
                alerts=s>=float(z['threshold__'+key])
                for group,mask in groups.items():
                    tp=int((alerts&y&mask).sum());fp=int((alerts&~y&mask).sum());fn=int((~alerts&y&mask).sum())
                    errors.append(dict(site=site,fold=fold,model=kind,group=group,n=int(mask.sum()),tp=tp,fp=fp,fn=fn,alerts=tp+fp))
                # Episode descriptors do not change any pointwise metric.
                detected=0;delays=[]
                for start in starts:
                    end=start+1
                    while end<len(y) and y[end] and timestamps[end]-timestamps[end-1]==np.timedelta64(15,'m'):end+=1
                    hits=np.flatnonzero(alerts[start:end])
                    if len(hits):detected+=1;delays.append(int(hits[0])*15)
                episode_rows.append(dict(site=site,fold=fold,model=kind,eligible_episodes=len(starts),detected_episodes=detected,
                    median_detected_delay_minutes=float(np.median(delays)) if delays else np.nan))
        if number%20==0:print('Analysis:',number,'/',len(final),'sites',flush=True)
    e=pd.DataFrame(errors);e.to_csv(OUT/'error_groups_by_site.csv',index=False)
    g=e.groupby(['model','group'],as_index=False)[['n','tp','fp','fn','alerts']].sum()
    g['precision']=g.tp/(g.tp+g.fp).replace(0,np.nan);g['recall']=g.tp/(g.tp+g.fn).replace(0,np.nan);g.to_csv(OUT/'error_groups.csv',index=False)
    curves=pd.DataFrame(curve_rows)
    # First average folds/seeds within each site, then sites; no pooled cross-site score thresholds.
    curves=curves.groupby(['site','model','label_set','recall'],as_index=False).precision.mean()
    curves.groupby(['model','label_set','recall'],as_index=False).precision.mean().to_csv(OUT/'precision_recall_curves.csv',index=False)
    pd.DataFrame(episode_rows).to_csv(OUT/'episode_descriptors.csv',index=False)

def main():
    aggregate();error_and_curves()
    print(pd.read_csv(OUT/'summary.csv').query("role=='final' and label_set=='label_set_1'")[['model','f1','precision','recall','false_alerts_per_observed_day']].to_string(index=False))
if __name__=='__main__':main()
