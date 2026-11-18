"""Causal single-section volume anomaly detection. Higher score means more unusual."""
from pathlib import Path
import hashlib,time,pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from numpy.lib.stride_tricks import sliding_window_view

PRIMARY=['if_raw_full','if_raw_pca2','if_context_full','if_context_pca2']
ALL_IF=['if_raw_full','if_raw_pca2','if_raw_pca4','if_raw_pca6','if_context_full','if_context_pca2','if_context_pca4','if_context_pca6']
BASELINES=['global_mad','time_of_week_mad','pca_reconstruction','sitdd_adapted']

def read_data(path,labels=True):
    d=pd.read_csv(path)
    if not {'Date','Volume'}<=set(d):raise ValueError('CSV requires Date and Volume')
    if d.empty:raise ValueError('CSV must contain observations')
    d['Date']=pd.to_datetime(d.Date,format='ISO8601') if str(d.Date.iloc[0])[4]=='-' else pd.to_datetime(d.Date,dayfirst=True,format='mixed')
    if d.Date.isna().any() or not d.Date.is_monotonic_increasing or d.Date.duplicated().any():raise ValueError('Timestamps must be ordered, unique and nonmissing')
    if ((d.Date.dt.minute%15!=0)|(d.Date.dt.second!=0)|(d.Date.dt.microsecond!=0)|(d.Date.dt.nanosecond!=0)).any():raise ValueError('Expected quarter-hour timestamps')
    d['Volume']=pd.to_numeric(d.Volume,errors='raise')
    if not np.isfinite(d.Volume).all() or (d.Volume<0).any():raise ValueError('Volumes must be finite and nonnegative; no imputation')
    if labels:
        for col in ['label_set_1','label_set_2']:
            if col not in d or not d[col].isin([0,1]).all():raise ValueError(f'Missing/nonbinary {col}')
    return d

def windows(d,width=8):
    x=d.Volume.to_numpy(float);ends=np.arange(width-1,len(d))
    if len(x)<width:raise ValueError('Need at least eight readings')
    dif=d.Date.diff().dt.total_seconds().to_numpy()[1:]
    gaps=np.r_[0,np.cumsum(dif!=900)]
    ok=(gaps[ends]-gaps[ends-width+1])==0
    return sliding_window_view(x,width)[ok].copy(),ends[ok]

def slot_numbers(d):
    return (d.Date.dt.day_of_week*96+d.Date.dt.hour*4+d.Date.dt.minute//15).to_numpy()

class Context:
    def fit(self,x,slots):
        self.median=float(np.median(x));self.scale=max(float(np.median(abs(x-self.median))),1.)
        g=pd.DataFrame({'x':x,'slot':slots}).groupby('slot').x
        self.centres=np.full(672,self.median);self.scales=np.full(672,self.scale);self.seen=np.zeros(672,bool)
        for s,a in g:
            self.centres[s]=np.median(a);self.scales[s]=max(float(np.median(abs(a-np.median(a)))),1.);self.seen[s]=True
        self.references={int(s):np.asarray(a,float) for s,a in g};self.global_reference=np.asarray(x,float)
        return self
    def transform(self,x,slots):return (x-self.centres[slots])/self.scales[slots]
    def knn(self,x,slots,k):
        out=np.empty(len(x))
        for s in np.unique(slots):
            indices=np.flatnonzero(slots==s);ref=self.references.get(int(s),self.global_reference)
            distance=abs(x[indices,None]-ref[None,:]);rank=min(k,len(ref))-1
            out[indices]=np.partition(distance,rank,axis=1)[:,rank]
        return out

class Representation:
    def __init__(self,kind):self.kind=kind
    def fit(self,w):
        self.scaler=StandardScaler().fit(w);z=self.scaler.transform(w);self.pca=None
        if 'pca' in self.kind:
            n=int(self.kind.rsplit('pca',1)[1]);self.pca=PCA(n_components=n,svd_solver='full').fit(z)
        return self
    def transform(self,w):
        z=self.scaler.transform(w);return self.pca.transform(z) if self.pca is not None else z

class Detector:
    def __init__(self,kind,max_samples=256,seed=42,n_estimators=100):
        if kind not in ALL_IF:raise ValueError(kind)
        self.kind=kind;self.max_samples=max_samples;self.seed=seed;self.n_estimators=n_estimators
    def fit(self,w):
        start=time.perf_counter();self.rep=Representation(self.kind).fit(w)
        self.forest=IsolationForest(n_estimators=self.n_estimators,max_samples=min(self.max_samples,len(w)),
            max_features=1.,contamination='auto',random_state=self.seed,n_jobs=1).fit(self.rep.transform(w))
        self.fit_seconds=time.perf_counter()-start;return self
    def score(self,w):return -self.forest.score_samples(self.rep.transform(w))

def threshold(scores,y):
    scores=np.asarray(scores,float);y=np.asarray(y,bool)
    if not np.isfinite(scores).all():raise ValueError('Nonfinite score')
    if not y.any():return float('inf')
    order=np.argsort(-scores,kind='stable');s=scores[order];tp=np.cumsum(y[order]);n=np.arange(1,len(s)+1)
    idx=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    f=2*tp[idx]/(n[idx]+y.sum());best=idx[np.argmax(f)]
    return float(s[best])

def metrics(y,s,t):
    y=np.asarray(y,bool);p=np.asarray(s)>=t
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    return dict(n=len(y),tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,
        recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        false_positive_rate=fp/(fp+tn) if fp+tn else 0.,false_alerts_per_observed_day=fp/(len(y)/96),
        average_precision=float(average_precision_score(y,s)) if y.any() else 0.)

def prepare_fold(d,fold,normal_only=False):
    if len(fold)!=3 or not 0<fold[0]<fold[1]<fold[2]<=1:raise ValueError('Invalid chronological fold endpoints')
    n=len(d);nf,nv,nt=[int(n*f) for f in fold];w,ends=windows(d)
    slots=slot_numbers(d);fitrows=np.arange(n)<nf
    if normal_only:fitrows&=(d.label_set_1.to_numpy()==0)
    if not fitrows.any():raise ValueError('No usable training observations')
    context=Context().fit(d.Volume.to_numpy()[fitrows],slots[fitrows])
    residual=context.transform(d.Volume.to_numpy(),slots)
    rw=sliding_window_view(residual,8)[ends-7].copy()
    fit=ends<nf;val=(ends>=nf)&(ends<nv);test=(ends>=nv)&(ends<nt)
    if normal_only:
        clean=sliding_window_view(d.label_set_1.to_numpy()==0,8).all(axis=1)[ends-7]
        fit&=clean
    if min(fit.sum(),val.sum(),test.sum())<8:raise ValueError('Insufficient eligible windows')
    return dict(w=w,rw=rw,ends=ends,slots=slots,fit=fit,val=val,test=test,context=context,nfit=nf,nval=nv,ntest=nt)

class ObservedBaseline:
    """Reusable baseline wrapper for the same observed-window inference interface."""
    def __init__(self,kind,context,k=3):
        if kind not in BASELINES:raise ValueError(kind)
        self.kind=kind;self.context=context;self.k=k
    def fit(self,w):
        if self.kind=='pca_reconstruction':self.rep=Representation('if_raw_pca2').fit(w)
        return self
    def score(self,w,slots):
        x=w[:,-1]
        if self.kind=='global_mad':return abs(x-self.context.median)/self.context.scale
        if self.kind=='time_of_week_mad':return abs(self.context.transform(x,slots))
        if self.kind=='sitdd_adapted':return self.context.knn(x,slots,self.k)
        z=self.rep.scaler.transform(w);return np.mean((z-self.rep.pca.inverse_transform(self.rep.pca.transform(z)))**2,axis=1)
