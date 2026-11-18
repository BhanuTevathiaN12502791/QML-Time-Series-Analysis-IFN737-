"""Fit the globally selected procedure to one road section and calibrate its threshold."""
import argparse,json,hashlib
from pathlib import Path
import joblib
from core import read_data,prepare_fold,Detector,threshold,metrics,ObservedBaseline,BASELINES
ROOT=Path(__file__).resolve().parent

def train(data,out,kind=None,fit_fraction=.7,validation_end=.85):
    if not 0<fit_fraction<validation_end<1:raise ValueError('Need 0 < fit fraction < validation end < 1')
    sel=json.loads((ROOT/'selected_parameters.json').read_text());kind=kind or sel['demo_model']
    d=read_data(data);a=prepare_fold(d,[fit_fraction,validation_end,1.]);tr,v,ts=a['fit'],a['val'],a['test']
    w=a['rw'] if 'context' in kind else a['w']
    if kind in BASELINES:
        model=ObservedBaseline(kind,a['context'],sel['sitdd_k']).fit(w[tr])
        score=lambda mask:model.score(w[mask],a['slots'][a['ends'][mask]])
    else:
        model=Detector(kind,sel['max_samples'],42,sel['n_estimators']).fit(w[tr])
        score=lambda mask:model.score(w[mask])
    y=d.label_set_1.to_numpy()[a['ends']];th=threshold(score(v),y[v])
    checkpoint=dict(site=Path(data).stem,model=model,context=a['context'],threshold=th,window=8,
        model_kind=kind,fit_last_timestamp=str(d.Date.iloc[a['nfit']-1]),validation_last_timestamp=str(d.Date.iloc[a['nval']-1]),
        source_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(),protocol_sha256=(ROOT/'protocol.sha256').read_text().strip())
    target=Path(out);target.mkdir(parents=True,exist_ok=True);joblib.dump(checkpoint,target/'detector.joblib',compress=3)
    result=dict(site=checkpoint['site'],model=kind,threshold=th,test_set1=metrics(y[ts],score(ts),th),
                test_set2=metrics(d.label_set_2.to_numpy()[a['ends'][ts]],score(ts),th),
                note='Single-site illustrative model. Default representation uses development validation; explicit --model choices are user specified. Global settings and local thresholds do not use this test result.')
    (target/'metadata.json').write_text(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default=str(ROOT/'data/0185-E.csv'))
    p.add_argument('--out',default=str(ROOT/'models/0185-E'));p.add_argument('--model',default=None)
    p.add_argument('--fit-fraction',type=float,default=.7);p.add_argument('--validation-end',type=float,default=.85)
    a=p.parse_args();print(json.dumps(train(a.data,a.out,a.model,a.fit_fraction,a.validation_end),indent=2))
if __name__=='__main__':main()
