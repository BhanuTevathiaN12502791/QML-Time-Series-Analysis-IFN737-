"""Score observed eight-reading windows. This detects after an interval is observed."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from core import read_data,windows,slot_numbers,BASELINES
ROOT=Path(__file__).resolve().parent

def detect(model_path,history,site=None):
    model=joblib.load(model_path)
    known={r['site'] for r in json.loads((ROOT/'protocol.json').read_text())['sites']}
    inferred=Path(history).stem if Path(history).stem in known else None
    if any(s is not None and s!=model['site'] for s in [site,inferred]):
        raise ValueError(f"Section mismatch: model is fitted to {model['site']}; train a separate model for the supplied section")
    d=read_data(history,labels=False);w,ends=windows(d,model['window'])
    if not len(ends):raise ValueError('No contiguous eight-reading windows')
    if 'context' in model['model_kind']:
        residual=model['context'].transform(d.Volume.to_numpy(),slot_numbers(d))
        w=sliding_window_view(residual,8)[ends-7].copy()
    scores=model['model'].score(w,slot_numbers(d)[ends]) if model['model_kind'] in BASELINES else model['model'].score(w)
    partition=np.where(d.Date.iloc[ends]<=pd.Timestamp(model['fit_last_timestamp']),'fit',np.where(d.Date.iloc[ends]<=pd.Timestamp(model['validation_last_timestamp']),'validation','after_calibration'))
    return pd.DataFrame(dict(site=model['site'],partition=partition,timestamp=d.Date.iloc[ends].to_numpy(),volume=d.Volume.iloc[ends].to_numpy(),
        anomaly_score=scores,threshold=model['threshold'],is_anomaly=scores>=model['threshold']))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',default=str(ROOT/'models/0185-E/detector.joblib'))
    p.add_argument('--history',default=str(ROOT/'data/0185-E.csv'));p.add_argument('--site',help='Assert the section ID when using a generic history filename');p.add_argument('--out');a=p.parse_args()
    result=detect(a.model,a.history,a.site)
    if result.timestamp.iloc[-1]!=read_data(a.history,labels=False).Date.iloc[-1]:raise ValueError('Latest readings cross a gap; no current detection can be made')
    if a.out:result.to_csv(a.out,index=False);print(f'Saved {len(result)} observed-window scores to {a.out}')
    print(result.tail(1).to_json(orient='records',date_format='iso',indent=2))
if __name__=='__main__':main()
