"""Serial isolated-process benchmark for identical single-site fold inputs."""
import argparse,json,time,subprocess,sys,platform,threading
try:
    import resource
except ImportError:
    resource=None
import psutil
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from core import ALL_IF,Detector,prepare_fold,read_data
ROOT=Path(__file__).resolve().parent

class SampledRSS:
    """Fallback where resource is unavailable; sampled RSS is an approximation."""
    def __init__(self):
        self.process=psutil.Process();self.peak=self.process.memory_info().rss;self.stop=threading.Event()
        self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start()
    def sample(self):
        while not self.stop.wait(.005):self.peak=max(self.peak,self.process.memory_info().rss)
    def finish(self):
        self.peak=max(self.peak,self.process.memory_info().rss);self.stop.set();self.thread.join()
        return self.peak/1024**2

def one(kind,seed):
    sampler=SampledRSS() if resource is None else None
    threadpool_limits(1);sel=json.loads((ROOT/'selected_parameters.json').read_text());d=read_data(ROOT/'data/0185-E.csv')
    start=time.perf_counter();a=prepare_fold(d,[.7,.85,1.]);pre=time.perf_counter()-start
    w=a['rw'] if 'context' in kind else a['w']
    before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else sampler.peak
    model=Detector(kind,sel['max_samples'],seed,sel['n_estimators']).fit(w[a['fit']]);start=time.perf_counter();model.score(w[a['test']]);scoring=time.perf_counter()-start
    import pickle
    # ru_maxrss units differ between macOS and Linux.
    div=1024**2 if platform.system()=='Darwin' else 1024
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/div if resource is not None else sampler.finish()
    baseline_mb=before/div if resource is not None else before/1024**2
    return dict(model=kind,seed=seed,site='0185-E',fit_windows=int(a['fit'].sum()),test_windows=int(a['test'].sum()),
        preprocessing_seconds=pre,fit_seconds=model.fit_seconds,test_score_seconds=scoring,
        process_peak_rss_mb=peak,baseline_peak_rss_mb=baseline_mb,
        rss_measurement='resource.ru_maxrss' if resource is not None else 'sampled_rss_5ms',
        serialized_model_bytes=len(pickle.dumps(model,protocol=5)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--one');p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.one:print(json.dumps(one(a.one,a.seed)));return
    rows=[]
    for seed in [17,42,91]:
        for kind in ALL_IF:
            result=subprocess.run([sys.executable,__file__,'--one',kind,'--seed',str(seed)],check=True,capture_output=True,text=True)
            rows.append(json.loads(result.stdout))
    pd.DataFrame(rows).to_csv(ROOT/'results/isolated_benchmark.csv',index=False)
if __name__=='__main__':main()
