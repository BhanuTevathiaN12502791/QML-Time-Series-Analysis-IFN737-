"""Audit source-to-score chronology, completeness, calibration, summaries and inference."""
import json,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np,pandas as pd,joblib
from core import metrics,read_data,prepare_fold,Detector,threshold,PRIMARY,BASELINES,ALL_IF,ObservedBaseline
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def independent_ends(d):
    # Sliding checks on the seven timestamp differences, independent of core's cumulative-gap algorithm.
    dif=np.diff(d.Date.to_numpy())==np.timedelta64(15,'m')
    return np.arange(7,len(d))[np.lib.stride_tricks.sliding_window_view(dif,7).all(axis=1)]

def main():
    p=json.loads((ROOT/'protocol.json').read_text());checks={};counter=0;errors=[]
    checks['protocol_sha256']=sha(ROOT/'protocol.json')==(ROOT/'protocol.sha256').read_text().strip()
    roles={r['site']:r['role'] for r in p['sites']};dev={s for s in roles if roles[s]=='development'};final=set(roles)-dev
    checks['frozen_disjoint_site_allocation']=len(dev)==30 and len(final)==117 and not dev&final
    checks['source_hashes']=all(sha(ROOT/'data'/f'{r["site"]}.csv')==r['sha256'] for r in p['sites'])
    tree=json.loads((ROOT/'source_tree.json').read_text());expected_git={Path(r['path']).stem:r['sha'] for r in tree['tree'] if r['path'].startswith('data/') and r['path'].endswith('.csv')}
    git_matches=True
    for site in roles:
        raw=(ROOT/'data'/f'{site}.csv').read_bytes();git_matches &= hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==expected_git[site]
    checks['all_147_git_object_hashes']=bool(git_matches)
    data={site:read_data(ROOT/'data'/f'{site}.csv') for site in roles}
    checks['source_weekday_agrees_with_parsed_date']=all((d.Weekday.str.lower()==d.Date.dt.day_name().str.lower()).all() for d in data.values())
    allgood=True;same=True;disjoint=True;source_aligned=True;complete=True;split_good=True;individual=True
    frames={}
    for folder in ['evaluation','normal_only']:
        df=pd.read_csv(ROOT/'results'/f'{folder}_metrics.csv');frames[folder]=df
        splitdf=pd.read_csv(ROOT/'results'/f'{folder}_splits.csv')
        wanted=set(roles) if folder=='evaluation' else final
        pairs={(k,0) for k in BASELINES}|{(k,s) for k in (ALL_IF if folder=='evaluation' else PRIMARY) for s in (p['forest_seeds'] if folder=='evaluation' else [42])}
        expected_rows={(k,s,label) for k,s in pairs for label in ['label_set_1','label_set_2']}
        complete &= set(df.site)==wanted and len(df)==len(wanted)*3*len(expected_rows)
        complete &= not df.duplicated(['site','fold','model','seed','label_set']).any()
        parts=[];splitparts=[]
        for site in sorted(wanted):
            parts.append(pd.read_csv(ROOT/'results'/folder/f'{site}_metrics.csv'))
            splitparts.append(pd.read_csv(ROOT/'results'/folder/f'{site}_splits.csv'))
            d=data[site];ends=independent_ends(d);testsets=[]
            for fold,fr in enumerate(p['folds']):
                g=df[(df.site==site)&(df.fold==fold)]
                complete &= set(zip(g.model,g.seed,g.label_set))==expected_rows and set(g.role)=={roles[site]}
                nf,nv,nt=[int(len(d)*f) for f in fr]
                fit=ends<nf;val=(ends>=nf)&(ends<nv);idx=ends[(ends>=nv)&(ends<nt)];testsets.append(set(idx))
                if folder=='normal_only':
                    clean=np.lib.stride_tricks.sliding_window_view(d.label_set_1.to_numpy()==0,8).all(axis=1)
                    fit &= clean[ends-7]
                saved_split=splitdf[(splitdf.site==site)&(splitdf.fold==fold)]
                if len(saved_split)!=1:split_good=False
                else:
                    sr=saved_split.iloc[0]
                    split_good &= all(int(sr[k])==v for k,v in dict(fit_end_row=nf,validation_end_row=nv,test_end_row=nt,
                        fit_windows=int(fit.sum()),validation_windows=int(val.sum()),test_windows=len(idx)).items())
                with np.load(ROOT/'results'/folder/f'{site}_fold{fold}.npz') as z:
                    same &= g.n.nunique()==1 and int(g.n.iloc[0])==len(idx)
                    source_aligned &= np.array_equal(z['indices'],idx)
                    source_aligned &= np.array_equal(pd.to_datetime(z['timestamps']).to_numpy(),d.Date.iloc[idx].to_numpy())
                    source_aligned &= np.array_equal(z['label1'],d.label_set_1.to_numpy()[idx]) and np.array_equal(z['label2'],d.label_set_2.to_numpy()[idx])
                    source_aligned &= np.allclose(z['volume'],d.Volume.to_numpy()[idx],atol=1e-5,rtol=1e-7)
                    expected_keys={'indices','timestamps','volume','label1','label2'}|{f'{k}__{s}' for k,s in pairs}|{f'threshold__{k}__{s}' for k,s in pairs}
                    complete &= set(z.files)==expected_keys
                    for row in g.itertuples():
                        key=f'{row.model}__{row.seed}';y=z['label1' if row.label_set=='label_set_1' else 'label2'];t=float(z['threshold__'+key]);scores=z[key]
                        ok=len(scores)==len(idx) and np.isfinite(scores).all()
                        calc=metrics(y,scores,t)
                        ok &= all(np.isclose(calc[k],getattr(row,k),rtol=1e-9,atol=1e-10) for k in calc)
                        ok &= bool(np.isclose(t,row.threshold))
                        allgood &= ok;counter+=1
                        if not ok:errors.append([site,int(fold),row.model,row.seed,row.label_set])
                    # The same threshold is reused under both inspector labels.
                    allgood &= bool((g.groupby(['model','seed']).threshold.nunique(dropna=False)==1).all())
            disjoint &= not any(testsets[i]&testsets[j] for i,j in [(0,1),(0,2),(1,2)])
        keys=['site','fold','model','seed','label_set'];actual=pd.concat(parts,ignore_index=True)
        individual &= df.sort_values(keys).reset_index(drop=True).equals(actual.sort_values(keys).reset_index(drop=True))
        actualsplit=pd.concat(splitparts,ignore_index=True)
        individual &= splitdf.sort_values(['site','fold']).reset_index(drop=True).equals(actualsplit.sort_values(['site','fold']).reset_index(drop=True))
    checks.update(complete_site_fold_method_seed_label_grid=bool(complete),saved_targets_times_labels_volumes_match_source=bool(source_aligned),
        saved_split_counts_match_independent_chronology=bool(split_good),individual_and_aggregate_files_match=bool(individual),
        all_methods_share_eligible_targets=bool(same),fold_test_targets_disjoint=bool(disjoint),saved_scores_reproduce_all_pointwise_metrics=bool(allgood))
    selection=json.loads((ROOT/'selected_parameters.json').read_text());dv=pd.read_csv(ROOT/'results/development_validation.csv')
    ranks=dv[dv.model.isin(PRIMARY)].groupby('max_samples').f1.mean().sort_values(ascending=False,kind='stable')
    kr=dv[dv.model=='sitdd_adapted'].groupby('k').f1.mean().sort_values(ascending=False,kind='stable')
    dr=dv[(dv.model.isin(PRIMARY))&(dv.max_samples==selection['max_samples'])].groupby('model').f1.mean().sort_values(ascending=False,kind='stable')
    dev_grid=True
    for site in dev:
        g=dv[dv.site==site]
        want={(f,k,s,n) for f in range(3) for k in PRIMARY for s in p['forest_seeds'] for n in p['max_samples_candidates']}
        actual=g[g.model.isin(PRIMARY)]
        dev_grid &= set(zip(actual.fold,actual.model,actual.seed,actual.max_samples))==want and len(actual)==len(want)
        actual=g[g.model=='sitdd_adapted'];want={(f,k) for f in range(3) for k in p['knn_candidates']}
        dev_grid &= set(zip(actual.fold,actual.k))==want and len(actual)==len(want)
    checks['complete_development_search_grid']=bool(dev_grid and set(dv.site)==dev)
    checks['global_selection_uses_development_validation']=int(ranks.index[0])==selection['max_samples'] and int(kr.index[0])==selection['sitdd_k'] and dr.index[0]==selection['demo_model']
    # Independent nested averaging: seeds -> fold -> site -> role/model/label summary.
    values=['precision','recall','f1','false_positive_rate','false_alerts_per_observed_day','average_precision']
    df=frames['evaluation'];expected=df.groupby(['role','site','fold','model','label_set'])[values].mean().groupby(['role','site','model','label_set']).mean().groupby(['role','model','label_set']).mean().sort_index()
    reported=pd.read_csv(ROOT/'results/summary.csv').set_index(['role','model','label_set']).sort_index()
    checks['macro_summaries_recompute']=expected.index.equals(reported.index) and bool(np.allclose(expected[values],reported[values],atol=1e-10,rtol=1e-10))
    # Refit a fixed pilot, additional development and final section; every IF representation and baseline, all folds.
    refit_sites=['0185-E',sorted(dev-{'0185-E','2414-E','2431-E','3385-S','4218-W','4443-W'})[0],sorted(final)[0]]
    refit_ok=True;refit_models=0
    for site in refit_sites:
        for fold,fr in enumerate(p['folds']):
            a=prepare_fold(data[site],fr);v=a['val'];ts=a['test'];y=data[site].label_set_1.to_numpy()[a['ends'][v]]
            with np.load(ROOT/'results/evaluation'/f'{site}_fold{fold}.npz') as z:
                for kind in ALL_IF+BASELINES:
                    w=a['rw'] if 'context' in kind else a['w'];seed=0 if kind in BASELINES else 42
                    if kind in BASELINES:
                        model=ObservedBaseline(kind,a['context'],selection['sitdd_k']).fit(w[a['fit']])
                        sv=model.score(w[v],a['slots'][a['ends'][v]]);st=model.score(w[ts],a['slots'][a['ends'][ts]])
                    else:
                        model=Detector(kind,selection['max_samples'],seed,p['n_estimators']).fit(w[a['fit']]);sv=model.score(w[v]);st=model.score(w[ts])
                    t=threshold(sv,y);key=f'{kind}__{seed}'
                    refit_ok &= bool(np.isclose(t,float(z['threshold__'+key]),atol=1e-12,rtol=1e-12)) and bool(np.allclose(st,z[key],atol=1e-12,rtol=1e-12));refit_models+=1
    checks['108_refits_reproduce_thresholds_and_predictions']=bool(refit_ok and refit_models==108)
    from detect import detect
    checkpoint_ok=True
    for modeldir,kind,seed in [('0185-E',selection['demo_model'],42),('0185-E_sitdd','sitdd_adapted',0)]:
        path=ROOT/'models'/modeldir/'detector.joblib';checkpoint=joblib.load(path);scored=detect(path,ROOT/'data/0185-E.csv');after=scored[scored.partition=='after_calibration']
        checkpoint_ok &= checkpoint['source_sha256']==sha(ROOT/'data/0185-E.csv') and checkpoint['protocol_sha256']==sha(ROOT/'protocol.json')
        with np.load(ROOT/'results/evaluation/0185-E_fold2.npz') as z:
            checkpoint_ok &= np.array_equal(after.timestamp.to_numpy(),pd.to_datetime(z['timestamps']).to_numpy())
            checkpoint_ok &= bool(np.allclose(after.anomaly_score,z[f'{kind}__{seed}'],atol=1e-12,rtol=1e-12))
            checkpoint_ok &= bool(np.isclose(checkpoint['threshold'],float(z[f'threshold__{kind}__{seed}']),atol=1e-12,rtol=1e-12))
    checks['both_delivered_checkpoints_match_recorded_test_predictions']=bool(checkpoint_ok)
    result=dict(verified_on=datetime.now(ZoneInfo('Australia/Brisbane')).date().isoformat(),checks=checks,metric_rows_recomputed=counter,
        refit_sites=refit_sites,models_refitted=refit_models,metric_mismatches=errors,passed=all(checks.values()))
    (ROOT/'results/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['passed']
if __name__=='__main__':main()
