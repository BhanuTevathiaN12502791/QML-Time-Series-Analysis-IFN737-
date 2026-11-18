import unittest,tempfile,json,hashlib
from unittest.mock import patch
from pathlib import Path
import numpy as np,pandas as pd
from core import Context,Detector,threshold,metrics,windows,prepare_fold,Representation,read_data
class ResearchTests(unittest.TestCase):
    def data(self,n=1000):
        return pd.DataFrame(dict(Date=pd.date_range('2020-01-01',periods=n,freq='15min'),Volume=20+np.arange(n)%96,label_set_1=0,label_set_2=0))
    def test_threshold_ties_choose_fewer_alerts(self):
        self.assertEqual(threshold([4,3,2,1],[1,0,0,1]),4.) # F1 tie: one alert versus four.
    def test_score_ties_are_never_split(self):
        t=threshold([2,2,1],[1,0,1]);self.assertEqual(t,1.)
    def test_validation_without_positive_labels(self):self.assertEqual(threshold([1,2],[0,0]),float('inf'))
    def test_zeros_are_preserved(self):
        d=self.data(12);d['Volume']=0;w,e=windows(d);self.assertEqual(len(w),5);self.assertTrue((w==0).all())
    def test_gaps_never_bridge(self):
        d=self.data(30).drop(index=15).reset_index(drop=True);w,e=windows(d);self.assertEqual(len(w),15)
        self.assertFalse(((e>=15)&(e<=21)).any())
    def test_current_observation_is_in_detection_window(self):
        d=self.data(12);w,e=windows(d);self.assertEqual(w[0,-1],d.Volume.iloc[7]);self.assertEqual(e[0],7)
    def test_future_changes_do_not_change_preprocessing(self):
        d=self.data();a=prepare_fold(d,[.4,.55,.7]);d.loc[400:,'Volume']=99999;b=prepare_fold(d,[.4,.55,.7])
        np.testing.assert_array_equal(a['context'].centres,b['context'].centres)
        np.testing.assert_array_equal(a['w'][a['fit']],b['w'][b['fit']])
    def test_labels_do_not_enter_primary_preprocessing(self):
        d=self.data();a=prepare_fold(d,[.4,.55,.7]);d.loc[:,'label_set_1']=1;b=prepare_fold(d,[.4,.55,.7])
        np.testing.assert_array_equal(a['rw'],b['rw']);np.testing.assert_array_equal(a['fit'],b['fit'])
    def test_pca_uses_only_fit_windows(self):
        d=self.data();a=prepare_fold(d,[.4,.55,.7]);r=Representation('if_raw_pca2').fit(a['w'][a['fit']])
        d.loc[400:,'Volume']=999999;b=prepare_fold(d,[.4,.55,.7]);other=Representation('if_raw_pca2').fit(b['w'][b['fit']])
        np.testing.assert_array_equal(r.scaler.mean_,other.scaler.mean_);np.testing.assert_array_equal(r.pca.components_,other.pca.components_)
    def test_normal_only_drops_contaminated_training_windows(self):
        d=self.data();d.loc[20,'label_set_1']=1;a=prepare_fold(d,[.4,.55,.7]);b=prepare_fold(d,[.4,.55,.7],True)
        self.assertEqual(a['fit'].sum()-b['fit'].sum(),8);np.testing.assert_array_equal(a['test'],b['test'])
    def test_sitdd_matches_original_distance_definition(self):
        c=Context().fit(np.array([1.,5.,8.,12.]),np.array([0,0,0,1]));s=c.knn(np.array([4.,9.]),np.array([0,1]),2)
        np.testing.assert_array_equal(s,[3.,3.])
    def test_unseen_context_slot_falls_back(self):
        c=Context().fit(np.array([1.,3.,5.]),np.array([0,0,0]));self.assertEqual(c.centres[20],3.);self.assertTrue(np.isfinite(c.knn(np.array([10.]),np.array([20]),3)).all())
    def test_metrics_are_pointwise(self):
        m=metrics([1,1,1,0],[3,0,0,0],2);self.assertEqual(m['tp'],1);self.assertEqual(m['fn'],2)
    def test_folds_have_disjoint_test_targets(self):
        d=self.data();sets=[]
        for fold in [[.4,.55,.7],[.55,.7,.85],[.7,.85,1.]]:
            a=prepare_fold(d,fold);sets.append(set(a['ends'][a['test']]))
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    def test_deterministic_forest_and_reload(self):
        import joblib
        x=np.random.default_rng(1).normal(size=(300,8));a=Detector('if_context_pca2',128,42).fit(x);b=Detector('if_context_pca2',128,42).fit(x)
        np.testing.assert_array_equal(a.score(x[:5]),b.score(x[:5]))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'m.joblib';joblib.dump(a,p);np.testing.assert_array_equal(a.score(x[:5]),joblib.load(p).score(x[:5]))
    def test_duplicate_timestamp_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.csv';d=self.data(20);d.loc[1,'Date']=d.loc[0,'Date'];d.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'unique'):read_data(p)
    def test_empty_csv_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'empty.csv';path.write_text('Date,Volume\n')
            with self.assertRaisesRegex(ValueError,'contain observations'):read_data(path,False)
    def test_fractional_quarter_hour_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'fraction.csv';d=self.data(20);d['Date']=d.Date+pd.Timedelta(microseconds=1);d.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'quarter-hour'):read_data(path)
    def test_invalid_fold_rejected(self):
        with self.assertRaisesRegex(ValueError,'endpoints'):prepare_fold(self.data(),[.7,.5,1.])
    def test_normal_only_without_training_rows_rejected(self):
        d=self.data();d['label_set_1']=1
        with self.assertRaisesRegex(ValueError,'No usable training'):prepare_fold(d,[.4,.55,.7],True)
    def test_threshold_matches_exhaustive_sklearn_search(self):
        from sklearn.metrics import f1_score
        rng=np.random.default_rng(8)
        for _ in range(100):
            scores=rng.integers(0,7,size=30);y=rng.integers(0,2,size=30).astype(bool)
            candidates=[float('inf'),*np.unique(scores).astype(float)]
            rows=[(t,f1_score(y,scores>=t,zero_division=0),int((scores>=t).sum())) for t in candidates]
            best=max(r[1] for r in rows);fewest=min(r[2] for r in rows if np.isclose(r[1],best,atol=1e-12,rtol=0))
            selected=threshold(scores,y)
            self.assertAlmostEqual(f1_score(y,scores>=selected,zero_division=0),best)
            self.assertEqual(int((scores>=selected).sum()),fewest)
    def test_known_section_mismatch_rejected(self):
        import joblib
        from detect import detect
        with tempfile.TemporaryDirectory() as td:
            model=Path(td)/'model.joblib';joblib.dump({'site':'0185-E'},model)
            with self.assertRaisesRegex(ValueError,'Section mismatch'):detect(model,Path(td)/'2414-E.csv')
    def test_explicit_section_mismatch_rejected(self):
        import joblib
        from detect import detect
        with tempfile.TemporaryDirectory() as td:
            model=Path(td)/'model.joblib';joblib.dump({'site':'0185-E'},model)
            with self.assertRaisesRegex(ValueError,'Section mismatch'):detect(model,Path(td)/'counts.csv',site='2414-E')
    def test_resume_detects_changed_outputs_code_and_data(self):
        import cache
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'data').mkdir();folder=root/'results/development';folder.mkdir(parents=True)
            data=root/'data/0185-E.csv';data.write_text('source')
            proto={'sites':[{'site':'0185-E','sha256':hashlib.sha256(data.read_bytes()).hexdigest()}]}
            (root/'protocol.json').write_text(json.dumps(proto));(root/'protocol.sha256').write_text(cache.sha(root/'protocol.json'))
            for name in ['core.py','experiment.py','cache.py']:(root/name).write_text('code')
            output=folder/'0185-E.csv';output.write_text('result')
            with patch.object(cache,'ROOT',root):
                cache.stamp('0185-E','development');self.assertTrue(cache.reusable('0185-E','development'))
                output.write_text('tampered')
                with self.assertRaisesRegex(ValueError,'outputs'):cache.reusable('0185-E','development')
                output.write_text('result');(root/'core.py').write_text('changed')
                with self.assertRaisesRegex(ValueError,'code/environment'):cache.reusable('0185-E','development')
                (root/'core.py').write_text('code');data.write_text('changed')
                with self.assertRaisesRegex(ValueError,'source data'):cache.reusable('0185-E','development')
    def test_portable_rss_fallback(self):
        import benchmark
        with patch.object(benchmark,'resource',None),patch.object(benchmark,'read_data',return_value=self.data()):
            result=benchmark.one('if_raw_full',42)
        self.assertEqual(result['rss_measurement'],'sampled_rss_5ms')
        self.assertGreaterEqual(result['process_peak_rss_mb'],result['baseline_peak_rss_mb'])
        self.assertTrue(np.isfinite(result['process_peak_rss_mb']))
    def selection_fixture(self):
        return dict(max_samples=256,sitdd_k=3,n_estimators=100,demo_model='if_context_full',protocol_sha256='fixed',
                    max_samples_validation_f1={'128':.2,'256':.3},knn_validation_f1={'3':.6,'5':.5})
    def test_frozen_selection_tolerates_roundoff_without_rewriting(self):
        from experiment import preserve_frozen_selection
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'selected.json';frozen=self.selection_fixture();target.write_text(json.dumps(frozen));original=target.read_bytes()
            candidate=self.selection_fixture();candidate['max_samples_validation_f1']['256']+=1e-12
            self.assertEqual(preserve_frozen_selection(target,candidate),frozen)
            self.assertEqual(target.read_bytes(),original)
    def test_changed_frozen_choice_is_rejected(self):
        from experiment import preserve_frozen_selection
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'selected.json';target.write_text(json.dumps(self.selection_fixture()));candidate=self.selection_fixture();candidate['max_samples']=128
            with self.assertRaisesRegex(ValueError,'Frozen selection changed: max_samples'):preserve_frozen_selection(target,candidate)
    def test_material_validation_drift_is_rejected(self):
        from experiment import preserve_frozen_selection
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'selected.json';target.write_text(json.dumps(self.selection_fixture()));candidate=self.selection_fixture();candidate['knn_validation_f1']['3']=.2
            with self.assertRaisesRegex(ValueError,'diagnostics changed materially'):preserve_frozen_selection(target,candidate)
    def test_runner_requires_source_before_starting_experiments(self):
        import run_all,sys
        with patch.object(sys,'argv',['run_all.py']),patch.object(run_all,'missing_source_files',return_value=[Path('missing.csv')]),patch.object(run_all.subprocess,'run') as execute:
            with self.assertRaises(SystemExit) as failure:run_all.main()
            self.assertEqual(failure.exception.code,2);execute.assert_not_called()
    def test_runner_fetch_is_explicit_and_creates_results_root(self):
        import run_all,sys
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with patch.object(run_all,'ROOT',root),patch.object(sys,'argv',['run_all.py','--fetch','--workers','2']),patch.object(run_all,'missing_source_files',side_effect=[[Path('missing.csv')],[]]),patch.object(run_all.subprocess,'run') as execute:
                run_all.main()
            self.assertTrue((root/'results').is_dir())
            self.assertEqual(Path(execute.call_args_list[0].args[0][1]).name,'fetch_data.py')
            self.assertEqual(len(execute.call_args_list),10)
if __name__=='__main__':unittest.main()
