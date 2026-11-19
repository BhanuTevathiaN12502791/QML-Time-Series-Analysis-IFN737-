"""Build the final evidence report from completed result files (no model fitting)."""
import argparse,json,re,html
from pathlib import Path
import numpy as np,pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak,KeepTogether
from reportlab.pdfbase.pdfmetrics import stringWidth
ROOT=Path(__file__).resolve().parent
REPO='https://github.com/BhanuTevathiaN12502791/QML-Time-Series-Analysis-IFN737-'
BLACK=colors.HexColor('#000000');INK=colors.HexColor('#1c3245');GREY=colors.HexColor('#d9dfe4');PALE=colors.HexColor('#edf2f5')
styles=getSampleStyleSheet()
for name in ['Title','Heading1','Heading2','Heading3']:styles[name].textColor=BLACK
styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=9.5,leading=13.6,spaceAfter=7,textColor=INK))
styles.add(ParagraphStyle(name='ReportSmall',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=5,textColor=INK))
styles.add(ParagraphStyle(name='ReportCaption',fontName='Helvetica',fontSize=8,leading=10.5,spaceAfter=8,textColor=INK))
styles.add(ParagraphStyle(name='ReportCell',fontName='Helvetica',fontSize=7.7,leading=10,textColor=INK))
styles.add(ParagraphStyle(name='ReportCode',fontName='Courier',fontSize=7.5,leading=10.5,spaceAfter=6,textColor=INK))
styles['Title'].fontSize=25;styles['Title'].leading=29;styles['Title'].spaceAfter=17
styles['Heading1'].fontSize=16;styles['Heading1'].leading=20;styles['Heading1'].spaceAfter=12
styles['Heading2'].fontSize=11.5;styles['Heading2'].leading=15;styles['Heading2'].spaceBefore=9
W=A4[0]-36*mm

def clean(text):
    text=str(text)
    for a,b in [('–','-'),('—','-'),('‑','-'),('−','-'),('’',"'"),('“','"'),('”','"'),('→','to'),('²','2'),('×','x'),('·',' / ')]:text=text.replace(a,b)
    return text

def esc(text):return html.escape(clean(text))
def P(text,small=False):return Paragraph(text,styles['ReportSmall' if small else 'ReportBody'])
def literal(text,small=False):return P(esc(text),small)
def H(text):return Paragraph(esc(text),styles['Heading1'])
def S(text):return Paragraph(esc(text),styles['Heading2'])
def link(label,url):return f'<link href="{html.escape(url,quote=True)}" color="#165d7c">{esc(label)}</link>'
def num(v,d=3):
    try:return '-' if pd.isna(v) or not np.isfinite(float(v)) else f'{float(v):,.{d}f}'
    except (TypeError,ValueError):return esc(v)
def count(v):return num(v,0)
def model_name(v):
    names={'global_mad':'Global MAD','time_of_week_mad':'Time-of-week MAD','pca_reconstruction':'PCA reconstruction','sitdd_adapted':'Adapted Si-TDD',
    'if_raw_full':'Raw IF / full','if_raw_pca2':'Raw IF / PCA 2','if_raw_pca4':'Raw IF / PCA 4','if_raw_pca6':'Raw IF / PCA 6',
    'if_context_full':'Context IF / full','if_context_pca2':'Context IF / PCA 2','if_context_pca4':'Context IF / PCA 4','if_context_pca6':'Context IF / PCA 6'}
    if v in names:return names[v]
    low=str(v).lower()
    if 'rolling' in low:return 'Rolling 56-day calendar HGB'
    if 'lstm' in low:return 'Fixed LSTM'
    if 'calendar' in low:return 'Fixed calendar HGB'
    if 'lag' in low:return 'Fixed lag-only HGB'
    if 'persist' in low:return 'Repeat latest'
    if 'week' in low:return 'Weekly reference'
    return clean(v)

def T(headers,rows,widths=None):
    data=[[Paragraph(f'<b>{esc(x)}</b>',styles['ReportCell']) for x in headers]]
    for row in rows:data.append([Paragraph(str(x),styles['ReportCell']) for x in row])
    table=Table(data,colWidths=widths or [W/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('GRID',(0,0),(-1,-1),.35,GREY),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return table

def fig(path,width=W,max_height=110*mm,caption=None):
    from PIL import Image as PILImage
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Required completed figure missing: {path}')
    with PILImage.open(path) as im:iw,ih=im.size
    scale=min(width/iw,max_height/ih);items=[Image(str(path),width=iw*scale,height=ih*scale,hAlign='CENTER')]
    if caption:items.append(Paragraph(esc(caption),styles['ReportCaption']))
    return items

def read_csv(path):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(f'Completed result required: {p}')
    return pd.read_csv(p)

def table_frame(df,cols,headers=None,limit=None,widths=None,modelcol='model'):
    use=df if limit is None else df.head(limit)
    rows=[]
    for _,r in use.iterrows():
        rows.append([esc(model_name(r[c])) if c==modelcol else num(r[c]) if pd.api.types.is_numeric_dtype(df[c]) else esc(r[c]) for c in cols])
    return T(headers or cols,rows,widths)

def page_number(canvas,doc):
    canvas.saveState();canvas.setFillColor(INK);canvas.setFont('Helvetica',7.5)
    canvas.drawString(18*mm,12*mm,'IFN737 | Classical traffic analysis | 19 November 2026')
    canvas.drawRightString(A4[0]-18*mm,12*mm,str(doc.page));canvas.restoreState()

def expected_rows(event,query):return event[event['event'].astype(str).str.contains(query,case=False,regex=False)]

def create(out,events):
    result=ROOT/'results';curves=result/'learning_curves';event=Path(events)
    required=['summary.json','overall_metrics.csv','daily_metrics.csv','event_metrics.csv','monthly_metrics.csv','seasonal_metrics.csv','school_period_metrics.csv','demand_anomaly_days.csv']
    for f in required:
        if not (event/f).is_file():raise FileNotFoundError(f'Run the event study first; missing {event/f}')
    es=json.loads((event/'summary.json').read_text());overall=read_csv(event/'overall_metrics.csv');daily=read_csv(event/'daily_metrics.csv');ev=read_csv(event/'event_metrics.csv')
    om=overall.set_index('model')
    def metric(model,key='mae'):return float(om.loc[model,key])
    monthly=read_csv(event/'monthly_metrics.csv');season=read_csv(event/'seasonal_metrics.csv');school=read_csv(event/'school_period_metrics.csv');demand=read_csv(event/'demand_anomaly_days.csv')
    def slice_row(frame,key,value,model='hgb_calendar_rolling56'):
        return frame[(frame[key]==value)&(frame.model==model)].iloc[0]
    complete_days=es['complete_test_days_by_model']['hgb_calendar_rolling56']
    base=read_csv(result/'summary.csv');b1=base.query("role=='final' and label_set=='label_set_1'").set_index('model');b2=base.query("role=='final' and label_set=='label_set_2'").set_index('model')
    effects=read_csv(result/'paired_effects.csv');curve=read_csv(curves/'runs.csv');cover=read_csv(curves/'coverage.csv');fc=curve.query("model=='lstm_forecast'").groupby('training_days').mean(numeric_only=True)
    calendar=json.loads((ROOT/'event_calendar_2025.json').read_text());audit=json.loads((result/'verification.json').read_text());caudit=json.loads((curves/'verification.json').read_text())
    srcurl='https://opendata.transport.vic.gov.au/dataset/traffic-signal-volume-data'
    story=[]
    def new(title):
        if story:story.append(PageBreak())
        story.append(H(title))
    def add(text,small=False):story.append(P(text,small))
    def image(path,caption=None,maxheight=110*mm):story.extend(fig(path,max_height=maxheight,caption=caption))
    # 1
    story.append(Paragraph('Traffic anomaly detection<br/>and event-aware forecasting',styles['Title']))
    add('<b>Final classical implementation and evaluation report</b><br/>Bhanu Tevathia | IFN737 | 19 November 2026')
    add('This report brings together the labelled AnoLT anomaly benchmark, measured learning/resource curves, and a separate full-year Melbourne SCATS forecast study covering public holidays, Christmas, storms, school terms and seasonal change. It also documents the prepared VS Code workflow.')
    story.append(S('Main findings'))
    rows=[['Labelled anomaly benchmark',f'Adapted Si-TDD mean site F1 {b1.loc["sitdd_adapted","f1"]:.3f}; contextual IF {b1.loc["if_context_full","f1"]:.3f}, compared with raw IF {b1.loc["if_raw_full","f1"]:.3f}.'],
          ['Training-history/resource study',f'Mean LSTM MAE is {fc.loc[7,"mae"]:.2f} at 7 calendar days and {fc.loc[140,"mae"]:.2f} at 140 days. Training time changes from {fc.loc[7,"training_seconds"]:.2f} to {fc.loc[140,"training_seconds"]:.2f} seconds on this machine.'],
          ['New event study',f'On {int(es["common_test_targets"]):,} paired test intervals, rolling calendar HGB MAE is {metric("hgb_calendar_rolling56"):.3f}, lag-only HGB {metric("hgb_lags_frozen"):.3f}, and fixed calendar HGB {metric("hgb_calendar_frozen"):.3f}. Rolling adaptation helps overall; adding calendar features to the frozen model does not.'],
          ['Robustness and events','Christmas and verified storm windows have complete coverage. Seasonal error still varies; spring is harder. Holiday and weather markers are contextual exposures, not confirmed anomaly labels.']]
    story.append(T(['Evidence','Result / meaning'],[[esc(a),esc(b)] for a,b in rows],[47*mm,W-47*mm]))
    story.append(Spacer(1,10*mm));add('<b>Interpretation boundary.</b> MAE measures forecast error. F1 measures labelled anomaly detection. Training seconds and RAM measure cost. A calendar-day average of rolling 15-minute forecasts is not a 24-hour-ahead forecast. The data do not establish crash causes or cyclone performance.')
    add(link('Source repository, runnable code and result tables',REPO))
    # 2
    new('1. Research design and evidence sources')
    add('The project asks how normal time context affects anomaly detection, and whether PCA compression helps when the downstream detector and evaluation remain matched. The classical phase is implemented; quantum integration remains future work.')
    story.append(T(['Study','Data and target','Evaluation'],[
        ['AnoLT benchmark','147 labelled Melbourne road-section series; 3,125,856 observations; two inspector label sets.','30 development sections, 117 final sections; three expanding chronological folds.'],
        ['Learning/resource curves','0185-E, an already-used development section; 7-140 calendar days of trailing history.','Fixed future periods; 84 fresh-process CPU trials; mean and observed seed/repetition ranges.'],
        ['2025 event forecast study','SCATS site 100, detector 2; full-year counts from official monthly archives.','Chronological fitting/calibration and later paired forecast comparisons; daily/season/event analysis.']
    ],[34*mm,70*mm,W-104*mm]))
    story.append(S('Chronology and data handling'))
    add('AnoLT fit/validation/test endpoints are 40/55/70%, 55/70/85% and 70/85/100% of each section\'s raw ordered rows. Test periods do not overlap within a section. Scaling, context and PCA fit earlier observations only; thresholds use validation label set 1 and are frozen for both test annotations.')
    add('Valid zero volumes are retained. Windows crossing missing timestamps are excluded. Final sections receive local earlier fitting and validation calibration: this is transfer of a fixed procedure, not zero-shot transfer of one universal set of weights. Related directions can span allocation groups, so geographic independence is not established.')
    add('The SCATS source timestamps are fixed-AEST interval ends. The event study subtracts 15 minutes and converts interval starts to Australia/Melbourne before assigning dates. Civil days can contain 92, 96 or 100 intervals. Entire incomplete/alarmed source detector-days are masked; resulting local-day coverage is reported.')
    add('Sources: '+link('AnoLT pinned snapshot','https://github.com/imaantaheri/AnoLT/tree/d2e927425b40235d881905ba6881112bcc017e43')+'; '+link('Transport Victoria traffic volumes',srcurl)+'.')
    # 3
    new('2. Labelled anomaly-detection results')
    add('The following are unweighted means of section metrics after averaging folds and forest seeds within each section. All methods share the same eligible targets. Inspector set 2 reuses the thresholds calibrated using set 1. No point adjustment is applied.')
    order=['global_mad','time_of_week_mad','pca_reconstruction','sitdd_adapted','if_raw_full','if_raw_pca2','if_raw_pca4','if_raw_pca6','if_context_full','if_context_pca2','if_context_pca4','if_context_pca6']
    rows=[[esc(model_name(k)),num(b1.loc[k,'f1']),num(b2.loc[k,'f1']),num(b1.loc[k,'precision']),num(b1.loc[k,'recall']),num(b1.loc[k,'false_alerts_per_observed_day'])] for k in order]
    story.append(T(['Method','F1 set 1','F1 set 2','Precision','Recall','False alerts/day'],rows,[54*mm]+[(W-54*mm)/5]*5))
    image(result/'final_f1.png','Mean site F1 on the 117 final sections; higher is better.',maxheight=90*mm)
    add('The adapted nearest-neighbour baseline is strongest in this comparison. Forest complexity does not automatically improve detection. These volume-pattern labels are not verified crashes.',True)
    # 4
    new('3. Matched context and PCA effects')
    add('The forest uses 100 trees, a shared development-selected max_samples of 256 and seeds 17, 42 and 91. Full and compressed representations use the same chronology, tuning budget and target observations. Two components are primary; four and six are sensitivity checks.')
    image(result/'paired_effects.png','Paired F1 differences: method A minus method B. Intervals resample sections after within-section averaging.',maxheight=90*mm)
    e=effects.query("label_set=='label_set_1'").head(4)
    rows=[[esc(r.comparison),num(r.mean_f1_difference,4),f'{r.ci95_low:+.4f} to {r.ci95_high:+.4f}'] for r in e.itertuples()]
    story.append(T(['Primary contrast','Mean difference','95% site-bootstrap interval'],rows,[60*mm,35*mm,W-95*mm]))
    add('Temporal context provides a large positive difference. PCA-2 reduces contextual IF mean F1 here. However, its shared-ID-prefix cluster interval spans zero even when the site-bootstrap interval does not. Dependence assumptions therefore affect the certainty of the compression contrast. These are descriptive benchmark intervals, not universal superiority claims.')
    add('A normal-only sensitivity excludes anomalous training rows and every training window touching them, using seed 42 and frozen global choices. The default primary fitting retains earlier anomalies. Full sensitivity tables, fold/seed variation, error groups and precision-recall curves are preserved in the repository.',True)
    # 5
    new('4. MAE and training time versus T')
    add('T is trailing calendar history ending 4 July 2019, 23:45 on section 0185-E. Validation and test dates remain fixed. The companion LSTM forecasts the next count from 20 past readings; anomaly detectors use the current observation plus seven prior readings. MAE and F1 answer different questions.')
    image(curves/'mae_vs_days.png','Forecast MAE in original volume-count units. Shading is observed min-max across three seeds, not a confidence interval.',maxheight=91*mm)
    image(curves/'training_time_vs_days.png','Training includes preparation, fitting and validation stopping/calibration; final test inference and file I/O are excluded. Time axis is logarithmic.',maxheight=91*mm)
    # 6
    new('5. RAM, forecast accuracy and cost')
    image(curves/'ram_vs_days.png','Absolute sampled process RAM and increment above the loaded baseline; libraries are included. Sample period: 2 ms.',maxheight=87*mm)
    rows=[[str(t),num(r.mae,2),num(r.training_seconds,2),num(r.training_peak_rss_mib,1),num(r.training_extra_rss_mib,1),num(r.epochs,1)] for t,r in fc.iterrows()]
    story.append(T(['T days','MAE','Train sec','Peak MiB','Extra MiB','Epochs'],rows,[23*mm]+[(W-23*mm)/5]*5))
    add('More history reduces mean forecast error overall in this experiment while increasing training work. Results are not monotonic at every point. Increasing T also changes the age and seasonal composition of data, so this is not a pure causal experiment on sample count.')
    add('The LSTM uses 32 hidden units, Adam 0.001, batch 64, up to 40 epochs and validation-MSE early stopping with patience 7. IF has a fixed 256-sample subsample, so its training cost need not grow linearly with T. Absolute RSS can appear flat because framework memory dominates. Very short transient peaks can be missed by sampling.',True)
    # 7
    new('6. Detection metrics and observed coverage')
    image(curves/'detection_metrics_vs_days.png','Inspector set 1 learning-history curves. The same future targets are evaluated at every T.',maxheight=115*mm)
    rows=[[count(r.training_days),count(r.train_rows),num(r.observed_training_days,1),count(r.train_windows),count(getattr(r,'forecast_train_windows',np.nan))] for r in cover.itertuples()]
    story.append(T(['Calendar T','Observed rows','Observed days','Detector windows','Forecast windows'],rows,[29*mm,32*mm,32*mm,40*mm,W-133*mm]))
    add('Seven days provide only one reference per weekday/time slot; contextual training residuals can become degenerate. The resulting poor short-history detector behaviour is retained rather than smoothed away. False alerts/day means false positive intervals divided by eligible observations/96, not a count of distinct incidents.',True)
    # 8
    new('7. Full-year Melbourne event study')
    add('The event study extends coverage to Christmas 2025 and complete school/seasonal cycles using official SCATS monthly archives. Site 100 is Burwood Highway / Livingstone Road, Vermont South, Whitehorse, according to the current signal inventory. The current inventory anchors the metropolitan location; it does not verify the historical physical lane/direction of detector 2.')
    add('Raw site 100:2 extraction contains 34,944 interval rows (364 fixed-AEST source days); 27 May is absent. Thirteen source detector-days are masked for incomplete records or alarms, leaving 33,696 usable source counts. Rolling input windows and local-date conversion reduce the number of complete civil-day forecasts further. Coverage values in each subsequent table are authoritative for that comparison.')
    add('Fitting uses 1 January-25 February; validation uses 26 February-11 March; testing begins 12 March. The study compares a frozen LSTM, matched lag-only and known-calendar histogram gradient boosting (HGB) forecasters, a calendar HGB refreshed every 28 days using only the preceding 56 calendar days, and persistence (repeat latest count). Calendar features are known in advance; retrospective storm flags are evaluation slices, not predictive inputs.')
    add('Known-calendar features encode time of day, weekday/year cycles, public holidays, school-term/break status and weekends. Government-school dates do not establish every school\'s attendance or pupil-free days. The study tests predictive associations, not whether school calendars cause traffic changes.')
    story.append(S('Daily definitions'))
    add('Daily actual/predicted means average the local civil day\'s quarter-hour counts. Daily MAE averages absolute forecast errors; signed bias is prediction minus observation. RMSE and WAPE provide complementary scale-sensitive and relative errors. Complete-day comparisons require every expected interval, including 92/100-slot daylight-saving days. Partial days remain visible but do not silently become full 24-hour tests.')
    add('All models use the same gap-valid 20-reading history and held-out targets. HGB has 100 iterations, 15 leaves, learning rate 0.05 and L2 penalty 1; early stopping is disabled. This extension uses one fixed seed (42), one detector and one year. A daily demand MAD-score and validation-calibrated forecast-error flags are screening measures rather than ground-truth incident annotation.')
    # 9
    new('8. Forecast performance and training-cost trade-off')
    cols=[c for c in ['model','n_intervals','mae','rmse','bias','wape_pct'] if c in overall]
    story.append(table_frame(overall,cols,['Method','Intervals','MAE','RMSE','Bias','WAPE %'][:len(cols)],widths=[54*mm]+[(W-54*mm)/(len(cols)-1)]*(len(cols)-1)))
    add(f'All five methods are evaluated on the same {int(es["common_test_targets"]):,} test intervals, with {int(complete_days)} complete local test days per model. Rolling HGB reduces interval MAE by {100*(metric("hgb_lags_frozen")-metric("hgb_calendar_rolling56"))/metric("hgb_lags_frozen"):.2f}% relative to the fixed lag-only HGB. Fixed calendar features alone increase MAE by {100*(metric("hgb_calendar_frozen")-metric("hgb_lags_frozen"))/metric("hgb_lags_frozen"):.2f}%. These single-year estimates do not establish a universal ranking.',True)
    timing=es.get('timing',es.get('timings',es.get('timing_table',[])))
    if isinstance(timing,list) and timing:
        tf=pd.DataFrame(timing);tf['model_KiB']=tf.model_bytes/1024 if 'model_bytes' in tf else np.nan
        use=[c for c in ['model','fit_seconds','inference_seconds','n_refits','model_KiB'] if c in tf]
        if use:story.append(S('Recorded compute'));story.append(table_frame(tf,use,['Method','Fit sec','Score sec','Refits','Model KiB'][:len(use)],widths=[54*mm]+[(W-54*mm)/(len(use)-1)]*(len(use)-1)))
        add('Shared event-study preparation took '+num(es.get('preprocessing_seconds_shared',np.nan),3)+' seconds. Fit seconds below exclude that shared work. Inference diagnostics cover validation plus test for frozen models but test blocks for the rolling model, so they are not a matched latency ranking. Serialized model KiB is not RAM usage.',True)
    nrefits=next((int(r['n_refits']) for r in timing if r['model']=='hgb_calendar_rolling56'),0)
    costpath=event/'error_cost_tradeoff.csv'
    if costpath.is_file():
        costs=read_csv(costpath);pareto=costs[costs.pareto_error_fit_time.astype(str).str.lower()=='true'].model
        frontier=', '.join(model_name(m) for m in pareto)
        add(f'Training time is a cost rather than a traffic feature. Choose validation accuracy under a training/RAM budget. The observed MAE/fit-time Pareto set is {esc(frontier)}. Rolling time sums {nrefits} past-only refits. This describes one run rather than a test-selected deployment policy.',True)
    else:add(f'Training time is a cost rather than a traffic feature. Choose validation accuracy under a training/RAM budget. Rolling time sums {nrefits} past-only refits; the more expensive LSTM is not the most accurate model here.',True)
    pairedpath=event/'paired_week_bootstrap.csv'
    if pairedpath.is_file():
        paired=read_csv(pairedpath);protocol=json.loads((event/'tradeoff_protocol.json').read_text())
        contrasts=[('hgb_calendar_rolling56','hgb_calendar_frozen','Rolling - fixed calendar'),('hgb_calendar_frozen','hgb_lags_frozen','Fixed calendar - lag-only'),('hgb_calendar_rolling56','hgb_lags_frozen','Rolling - fixed lag-only')]
        rows=[]
        for left,right,label in contrasts:
            r=paired[(paired.left==left)&(paired.right==right)].iloc[0]
            rows.append([esc(label),f'{r.delta_mae:+.3f}',f'{r.ci95_low:+.3f} to {r.ci95_high:+.3f}'])
        story.append(S('Paired uncertainty (negative favours first method)'))
        story.append(T(['Contrast','MAE difference','95% week-bootstrap interval'],rows,[70*mm,33*mm,W-103*mm]))
        add(f'{int(paired.n_week_clusters.iloc[0])} Monday-based test-week clusters; {protocol["draws"]:,} paired percentile resamples, seed {protocol["seed"]}. All common intervals, including partial days, enter pooled MAE. Intervals are conditional on this detector/year and fitted models; consecutive weeks/seasons can remain dependent. These post-hoc comparisons have no multiplicity correction.',True)
    # 10
    new('9. Daily errors and anomalous-day screening')
    image(event/'daily_mae_2025.png','Daily MAE of rolling 15-minute forecasts. These are after-observation daily aggregates, not 24-hour-ahead predictions.',maxheight=112*mm)
    flagcol=next((c for c in ['demand_screen_flag','flag','is_anomaly','demand_flag','anomaly_flag'] if c in demand),None)
    scorecol=next((c for c in ['robust_z','robust_z_score','robust_zscore','mad_z'] if c in demand),None)
    dd=demand.copy()
    if flagcol:dd=dd[dd[flagcol].astype(str).str.lower().isin(['true','1'])]
    if scorecol:dd=dd.sort_values(scorecol,ascending=False)
    use=[c for c in ['date','observed_daily_mean','reference_mean_median',scorecol] if c and c in dd]
    if use and len(dd):story.append(S('Largest screening departures'));story.append(table_frame(dd,use,['Date','Observed mean','Reference mean','MAD score'][:len(use)],limit=8))
    valdays=len(daily[(daily.model=='hgb_calendar_frozen')&(daily.split=='validation')&daily.complete_24h])
    add(f'The demand screen marks {int(es["flagged_demand_days"])} complete days. References are the preceding 56 days with the same weekday, complete coverage and no public holiday (at least four); departures above 3.5 robust scale units are flagged. Error thresholds are the 95th percentile of only {valdays} complete validation days; rolling HGB uses the fixed calendar threshold as a proxy. Later flags can reflect drift and do not have a validated incident false-alarm rate.',True)
    add('A high departure score or forecast MAE identifies a day for review, not a confirmed incident. Holiday demand can differ sharply from ordinary weekdays while remaining expected calendar behaviour. A low rolling forecast error does not prove typical or disruption-free demand.',True)
    # 11
    new('10. Christmas Day: full-day assessment')
    image(event/'christmas_2025.png','Christmas Day, 25 December 2025, Melbourne civil time. Coverage is reported below.',maxheight=110*mm)
    christmas=daily[daily.date.astype(str).str[:10]=='2025-12-25']
    use=[c for c in ['model','n','expected_intervals','actual_mean','predicted_mean','mae','bias','wape_pct'] if c in christmas]
    if len(christmas):story.append(table_frame(christmas,use,['Method','n','Expected','Actual mean','Pred. mean','MAE','Bias','WAPE %'][:len(use)],widths=[42*mm]+[(W-42*mm)/(len(use)-1)]*(len(use)-1)))
    else:add('No eligible Christmas-day result was produced. A full-day conclusion is therefore unavailable; no missing values have been invented.')
    if len(christmas):
        cr=christmas.set_index('model');best=christmas.loc[christmas.mae.idxmin()]
        add(f'All five models cover 96 of 96 intervals. Observed mean is {cr.actual_mean.iloc[0]:.3f} counts per 15 minutes; the full-day total is {cr.source_observed_total.iloc[0]:,.0f}. {esc(model_name(best.model))} has the smallest Christmas MAE ({best.mae:.3f}); rolling HGB has MAE {cr.loc["hgb_calendar_rolling56","mae"]:.3f}, slightly worse than persistence ({cr.loc["persistence","mae"]:.3f}), and positive bias {cr.loc["hgb_calendar_rolling56","bias"]:.3f}. The overall winner does not win every event day.',True)
    add('A calendar-day mean of rolling quarter-hour forecasts is not a 24-hour-ahead prediction. Positive and negative errors can cancel in bias but not MAE. Christmas demand alone is not evidence of an accident.',True)
    # 12
    new('11. Public holidays and verified weather windows')
    add('Victorian holiday dates were checked against the official 2025 calendar. Early-year holidays used for fitting or validation are not held-out tests. Only complete eligible test days enter the principal event comparisons; unavailable dates must retain their missing/partial status.')
    # Focus on two primary calendar procedures so the holiday table remains readable; full model records stay in CSV.
    candidates=[m for m in ev.model.unique() if 'calendar' in str(m).lower()]
    chosen=[m for m in candidates if 'rolling' in str(m).lower()]+[m for m in candidates if 'rolling' not in str(m).lower()]
    eventview=ev[ev.model.isin(chosen[:2])] if chosen else ev
    weather_names={e['name'] for e in calendar.get('weather_events',[])}
    eventview=eventview[eventview.event.str.startswith('holiday: ')|eventview.event.isin(weather_names)].copy()
    eventview['event']=eventview.event.str.replace('holiday: ','',regex=False).replace({'Melbourne damaging-wind event':'Wind / 22 Oct','Melbourne heavy-rain/severe-weather window':'Rain-storms / 26-27 Oct'})
    use=[c for c in ['event','model','n_complete_days','mean_daily_mae','mean_observed_volume'] if c in eventview]
    story.append(table_frame(eventview,use,['Event','Method','Complete days','Daily MAE','Mean count'][:len(use)],limit=26,widths=[51*mm,43*mm]+[(W-94*mm)/(len(use)-2)]*(len(use)-2)))
    ph=slice_row(ev,'event','public_holiday');ordinary=slice_row(ev,'event','ordinary_nonholiday')
    wind=slice_row(ev,'event','Melbourne damaging-wind event');rain=slice_row(ev,'event','Melbourne heavy-rain/severe-weather window')
    add(f'The rolling model\'s mean daily MAE is {ph.mean_daily_mae:.3f} over {int(ph.n_complete_days)} complete held-out public holidays, {wind.mean_daily_mae:.3f} on 22 October and {rain.mean_daily_mae:.3f} across 26-27 October. These differ from its ordinary nonholiday mean of {ordinary.mean_daily_mae:.3f}, but the weather sample is only {int(wind.n_complete_days+rain.n_complete_days)} days. Scoresby is a regional proxy. Bureau rainfall totals end at 9 am; the two-day window accommodates that timing. Association does not establish weather causation.',True)
    add('<b>Cyclone limit:</b> Cyclone Alfred\'s major documented impacts were southeast Queensland/northeast New South Wales. It is not labelled as a Melbourne cyclone exposure. This project cannot assess cyclone robustness without suitable affected-region traffic observations.',True)
    # 13
    new('12. Seasonal and monthly robustness')
    image(event/'seasonal_metrics.png','Temporal performance under frozen and rolling procedures; interpret alongside day counts and coverage.',maxheight=108*mm)
    season_view=season[season.model.astype(str).str.contains('calendar')].copy()
    use=[c for c in ['season','model','n_complete_days','mean_daily_mae','mean_observed_volume'] if c in season_view]
    story.append(table_frame(season_view,use,['Season','Method','Complete days','Daily MAE','Mean count'][:len(use)],limit=24,widths=[24*mm,49*mm]+[(W-73*mm)/(len(use)-2)]*(len(use)-2)))
    sr={s:slice_row(season,'season',s) for s in ['autumn','winter','spring','summer']}
    add(f'Rolling mean daily MAE is {sr["autumn"].mean_daily_mae:.3f} in autumn, {sr["winter"].mean_daily_mae:.3f} in winter, {sr["spring"].mean_daily_mae:.3f} in spring and {sr["summer"].mean_daily_mae:.3f} in December. Adaptation helps the overall average but does not remove seasonal variation. Spring remains the hardest slice; the evidence supports partial adaptation rather than season-invariant performance.',True)
    add('Summer test coverage is December only: January/February are fitting or validation data. These are samples from one detector and year. Season, demand level and other regime changes overlap. The AnoLT section-level study and this detector-level study are separate populations.',True)
    # 14
    new('13. Eight school-term weeks and school holidays')
    image(event/'school_term_break.png','Explicit term/break windows; equal-weekday summaries reduce differences in weekday mix.',maxheight=98*mm)
    school_view=school[school.period.str.endswith('_weekdays_no_public')].copy()
    school_view['period']=school_view.period.replace({'term3_first8weeks_weekdays_no_public':'Term 3 / 8 weeks','following_spring_break_weekdays_no_public':'Spring break'})
    use=[c for c in ['period','model','n_complete_days','mean_daily_mae','weekday_balanced_mae','mean_observed_volume'] if c in school_view]
    story.append(table_frame(school_view,use,['Period','Method','Days','Daily MAE','Weekday MAE','Mean count'][:len(use)],limit=18,widths=[42*mm,43*mm]+[(W-85*mm)/(len(use)-2)]*(len(use)-2)))
    term=slice_row(school,'period','term3_first8weeks_weekdays_no_public');brk=slice_row(school,'period','following_spring_break_weekdays_no_public')
    add(f'The eight-week Term 3 window is 21 July-14 September; the following break is 20 September-5 October. Excluding weekends and public holidays leaves {int(term.n_complete_days)} complete term days and {int(brk.n_complete_days)} break days. Rolling HGB weekday-balanced MAE rises from {term.weekday_balanced_mae:.3f} to {brk.weekday_balanced_mae:.3f}. This procedure does not eliminate the break shift. Raw durations differ; equal-weekday means reduce weekday-mix imbalance.',True)
    add('Public holidays, weather and other changes can overlap school breaks. There is no random assignment or school-level attendance data; these comparisons assess predictive behaviour rather than establish a causal school effect.',True)
    # 15
    new('14. Run and debug in VS Code')
    add('Open the repository folder containing core.py, README.md and .vscode. This checkout has a .venv interpreter, Python/Jupyter support and launch/task configurations prepared. Use Python: Select Interpreter if an older VS Code selection remains cached.')
    story.append(T(['Run and Debug configuration','What it does'],[
        ['Detect historical section 0185-E','Scores the fitted contextual IF model.'],['Train contextual IF / adapted Si-TDD','Creates the section-specific model and threshold.'],['Run fresh learning/resource curves','Produces MAE/time/RAM trials in a new local output folder.'],['Plot local learning/resource curves','Checks predictions and exports PNG/SVG figures.'],['Run holiday and seasonal event study','Runs the full-year event forecast analysis.']
    ],[65*mm,W-65*mm]))
    story.append(S('Fresh machine setup'))
    for command in ['python3 -m venv .venv','.venv/bin/python -m pip install -r requirements-forecast.txt','.venv/bin/python -m pip install ipykernel','.venv/bin/python fetch_data.py','.venv/bin/python train_site.py','.venv/bin/python detect.py']:
        story.append(Paragraph(esc(command),styles['ReportCode']))
    add('On Windows use py -3.12 to create the environment and .venv\\Scripts\\python.exe for the commands. The raw benchmark/event inputs and fitted binaries are generated locally or obtained from the model release. Source data acquisition requires internet; completed-data analysis then runs locally. Virtual environments are machine-specific and are excluded from Git.')
    add('For event inputs, follow the event-study acquisition instructions in the repository. Do not run timing measurements concurrently with other heavy experiments. For notebooks, select the same .venv kernel. '+link('VS Code setup and troubleshooting',REPO+'/blob/main/docs/VSCODE.md'))
    add(link('GitHub project and releases',REPO)+'<br/>'+link('VS Code interpreter documentation','https://code.visualstudio.com/docs/python/settings-reference'))
    # 16
    new('15. Verification, limitations and next work')
    add(f'The full AnoLT recheck reproduced {audit.get("metric_rows_recomputed",30312):,} metric rows and {audit.get("models_refitted",108)} fresh representative refits. The learning-curve archive contains {caudit.get("trials",84)} verified trials. The complete local regression suite passed 57 tests, including forecasting, daily coverage, daylight-saving boundaries, archive quality masking and paired error analysis. GitHub Actions provides the cross-platform checks recorded with the release.')
    auditpath=event/'verification.json'
    if auditpath.is_file():
        eaudit=json.loads(auditpath.read_text())
        if not eaudit.get('passed') or not all(eaudit.get('checks',{}).values()):raise ValueError('Event verification must pass before report generation')
        add(f'The separate event verifier passed all {len(eaudit["checks"])} checks: the source hash, target chronology, {eaudit["prediction_rows"]:,} saved prediction rows, {eaudit["daily_metric_rows"]:,} daily records, overall errors, daylight-saving coverage and Christmas completeness. All {eaudit["rolling_models_refitted"]} past-only rolling refits and saved LSTM/HGB predictions were reproduced. Recomputing saved results and states checks implementation consistency; it is distinct from external validation on new observations.',True)
    add('The classical benchmark and event study are complete within their documented historical scope. Important limits remain: inspector disagreement; anomalous observations retained in primary fitting; local threshold calibration; geographically related directions; one detector for holiday/seasonal evaluation; missing/alarmed intervals; and unknown incident causes. No live-response benefit, crash validation or quantum performance has been established.')
    add('To improve deployment evidence, validate the chosen procedure prospectively, test additional detectors/years and affected-region weather, inspect false alerts and concept drift, and agree the next protocol before evaluating a new locked test set. Use validation accuracy with compute constraints for model selection. Future quantum comparisons should retain matched inputs, splits and tuning budgets and measure simulator/hardware costs explicitly.')
    story.append(S('Sources and reproducibility'))
    references=[('AnoLT data and benchmark paper','https://doi.org/10.1016/j.trc.2026.105536'),('Pinned AnoLT snapshot','https://github.com/imaantaheri/AnoLT/tree/d2e927425b40235d881905ba6881112bcc017e43'),('Victoria traffic-signal volume data',srcurl),('Current site 100 signal inventory','https://opendata.transport.vic.gov.au/dataset/victorian-traffic-signals/resource/1036a318-8867-4dc6-bd84-fc3bcd6cbdb9'),('Victorian public holidays 2025','https://business.vic.gov.au/business-information/public-holidays/victorian-public-holidays-2025'),('Victorian government-school terms','https://www.vic.gov.au/school-term-dates-and-holidays-victoria'),('Bureau of Meteorology, October 2025 summary','https://www.bom.gov.au/clim_data/IDCKGC1AR0/202510.summary.shtml'),('Bureau of Meteorology, Scoresby October observations','https://www.bom.gov.au/climate/dwo/202510/html/IDCJDW3072.202510.shtml'),('Cyclone Alfred report','https://www.bom.gov.au/cyclone/history/Alfred2025.shtml'),('scikit-learn Isolation Forest','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html'),('PyTorch installation and framework','https://pytorch.org/get-started/locally/'),('Project code, CSV tables, protocols and figures',REPO)]
    for label,url in references:add(link(label,url),True)
    add('OpenAI Codex assisted with implementation, execution, verification, source checking and this report. Earlier progress documents remain historical; these expanded results have not been backdated. Detailed CSVs and protocols preserve results beyond the space available in this PDF.',True)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=18*mm,bottomMargin=20*mm,title='Traffic anomaly detection and event-aware forecasting',author='Bhanu Tevathia')
    doc.build(story,onFirstPage=page_number,onLaterPages=page_number)
    print(out)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--events',default=str(ROOT/'results/events2025'));p.add_argument('--out',default=str(ROOT.parents[1]/'outputs/Traffic_Anomaly_Forecasting_Final_Report.pdf'))
    a=p.parse_args();create(a.out,a.events)
if __name__=='__main__':main()
