"""Post-hoc paired error/cost analysis of the completed event study.

No models are selected or fitted here. Weekly clusters retain within-week paired
errors; intervals include incomplete days, matching the overall test estimand.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_weekly_interval(left, right, draws=10000, seed=42):
    """Difference in pooled MAE, resampling paired Monday-based calendar weeks."""
    keys = ['timestamp_utc', 'date']
    p = left[keys + ['actual', 'prediction']].merge(
        right[keys + ['actual', 'prediction']], on=keys,
        suffixes=('_left', '_right'), validate='one_to_one')
    if len(p) != len(left) or len(p) != len(right):
        raise ValueError('Paired models must have exactly the same targets')
    if not np.array_equal(p.actual_left, p.actual_right):
        raise ValueError('Paired actuals differ')
    date = pd.to_datetime(p.date)
    p['week'] = date - pd.to_timedelta(date.dt.day_of_week, unit='D')
    p['difference'] = abs(p.prediction_left-p.actual_left) - abs(p.prediction_right-p.actual_right)
    clusters = p.groupby('week').difference.agg(['sum', 'count'])
    if len(clusters) < 2:
        raise ValueError('At least two weekly clusters required')
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(clusters), size=(draws, len(clusters)))
    means = clusters['sum'].to_numpy()[picks].sum(axis=1) / clusters['count'].to_numpy()[picks].sum(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return dict(delta_mae=float(p.difference.mean()), ci95_low=float(low), ci95_high=float(high),
                n_intervals=len(p), n_week_clusters=len(clusters))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('results/events2025'))
    p.add_argument('--draws', type=int, default=10000)
    a = p.parse_args()
    path = a.out/'predictions.csv'
    predictions = pd.read_csv(path if path.exists() else path.with_suffix('.csv.gz'))
    predictions = predictions[predictions.split == 'test']
    rows = []
    comparisons = [
        ('hgb_calendar_rolling56', 'hgb_calendar_frozen', 'rolling history versus frozen history; same calendar features'),
        ('hgb_calendar_frozen', 'hgb_lags_frozen', 'matched frozen calendar-feature ablation'),
        ('hgb_calendar_rolling56', 'hgb_lags_frozen', 'combined calendar and rolling-history change'),
        ('hgb_calendar_rolling56', 'persistence', 'rolling model versus persistence'),
        ('lstm_frozen', 'persistence', 'frozen LSTM versus persistence'),
    ]
    for left, right, meaning in comparisons:
        values = paired_weekly_interval(predictions[predictions.model == left],
                                        predictions[predictions.model == right], a.draws)
        rows.append(dict(left=left, right=right, interpretation=meaning, **values))
    pd.DataFrame(rows).to_csv(a.out/'paired_week_bootstrap.csv', index=False)
    overall = pd.read_csv(a.out/'overall_metrics.csv')
    timing = pd.read_csv(a.out/'model_timing.csv')
    cost = overall.merge(timing, on='model', validate='one_to_one')
    cost['pareto_error_fit_time'] = [not (((cost.mae <= row.mae) & (cost.fit_seconds <= row.fit_seconds)) &
        ((cost.mae < row.mae) | (cost.fit_seconds < row.fit_seconds))).any() for row in cost.itertuples()]
    cost.to_csv(a.out/'error_cost_tradeoff.csv', index=False)
    notes = dict(draws=a.draws, seed=42, differences='left minus right; negative favors left',
        estimand='Pooled 15-minute test MAE on the common targets, including partial days',
        method='Paired calendar-week cluster percentile bootstrap, Monday-based weeks, 10000 resamples by default',
        limitations=['Descriptive uncertainty conditional on one detector/year and fitted models.',
                    'Weekly clusters do not establish independence across seasons or consecutive weeks.',
                    'No correction for multiple post-hoc comparisons; no model was tuned using these intervals.',
                    'Fit times are a single machine run; rolling sums 11 refits, LSTM includes validation early stopping.',
                    'Pareto flags describe observed error/cost and do not constitute validation-based model selection.'])
    (a.out/'tradeoff_protocol.json').write_text(json.dumps(notes, indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
