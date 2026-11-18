# Trained example detectors

Two fitted examples for AnoLT section **0185-E** are distributed as a GitHub Release asset rather than committed binaries:

- `models/0185-E/detector.joblib`: contextual, full-window Isolation Forest; 100 trees, max samples 256, seed 42.
- `models/0185-E_sitdd/detector.joblib`: adapted same-slot kth-distance detector, k = 3.

Both fit the first 70% of the section's ordered observations and calibrate thresholds against inspector set 1 on the next 15%, leaving the last 15% for historical evaluation. The source is the pinned AnoLT snapshot recorded in `protocol.json`. Metadata files record the example's evaluation; the full research tables report section-averaged results across 117 final sections and should not be confused with this example's score.

## Use

Install `requirements.txt` (including scikit-learn 1.9.1), acquire data with `python fetch_data.py`, and extract the release archive into the repository root. Then run:

```sh
python detect.py
python detect.py --model models/0185-E_sitdd/detector.joblib --history data/0185-E.csv
```

A window contains the current observed count and seven preceding consecutive quarter-hour counts. Detection occurs after the interval is observed; it is not forecasting. Scores are relative measures, not probabilities. Thresholds were calibrated using labelled historical data. There is no live feed or verified crash detection.

These examples apply to their fitted section. Train and calibrate separately for another section; no shared-weight or zero-shot generalisation claim is made. The default IF representation was chosen among primary IF candidates using development validation; the adapted nearest-neighbour baseline is stronger in the broader benchmark.

`joblib` serialization is version-sensitive. The release manifest records file hashes and the source revision. Use the published artifacts or models you fit yourself. Raw CSVs and research/course documents are not included in this model archive.
