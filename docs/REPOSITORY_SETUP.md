# Repository distribution

The source checkout contains the classical implementation, frozen protocol/source manifests, compact derived metrics, figures and the historical full-output audit record. It excludes raw CSV data, per-fold score arrays, generated cache directories, fitted binaries and historical course documents from Git history.

The full local experiment was rechecked on 19 November 2026: 30,312 metric rows and 108 model refits reproduced the recorded results. The repository packaging additionally:

- tests with synthetic data so CI does not require downloading the benchmark;
- creates fresh result directories as needed;
- keeps frozen selected parameters strict while allowing negligible floating-point differences in diagnostic means;
- supplies a guarded runner with an explicit `--fetch` option and a missing-data error;
- provides a downloadable model release, plus checksum verification, for cross-device recovery.

These packaging changes do not change feature/scoring mathematics, the original section allocation or headline metrics. The saved historical verification record describes the full local outputs, not an audit of bulk artifacts present in a reduced clone. To regenerate those artifacts, fetch data and run `python run_all.py`.

## Local source-checkout checks

The repository was checked with Python 3.13.5 on macOS:

- acquired the pinned upstream snapshot and verified all 147 CSV hashes;
- ran 30 regression tests with both data and model directories absent;
- fitted and scored IF examples for 0185-E and 2414-E;
- fitted the adapted nearest-neighbour example for 0185-E;
- checked that input/protocol/source-scoring definitions match the completed experiment;
- checked repository paths, ignored files and publication contents.

GitHub Actions separately runs synthetic tests on Linux (Python 3.12 and 3.13) and Windows (Python 3.12). Its live status is shown in the repository's Actions tab.
