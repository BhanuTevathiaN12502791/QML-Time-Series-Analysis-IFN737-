# Sources and attribution

The implementation and derived evaluation outputs in this repository were prepared for Bhanu Tevathia's IFN737 research project with OpenAI Codex assistance. Quantum integration is a future phase.

## AnoLT data and reference algorithm

- Repository: https://github.com/imaantaheri/AnoLT
- Pinned snapshot: `d2e927425b40235d881905ba6881112bcc017e43`
- Reference scoring definition: the snapshot's `data_to_score/Si-TDD.py`
- Paper: Taheri Sarteshnizi et al. (2026), *On the applicability of time series anomaly detection methods to real-world traffic volume data*, Transportation Research Part C, 184, 105536. https://doi.org/10.1016/j.trc.2026.105536

Raw dataset files, downloaded research papers, historical assignment documents and the upstream Si-TDD script are not redistributed in this Git repository. `fetch_data.py` downloads the frozen CSV snapshot directly from upstream and verifies its hashes. The upstream materials retain their original terms; no new licence over them is asserted here.

The adapted Si-TDD-style implementation uses same-weekday/time-slot kth absolute distance. This project's split/calibration protocol and fallback rules differ from the original reported experiment, so its scores are not presented as an exact replication of published paper scores.

## Dependencies

The implementation uses NumPy, pandas, scikit-learn, joblib, Matplotlib, psutil and threadpoolctl under their respective upstream terms. Their source and documentation are available through the package metadata and official project sites. No dependency source code is vendored here.
