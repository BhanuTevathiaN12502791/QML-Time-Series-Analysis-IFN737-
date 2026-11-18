# Implementation alignment with the final research plan

The historical final 2C report and aligned 2B presentation supplied the research plan. Those private course documents are not republished in this source repository. This file maps their planned classical work to the implementation and archived derived results; it does not imply these experiments had been completed when those documents were written.

| Research-plan requirement | Implemented evidence |
|---|---|
| Labelled AnoLT volume task on individual road sections | `data/`, pinned hashes in `protocol.json`, strict schema checks in `core.py` |
| Preserve valid zero counts and exclude gap-crossing trailing windows | `core.windows`, source audit and pipeline tests |
| RQ1: effect of weekday/time-of-day context | Matched raw/context IF variants; global/time-of-week MAD comparison; `paired_effects.csv` |
| RQ2: hold detector fixed while testing PCA | Shared selected IF setting, matched seeds/folds; full versus PCA 2, 4 and 6 dimensions |
| Development/final-site separation | Six original pilot + 24 seeded additional development sections, 117 final sections; frozen allocation and digest |
| Three expanding chronological folds | Fit/validation/test endpoints 40/55/70%, 55/70/85%, 70/85/100%; saved splits |
| Training-only preprocessing | Context, standardisation, PCA and IF fit only before the fold's fit boundary |
| Validation-only threshold selection | Set-1 validation F1, grouped score ties favour fewer alerts; unchanged thresholds for both test annotations |
| Reproduce selected paper baseline | Si-TDD-style same-slot kth distance implemented independently, source definition checked; chronological calibration differs, explicitly adapted rather than claiming original score replication |
| Pointwise detection metrics, no point adjustment | Saved confusion counts, F1, precision, recall, AP, FPR and false alerts per observed day |
| Both inspector labels and uncertainty | Set-2 metrics with unchanged thresholds, 1,000 paired site-bootstrap draws and shared-ID-prefix sensitivity |
| Robustness across periods/seeds | All final sections across three folds and IF seeds 17, 42, 91; fold/seed tables |
| Training contamination sensitivity | Label-informed normal-only training, seed 42, no new global tuning |
| Runtime/memory and cost of PCA | Same-section/fold fresh-process serial benchmark, model bytes, fit/scoring time and peak RSS |
| Error analysis | Zero/low/high counts, overnight, annotation disagreement and gap proximity; descriptive episode summaries |
| Reusable model and reproducibility | IF and adapted nearest-neighbour training/inference CLIs, notebook, frozen protocol, source fetcher, synthetic tests and full-output numerical verification; weights are trained locally |
| Spatial/crash/QML work later | Excluded from the present classical implementation and explicitly recorded as unevaluated extensions |

The earlier plan left forest hyperparameters, seed values and exact additional-site allocation unspecified. These details were frozen before this expanded experiment. One shared `max_samples` was chosen from [128, 256] using development validation averages across the four primary IF representations. All IF models use 100 trees. PCA is standardised full-SVD compression without whitening; two dimensions are primary, four and six are sensitivity checks. Forest settings and data splits remain matched across representations.

Final sections are excluded from global selection but still use local earlier observations and local validation annotations for fitting/calibration. This evaluates transfer of a fixed procedure, not a universal set of weights or geographic independence. Opposite-direction sections with a common ID prefix may span development and final groups; the report acknowledges this limitation.

The trained default IF example was chosen among primary IF representations on development validation data. Baselines remain visible in the complete comparison, and an adapted nearest-neighbour checkpoint can also be trained locally. Binary weights are not committed. No claim that the IF example is automatically the strongest method is made.

Read `report.html` for the executed results. `results/verification.json` preserves the historical full local audit. Re-executing `verify_results.py` requires the full locally reproduced output archive, which is excluded from Git. `README.md` explains changing data, training another section and rerunning the experiment.
