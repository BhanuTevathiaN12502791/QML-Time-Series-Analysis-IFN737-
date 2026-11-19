# Run the project in VS Code

Open the **repository folder**, not its parent folder or a single Python file. The folder should contain `core.py`, `README.md`, `.vscode/` and `requirements.txt`.

## This Mac

VS Code 1.137.0 (Apple silicon) and the Python, Python Debugger, Pylance and Jupyter extensions were found installed during setup. The `code` shell command is not currently on PATH; use **File → Open Folder…** in VS Code, or run this from the repository folder:

```sh
open -a "Visual Studio Code" .
```

The prepared Mac checkout now has a working `.venv` link to the existing project environment, with forecasting, PDF and notebook dependencies installed. The repository has been opened in VS Code. This is a local setup; the environment link itself is excluded from Git. Fresh clones on another computer must create their own environment as described below.

The project expects its environment in `.venv/`. After the environment has been prepared, open the Command Palette and run **Python: Select Interpreter**. Choose the interpreter inside this project's `.venv` folder. On macOS/Linux it is `.venv/bin/python`; on Windows it is `.venv\Scripts\python.exe`.

Workspace settings point to the portable `.venv` folder. If VS Code previously selected another interpreter, choose the correct one manually: changing `python.defaultInterpreterPath` does not replace an already selected interpreter.

## Fresh clone setup

Use Python 3.12 or 3.13. Open **Terminal → New Terminal** in the repository folder.

macOS/Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-forecast.txt
.venv/bin/python -m pip install -r requirements-report.txt ipykernel debugpy
.venv/bin/python fetch_data.py
.venv/bin/python train_site.py
.venv/bin/python detect.py
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-forecast.txt
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-report.txt ipykernel debugpy
& ".\.venv\Scripts\python.exe" fetch_data.py
& ".\.venv\Scripts\python.exe" train_site.py
& ".\.venv\Scripts\python.exe" detect.py
```

The forecasting requirements include the base requirements and add PyTorch. They are needed for the companion MAE/resource curves and any forecast-based event study. The anomaly detector can run with only `requirements.txt`. Installing packages and fetching data need internet access. Input data and fitted models are intentionally excluded from Git; fetch/train them or obtain the published model release before inference.

For this prepared checkout, do not recreate an existing environment unnecessarily. The **Python: Create .venv (fresh clone)** task is intended for a new clone. A virtual environment is machine-specific and should not be committed or copied between operating systems.

## Run and debug

Open **Run and Debug** and choose a configuration, then press **F5**:

| Configuration | Purpose | Prerequisite |
|---|---|---|
| Detect historical section 0185-E | Score the supplied historical series with contextual IF | Data and fitted `models/0185-E/detector.joblib` |
| Train contextual IF for 0185-E | Fit/calibrate the example detector | AnoLT data |
| Train adapted Si-TDD for 0185-E | Fit/calibrate the nearest-neighbour example | AnoLT data |
| Detect with adapted Si-TDD | Score with the baseline checkpoint | Its fitted checkpoint |
| Run fresh learning/resource curves | Measure MAE, time, RAM and detector metrics in a local output folder | Data and optional forecast dependencies |
| Plot local learning/resource curves | Render and verify the completed local measurements | Completed local curve trials |
| Run holiday and seasonal event study | Run the project's event-analysis entry point | Extracted 2025 event data and forecasting dependencies |
| Build final results PDF | Write `results/final_report.pdf` | Report dependencies and completed analysis outputs |

All configurations use the selected interpreter, the repository as working directory and single-thread CPU environment variables. They do not connect to a live feed. Detection happens after an observed interval. An anomaly flag does not establish a crash cause.

The event configuration calls `event_study.py` with its default arguments. Its source/data documentation defines the event-study defaults; the launch file does not invent holiday/weather labels.

## Terminal tasks

Use **Terminal → Run Task…** for environment setup, dependency installation, data acquisition, training, detection, tests, learning curves or event analysis. Use **Tasks: Run Test Task** for the synthetic regression suite.

The configured task commands select `.venv/bin/python` on macOS/Linux and `.venv\Scripts\python.exe` on Windows. They avoid global-package installation and do not require changing PowerShell execution policy.

The full benchmark and learning-curve tasks can take several minutes or longer. Run cost measurements serially with other heavy work stopped. Measured time/RAM depends on hardware and loaded libraries; the provided experimental report records its measurement definitions.

## Holiday/event data and final PDF

For a quick replay, download `Melbourne_2025_Inputs_and_Models.zip` from [release v1.1.0](https://github.com/BhanuTevathiaN12502791/QML-Time-Series-Analysis-IFN737-/releases/tag/v1.1.0) and extract it into the repository root. It supplies the small extracted traffic series and fitted event states. The committed result tables and compressed predictions then support `python verify_event_study.py`, and the report can be built without downloading twelve full archives. Use the recorded `results/events2025/environment.json` versions when replaying saved estimators. The existing prepared checkout already has these files.

The annual event study uses a separate 2025 SCATS traffic series; it does not extend the labelled AnoLT data by inventing Christmas observations. In a fresh clone, run **Data: Fetch 2025 holiday/event traffic** once, or use:

```sh
.venv/bin/python fetch_event_data.py
```

This downloads approximately **1.6 GB** of verified source archives across the full year, so allow time and internet bandwidth. Each archive is read as a stream to extract only the study detectors; all road data are not expanded to disk. The default output is `data_events/` and the download cache is `work/event_cache/`. Newly downloaded archives are removed after extraction unless `--keep-archives` is given. Existing cached archives are retained. Data and caches are excluded from Git.

After the data are ready, use **Events: Run holiday and seasonal study**. The event-study documentation explains the location, public-holiday/school-calendar/weather definitions and evaluation boundaries. A cyclone affecting another region must not be treated as a local Melbourne traffic event.

Use **Python: Install report and notebook dependencies** for the report/Jupyter tools, then **Report: Build final PDF** or the **Build final results PDF** debug configuration. Both pass `--out` with the portable destination `${workspaceFolder}/results/final_report.pdf`. The report builder consumes completed analysis outputs; it is not a substitute for running missing experiments.

```sh
.venv/bin/python build_final_report.py --out results/final_report.pdf
```

On Windows, use `.\.venv\Scripts\python.exe` for the same script arguments.

## Tests and notebook

The Testing view discovers `unittest` files matching `test_*.py`. The baseline synthetic suite does not require downloading AnoLT. A forecasting smoke test runs when PyTorch is installed. Select the same `.venv` interpreter in the Testing view.

Open `experiment.ipynb`, choose the project's `.venv` kernel, and install the recommended Jupyter extension if prompted. If Jupyter reports a missing kernel package, install `ipykernel` into the same environment:

```sh
.venv/bin/python -m pip install ipykernel
```

On Windows use `.\.venv\Scripts\python.exe -m pip install ipykernel`. Viewing committed figures and CSVs needs no notebook execution. Cells that read input data, fit models or recompute a full audit need their documented local inputs.

## Troubleshooting

- **ModuleNotFoundError:** verify the selected interpreter, then install requirements using that interpreter's `-m pip`.
- **Missing data/model:** run the acquisition/training tasks or download the model release; Git does not contain bulk generated inputs.
- **A different section is rejected:** train/calibrate a separate section model and use the matching history. For generically named histories, pass an explicit `--site` assertion.
- **Cached experiment mismatch:** preserve the old results and choose a fresh output folder; do not relabel old outputs as a new experiment.
- **Interpreter selection stays global:** run **Python: Select Interpreter** again and select the project's `.venv` explicitly.

## Official VS Code references

- [Python interpreter settings](https://code.visualstudio.com/docs/python/settings-reference)
- [Python environments](https://code.visualstudio.com/docs/python/environments)
- [Python debugging](https://code.visualstudio.com/docs/python/debugging)
- [Tasks](https://code.visualstudio.com/docs/debugtest/tasks)
