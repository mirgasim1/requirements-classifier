# Software requirements classifier

The application predicts Functional or Non-functional for one English software requirement at a time. It runs locally using the original TF-IDF and linear SVM model. The additional experiments compare other settings and models. They do not replace the default application.

In this project, “original dataset” means the FR_NFR dataset. “Original training” and “original test” refer to its prepared training and test subsets, as explained in the thesis.

## Run on Windows

1. Download and extract the complete repository.
2. Install Python 3.12 with the Python launcher if needed: [Python downloads for Windows](https://www.python.org/downloads/windows/).
3. Double-click `RUN.cmd`. The first run needs internet access to install packages.
4. Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) if the browser does not open automatically.
5. Keep the command window open. Press Ctrl+C in it to stop the application.

Predictions may be incorrect and should be reviewed.

## Files and experiments

| Location | Contents |
| --- | --- |
| `app.py`, `index.html`, `classifier.joblib` | Server, interface and default saved model |
| `evidence/original/`, `evidence/prepared/` | Original FR_NFR dataset, prepared data and exclusions |
| `evidence/experiment_01/` | Original split, predictions and baseline results |
| `evidence/experiment_02/` | Three-classifier comparison, SecReq preparation, case review and application checks |
| `evidence/experiment_03/` | TF-IDF and SVM tuning |
| `evidence/experiment_04/` | Frozen encoder, PROMISE+ preparation and broader training |
| `evidence/experiment_05/` | Fine-tuning, validation records, candidate models and launcher |

The root `classifier.joblib` file is the baseline model. The original data, split and cross-validation membership are shared between experiments rather than stored as repeated copies. Model settings and source hashes are kept in the recorded protocols. `evidence/check_files.py` resolves old file paths in those records to their current locations.

## Check saved results

Run these commands from the project folder after starting `RUN.cmd` once to install the application packages:

```text
.venv\Scripts\python.exe evidence\check_files.py
.venv\Scripts\python.exe verify_app.py
.venv\Scripts\python.exe evidence\experiment_02\verify_application_75.py
.venv\Scripts\python.exe evidence\experiment_02\verify_experiment.py
.venv\Scripts\python.exe evidence\experiment_03\verify_tuning.py
```

For experiments 4 and 5, install the additional packages in a separate environment:

```text
py -3.12 -m venv .experiment_env
.experiment_env\Scripts\python.exe -m pip install -r evidence\requirements_advanced.txt
.experiment_env\Scripts\python.exe evidence\experiment_04\download_model.py
.experiment_env\Scripts\python.exe evidence\experiment_05\download_model.py
.experiment_env\Scripts\python.exe evidence\experiment_04\verify_comparison.py
.experiment_env\Scripts\python.exe evidence\experiment_05\verify_finetuning.py --model-dir evidence\experiment_05\model_assets
.experiment_env\Scripts\python.exe evidence\experiment_05\verify_candidate_app.py --model-dir evidence\experiment_05\model_assets
```

The download scripts obtain the specified public base-model versions and check their hashes. The trained candidate weights are already stored in `evidence/experiment_05/results/models/`.

The saved verification JSON files record the earlier checks. Running verification again prints a fresh report and preserves an existing verification file.

## Repeat training

The training commands below write to new `results_repeat` folders within their experiment directories and refuse to overwrite them. Choose another output name for later runs. The default application model stays unchanged.

The `prepare_external.py` command checks that the PROMISE+ preparation reproduces the saved partition and exclusions. Without an `--output` argument, it does not create a new output folder.

```text
.venv\Scripts\python.exe evidence\train_classifier.py
.venv\Scripts\python.exe evidence\experiment_02\run_experiment.py --output results_repeat
.venv\Scripts\python.exe evidence\experiment_03\run_tuning.py --output results_repeat
.experiment_env\Scripts\python.exe evidence\experiment_04\prepare_external.py
.experiment_env\Scripts\python.exe evidence\experiment_04\run_comparison.py --output results_repeat
.experiment_env\Scripts\python.exe evidence\experiment_05\run_finetuning.py --model-dir evidence\experiment_05\model_assets --output results_repeat
```

The first command saves the repeated baseline experiment in `evidence/experiment_01/results_repeat/`.

To verify a repeat run of experiments 2–5, add `--results results_repeat` to the corresponding experiment verification command. If another output name was used, use that name instead.

To demonstrate a saved fine-tuned candidate using the existing interface:

```text
.experiment_env\Scripts\python.exe evidence\experiment_05\run_candidate_app.py --condition original_only
```

Use `--condition augmented` for the broader-training candidate. Both candidates use port 8001 by default and do not change `RUN.cmd` or the root model.

## Data and sources

- [FR_NFR_dataset](https://doi.org/10.17632/4ysx9fyzv4.1), Sonali and Thamada (2024).
- [Curated workbook supplying the SecReq CPN/ePurse requirements](https://doi.org/10.5281/zenodo.15082900), Xanthopoulou et al. (2025).
- [PROMISE+](https://doi.org/10.5281/zenodo.12805484), Carvalho da Silva et al. (2024).
- [Arctic-Embed technical report](https://doi.org/10.48550/arXiv.2405.05374), Merrick et al. (2024).
- [Base model and model card](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs).

The datasets are provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Requirement texts and retained labels were preserved. Exclusions and the PROMISE+ binary label mapping are recorded with the data.

The base model uses the Apache 2.0 licence. Its licence and derivation notice are retained in experiment 5. Further source details and model revisions are recorded in the experiments’ source and protocol JSON files.

## Evaluation notes

The original dataset contains 5,936 prepared requirements, split into 4,748 training and 1,188 test records. SecReq contains 253 selected requirements. PROMISE+ supplies 1,496 additional training requirements and 802 requirements from reserved projects.

Scores measure agreement with the supplied reference labels. They do not establish that every reference label is correct. SecReq is security-focused, and performance on requirements from other sources remains limited. The case-review interpretations are tentative and do not replace the reference labels.

PROMISE+ preparation replays the recorded near-match pairs and checks them against the source text. The original top-ten candidate search did not specify how candidates with equal counts were ordered. The preparation script therefore keeps the verified exclusions and the fixed evaluation sample.

Overlap checks do not guarantee complete independence between the underlying projects. Broader-training models have seen other PROMISE+ projects, while SecReq was not used for training. Later experiments were exploratory because earlier test results were already known.

Application checks cover the tested local requests. They do not establish usability with participants, performance during concurrent use or measured time savings.

This repository uses a simplified file layout. Repeated input copies, separate READMEs and repeated file-preservation lists were consolidated. The recorded protocols, data, model weights, predictions and numerical results were retained. `evidence/file_hashes.json` and `evidence/check_files.py` check the retained fixed files in this layout.
