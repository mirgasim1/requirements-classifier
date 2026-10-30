# Software requirements classifier

This application classifies one English software requirement as Functional or Non-functional. It runs locally using a saved TF-IDF vectorizer and linear SVM.

The thesis experiments compare classifiers, model settings and training data. The original SVM remains the default application model. Fine-tuned candidates can be run separately using the same interface.

## Running on Windows

1. Select **Code → Download ZIP** and extract the entire ZIP.
2. Install [Python 3.12](https://www.python.org/downloads/windows/) if needed. Include the Python launcher.
3. Double-click `RUN.cmd`. The first run needs an internet connection to install packages.
4. The browser opens at http://127.0.0.1:8000/. Enter a requirement and click **Classify**.
5. Keep the command window open. Press **Ctrl+C** in it to stop the application.

If the browser does not open automatically, open the address manually.

Empty or spaces-only input is rejected. Submitted text remains visible for editing. Predictions may be incorrect and should be reviewed.

## Main files

| File or folder | Purpose |
| --- | --- |
| `app.py` | Python server, input validation and prediction handling |
| `index.html` | Browser interface |
| `classifier.joblib` | Default saved TF-IDF and SVM model |
| `requirements.txt` | Packages needed for the default application |
| `verify_app.py` | Checks selected application functions |
| `evidence/` | Source data, preparation records, experiments and verification files |

`evidence/requirements_advanced.txt` lists the additional packages for experiments 4 and 5. `evidence/check_files.py` uses `file_hashes.json` to check that the recorded files are present and unchanged.

## Experiments and saved results

All experiment folders are inside `evidence/`.

| Folder | Work completed |
| --- | --- |
| `experiment_01` | Original TF-IDF and SVM experiment |
| `experiment_02` | SVM, logistic regression and Naive Bayes comparison; SecReq preparation, case review and application checks |
| `experiment_03` | TF-IDF features, class weighting and SVM tuning |
| `experiment_04` | Frozen pretrained encoder and broader training using additional PROMISE+ requirements |
| `experiment_05` | Supervised fine-tuning, validation-based training duration and candidate application checks |

The original results are in `experiment_01/results.json`. Experiments 2–5 have a `results` folder containing `model_comparison.csv`, `results.json` and individual predictions. Settings, data preparation and verification records are kept with the experiments. Metric values in the CSV files are fractions between 0 and 1.

Saved results can be inspected without repeating training.

## Data and model sources

- [FR_NFR_dataset — Sonali and Thamada (2024)](https://doi.org/10.17632/4ysx9fyzv4.1): 5,936 prepared requirements, divided into 4,748 training and 1,188 test records.
- [Curated requirements workbook — Xanthopoulou et al. (2025)](https://doi.org/10.5281/zenodo.15082900): supplies the SecReq CPN and ePurse requirements. The prepared evaluation subset contains 253 records.
- [PROMISE+ — Carvalho da Silva et al. (2024)](https://doi.org/10.5281/zenodo.12805484): supplies 1,496 additional training requirements and 802 evaluation requirements from reserved projects.

These dataset releases use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preparation records document exclusions, retained labels, PROMISE+ binary label mapping and subset assignments.

Experiments 4 and 5 use [Snowflake Arctic-Embed XS](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs), described in the [Arctic-Embed technical report](https://doi.org/10.48550/arXiv.2405.05374). The base model uses Apache 2.0. Its licence and the notice describing the fine-tuned derivatives are retained in `experiment_05`. The fixed model revision and download hashes are recorded in the source files.

## Running a fine-tuned candidate

This is optional. Open a terminal in the project folder containing `app.py`.

Check that these files are inside `evidence/experiment_05/results/models/`:

- `original_only.safetensors`
- `augmented.safetensors`

Install the additional packages and download the required public base-model files:

```text
py -3.12 -m venv .experiment_env
.experiment_env\Scripts\python.exe -m pip install -r evidence\requirements_advanced.txt
.experiment_env\Scripts\python.exe evidence\experiment_05\download_model.py
```

Start the original-training candidate:

```text
.experiment_env\Scripts\python.exe evidence\experiment_05\run_candidate_app.py --condition original_only
```

The candidate opens at http://127.0.0.1:8001/. Stop it with **Ctrl+C** before running the broader-training candidate:

```text
.experiment_env\Scripts\python.exe evidence\experiment_05\run_candidate_app.py --condition augmented
```

These commands load the saved candidates without retraining or replacing the default application model.

## Checking the recorded work

After running `RUN.cmd` once, these commands check the files, default application and experiments 2–3:

```text
.venv\Scripts\python.exe evidence\check_files.py
.venv\Scripts\python.exe verify_app.py
.venv\Scripts\python.exe evidence\experiment_02\verify_application_75.py
.venv\Scripts\python.exe evidence\experiment_02\verify_experiment.py
.venv\Scripts\python.exe evidence\experiment_03\verify_tuning.py
```

After completing the additional setup above, check experiments 4–5:

```text
.experiment_env\Scripts\python.exe evidence\experiment_04\download_model.py
.experiment_env\Scripts\python.exe evidence\experiment_04\verify_comparison.py
.experiment_env\Scripts\python.exe evidence\experiment_05\verify_finetuning.py --model-dir evidence\experiment_05\model_assets
.experiment_env\Scripts\python.exe evidence\experiment_05\verify_candidate_app.py --model-dir evidence\experiment_05\model_assets
```

## Repeating training

Run the following commands after installing the relevant packages and downloading the base-model files. Training outputs go into new `results_repeat` folders. Existing output folders are not overwritten; choose another output name for later runs.

```text
.venv\Scripts\python.exe evidence\train_classifier.py
.venv\Scripts\python.exe evidence\experiment_02\run_experiment.py --output results_repeat
.venv\Scripts\python.exe evidence\experiment_03\run_tuning.py --output results_repeat
.experiment_env\Scripts\python.exe evidence\experiment_04\prepare_external.py
.experiment_env\Scripts\python.exe evidence\experiment_04\run_comparison.py --output results_repeat
.experiment_env\Scripts\python.exe evidence\experiment_05\run_finetuning.py --model-dir evidence\experiment_05\model_assets --output results_repeat
```

## Interpreting the results

Scores measure agreement with the supplied reference labels, which have not all been independently checked. SecReq covers two security-focused specifications. Broader-training models use other PROMISE+ projects, so their reserved-project evaluation is not evaluation on a completely unseen source.

Later experiments were exploratory because earlier test results were already known. Performance varies between evaluation groups, and reliable classification across new projects has not been established. Application checks cover selected local functions; human usability and concurrent-use performance were not evaluated.
