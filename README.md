# Software requirements classifier

This application predicts whether an English software requirement is Functional or Non-functional. It runs locally using a saved TF-IDF and linear SVM model.

Other models were tested in the thesis experiments. They are saved separately and do not replace the default model.

## Run the application

1. Click **Code → Download ZIP** and extract the ZIP.
2. Install [Python 3.12](https://www.python.org/downloads/windows/) if needed. Include the Python launcher.
3. Double-click `RUN.cmd`. The first run needs internet access to install packages.
4. Enter a requirement in the browser and click **Classify**.
5. Keep the command window open. Press **Ctrl+C** to stop.

If the browser does not open, go to http://127.0.0.1:8000/.

Empty or spaces-only input is rejected. The entered text stays visible so it can be edited. Predictions may be incorrect and should be checked before use.

## Files

- `app.py`: Python server and prediction handling.
- `index.html`: browser interface.
- `classifier.joblib`: default saved model.
- `requirements.txt`: application packages.
- `verify_app.py`: application checks.
- `evidence/`: datasets, preparation records and experiments.

The experiment folders contain:

| Folder | Contents |
| --- | --- |
| `experiment_01` | Original TF-IDF and SVM results |
| `experiment_02` | Comparison of three classifiers, SecReq preparation and case review |
| `experiment_03` | TF-IDF and SVM tuning |
| `experiment_04` | Frozen encoder and additional PROMISE+ training data |
| `experiment_05` | Fine-tuned models, training records and candidate application checks |

The baseline results are in `experiment_01/results.json`. In experiments 2–5, open `results/model_comparison.csv` for the scores. The folders also contain individual predictions, settings and verification records. Metric values in the CSV files range from 0 to 1.

## Data and sources

- [FR_NFR_dataset — Sonali and Thamada (2024)](https://doi.org/10.17632/4ysx9fyzv4.1): 5,936 prepared requirements, with 4,748 used for training and 1,188 for testing.
- [Requirements workbook — Xanthopoulou et al. (2025)](https://doi.org/10.5281/zenodo.15082900): provides the SecReq requirements. The evaluation uses 253 selected records.
- [PROMISE+ — Carvalho da Silva et al. (2024)](https://doi.org/10.5281/zenodo.12805484): provides 1,496 additional training records and 802 evaluation records from reserved projects.

The dataset releases use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Exclusions, label mapping and data splits are recorded in the experiment files.

Experiments 4 and 5 use [Snowflake Arctic-Embed XS](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs). The model revision and download hashes are recorded in the project. Its Apache 2.0 licence and the fine-tuning notice are kept in `experiment_05`.

## Run a fine-tuned model

This is optional. Open a terminal in the project folder containing `app.py`.

Both `original_only.safetensors` and `augmented.safetensors` must be in:

```text
evidence/experiment_05/results/models/
```

Run:

```text
py -3.12 -m venv .experiment_env
.experiment_env\Scripts\python.exe -m pip install -r evidence\requirements_advanced.txt
.experiment_env\Scripts\python.exe evidence\experiment_05\download_model.py
.experiment_env\Scripts\python.exe evidence\experiment_05\run_candidate_app.py --condition original_only
```

The setup needs internet access. The candidate opens at http://127.0.0.1:8001/.

To use the model trained with additional data, stop the first candidate with **Ctrl+C**, then run:

```text
.experiment_env\Scripts\python.exe evidence\experiment_05\run_candidate_app.py --condition augmented
```

## Check or repeat the work

After running `RUN.cmd` once, check the recorded files and default application with:

```text
.venv\Scripts\python.exe evidence\check_files.py
.venv\Scripts\python.exe verify_app.py
```

`check_files.py` uses `evidence/file_hashes.json` to check for missing or changed files. Each additional experiment also contains verification scripts and saved check results.

To repeat training, use the relevant commands below. Experiments 4 and 5 need the additional setup above.

```text
.venv\Scripts\python.exe evidence\train_classifier.py
.venv\Scripts\python.exe evidence\experiment_02\run_experiment.py --output results_repeat
.venv\Scripts\python.exe evidence\experiment_03\run_tuning.py --output results_repeat
.experiment_env\Scripts\python.exe evidence\experiment_04\download_model.py
.experiment_env\Scripts\python.exe evidence\experiment_04\prepare_external.py
.experiment_env\Scripts\python.exe evidence\experiment_04\run_comparison.py --output results_repeat
.experiment_env\Scripts\python.exe evidence\experiment_05\run_finetuning.py --model-dir evidence\experiment_05\model_assets --output results_repeat
```

Training results go into new `results_repeat` folders. Existing output folders are not overwritten. The default application model stays unchanged.

## Limitations

The scores measure agreement with the supplied dataset labels, which have not all been checked manually. SecReq is security-focused. Broader-training models use other PROMISE+ projects, so their PROMISE+ evaluation is on reserved projects within the same source.

Later experiments were exploratory because earlier test results were already known. Performance on new requirements varies. Application checks cover selected local functions; usability and concurrent-use performance were not evaluated.
