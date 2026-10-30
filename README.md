# Software requirements classifier

The application predicts Functional or Non-functional for one English software requirement. It runs locally using the original TF-IDF and linear SVM model. The additional experiments compare other settings and models; they do not replace the default application.

## Run on Windows

1. Download and extract the complete repository.
2. Install Python 3.12, including the Python launcher, if needed: https://www.python.org/downloads/windows/
3. Double-click `RUN.cmd`. The first run needs internet access to install packages.
4. Open http://127.0.0.1:8000/ if the browser does not open automatically.
5. Keep the command window open. Press Ctrl+C in it to stop.

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

The root model is the baseline. Original data, split and cross-validation membership are shared rather than copied between experiments. Model settings and source hashes remain in the recorded protocols. Old filenames in those historical records are resolved by `evidence/check_files.py`.

## Check saved results

Run these commands from the project folder after starting `RUN.cmd` once:

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

The download scripts obtain the pinned public base model and check its hashes. The trained candidate weights are already in `experiment_05/results/models/`. Saved verification JSONs record the original checks. Rechecking prints a fresh report and preserves an existing verification file.

## Repeat training

Each command writes to a new `results_repeat` folder and refuses to overwrite it. Choose another output name for subsequent runs. The default application model stays unchanged.

```text
.venv\Scripts\python.exe evidence\train_classifier.py
.venv\Scripts\python.exe evidence\experiment_02\run_experiment.py --output results_repeat
.venv\Scripts\python.exe evidence\experiment_03\run_tuning.py --output results_repeat
.experiment_env\Scripts\python.exe evidence\experiment_04\prepare_external.py
.experiment_env\Scripts\python.exe evidence\experiment_04\run_comparison.py --output results_repeat
.experiment_env\Scripts\python.exe evidence\experiment_05\run_finetuning.py --model-dir evidence\experiment_05\model_assets --output results_repeat
```

To verify a repeat run, add `--results results_repeat` to its verification command. To demonstrate a saved fine-tuned candidate using the existing interface:

```text
.experiment_env\Scripts\python.exe evidence\experiment_05\run_candidate_app.py --condition original_only
```

Use `--condition augmented` for the broader-training candidate. Candidates use port 8001 and do not change `RUN.cmd` or the root model.

## Data and sources

- FR_NFR_dataset, Sonali and Thamada (2024): https://doi.org/10.17632/4ysx9fyzv4.1
- Curated workbook supplying the SecReq CPN/ePurse requirements, Xanthopoulou et al. (2025): https://doi.org/10.5281/zenodo.15082900
- PROMISE+, Carvalho da Silva et al. (2024): https://doi.org/10.5281/zenodo.12805484
- Arctic-Embed technical report, Merrick et al. (2024): https://doi.org/10.48550/arXiv.2405.05374
- Pinned model and model card: https://huggingface.co/Snowflake/snowflake-arctic-embed-xs

The datasets are provided under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/. Source texts and retained labels were preserved. Exclusions and PROMISE+ binary mapping are recorded with the data. The base model uses Apache 2.0; its licence and derivation notice are retained in experiment 5. Source details and model revisions are in the experiments' source and protocol JSONs.

## Evaluation notes

The original dataset has 5,936 prepared requirements, split into 4,748 training and 1,188 test records. SecReq contains 253 selected requirements. PROMISE+ supplies 1,496 additional training requirements and 802 requirements from reserved projects. Scores measure agreement with supplied labels, not independently verified correctness. SecReq is security-focused; cross-source performance remains limited. The case review is provisional and is not a replacement gold-labelled dataset.

PROMISE+ preparation replays the recorded, individually verified near-match pairs. The original top-ten candidate shortlist did not specify equal-count ordering; the reproducer retains the fixed exclusions instead of changing the evaluation sample. Overlap screening does not establish complete underlying project independence. Broader-training models have seen other PROMISE+ projects. Later experiments were exploratory because earlier test outcomes were known. No human-usability, concurrent-load or measured time-saving claim is made.

This copy has a simplified file layout: repeated input copies, separate READMEs and repeated file-preservation lists were consolidated. The recorded protocols, data, model weights, predictions and numerical results were retained. `evidence/file_hashes.json` and `evidence/check_files.py` check the retained fixed files in this layout.
