# Software requirements classifier

This application classifies one English software requirement as Functional or Non-functional. It runs locally using a saved TF-IDF vectorizer and linear support vector machine (SVM).

The thesis experiments compare classifiers and training changes across different requirements collections. The original model remains the default application model; experimental candidates are saved separately.

- [View experiment results](evidence/75_percent_results/RESULTS.md)
- [Download candidate models](https://github.com/mirgasim1/requirements-classifier/releases)
- [Run the application](#running-on-windows)

## Running on Windows

1. Select **Code → Download ZIP** and extract the entire ZIP.
2. Install [Python 3.12](https://www.python.org/downloads/windows/) if necessary. Include the Python launcher.
3. Double-click `RUN.cmd`. The first run requires an internet connection to install dependencies.
4. The application opens at <http://127.0.0.1:8000/>. Enter a requirement and click **Classify**.
5. Keep the command window open. Press **Ctrl+C** in it to stop the server.

If the browser does not open automatically, open <http://127.0.0.1:8000/> manually.

The submitted text remains visible for editing and resubmission. Empty or spaces-only input is rejected. Predictions should be reviewed before use.

## Files and experiments

The application files are:

- `app.py`: Python server and prediction handling.
- `index.html`: browser interface.
- `classifier.joblib`: default TF-IDF and SVM model.
- `requirements.txt`: application dependencies.
- `evidence/`: datasets, preparation records, experiments and verification evidence.

| Folder | Completed work |
|---|---|
| [experiment_01](evidence/experiment_01) | Original TF-IDF and SVM baseline |
| [experiment_02](evidence/experiment_02/README.md) | Three-classifier comparison, SecReq preparation, selected case review and application checks |
| [experiment_03](evidence/experiment_03/README.md) | TF-IDF features, class weighting and SVM tuning |
| [experiment_04](evidence/experiment_04/README.md) | Frozen pretrained encoder and broader-training comparisons using PROMISE+ |
| [experiment_05](evidence/experiment_05/README.md) | Fine-tuning, validation-based training duration and separate candidate applications |

Each experiment contains its method, scripts, configurations, predictions, results and verification records.

Application verification records are available for the [original application](evidence/experiment_02/application_checks/verification.json) and the [fine-tuned candidates](evidence/experiment_05/results/candidate_application_checks.json).

## Data and results

The prepared data used were:

- [FR_NFR — Sonali and Thamada (2024)](https://data.mendeley.com/datasets/4ysx9fyzv4/1): 5,936 requirements, split into 4,748 training and 1,188 test records.
- [Curated SecReq release — Xanthopoulou et al. (2025)](https://doi.org/10.5281/zenodo.15082900): 253 selected requirements used for additional evaluation.
- [PROMISE+ — Carvalho da Silva et al. (2024)](https://doi.org/10.5281/zenodo.12805484): 1,496 additional training requirements and 802 evaluation requirements from separate project groups.

These dataset releases use the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/). Preparation records document exclusions, label mapping and subset assignments.

The [combined results](evidence/75_percent_results/RESULTS.md) present all tested configurations. Supporting files include [overall metrics](evidence/75_percent_results/all_model_metrics.csv), [class-level metrics](evidence/75_percent_results/per_class_metrics.csv) and [confusion matrices](evidence/75_percent_results/confusion_matrices.csv). Scores in the CSV files are fractions between 0 and 1.

Different configurations performed best on different evaluation groups. Increased NFR recall did not consistently improve overall performance. The findings remain exploratory and measure agreement with retained reference labels.

## Running a fine-tuned candidate

This is optional. `RUN.cmd` continues to run the original application.

1. Download `requirements-classifier_finetuned_models.zip` from [Releases](https://github.com/mirgasim1/requirements-classifier/releases).
2. Copy `original_only.safetensors` and `augmented.safetensors` from that archive into `evidence/experiment_05/results/models/`. Create the folder if necessary.
3. Open a terminal in the project folder containing `app.py` and run:

```text
py -3.12 -m venv .experiment_env
.experiment_env\Scripts\python.exe -m pip install -r evidence\experiment_05\requirements.txt
.experiment_env\Scripts\python.exe evidence\experiment_05\download_model.py
.experiment_env\Scripts\python.exe evidence\experiment_05\run_candidate_app.py --condition original_only
```

The setup downloads dependencies and the required public model assets. The candidate opens at <http://127.0.0.1:8001/>.

To demonstrate the broader-training candidate, stop the first candidate with **Ctrl+C**, then run:

```text
.experiment_env\Scripts\python.exe evidence\experiment_05\run_candidate_app.py --condition augmented
```

Both candidates use the existing interface. Using the supplied weights does not require retraining.

The encoder experiments use [Snowflake Arctic-Embed XS](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs), distributed under Apache 2.0. Model sources and licence notices are retained in the experiment folders and model archive.

## Repeating experiments

Saved results can be inspected without repeating training. Use a separate project copy when reproducing experiments.

After `RUN.cmd` has installed the application dependencies, the original experiment can be repeated with:

```text
.venv\Scripts\python.exe evidence\train_classifier.py
```

This replaces the output files in `evidence/experiment_01`; it does not replace the application's root `classifier.joblib`.

Reproduction commands for experiments 02–05 are in their linked README files. Experiments 04 and 05 require their additional dependencies.

## Limitations

The reference labels have not all been independently checked manually. SecReq covers two security-heavy specifications. For broader-training models, PROMISE+ evaluation concerns reserved projects within a source also represented in training.

Local application checks assess selected functions. They do not establish human usability, concurrent-use performance or reliable classification across every new project.
