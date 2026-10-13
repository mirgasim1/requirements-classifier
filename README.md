# Software requirements classifier

This application classifies one English software requirement as Functional or Non-functional using a trained TF-IDF vectorizer and linear SVM. It runs locally on the user's computer.

## Running on Windows

1. Select Code → Download ZIP and extract the ZIP.
2. Install Python 3.12 from https://www.python.org/downloads/windows/ if needed. Include the Python launcher.
3. Double-click `RUN.cmd`. The first run needs an internet connection to install the packages.
4. The application opens at http://127.0.0.1:8000/. Enter a requirement and click Classify.
5. Keep the command window open. Press Ctrl+C in it to stop the application.

If the browser does not open automatically, open http://127.0.0.1:8000/ manually.

## Files

- `app.py`: the Python server.
- `index.html`: the browser interface.
- `classifier.joblib`: the saved vectorizer and classifier.
- `requirements.txt`: the required package versions.
- `verify_app.py`: checks selected application functions.
- `evidence/`: datasets, preparation records, training script and experiment results.

## Dataset and results

Dataset: Sonali, Sonali; Thamada, Srinivasarao (2024). FR_NFR_dataset, Version 1. Mendeley Data.

Source: https://data.mendeley.com/datasets/4ysx9fyzv4/1  
Licence: CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/

Invalid labels, extra duplicate copies and statements with conflicting labels were removed. The prepared dataset contains 5,936 requirements: 4,748 for training and 1,188 for testing.

The first experiment achieved 87.96% accuracy, 87.61% macro precision, 85.28% macro recall and 86.25% macro F1-score.

These results apply to one test subset. Performance on new projects has not been evaluated, and the dataset labels have not all been checked manually. Predictions should be reviewed before use.

## Repeating the experiment

After running `RUN.cmd` to install the packages, run:

```text
.venv\Scripts\python.exe evidence\train_classifier.py
```

This uses the prepared dataset and replaces the output files in `evidence\experiment_01`. It does not replace the root `classifier.joblib` used by the application.
