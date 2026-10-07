SOFTWARE REQUIREMENTS CLASSIFIER

Purpose
This local prototype predicts Functional or Non-functional for one English
software requirement using the trained TF-IDF and linear SVM model.
It uses learned model predictions, not rules assigning labels to keywords.

HOW TO START ON WINDOWS
1. Extract the entire ZIP into a folder. Do not run files from inside the ZIP.
2. Install Python 3.12 from https://www.python.org/downloads/windows/ if needed.
   Include the Python launcher when installing.
3. Double-click RUN.cmd.
4. On the first run, the required packages are downloaded into the .venv folder.
   An internet connection is needed for this installation.
5. A browser opens at http://127.0.0.1:8000/.
6. Enter a requirement and click Classify. Edit the text and click again to retry.
7. Keep the command window open. Press Ctrl+C in it to stop the server.

If the browser does not open, enter http://127.0.0.1:8000/ in your browser.
If port 8000 is already in use, run:
  .venv\Scripts\python.exe app.py --port 8001
The application is accessible only on this computer.

FILES
app.py: receives text, calls the saved model and returns the result page.
index.html: the browser form and page appearance.
classifier.joblib: saved vectorizer and classifier from the first experiment.
requirements.txt: exact package versions used for the experiment.
evidence/: experiment results, predictions, split membership, dataset
preparation records, original dataset and prepared dataset.

INITIAL EXPERIMENT
Dataset source: https://data.mendeley.com/datasets/4ysx9fyzv4/1
Sonali, Sonali; Thamada, Srinivasarao (2024). FR_NFR_dataset, Version 1.
Mendeley Data. DOI: 10.17632/4ysx9fyzv4.1. Licence: CC BY 4.0.
Dataset licence: https://creativecommons.org/licenses/by/4.0/
The prepared derivative excludes invalid labels, conflicting duplicates and
repeated same-label statements. Retained texts and labels are unchanged.
Prepared data: 5,936 records. Training: 4,748. Test: 1,188.
The stratified 80/20 split uses random_state=42.
TF-IDF uses lowercase word unigrams, no stop-word removal and default L2
normalization and smoothed IDF. LinearSVC uses C=1.0 and no class weighting.
Only the training data were used to fit the vectorizer and classifier.
Accuracy: 87.96%. Macro precision: 87.61%. Macro recall: 85.28%.
Macro F1: 86.25%. These are measurements on one random held-out subset.
Remaining labels have not all been manually validated. Similar requirements
may remain, and no evaluation on completely new projects has been performed.
The model may return an incorrect category even when the web interface works.

TO REPEAT THE EXPERIMENT
After RUN.cmd has installed dependencies:
  .venv\Scripts\python.exe evidence\train_classifier.py
This writes a new run to evidence\experiment_01 and uses
evidence\prepared\FR_NFR_prepared.csv. The app's classifier.joblib is not
automatically replaced by retraining; replacement should be deliberate.
