"""Run the first FR/NFR experiment on the prepared dataset."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys
from collections import Counter

BASE = Path(__file__).resolve().parent
if (BASE / "runtime").exists():
    sys.path.insert(0, str(BASE / "runtime"))

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA = BASE / "prepared" / "FR_NFR_prepared.csv"
OUTPUT = BASE / "experiment_01"


def main():
    OUTPUT.mkdir(exist_ok=True)
    with DATA.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    keys = [re.sub(r"\s+", " ", row["requirement_text"]).strip().casefold() for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated requirement text found in prepared data.")
    if any(row["label"] not in ("FR", "NFR") or not row["requirement_text"].strip() for row in rows):
        raise ValueError("Invalid requirement text or label found.")

    # Preserve approximately the same class proportions in both subsets.
    train, test = train_test_split(
        rows, test_size=0.20, random_state=42,
        stratify=[row["label"] for row in rows],
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)),
        ("svm", LinearSVC(C=1.0, class_weight=None, random_state=42, max_iter=10000)),
    ])
    # Fit the vocabulary, IDF weights and classifier on training data only.
    model.fit([row["requirement_text"] for row in train], [row["label"] for row in train])
    truth = [row["label"] for row in test]
    predictions = model.predict([row["requirement_text"] for row in test])
    report = classification_report(truth, predictions, labels=["FR", "NFR"], output_dict=True, zero_division=0)
    majority_label = Counter(row["label"] for row in train).most_common(1)[0][0]
    results = {
        "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "python_version": sys.version.split()[0], "scikit_learn_version": sklearn.__version__,
        "split": {"test_fraction": 0.20, "random_state": 42, "stratified": True},
        "training_records": len(train), "test_records": len(test),
        "training_label_counts": dict(Counter(row["label"] for row in train)),
        "test_label_counts": dict(Counter(truth)),
        "model": "TF-IDF unigrams and linear SVM, C=1.0, no class weighting",
        "tfidf_settings": {"lowercase": True, "stop_words": None, "ngram_range": [1, 1], "norm": "l2", "smooth_idf": True},
        "accuracy": accuracy_score(truth, predictions), "classification_report": report,
        "confusion_matrix_label_order": ["FR", "NFR"],
        "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(truth, predictions, labels=["FR", "NFR"]).tolist(),
        "majority_baseline_label": majority_label,
        "majority_baseline_accuracy": accuracy_score(truth, [majority_label] * len(test)),
        "limitations": "Single random holdout; labels not fully manually validated; similar requirements may remain; no project-level generalization test; no tuning performed.",
    }
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (OUTPUT / "split_membership.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_row", "subset", "label"])
        for subset, subset_rows in [("train", train), ("test", test)]:
            writer.writerows((row["source_row"], subset, row["label"]) for row in subset_rows)
    with (OUTPUT / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_row", "requirement_text", "actual_label", "predicted_label", "correct"])
        for row, prediction in zip(test, predictions):
            writer.writerow([row["source_row"], row["requirement_text"], row["label"], prediction, row["label"] == prediction])
    joblib.dump(model, OUTPUT / "classifier.joblib")
    versions = {name: __import__(name).__version__ for name in ["sklearn", "numpy", "scipy", "joblib"]}
    names = {"sklearn": "scikit-learn", "numpy": "numpy", "scipy": "scipy", "joblib": "joblib"}
    (BASE / "requirements.txt").write_text("\n".join(f"{names[name]}=={version}" for name, version in versions.items()) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
