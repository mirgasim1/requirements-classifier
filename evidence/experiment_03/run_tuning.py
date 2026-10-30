"""Compare a fixed set of SVM settings using the original training data."""

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


BASE = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(BASE.parent))
from check_files import check_files, recorded_path
PROJECT = BASE.parents[1]
PREVIOUS = BASE.parent / "experiment_02"
LABELS = ["FR", "NFR"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pipeline(config):
    word = TfidfVectorizer(lowercase=True, ngram_range=tuple(config["word_ngram_range"]), stop_words=None)
    character = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, lowercase=True)
    if config["features"] == "word":
        features = word
    elif config["features"] == "character":
        features = character
    elif config["features"] == "word_and_character":
        features = FeatureUnion([("word", word), ("character", character)], n_jobs=1)
    else:
        raise ValueError("Unknown feature configuration")
    return Pipeline([
        ("features", features),
        ("classifier", LinearSVC(C=config["C"], class_weight=config["class_weight"], random_state=42, max_iter=10000)),
    ])


def metrics(truth, predictions):
    report = classification_report(truth, predictions, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "FR": report["FR"], "NFR": report["NFR"],
        "confusion_matrix": confusion_matrix(truth, predictions, labels=LABELS).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results", help="A new output folder in experiment_03")
    args = parser.parse_args()
    output = (BASE / args.output).resolve()
    if not output.is_relative_to(BASE) or output == BASE or output.exists():
        raise ValueError("Choose a new output folder in experiment_03; existing files are not replaced.")
    protocol_path = BASE / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    check_files()
    for name, expected in protocol["input_sha256"].items():
        if digest(recorded_path(PREVIOUS / name)) != expected:
            raise ValueError(f"Input changed: {name}")
    output.mkdir()
    events = []

    def event(name):
        events.append({"stage": name, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(output / "execution_order.json", events)
        print(name, flush=True)

    event("Protocol and preservation checks completed before fitting")
    original = read_csv(recorded_path(PREVIOUS / "inputs/original_prepared.csv"))
    by_id = {row["source_row"]: row for row in original}
    membership = read_csv(recorded_path(PREVIOUS / "inputs/original_split_membership.csv"))
    train = [by_id[row["source_row"]] for row in membership if row["subset"] == "train"]
    test = [by_id[row["source_row"]] for row in membership if row["subset"] == "test"]
    assert (len(train), len(test)) == (4748, 1188)
    assert not ({row["source_row"] for row in train} & {row["source_row"] for row in test})
    xtrain = np.asarray([row["requirement_text"] for row in train], dtype=object)
    ytrain = np.asarray([row["label"] for row in train])
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(xtrain, ytrain))
    fold_rows = [
        {"source_row": train[index]["source_row"], "label": train[index]["label"], "validation_fold": number}
        for number, (_, validation) in enumerate(folds, start=1) for index in validation
    ]
    old_folds = read_csv(PREVIOUS / "results/cv_fold_membership.csv")
    assert [(r["source_row"], r["label"], str(r["validation_fold"])) for r in fold_rows] == [
        (r["source_row"], r["label"], r["validation_fold"]) for r in old_folds
    ]
    score_rows, validation_rows, cv = [], [], {}
    for config in protocol["candidates"]:
        scores = []
        for number, (fitting, validation) in enumerate(folds, start=1):
            model = pipeline(config)
            model.fit(xtrain[fitting], ytrain[fitting])
            prediction = model.predict(xtrain[validation])
            score = float(f1_score(ytrain[validation], prediction, average="macro", zero_division=0))
            scores.append(score)
            score_rows.append({"candidate": config["name"], "fold": number, "macro_f1": score})
            validation_rows.extend({
                "candidate": config["name"], "fold": number, "source_row": train[index]["source_row"],
                "reference_label": train[index]["label"], "predicted_label": str(label),
            } for index, label in zip(validation, prediction))
        cv[config["name"]] = {"fold_macro_f1": scores, "mean_macro_f1": float(np.mean(scores)), "std_macro_f1_population": float(np.std(scores))}
        write_csv(output / "cv_scores.csv", ["candidate", "fold", "macro_f1"], score_rows)
        event(f"Training-only five-fold evaluation completed: {config['name']}")
    selected = max(cv, key=lambda name: cv[name]["mean_macro_f1"])
    write_csv(output / "cv_predictions.csv", ["candidate", "fold", "source_row", "reference_label", "predicted_label"], validation_rows)
    write_json(output / "model_selection.json", {
        "selected_candidate": selected,
        "criterion": protocol["selection_criterion"], "cross_validation": cv,
        "test_data_used_to_rank_candidates": False,
        "qualification": "This is an exploratory follow-up to inspected results, not a new untouched-test claim. The best validation score is subject to selection optimism.",
    })
    event(f"Candidate selected and recorded before follow-up test predictions: {selected}")
    chosen = next(config for config in protocol["candidates"] if config["name"] == selected)
    model = pipeline(chosen).fit(xtrain, ytrain)
    (output / "models").mkdir()
    joblib.dump(model, output / "models/selected_candidate.joblib", compress=3)
    original_model = joblib.load(recorded_path(PREVIOUS / "inputs/original_saved_classifier.joblib"))
    external = read_csv(PREVIOUS / "results/SecReq_FR_NFR_prepared.csv")
    datasets = {"original_test": test, "secreq_combined": external}
    score_table, predictions, result = [], [], {}
    for model_name, fitted in [("original_app_svm", original_model), (selected, model)]:
        result[model_name] = {}
        for dataset_name, rows in datasets.items():
            texts = [row["requirement_text"] for row in rows]
            prediction = fitted.predict(texts)
            groups = [(dataset_name, rows, prediction)]
            if dataset_name == "secreq_combined":
                for project in ["CPN", "ePurse"]:
                    indices = [i for i, row in enumerate(rows) if row["source_project"] == project]
                    groups.append((f"secreq_{project}", [rows[i] for i in indices], prediction[indices]))
            for group_name, group_rows, group_predictions in groups:
                scores = metrics([row["label"] for row in group_rows], group_predictions)
                result[model_name][group_name] = scores
                score_table.append({"model": model_name, "dataset": group_name, "records": len(group_rows), **{
                    key: scores[key] for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
                }, "FR_recall": scores["FR"]["recall"], "NFR_recall": scores["NFR"]["recall"]})
            predictions.extend({
                "model": model_name, "dataset": dataset_name, "source_row": row.get("source_row", ""),
                "workbook_row": row.get("workbook_row", ""), "source_project": row.get("source_project", ""),
                "requirement_text": row["requirement_text"], "reference_label": row["label"],
                "predicted_label": str(label), "matches_reference": int(row["label"] == label),
            } for row, label in zip(rows, prediction))
        event(f"Test predictions recorded: {model_name}")
    write_csv(output / "model_comparison.csv", ["model", "dataset", "records", "accuracy", "macro_precision", "macro_recall", "macro_f1", "FR_recall", "NFR_recall"], score_table)
    write_csv(output / "predictions.csv", ["model", "dataset", "source_row", "workbook_row", "source_project", "requirement_text", "reference_label", "predicted_label", "matches_reference"], predictions)
    write_json(output / "results.json", {
        "selected_candidate": selected, "selected_configuration": chosen, "models": result,
        "training_records": len(train), "original_test_records": len(test), "external_records": len(external),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "protocol_sha256": digest(protocol_path),
        "app_model_replaced": False,
        "qualification": "Exploratory follow-up. Existing test results and case review were known before this search. Retained labels and data are unchanged. The external set is not used to rank configurations. No new independent test set was introduced.",
        "confusion_matrix_order": LABELS, "confusion_matrix_orientation": "Rows actual, columns predicted",
    })
    check_files()
    event("Recorded data and results checked; application model not replaced")
    print(json.dumps({"selected": selected, "results": result[selected]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
