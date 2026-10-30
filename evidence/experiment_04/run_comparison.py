"""Compare frozen semantic features and fixed additional-training conditions."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from semantic_features import FrozenEncoder

BASE = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(BASE.parent))
from check_files import check_files, recorded_path
PROJECT = BASE.parents[1]
PREVIOUS = BASE.parent / "experiment_02"
LABELS = ["FR", "NFR"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def metrics(truth, prediction):
    report = classification_report(truth, prediction, labels=LABELS, output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(truth, prediction)), "macro_precision": float(report["macro avg"]["precision"]), "macro_recall": float(report["macro avg"]["recall"]), "macro_f1": float(report["macro avg"]["f1-score"]), "FR": report["FR"], "NFR": report["NFR"], "confusion_matrix": confusion_matrix(truth, prediction, labels=LABELS).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    parser.add_argument("--model-dir", type=Path, default=BASE / "model_assets")
    args = parser.parse_args()
    output = (BASE / args.output).resolve()
    if not output.is_relative_to(BASE) or output == BASE or output.exists():
        raise ValueError("Choose a new output folder inside experiment_04")
    protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    check_files()
    for name, expected in protocol["input_sha256"].items():
        assert digest(BASE / name) == expected, f"Input changed: {name}"
    output.mkdir()
    (output / "models").mkdir()
    events = []

    def event(stage):
        events.append({"stage": stage, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(output / "execution_order.json", events)
        print(stage, flush=True)

    event("Protocol and source hashes verified before fitting")
    original = read_csv(recorded_path(PREVIOUS / "inputs/original_prepared.csv"))
    by_id = {r["source_row"]: r for r in original}
    membership = read_csv(recorded_path(PREVIOUS / "inputs/original_split_membership.csv"))
    train = [by_id[r["source_row"]] for r in membership if r["subset"] == "train"]
    original_test = [by_id[r["source_row"]] for r in membership if r["subset"] == "test"]
    partition = read_csv(BASE / "prepared/Promise_plus_partition.csv")
    additional = [r for r in partition if r["subset"] == "additional_training"]
    promise_test = [r for r in partition if r["subset"] == "external_test"]
    assert (len(train), len(original_test), len(additional), len(promise_test)) == (4748, 1188, protocol["additional_training"]["records"], protocol["promise_plus_test"]["records"])
    assert not ({r["project_id"] for r in additional} & {r["project_id"] for r in promise_test})
    xtrain = [r["requirement_text"] for r in train]
    ytrain = np.asarray([r["label"] for r in train])
    encoder = FrozenEncoder(args.model_dir)
    embeddings, encoding_train = encoder.encode(xtrain, name="original training")
    np.save(output / "original_training_embeddings.npy", embeddings, allow_pickle=False)
    event("Original training vectors encoded with fixed public model")
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(embeddings, ytrain))
    old_folds = {r["source_row"]: int(r["validation_fold"]) for r in read_csv(PREVIOUS / "results/cv_fold_membership.csv")}
    assert all(old_folds[train[index]["source_row"]] == number for number, (_, validation) in enumerate(folds, start=1) for index in validation)
    cv, cv_rows, cv_predictions = {}, [], []
    for config in protocol["head_candidates"]:
        scores = []
        for number, (fitting, validation) in enumerate(folds, start=1):
            head = LinearSVC(C=config["C"], class_weight=config["class_weight"], random_state=42, max_iter=20000)
            head.fit(embeddings[fitting], ytrain[fitting])
            labels = head.predict(embeddings[validation])
            score = float(f1_score(ytrain[validation], labels, average="macro", zero_division=0))
            scores.append(score)
            cv_rows.append({"candidate": config["name"], "fold": number, "macro_f1": score})
            cv_predictions.extend({"candidate": config["name"], "fold": number, "source_row": train[index]["source_row"], "reference_label": train[index]["label"], "predicted_label": str(label)} for index, label in zip(validation, labels))
        cv[config["name"]] = {"fold_macro_f1": scores, "mean_macro_f1": float(np.mean(scores)), "std_macro_f1_population": float(np.std(scores))}
        event(f"Training-only semantic-head validation complete: {config['name']}")
    selected = max(cv, key=lambda name: cv[name]["mean_macro_f1"])
    chosen = next(c for c in protocol["head_candidates"] if c["name"] == selected)
    write_csv(output / "cv_scores.csv", ["candidate", "fold", "macro_f1"], cv_rows)
    write_csv(output / "cv_predictions.csv", ["candidate", "fold", "source_row", "reference_label", "predicted_label"], cv_predictions)
    write_json(output / "model_selection.json", {"semantic_head_selected": selected, "selected_configuration": chosen, "cross_validation": cv, "selection_data": "Original training folds only", "external_data_used_for_selection": False, "selection_is_not_a_deployment_decision": True})
    event(f"Semantic configuration recorded before any test prediction: {selected}")
    additional_vectors, encoding_additional = encoder.encode([r["requirement_text"] for r in additional], name="additional training")
    np.save(output / "additional_training_embeddings.npy", additional_vectors, allow_pickle=False)
    semantic_original = LinearSVC(C=chosen["C"], class_weight=chosen["class_weight"], random_state=42, max_iter=20000).fit(embeddings, ytrain)
    semantic_augmented = LinearSVC(C=chosen["C"], class_weight=chosen["class_weight"], random_state=42, max_iter=20000).fit(np.vstack([embeddings, additional_vectors]), np.concatenate([ytrain, np.asarray([r["label"] for r in additional])]))
    word_augmented = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words=None)), ("classifier", LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=10000))]).fit(xtrain + [r["requirement_text"] for r in additional], np.concatenate([ytrain, np.asarray([r["label"] for r in additional])]))
    for name, model in [("semantic_original_training", semantic_original), ("semantic_augmented_training", semantic_augmented), ("word_augmented_training", word_augmented)]:
        joblib.dump(model, output / "models" / f"{name}.joblib", compress=3)
    event("All original-only and augmented candidates fitted before test evaluation")
    models = {
        "original_app_svm": (joblib.load(PROJECT / "classifier.joblib"), "text"),
        "tuned_word_original_training": (joblib.load(BASE.parent / "experiment_03/results/models/selected_candidate.joblib"), "text"),
        "semantic_original_training": (semantic_original, "embedding"),
        "word_augmented_training": (word_augmented, "text"),
        "semantic_augmented_training": (semantic_augmented, "embedding"),
    }
    datasets = {"original_test": original_test, "secreq": read_csv(PREVIOUS / "results/SecReq_FR_NFR_prepared.csv"), "promise_plus_heldout_projects": promise_test}
    predictions, table, class_rows, per_project_rows, all_results, baselines, encoding_test = [], [], [], [], {}, {}, {}
    for dataset, rows in datasets.items():
        texts = [r["requirement_text"] for r in rows]
        truth = [r["label"] for r in rows]
        vectors, encoding_test[dataset] = encoder.encode(texts, name=dataset)
        np.save(output / f"{dataset}_embeddings.npy", vectors, allow_pickle=False)
        baselines[dataset] = {label: metrics(truth, [label] * len(rows)) for label in LABELS}
        for name, (model, feature_type) in models.items():
            labels = model.predict(vectors if feature_type == "embedding" else texts)
            scores = metrics(truth, labels)
            all_results.setdefault(name, {})[dataset] = scores
            table.append({"model": name, "dataset": dataset, "records": len(rows), **{key: scores[key] for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}, "FR_recall": scores["FR"]["recall"], "NFR_recall": scores["NFR"]["recall"]})
            class_rows.extend({"model": name, "dataset": dataset, "label": label, **scores[label]} for label in LABELS)
            predictions.extend({"model": name, "dataset": dataset, "source_row": r.get("source_row", ""), "source_line": r.get("source_line", ""), "workbook_row": r.get("workbook_row", ""), "project_id": r.get("project_id", r.get("source_project", "")), "requirement_text": r["requirement_text"], "source_label": r.get("source_label", r["label"]), "reference_label": r["label"], "predicted_label": str(label), "matches_reference": int(r["label"] == label)} for r, label in zip(rows, labels))
            if dataset != "original_test":
                for project in sorted({r.get("project_id", r.get("source_project")) for r in rows}):
                    ids = [i for i, r in enumerate(rows) if r.get("project_id", r.get("source_project")) == project]
                    group_scores = metrics([truth[i] for i in ids], labels[ids])
                    per_project_rows.append({"model": name, "dataset": dataset, "project_id": project, "records": len(ids), **{k: group_scores[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}})
        event(f"Fixed test predictions recorded for all five models: {dataset}")
    write_csv(output / "predictions.csv", ["model", "dataset", "source_row", "source_line", "workbook_row", "project_id", "requirement_text", "source_label", "reference_label", "predicted_label", "matches_reference"], predictions)
    write_csv(output / "model_comparison.csv", ["model", "dataset", "records", "accuracy", "macro_precision", "macro_recall", "macro_f1", "FR_recall", "NFR_recall"], table)
    write_csv(output / "per_class_metrics.csv", ["model", "dataset", "label", "precision", "recall", "f1-score", "support"], class_rows)
    write_csv(output / "per_project_metrics.csv", ["model", "dataset", "project_id", "records", "accuracy", "macro_precision", "macro_recall", "macro_f1"], per_project_rows)
    write_json(output / "results.json", {
        "protocol_sha256": digest(BASE / "protocol.json"), "semantic_configuration": chosen, "models": all_results, "constant_label_baselines": baselines,
        "original_training_records": len(train), "additional_training_records": len(additional), "augmented_training_records": len(train) + len(additional),
        "dataset_counts": {name: {"records": len(rows), "labels": dict(Counter(r["label"] for r in rows))} for name, rows in datasets.items()},
        "encoding": {"original_training": encoding_train, "additional_training": encoding_additional, **encoding_test},
        "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn", "joblib", "onnxruntime", "tokenizers"]}},
        "app_model_replaced": False, "external_scores_used_to_select_or_deploy": False,
        "qualification": "All comparisons are retained. Existing original/SecReq evidence was inspected earlier. New PROMISE+ partition fixed before prediction. Augmented conditions have seen other projects from the same new source and are not source-unseen tests.",
        "confusion_matrix_label_order": LABELS, "confusion_matrix_orientation": "Rows reference, columns prediction",
    })
    check_files()
    event("Previous project files preserved; all recorded comparisons complete")
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()
