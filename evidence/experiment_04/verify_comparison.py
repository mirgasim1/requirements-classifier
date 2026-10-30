"""Independently verify saved predictions, metrics, inputs and encoder alignment."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from semantic_features import FrozenEncoder

BASE = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(BASE.parent))
from check_files import check_files, recorded_path
PROJECT = BASE.parents[1]
PREVIOUS = BASE.parent / "experiment_02"
LABELS = ["FR", "NFR"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_metrics(truth, prediction, expected):
    stats = classification_report(truth, prediction, labels=LABELS, output_dict=True, zero_division=0)
    assert np.isclose(accuracy_score(truth, prediction), expected["accuracy"])
    for field, key in [("macro_precision", "precision"), ("macro_recall", "recall"), ("macro_f1", "f1-score")]:
        assert np.isclose(stats["macro avg"][key], expected[field])
    for label in LABELS:
        for key in ["precision", "recall", "f1-score", "support"]:
            assert np.isclose(stats[label][key], expected[label][key])
    assert confusion_matrix(truth, prediction, labels=LABELS).tolist() == expected["confusion_matrix"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--model-dir", type=Path, default=BASE / "model_assets")
    args = parser.parse_args()
    results = (BASE / args.results).resolve()
    if not results.is_relative_to(BASE) or not results.is_dir():
        raise ValueError("Invalid results folder")
    target = results / "verification.json"
    recorded = read_json(results / "results.json")
    protocol = read_json(BASE / "protocol.json")
    assert digest(BASE / "protocol.json") == recorded["protocol_sha256"]
    assert all(digest(BASE / name) == expected for name, expected in protocol["input_sha256"].items())
    checked_files = check_files()
    original = read_csv(recorded_path(PREVIOUS / "inputs/original_prepared.csv"))
    original_by_id = {r["source_row"]: r for r in original}
    membership = read_csv(recorded_path(PREVIOUS / "inputs/original_split_membership.csv"))
    train = [original_by_id[r["source_row"]] for r in membership if r["subset"] == "train"]
    original_test = [original_by_id[r["source_row"]] for r in membership if r["subset"] == "test"]
    partition = read_csv(BASE / "prepared/Promise_plus_partition.csv")
    additional = [r for r in partition if r["subset"] == "additional_training"]
    promise = [r for r in partition if r["subset"] == "external_test"]
    assert not ({r["project_id"] for r in additional} & {r["project_id"] for r in promise})
    parsed = {r["source_line"]: r for r in read_csv(BASE / "prepared/Promise_plus_parsed.csv")}
    mapping = read_json(BASE / "prepared/Promise_plus_audit.json")["binary_mapping"]
    for row in partition:
        source = parsed[row["source_line"]]
        for key in ["project_id", "requirement_text", "source_label"]:
            assert row[key] == source[key]
        assert row["label"] == mapping[source["source_label"]]
    folds = {r["source_row"]: int(r["validation_fold"]) for r in read_csv(PREVIOUS / "results/cv_fold_membership.csv")}
    train_by_id = {r["source_row"]: r for r in train}
    cv = read_csv(results / "cv_predictions.csv")
    groups = defaultdict(list)
    for row in cv:
        assert row["source_row"] in train_by_id
        assert row["reference_label"] == train_by_id[row["source_row"]]["label"]
        assert int(row["fold"]) == folds[row["source_row"]]
        groups[(row["candidate"], int(row["fold"]))].append(row)
    selection = read_json(results / "model_selection.json")
    means = {}
    for candidate in protocol["head_candidates"]:
        scores, ids = [], []
        for fold in range(1, 6):
            rows = groups[(candidate["name"], fold)]
            ids.extend(r["source_row"] for r in rows)
            score = float(f1_score([r["reference_label"] for r in rows], [r["predicted_label"] for r in rows], average="macro", zero_division=0))
            assert np.isclose(score, selection["cross_validation"][candidate["name"]]["fold_macro_f1"][fold - 1])
            scores.append(score)
        assert set(ids) == set(train_by_id) and len(ids) == len(train)
        means[candidate["name"]] = float(np.mean(scores))
        assert np.isclose(means[candidate["name"]], selection["cross_validation"][candidate["name"]]["mean_macro_f1"])
    assert max(means, key=means.get) == selection["semantic_head_selected"]
    models = {
        "original_app_svm": (joblib.load(PROJECT / "classifier.joblib"), "text"),
        "tuned_word_original_training": (joblib.load(BASE.parent / "experiment_03/results/models/selected_candidate.joblib"), "text"),
        "semantic_original_training": (joblib.load(results / "models/semantic_original_training.joblib"), "embedding"),
        "word_augmented_training": (joblib.load(results / "models/word_augmented_training.joblib"), "text"),
        "semantic_augmented_training": (joblib.load(results / "models/semantic_augmented_training.joblib"), "embedding"),
    }
    datasets = {"original_test": original_test, "secreq": read_csv(PREVIOUS / "results/SecReq_FR_NFR_prepared.csv"), "promise_plus_heldout_projects": promise}
    encoder = FrozenEncoder(args.model_dir)
    embedding_error, sample_records = 0.0, 0
    stored_predictions = read_csv(results / "predictions.csv")
    predictions_checked, metric_groups, baseline_groups = 0, 0, 0
    pooled_rows = read_csv(results / "model_comparison.csv")
    project_rows = read_csv(results / "per_project_metrics.csv")
    for dataset, rows in datasets.items():
        truth = [r["label"] for r in rows]
        vectors = np.load(results / f"{dataset}_embeddings.npy", allow_pickle=False)
        assert vectors.shape == (len(rows), 384) and np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
        indices = sorted(set([0, len(rows) // 3, len(rows) // 2, len(rows) - 1]))
        fresh, _ = encoder.encode([rows[i]["requirement_text"] for i in indices], name="verification")
        difference = float(np.abs(fresh - vectors[indices]).max())
        assert difference < 2e-5, f"Encoding alignment failed: {dataset}"
        embedding_error = max(embedding_error, difference)
        sample_records += len(indices)
        for name, (model, features) in models.items():
            predicted = model.predict(vectors if features == "embedding" else [r["requirement_text"] for r in rows])
            saved = [r for r in stored_predictions if r["dataset"] == dataset and r["model"] == name]
            assert len(saved) == len(rows)
            for source, stored, label in zip(rows, saved, predicted):
                assert source["requirement_text"] == stored["requirement_text"]
                assert source["label"] == stored["reference_label"]
                assert str(label) == stored["predicted_label"]
                assert int(source["label"] == label) == int(stored["matches_reference"])
            check_metrics(truth, predicted, recorded["models"][name][dataset])
            table = next(r for r in pooled_rows if r["model"] == name and r["dataset"] == dataset)
            for field in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
                assert np.isclose(float(table[field]), recorded["models"][name][dataset][field])
            for summary in [r for r in project_rows if r["model"] == name and r["dataset"] == dataset]:
                group_indices = [i for i, row in enumerate(rows) if row.get("project_id", row.get("source_project")) == summary["project_id"]]
                assert len(group_indices) == int(summary["records"])
                actual = [truth[i] for i in group_indices]
                labels = predicted[group_indices]
                stats = classification_report(actual, labels, labels=LABELS, output_dict=True, zero_division=0)
                assert np.isclose(accuracy_score(actual, labels), float(summary["accuracy"]))
                assert np.isclose(stats["macro avg"]["f1-score"], float(summary["macro_f1"]))
            predictions_checked += len(rows)
            metric_groups += 1
        for label in LABELS:
            check_metrics(truth, [label] * len(rows), recorded["constant_label_baselines"][dataset][label])
            baseline_groups += 1
    previous = read_json(BASE.parent / "experiment_03/results/results.json")["models"]
    for name, old_name in [("original_app_svm", "original_app_svm"), ("tuned_word_original_training", "word_pairs_balanced")]:
        for dataset, old_dataset in [("original_test", "original_test"), ("secreq", "secreq_combined")]:
            for field in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
                assert np.isclose(recorded["models"][name][dataset][field], previous[old_name][old_dataset][field])
    check_files()
    report = {
        "status": "passed", "recorded_files_checked": checked_files,
        "training_validation_predictions_checked": len(cv), "saved_test_predictions_recomputed": predictions_checked,
        "pooled_model_metric_groups_recomputed": metric_groups, "baseline_metric_groups_recomputed": baseline_groups,
        "per_project_metric_rows_checked": len(project_rows), "heldout_project_ids_disjoint_from_additional_training": True,
        "retained_source_text_and_labels_unchanged": True, "previous_original_and_secreq_results_reproduced": True,
        "fresh_encoder_alignment_records": sample_records, "maximum_fresh_embedding_difference": embedding_error,
        "encoder_weights_pinned_and_unchanged": True, "cv_means_recomputed": means,
        "app_model_replaced": False, "reference_labels_independently_reannotated": False,
    }
    if not target.exists():
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
