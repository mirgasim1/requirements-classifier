"""Reload final weights and independently check every recorded test prediction."""

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from text_model import LABELS, RequirementClassifier, predict, tokenize_rows

BASE = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(BASE.parent))
from check_files import check_files, recorded_path
PROJECT = BASE.parents[1]
PREVIOUS = BASE.parent / "experiment_02"
FOUR = BASE.parent / "experiment_04"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_metrics(truth, labels, expected):
    stats = classification_report(truth, labels, labels=LABELS, output_dict=True, zero_division=0)
    assert np.isclose(accuracy_score(truth, labels), expected["accuracy"])
    for field, key in [("macro_precision", "precision"), ("macro_recall", "recall"), ("macro_f1", "f1-score")]:
        assert np.isclose(stats["macro avg"][key], expected[field])
    for label in LABELS:
        for field in ["precision", "recall", "f1-score", "support"]:
            assert np.isclose(stats[label][field], expected[label][field])
    assert confusion_matrix(truth, labels, labels=LABELS).tolist() == expected["confusion_matrix"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    output = (BASE / args.results).resolve()
    if not output.is_relative_to(BASE):
        raise ValueError("Results must be inside experiment_05")
    target = output / "verification.json"
    result = read_json(output / "results.json")
    protocol = read_json(BASE / "protocol.json")
    assert result["protocol_sha256"] == digest(BASE / "protocol.json")
    assert digest(args.model_dir / "model.safetensors") == protocol["weights_sha256"]
    checked_files = check_files()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    original = read_csv(recorded_path(PREVIOUS / "inputs/original_prepared.csv"))
    original_by_id = {r["source_row"]: {**r, "source_id": "original:" + r["source_row"]} for r in original}
    membership = read_csv(recorded_path(PREVIOUS / "inputs/original_split_membership.csv"))
    original_test = [original_by_id[r["source_row"]] for r in membership if r["subset"] == "test"]
    partition = [{**r, "source_id": "promise_plus:" + r["source_line"]} for r in read_csv(FOUR / "prepared/Promise_plus_partition.csv")]
    datasets = {
        "original_test": original_test,
        "secreq": [{**r, "source_id": "secreq:" + r["workbook_row"]} for r in read_csv(PREVIOUS / "results/SecReq_FR_NFR_prepared.csv")],
        "promise_plus_heldout_projects": [r for r in partition if r["subset"] == "external_test"],
    }
    splits = read_csv(output / "training_validation_membership.csv")
    predictions = read_csv(output / "predictions.csv")
    validation_predictions_checked, test_predictions_checked, metric_groups, maximum_logit_error = 0, 0, 0, 0.0
    selected = read_json(output / "epoch_selection.json")
    for condition in ["original_only", "augmented"]:
        fit = [r for r in splits if r["condition"] == condition and r["subset"] == "fitting"]
        validation = [r for r in splits if r["condition"] == condition and r["subset"] == "validation"]
        refit = [r for r in splits if r["condition"] == condition and r["subset"] == "refit"]
        assert not ({r["source_id"] for r in fit} & {r["source_id"] for r in validation})
        assert {r["source_id"] for r in refit} == {r["source_id"] for r in fit + validation}
        all_test_ids = {r["source_id"] for rows in datasets.values() for r in rows}
        assert not ({r["source_id"] for r in refit} & all_test_ids)
        if condition == "augmented":
            validation_projects = {r["project_id"] for r in validation if r["project_id"]}
            assert not (validation_projects & {r["project_id"] for r in fit if r["project_id"]})
            assert not ({r["project_id"] for r in refit if r["project_id"]} & {r["project_id"] for r in datasets["promise_plus_heldout_projects"]})
        history = read_json(output / f"{condition}_selection_history.json")
        validation_predictions = read_csv(output / f"{condition}_validation_predictions.csv")
        for epoch in [1, 2, 3]:
            rows = [r for r in validation_predictions if int(r["epoch"]) == epoch]
            assert len(rows) == len(validation)
            assert [r["source_id"] for r in rows] == [r["source_id"] for r in validation]
            assert [r["reference_label"] for r in rows] == [r["label"] for r in validation]
            logits = np.asarray([[float(r["logit_FR"]), float(r["logit_NFR"])] for r in rows])
            labels = np.asarray(LABELS)[logits.argmax(axis=1)]
            assert labels.tolist() == [r["predicted_label"] for r in rows]
            check_metrics([r["reference_label"] for r in rows], labels, history[epoch - 1]["validation"])
            validation_predictions_checked += len(rows)
        best = max(history, key=lambda item: item["validation"]["macro_f1"])
        assert best["epoch"] == selected[condition]["selected_epochs"]
        assert len(read_json(output / f"{condition}_refit_history.json")) == best["epoch"]
        weights = output / "models" / f"{condition}.safetensors"
        assert digest(weights) == result["model_sha256"][condition]
        model = RequirementClassifier(args.model_dir)
        model.load_state_dict(load_file(str(weights)), strict=True)
        for dataset, rows in datasets.items():
            tokens, _ = tokenize_rows(rows, args.model_dir)
            labels, logits = predict(model, tokens)
            saved = [r for r in predictions if r["model"] == condition and r["dataset"] == dataset]
            assert len(rows) == len(saved)
            for source, stored in zip(rows, saved):
                assert source["source_id"] == stored["source_id"]
                assert source["requirement_text"] == stored["requirement_text"]
                assert source["label"] == stored["reference_label"]
            assert labels.tolist() == [r["predicted_label"] for r in saved]
            stored_logits = np.asarray([[float(r["logit_FR"]), float(r["logit_NFR"])] for r in saved])
            error = float(np.abs(logits - stored_logits).max())
            assert error < 1e-5
            maximum_logit_error = max(maximum_logit_error, error)
            check_metrics([r["label"] for r in rows], labels, result["models"][condition][dataset])
            test_predictions_checked += len(rows)
            metric_groups += 1
            print(f"Verified {condition}: {dataset}", flush=True)
        del model
    check_files()
    report = {
        "status": "passed", "recorded_files_checked": checked_files,
        "validation_predictions_checked": validation_predictions_checked,
        "test_predictions_recomputed_from_saved_weights": test_predictions_checked,
        "test_metric_groups_recomputed": metric_groups, "maximum_reloaded_logit_difference": maximum_logit_error,
        "epoch_selection_recomputed_from_validation_only": True,
        "all_test_ids_excluded_from_fitting_validation_and_refit": True,
        "auxiliary_validation_and_test_project_ids_disjoint": True,
        "saved_weights_sha256_verified": True, "app_model_replaced": False,
        "qualification": "These integrity checks do not independently validate reference labels or make inspected test sets untouched again.",
    }
    if not target.exists():
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
