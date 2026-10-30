"""Recompute follow-up predictions and test the candidate with the web handler."""

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import threading
from collections import Counter, defaultdict
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight


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


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    output = (BASE / args.results).resolve()
    if not output.is_relative_to(BASE) or not output.is_dir():
        raise ValueError("Results must be inside experiment_03")
    report_path = output / "verification.json"
    result = read_json(output / "results.json")
    selection = read_json(output / "model_selection.json")
    protocol = read_json(BASE / "protocol.json")
    assert digest(BASE / "protocol.json") == result["protocol_sha256"]
    selected = result["selected_candidate"]
    cv_rows = read_csv(output / "cv_predictions.csv")
    groups = defaultdict(list)
    for row in cv_rows:
        groups[(row["candidate"], int(row["fold"]))].append(row)
    cv_summary = {}
    for config in protocol["candidates"]:
        scores, identifiers = [], []
        for fold in range(1, 6):
            rows = groups[(config["name"], fold)]
            identifiers.extend(row["source_row"] for row in rows)
            score = f1_score([row["reference_label"] for row in rows], [row["predicted_label"] for row in rows], average="macro", zero_division=0)
            assert np.isclose(score, selection["cross_validation"][config["name"]]["fold_macro_f1"][fold - 1])
            scores.append(float(score))
        assert len(identifiers) == len(set(identifiers)) == 4748
        mean = float(np.mean(scores))
        assert np.isclose(mean, selection["cross_validation"][config["name"]]["mean_macro_f1"])
        cv_summary[config["name"]] = mean
    assert selected == max(cv_summary, key=cv_summary.get)
    original = read_csv(recorded_path(PREVIOUS / "inputs/original_prepared.csv"))
    original_by_id = {row["source_row"]: row for row in original}
    membership = read_csv(recorded_path(PREVIOUS / "inputs/original_split_membership.csv"))
    train = [original_by_id[row["source_row"]] for row in membership if row["subset"] == "train"]
    test = [original_by_id[row["source_row"]] for row in membership if row["subset"] == "test"]
    external = read_csv(PREVIOUS / "results/SecReq_FR_NFR_prepared.csv")
    datasets = {"original_test": test, "secreq_combined": external}
    models = {
        "original_app_svm": joblib.load(PROJECT / "classifier.joblib"),
        selected: joblib.load(output / "models/selected_candidate.joblib"),
    }
    predicted_rows = read_csv(output / "predictions.csv")
    predictions_recomputed, metric_groups = 0, 0
    paired_changes = {}
    for name, model in models.items():
        for dataset, rows in datasets.items():
            saved = [row for row in predicted_rows if row["model"] == name and row["dataset"] == dataset]
            fresh = model.predict([row["requirement_text"] for row in rows])
            assert len(saved) == len(rows)
            for source, stored, label in zip(rows, saved, fresh):
                assert source["requirement_text"] == stored["requirement_text"]
                assert source["label"] == stored["reference_label"]
                assert str(label) == stored["predicted_label"]
                assert int(source["label"] == label) == int(stored["matches_reference"])
            predictions_recomputed += len(fresh)
            subsets = [(dataset, list(range(len(rows))))]
            if dataset == "secreq_combined":
                subsets += [(f"secreq_{project}", [i for i, row in enumerate(rows) if row["source_project"] == project]) for project in ["CPN", "ePurse"]]
            for subset, indices in subsets:
                truth = [rows[i]["label"] for i in indices]
                labels = fresh[indices]
                expected = result["models"][name][subset]
                stats = classification_report(truth, labels, labels=LABELS, output_dict=True, zero_division=0)
                assert np.isclose(accuracy_score(truth, labels), expected["accuracy"])
                for field, key in [("macro_precision", "precision"), ("macro_recall", "recall"), ("macro_f1", "f1-score")]:
                    assert np.isclose(stats["macro avg"][key], expected[field])
                assert confusion_matrix(truth, labels, labels=LABELS).tolist() == expected["confusion_matrix"]
                for label in LABELS:
                    for key in ["precision", "recall", "f1-score", "support"]:
                        assert np.isclose(stats[label][key], expected[label][key])
                metric_groups += 1
    for dataset in datasets:
        old = [r for r in predicted_rows if r["model"] == "original_app_svm" and r["dataset"] == dataset]
        new = [r for r in predicted_rows if r["model"] == selected and r["dataset"] == dataset]
        assert len(old) == len(new)
        counts = Counter()
        for a, b in zip(old, new):
            assert (a["source_row"], a["workbook_row"]) == (b["source_row"], b["workbook_row"])
            counts[f"original_{a['matches_reference']}_candidate_{b['matches_reference']}"] += 1
        paired_changes[dataset] = dict(counts)
    candidate = models[selected]
    config = result["selected_configuration"]
    assert candidate.named_steps["classifier"].C == config["C"]
    assert candidate.named_steps["classifier"].class_weight == config["class_weight"]
    weights = compute_class_weight("balanced", classes=np.asarray(LABELS), y=[row["label"] for row in train]).tolist()
    old_baseline = read_json(PREVIOUS / "results/results.json")["models"]["linear_svm"]
    for dataset in ["original_test", "secreq_combined", "secreq_CPN", "secreq_ePurse"]:
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            assert np.isclose(result["models"]["original_app_svm"][dataset][key], old_baseline[dataset][key])
    specification = importlib.util.spec_from_file_location("candidate_app_check", PROJECT / "app.py")
    app = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(app)
    app.MODEL = candidate
    original_fit = candidate.fit

    def refuse_fit(*args, **kwargs):
        raise AssertionError("Application tried to retrain the candidate")

    candidate.fit = refuse_fit
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{server.server_address[1]}"
    checks = []
    try:
        with urlopen(address + "/") as response:
            assert response.status == 200 and '<textarea' in response.read().decode()
        checks.append({"check": "form available with candidate model", "passed": True})
        for text in ["", "   ", "\n\t"]:
            try:
                urlopen(address + "/classify", data=urlencode({"requirement": text}).encode())
                raise AssertionError("Empty input was accepted")
            except HTTPError as error:
                body = error.read().decode()
                assert error.code == 400 and "Enter a requirement statement before classifying." in body
                assert "Predicted category" not in body
        checks.append({"check": "empty and whitespace inputs rejected", "passed": True})
        for text in ["The system shall allow users to reset their passwords.", "The system shall respond to user requests within two seconds.", '<script>alert("test")</script>']:
            expected_label = app.LABELS[candidate.predict([text])[0]]
            with urlopen(address + "/classify", data=urlencode({"requirement": text}).encode()) as response:
                body = response.read().decode()
                assert response.status == 200
                assert html.escape(text) in body
                assert f'class="category">{expected_label}</p>' in body
                assert "This prediction may be incorrect." in body
        checks.append({"check": "valid inputs use candidate predictions, retain escaped text and show warning", "passed": True})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        candidate.fit = original_fit
    checked_files = check_files()
    verification = {
        "status": "passed", "cv_predictions_checked": len(cv_rows), "saved_test_predictions_recomputed": predictions_recomputed,
        "metric_groups_recomputed": metric_groups, "recorded_files_checked": checked_files,
        "original_and_external_data_unchanged": True, "app_model_replaced": False,
        "training_only_selected_candidate": selected, "cv_means_recomputed": cv_summary,
        "balanced_training_class_weights_FR_NFR": weights, "paired_match_changes": paired_changes,
        "local_candidate_integration_checks": checks,
        "integration_scope": "Local HTTP/HTML checks with candidate substituted in memory; application file and saved model remain unchanged. Not a concurrent-use or human-usability study.",
    }
    if not report_path.exists():
        report_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
