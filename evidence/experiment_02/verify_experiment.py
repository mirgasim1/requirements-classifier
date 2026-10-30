"""Check saved predictions, metrics and data records without retraining."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import joblib

from run_experiment import FIELDS, LABELS, INPUT, match_key, workbook_rows


BASE = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(BASE.parent))
from check_files import check_files, recorded_path
PROJECT = BASE.parents[1]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_metrics(rows, predictions):
    counts = Counter((row["label"], prediction) for row, prediction in zip(rows, predictions))
    class_metrics = {}
    for label in LABELS:
        correct = counts[label, label]
        actual = sum(counts[label, other] for other in LABELS)
        predicted = sum(counts[other, label] for other in LABELS)
        precision = correct / predicted if predicted else 0.0
        recall = correct / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[label] = {"precision": precision, "recall": recall, "f1-score": f1, "support": actual}
    return {
        "accuracy": sum(counts[label, label] for label in LABELS) / len(rows),
        "macro_precision": sum(class_metrics[label]["precision"] for label in LABELS) / 2,
        "macro_recall": sum(class_metrics[label]["recall"] for label in LABELS) / 2,
        "macro_f1": sum(class_metrics[label]["f1-score"] for label in LABELS) / 2,
        "confusion_matrix": [[counts[actual, predicted] for predicted in LABELS] for actual in LABELS],
        **class_metrics,
    }


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12), (actual, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    output = (BASE / args.results).resolve()
    if not output.is_relative_to(BASE) or not output.is_dir():
        raise ValueError("Choose an existing result folder within this experiment.")
    check_files()
    sources = json.loads((BASE / "sources.json").read_text(encoding="utf-8"))
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    for name, expected in sources["input_sha256"].items():
        assert sha256(recorded_path(INPUT / name)) == expected

    original = read_csv(recorded_path(INPUT / "original_prepared.csv"))
    original_by_id = {row["source_row"]: row for row in original}
    split = read_csv(recorded_path(INPUT / "original_split_membership.csv"))
    train = [original_by_id[row["source_row"]] for row in split if row["subset"] == "train"]
    test = [original_by_id[row["source_row"]] for row in split if row["subset"] == "test"]
    assert len(train) == 4748 and len(test) == 1188
    assert not ({row["source_row"] for row in train} & {row["source_row"] for row in test})
    raw = workbook_rows(INPUT / "dcai24_src_dataset.xlsx")
    raw_by_id = {str(row["workbook_row"]): row for row in raw}
    prepared = read_csv(output / "SecReq_FR_NFR_prepared.csv")
    excluded = read_csv(output / "excluded_records.csv")
    assert len(prepared) == 253 and len(prepared) + len(excluded) == len(raw)
    assert Counter(row["label"] for row in prepared) == {"FR": 106, "NFR": 147}
    kept_ids = {row["workbook_row"] for row in prepared}
    excluded_ids = {row["workbook_row"] for row in excluded}
    assert len(kept_ids) == len(prepared) and len(excluded_ids) == len(excluded)
    assert not (kept_ids & excluded_ids) and kept_ids | excluded_ids == set(raw_by_id)
    for row in prepared:
        source = raw_by_id[row["workbook_row"]]
        for field in FIELDS:
            assert str(source[field]) == row[field]
    assert len({match_key(row["requirement_text"]) for row in prepared}) == len(prepared)
    assert not ({match_key(row["requirement_text"]) for row in prepared} & {match_key(row["requirement_text"]) for row in original})
    assert Counter(row["exclusion_reason"] for row in excluded) == {"outside_selected_projects": 3148, "conflicting_binary_label_group": 34, "additional_same_label_duplicate": 47}

    folds = read_csv(output / "cv_fold_membership.csv")
    assert len(folds) == len(train)
    assert Counter(row["source_row"] for row in folds) == Counter(row["source_row"] for row in train)
    assert {row["validation_fold"] for row in folds} == {"1", "2", "3", "4", "5"}
    cv = read_csv(output / "cv_scores.csv")
    selection = json.loads((output / "model_selection.json").read_text(encoding="utf-8"))
    means = {}
    for model in results["models"]:
        scores = [float(row["macro_f1"]) for row in cv if row["model"] == model]
        assert len(scores) == 5
        means[model] = sum(scores) / len(scores)
        close(means[model], selection["cross_validation"][model]["mean_macro_f1"])
    assert results["selected_model"] == selection["selected_model"] == max(means, key=means.get)
    assert selection["external_data_used_for_selection"] is False
    stages = json.loads((output / "execution_order.json").read_text(encoding="utf-8"))
    selection_stage = next(i for i, event in enumerate(stages) if event["stage"].startswith("Model selected"))
    evaluation_stages = [i for i, event in enumerate(stages) if event["stage"].startswith("Fitting and test evaluation")]
    assert evaluation_stages and selection_stage < min(evaluation_stages)

    summaries = {(row["model"], row["dataset"]): row for row in read_csv(output / "model_comparison.csv")}
    class_rows = {(row["model"], row["dataset"], row["label"]): row for row in read_csv(output / "per_class_metrics.csv")}
    matrix_rows = {(row["model"], row["dataset"], row["reference_label"], row["predicted_label"]): int(row["count"]) for row in read_csv(output / "confusion_matrices.csv")}
    datasets = {"original_test": test, "secreq_combined": prepared}
    checked_predictions, selected_disagreements = 0, set()
    app_model = joblib.load(recorded_path(INPUT / "original_saved_classifier.joblib"))
    for name, model_result in results["models"].items():
        model = joblib.load(output / "models" / f"{name}.joblib")
        for dataset, rows in datasets.items():
            stored = [row for row in read_csv(output / f"predictions_{dataset}.csv") if row["model"] == name]
            assert len(stored) == len(rows)
            actual_predictions = list(model.predict([row["requirement_text"] for row in rows]))
            assert actual_predictions == [row["predicted_label"] for row in stored]
            if name == "linear_svm":
                assert actual_predictions == list(app_model.predict([row["requirement_text"] for row in rows]))
            for source, prediction in zip(rows, stored):
                assert source["requirement_text"] == prediction["requirement_text"]
                assert source["label"] == prediction["reference_label"]
                assert int(prediction["matches_reference"]) == int(prediction["reference_label"] == prediction["predicted_label"])
                if dataset == "secreq_combined" and name == results["selected_model"] and prediction["matches_reference"] == "0":
                    selected_disagreements.add(prediction["workbook_row"])
            checked_predictions += len(stored)
            groups = {dataset: (rows, actual_predictions)}
            if dataset == "secreq_combined":
                for project in ["CPN", "ePurse"]:
                    indices = [i for i, row in enumerate(rows) if row["source_project"] == project]
                    groups[f"secreq_{project}"] = ([rows[i] for i in indices], [actual_predictions[i] for i in indices])
            for group, (group_rows, predictions) in groups.items():
                calculated = independent_metrics(group_rows, predictions)
                close(summaries[name, group]["records"], len(group_rows))
                for field in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
                    close(calculated[field], model_result[group][field])
                    close(calculated[field], summaries[name, group][field])
                assert calculated["confusion_matrix"] == model_result[group]["confusion_matrix"]
                for i, actual in enumerate(LABELS):
                    for j, predicted in enumerate(LABELS):
                        assert calculated["confusion_matrix"][i][j] == matrix_rows[name, group, actual, predicted]
                for label in LABELS:
                    for field, value in calculated[label].items():
                        close(value, class_rows[name, group, label][field])
    assert checked_predictions == 4323 and len(summaries) == 12
    errors = read_csv(output / "selected_model_disagreements.csv")
    assert len(errors) == len(selected_disagreements) == results["external_disagreements_selected_model"]
    assert {row["workbook_row"] for row in errors} == selected_disagreements
    assert all(not row["manual_review_notes"] for row in errors)

    baseline_rows = []
    baseline_sets = {**datasets, **{f"secreq_{project}": [row for row in prepared if row["source_project"] == project] for project in ["CPN", "ePurse"]}}
    for dataset, rows in baseline_sets.items():
        for label in LABELS:
            scores = independent_metrics(rows, [label] * len(rows))
            baseline_rows.append({"dataset": dataset, "constant_prediction": label, "records": len(rows), **{field: scores[field] for field in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}})
    with (output / "constant_label_baselines.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(baseline_rows)
    verification = {
        "status": "passed", "input_hashes_match": True, "canonical_original_input_hashes_match": True,
        "raw_text_and_labels_preserved": True, "all_source_rows_accounted_for": True,
        "prepared_records": len(prepared), "prepared_label_counts": dict(Counter(row["label"] for row in prepared)),
        "normalized_overlap_with_original": 0, "original_split_preserved": True,
        "training_only_model_selection_checked": True, "saved_predictions_recomputed": checked_predictions,
        "metric_groups_independently_recomputed": len(summaries),
        "selected_model_disagreements": len(errors), "saved_application_predictions_match_svm": True,
        "constant_label_baselines_calculated": len(baseline_rows),
        "scope": "Structural, prediction and numerical checks. This does not independently establish semantic label correctness, independence of all underlying projects or generalisation to all domains.",
    }
    if not (output / "verification.json").exists():
        (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
