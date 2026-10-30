"""Compare three classifiers using the original split and the SecReq sections."""

import argparse
import csv
import difflib
import hashlib
import json
import platform
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASE = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(BASE.parent))
from check_files import check_files, recorded_path
INPUT = BASE / "inputs"
LABELS = ["FR", "NFR"]
PROJECTS = {
    "SecReq Customer Premises Network (CPN)'": "CPN",
    "SecReq Common Electronic Purse (ePurse)'": "ePurse",
}
WORKBOOK_SHA256 = "216e0debeada8cd3741ec3bf9c7cd23f6c68fa98087a1200fc772406bf2df3ec"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
FIELDS = ["workbook_row", "source_project", "source_name", "requirement_text", "label", "specific_type"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_key(text):
    # Matching does not change the text passed to a classifier.
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def workbook_rows(path):
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(node.itertext()) for node in root.findall("s:si", NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for element in sheet.findall("s:sheetData/s:row", NS):
            cells = {}
            for cell in element.findall("s:c", NS):
                column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                value = cell.find("s:v", NS)
                if cell.get("t") == "s":
                    text = strings[int(value.text)] if value is not None else ""
                elif cell.get("t") == "inlineStr":
                    text = "".join(cell.find("s:is", NS).itertext())
                else:
                    text = value.text if value is not None else ""
                cells[column] = text
            if element.attrib["r"] == "1":
                expected = {"A": "Requirement", "B": "Type", "C": "Specific_Type", "D": "Security_Category", "E": "Dataset_Name"}
                if any(cells.get(column) != name for column, name in expected.items()):
                    raise ValueError("Unexpected workbook columns.")
            elif any(cells.get(column, "") for column in "ABCDE"):
                rows.append({
                    "workbook_row": int(element.attrib["r"]),
                    "source_project": PROJECTS.get(cells.get("E"), ""),
                    "source_name": cells.get("E", ""),
                    "requirement_text": cells.get("A", ""),
                    "label": cells.get("B", ""),
                    "specific_type": cells.get("C", ""),
                })
    return rows


def near_match_screen(external, original):
    original_keys = [match_key(row["requirement_text"]) for row in original]
    grams, index = [], defaultdict(list)
    for number, key in enumerate(original_keys):
        words = key.split()
        group = set(zip(words, words[1:], words[2:]))
        grams.append(group)
        for gram in group:
            index[gram].append(number)
    candidates = []
    for row in external:
        key = match_key(row["requirement_text"])
        words = key.split()
        group = set(zip(words, words[1:], words[2:]))
        matches = Counter(number for gram in group for number in index.get(gram, []))
        for number, shared in matches.most_common(10):
            jaccard = shared / len(group | grams[number])
            if jaccard < 0.30:
                continue
            ratio = difflib.SequenceMatcher(None, key, original_keys[number], autojunk=False).ratio()
            if ratio >= 0.80 or jaccard >= 0.70:
                candidates.append({
                    "workbook_row": row["workbook_row"], "source_project": row["source_project"],
                    "original_source_row": original[number]["source_row"],
                    "sequence_similarity": ratio, "word_trigram_jaccard": jaccard,
                    "external_text": row["requirement_text"], "original_text": original[number]["requirement_text"],
                })
    return candidates


def prepare(original, output):
    source = INPUT / "dcai24_src_dataset.xlsx"
    if digest(source) != WORKBOOK_SHA256:
        raise ValueError("The source workbook differs from the inspected version.")
    raw = workbook_rows(source)
    groups = defaultdict(list)
    for row in raw:
        if row["label"] in LABELS and match_key(row["requirement_text"]):
            groups[match_key(row["requirement_text"])].append(row)
    conflicts = {key for key, group in groups.items() if len({row["label"] for row in group}) > 1}
    original_keys = {match_key(row["requirement_text"]) for row in original}
    kept, excluded, seen = [], [], set()
    for row in raw:
        key = match_key(row["requirement_text"])
        reason = ""
        if not row["source_project"]:
            reason = "outside_selected_projects"
        elif row["label"] not in LABELS:
            reason = "invalid_or_missing_binary_label"
        elif not key:
            reason = "empty_requirement_text"
        elif row["requirement_text"] in {"#NAME?", "#N/A", "#VALUE!", "#REF!"}:
            reason = "spreadsheet_error_in_requirement_text"
        elif key in conflicts:
            reason = "conflicting_binary_label_group"
        elif key in original_keys:
            reason = "normalized_word_match_with_original_dataset"
        elif key in seen:
            reason = "additional_same_label_duplicate"
        if reason:
            excluded.append({**row, "exclusion_reason": reason})
        else:
            seen.add(key)
            kept.append(row)
    write_csv(output / "SecReq_FR_NFR_prepared.csv", FIELDS, kept)
    write_csv(output / "excluded_records.csv", FIELDS + ["exclusion_reason"], excluded)
    conflict_rows = [{**row, "matching_key": match_key(row["requirement_text"])} for row in raw if row["source_project"] and match_key(row["requirement_text"]) in conflicts]
    write_csv(output / "conflicting_label_records.csv", FIELDS + ["matching_key"], conflict_rows)
    near_matches = near_match_screen(kept, original)
    write_csv(output / "near_match_candidates.csv", ["workbook_row", "source_project", "original_source_row", "sequence_similarity", "word_trigram_jaccard", "external_text", "original_text"], near_matches)
    if near_matches:
        raise ValueError("Review the flagged near matches before running predictions.")
    assert len(kept) + len(excluded) == len(raw)
    assert len(seen) == len(kept) and not (seen & original_keys)
    assert all(row["label"] in LABELS and row["source_project"] for row in kept)
    assert not any(match_key(row["requirement_text"]) in conflicts for row in kept)
    summary = {
        "raw_workbook_records": len(raw),
        "selected_project_records_before_cleaning": sum(bool(row["source_project"]) for row in raw),
        "raw_project_label_counts": {project: dict(Counter(row["label"] for row in raw if row["source_project"] == project)) for project in PROJECTS.values()},
        "excluded_counts": dict(Counter(row["exclusion_reason"] for row in excluded)),
        "prepared_records": len(kept),
        "prepared_label_counts": dict(Counter(row["label"] for row in kept)),
        "prepared_project_label_counts": {project: dict(Counter(row["label"] for row in kept if row["source_project"] == project)) for project in PROJECTS.values()},
        "selected_conflicting_groups": len({match_key(row["requirement_text"]) for row in conflict_rows}),
        "remaining_normalized_matches_with_original": len(seen & original_keys),
        "near_match_candidates": len(near_matches),
        "source_sha256": digest(source),
        "prepared_sha256": digest(output / "SecReq_FR_NFR_prepared.csv"),
        "matching_rule": "NFKC normalization, casefold and Unicode word tokens. Used only for matching.",
        "duplicate_rule": "Conflicting FR/NFR groups are identified across the whole workbook. Within the selected projects, keep the first remaining same-label occurrence in workbook order.",
        "near_match_rule": "Use shared word trigrams to find ten candidate original records per external statement. Check candidates with trigram Jaccard >= 0.30; flag sequence similarity >= 0.80 or trigram Jaccard >= 0.70. This is not an exhaustive semantic-overlap test.",
        "input_text": "The Requirement cell only. Text and supplied Type labels of retained records are unchanged. IDs, source names and category fields are not classifier inputs.",
        "label_review": "No independent relabelling of the retained requirements. Disagreements measure differences from the published reference labels, not necessarily proven model mistakes.",
        "scope": "The CPN and ePurse sections only. These are two security-heavy specifications, not the whole merged dataset or a representative sample of all software domains.",
    }
    write_json(output / "preparation_summary.json", summary)
    return kept, summary


def metrics(truth, predictions):
    report = classification_report(truth, predictions, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "FR": report["FR"], "NFR": report["NFR"],
        "confusion_matrix": confusion_matrix(truth, predictions, labels=LABELS).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results", help="A new output folder inside this experiment folder.")
    args = parser.parse_args()
    output = (BASE / args.output).resolve()
    if not output.is_relative_to(BASE) or output == BASE or output.exists():
        raise ValueError("Choose a new output folder inside this experiment folder. Existing folders are not replaced.")
    check_files()
    sources = json.loads((BASE / "sources.json").read_text(encoding="utf-8"))
    for name, expected in sources["input_sha256"].items():
        if digest(recorded_path(INPUT / name)) != expected:
            raise ValueError(f"Input file changed: {name}")
    output.mkdir(parents=True)
    events = []

    def event(name):
        events.append({"stage": name, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(output / "execution_order.json", events)
        print(name, flush=True)

    original = read_csv(recorded_path(INPUT / "original_prepared.csv"))
    train, test = train_test_split(original, test_size=0.20, random_state=42, stratify=[row["label"] for row in original])
    membership = read_csv(recorded_path(INPUT / "original_split_membership.csv"))
    for subset, rows in [("train", train), ("test", test)]:
        assert [row["source_row"] for row in rows] == [row["source_row"] for row in membership if row["subset"] == subset]
    assert (len(train), len(test)) == (4748, 1188)
    assert not ({row["source_row"] for row in train} & {row["source_row"] for row in test})
    external, preparation = prepare(original, output)
    event("Preparation finished before training and evaluation")

    xtrain = [row["requirement_text"] for row in train]
    ytrain = [row["label"] for row in train]
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(xtrain, ytrain))
    fold_rows = []
    for number, (_, validation) in enumerate(folds, start=1):
        fold_rows.extend({"source_row": train[index]["source_row"], "label": train[index]["label"], "validation_fold": number} for index in validation)
    write_csv(output / "cv_fold_membership.csv", ["source_row", "label", "validation_fold"], fold_rows)

    classifiers = {
        "linear_svm": LinearSVC(C=1.0, class_weight=None, random_state=42, max_iter=10000),
        "logistic_regression": LogisticRegression(C=1.0, class_weight=None, solver="liblinear", random_state=42, max_iter=10000),
        "multinomial_naive_bayes": MultinomialNB(alpha=1.0, fit_prior=True),
    }
    models = {
        name: Pipeline([("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)), ("classifier", classifier)])
        for name, classifier in classifiers.items()
    }
    cv, cv_rows = {}, []
    for name, model in models.items():
        scores = cross_val_score(model, xtrain, ytrain, cv=folds, scoring="f1_macro", n_jobs=1, error_score="raise")
        cv[name] = {"fold_macro_f1": scores.tolist(), "mean_macro_f1": float(scores.mean()), "std_macro_f1_population": float(scores.std(ddof=0))}
        cv_rows.extend({"model": name, "fold": number, "macro_f1": float(score)} for number, score in enumerate(scores, start=1))
        event(f"Training-only cross-validation finished: {name}")
    selected = max(cv, key=lambda name: cv[name]["mean_macro_f1"])
    write_csv(output / "cv_scores.csv", ["model", "fold", "macro_f1"], cv_rows)
    write_json(output / "model_selection.json", {
        "criterion": "Highest mean five-fold macro F1 within the original training data only",
        "tie_rule": "First model in the recorded classifier order if means are exactly equal",
        "selected_model": selected, "cross_validation": cv,
        "external_data_used_for_selection": False,
    })
    event(f"Model selected before either test set was evaluated: {selected}")

    datasets = {"original_test": test, "secreq_combined": external}
    predictions_for_csv = {name: [] for name in datasets}
    comparisons, class_rows, matrix_rows, disagreements = [], [], [], []
    results = {}
    (output / "models").mkdir()
    recorded_predictions = {row["source_row"]: row["predicted_label"] for row in read_csv(recorded_path(INPUT / "original_test_predictions.csv"))}
    saved_model = joblib.load(recorded_path(INPUT / "original_saved_classifier.joblib"))
    app_matches_svm = {}

    def add_scores(model_name, dataset_name, rows, predictions):
        scores = metrics([row["label"] for row in rows], predictions)
        comparisons.append({"model": model_name, "dataset": dataset_name, "records": len(rows), "cv_macro_f1_mean": cv[model_name]["mean_macro_f1"], **{field: scores[field] for field in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}})
        for label in LABELS:
            class_rows.append({"model": model_name, "dataset": dataset_name, "label": label, **scores[label]})
        for i, actual in enumerate(LABELS):
            for j, predicted in enumerate(LABELS):
                matrix_rows.append({"model": model_name, "dataset": dataset_name, "reference_label": actual, "predicted_label": predicted, "count": scores["confusion_matrix"][i][j]})
        return scores

    for name, model in models.items():
        model.fit(xtrain, ytrain)
        model_results = {}
        for dataset, rows in datasets.items():
            texts = [row["requirement_text"] for row in rows]
            predictions = model.predict(texts)
            model_results[dataset] = add_scores(name, dataset, rows, predictions)
            if name == "linear_svm":
                app_matches_svm[dataset] = bool(np.array_equal(predictions, saved_model.predict(texts)))
                if not app_matches_svm[dataset]:
                    raise ValueError("The refitted SVM differs from the saved application model.")
                if dataset == "original_test":
                    assert all(prediction == recorded_predictions[row["source_row"]] for row, prediction in zip(rows, predictions))
            for row, prediction in zip(rows, predictions):
                item = {
                    "model": name, "source_row": row.get("source_row", ""),
                    "workbook_row": row.get("workbook_row", ""), "source_project": row.get("source_project", ""),
                    "requirement_text": row["requirement_text"], "reference_label": row["label"],
                    "predicted_label": str(prediction), "matches_reference": int(row["label"] == prediction),
                }
                predictions_for_csv[dataset].append(item)
                if dataset == "secreq_combined" and name == selected and row["label"] != prediction:
                    disagreements.append({**item, "manual_review_notes": ""})
            if dataset == "secreq_combined":
                for project in PROJECTS.values():
                    indices = [i for i, row in enumerate(rows) if row["source_project"] == project]
                    project_rows = [rows[i] for i in indices]
                    model_results[f"secreq_{project}"] = add_scores(name, f"secreq_{project}", project_rows, predictions[indices])
        results[name] = {"cross_validation": cv[name], **model_results}
        joblib.dump(model, output / "models" / f"{name}.joblib", compress=3)
        event(f"Fitting and test evaluation finished: {name}")

    prediction_fields = ["model", "source_row", "workbook_row", "source_project", "requirement_text", "reference_label", "predicted_label", "matches_reference"]
    for dataset, rows in predictions_for_csv.items():
        write_csv(output / f"predictions_{dataset}.csv", prediction_fields, rows)
    write_csv(output / "selected_model_disagreements.csv", prediction_fields + ["manual_review_notes"], disagreements)
    write_csv(output / "model_comparison.csv", ["model", "dataset", "records", "cv_macro_f1_mean", "accuracy", "macro_precision", "macro_recall", "macro_f1"], comparisons)
    write_csv(output / "per_class_metrics.csv", ["model", "dataset", "label", "precision", "recall", "f1-score", "support"], class_rows)
    write_csv(output / "confusion_matrices.csv", ["model", "dataset", "reference_label", "predicted_label", "count"], matrix_rows)
    majority = Counter(ytrain).most_common(1)[0][0]
    baselines = {name: metrics([row["label"] for row in rows], [majority] * len(rows)) for name, rows in datasets.items()}
    original_result = json.loads((recorded_path(INPUT / "original_results.json")).read_text(encoding="utf-8"))
    assert np.isclose(results["linear_svm"]["original_test"]["accuracy"], original_result["accuracy"])
    event("All prediction and metric files written")
    result = {
        "experiment_date": "2026-10-22", "experiment": "Original-data model comparison and SecReq external case study",
        "selected_model": selected,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__},
        "training_records": len(train), "original_test_records": len(test), "external_test_records": len(external),
        "training_counts": dict(Counter(ytrain)), "original_test_counts": dict(Counter(row["label"] for row in test)),
        "tfidf_settings": models[selected].named_steps["tfidf"].get_params(),
        "classifier_settings": {name: classifier.get_params() for name, classifier in classifiers.items()},
        "validation": "Five identical stratified folds within the original training data only; shuffle=True, random_state=42. Each pipeline refits TF-IDF inside each fold. Three fixed configurations, with no parameter search.",
        "input_features": "Requirement text only; no IDs, labels, source names or subcategory columns.",
        "models": results, "training_majority_label": majority, "training_majority_baselines": baselines,
        "external_disagreements_selected_model": len(disagreements), "preparation": preparation,
        "confusion_matrix_label_order": LABELS, "confusion_matrix_orientation": "Rows are reference labels; columns are predicted labels",
        "input_sha256": {name: digest(recorded_path(INPUT / name)) for name in sorted(sources["input_sha256"])},
        "verification": {"original_split_and_order_match": True, "svm_matches_recorded_original_predictions": True, "saved_application_matches_refitted_svm": app_matches_svm},
        "limitations": [
            "External evaluation covers the CPN and ePurse sections only, not the complete merged workbook or all software domains.",
            "These specifications contain many security requirements, and the supplied labels reflect the source authors' classification conventions.",
            "The supplied FR/NFR labels have not been independently reannotated. A disagreement is not automatically a proven classifier error.",
            "Exact normalized overlap and a limited near-match screen were checked. Complete semantic or underlying project independence from the original mixed corpus is not established.",
            "Different wording, label conventions and category distributions can affect scores. This experiment does not isolate a causal domain-shift effect.",
            "The original holdout was already inspected during the 50% stage. It is not a newly untouched test set.",
            "The curated dataset version was released in 2025. Its underlying specifications are older; the release date is not the collection date.",
            "Three fixed model configurations are compared, not every possible model or tuning setting.",
            "No user study, measured time saving or concurrent-use evaluation is included in this experiment.",
        ],
    }
    write_json(output / "results.json", result)
    print(json.dumps({"output": str(output), "selected_model": selected, "external_records": len(external), "external_accuracy": results[selected]["secreq_combined"]["accuracy"], "external_macro_f1": results[selected]["secreq_combined"]["macro_f1"], "external_disagreements": len(disagreements)}, indent=2))


if __name__ == "__main__":
    main()
