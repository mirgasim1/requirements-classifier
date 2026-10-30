"""Select 40 cases and calculate diagnostics without fitting or relabelling."""

import csv
import difflib
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

from run_experiment import match_key


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
REVIEW = BASE / "review"
SELECTION_ID = "2026-10-22-secreq-review-v1"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    REVIEW.mkdir(exist_ok=True)
    notes = json.loads((REVIEW / "case_notes.json").read_text(encoding="utf-8"))
    external = [row for row in read_csv(RESULTS / "predictions_secreq_combined.csv") if row["model"] == "linear_svm"]
    original = [row for row in read_csv(RESULTS / "predictions_original_test.csv") if row["model"] == "linear_svm"]
    groups = defaultdict(list)
    for row in external:
        groups[row["source_project"], row["reference_label"], row["matches_reference"]].append(row)
    selected, selection_groups = [], []
    for group in sorted(groups):
        def rank(row):
            key = SELECTION_ID + "|" + "|".join(group) + "|" + row["workbook_row"]
            return hashlib.sha256(key.encode()).hexdigest()
        sample = sorted(groups[group], key=rank)[:5]
        assert len(sample) == 5
        selected.extend(sample)
        selection_groups.append({"project": group[0], "reference_label": group[1], "matches_reference": int(group[2]), "available": len(groups[group]), "selected_workbook_rows": [int(row["workbook_row"]) for row in sample]})
    selected.sort(key=lambda row: int(row["workbook_row"]))
    assert len(selected) == 40 and set(notes["notes"]) == {row["workbook_row"] for row in selected}

    model_path = RESULTS / "models/linear_svm.joblib"
    model = joblib.load(model_path)
    vectorizer, classifier = model.named_steps["tfidf"], model.named_steps["classifier"]
    assert list(classifier.classes_) == ["FR", "NFR"]
    analyse_tokens = vectorizer.build_analyzer()
    vocabulary = set(vectorizer.vocabulary_)
    feature_names = vectorizer.get_feature_names_out()
    all_diagnostics = []
    unknown_counts, unknown_record_counts = defaultdict(Counter), defaultdict(Counter)
    for dataset, rows in [("original_test", original), ("secreq_combined", external)]:
        matrix = vectorizer.transform([row["requirement_text"] for row in rows])
        for i, row in enumerate(rows):
            tokens = analyse_tokens(row["requirement_text"])
            missing = [token for token in tokens if token not in vocabulary]
            item = {
                "dataset": dataset, "source_row": row["source_row"], "workbook_row": row["workbook_row"],
                "source_project": row["source_project"], "reference_label": row["reference_label"],
                "predicted_label": row["predicted_label"], "matches_reference": int(row["matches_reference"]),
                "tokens": len(tokens), "unseen_tokens": len(missing),
                "unseen_token_fraction": len(missing) / len(tokens) if tokens else 0.0,
                "nonzero_tfidf_features": matrix.getrow(i).nnz,
                "encoding_artifact_marker": int(any(marker in row["requirement_text"] for marker in ["\u0192", "_x009d_", "\ufffd"])),
                "unseen_terms": "; ".join(sorted(set(missing))),
            }
            all_diagnostics.append(item)
            diagnostic_groups = [dataset] + ([f"secreq_{row['source_project']}"] if dataset == "secreq_combined" else [])
            for name in diagnostic_groups:
                unknown_counts[name].update(missing)
                unknown_record_counts[name].update(set(missing))
    write_csv(REVIEW / "vocabulary_records.csv", all_diagnostics)

    diagnostic_sets = {
        "original_test": [row for row in all_diagnostics if row["dataset"] == "original_test"],
        "secreq_combined": [row for row in all_diagnostics if row["dataset"] == "secreq_combined"],
        **{f"secreq_{project}": [row for row in all_diagnostics if row["source_project"] == project] for project in ["CPN", "ePurse"]},
    }
    vocabulary_summary = []
    for name, rows in diagnostic_sets.items():
        total = sum(row["tokens"] for row in rows)
        missing = sum(row["unseen_tokens"] for row in rows)
        vocabulary_summary.append({
            "dataset": name, "records": len(rows), "training_vocabulary_terms": len(vocabulary),
            "total_tokens": total, "unseen_tokens": missing, "pooled_unseen_token_fraction": missing / total if total else 0,
            "mean_record_unseen_token_fraction": float(np.mean([row["unseen_token_fraction"] for row in rows])),
            "mean_tokens_per_record": total / len(rows),
            "zero_feature_records": sum(row["nonzero_tfidf_features"] == 0 for row in rows),
            "encoding_artifact_marker_records": sum(row["encoding_artifact_marker"] for row in rows),
        })
    write_csv(REVIEW / "vocabulary_summary.csv", vocabulary_summary)
    missing_terms = [{"dataset": name, "term": term, "token_occurrences": count, "records_with_term": unknown_record_counts[name][term]} for name in diagnostic_sets for term, count in unknown_counts[name].most_common(20)]
    write_csv(REVIEW / "frequent_unseen_terms.csv", missing_terms)

    close_pairs = []
    prepared_by_id = {row["workbook_row"]: row for row in read_csv(RESULTS / "SecReq_FR_NFR_prepared.csv")}
    for i, first in enumerate(external):
        for second in external[i + 1:]:
            if first["reference_label"] == second["reference_label"]:
                continue
            first_key, second_key = match_key(first["requirement_text"]), match_key(second["requirement_text"])
            if 2 * min(len(first_key), len(second_key)) / (len(first_key) + len(second_key)) < 0.90:
                continue
            matcher = difflib.SequenceMatcher(None, first_key, second_key, autojunk=False)
            if matcher.quick_ratio() < 0.90:
                continue
            score = matcher.ratio()
            if score >= 0.90:
                close_pairs.append({
                    "first_workbook_row": first["workbook_row"], "second_workbook_row": second["workbook_row"],
                    "first_reference_label": first["reference_label"], "second_reference_label": second["reference_label"],
                    "first_specific_type": prepared_by_id[first["workbook_row"]]["specific_type"],
                    "second_specific_type": prepared_by_id[second["workbook_row"]]["specific_type"],
                    "normalized_sequence_similarity": score,
                    "first_text": first["requirement_text"], "second_text": second["requirement_text"],
                    "first_prediction": first["predicted_label"], "second_prediction": second["predicted_label"],
                })
    write_csv(REVIEW / "near_conflicting_pairs.csv", close_pairs, ["first_workbook_row", "second_workbook_row", "first_reference_label", "second_reference_label", "first_specific_type", "second_specific_type", "normalized_sequence_similarity", "first_text", "second_text", "first_prediction", "second_prediction"])

    reviewed = []
    diagnostic_lookup = {row["workbook_row"]: row for row in all_diagnostics if row["dataset"] == "secreq_combined"}
    matrix = vectorizer.transform([row["requirement_text"] for row in selected])
    margins = model.decision_function([row["requirement_text"] for row in selected])
    reconstructed = []
    for i, row in enumerate(selected):
        vector = matrix.getrow(i)
        weights = vector.data * classifier.coef_[0, vector.indices]
        terms = [(str(feature_names[index]), float(weight)) for index, weight in zip(vector.indices, weights)]
        positive = sorted((pair for pair in terms if pair[1] > 0), key=lambda pair: -pair[1])[:4]
        negative = sorted((pair for pair in terms if pair[1] < 0), key=lambda pair: pair[1])[:4]
        score = float(classifier.intercept_[0] + weights.sum())
        assert np.isclose(score, margins[i], rtol=0, atol=1e-10)
        assert ("NFR" if score > 0 else "FR") == row["predicted_label"]
        reconstructed.append(score)
        diagnostic = diagnostic_lookup[row["workbook_row"]]
        reviewed.append({
            **row, **notes["notes"][row["workbook_row"]],
            "unseen_token_fraction": diagnostic["unseen_token_fraction"],
            "encoding_artifact_marker": diagnostic["encoding_artifact_marker"],
            "svm_score_toward_NFR": score,
            "largest_NFR_contributions": json.dumps(positive, ensure_ascii=False),
            "largest_FR_contributions": json.dumps(negative, ensure_ascii=False),
            "review_status": "Provisional interpretation; not independently adjudicated",
        })
    write_csv(REVIEW / "case_review.csv", reviewed)
    summary = {
        "date": "2026-10-22", "selection_id": SELECTION_ID,
        "sample_records": len(reviewed), "strata": selection_groups,
        "selection_method": "Select five records by SHA-256 rank within each project x published label x SVM-match-status stratum. Twenty match and twenty disagree by design.",
        "sampling_limitation": "Outcome-stratified post-hoc diagnostic sample. Interpretation counts are not estimates of external error prevalence, dataset label-error rates or corrected accuracy.",
        "review_status": notes["review_status"], "definition_basis": notes["definition_basis"],
        "interpretation_counts": dict(Counter(row["interpretation"] for row in reviewed)),
        "reference_alignment_counts": dict(Counter(row["reference_alignment"] for row in reviewed)),
        "plausible_reference_disagreements_in_sample": sum(row["reference_alignment"] == "plausible" and row["matches_reference"] == "0" for row in reviewed),
        "near_conflicting_pairs_at_0_90": len(close_pairs),
        "vocabulary_diagnostics": vocabulary_summary,
        "coefficient_reconstruction_max_error": float(np.max(np.abs(np.asarray(reconstructed) - margins))),
        "score_note": "The signed linear-SVM score is not a calibrated probability. Term contributions explain the fitted linear score, not the causal reason for generalisation failure.",
        "inputs_sha256": {"predictions_external": sha256(RESULTS / "predictions_secreq_combined.csv"), "predictions_original": sha256(RESULTS / "predictions_original_test.csv"), "svm_model": sha256(model_path)},
        "reference_sources": {
            "report_definition": "50% report, Sections 1 and 1.1, printed pages 7-8",
            "dataset": "https://doi.org/10.5281/zenodo.15082900",
            "linear_score_method": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html",
        },
        "no_training_or_relabelling": True,
    }
    (REVIEW / "review_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
