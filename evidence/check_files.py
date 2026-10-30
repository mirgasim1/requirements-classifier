"""Check fixed experiment files and resolve historical input filenames."""

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
ALIASES = {
    "evidence/experiment_02/inputs/original_prepared.csv": "evidence/prepared/FR_NFR_prepared.csv",
    "evidence/experiment_02/inputs/original_results.json": "evidence/experiment_01/results.json",
    "evidence/experiment_02/inputs/original_saved_classifier.joblib": "classifier.joblib",
    "evidence/experiment_02/inputs/original_split_membership.csv": "evidence/experiment_01/split_membership.csv",
    "evidence/experiment_02/inputs/original_test_predictions.csv": "evidence/experiment_01/test_predictions.csv",
    "evidence/experiment_01/classifier.joblib": "classifier.joblib",
    "evidence/experiment_03/results/cv_fold_membership.csv": "evidence/experiment_02/results/cv_fold_membership.csv",
}


def recorded_path(path):
    path = Path(path).resolve()
    name = path.relative_to(PROJECT).as_posix()
    return PROJECT / ALIASES.get(name, name)


def check_files():
    record = json.loads((PROJECT / "evidence/file_hashes.json").read_text(encoding="utf-8"))
    for name, expected in record["files"].items():
        path = (PROJECT / name).resolve()
        if not path.is_relative_to(PROJECT):
            raise ValueError(f"File path is outside the project: {name}")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Recorded file is missing or changed: {name}")
    return len(record["files"])


if __name__ == "__main__":
    print(f"Passed: {check_files()} recorded files checked.")
