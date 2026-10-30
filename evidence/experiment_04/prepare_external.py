"""Reproduce the published-data parsing, overlap exclusions and project split."""

import argparse
import csv
import difflib
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(BASE.parent))
from check_files import check_files, recorded_path
PREVIOUS = BASE.parent / "experiment_02"
MAPPING = {name: "NFR" for name in ["A", "L", "LF", "MN", "O", "PE", "SC", "SE", "US", "FT", "PO"]}
MAPPING["F"] = "FR"
FIELDS = ["source_line", "project_id", "requirement_text", "source_label", "label", "subset"]


def key(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def overlap_matches(rows, references):
    reference_keys = {key(r["requirement_text"]) for r in references}
    grams, index = [], defaultdict(set)
    for n, row in enumerate(references):
        words = key(row["requirement_text"]).split()
        group = set(zip(words, words[1:], words[2:]))
        grams.append(group)
        for gram in group:
            index[gram].add(n)
    reasons = {}
    for row in rows:
        text = key(row["requirement_text"])
        if text in reference_keys:
            reasons[row["source_line"]] = "normalized_match_with_original_dataset"
            continue
        words = text.split()
        group = set(zip(words, words[1:], words[2:]))
        common = Counter(n for gram in group for n in index.get(gram, []))
        for n, shared in sorted(common.items(), key=lambda item: (-item[1], item[0]))[:10]:
            jaccard = shared / len(group | grams[n])
            if jaccard < 0.30:
                continue
            similarity = difflib.SequenceMatcher(None, text, key(references[n]["requirement_text"]), autojunk=False).ratio()
            if jaccard >= 0.70 or similarity >= 0.80:
                reasons[row["source_line"]] = "near_match_with_original_dataset"
                break
    return reasons


def prepare():
    lines = (BASE / "inputs/Promise_plus_original.arff").read_text(encoding="utf-8-sig").splitlines()
    raw, records, current, start, active = [], [], [], None, False
    for number, line in enumerate(lines, start=1):
        if line.strip().upper() == "@DATA":
            active = True
            continue
        if not active or not line.strip() or line.lstrip().startswith("%"):
            continue
        if re.match(r"^\s*\d+\s*,\s*'", line):
            if current:
                records.append((start, "\n".join(current)))
            start, current = number, [line]
        else:
            current.append(line)
    if current:
        records.append((start, "\n".join(current)))
    for number, record in records:
        match = re.fullmatch(r"\s*(\d+)\s*,\s*'(.*)'\s*,?\s*([A-Z]+)\s*", record, flags=re.S)
        if not match:
            raise ValueError(f"Unexpected unparsed source record at line {number}")
        project, text, label = match.groups()
        raw.append({"source_line": number, "project_id": project, "requirement_text": text, "source_label": label, "label": MAPPING.get(label, "")})
    # Replay the archived screening decisions and verify each against raw text.
    # The initial top-ten search did not specify how equal shortlist counts were
    # ordered. Replaying its verified pairs preserves the already fixed sample.
    original = read_csv(recorded_path(PREVIOUS / "inputs/original_prepared.csv"))
    original_keys = {key(row["requirement_text"]) for row in original}
    original_by_id = {row["source_row"]: row for row in original}
    raw_by_line = {str(row["source_line"]): row for row in raw}
    reasons = {row["source_line"]: "normalized_match_with_original_dataset" for row in raw if key(row["requirement_text"]) in original_keys}
    for pair in read_csv(BASE / "prepared/Promise_plus_near_matches.csv"):
        row = raw_by_line[pair["source_line"]]
        reference = original_by_id[pair["original_source_row"]]
        assert row["requirement_text"] == pair["requirement_text"]
        assert reference["requirement_text"] == pair["original_text"]
        words_a, words_b = key(row["requirement_text"]).split(), key(reference["requirement_text"]).split()
        grams_a = set(zip(words_a, words_a[1:], words_a[2:]))
        grams_b = set(zip(words_b, words_b[1:], words_b[2:]))
        jaccard = len(grams_a & grams_b) / len(grams_a | grams_b)
        similarity = difflib.SequenceMatcher(None, key(row["requirement_text"]), key(reference["requirement_text"]), autojunk=False).ratio()
        assert jaccard >= 0.30 and (similarity >= 0.80 or jaccard >= 0.70)
        assert row["source_line"] not in reasons
        reasons[row["source_line"]] = "near_match_with_original_dataset"
    overlap_projects = {r["project_id"] for r in raw if r["source_line"] in reasons}
    groups = defaultdict(list)
    for row in raw:
        groups[key(row["requirement_text"])].append(row)
    conflicts = {text for text, group in groups.items() if len({r["label"] for r in group}) > 1}
    kept, excluded, seen = [], [], set()
    for row in raw:
        text = key(row["requirement_text"])
        reason = reasons.get(row["source_line"], "")
        if not reason:
            if not row["label"]:
                reason = "unknown_label"
            elif not text:
                reason = "empty_text"
            elif row["project_id"] in overlap_projects:
                reason = "project_has_observed_overlap_with_original"
            elif text in conflicts:
                reason = "conflicting_binary_label_group"
            elif text in seen:
                reason = "additional_same_label_duplicate"
        if reason:
            excluded.append({**row, "reason": reason})
        else:
            kept.append(row)
            seen.add(text)
    projects = sorted({r["project_id"] for r in kept}, key=int)
    random.Random(42).shuffle(projects)
    test_projects = set(projects[:(len(projects) + 1) // 2])
    for row in kept:
        row["subset"] = "external_test" if row["project_id"] in test_projects else "additional_training"
    additional = [r for r in kept if r["subset"] == "additional_training"]
    test = [r for r in kept if r["subset"] == "external_test"]
    auxiliary_near = overlap_matches(additional, test + read_csv(PREVIOUS / "results/SecReq_FR_NFR_prepared.csv"))
    final = [r for r in kept if r["source_line"] not in auxiliary_near]
    return raw, final, excluded, auxiliary_near


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Optional new folder inside experiment_04 for a reproduced partition")
    args = parser.parse_args()
    raw, partition, excluded, auxiliary_near = prepare()
    expected = read_csv(BASE / "prepared/Promise_plus_partition.csv")
    assert [{f: str(r[f]) for f in FIELDS} for r in partition] == expected
    stored_exclusions = read_csv(BASE / "prepared/Promise_plus_exclusions.csv")
    assert [{f: str(r[f]) for f in FIELDS[:-1] + ["reason"]} for r in excluded] == stored_exclusions
    assert len(auxiliary_near) == len(read_csv(BASE / "prepared/additional_training_overlap_exclusions.csv"))
    if args.output:
        output = (BASE / args.output).resolve()
        if not output.is_relative_to(BASE) or output.exists():
            raise ValueError("Choose a new output directory inside experiment_04")
        output.mkdir()
        with (output / "Promise_plus_partition.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(partition)
    print(f"Preparation reproduced: {len(raw)} raw, {len(excluded)} excluded, {len(partition)} partitioned; {len(auxiliary_near)} additional training overlaps.")
