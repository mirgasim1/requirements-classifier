"""Choose training duration on validation data, then refit and evaluate."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from text_model import LABELS, RequirementClassifier, collate, predict, seed_everything, tokenize_rows

BASE = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(BASE.parent))
from check_files import check_files, recorded_path
PROJECT = BASE.parents[1]
PREVIOUS = BASE.parent / "experiment_02"
FOUR = BASE.parent / "experiment_04"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def metrics(truth, labels):
    stats = classification_report(truth, labels, labels=LABELS, output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(truth, labels)), "macro_precision": float(stats["macro avg"]["precision"]), "macro_recall": float(stats["macro avg"]["recall"]), "macro_f1": float(stats["macro avg"]["f1-score"]), "FR": stats["FR"], "NFR": stats["NFR"], "confusion_matrix": confusion_matrix(truth, labels, labels=LABELS).tolist()}


def fit(tokens, epochs, model_dir, protocol, condition, phase, output, validation_tokens=None, validation_rows=None):
    seed_everything(protocol["seed"])
    model = RequirementClassifier(model_dir)
    counts = Counter(item["label"] for item in tokens)
    weights = torch.tensor([len(tokens) / (2 * counts[label]) for label in range(2)], dtype=torch.float32)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    generator = torch.Generator().manual_seed(protocol["seed"])
    loader = DataLoader(tokens, batch_size=protocol["batch_size"], shuffle=True, generator=generator, collate_fn=collate, num_workers=0)
    settings = protocol["optimizer"]
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": settings["encoder_learning_rate"]},
        {"params": model.classifier.parameters(), "lr": settings["new_head_learning_rate"]},
    ], weight_decay=settings["weight_decay"])
    total_steps = epochs * len(loader)
    warmup = max(1, int(0.10 * total_steps))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    history, validation_predictions = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for number, (batch, labels) in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch)
            loss = loss_function(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clipping_max_norm"])
            optimizer.step()
            schedule.step()
            losses.append(float(loss.detach()))
            if number % 40 == 0:
                print(f"{condition} {phase} epoch {epoch}/{epochs}, batch {number}/{len(loader)}", flush=True)
        item = {"epoch": epoch, "mean_batch_weighted_loss": float(np.mean(losses)), "fitting_records": len(tokens), "training_class_weights_FR_NFR": weights.tolist()}
        if validation_tokens is not None:
            labels, logits = predict(model, validation_tokens)
            scores = metrics([row["label"] for row in validation_rows], labels)
            item["validation"] = scores
            validation_predictions.extend({"epoch": epoch, "source_id": row["source_id"], "reference_label": row["label"], "predicted_label": str(label), "logit_FR": float(score[0]), "logit_NFR": float(score[1])} for row, label, score in zip(validation_rows, labels, logits))
            print(f"{condition} validation epoch {epoch}: macro F1 {scores['macro_f1']:.6f}", flush=True)
        history.append(item)
        write_json(output / f"{condition}_{phase}_history.json", history)
    if validation_predictions:
        write_csv(output / f"{condition}_validation_predictions.csv", ["epoch", "source_id", "reference_label", "predicted_label", "logit_FR", "logit_NFR"], validation_predictions)
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    output = (BASE / args.output).resolve()
    if not output.is_relative_to(BASE) or output == BASE or output.exists():
        raise ValueError("Choose a new output folder inside experiment_05")
    protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    check_files()
    assert all(digest(recorded_path(PROJECT / name)) == expected for name, expected in protocol["source_files"].items())
    assert digest(args.model_dir / "model.safetensors") == protocol["weights_sha256"]
    torch.set_num_threads(protocol["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir()
    (output / "models").mkdir()
    events = []

    def event(stage):
        events.append({"stage": stage, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(output / "execution_order.json", events)
        print(stage, flush=True)

    event("Pinned inputs and pretrained weights checked before training")
    original = read_csv(recorded_path(PREVIOUS / "inputs/original_prepared.csv"))
    by_id = {row["source_row"]: {**row, "source_id": "original:" + row["source_row"]} for row in original}
    membership = read_csv(recorded_path(PREVIOUS / "inputs/original_split_membership.csv"))
    train = [by_id[row["source_row"]] for row in membership if row["subset"] == "train"]
    original_test = [by_id[row["source_row"]] for row in membership if row["subset"] == "test"]
    fitting, validation = train_test_split(train, test_size=0.20, stratify=[r["label"] for r in train], random_state=42)
    partition = [{**r, "source_id": "promise_plus:" + r["source_line"]} for r in read_csv(FOUR / "prepared/Promise_plus_partition.csv")]
    auxiliary = [r for r in partition if r["subset"] == "additional_training"]
    projects = sorted({r["project_id"] for r in auxiliary}, key=int)
    random.Random(42).shuffle(projects)
    auxiliary_validation_projects = set(projects[:math.ceil(0.20 * len(projects))])
    auxiliary_fit = [r for r in auxiliary if r["project_id"] not in auxiliary_validation_projects]
    auxiliary_validation = [r for r in auxiliary if r["project_id"] in auxiliary_validation_projects]
    conditions = {
        "original_only": {"fitting": fitting, "validation": validation, "refit": train},
        "augmented": {"fitting": fitting + auxiliary_fit, "validation": validation + auxiliary_validation, "refit": train + auxiliary},
    }
    split_rows = [{"condition": name, "subset": subset, "source_id": r["source_id"], "project_id": r.get("project_id", ""), "label": r["label"]} for name, subsets in conditions.items() for subset, rows in subsets.items() for r in rows]
    write_csv(output / "training_validation_membership.csv", ["condition", "subset", "source_id", "project_id", "label"], split_rows)
    assert not ({r["source_id"] for r in fitting} & {r["source_id"] for r in validation})
    assert not ({r["project_id"] for r in auxiliary_fit} & {r["project_id"] for r in auxiliary_validation})
    selection, truncation = {}, {}
    for name, subsets in conditions.items():
        encoded = {}
        truncation[name] = {}
        for subset, rows in subsets.items():
            encoded[subset], truncation[name][subset] = tokenize_rows(rows, args.model_dir)
        _, history = fit(encoded["fitting"], 3, args.model_dir, protocol, name, "selection", output, encoded["validation"], subsets["validation"])
        best = max(history, key=lambda item: item["validation"]["macro_f1"])
        selection[name] = {"selected_epochs": best["epoch"], "validation_macro_f1": best["validation"]["macro_f1"], "criterion": "Validation macro F1 only; earliest exact tie", "fitting_records": len(subsets["fitting"]), "validation_records": len(subsets["validation"]), "final_refit_records": len(subsets["refit"])}
        write_json(output / "epoch_selection.json", selection)
        event(f"Duration selected before final test predictions: {name}, {best['epoch']} epochs")
        model, _ = fit(encoded["refit"], best["epoch"], args.model_dir, protocol, name, "refit", output)
        save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(output / "models" / f"{name}.safetensors"))
        del model
        event(f"Final model saved without using test results: {name}")
    datasets = {
        "original_test": original_test,
        "secreq": [{**r, "source_id": "secreq:" + r["workbook_row"]} for r in read_csv(PREVIOUS / "results/SecReq_FR_NFR_prepared.csv")],
        "promise_plus_heldout_projects": [r for r in partition if r["subset"] == "external_test"],
    }
    from safetensors.torch import load_file
    comparison, predictions, results, project_metrics = [], [], {}, []
    for name in conditions:
        model = RequirementClassifier(args.model_dir)
        model.load_state_dict(load_file(str(output / "models" / f"{name}.safetensors")), strict=True)
        results[name] = {}
        for dataset, rows in datasets.items():
            tokens, truncation[f"test_{dataset}"] = tokenize_rows(rows, args.model_dir)
            labels, logits = predict(model, tokens)
            scores = metrics([r["label"] for r in rows], labels)
            results[name][dataset] = scores
            comparison.append({"model": name, "dataset": dataset, "records": len(rows), **{field: scores[field] for field in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}, "FR_recall": scores["FR"]["recall"], "NFR_recall": scores["NFR"]["recall"]})
            predictions.extend({"model": name, "dataset": dataset, "source_id": row["source_id"], "project_id": row.get("project_id", row.get("source_project", "")), "requirement_text": row["requirement_text"], "reference_label": row["label"], "predicted_label": str(label), "logit_FR": float(score[0]), "logit_NFR": float(score[1]), "matches_reference": int(row["label"] == label)} for row, label, score in zip(rows, labels, logits))
            if dataset != "original_test":
                for project in sorted({r.get("project_id", r.get("source_project")) for r in rows}):
                    indices = [i for i, r in enumerate(rows) if r.get("project_id", r.get("source_project")) == project]
                    score = metrics([rows[i]["label"] for i in indices], labels[indices])
                    project_metrics.append({"model": name, "dataset": dataset, "project_id": project, "records": len(indices), **{key: score[key] for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}})
            event(f"Fixed test scored: {name}, {dataset}, accuracy={scores['accuracy']:.6f}")
        del model
    write_csv(output / "model_comparison.csv", ["model", "dataset", "records", "accuracy", "macro_precision", "macro_recall", "macro_f1", "FR_recall", "NFR_recall"], comparison)
    write_csv(output / "predictions.csv", ["model", "dataset", "source_id", "project_id", "requirement_text", "reference_label", "predicted_label", "logit_FR", "logit_NFR", "matches_reference"], predictions)
    write_csv(output / "per_project_metrics.csv", ["model", "dataset", "project_id", "records", "accuracy", "macro_precision", "macro_recall", "macro_f1"], project_metrics)
    write_json(output / "results.json", {
        "models": results, "epoch_selection": selection, "tokenization": truncation,
        "protocol_sha256": digest(BASE / "protocol.json"),
        "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ["numpy", "torch", "transformers", "safetensors", "tokenizers", "scikit-learn"]}},
        "model_sha256": {name: digest(output / "models" / f"{name}.safetensors") for name in conditions},
        "app_model_replaced": False, "qualification": "Exploratory fine-tuning after prior outcomes were inspected. Test records and labels unchanged. Epoch selection uses validation only. Augmented training includes other PROMISE+ projects, so that condition is not an unseen-source test.",
    })
    check_files()
    event("Both training conditions completed; original application and evidence preserved")
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
