"""The pretrained encoder with a binary classification layer."""

import random

import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer
from transformers import BertModel

LABELS = ["FR", "NFR"]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RequirementClassifier(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.encoder = BertModel.from_pretrained(str(model_dir), add_pooling_layer=False, local_files_only=True, attn_implementation="sdpa")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(hidden))


def tokenize_rows(rows, model_dir):
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=256)
    encoded = tokenizer.encode_batch([row["requirement_text"] for row in rows])
    return [
        {"input_ids": item.ids, "attention_mask": item.attention_mask, "token_type_ids": item.type_ids, "label": LABELS.index(row["label"])}
        for row, item in zip(rows, encoded)
    ], {"records": len(rows), "truncated_to_256_tokens": sum(bool(item.overflowing) for item in encoded)}


def collate(items):
    length = max(len(item["input_ids"]) for item in items)
    batch = {
        name: torch.tensor([item[name] + [0] * (length - len(item[name])) for item in items], dtype=torch.long)
        for name in ["input_ids", "attention_mask", "token_type_ids"]
    }
    labels = torch.tensor([item["label"] for item in items], dtype=torch.long)
    return batch, labels


def predict(model, tokens, batch_size=16):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            batch, _ = collate(tokens[start:start + batch_size])
            outputs.append(model(**batch).cpu().numpy())
    logits = np.vstack(outputs)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite prediction logits")
    return np.asarray(LABELS)[logits.argmax(axis=1)], logits
