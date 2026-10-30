"""Frozen sentence vectors from the pinned public Arctic encoder."""

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

BASE = Path(__file__).resolve().parent


def model_files(model_dir, download=False):
    source = json.loads((BASE / "sources.json").read_text(encoding="utf-8"))["model"]
    model_dir = Path(model_dir)
    for name, expected in source["files_sha256"].items():
        path = model_dir / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing model asset {name}; run download_model.py first.")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            url = f"https://huggingface.co/{source['name']}/resolve/{source['revision']}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Model asset checksum differs: {name}")
            temporary.rename(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Model asset checksum differs: {name}")
    return model_dir


class FrozenEncoder:
    def __init__(self, model_dir):
        path = model_files(model_dir)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path / "onnx/model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=512)

    def encode(self, texts, batch_size=32, name="requirements"):
        vectors, truncated = [], 0
        for start in range(0, len(texts), batch_size):
            encodings = self.tokenizer.encode_batch(texts[start:start + batch_size])
            truncated += sum(bool(item.overflowing) for item in encodings)
            inputs = {
                "input_ids": np.asarray([item.ids for item in encodings], dtype=np.int64),
                "attention_mask": np.asarray([item.attention_mask for item in encodings], dtype=np.int64),
                "token_type_ids": np.asarray([item.type_ids for item in encodings], dtype=np.int64),
            }
            output = self.session.run(["last_hidden_state"], inputs)[0][:, 0, :]
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            if not np.isfinite(output).all() or (norms <= 0).any():
                raise ValueError("Invalid encoder output")
            vectors.append((output / norms).astype(np.float32))
            if (start // batch_size + 1) % 20 == 0:
                print(f"Encoded {name}: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        result = np.vstack(vectors)
        assert result.shape == (len(texts), 384)
        return result, {"records": len(texts), "truncated_to_512_tokens": truncated, "dimensions": 384}
