"""Run a saved fine-tuned candidate with the existing browser interface."""

import argparse
import importlib.util
import threading
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from text_model import LABELS, RequirementClassifier

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]


class CandidatePredictor:
    def __init__(self, model_dir, weights):
        self.model = RequirementClassifier(model_dir)
        self.model.load_state_dict(load_file(str(weights)), strict=True)
        self.model.eval()
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=256)

    def predict(self, texts):
        items = self.tokenizer.encode_batch(texts)
        inputs = {
            "input_ids": torch.tensor([item.ids for item in items], dtype=torch.long),
            "attention_mask": torch.tensor([item.attention_mask for item in items], dtype=torch.long),
            "token_type_ids": torch.tensor([item.type_ids for item in items], dtype=torch.long),
        }
        with torch.inference_mode():
            logits = self.model(**inputs).cpu().numpy()
        return np.asarray(LABELS)[logits.argmax(axis=1)]


def application(predictor):
    spec = importlib.util.spec_from_file_location("requirements_candidate_app", PROJECT / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    app.MODEL = predictor
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["original_only", "augmented"], required=True)
    parser.add_argument("--model-dir", type=Path, default=BASE / "model_assets")
    parser.add_argument("--results", type=Path, default=BASE / "results")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    predictor = CandidatePredictor(args.model_dir, args.results / "models" / f"{args.condition}.safetensors")
    app = application(predictor)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), app.Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Candidate {args.condition} running at {url}. Keep this window open; press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
