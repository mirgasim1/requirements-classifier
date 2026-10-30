"""Check the saved fine-tuned candidates with the unchanged application handler."""

import argparse
import hashlib
import html
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import torch

from run_candidate_app import CandidatePredictor, application

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    output = (BASE / args.results).resolve()
    if not output.is_relative_to(BASE):
        raise ValueError("Results must be inside experiment_05")
    target = output / "candidate_application_checks.json"
    protected = {name: digest(PROJECT / name) for name in ["app.py", "index.html", "classifier.joblib", "requirements.txt", "RUN.cmd"]}
    torch.set_num_threads(4)
    report = {}
    for condition in ["original_only", "augmented"]:
        predictor = CandidatePredictor(args.model_dir, output / "models" / f"{condition}.safetensors")
        calls = []

        class Tracked:
            def predict(self, texts):
                calls.extend(texts)
                return predictor.predict(texts)

        app = application(Tracked())
        server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        address = f"http://127.0.0.1:{server.server_address[1]}"
        checks = []
        try:
            with urlopen(address + "/") as response:
                page = response.read().decode()
                assert response.status == 200 and page.count("<textarea") == 1 and "Classify" in page
            checks.append("one statement field and classify button present")
            for text in ["", "   ", "\n\t"]:
                before = len(calls)
                try:
                    urlopen(address + "/classify", data=urlencode({"requirement": text}).encode())
                    raise AssertionError("Empty input accepted")
                except HTTPError as error:
                    page = error.read().decode()
                    assert error.code == 400 and "Enter a requirement statement before classifying." in page
                    assert "Predicted category" not in page and len(calls) == before
            checks.append("empty and whitespace-only input rejected before model call")
            for text in ["The system shall allow users to reset their passwords.", "The system shall respond to user requests within two seconds.", '<script>alert("test")</script>']:
                category = app.LABELS[predictor.predict([text])[0]]
                with urlopen(address + "/classify", data=urlencode({"requirement": text}).encode()) as response:
                    page = response.read().decode()
                    assert response.status == 200 and f'class="category">{category}</p>' in page
                    assert html.escape(text) in page and "This prediction may be incorrect." in page
            checks.append("saved model output displayed with retained escaped input and warning")
            previous_model = app.MODEL

            class FailedModel:
                def predict(self, texts):
                    raise RuntimeError("simulated prediction failure")

            app.MODEL = FailedModel()
            try:
                urlopen(address + "/classify", data=urlencode({"requirement": "A valid statement."}).encode())
                raise AssertionError("Prediction failure did not return an error")
            except HTTPError as error:
                page = error.read().decode()
                assert error.code == 500 and "Classification could not be completed." in page
                assert "Predicted category" not in page
            finally:
                app.MODEL = previous_model
            checks.append("prediction failure returns an error without a category")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        report[condition] = {"status": "passed", "checks": checks, "scope": "Local HTTP and returned HTML; no usability or concurrent-load claim. Output categories are compared with the saved model, not independently relabelled gold examples."}
        print("Candidate interface verified:", condition, flush=True)
    assert all(digest(PROJECT / name) == expected for name, expected in protected.items())
    report["original_application_files_unchanged"] = True
    if not target.exists():
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
