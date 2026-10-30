"""Check the eight application requirements in Section 2.1 through local HTTP."""

import hashlib
import html
import importlib.util
import json
import threading
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np


BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]


class FormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.textareas, self.forms, self.labels = [], [], []

    def handle_starttag(self, tag, attrs):
        if tag == "textarea":
            self.textareas.append(dict(attrs))
        elif tag == "form":
            self.forms.append(dict(attrs))
        elif tag == "label":
            self.labels.append(dict(attrs))


def main():
    model_path = PROJECT / "classifier.joblib"
    model_hash_before = hashlib.sha256(model_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("application_check_75", PROJECT / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    fitted = app.MODEL
    vectorizer = next(step for step in fitted.named_steps.values() if hasattr(step, "vocabulary_"))
    classifier = list(fitted.named_steps.values())[-1]
    vocabulary_before = dict(vectorizer.vocabulary_)
    idf_before = vectorizer.idf_.copy()
    examples = [
        "The system shall allow users to reset their passwords.",
        "The system shall respond to user requests within two seconds.",
        '</textarea><script>alert("test")</script> The system shall display text.',
    ]
    expected = [app.LABELS[label] for label in fitted.predict(examples)]
    transform_calls = []
    transform = vectorizer.transform

    def tracked_transform(*args, **kwargs):
        transform_calls.append(list(args[0]))
        return transform(*args, **kwargs)

    def block_fit(*args, **kwargs):
        raise AssertionError("A prediction request tried to fit the saved model again.")

    vectorizer.transform = tracked_transform
    vectorizer.fit = block_fit
    vectorizer.fit_transform = block_fit
    classifier.fit = block_fit
    fitted.fit = block_fit

    class CountingModel:
        def __init__(self):
            self.calls = []

        def predict(self, texts):
            self.calls.append(list(texts))
            return fitted.predict(texts)

    counting_model = CountingModel()
    app.MODEL = counting_model
    server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    def request(path="/", statement=None):
        data = None if statement is None else urlencode({"requirement": statement}).encode("utf-8")
        try:
            response = urlopen(base_url + path, data=data, timeout=10)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.read().decode("utf-8")

    checks, observations = [], []
    try:
        status, page = request()
        parser = FormParser()
        parser.feed(page)
        assert status == 200 and len(parser.textareas) == 1
        assert parser.textareas[0].get("name") == "requirement"
        checks.append({"requirement": "FR1", "status": "passed", "evidence": "GET / returns one textarea for one submitted requirement statement."})
        assert any(form.get("action") == "/classify" and form.get("method", "").lower() == "post" for form in parser.forms)
        assert '>Classify</button>' in page
        checks.append({"requirement": "FR2", "status": "passed", "evidence": "The form has a submit button and posts to /classify; actual POST requests were checked below."})
        for statement in ["", "   ", "\n\t"]:
            status, body = request("/classify", statement)
            assert status == 400 and "Enter a requirement statement before classifying." in body and 'class="category"' not in body
        assert counting_model.calls == [] and transform_calls == []
        checks.append({"requirement": "FR4", "status": "passed", "evidence": "Empty, spaces-only and newline/tab-only submissions return 400 and an input message. The model and vectorizer are not called."})

        returned_pages = []
        for statement, category in zip(examples, expected):
            status, body = request("/classify", statement)
            assert status == 200 and f'class="category">{category}</p>' in body
            assert html.escape(statement) in body
            assert "This prediction may be incorrect." in body
            returned_pages.append(body)
            observations.append({"statement": statement, "displayed_category": category, "matches_saved_model": True, "submitted_text_preserved": True})
        checks.append({"requirement": "FR3", "status": "passed", "evidence": "Three valid POST submissions display the saved model's Functional/Non-functional category and retain the input. This verifies model integration, not semantic classification accuracy."})
        assert html.escape(examples[1]) in returned_pages[1] and html.escape(examples[0]) not in returned_pages[1]
        checks.append({"requirement": "FR5", "status": "passed", "evidence": "An edited second submission is processed and displayed with the new text; the old submitted text is not returned in its place."})
        assert examples[2] not in returned_pages[2] and "&lt;script&gt;" in returned_pages[2]

        class FailingModel:
            def predict(self, texts):
                raise RuntimeError("Deliberate verification failure")

        app.MODEL = FailingModel()
        status, body = request("/classify", "The system shall display a result.")
        assert status == 500 and "Classification could not be completed." in body and 'class="category"' not in body
        checks.append({"requirement": "FR6", "status": "passed", "evidence": "A simulated prediction failure returns 500 and an error message without a predicted category."})
        app.MODEL = counting_model

        assert any(label.get("for") == "requirement" for label in parser.labels)
        assert "Requirement statement" in page and "Classify" in page and "Predicted category" in returned_pages[0]
        checks.append({"requirement": "NFR1", "status": "passed", "evidence": "Returned HTML contains an associated Requirement statement label, Classify button and Predicted category heading.", "scope": "HTML-label presence. Visual usability and user comprehension are not measured by this check."})
        assert counting_model.calls == [[statement] for statement in examples]
        assert transform_calls == [[statement] for statement in examples]
        assert vectorizer.vocabulary_ == vocabulary_before and np.array_equal(vectorizer.idf_, idf_before)
        checks.append({"requirement": "NFR2", "status": "passed", "evidence": "Every successful request calls the fitted TF-IDF transform. Fit methods were guarded against use; vocabulary and IDF values remained unchanged."})
        status, body = request("/missing")
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=10)
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_hash_before
    assert {row["requirement"] for row in checks} == {"FR1", "FR2", "FR3", "FR4", "FR5", "FR6", "NFR1", "NFR2"}
    result = {
        "date": "2026-10-22", "status": "passed", "requirements_checked": 8,
        "requirement_source": "50% report, Section 2.1, printed page 16",
        "checks": sorted(checks, key=lambda row: row["requirement"]), "prediction_observations": observations,
        "additional_checks": ["Submitted HTML is escaped before display", "Unknown GET route returns 404", "Model file hash is unchanged"],
        "model_sha256": model_hash_before,
        "limitations": "Local HTTP and HTML checks. No concurrent-use, browser automation, usability study or classification-accuracy claim is made here.",
    }
    output = BASE / "application_checks"
    output.mkdir(exist_ok=True)
    if not (output / "verification.json").exists():
        (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
