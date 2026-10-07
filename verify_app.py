"""Verify actual HTTP responses against the trained model."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

BASE = Path(__file__).resolve().parent
runtime = BASE.parent / "dataset_review" / "runtime"
if runtime.exists():
    sys.path.insert(0, str(runtime))
spec = importlib.util.spec_from_file_location("classifier_app", BASE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{server.server_port}"
checks = []


def request(path, text=None):
    data = None if text is None else urlencode({"requirement": text}).encode()
    try:
        response = urlopen(URL + path, data=data, timeout=10)
    except HTTPError as error:
        response = error
    with response:
        return response.status, response.read().decode("utf-8")


try:
    status, body = request("/")
    assert status == 200 and '<form action="/classify"' in body
    checks.append("Initial page provides labelled text input and classification button.")
    for text in ["", "   ", "\n\t"]:
        status, body = request("/classify", text)
        assert status == 400 and "Enter a requirement statement" in body and 'class="category"' not in body
    checks.append("Empty and whitespace-only input rejected without a prediction.")
    for text in [
        "The system shall allow users to reset their password.",
        "The system shall respond within two seconds.",
        "The system shall allow registered users to log in.",
        "The application must protect confidential information.",
    ]:
        status, body = request("/classify", text)
        label = app.LABELS[app.MODEL.predict([text])[0]]
        assert status == 200 and f'class="category">{label}</p>' in body and app.html.escape(text) in body
    checks.append("Four submissions display the actual saved-model prediction and preserve input text.")
    status, body = request("/classify", '<script>alert("test")</script> The system shall display text.')
    assert status == 200 and '<script>alert("test")</script>' not in body and "&lt;script&gt;" in body
    checks.append("Submitted HTML is escaped before display.")
    original = app.MODEL
    class FailingModel:
        def predict(self, texts):
            raise RuntimeError("Simulated failure")
    app.MODEL = FailingModel()
    status, body = request("/classify", "The system shall display text.")
    assert status == 500 and "Classification could not be completed" in body and 'class="category"' not in body
    app.MODEL = original
    checks.append("A simulated model failure displays an error without a prediction.")
    status, body = request("/missing")
    assert status == 404
    checks.append("Unknown page returns 404.")
    result = {"status": "passed", "checks": checks, "scope": "Local HTTP behaviour; model accuracy assessed separately in experiment_01."}
    (BASE / "evidence" / "application_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
finally:
    server.shutdown()
    server.server_close()
