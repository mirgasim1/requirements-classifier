"""A local web interface for the trained functional/non-functional classifier."""
import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import parse_qs
import webbrowser

import joblib

BASE = Path(__file__).resolve().parent
MODEL = joblib.load(BASE / "classifier.joblib")
LABELS = {"FR": "Functional", "NFR": "Non-functional"}


def page(requirement="", prediction=None, error=None):
    template = (BASE / "index.html").read_text(encoding="utf-8")
    result = ""
    if prediction is not None:
        result = '<section class="result" role="status"><h2>Predicted category</h2><p class="category">' + html.escape(prediction) + '</p><p>This prediction may be incorrect. Review it before using it.</p></section>'
    if error:
        result = '<p class="error" role="alert">' + html.escape(error) + '</p>'
    # Escape all submitted text before displaying it in HTML.
    return template.replace("{{result}}", result).replace("{{requirement}}", html.escape(requirement))


class Handler(BaseHTTPRequestHandler):
    def send_page(self, content, status=200):
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        self.send_page(page())

    def do_POST(self):
        if self.path != "/classify":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 0 or size > 1000000:
                self.send_page(page(error="The submitted request is too large."), 413)
                return
            form = parse_qs(self.rfile.read(size).decode("utf-8"), keep_blank_values=True)
            requirement = form.get("requirement", [""])[0]
        except (ValueError, UnicodeDecodeError):
            self.send_page(page(error="The submitted text could not be read."), 400)
            return
        if not requirement.strip():
            self.send_page(page(requirement, error="Enter a requirement statement before classifying."), 400)
            return
        try:
            label = MODEL.predict([requirement])[0]
            prediction = LABELS[label]
        except Exception:
            self.send_page(page(requirement, error="Classification could not be completed. Please try again."), 500)
            return
        self.send_page(page(requirement, prediction=prediction))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Classifier running at {url}\nKeep this window open. Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
