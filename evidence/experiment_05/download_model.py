"""Download and verify the public base weights needed to repeat fine-tuning."""

import hashlib
import json
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent

if __name__ == "__main__":
    source = json.loads((BASE.parent / "experiment_04/sources.json").read_text(encoding="utf-8"))["model"]
    protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    hashes = {name: expected for name, expected in source["files_sha256"].items() if name != "onnx/model.onnx"}
    hashes["model.safetensors"] = protocol["weights_sha256"]
    destination = BASE / "model_assets"
    for name, expected in hashes.items():
        path = destination / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            url = f"https://huggingface.co/{source['name']}/resolve/{source['revision']}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Checksum differs for {name}")
            temporary.rename(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum differs for {name}")
        print("Verified", name, flush=True)
    print("Base model is ready:", destination)
