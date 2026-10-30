"""Download and verify the exact model revision used in experiment 04."""

from pathlib import Path
from semantic_features import model_files

if __name__ == "__main__":
    print(model_files(Path(__file__).resolve().parent / "model_assets", download=True))
