"""A small local model bank for fitted sklearn pipelines."""

import json
from pathlib import Path

import joblib


def save_model(pipeline, metadata: dict, model_bank: str | Path) -> Path:
    """Write the complete fitted pipeline and readable metadata under its version."""
    model_directory = Path(model_bank) / metadata["model_version"]
    model_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_directory / "model.joblib")
    (model_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return model_directory


def load_model(model_bank: str | Path, model_version: str):
    """Deserialize a frozen pipeline and its metadata; never fit on load."""
    model_directory = Path(model_bank) / model_version
    metadata = json.loads((model_directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["model_version"] != model_directory.name:
        raise ValueError("Model version does not match its model-bank folder")
    pipeline = joblib.load(model_directory / "model.joblib")
    return pipeline, metadata
