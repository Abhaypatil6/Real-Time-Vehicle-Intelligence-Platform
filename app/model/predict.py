from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib

MODEL_DIR = Path(__file__).resolve().parent
MODEL_REGISTRY: Dict[str, str] = {
    "v1": "model.pkl",
    "v2": "model.pkl",
}


class ModelManager:
    def __init__(self) -> None:
        self._models = {}

    def load_models(self) -> None:
        for version, file_name in MODEL_REGISTRY.items():
            model_path = MODEL_DIR / file_name
            if not model_path.exists():
                raise FileNotFoundError(f"Model for {version} not found at {model_path}")
            self._models[version] = joblib.load(model_path)

    def predict(self, feature1: float, feature2: float, version: str = "v1") -> str:
        if version not in self._models:
            raise ValueError(f"Unsupported model version: {version}")
        pred = self._models[version].predict([[feature1, feature2]])
        return str(pred[0])

    def batch_predict(self, items: list[tuple[float, float]], version: str = "v1") -> list[str]:
        if version not in self._models:
            raise ValueError(f"Unsupported model version: {version}")
        pred = self._models[version].predict(items)
        return [str(x) for x in pred]
