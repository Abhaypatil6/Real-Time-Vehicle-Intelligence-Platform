import time
from collections import defaultdict, deque
from dataclasses import dataclass

from app.model.predict import ModelManager
from app.schemas.input_schema import PredictionInput


@dataclass
class InferenceResult:
    prediction: str
    latency_ms: float


class SimpleRateLimiter:
    def __init__(self, limit: int = 60, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests = defaultdict(deque)

    def allow(self, client_key: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        q = self._requests[client_key]

        while q and q[0] < window_start:
            q.popleft()

        if len(q) >= self.limit:
            return False

        q.append(now)
        return True


class InferenceService:
    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    def predict(self, payload: PredictionInput, model_version: str = "v1") -> InferenceResult:
        start = time.perf_counter()
        prediction = self.model_manager.predict(
            feature1=payload.feature1,
            feature2=payload.feature2,
            version=model_version,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return InferenceResult(prediction=prediction, latency_ms=latency_ms)

    def batch_predict(self, items: list[PredictionInput], model_version: str = "v1") -> InferenceResult:
        start = time.perf_counter()
        features = [(item.feature1, item.feature2) for item in items]
        preds = self.model_manager.batch_predict(features, version=model_version)
        latency_ms = (time.perf_counter() - start) * 1000
        return InferenceResult(prediction=",".join(preds), latency_ms=latency_ms)
