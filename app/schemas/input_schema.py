from typing import List

from pydantic import BaseModel, Field, conlist


class PredictionInput(BaseModel):
    feature1: float = Field(..., description="First numeric feature")
    feature2: float = Field(..., description="Second numeric feature")


class BatchPredictionInput(BaseModel):
    instances: conlist(PredictionInput, min_length=1, max_length=100)


class PredictionOutput(BaseModel):
    prediction: str
    model_version: str
    latency_ms: float


class BatchPredictionOutput(BaseModel):
    predictions: List[str]
    model_version: str
    latency_ms: float
