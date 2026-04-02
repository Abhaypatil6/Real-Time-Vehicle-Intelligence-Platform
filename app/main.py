import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.model.predict import ModelManager
from app.schemas.input_schema import (
    BatchPredictionInput,
    BatchPredictionOutput,
    PredictionInput,
    PredictionOutput,
)
from app.services.inference_service import InferenceService, SimpleRateLimiter
from app.utils.logger import setup_logger

logger = setup_logger()
model_manager = ModelManager()
inference_service = InferenceService(model_manager)
rate_limiter = SimpleRateLimiter(
    limit=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
    window_seconds=60,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_manager.load_models()
    logger.info("Models loaded successfully")
    yield


app = FastAPI(title="Real-Time Vehicle Intelligence Platform", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s: %s", request.url.path, str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


@app.get("/")
async def root():
    return {"status": "healthy", "service": "inference-api"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
async def predict(payload: PredictionInput, request: Request, model_version: str = Query("v1")):
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        result = inference_service.predict(payload, model_version=model_version)
        logger.info(
            "predict | client=%s | input=%s | prediction=%s | version=%s | latency_ms=%.2f",
            client_ip,
            payload.model_dump(),
            result.prediction,
            model_version,
            result.latency_ms,
        )
        return PredictionOutput(
            prediction=result.prediction,
            model_version=model_version,
            latency_ms=result.latency_ms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def batch_predict(
    payload: BatchPredictionInput,
    request: Request,
    model_version: str = Query("v1"),
):
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        result = inference_service.batch_predict(payload.instances, model_version=model_version)
        predictions = result.prediction.split(",") if result.prediction else []
        logger.info(
            "batch_predict | client=%s | count=%d | version=%s | latency_ms=%.2f",
            client_ip,
            len(payload.instances),
            model_version,
            result.latency_ms,
        )
        return BatchPredictionOutput(
            predictions=predictions,
            model_version=model_version,
            latency_ms=result.latency_ms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
