# Real-Time Vehicle Intelligence Platform

Production-ready real-time machine learning inference service built with **FastAPI + scikit-learn**, designed for low-latency (<100ms typical) online predictions.

## Architecture (text diagram)

```text
Client
  │
  ├── POST /predict or /predict/batch
  ▼
FastAPI App (app/main.py)
  ├── Input validation (Pydantic schemas)
  ├── Rate limiting (in-memory)
  ├── Inference service (latency tracking)
  ├── Model manager (versioned model loading)
  └── Structured logging to logs/inference.log
  ▼
scikit-learn model (app/model/model.pkl)
  ▼
JSON response
```

## Project Structure

```text
project/
│
├── app/
│   ├── main.py
│   ├── model/
│   │   ├── model.pkl
│   │   └── predict.py
│   ├── schemas/
│   │   └── input_schema.py
│   ├── services/
│   │   └── inference_service.py
│   ├── utils/
│   │   └── logger.py
│
├── training/
│   └── train_model.py
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
```

## Features

- Model training and persistence (`model.pkl` via `joblib`)
- FastAPI endpoints:
  - `GET /` health summary
  - `GET /health` liveness endpoint
  - `POST /predict` single prediction
  - `POST /predict/batch` batch prediction
- Model loaded once at startup
- Request + prediction logging to file
- Global error handling with structured JSON
- Model version query support (`?model_version=v1|v2`)
- Async API endpoints
- Request latency tracking in responses
- Built-in per-IP in-memory rate limiter
- Docker + Docker Compose support

## Setup (Local)

1. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train and save model:
   ```bash
   python training/train_model.py
   ```
4. Run API:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
5. Open docs:
   - Swagger UI: `http://localhost:8000/docs`

## API Usage

### Health
```bash
curl -X GET http://localhost:8000/
```

### Single prediction
```bash
curl -X POST 'http://localhost:8000/predict?model_version=v1' \
  -H 'Content-Type: application/json' \
  -d '{"feature1": 5.1, "feature2": 3.5}'
```

Example response:
```json
{
  "prediction": "1",
  "model_version": "v1",
  "latency_ms": 1.23
}
```

### Batch prediction
```bash
curl -X POST 'http://localhost:8000/predict/batch?model_version=v1' \
  -H 'Content-Type: application/json' \
  -d '{"instances": [{"feature1": 5.1, "feature2": 3.5}, {"feature1": 0.2, "feature2": -1.1}]}'
```

## Testing

```bash
pytest -q
```

## Docker

### Build and run
```bash
docker build -t realtime-ml-platform .
docker run -p 8000:8000 realtime-ml-platform
```

### Docker Compose
```bash
docker compose up --build
```

## Environment Variables

- `RATE_LIMIT_PER_MINUTE` (default: `60`)

## AWS EC2 Deployment (Step-by-step)

1. Launch EC2 (Ubuntu 22.04 or newer), allow inbound TCP ports `22` and `8000`.
2. SSH into instance:
   ```bash
   ssh -i <key.pem> ubuntu@<ec2-public-ip>
   ```
3. Install Docker:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose-plugin
   sudo usermod -aG docker $USER
   newgrp docker
   ```
4. Clone repository and run:
   ```bash
   git clone <your-repo-url>
   cd Real-Time-Vehicle-Intelligence-Platform
   docker compose up --build -d
   ```
5. Verify service:
   ```bash
   curl http://<ec2-public-ip>:8000/health
   ```
6. (Optional production hardening):
   - Place Nginx in front of FastAPI
   - Add TLS certificate with Let's Encrypt
   - Enable CloudWatch / centralized logs

## Commands Quick Start

```bash
pip install -r requirements.txt
python training/train_model.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
