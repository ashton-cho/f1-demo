# F1 Live Data Demo

## Overview

This project is a demo of a real-time F1 telemetry analytics dashboard, featuring:

- **Backend:** FastAPI (serves session-based slices of raw F1 data)
- **Frontend:** Dash (aggregates, processes, predicts, and visualizes live data)
- **ML Model:** LSTM (pre-trained, with encoder/scaler in `/frontend/model/`)

---

## Quickstart

### Option 1: Local Development (with Poetry, recommended for developers)

#### 1. Prerequisites

- Python 3.11 installed
- [Poetry](https://python-poetry.org/docs/#installation) installed

#### 2. Start Backend

```sh
cd backend
poetry install
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

#### 3. Start Frontend

Open a new terminal window/tab:

```sh
cd frontend
poetry install
poetry run python -m src.app
```

#### 4. Open Dashboard

Go to http://localhost:8050 in your browser.

<br>

### Option 2: Docker (Run Everything with One Command)

#### 1. Prerequisites

- Docker installed

#### 2. Build and Run with Docker Compose

From your project root:

```sh
docker-compose up --build
```

This will build images for both backend and frontend, install all dependencies, and start both services.

#### 3. Open Dashboard

Go to http://localhost:8050 in your browser.

---

## Project Structure

- backend/
  - src/ — FastAPI backend app code (`main.py`, `config.py`)
  - demo_data/ — Raw data files
  - Dockerfile, poetry.lock, pyproject.toml, README.md
- frontend/
  - src/ — Dash app code (`app.py`, `config.py`, `etl.py`)
  - model/ — Pretrained ML model, encoder, scaler dict
  - Dockerfile, poetry.lock, pyproject.toml, README.md
- docker-compose.yml — Multi-service orchestration
- README.md, .gitignore

---

## Adding Your Own Data

Pull raw session data for selected year and event with backend/demo_data/data_pull.py.

---

## Notes

- For best reproducibility, use Python 3.11 everywhere.
- All dependencies are managed with Poetry in both backend and frontend.
