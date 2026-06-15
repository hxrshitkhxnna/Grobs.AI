# GrobsAI Backend

Monorepo backend built with **FastAPI** for GrobsAI (AI-powered career platform).

## Structure

- `app/` - FastAPI application code (routers, services, models)
- `migrations/` - Alembic migrations
- `tests/` - Pytest test suite

## Getting started (local)

### 1) Create virtual environment

```powershell
python -m venv venv
.
venv\Scripts\activate
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Configure environment

Copy `.env.example` to `.env` (create `.env` if needed) and set your variables.

### 4) Start the API (FastAPI)

```powershell
uvicorn app.main:app --reload --port 8000
```

After starting, open API docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Run tests

```powershell
pytest
```

## Notes

- Database migrations are managed with Alembic (see `migrations/`).

