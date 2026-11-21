#!/usr/bin/env bash
set -euo pipefail

echo "Waiting for PostgreSQL at ${DATABASE_URL%%\?*} ..."
python - <<'PY'
import os
import time

from sqlalchemy import create_engine, text

url = os.environ["DATABASE_URL"]
if url.startswith("postgres://"):
    url = url.replace("postgres://", "postgresql+psycopg://", 1)
elif url.startswith("postgresql://"):
    url = url.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(url, connect_args={"connect_timeout": 5})
for attempt in range(60):
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("Database is ready")
        break
    except Exception as exc:  # noqa: BLE001
        print(f"  attempt {attempt + 1}/60: {exc.__class__.__name__}")
        time.sleep(2)
else:
    raise SystemExit("PostgreSQL did not become ready in time")
engine.dispose()
PY

echo "Applying database migrations ..."
alembic upgrade head

echo "Starting DocuMind API ..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips='*'
