FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml alembic.ini ./
COPY app ./app
COPY alembic ./alembic
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh && pip install --no-cache-dir ".[dev]"

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
