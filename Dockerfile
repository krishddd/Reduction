FROM python:3.12-slim AS base

WORKDIR /srv

COPY pyproject.toml README.md ./
COPY reduction ./reduction
COPY simulator ./simulator

RUN pip install --no-cache-dir ".[gateway]"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uvicorn", "reduction.gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
