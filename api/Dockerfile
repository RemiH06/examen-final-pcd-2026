# Contexto de construcción: raíz del repositorio (necesita pyproject.toml y uv.lock).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app
# Primero sólo el lock: esta capa se reutiliza mientras no cambien las dependencias.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-install-project --no-default-groups --group api

COPY api/ api/
RUN useradd --create-home nimbo && chown -R nimbo /app
USER nimbo

EXPOSE 8060
# Las credenciales de Databricks llegan en tiempo de ejecución (env_file), nunca en la imagen.
CMD ["/app/.venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8060"]
