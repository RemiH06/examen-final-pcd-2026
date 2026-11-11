# Contexto de construcción: raíz del repositorio (necesita pyproject.toml y uv.lock).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Usuario sin privilegios creado antes de instalar: evita un chown -R que duplicaría el ambiente en otra capa.
RUN useradd --create-home nimbo && mkdir /app && chown nimbo /app
USER nimbo
WORKDIR /app
COPY --chown=nimbo pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-install-project --no-default-groups --group ui

COPY --chown=nimbo ui/ ui/

EXPOSE 8560
# Sin credenciales: la UI sólo conoce la URL de la API (API_URL).
CMD ["/app/.venv/bin/streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8560", "--server.headless", "true"]
