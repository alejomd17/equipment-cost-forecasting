FROM python:3.12-slim

# libgomp1 es requisito de LightGBM y no viene en la imagen slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Las dependencias se instalan antes de copiar el codigo para que la capa
# quede cacheada y un cambio en src no obligue a reinstalar todo
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8501

# Por defecto levanta el agente. Para correr solo el pipeline:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
#     equipment-cost-forecasting uv run python -m src.forecast.run
CMD ["uv", "run", "--no-dev", "streamlit", "run", "src/agent/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]