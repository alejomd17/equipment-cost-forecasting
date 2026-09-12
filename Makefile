.PHONY: setup data train forecast agent all docker-build docker-run

setup:
	uv sync

data:
	uv run python -m src.ingest.build

train: data
	uv run python -m src.models.train

forecast: train
	uv run python -m src.forecast.run

agent:
	uv run streamlit run src/agent/app.py

all: data train forecast

docker-build:
	docker build -t equipment-cost-forecasting .

docker-run:
	docker run --rm -p 8501:8501 --env-file .env equipment-cost-forecasting