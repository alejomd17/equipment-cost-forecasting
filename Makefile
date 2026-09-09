.PHONY: setup data train forecast agent all

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