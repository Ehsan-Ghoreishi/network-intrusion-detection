.PHONY: install data train supervised autoencoder baselines robustness plots all test lint clean

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

data:
	nids download

supervised: data
	nids supervised

autoencoder: data
	nids autoencoder

baselines: data
	nids baselines

robustness: data
	nids robustness --seeds 5

plots:
	nids plots

train: supervised autoencoder baselines

all: train robustness plots

test:
	pytest

lint:
	ruff check src tests

clean:
	rm -rf models .pytest_cache .ruff_cache
