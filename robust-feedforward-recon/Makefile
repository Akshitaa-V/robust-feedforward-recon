.PHONY: install test train evaluate report all

install:
	pip install -e ".[dev]"

test:
	pytest -q

all:
	bash scripts/run_all.sh

evaluate:
	PYTHONPATH=src python -m robustrecon.evaluate --config configs/default.yaml

report:
	PYTHONPATH=src python -m robustrecon.report --config configs/default.yaml
