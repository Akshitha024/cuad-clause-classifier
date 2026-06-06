.PHONY: help install lint typecheck test data train eval plots clean

LIMIT ?= 2000
EPOCHS ?= 3
BASE_MODEL ?= nlpaueb/legal-bert-small-uncased

help:
	@echo "make install                          - install deps via uv"
	@echo "make lint / typecheck / test          - standard quality gates"
	@echo "make data LIMIT=N                     - build the per-clause classification dataset from CUAD"
	@echo "make train EPOCHS=N BASE_MODEL=...    - LoRA fine-tune on the prepared dataset"
	@echo "make eval                             - per-clause F1 + macro/micro + confusion summary"
	@echo "make plots                            - bar chart of per-clause F1"

install:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -m "not slow and not needs_gpu"

data:
	uv run clause-x data prepare --limit $(LIMIT)

train:
	uv run clause-x train run --epochs $(EPOCHS) --base-model $(BASE_MODEL)

eval:
	uv run clause-x eval run

plots:
	uv run clause-x plots

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +


.PHONY: pdf test-artifacts
pdf:
	cd docs/_report && pandoc research_report.md -o ../research_report.pdf --pdf-engine=xelatex || echo "pandoc + xelatex required; see https://pandoc.org/installing.html"

test-artifacts:
	uv run python ../../_meta/retrofit.py "$(notdir $(CURDIR))" "$(notdir $(CURDIR))"
