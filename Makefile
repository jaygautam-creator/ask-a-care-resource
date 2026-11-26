PY := .venv/bin/python

.PHONY: setup ingest eval report all app api test lint clean

setup:            ## create the virtualenv and install (needs uv: https://docs.astral.sh/uv/)
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"
	@test -f .env || (cp .env.example .env && echo "created .env - add your API keys")

ingest:           ## raw snapshot -> clean -> chunk -> embed -> index
	$(PY) -m care_rag.cli ingest

eval:             ## full evaluation (retrieval for all configs + answer experiment + report)
	$(PY) -m care_rag.cli eval

report:
	$(PY) -m care_rag.cli report

all: ingest eval  ## everything after setup

app:              ## Streamlit UI
	$(PY) -m streamlit run app.py

api:              ## FastAPI on :8000 (docs at /docs)
	$(PY) -m uvicorn api:app --port 8000

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests app.py api.py

clean:            ## remove rebuilt artefacts (keeps the raw snapshot and results)
	rm -rf data/processed data/index .cache
