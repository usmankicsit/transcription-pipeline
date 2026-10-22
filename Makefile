PYTHON ?= python3.12

.PHONY: install test sample api

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/pip install -e .

test:
	.venv/bin/pytest -q

sample:
	bash scripts/make_sample.sh
	.venv/bin/transcribe samples/sample.mp3 --model tiny -f json -o examples/sample.transcript.json
	.venv/bin/transcribe samples/sample.mp3 --model tiny -f srt -o examples/sample.srt

api:
	.venv/bin/transcribe-api
