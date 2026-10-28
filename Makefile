.PHONY: checks
checks:
	python3 src/ledger.py --check
	python3 src/ingest.py --check
	.venv/bin/python src/calib.py --check
