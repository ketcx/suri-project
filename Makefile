.PHONY: checks
checks:
	python3 src/ledger.py --check
	python3 src/ingest.py --check
	.venv/bin/python src/sam.py --check
	.venv/bin/python src/clips.py --check
	python3 src/kinematics.py --check
	python3 src/agent.py --check
	python3 src/report.py --check
	python3 src/metrics.py --check
	.venv/bin/python src/verify_live.py --check
	.venv/bin/python src/calib.py --check
