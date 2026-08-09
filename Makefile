.PHONY: setup verify full report audit test

setup:
	python -m pip install -r requirements-lock.txt
	python -m pip install -e .

verify:
	python scripts/reproduce_report.py verify

full:
	python scripts/reproduce_report.py full

report:
	python report/make_figures.py
	cd report && tectonic main.tex

audit:
	python scripts/public_release_audit.py

test:
	python -m pytest
