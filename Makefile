.PHONY: all setup run clean

setup:
	bash scripts/setup_ubuntu.sh

run:
	python -m src.run_all

all: run

clean:
	rm -rf data_stage/* outputs/* || true

# One-click: split train/validate, evaluate, predict latest, reports, site
oneclick:
	PYTENSOR_FLAGS=base_compiledir=./.pytensor python -m src.one_click
