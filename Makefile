.PHONY: all setup run clean

setup:
	bash scripts/setup_ubuntu.sh

run:
	python -m src.run_all

all: run

clean:
	rm -rf data_stage/* outputs/* || true
