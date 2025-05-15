VENV = .venv
REQUIREMENTS = requirements.txt
PYTHON = /usr/bin/env python3
SRC = ./src
MODULE = ss_reporting_tool
export PYTHONPATH = $(SRC)
.PHONY: install test clean run get set feedback

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip

install: requirements.txt | $(VENV)
	$(VENV)/bin/pip install -r $(REQUIREMENTS)

test:
	$(VENV)/bin/python main.py

clean:
	rm -rf $(VENV)
