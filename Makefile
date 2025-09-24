#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = pretrain-braindecode-models
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python${PYTHON_VERSION}

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
## Make sure that we're inside a virtualenv by testing whether $$VIRTUAL_ENV is set
## or $$CONDA_PREFIX is set
## Installs torch first (requirements.txt), then PyTorch3D, otherwise it doesn't work
## ${PYTHON_INTERPRETER} -m pip install git+https://github.com/facebookresearch/pytorch3d.git@stable 
.PHONY: requirements
requirements:
	@echo "Python interpreter: $(PYTHON_INTERPRETER)"
	@echo "Python path: $(shell which $(PYTHON_INTERPRETER))"
	@if [ -z "$(VIRTUAL_ENV)" ] && [ -z "$(CONDA_PREFIX)" ]; then \
		echo "Not inside a virtualenv. Please activate it first."; \
		exit 1; \
	fi
	@echo "VIRTUALENV path: $(VIRTUAL_ENV)"
	@echo "CONDA_PREFIX path: $(CONDA_PREFIX)"
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff check pretrain_braindecode_models

## Format source code with ruff
.PHONY: format
format:
	ruff format pretrain_braindecode_models
	ruff check --fix pretrain_braindecode_models

## Set up python interpreter environment with virtualenvwrapper
## This will create a new virtualenv with the name of the project
.PHONY: environment_venv
environment_venv:
	@bash -c "source $(HOME)/.bashrc && if command -v mkvirtualenv &> /dev/null; then mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else if [ ! -z \"$(which virtualenvwrapper.sh)\" ]; then source $(which virtualenvwrapper.sh); mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi; fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"

## Set up python interpreter environment with micromamba
## This will create a new virtualenv with the name of the project
.PHONY: environment_micromamba
environment_micromamba:
	@bash -c "source $(HOME)/.bashrc && if command -v micromamba &> /dev/null; then micromamba create -n $(PROJECT_NAME) python=${PYTHON_VERSION} -c conda-forge -c nvidia -c pytorch -y && micromamba install -y gxx_linux-64=14.2 gcc_linux-64=14.2 gcc=14.2 gxx=14.2; fi"
	@echo ">>> New micromamba environment created. Activate with:\nmicromamba activate $(PROJECT_NAME)"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
