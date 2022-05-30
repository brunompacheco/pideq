.ONESHELL:

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SHELL=/bin/bash

CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

default: help

jupyter:
	@$(CONDA_ACTIVATE) base
	@jupyter notebook $(PROJECT_DIR)/notebooks/.

help:
	@echo "Sadly, still a WIP :("
