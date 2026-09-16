PYTHON ?= python3
.DEFAULT_GOAL := help

help:
	@echo "Run commands from the repository root:"
	@echo "  make all             Build C and C++ inference programs"
	@echo "  make test-software   Check golden model, parser, gradients, and deployment"
	@echo "  make test-hardware   Run VHDL simulations and synthesis (requires GHDL)"
	@echo "  make docs-check      Check local documentation links"
	@echo "  make test            Run all three check groups"
	@echo "  make asan            Run native memory/undefined-behavior checks"
	@echo "  make download        Download and verify MNIST archives"
	@echo "  make train           Train the default CNN into build/trained/"
	@echo "  make verify-golden   Check the frozen teaching model"
	@echo "  make verify-trained  Check exported trained weights in both engines"
	@echo "  make benchmark       Measure verified native CPU inference"
	@echo "  make demo            Generate build/trained/demo.html"
	@echo "Optional: append PYTHON=/absolute/path/to/python"

all:
	$(MAKE) -C lessons/day7 all
	$(MAKE) -C trained all

test: test-software test-hardware docs-check

test-software:
	$(MAKE) -C lessons/day7 verify test-robust test-engine PYTHON=$(PYTHON)
	$(MAKE) -C trained test PYTHON=$(PYTHON)

test-hardware:
	$(MAKE) -C hardware/vhdl test synth

docs-check:
	$(PYTHON) scripts/check_docs.py

verify-golden:
	$(MAKE) -C lessons/day7 verify PYTHON=$(PYTHON)

asan:
	$(MAKE) -C lessons/day7 asan PYTHON=$(PYTHON)
	$(MAKE) -C trained asan PYTHON=$(PYTHON)

train:
	$(PYTHON) -m trained.train

download:
	$(PYTHON) -m trained.download

demo: all
	$(PYTHON) -m trained.demo

verify-trained: all
	$(PYTHON) -m trained.verify

benchmark: all
	$(PYTHON) -m trained.benchmark

.PHONY: help all test test-software test-hardware docs-check verify-golden asan train download demo verify-trained benchmark
