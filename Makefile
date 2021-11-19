.ONESHELL: # Applies to every targets in the file!
.SHELLFLAGS += -e

# Default target executed when no arguments are given to make.
default_target: all
.PHONY: default_target

clean:
	rm -rf build/
.PHONY: clean

prepare:
	cmake -B build
	@echo "=== successfully prepared ==="
	@echo ""
.PHONY: prepare

# core

test_core: prepare
	$(MAKE) -C build core_tests
	$(MAKE) -C build test ARGS="-R '^core_tests_'"
	@echo "=== core is successfully tested ==="
	@echo ""
.PHONY: test_core

# cpu_sim

cpu_sim: prepare
	$(MAKE) -C build cpu_sim
	@echo "=== cpu_sim is successfully built ==="
	@echo ""
.PHONY: cpu_sim

run_cpu_sim: cpu_sim
	./build/cpu_sim/cpu_sim
.PHONY: run_cpu_sim

# Check whether NVCC exists
NVCC_RESULT := $(shell which nvcc)
NVCC_TEST := $(notdir $(NVCC_RESULT))

ifeq ($(NVCC_TEST),nvcc) # NVCC exists

# tus

tus: prepare
	$(MAKE) -C build tus
	@echo "=== tus is successfully built ==="
	@echo ""
.PHONY: tus

run_tus: tus
	./build/tus/tus
.PHONY: run_tus

# all

all: test_core cpu_sim tus
.PHONY: all

else # NVCC does not exist

# all

all: test_core cpu_sim
.PHONY: all

endif