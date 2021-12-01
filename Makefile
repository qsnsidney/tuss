.ONESHELL: # Applies to every targets in the file!
.SHELLFLAGS += -e

# Default target executed when no arguments are given to make.
default_target: all
.PHONY: default_target

clean:
	rm -rf build/
.PHONY: clean

# CMAKE_ARGS="-DENABLE_FFAST_MATH=ON"
prepare:
	cmake ${CMAKE_ARGS} -B build
	@echo [=== cmake is successfully prepared ===]
	@echo 
.PHONY: prepare

# core

test_core: prepare
	$(MAKE) -C build core_tests
	$(MAKE) -C build test ARGS="-R '^core_tests_'"
	@echo [=== core is successfully tested ===]
	@echo 
.PHONY: test_core

# cpusim

cpusim: prepare
	$(MAKE) -C build cpusim_all
	@echo [=== cpusim is successfully built ===]
	@echo 
.PHONY: cpusim

run_cpusim: cpusim
	./build/cpusim/cpusim_exe ${ARGS}
.PHONY: run_cpusim

# Check whether NVCC exists
NVCC_RESULT := $(shell which nvcc)
NVCC_TEST := $(notdir $(NVCC_RESULT))

# tus

ifeq ($(NVCC_TEST),nvcc) # NVCC exists
tus: prepare
	$(MAKE) -C build tus_exe
	@echo [=== tus is successfully built ===]
	@echo 
.PHONY: tus

run_tus: tus
	./build/tus/tus_exe ${ARGS}
.PHONY: run_tus
else # NVCC does not exist
tus: prepare
	@echo [=== tus is not supported ===]
	@echo 
.PHONY: tus

# TODO: move run_tus to a separete .sh if passing arguments
run_tus: tus
.PHONY: run_tus
endif

# all

all: test_core cpusim tus
.PHONY: all