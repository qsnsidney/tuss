.ONESHELL: # Applies to every targets in the file!
.SHELLFLAGS += -e

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

clean:
	rm -rf build/
.PHONY: clean

# core

prepare_core:
	mkdir -p build/core
	cmake -S src/core -B build/core
.PHONY: prepare_core

test_core: prepare_core
	$(MAKE) -C build/core all
	$(MAKE) -C build/core test
	@echo "=== core is successfully tested ==="
	@echo ""
.PHONY: test_core

# cpu_sim

prepare_cpu_sim:
	mkdir -p build/cpu_sim
	cmake -S src/cpu_sim -B build/cpu_sim
.PHONY: prepare_cpu_sim

cpu_sim: prepare_cpu_sim
	$(MAKE) -C build/cpu_sim all
	@echo "=== cpu_sim is successfully built ==="
	@echo ""
.PHONY: cpu_sim

run_cpu_sim: cpu_sim
	./build/cpu_sim/cpu_sim
.PHONY: run_cpu_sim

# tus

prepare_tus:
	mkdir -p build/tus
	cmake -S src/tus -B build/tus
.PHONY: prepare_tus

tus: prepare_tus
	$(MAKE) -C build/tus all
	@echo "=== tus is successfully built ==="
	@echo ""
.PHONY: tus

run_tus: tus
	./build/tus/tus
.PHONY: run_tus

# all

# tus is at the last, so others can still be built even
# when no nvcc is available.
all: test_core cpu_sim tus
.PHONY: all