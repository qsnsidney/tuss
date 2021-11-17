.ONESHELL: # Applies to every targets in the file!
.SHELLFLAGS += -e

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

clean:
	rm -rf build/
.PHONY: clean

# cpu_sim

prepare_cpu_sim:
	mkdir -p build/cpu_sim;
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

# gpu

gpu:
	./compile_cuda.sh main
	@echo "=== gpu is successfully built ==="
	@echo ""
.PHONY: gpu

run_gpu: gpu
	./build/gpu/main
.PHONY: run_gpu

# all

all: cpu_sim gpu
.PHONY: all