# Tiny Universe Simulator (TUS)

## Algorithm
Paper reference:  
[Fast Simulations of Gravitational Many-body Problem on RV770 GPU](https://arxiv.org/pdf/0904.3659.pdf)

## Executables

### tus (main project)
- The GPU simulator written in CUDA
- Located in ./src/tus/

### cpu_sim
- The CPU simulator
- Located in ./src/cpu_sim

### core
- The core common code
- core_lib
- Located in ./src/core

### bicgen
- Bodies Initial Condition GENerator
- Located in ./scripts/bicgen
- `python3 -m scripts.bicgen`

## Makefile

### all
```
# Compile all and run test_core
make
```

### tus
```
# Compile only
make tus
# Compile and run
make run_tus
```

### cpu_sim
```
# Compile only
make cpu_sim
# Compile and run
make run_cpu_sim
```

### core
```
# Compile and test
make test_core
```
