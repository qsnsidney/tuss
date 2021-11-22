# Tiny Universe Simulator System (TUSS)

[![Build Status][actions-badge]][actions-url]

[actions-badge]: https://github.com/qsnsidney/tuss/actions/workflows/makefile-src.yml/badge.svg
[actions-url]: https://github.com/qsnsidney/tuss/actions?query=workflow%3Amakefile-src

## Algorithm
Paper reference:  
[Fast Simulations of Gravitational Many-body Problem on RV770 GPU](https://arxiv.org/pdf/0904.3659.pdf)

## Executables

### tus (main project)
- The GPU simulator written in CUDA
- Located in ./src/tus

### cpusim
- The CPU simulator
- Located in ./src/cpusim

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
# Run with arguments
make run_tus ARGS="any_args"
```

### cpusim
```
# Compile only
make cpusim
# Compile and run
make run_cpusim
# Run with arguments
make run_cpusim ARGS="any_args"
```

### core
```
# Compile and test
make test_core
# With more verbose output
make test_core ARGS=-V
# See more options available
make test_core ARGS=-h
```
