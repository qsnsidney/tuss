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

## Input/Output Data

### `BODY_STATE`
The state of a single body, consists of `(POS, VEC, MASS)`.  
The serialization format expands each field, so that looks like:
`(POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS)`

### `BODY_STATE_VEC`
A collection of `BODY_STATE`, each represents individual `BODY_STATE`.

### `BODY_STATE_VEC` in CSV
Print out each `BODY_STATE` in serialization format.  
Each `BODY_STATE` is represented as individual row with strings representing floating values, with comma to separate each field, and ends with a new line.  
This format is not encouraged due to its large file size, and slow deserialization speed due to string processing.

### `BODY_STATE_VEC` in BIN
Print out each `BODY_STATE` in binary serialization format.  
Each `BODY_STATE` is represented as a sequence of bytes, with bytes representing floating values.
The specific format looks like the following:
- first 4 bytes: size of floating type (ie., 4 for floating, 8 for double)
- second 4 bytes: number of bodies
- rest: `(POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS)` for each `BODY_STATE`
  
This format is the recommended format.

## Demo
```
mkdir -p tmp/2_body_log
make run_cpusim ARGS="./benchmark/ic/benchmark_2_simple_2.csv -1 0.0001 6000 ./tmp/2_body_log"
python3 -m scripts.tuss_gui
```
