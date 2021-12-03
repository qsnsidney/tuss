# Tiny Universe Simulator System (TUSS)

[![Build Status][actions-badge]][actions-url]

[actions-badge]: https://github.com/qsnsidney/tuss/actions/workflows/makefile-src.yml/badge.svg
[actions-url]: https://github.com/qsnsidney/tuss/actions?query=workflow%3Amakefile-src


## Algorithm
Paper reference:  
[Fast Simulations of Gravitational Many-body Problem on RV770 GPU](https://arxiv.org/pdf/0904.3659.pdf)


## Executables
```
# Compile all and run test_core
make
```

### tus (main project)
- Tiny Universe Simulator
- The GPU simulator written in CUDA
- Located in ./src/tus
```
# Compile only
make tus
# Compile and run
make run_tus
# Run with arguments
make run_tus ARGS="any_args"
```

### cpusim
- CPU SIMulator
- Located in ./src/cpusim
```
# Compile only
make cpusim
# Compile and run
make run_cpusim
# Run with arguments
make run_cpusim ARGS="any_args"
```

### core
- The core common code
- core_lib
- Located in ./src/core
```
# Compile and test
make test_core
# With more verbose output
make test_core ARGS=-V
# See more options available
make test_core ARGS=-h
```

### bicgen
- Bodies Initial Condition GENerator
- Located in ./scripts/bicgen
- `python3 -m scripts.bicgen`

### tussgui
- Tiny Universe Simulator System GUI
- Visualiation of trajectories via `SYSTEM_STATE`s
- Input files are expected to be ordered by integer
- Supports Live and Still trajectory mode
- `python3 -m scripts.tussgui`


## Input/Output Data

### `BODY_STATE`
The state of a single body, consists of `(POS, VEC, MASS)`.  
The serialization format expands each field, so that looks like:
`(POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS)`

### `SYSTEM_STATE`
A collection of `BODY_STATE`, each represents individual `BODY_STATE`.

### `SYSTEM_STATE` in CSV
Print out each `BODY_STATE` in serialization format.  
Each `BODY_STATE` is represented as individual row with strings representing floating values, with comma to separate each field, and ends with a new line.  
This format is not encouraged due to its large file size, and slow deserialization speed due to string processing.

### `SYSTEM_STATE` in BIN
Print out each `BODY_STATE` in binary serialization format.  
Each `BODY_STATE` is represented as a sequence of bytes, with bytes representing floating values.
The specific format looks like the following:
- first 4 bytes: size of floating type (ie., 4 for floating, 8 for double)
- second 4 bytes: number of bodies
- rest: `(POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS)` for each `BODY_STATE`
  
This format is the recommended format.


## Demo

### Visualization

#### cpusim
```
mkdir -p ./tmp/solar_sys_cpu_log
make run_cpusim ARGS="-i ./benchmark/ic/solar_system.csv -d 0.05 -n 10000 -o ./tmp/solar_sys_cpu_log"
python3 -m scripts.tussgui still ./tmp/solar_sys_cpu_log
# Or, some animation
python3 -m scripts.tussgui live ./tmp/solar_sys_cpu_log
rm -rf ./tmp/solar_sys_cpu_log
```

#### tus
```
mkdir -p ./tmp/solar_sys_tus_log
make run_tus ARGS="-i ./benchmark/ic/solar_system.csv -d 0.05 -n 10000 -o ./tmp/solar_sys_tus_log"
python3 -m scripts.tussgui still ./tmp/solar_sys_tus_log
# Or, some animation
python3 -m scripts.tussgui live ./tmp/solar_sys_tus_log
rm -rf ./tmp/solar_sys_tus_log
```

### Performance

#### cpusim
```
make run_cpusim ARGS="-i ./benchmark/ic/solar_system.csv -d 0.05 -n 10000"
make run_cpusim ARGS="-i ./benchmark/ic/benchmark_100000.bin -d 0.001 -n 10 -t 1"
make run_cpusim ARGS="-i ./benchmark/ic/benchmark_100000.bin -d 0.001 -n 10 -t 4"
```

#### tus
```
make run_tus ARGS="-i ./benchmark/ic/solar_system.csv -d 0.05 -n 10000"
make run_tus ARGS="-i ./benchmark/ic/benchmark_100000.bin -d 0.001 -n 10"
```

### Verification

#### cpusim
```
make run_cpusim ARGS="-i ./benchmark/ic/solar_system.csv -d 0.05 -n 1 --verify"
make run_cpusim ARGS="-i ./benchmark/ic/benchmark_100000.bin -d 0.001 -n 10 -t 4 --verify"
```

#### tus
```
make run_tus ARGS="-i ./benchmark/ic/solar_system.csv -d 0.05 -n 1 --verify"
make run_tus ARGS="-i ./benchmark/ic/benchmark_100000.bin -d 0.001 -n 10 --verify"
```

## CMake and Makefile

### Arg Passing to exe
```
ARGS="<your_args>"
```

### Enable -ffast-math
```
CMAKE_ARGS="-DENABLE_FFAST_MATH=ON"
```