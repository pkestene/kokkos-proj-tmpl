# kokkos-proj-tmpl
A minimal cmake based project skeleton for developping a kokkos application

## Download this skeleton

```bash
git clone --recursive git@github.com:pkestene/kokkos-proj-tmpl.git
```

## How to build ?

### Build with target device OpenMP

```bash
mkdir build_openmp
cd build_openmp
CXX=YOUR_COMPILER_HERE cmake -DKOKKOS_ENABLE_OPENMP=ON ..
make
# then you can run the application
./src/saxpy_kokkos_lambda.openmp
```

Optionnally you can enable HWLOC by passing -DKOKKOS_ENABLE_HWLOC=ON on cmake's command line (or in ccmake curse gui).

### Build with target device CUDA

You **NEED** to use nvcc_wrapper as the CXX compiler. nvcc_wrapper is located in kokkos sources (cloned as git submodule), int the bin subdirectory. You can set the CXX env variable, like this

```bash
mkdir build_cuda
cd build_cuda
export CXX=/path/to/kokkos-proj-tmpl/external/kokkos/bin/nvcc_wrapper
cmake -DKOKKOS_ENABLE_CUDA=ON -DKOKKOS_ENABLE_CUDA_LAMBDA=ON -DKOKKOS_ARCH=Maxwell50 ..
make
# then you can run the application as before
./src/saxpy_kokkos_lambda.cuda
```

Of course, you will need to adapt variable **KOKKOS_ARCH** to your actual GPU architecture (use cuda sample device_query to probe the architecture).

Depending on your OS, you may need to set variable **KOKKOS_CUDA_DIR** to point to your CUDA SDK (if cmake is not able to figure out by itself); e.g. /usr/local/cuda-9.0


## Additional notes

The stream benchmark source code is slightly adapted from [BabelStream](https://github.com/UoB-HPC/BabelStream).
