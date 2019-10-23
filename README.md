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
CXX=YOUR_COMPILER_HERE cmake -DKokkos_ENABLE_OPENMP=ON ..
make
# then you can run the application
./src/saxpy_kokkos_lambda.openmp
```

Optionnally you can enable HWLOC by passing -DKokkos_ENABLE_HWLOC=ON on cmake's command line (or in ccmake curse gui).

### Build with target device CUDA

You **NEED** to use nvcc_wrapper as the CXX compiler. nvcc_wrapper is located in kokkos sources (cloned as git submodule), int the bin subdirectory. You can set the CXX env variable, like this

```bash
mkdir build_cuda
cd build_cuda
export CXX=/path/to/kokkos-proj-tmpl/external/kokkos/bin/nvcc_wrapper
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_MAXWELL50=ON ..
make
# then you can run the application as before
./src/saxpy_kokkos_lambda.cuda
```

Of course, you will need to adapt variable **Kokkos_ARCH** to your actual GPU architecture (use cuda sample device_query to probe the architecture).

Depending on your OS, you may need to set variable **Kokkos_CUDA_DIR** to point to your CUDA SDK (if cmake is not able to figure out by itself); e.g. /usr/local/cuda-9.0


## Additional notes

### Stream benchmark

The stream benchmark source code is slightly adapted from [BabelStream](https://github.com/UoB-HPC/BabelStream).

### Stencil benchmark

Here are the results obtained on different computing platforms:

Intel Skylake (2x20 cores, Intel Xeon Gold 5115, icpc 2018.0.128)

![stencil bench skylake icpc](https://github.com/pkestene/kokkos-proj-tmpl/raw/master/doc/stencil/stencil_bench_alfven_skylake_icpc.png "Skylake (2x20 cores, Intel Xeon Gold 5115, icpc 2018.0.128)")

Intel KNL (icpc 2017.0.6.256, OMP_NUM_THREADS=64)

![stencil bench knl icpc_omp_64](https://github.com/pkestene/kokkos-proj-tmpl/raw/master/doc/stencil/stencil_bench_irene_knl_omp_64.png "Skylake (Intel KNL, icpc 2017.0.6.256, 64 threads)")

Nvidia K80, cuda 9.2

![stencil bench nvidia k80](https://github.com/pkestene/kokkos-proj-tmpl/raw/master/doc/stencil/stencil_bench_ouessant_k80.png "Nvidia K80, cuda 9.2")

Nvidia P100, cuda 9.2

![stencil bench nvidia p100](https://github.com/pkestene/kokkos-proj-tmpl/raw/master/doc/stencil/stencil_bench_ouessant_p100.png "Nvidia P100, cuda 9.2")

