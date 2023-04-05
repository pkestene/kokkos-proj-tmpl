# kokkos-proj-tmpl

A minimal cmake based project skeleton for developping a kokkos application

## Download this skeleton

```bash
git clone --recursive git@github.com:pkestene/kokkos-proj-tmpl.git
```

## How to build ?

### Requirement

- [cmake](https://cmake.org/) version 3.16

- **note**: if you are on a fairly recent OS (ex: Ubuntu 21.10, or any OS using glibc >= 2.34), you may need to turn off linking with libdl when using kokkos/cuda backend. See [this issue](https://github.com/kokkos/kokkos/issues/4824), as nvcc (even version 11.6) apparently doesn't seem to handle empty file `/usr/lib/x86_64-linux-gnu/libdl.a` (stub, libdl is integrated into glibc). Hopefully this will be solved in an upcoming cuda release.

```shell
# run this to know your glibc version
ldd --version
```

### Build with target device OpenMP

Default behavior is to download and build kokkos from source; thus you need to specifiy for which hardware target (aka Kokkos backend) you want

```bash
mkdir -p build/openmp
cd build/openmp
CXX=YOUR_COMPILER_HERE cmake -DKOKKOS_PROJ_TMPL_BACKEND=OpenMP ../..
make
# then you can run the application
./src/saxpy_kokkos_lambda.openmp
```

Note that option `-DKokkos_ENABLE_HWLOC=ON` is enabled by default.

If you already have build and installed kokkos for some target backend (OpenMP, Cuda, HIP, etc...), you don't need to specify cmake option `KOKKOS_PROJ_TMPL_BACKEND`, it will determine by the build system but of course you need to set env variable `CMAKE_PREFIX_PATH` to the directory containing file `KokkosConfig.cmake` inside your kokkos installation.

### Build with target device CUDA

You need to have Nvidia compiler `nvcc` in your PATH.

CMake and Kokkos will set the compiler to `nvcc_wrapper` (located in kokkos sources, cloned as git submodule).

```bash
mkdir -p build/cuda
cd build/cuda
cmake -DKOKKOS_PROJ_TMPL_BACKEND=Cuda -DKokkos_ARCH_AMPERE86=ON ../..
make
# then you can run the application as before
./src/saxpy_kokkos_lambda.cuda
```

Of course, you will need to adapt variable **Kokkos_ARCH** to your actual GPU architecture (use cuda sample device_query to probe the architecture).

Depending on your OS, you may need to set variable **Kokkos_CUDA_DIR** to point to your CUDA SDK (if cmake is not able to figure out by itself); e.g. /usr/local/cuda-9.0

### Build with target device HIP (AMD GPU)

CMake and Kokkos will set the compiler to `hipcc` (located in kokkos sources, cloned as git submodule).

Example:
```bash
mkdir build_hip
cd build_hip
cmake -DKOKKOS_PROJ_TMPL_BACKEND=HIP -DKokkos_ARCH_VEGA908=ON ..
make
# then you can run the application as before
./src/saxpy_kokkos_lambda.hip
```

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

### glibc 2.34 and nvlink error

If using glibc version >= 2.34 you get the following link error when building with Cuda backend:

```shell
[ 65%] Linking CXX executable saxpy_kokkos_lambda.cuda
nvlink fatal   : Could not open input file '/usr/lib/x86_64-linux-gnu/libdl.a'
make[2]: *** [src/CMakeFiles/saxpy_kokkos_lambda.cuda.dir/build.make:118: src/saxpy_kokkos_lambda.cuda] Error 1
make[1]: *** [CMakeFiles/Makefile2:1047: src/CMakeFiles/saxpy_kokkos_lambda.cuda.dir/all] Error 2
make: *** [Makefile:136: all] Error 2
```

One temporary solution (until fixed in a future nvcc release ?) is mentionned here:
https://matsci.org/t/lammps-users-kokkos-linker-error-nvidia-libdl-a/41050

for simplicity, you just need to create a fake `libdl.a`, e.g. in current build dirrector
```shell
touch empty.c
gcc -fpic -c empty.c
ar rcsv libdl.a empty.o
```
and then reconfigure cmake with additionnal flag `-DLIBDL_LIBRARY=$PWD/libdl.a` and the build will work as expected.
