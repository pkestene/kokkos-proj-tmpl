# use minimum version required by Kokkos
cmake_minimum_required(VERSION 3.16)

project(kokkos-proj-tmpl
  LANGUAGES CXX)

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++17 is for Kokkos (to be able to use KOKKOS_CLASS_LAMBDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(USE_SIMD_KOKKOS "Should we enable simd from kokkos" OFF)
option(USE_CBLAS_SERIAL "Should we use serial cblas for BatchedDotProduct benchmark ?" OFF)
if (USE_CBLAS_SERIAL)
  message(STATUS "[FYI] Make sure to have openblas-serial installed.")
  message(STATUS "[FYI] To make sure openblas-serial will be selected use env variable CMAKE_PREFIX_PATH")
  message(STATUS "[FYI] e.g. CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/openblas-serial")
  #set(BLA_VENDOR OpenBLAS)
  #set(BLA_PREFER_PKGCONFIG ON)
  #find_package(BLAS)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(OPENBLAS QUIET IMPORTED_TARGET openblas)
  if(OPENBLAS_FOUND)
    message("openblas found via pkg-config")
    add_library(openblas::openblas ALIAS PkgConfig::OPENBLAS)
  else()
    message(FATAL_ERROR "openblas NOT FOUND, selected another flavor or adjust env variable PKG_CONFIG_PATH")
  endif()

endif()

#
# sources
#

# configure kokkos
include(cmake/build_or_find_kokkos.cmake)

# build the miniapp
add_subdirectory(src)

##################### PRINT CONFIGURE STATUS ######################
message("//===================================================")
message("// ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
                         "${CMAKE_CXX_COMPILER_VERSION} "
                         "${CMAKE_CXX_COMPILER_WRAPPER}")
message("                 ${CMAKE_CXX_COMPILER}")

message("  Kokkos version               = ${Kokkos_VERSION}")
message("  Kokkos_CXX_COMPILER          = ${Kokkos_CXX_COMPILER}")
message("  Kokkos_CXX_COMPILER_ID       = ${Kokkos_CXX_COMPILER_ID}")
message("  Kokkos_CXX_STANDARD          = ${Kokkos_CXX_STANDARD}")
message("  Kokkos_OPTIONS               = ${Kokkos_OPTIONS}")
message("  Kokkos_TPLS                  = ${Kokkos_TPLS}")
message("  Kokkos_DIRS                  = ${Kokkos_DIR}")

if(Kokkos_ENABLE_OPENMP)
  message("  Kokkos_ENABLE_OPENMP         = ${Kokkos_ENABLE_OPENMP}")
endif()

if(Kokkos_ENABLE_CUDA)
  message("  Kokkos_ENABLE_CUDA           = ${Kokkos_ENABLE_CUDA}")

  if( (${Kokkos_CUDA_LAMBDA_ENABLED}) OR (${Kokkos_ENABLE_CUDA_LAMBDA}))
    message("  Kokkos_ENABLE_CUDA_LAMBDA    = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_LAMBDA    = OFF")
  endif()

  if( (${Kokkos_CUDA_CONSTEXPR_ENABLED}) OR (${Kokkos_ENABLE_CUDA_CONSTEXPR}))
    message("  Kokkos_ENABLE_CUDA_CONSTEXPR = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_CONSTEXPR = OFF")
  endif()

  if( (${Kokkos_CUDA_UVM_ENABLED}) OR (${Kokkos_ENABLE_CUDA_UVM}))
    message("  Kokkos_ENABLE_CUDA_UVM       = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_UVM       = OFF")
  endif()

  message("  Kokkos CUDA   flags          = ${KOKKOS_CUDA_OPTIONS}")
  #message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
  #message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
  #message("  CUDA Compile flags : ${CUDA_NVCC_FLAGS}")
endif(Kokkos_ENABLE_CUDA)

if(Kokkos_ENABLE_HIP)
  message("  Kokkos_ENABLE_HIP            = ${Kokkos_ENABLE_HIP}")
endif(Kokkos_ENABLE_HIP)

if ( (${Kokkos_TPLS_HWLOC_ENABLED}) OR (${Kokkos_ENABLE_HWLOC}) )
  message("  Kokkos_ENABLE_HWLOC          = ON")
else()
  message("  Kokkos_ENABLE_HWLOC          = OFF")
endif()

message("  Kokkos architecture          = ${Kokkos_ARCH}")

message("")
message("  USE_SIMD_KOKKOS              = ${USE_SIMD_KOKKOS}")
message("  USE_CBLAS_SERIAL             = ${USE_CBLAS_SERIAL}")
message("")
