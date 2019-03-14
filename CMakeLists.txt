cmake_minimum_required(VERSION 3.3)

project(kokkos-proj-tmpl CXX)

# C++11 is for Kokkos
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

#
# sources
#

# first build kokkos
add_subdirectory(external/kokkos)

# pass Kokkos include directories to our target application
include_directories(${Kokkos_INCLUDE_DIRS_RET})

# build the miniapp
add_subdirectory(src)
