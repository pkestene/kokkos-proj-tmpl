# define executable name
if (KOKKOS_ENABLE_CUDA)
  set(EXE_NAME "saxpy_kokkos_lambda.cuda")
else()
  set(EXE_NAME "saxpy_kokkos_lambda.openmp")
endif()

# define list of source files
set (EXE_NAME_SRC
  saxpy_kokkos_lambda.cpp
  )

# define how to build and link
add_executable(${EXE_NAME}
  ${EXE_NAME_SRC})

target_link_libraries(${EXE_NAME} kokkos)
if (KOKKOS_ENABLE_HWLOC)
  target_link_libraries(${EXE_NAME} hwloc)
endif()
