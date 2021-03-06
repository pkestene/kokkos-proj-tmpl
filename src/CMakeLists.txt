#
# define a macro to build an executable
#
macro(build_exec name)

  # define executable name
  if (Kokkos_ENABLE_CUDA)
    set(EXE_NAME "${name}.cuda")
  elseif(Kokkos_ENABLE_OPENMP)
    set(EXE_NAME "${name}.openmp")
  elseif(Kokkos_ENABLE_PTHREAD)
    set(EXE_NAME "${name}.pthread")
  else()
    set(EXE_NAME "${name}.serial")
  endif()
  
  # define list of source files
  set (EXE_NAME_SRC
    ${name}.cpp
    )

  # add timer
  if (Kokkos_ENABLE_CUDA)
    set (EXE_NAME_SRC
      ${EXE_NAME_SRC}
      CudaTimer.h
      )
  elseif (Kokkos_ENABLE_OPENMP)
    set (EXE_NAME_SRC
      ${EXE_NAME_SRC}
      OpenMPTimer.h
      OpenMPTimer.cpp
      )
  else()
    set (EXE_NAME_SRC
      ${EXE_NAME_SRC}
      SimpleTimer.h
      SimpleTimer.cpp
      )
  endif()

  if (${name} STREQUAL KokkosStream_bench)
    set (EXE_NAME_SRC
      ${EXE_NAME_SRC}
      Stream.h
      KokkosStream.hpp
      KokkosStream.cpp
      )
  endif()
    
  # define how to build and link
  add_executable(${EXE_NAME}
    ${EXE_NAME_SRC})
  
  target_link_libraries(${EXE_NAME} kokkos)
  if (Kokkos_ENABLE_HWLOC)
    target_link_libraries(${EXE_NAME} hwloc)
  endif()
  
endmacro(build_exec)

####################################################
build_exec(saxpy_kokkos_lambda)
build_exec(compute_pi)
build_exec(deep_copy)
build_exec(stencil)
build_exec(KokkosStream_bench)
  

