# Two alternatives:
# 1. If KOKKOS_PROJ_TMPL_BUILD_KOKKOS is ON, we download kokkos sources and build them using FetchContent (which actually uses add_subdirectory)
# 2. If KOKKOS_PROJ_TMPL_BUILD_KOKKOS is OFF (default), we don't build kokkos, but use find_package for setup (you must have kokkos already installed)

# NOTE about required C++ standard
# we better chose to set the minimum C++ standard level if not already done:
# - when building kokkos <  4.0.00, it defaults to c++-14
# - when building kokkos >= 4.0.00, it defaults to c++-17
# - when using installed kokkos, we set C++ standard according to kokkos version

#
# Do we want to build kokkos (https://github.com/kokkos/kokkos) ?
#
option(KOKKOS_PROJ_TMPL_BUILD_KOKKOS "Turn ON if you want to build kokkos (default: ON)" ON)

#
# Option to use git (instead of tarball release) for downloading kokkos
#
option(KOKKOS_PROJ_TMPL_USE_GIT_KOKKOS "Turn ON if you want to use git to download Kokkos sources (default: OFF)" OFF)

#
# Options to specify target device backend
#

# set default backend
set(KOKKOS_PROJ_TMPL_BACKEND "Undefined" CACHE STRING
  "Kokkos default backend device")

# Set the possible values for kokkos backend device
set_property(CACHE KOKKOS_PROJ_TMPL_BACKEND PROPERTY STRINGS
  "OpenMP" "Cuda" "HIP" "Undefined")

# check if user requested a build of kokkos
if(KOKKOS_PROJ_TMPL_BUILD_KOKKOS)

  message("[kokkos_proj_tmpl / kokkos] Building kokkos from source")

  # Kokkos default build options


  # use predefined cmake args
  # can be override on the command line
  if (KOKKOS_PROJ_TMPL_BACKEND MATCHES "Cuda")

    if ((NOT DEFINED Kokkos_ENABLE_HWLOC) OR (NOT Kokkos_ENABLE_HWLOC))
      set(Kokkos_ENABLE_HWLOC ON CACHE BOOL "")
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_OPENMP) OR (NOT Kokkos_ENABLE_OPENMP))
      set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_CUDA) OR (NOT Kokkos_ENABLE_CUDA))
      set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_CUDA_LAMBDA) OR (NOT Kokkos_ENABLE_CUDA_LAMBDA))
      set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "")
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_CUDA_CONSTEXPR) OR (NOT Kokkos_ENABLE_CUDA_CONSTEXPR))
      set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "")
    endif()

    # Note : cuda architecture will probed by kokkos cmake configure

  elseif(KOKKOS_PROJ_TMPL_BACKEND MATCHES "HIP")

    if ((NOT DEFINED Kokkos_ENABLE_HWLOC) OR (NOT Kokkos_ENABLE_HWLOC))
      set(Kokkos_ENABLE_HWLOC ON CACHE BOOL "")
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_OPENMP) OR (NOT Kokkos_ENABLE_OPENMP))
      set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_HIP) OR (NOT Kokkos_ENABLE_HIP))
      set(Kokkos_ENABLE_HIP ON CACHE BOOL "")
    endif()

  elseif(KOKKOS_PROJ_TMPL_BACKEND MATCHES "OpenMP")

    if ((NOT DEFINED Kokkos_ENABLE_HWLOC) OR (NOT Kokkos_ENABLE_HWLOC))
      set(Kokkos_ENABLE_HWLOC ON CACHE BOOL "")
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_OPENMP) OR (NOT Kokkos_ENABLE_OPENMP))
      set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
    endif()

  elseif(KOKKOS_PROJ_TMPL_BACKEND MATCHES "Undefined")

    message(FATAL_ERROR "[kokkos_proj_tmpl / kokkos] You must chose a valid KOKKOS_PROJ_TMPL_BACKEND !")

  endif()

  # we set c++ standard to c++-17 as kokkos >= 4.0.00 requires at least c++-17
  if (NOT "${CMAKE_CXX_STANDARD}")
    set(CMAKE_CXX_STANDARD 17)
  endif()

  #find_package(Git REQUIRED)
  include (FetchContent)

  if (KOKKOS_PROJ_TMPL_USE_GIT_KOKKOS)
    FetchContent_Declare( kokkos_external
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG 4.0.00
      )
  else()
    FetchContent_Declare( kokkos_external
      URL https://github.com/kokkos/kokkos/archive/refs/tags/4.0.00.tar.gz
      )
  endif()

  # Import kokkos targets (download, and call add_subdirectory)
  FetchContent_MakeAvailable(kokkos_external)

  if(TARGET Kokkos::kokkos)
    message("[kokkos_proj_tmpl / kokkos] Kokkos found (using FetchContent)")
    set(KOKKOS_PROJ_TMPL_KOKKOS_FOUND True)
    set(HAVE_KOKKOS 1)
  else()
    message("[kokkos_proj_tmpl / kokkos] we shouldn't be here. We've just integrated kokkos build into kokkos_proj_tmpl build !")
  endif()

  set(KOKKOS_PROJ_TMPL_BUILTIN TRUE)

else()

  #
  # check if an already installed kokkos exists
  #
  find_package(Kokkos 3.7.00 REQUIRED)

  if(TARGET Kokkos::kokkos)

    # set default c++ standard according to Kokkos version
    # Kokkos >= 4.0.00 requires c++-17
    if (NOT "${CMAKE_CXX_STANDARD}")
      if ( ${Kokkos_VERSION} VERSION_LESS 4.0.00)
        set(CMAKE_CXX_STANDARD 14)
      else()
        set(CMAKE_CXX_STANDARD 17)
      endif()
    endif()

    # kokkos_check is defined in KokkosConfigCommon.cmake
    kokkos_check( DEVICES "OpenMP" RETURN_VALUE KOKKOS_DEVICE_ENABLE_OPENMP)
    kokkos_check( DEVICES "Cuda" RETURN_VALUE KOKKOS_DEVICE_ENABLE_CUDA)
    kokkos_check( DEVICES "HIP" RETURN_VALUE KOKKOS_DEVICE_ENABLE_HIP)

    kokkos_check( TPLS "HWLOC" RETURN_VALUE Kokkos_TPLS_HWLOC_ENABLED)

    if(KOKKOS_DEVICE_ENABLE_CUDA)
      set(KOKKOS_PROJ_TMPL_BACKEND "Cuda")
      kokkos_check( OPTIONS CUDA_LAMBDA RETURN_VALUE Kokkos_CUDA_LAMBDA_ENABLED)
      kokkos_check( OPTIONS CUDA_CONSTEXPR RETURN_VALUE Kokkos_CUDA_CONSTEXPR_ENABLED)
      kokkos_check( OPTIONS CUDA_UVM RETURN_VALUE Kokkos_CUDA_UVM_ENABLED)
    elseif(KOKKOS_DEVICE_ENABLE_HIP)
      set(KOKKOS_PROJ_TMPL_BACKEND "HIP")
    elseif(KOKKOS_DEVICE_ENABLE_OPENMP)
      set(KOKKOS_PROJ_TMPL_BACKEND "OpenMP")
    endif()

    message("[kokkos_proj_tmpl / kokkos] Kokkos found via find_package; default backend is ${KOKKOS_PROJ_TMPL_BACKEND}")
    set(KOKKOS_PROJ_TMPL_KOKKOS_FOUND True)
    set(HAVE_KOKKOS 1)

  else()

    message(FATAL_ERROR "[kokkos_proj_tmpl / kokkos] Kokkos is required but not found by find_package. Please adjet your env variable CMAKE_PREFIX_PATH (or Kokkos_ROOT) to where Kokkos is installed on your machine !")

  endif()

endif()
