if("x${CMAKE_SOURCE_DIR}" STREQUAL "x${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "\
In-source build is not a good practice.
Please use:
  mkdir build
  cd build
  cmake ..
to build this project"
  )
endif()

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(languages CXX)
set(_FT_WITH_CUDA ON)

# the following settings are modified from cub/CMakeLists.txt
set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")


find_program(FT_HAS_NVCC nvcc)
if(NOT FT_HAS_NVCC AND "$ENV{CUDACXX}" STREQUAL "")
  message(STATUS "No NVCC detected. Disable CUDA support")
  set(_FT_WITH_CUDA OFF)
endif()

if(APPLE OR (DEFINED FT_WITH_CUDA AND NOT FT_WITH_CUDA))
  if(_FT_WITH_CUDA)
    message(STATUS "Disable CUDA support")
    set(_FT_WITH_CUDA OFF)
  endif()
endif()

if(_FT_WITH_CUDA)
  set(languages ${languages} CUDA)
  if(NOT DEFINED FT_WITH_CUDA)
    set(FT_WITH_CUDA ON)
  endif()
endif()

message(STATUS "Enabled languages: ${languages}")

project(fast_rnnt ${languages})

set(FT_VERSION "1.0")

set(ALLOWABLE_BUILD_TYPES Debug Release RelWithDebInfo MinSizeRel)
set(DEFAULT_BUILD_TYPE "Release")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${ALLOWABLE_BUILD_TYPES}")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # CMAKE_CONFIGURATION_TYPES: with config type values from other generators (IDE).
  message(STATUS "No CMAKE_BUILD_TYPE given, default to ${DEFAULT_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}")
elseif(NOT CMAKE_BUILD_TYPE IN_LIST ALLOWABLE_BUILD_TYPES)
  message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}, \
    choose one from ${ALLOWABLE_BUILD_TYPES}")
endif()

option(FT_BUILD_TESTS "Whether to build tests or not" ON)
option(BUILD_SHARED_LIBS "Whether to build shared libs" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_RPATH "$ORIGIN")

if(FT_WITH_CUDA)
  add_definitions(-DFT_WITH_CUDA)
  # Force CUDA C++ standard to be the same as the C++ standard used.
  #
  # Now, CMake is unaligned with reality on standard versions: https://gitlab.kitware.com/cmake/cmake/issues/18597
  # which means that using standard CMake methods, it's impossible to actually sync the CXX and CUDA versions for pre-11
  # versions of C++; CUDA accepts 98 but translates that to 03, while CXX doesn't accept 03 (and doesn't translate that to 03).
  # In case this gives You, dear user, any trouble, please escalate the above CMake bug, so we can support reality properly.
  if(DEFINED CMAKE_CUDA_STANDARD)
    message(WARNING "You've set CMAKE_CUDA_STANDARD; please note that this variable is ignored, and CMAKE_CXX_STANDARD"
      " is used as the C++ standard version for both C++ and CUDA.")
  endif()


  unset(CMAKE_CUDA_STANDARD CACHE)
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

  include(cmake/select_compute_arch.cmake)
  cuda_select_nvcc_arch_flags(FT_COMPUTE_ARCH_FLAGS)
  message(STATUS "FT_COMPUTE_ARCH_FLAGS: ${FT_COMPUTE_ARCH_FLAGS}")

  # set(OT_COMPUTE_ARCHS 30 32 35 50 52 53 60 61 62 70 72)
  # message(WARNING "arch 62/72 are not supported for now")

  # see https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  # https://www.myzhar.com/blog/tutorials/tutorial-nvidia-gpu-cuda-compute-capability/
  set(FT_COMPUTE_ARCH_CANDIDATES 35 50 60 61 70 75)
  if(CUDA_VERSION VERSION_GREATER "11.0")
    list(APPEND FT_COMPUTE_ARCH_CANDIDATES 80 86)
  endif()
  message(STATUS "FT_COMPUTE_ARCH_CANDIDATES ${FT_COMPUTE_ARCH_CANDIDATES}")

  set(FT_COMPUTE_ARCHS)

  foreach(COMPUTE_ARCH IN LISTS FT_COMPUTE_ARCH_CANDIDATES)
    if("${FT_COMPUTE_ARCH_FLAGS}" MATCHES ${COMPUTE_ARCH})
      message(STATUS "Adding arch ${COMPUTE_ARCH}")
      list(APPEND FT_COMPUTE_ARCHS ${COMPUTE_ARCH})
    else()
      message(STATUS "Skipping arch ${COMPUTE_ARCH}")
    endif()
  endforeach()

  if(NOT FT_COMPUTE_ARCHS)
    set(FT_COMPUTE_ARCHS ${FT_COMPUTE_ARCH_CANDIDATES})
  endif()

  message(STATUS "FT_COMPUTE_ARCHS: ${FT_COMPUTE_ARCHS}")

  foreach(COMPUTE_ARCH IN LISTS FT_COMPUTE_ARCHS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -gencode arch=compute_${COMPUTE_ARCH},code=sm_${COMPUTE_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES "${COMPUTE_ARCH}-real;${COMPUTE_ARCH}-virtual;${CMAKE_CUDA_ARCHITECTURES}")
  endforeach()
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(pybind11)
include(torch)
if(FT_WITH_CUDA AND CUDA_VERSION VERSION_LESS 11.0)
  # CUB is included in CUDA toolkit 11.0 and above
  include(cub)
endif()

if(FT_BUILD_TESTS)
  enable_testing()
  include(googletest)
endif()

add_subdirectory(fast_rnnt)
