include_directories(${CMAKE_SOURCE_DIR})

# it is located in fast_rnnt/cmake/transform.cmake
include(transform)

set(srcs
  mutual_information_cpu.cu
  utils.cu
)

if(NOT FT_WITH_CUDA)
  transform(OUTPUT_VARIABLE srcs SRCS ${srcs})
else()
  list(APPEND srcs mutual_information_cuda.cu)
endif()

add_library(mutual_information_core ${srcs})
target_link_libraries(mutual_information_core PUBLIC ${TORCH_LIBRARIES})
# for <torch/extension.h>
target_include_directories(mutual_information_core PUBLIC ${PYTHON_INCLUDE_DIRS})
# see https://github.com/NVIDIA/thrust/issues/1401
# and https://github.com/k2-fsa/k2/pull/917
target_compile_definitions(mutual_information_core PUBLIC CUB_WRAPPED_NAMESPACE=fast_rnnt)
target_compile_definitions(mutual_information_core PUBLIC THRUST_NS_QUALIFIER=thrust)
if(FT_WITH_CUDA AND CUDA_VERSION VERSION_LESS 11.0)
  target_link_libraries(mutual_information_core PUBLIC cub)
endif()
