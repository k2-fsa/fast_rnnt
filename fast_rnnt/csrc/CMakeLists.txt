include_directories(${CMAKE_SOURCE_DIR})

set(srcs
  mutual_information_cpu.cc
)

add_library(mutual_information_core ${srcs})
target_link_libraries(mutual_information_core PUBLIC ${TORCH_LIBRARIES})

if(FT_WITH_CUDA)
  set(cuda_srcs mutual_information_cuda.cu)
  add_library(mutual_information_core_cuda ${cuda_srcs})
  target_link_libraries(mutual_information_core_cuda PUBLIC ${TORCH_LIBRARIES})
  # for <torch/extension.h>
  target_include_directories(mutual_information_core_cuda PUBLIC ${PYTHON_INCLUDE_DIRS})
  target_link_libraries(mutual_information_core PUBLIC mutual_information_core_cuda)
endif()
