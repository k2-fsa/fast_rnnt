include_directories(${CMAKE_SOURCE_DIR})

pybind11_add_module(_fast_rnnt
  mutual_information.cu
)
target_link_libraries(_fast_rnnt PRIVATE mutual_information_core)

if(UNIX AND NOT APPLE)
  target_link_libraries(_fast_rnnt
    PRIVATE
      ${PYTHON_LIBRARY}
      ${TORCH_DIR}/lib/libtorch_python.so
  )
endif()

