function(fast_rnnt_add_py_test source)
  get_filename_component(name ${source} NAME_WE)
  set(name "${name}_py")

  add_test(NAME ${name}
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
  )

  get_filename_component(fast_rnnt_path ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

  set_property(TEST ${name}
    PROPERTY ENVIRONMENT "PYTHONPATH=${fast_rnnt_path}:$<TARGET_FILE_DIR:_fast_rnnt>:$ENV{PYTHONPATH}"
  )
endfunction()

# please sort the files in alphabetic order
set(py_test_files
  mutual_information_test.py
  rnnt_loss_test.py
)

foreach(source IN LISTS py_test_files)
  fast_rnnt_add_py_test(${source})
endforeach()
