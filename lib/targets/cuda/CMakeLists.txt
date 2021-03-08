# add target specific files / options
target_sources(quda_cpp PRIVATE quda_api.cpp device.cpp malloc.cpp tune.cpp blas_lapack_cublas.cpp)

if(QUDA_BACKWARDS)
  set_property(SOURCE malloc.cpp DIRECTORY ${CMAKE_SOURCE_DIR}/lib APPEND PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE malloc.cpp DIRECTORY ${CMAKE_SOURCE_DIR}/lib APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
  message("BACKWARDS")
endif()