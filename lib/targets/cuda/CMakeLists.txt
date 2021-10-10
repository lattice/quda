# add target specific files / options
target_sources(quda_cpp PRIVATE quda_api.cpp device.cpp malloc.cpp blas_lapack_cublas.cpp comm_target.cpp)

if(QUDA_JITIFY)
  target_sources(quda_cpp PRIVATE jitify_helper.cpp)
endif()

if(QUDA_BACKWARDS)
  set_property(SOURCE malloc.cpp DIRECTORY ${CMAKE_SOURCE_DIR}/lib APPEND PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE malloc.cpp DIRECTORY ${CMAKE_SOURCE_DIR}/lib APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
endif()


target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xfatbin=-compress-all>)
