<<<<<<< HEAD
# generate an object library for all target specific files
add_library(quda_hip_target OBJECT device.cpp malloc.cpp tune.cpp blas_lapack_hipblas.cpp quda_api.cpp comm_common.cpp)
if(QUDA_BUILD_SHAREDLIB)
  set_target_properties(quda_hip_target PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
endif()
# this part needs cleanup, thinking about what is needed here
target_compile_definitions(quda_hip_target PRIVATE $<TARGET_PROPERTY:quda_cpp,COMPILE_DEFINITIONS>)
target_include_directories(quda_hip_target PRIVATE $<TARGET_PROPERTY:quda_cpp,INCLUDE_DIRECTORIES>)
target_compile_options(quda_hip_target PRIVATE $<TARGET_PROPERTY:quda_cpp,COMPILE_OPTIONS>)
=======
# add target specific files / options
target_sources(quda_cpp PRIVATE  device.cpp malloc.cpp tune.cpp blas_lapack_hipblas.cpp)
>>>>>>> feature/generic_kernel

