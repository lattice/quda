<<<<<<< HEAD
# generate an object library for all target specific files
add_library(quda_cuda_target OBJECT quda_api.cpp device.cpp malloc.cpp tune.cpp blas_lapack_cublas.cpp quda_fortran.cpp
comm_common.cpp
# CUDA specific kernels that need to be generalized
gauge_fix_ovr.cu
)


if(QUDA_BACKWARDS)
  target_include_directories(quda_cpp SYSTEM PRIVATE ${backward-cpp_SOURCE_DIR})
  set_property(
    SOURCE comm_common.cpp
    APPEND
    PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE comm_common.cpp APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
  target_link_libraries(quda PUBLIC ${BACKWARD_LIBRARIES})
endif()



if(QUDA_BUILD_SHAREDLIB)
  set_target_properties(quda_cuda_target PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
=======
# add target specific files / options
target_sources(quda_cpp PRIVATE quda_api.cpp device.cpp malloc.cpp tune.cpp blas_lapack_cublas.cpp)

if(QUDA_JITIFY)
  target_sources(quda_cpp PRIVATE jitify_helper.cpp)
>>>>>>> feature/generic_kernel
endif()
