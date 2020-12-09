# add target specific files / options
target_sources(quda_cpp PRIVATE quda_api.cpp device.cpp malloc.cpp tune.cpp blas_lapack_cublas.cpp comm_common.cpp gauge_fix_over.cu)

if(QUDA_JITIFY)
  target_sources(quda_cpp PRIVATE jitify_helper.cpp)
endif()
