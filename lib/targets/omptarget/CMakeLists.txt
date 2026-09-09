# add target specific files / options
target_sources(quda_cpp PRIVATE blas_lapack_mkl.cpp comm_target.cpp device.cpp malloc.cpp quda_api.cpp)
target_sources(quda_cpp PRIVATE reproducible_floating_accumulator.cpp)
