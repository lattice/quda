
include_directories(. ../tests ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

set(SOURCES ../tests/wilson_dslash_reference.cpp ../tests/clover_reference.cpp ../tests/domain_wall_dslash_reference.cpp ../tests/blas_reference.cpp)

#set(HEADERS ../tests/wilson_dslash_reference.h )

set(CMAKE_CXX_FLAGS_DEVEL  "${CMAKE_CXX_FLAGS_DEVEL} ${Trilinos_CXX_COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS_HOSTDEBUG "${CMAKE_CXX_FLAGS_HOSTDEBUG} ${Trilinos_CXX_COMPILER_FLAGS}")

set(CMAKE_CUDA_FLAGS_DEVEL  "${CMAKE_CUDA_FLAGS_DEVEL} ${Trilinos_CXX_COMPILER_FLAGS}")
set(CMAKE_CUDA_FLAGS_STRICT "${CMAKE_CUDA_FLAGS_STRICT} ${Trilinos_CXX_COMPILER_FLAGS}")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} ${Trilinos_CXX_COMPILER_FLAGS}")
set(CMAKE_CUDA_FLAGS_HOSTDEBUG "${CMAKE_CUDA_FLAGS_HOSTDEBUG} ${Trilinos_CXX_COMPILER_FLAGS}")
set(CMAKE_CUDA_FLAGS_DEVICEDEBUG "${CMAKE_CUDA_FLAGS_DEVICEDEBUG} ${Trilinos_CXX_COMPILER_FLAGS}")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} ${Trilinos_CXX_COMPILER_FLAGS}")

## enable trilinos tests
#build a common library for all test utilities
set(QUDATEST_LIBS quda quda_test )

if(QUDA_TRILINOS)

cuda_add_executable(trilinos_block_invert_test trilinos_block_invert_test.cpp ${SOURCES})
target_link_libraries(trilinos_block_invert_test ${QUDATEST_LIBS} ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
cuda_add_executable(trilinos_invert_test trilinos_invert_test.cpp ${SOURCES})
target_link_libraries(trilinos_invert_test ${QUDATEST_LIBS} ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

endif()

