
include_directories(.)

#build a common library for all test utilities
set(QUDA_TEST_COMMON gtest-all.cc test_util.cpp misc.cpp)
cuda_add_library(quda_test STATIC ${QUDA_TEST_COMMON})

# if CUDA_CUDA_LIBRARY (driver api) is not found look for stubs
FIND_LIBRARY(CUDA_CUDA_LIBRARY cuda HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib/ ${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs)

set(TEST_LIBS quda quda_test ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIBS} ${CUDA_CUDA_LIBRARY})

if(QUDA_QIO)
  LIST(APPEND TEST_LIBS ${QIO_LIB} ${LIME_LIB})
endif()

if(QUDA_QMP)
  LIST(APPEND TEST_LIBS ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MPI)
  LIST(APPEND TEST_LIBS ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_NUMA_NVML)
  LIST(APPEND TEST_LIBS ${NVML_LIBRARY})
endif()


MACRO (QUDA_CHECKBUILDTEST mytarget qudabuildtests)
  IF(NOT ${qudabuildtests})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
  ENDIF()
ENDMACRO()
#define tests

if(QUDA_DIRAC_WILSON OR QUDA_DIRAC_CLOVER OR QUDA_DIRAC_TWISTED_MASS OR QUDA_DIRAC_TWISTED_CLOVER OR QUDA_DIRAC_DOMAIN_WALL)
  cuda_add_executable(dslash_test dslash_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp)
  target_link_libraries(dslash_test ${TEST_LIBS} )
  QUDA_CHECKBUILDTEST(dslash_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(invert_test invert_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp)
  target_link_libraries(invert_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(invert_test QUDA_BUILD_ALL_TESTS)


  if(QUDA_BLOCKSOLVER)
    cuda_add_executable(invertmsrc_test invertmsrc_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp)
    target_link_libraries(invertmsrc_test ${TEST_LIBS})
    QUDA_CHECKBUILDTEST(invertmsrc_test QUDA_BUILD_ALL_TESTS)
  endif()

endif()

if(QUDA_DIRAC_STAGGERED)
  cuda_add_executable(staggered_dslash_test staggered_dslash_test.cpp staggered_dslash_reference.cpp blas_reference.cpp)
  target_link_libraries(staggered_dslash_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(staggered_dslash_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(staggered_invert_test staggered_invert_test.cpp staggered_dslash_reference.cpp blas_reference.cpp)
  target_link_libraries(staggered_invert_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(staggered_invert_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_MULTIGRID)
  cuda_add_executable(multigrid_invert_test multigrid_invert_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp)
  target_link_libraries(multigrid_invert_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(multigrid_invert_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(multigrid_benchmark_test multigrid_benchmark_test.cu)
  target_link_libraries(multigrid_benchmark_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(multigrid_benchmark_test QUDA_BUILD_ALL_TESTS)
endif()

cuda_add_executable(su3_test su3_test.cpp)
target_link_libraries(su3_test ${TEST_LIBS})
QUDA_CHECKBUILDTEST(su3_test QUDA_BUILD_ALL_TESTS)

cuda_add_executable(pack_test pack_test.cpp)
target_link_libraries(pack_test ${TEST_LIBS})
QUDA_CHECKBUILDTEST(pack_test QUDA_BUILD_ALL_TESTS)

cuda_add_executable(blas_test blas_test.cu)
target_link_libraries(blas_test ${TEST_LIBS})
QUDA_CHECKBUILDTEST(blas_test QUDA_BUILD_ALL_TESTS)

