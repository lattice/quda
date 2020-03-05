include_directories(. googletest/include googletest)

# enable_language(Fortran)

# enable tests build a common library for all test utilities
set(QUDA_TEST_COMMON googletest/src/gtest-all.cc test_util.cpp test_params.cpp misc.cpp)
if(${QUDA_BACKEND_TYPE} STREQUAL "CUDA")
  cuda_add_library(quda_test STATIC ${QUDA_TEST_COMMON})
else()
  add_library(quda_test STATIC ${QUDA_TEST_COMMON})
endif()

set(TEST_LIBS quda_test quda)

macro(QUDA_CHECKBUILDTEST mytarget qudabuildtests)
  if(NOT ${qudabuildtests})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
  endif()

endmacro()

# define tests

# if we build with QDP JIT the tests cannot run anyway


if(QUDA_DIRAC_WILSON OR QUDA_DIRAC_CLOVER OR QUDA_DIRAC_TWISTED_MASS OR QUDA_DIRAC_TWISTED_CLOVER OR QUDA_DIRAC_NDEG_TWISTED_MASS OR QUDA_DIRAC_DOMAIN_WALL OR QUDA_DIRAC_STAGGERED)
  if(${QUDA_BACKEND_TYPE} STREQUAL "CUDA")
    cuda_add_executable(blas_test blas_test.cu)
  else()
    #set_source_files_properties(blas_test.cu PROPERTIES LANGUAGE CXX)
    #add_executable(blas_test blas_test.cu)
    add_executable(blas_test blas_test.cpp)
  endif()
  target_link_libraries(blas_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(blas_test QUDA_BUILD_ALL_TESTS)
endif()

add_executable(blas_test blas_test.cpp)
target_link_libraries(blas_test ${TEST_LIBS})

if(${QUDA_BACKEND_TYPE} STREQUAL "CUDA")
  cuda_add_executable(plaq_test plaq_test.cpp)
  target_link_libraries(plaq_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(plaq_test QUDA_BUILD_ALL_TESTS)
else()
  #add_executable(plaq_test plaq_test.cpp)
  #target_link_libraries(plaq_test ${TEST_LIBS})
  #QUDA_CHECKBUILDTEST(plaq_test QUDA_BUILD_ALL_TESTS)
endif()

#cuda_add_executable(su3_test su3_test.cpp)
#target_link_libraries(su3_test ${TEST_LIBS})
#quda_checkbuildtest(su3_test QUDA_BUILD_ALL_TESTS)

# use FindMPI variables for QUDA_CTEST_LAUNCH set MPIEXEC_MAX_NUMPROCS to the number of ranks you want to launch
set(QUDA_CTEST_LAUNCH ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS})

# BLAS test

if(QUDA_DIRAC_WILSON
   OR QUDA_DIRAC_CLOVER
   OR QUDA_DIRAC_TWISTED_MASS
   OR QUDA_DIRAC_NDEG_TWISTED_MASS
   OR QUDA_DIRAC_TWISTED_CLOVER
   OR QUDA_DIRAC_DOMAIN_WALL)
  add_test(NAME blas_test_parity_wilson
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_test> ${MPIEXEC_POSTFLAGS}
                   --dim 2 4 6 8
                   --solve-type direct-pc
                   --gtest_output=xml:blas_test_parity.xml)
  add_test(NAME blas_test_full_wilson
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_test> ${MPIEXEC_POSTFLAGS}
                   --dim 2 4 6 8
                   --solve-type direct
                   --gtest_output=xml:blas_test_full.xml)
endif()

