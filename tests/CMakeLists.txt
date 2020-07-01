include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_FULL_LIBDIR})

# enable tests build a common library for all test utilities
add_library(quda_test STATIC googletest/src/gtest-all.cc)
target_include_directories(quda_test PUBLIC googletest/include googletest)

set(TEST_LIBS quda quda_test quda_utils)

macro(QUDA_CHECKBUILDTEST mytarget qudabuildtests)
  # adding the linker language here as a workaround -- was not needed for cmake 3.16
  set_target_properties(${mytarget} PROPERTIES LINKER_LANGUAGE CUDA)
  if(NOT ${qudabuildtests})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
    set(QUDA_EXCLUDE_FROM_INSTALL "EXCLUDE_FROM_ALL")
  endif()

endmacro()

if(NOT ${QUDA_INSTALL_ALL_TESTS})
  set(QUDA_EXCLUDE_FROM_INSTALL "EXCLUDE_FROM_ALL")
endif()

# define tests

if(QUDA_DIRAC_WILSON
   OR QUDA_DIRAC_CLOVER
   OR QUDA_DIRAC_TWISTED_MASS
   OR QUDA_DIRAC_TWISTED_CLOVER
   OR QUDA_DIRAC_NDEG_TWISTED_MASS
   OR QUDA_DIRAC_DOMAIN_WALL
   OR QUDA_DIRAC_STAGGERED)
  add_executable(blas_test blas_test.cpp)
  target_link_libraries(blas_test ${TEST_LIBS})
  quda_checkbuildtest(blas_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS blas_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# use FindMPI variables for QUDA_CTEST_LAUNCH set MPIEXEC_MAX_NUMPROCS to the
# number of ranks you want to launch
set(QUDA_CTEST_LAUNCH ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
                      ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS})

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

