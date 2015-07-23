
if(${BUILD_WILSON_DIRAC})
  set(DIRAC_TEST dslash_test invert_test)
endif(${BUILD_WILSON_DIRAC})

set(QUDA_TEST_COMMON gtest-all.cc test_util.cpp  misc.cpp )

cuda_add_executable(staggered_dslash_test staggered_dslash_test.cpp staggered_dslash_reference.cpp ${QUDA_TEST_COMMON})
target_link_libraries(staggered_dslash_test quda)
