
include_directories(.)

set(QUDA_TEST_COMMON gtest-all.cc test_util.cpp  misc.cpp )

if(${BUILD_WILSON_DIRAC} OR ${BUILD_DOMAIN_WALL_DIRAC})
  cuda_add_executable(dslash_test dslash_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp ${QIO_UTIL} ${QUDA_TEST_COMMON})
  target_link_libraries(dslash_test quda)

  cuda_add_executable(invert_test invert_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp ${QIO_UTIL} ${QUDA_TEST_COMMON})
  target_link_libraries(invert_test quda)
endif()

cuda_add_executable(deflation_test deflation_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp ${QIO_UTIL} ${QUDA_TEST_COMMON})
target_link_libraries(deflation_test quda)

if(${BUILD_STAGGERED_DIRAC})
  cuda_add_executable(staggered_dslash_test staggered_dslash_test.cpp   staggered_dslash_reference.cpp  ${QUDA_TEST_COMMON})
  target_link_libraries(staggered_dslash_test quda)

  cuda_add_executable(staggered_invert_test staggered_invert_test.cpp  staggered_dslash_reference.cpp  blas_reference.cpp ${QUDA_TEST_COMMON})
  target_link_libraries(staggered_invert_test quda)
endif()

cuda_add_executable(su3_test su3_test.cpp   ${QIO_UTIL} ${QUDA_TEST_COMMON})
target_link_libraries(su3_test quda)

cuda_add_executable(pack_test pack_test.cpp   ${QUDA_TEST_COMMON})
target_link_libraries(pack_test quda)

cuda_add_executable(blas_test blas_test.cu    ${QUDA_TEST_COMMON})
target_link_libraries(blas_test quda)

if(${BUILD_FATLINK} OR {BUILD_HISQLINK})
  cuda_add_executable(llfat_test llfat_test.cpp llfat_reference.cpp   ${QUDA_TEST_COMMON})
  target_link_libraries(staggered_dslash_test quda)
endif()

if(${BUILD_HISQLINK})
  cuda_add_executable(unitarize_link_test unitarize_link_test.cpp   ${QUDA_TEST_COMMON})
  target_link_libraries(unitarize_link_test quda)
endif()

if(${BUILD_GAUGE_FORCE})
  cuda_add_executable(gauge_force_test gauge_force_test.cpp gauge_force_reference.cpp   ${QUDA_TEST_COMMON})
  target_link_libraries(gauge_force_test quda)
endif()

if(${BUILD_FERMION_FORCE})
  cuda_add_executable(fermion_force_test fermion_force_test.cpp fermion_force_reference.cpp   ${QUDA_TEST_COMMON})
  target_link_libraries(fermion_force_test quda)
endif()

if(${BUILD_HISQ_FORCE})
  cuda_add_executable(hisq_paths_force_test hisq_paths_force_test.cpp hisq_force_reference.cpp hisq_force_reference2.cpp fermion_force_reference.cpp   ${QUDA_TEST_COMMON})
  target_link_libraries(hisq_paths_force_test quda)

  cuda_add_executable(hisq_unitarize_force_test hisq_unitarize_force_test.cpp hisq_force_reference.cpp )
  target_link_libraries(hisq_unitarize_force_test quda)
endif()
