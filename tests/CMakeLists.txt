
include_directories(.)

#build a common library for all test utilities
set(QUDA_TEST_COMMON gtest-all.cc test_util.cpp misc.cpp)
cuda_add_library(quda_test STATIC ${QUDA_TEST_COMMON})
set(TEST_LIBS quda quda_test)


MACRO (QUDA_CHECKBUILDTEST mytarget qudabuildtests)
  IF(NOT ${qudabuildtests})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
  ENDIF()
ENDMACRO()
#define tests

if(QUDA_DIRAC_WILSON OR QUDA_DIRAC_CLOVER OR QUDA_DIRAC_TWISTED_MASS OR QUDA_DIRAC_TWISTED_CLOVER OR QUDA_DIRAC_DOMAIN_WALL)
  cuda_add_executable(dslash_test dslash_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp clover_reference.cpp blas_reference.cpp)
  target_link_libraries(dslash_test ${TEST_LIBS} )
  QUDA_CHECKBUILDTEST(dslash_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(invert_test invert_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp clover_reference.cpp blas_reference.cpp)
  target_link_libraries(invert_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(invert_test QUDA_BUILD_ALL_TESTS)


  if(QUDA_BLOCKSOLVER)
    cuda_add_executable(invertmsrc_test invertmsrc_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp)
    target_link_libraries(invertmsrc_test ${TEST_LIBS})
    QUDA_CHECKBUILDTEST(invertmsrc_test QUDA_BUILD_ALL_TESTS)
  endif()

endif()

cuda_add_executable(deflation_test deflation_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp)
target_link_libraries(deflation_test ${TEST_LIBS})
QUDA_CHECKBUILDTEST(deflation_test QUDA_BUILD_ALL_TESTS)

if(QUDA_DIRAC_STAGGERED)
  cuda_add_executable(staggered_dslash_test staggered_dslash_test.cpp staggered_dslash_reference.cpp blas_reference.cpp)
  target_link_libraries(staggered_dslash_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(staggered_dslash_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(staggered_invert_test staggered_invert_test.cpp staggered_dslash_reference.cpp blas_reference.cpp)
  target_link_libraries(staggered_invert_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(staggered_invert_test QUDA_BUILD_ALL_TESTS)

  if(QUDA_BLOCKSOLVER)
    cuda_add_executable(staggered_invertmsrc_test staggered_invertmsrc_test.cpp  staggered_dslash_reference.cpp  blas_reference.cpp)
    target_link_libraries(staggered_invertmsrc_test ${TEST_LIBS})
    QUDA_CHECKBUILDTEST(staggered_invertmsrc_test QUDA_BUILD_ALL_TESTS)
  endif()
endif()

if(QUDA_MULTIGRID)
  cuda_add_executable(multigrid_invert_test multigrid_invert_test.cpp wilson_dslash_reference.cpp clover_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp)
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

if(QUDA_LINK_ASQTAD OR QUDA_LINK_HISQ)
  cuda_add_executable(llfat_test llfat_test.cpp llfat_reference.cpp)
  target_link_libraries(llfat_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(llfat_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_LINK_HISQ)
  cuda_add_executable(unitarize_link_test unitarize_link_test.cpp)
  target_link_libraries(unitarize_link_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(unitarize_link_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_FORCE_GAUGE)
  cuda_add_executable(gauge_force_test gauge_force_test.cpp gauge_force_reference.cpp)
  target_link_libraries(gauge_force_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(gauge_force_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_FORCE_ASQTAD)
  cuda_add_executable(fermion_force_test fermion_force_test.cpp fermion_force_reference.cpp)
  target_link_libraries(fermion_force_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(fermion_force_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_GAUGE_ALG)
  cuda_add_executable(gauge_alg_test gauge_alg_test.cpp)
  target_link_libraries(gauge_alg_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(gauge_alg_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_FORCE_HISQ)
  cuda_add_executable(hisq_paths_force_test hisq_paths_force_test.cpp hisq_force_reference.cpp hisq_force_reference2.cpp fermion_force_reference.cpp   )
  target_link_libraries(hisq_paths_force_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(hisq_paths_force_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(hisq_unitarize_force_test hisq_unitarize_force_test.cpp hisq_force_reference.cpp )
  target_link_libraries(hisq_unitarize_force_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(hisq_unitarize_force_test QUDA_BUILD_ALL_TESTS)
endif()
