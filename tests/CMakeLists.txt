
include_directories(. googletest/include googletest)



## enable tests
#build a common library for all test utilities
set(QUDA_TEST_COMMON googletest/src/gtest-all.cc test_util.cpp misc.cpp face_gauge.cpp)
cuda_add_library(quda_test STATIC ${QUDA_TEST_COMMON})

set(TEST_LIBS quda quda_test )

MACRO (QUDA_CHECKBUILDTEST mytarget qudabuildtests)
  IF(NOT ${qudabuildtests})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
  ENDIF()
ENDMACRO()

if(NOT QUDA_BUILD_ALL_TESTS)
  set(BUILD_TESTING OFF)
endif()

#define tests

# if we build with QDP JIT the tests cannot run anyway
if(QUDA_INTERFACE_QDPJIT)
  set(QUDA_BUILD_ALL_TESTS OFF)
endif()

if(QUDA_DIRAC_WILSON OR QUDA_DIRAC_CLOVER OR QUDA_DIRAC_TWISTED_MASS OR QUDA_DIRAC_TWISTED_CLOVER OR QUDA_DIRAC_NDEG_TWISTED_MASS OR QUDA_DIRAC_DOMAIN_WALL)
  cuda_add_executable(dslash_test dslash_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp clover_reference.cpp blas_reference.cpp)
  cuda_add_executable(dslash_ctest dslash_ctest.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp clover_reference.cpp blas_reference.cpp)
  target_link_libraries(dslash_test ${TEST_LIBS} )
  target_link_libraries(dslash_ctest ${TEST_LIBS} )
  QUDA_CHECKBUILDTEST(dslash_test QUDA_BUILD_ALL_TESTS)
  QUDA_CHECKBUILDTEST(dslash_ctest BUILD_TESTING)

  cuda_add_executable(invert_test invert_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp clover_reference.cpp blas_reference.cpp)
  target_link_libraries(invert_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(invert_test QUDA_BUILD_ALL_TESTS)

  if(QUDA_BLOCKSOLVER)
    cuda_add_executable(invertmsrc_test invertmsrc_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp)
    target_link_libraries(invertmsrc_test ${TEST_LIBS})
    QUDA_CHECKBUILDTEST(invertmsrc_test QUDA_BUILD_ALL_TESTS)
  endif()
endif()

if(QUDA_DIRAC_WILSON OR QUDA_DIRAC_CLOVER OR QUDA_DIRAC_TWISTED_MASS OR QUDA_DIRAC_TWISTED_CLOVER OR QUDA_DIRAC_DOMAIN_WALL OR QUDA_DIRAC_STAGGERED)
  cuda_add_executable(deflated_invert_test deflated_invert_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp)
  target_link_libraries(deflated_invert_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(deflated_invert_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_DIRAC_STAGGERED)
  cuda_add_executable(staggered_dslash_test staggered_dslash_test.cpp staggered_dslash_reference.cpp blas_reference.cpp llfat_reference.cpp)
  target_link_libraries(staggered_dslash_test ${TEST_LIBS})
  cuda_add_executable(staggered_dslash_ctest staggered_dslash_ctest.cpp staggered_dslash_reference.cpp blas_reference.cpp llfat_reference.cpp)
  target_link_libraries(staggered_dslash_ctest ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(staggered_dslash_test QUDA_BUILD_ALL_TESTS)
  QUDA_CHECKBUILDTEST(staggered_dslash_ctest BUILD_TESTING)

  cuda_add_executable(staggered_invert_test staggered_invert_test.cpp staggered_dslash_reference.cpp blas_reference.cpp llfat_reference.cpp)
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

  if(${QUDA_GAUGE_ALG})
    cuda_add_executable(multigrid_evolve_test multigrid_evolve_test.cpp wilson_dslash_reference.cpp clover_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp)
    target_link_libraries(multigrid_evolve_test ${TEST_LIBS})
    QUDA_CHECKBUILDTEST(multigrid_evolve_test QUDA_BUILD_ALL_TESTS)
  endif()

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

cuda_add_executable(covdev_test covdev_test.cpp  covdev_reference.cpp)
target_link_libraries(covdev_test ${TEST_LIBS})
QUDA_CHECKBUILDTEST(covdev_test QUDA_BUILD_ALL_TESTS)

if(QUDA_LINK_ASQTAD OR QUDA_LINK_HISQ)
  cuda_add_executable(llfat_test llfat_test.cpp llfat_reference.cpp)
  target_link_libraries(llfat_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(llfat_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_LINK_HISQ)
  cuda_add_executable(unitarize_link_test unitarize_link_test.cpp)
  target_link_libraries(unitarize_link_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(unitarize_link_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(hisq_stencil_test hisq_stencil_test.cpp llfat_reference.cpp)
  target_link_libraries(hisq_stencil_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(hisq_stencil_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_FORCE_GAUGE)
  cuda_add_executable(gauge_force_test gauge_force_test.cpp gauge_force_reference.cpp)
  target_link_libraries(gauge_force_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(gauge_force_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_GAUGE_ALG)
  cuda_add_executable(gauge_alg_test gauge_alg_test.cpp)
  target_link_libraries(gauge_alg_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(gauge_alg_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(heatbath_test heatbath_test.cpp)
  target_link_libraries(heatbath_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(heatbath_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_FORCE_HISQ)
  cuda_add_executable(hisq_paths_force_test hisq_paths_force_test.cpp hisq_force_reference.cpp hisq_force_reference2.cpp )
  target_link_libraries(hisq_paths_force_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(hisq_paths_force_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(hisq_unitarize_force_test hisq_unitarize_force_test.cpp hisq_force_reference.cpp )
  target_link_libraries(hisq_unitarize_force_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(hisq_unitarize_force_test QUDA_BUILD_ALL_TESTS)
endif()



#  use FindMPI variables for QUDA_CTEST_LAUNCH
# set MPIEXEC_MAX_NUMPROCS to the number of ranks you want to launch
set(QUDA_CTEST_LAUNCH ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS})

## BLAS test

if(QUDA_DIRAC_WILSON OR QUDA_DIRAC_CLOVER OR QUDA_DIRAC_TWISTED_MASS OR QUDA_DIRAC_TWISTED_CLOVER OR QUDA_DIRAC_DOMAIN_WALL)
  add_test(NAME blas_test_parity_wilson COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_test> ${MPIEXEC_POSTFLAGS} --dim 2 4 6 8 --solve-type direct-pc --gtest_output=xml:blas_test_parity.xml)
  add_test(NAME blas_test_full_wilson COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_test> ${MPIEXEC_POSTFLAGS} --dim 2 4 6 8 --solve-type direct --gtest_output=xml:blas_test_full.xml)
endif()

if(QUDA_DIRAC_STAGGERED)
  add_test(NAME blas_test_parity_staggered COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_test> ${MPIEXEC_POSTFLAGS} --dim 2 4 6 8 --dslash-type staggered --solve-type direct-pc --gtest_output=xml:blas_test_parity.xml)
  add_test(NAME blas_test_full_staggered COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_test> ${MPIEXEC_POSTFLAGS} --dim 2 4 6 8 --dslash-type staggered --solve-type direct --gtest_output=xml:blas_test_full.xml)
endif()

# loop over Dslash policies
if(QUDA_CTEST_SEP_DSLASH_POLICIES)
  set(DSLASH_POLICIES 0 1 6 7 8 9 -1)
  if(DEFINED ENV{QUDA_ENABLE_GDR})
    if($ENV{QUDA_ENABLE_GDR} EQUAL 1)
      set(DSLASH_POLICIES 0 1 2 3 4 5 6 7 8 9 10 11 -1)
      message(STATUS "QUDA_ENABLE_GDR=1: enabling GDR-enabled dslash policies in ctest")
    else()
      message(STATUS "QUDA_ENABLE_GDR!=1: disabling GDR-enabled dslash policies in ctest")
    endif()
  else()
    message(STATUS "QUDA_ENABLE_GDR not set: disabling GDR-enabled dslash policies in ctest")
  endif()
else()
  set(DSLASH_POLICIES -1)
endif()


foreach(pol IN LISTS DSLASH_POLICIES)

  if(${pol} LESS 0)
    set(pol2 "tune")
    set(polenv OFF)
  else()
    set(pol2 ${pol})
    set(polenv ON)
  endif()

  if(QUDA_DIRAC_WILSON)
    add_test(NAME dslash_wilson-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type wilson --test 3 --dim 2 4 6 8 --gtest_output=xml:dslash_wilson_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_wilson-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_CLOVER)
    # symmetric preconditioning
    add_test(NAME dslash_clover-sym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type clover --test 3 --matpc even-even --dim 2 4 6 8 --gtest_output=xml:dslash_clover_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_clover-sym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_clover-asym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type clover --test 3 --matpc even-even-asym --dim 2 4 6 8 --gtest_output=xml:dslash_clover_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_clover-asym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_TWISTED_MASS)
    add_test(NAME dslash_twisted-mass-sym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type twisted-mass --test 3 --matpc even-even --dim 2 4 6 8 --gtest_output=xml:dslash_twisted-mass_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_twisted-mass-sym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_twisted-mass-asym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type twisted-mass --test 3 --matpc even-even-asym --dim 2 4 6 8 --gtest_output=xml:dslash_twisted-mass_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_twisted-mass-asym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_NDEG_TWISTED_MASS)
    # symmetric preconditioning
    add_test(NAME dslash_ndeg-twisted-mass-sym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type twisted-mass --test 3 --matpc even-even --flavor nondeg-doublet --dim 2 4 6 8 --gtest_output=xml:dslash_ndeg-twisted-mass_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_ndeg-twisted-mass-sym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_ndeg-twisted-mass-asym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type twisted-mass --test 3 --matpc even-even-asym --flavor nondeg-doublet --dim 2 4 6 8 --gtest_output=xml:dslash_ndeg-twisted-mass_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_ndeg-twisted-mass-asym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_TWISTED_CLOVER)
    # symmetric preconditioning
    add_test(NAME dslash_twisted-clover-sym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type twisted-clover --test 3 --matpc even-even --dim 2 4 6 8 --gtest_output=xml:dslash_twisted-clover_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_twisted-clover-sym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_twisted-clover-asym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type twisted-clover --test 3 --matpc even-even-asym --dim 2 4 6 8 --gtest_output=xml:dslash_twisted-clover_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_twisted-clover-asym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_DOMAIN_WALL)
    add_test(NAME dslash_domain-wall-sym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH}  $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type domain-wall --test 3 --matpc even-even --dim 2 4 6 8 --Lsdim 4 --gtest_output=xml:dslash_domain-wall_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_domain-wall-sym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # symmetric 4-d preconditioning
    add_test(NAME dslash_domain-wall-4d-sym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH}  $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type domain-wall-4d --test 4 --matpc even-even --dim 2 4 6 8 --Lsdim 4 --gtest_output=xml:dslash_domain-wall-4d_test_pol${pol2}.xml)
    add_test(NAME dslash_mobius-sym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH}  $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type mobius --test 5 --matpc even-even --dim 2 4 6 8 --Lsdim 4 --gtest_output=xml:dslash_mobius_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_domain-wall-4d-sym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
      set_tests_properties(dslash_mobius-sym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric 4-d preconditioning
    add_test(NAME dslash_domain-wall-4d-asym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH}  $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type domain-wall-4d --test 4 --matpc even-even-asym  --dim 2 4 6 8 --Lsdim 4 --gtest_output=xml:dslash_domain-wall-4d_test_pol${pol2}.xml)
    add_test(NAME dslash_mobius-asym-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH}  $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type mobius --test 5 --matpc even-even-asym  --dim 2 4 6 8 --Lsdim 4 --gtest_output=xml:dslash_mobius_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_domain-wall-4d-asym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
      set_tests_properties(dslash_mobius-asym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_STAGGERED)
    add_test(NAME dslash_improved_staggered-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type asqtad --test 1 --dim 6 8 10 12 --gtest_output=xml:dslash_improved_staggered_test_pol${pol2}.xml)
    add_test(NAME dslash_naive_staggered-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH}  $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type staggered --test 1 --dim 2 4 6 8 --gtest_output=xml:dslash_naive_staggered_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_improved_staggered-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol2})
      set_tests_properties(dslash_naive_staggered-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol2})
    endif()
    if(QUDA_LINK_HISQ)
      add_test(NAME dslash_improved_staggered_build-policy${pol2} COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS} --dslash-type asqtad --test 1 --dim 6 8 10 12 --compute-fat-long true --epsilon-naik -0.01 --tadpole-coeff 0.9 --gtest_output=xml:dslash_improved_staggered_build_test_pol${pol2}.xml)
      if (polenv)
        set_tests_properties(dslash_improved_staggered_build-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol2})
      endif()
    endif()
  endif()

endforeach(pol)

