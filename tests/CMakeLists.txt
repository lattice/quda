include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_FULL_LIBDIR})

# enable tests build a common library for all test utilities
add_library(quda_test STATIC googletest/src/gtest-all.cc)
target_include_directories(quda_test PUBLIC googletest/include googletest)

if(QUDA_QIO
   AND QUDA_DOWNLOAD_USQCD
   AND NOT QIO_FOUND)
  add_dependencies(quda_test QIO)
endif()

if(QUDA_QMP
   AND QUDA_DOWNLOAD_USQCD
   AND NOT QMP_FOUND)
  add_dependencies(quda_test QMP)
endif()

set(TEST_LIBS quda quda_test quda_reference quda_utils)

macro(QUDA_CHECKBUILDTEST mytarget qudabuildtests)
  # adding the linker language here as a workaround -- was not needed for cmake 3.16
  set_target_properties(${mytarget} PROPERTIES LINKER_LANGUAGE CUDA)
  if(NOT ${qudabuildtests})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
    set(QUDA_EXCLUDE_FROM_INSTALL "EXCLUDE_FROM_ALL")
  endif()

  if(QUDA_QIO
     AND QUDA_DOWNLOAD_USQCD
     AND NOT QIO_FOUND)
    add_dependencies(${mytarget} QIO)
  endif()

  if(QUDA_QMP
     AND QUDA_DOWNLOAD_USQCD
     AND NOT QMP_FOUND)
    add_dependencies(${mytarget} QMP)
  endif()
endmacro()

if(NOT ${QUDA_INSTALL_ALL_TESTS})
  set(QUDA_EXCLUDE_FROM_INSTALL "EXCLUDE_FROM_ALL")
endif()

if(QUDA_ARPACK)
  list(APPEND TEST_LIBS ${ARPACK})
  if(QUDA_MPI OR QUDA_QMP)
    list(APPEND TEST_LIBS ${PARPACK})
  endif()
endif()

# define tests

# if we build with QDP JIT the tests cannot run anyway
if(QUDA_QDPJIT)
  set(QUDA_BUILD_ALL_TESTS OFF)
endif()

if(QUDA_DIRAC_WILSON
   OR QUDA_DIRAC_CLOVER
   OR QUDA_DIRAC_TWISTED_MASS
   OR QUDA_DIRAC_TWISTED_CLOVER
   OR QUDA_DIRAC_NDEG_TWISTED_MASS
   OR QUDA_DIRAC_DOMAIN_WALL)
  add_executable(dslash_test dslash_test.cpp)
  target_link_libraries(dslash_test ${TEST_LIBS})
  quda_checkbuildtest(dslash_test QUDA_BUILD_ALL_TESTS)

  add_executable(dslash_ctest dslash_ctest.cpp)
  target_link_libraries(dslash_ctest ${TEST_LIBS})
  quda_checkbuildtest(dslash_ctest QUDA_BUILD_ALL_TESTS)
  install(TARGETS dslash_test dslash_ctest ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(invert_test invert_test.cpp)
  target_link_libraries(invert_test ${TEST_LIBS})
  quda_checkbuildtest(invert_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS invert_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(eigensolve_test eigensolve_test.cpp)
  target_link_libraries(eigensolve_test ${TEST_LIBS})
  quda_checkbuildtest(eigensolve_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS eigensolve_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  if(QUDA_BLOCKSOLVER)
    add_executable(invertmsrc_test invertmsrc_test.cpp)
    target_link_libraries(invertmsrc_test ${TEST_LIBS})
    quda_checkbuildtest(invertmsrc_test QUDA_BUILD_ALL_TESTS)
    install(TARGETS invertmsrc_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

if(QUDA_DIRAC_WILSON
   OR QUDA_DIRAC_CLOVER
   OR QUDA_DIRAC_TWISTED_MASS
   OR QUDA_DIRAC_TWISTED_CLOVER
   OR QUDA_DIRAC_DOMAIN_WALL
   OR QUDA_DIRAC_STAGGERED)
  add_executable(deflated_invert_test deflated_invert_test.cpp)
  target_link_libraries(deflated_invert_test ${TEST_LIBS})
  quda_checkbuildtest(deflated_invert_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS deflated_invert_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(QUDA_DIRAC_STAGGERED)
  add_executable(staggered_dslash_test staggered_dslash_test.cpp)
  target_link_libraries(staggered_dslash_test ${TEST_LIBS})
  quda_checkbuildtest(staggered_dslash_test QUDA_BUILD_ALL_TESTS)

  add_executable(staggered_dslash_ctest staggered_dslash_ctest.cpp)
  target_link_libraries(staggered_dslash_ctest ${TEST_LIBS})
  quda_checkbuildtest(staggered_dslash_ctest QUDA_BUILD_ALL_TESTS)
  install(TARGETS staggered_dslash_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(staggered_invert_test staggered_invert_test.cpp)
  target_link_libraries(staggered_invert_test ${TEST_LIBS})
  quda_checkbuildtest(staggered_invert_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS staggered_invert_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(staggered_eigensolve_test staggered_eigensolve_test.cpp)
  target_link_libraries(staggered_eigensolve_test ${TEST_LIBS})
  quda_checkbuildtest(staggered_eigensolve_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS staggered_eigensolve_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  if(QUDA_BLOCKSOLVER)
    add_executable(staggered_invertmsrc_test staggered_invertmsrc_test.cpp)
    target_link_libraries(staggered_invertmsrc_test ${TEST_LIBS})
    quda_checkbuildtest(staggered_invertmsrc_test QUDA_BUILD_ALL_TESTS)
    install(TARGETS staggered_invertmsrc_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()

endif()

if(QUDA_DIRAC_WILSON
   OR QUDA_DIRAC_CLOVER
   OR QUDA_DIRAC_TWISTED_MASS
   OR QUDA_DIRAC_TWISTED_CLOVER
   OR QUDA_DIRAC_NDEG_TWISTED_MASS
   OR QUDA_DIRAC_DOMAIN_WALL
   OR QUDA_DIRAC_STAGGERED)
  add_executable(blas_test blas_test.cu)
  target_link_libraries(blas_test ${TEST_LIBS})
  quda_checkbuildtest(blas_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS blas_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(QUDA_MULTIGRID)
  add_executable(multigrid_benchmark_test multigrid_benchmark_test.cu)
  target_link_libraries(multigrid_benchmark_test ${TEST_LIBS})
  quda_checkbuildtest(multigrid_benchmark_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS multigrid_benchmark_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  if(${QUDA_GAUGE_ALG})
    add_executable(multigrid_evolve_test multigrid_evolve_test.cpp)
    target_link_libraries(multigrid_evolve_test ${TEST_LIBS})
    quda_checkbuildtest(multigrid_evolve_test QUDA_BUILD_ALL_TESTS)
    install(TARGETS multigrid_evolve_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()

endif()

add_executable(plaq_test plaq_test.cpp)
target_link_libraries(plaq_test ${TEST_LIBS})
quda_checkbuildtest(plaq_test QUDA_BUILD_ALL_TESTS)
install(TARGETS plaq_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(su3_test su3_test.cpp)
target_link_libraries(su3_test ${TEST_LIBS})
quda_checkbuildtest(su3_test QUDA_BUILD_ALL_TESTS)
install(TARGETS su3_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(pack_test pack_test.cpp)
target_link_libraries(pack_test ${TEST_LIBS})
quda_checkbuildtest(pack_test QUDA_BUILD_ALL_TESTS)
install(TARGETS pack_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

if(QUDA_COVDEV)
  add_executable(covdev_test covdev_test.cpp)
  target_link_libraries(covdev_test ${TEST_LIBS})
  quda_checkbuildtest(covdev_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS covdev_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(QUDA_CONTRACT)
  add_executable(contract_test contract_test.cpp)
  target_link_libraries(contract_test ${TEST_LIBS})
  quda_checkbuildtest(contract_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS contract_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(QUDA_DIRAC_STAGGERED)
  add_executable(llfat_test llfat_test.cpp)
  target_link_libraries(llfat_test ${TEST_LIBS})
  quda_checkbuildtest(llfat_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS llfat_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(unitarize_link_test unitarize_link_test.cpp)
  target_link_libraries(unitarize_link_test ${TEST_LIBS})
  quda_checkbuildtest(unitarize_link_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS unitarize_link_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(hisq_stencil_test hisq_stencil_test.cpp)
  target_link_libraries(hisq_stencil_test ${TEST_LIBS})
  quda_checkbuildtest(hisq_stencil_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS hisq_stencil_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(QUDA_FORCE_GAUGE)
  add_executable(gauge_force_test gauge_force_test.cpp)
  target_link_libraries(gauge_force_test ${TEST_LIBS})
  quda_checkbuildtest(gauge_force_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS gauge_force_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(QUDA_GAUGE_ALG)
  add_executable(gauge_alg_test gauge_alg_test.cpp)
  target_link_libraries(gauge_alg_test ${TEST_LIBS})
  quda_checkbuildtest(gauge_alg_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS gauge_alg_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(heatbath_test heatbath_test.cpp)
  target_link_libraries(heatbath_test ${TEST_LIBS})
  quda_checkbuildtest(heatbath_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS heatbath_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(QUDA_FORCE_HISQ)
  add_executable(hisq_paths_force_test hisq_paths_force_test.cpp)
  target_link_libraries(hisq_paths_force_test ${TEST_LIBS})
  quda_checkbuildtest(hisq_paths_force_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS hisq_paths_force_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(hisq_unitarize_force_test hisq_unitarize_force_test.cpp)
  target_link_libraries(hisq_unitarize_force_test ${TEST_LIBS})
  quda_checkbuildtest(hisq_unitarize_force_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS hisq_unitarize_force_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
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

if(QUDA_DIRAC_STAGGERED)
  add_test(NAME blas_test_parity_staggered
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_test> ${MPIEXEC_POSTFLAGS}
                   --dim 2 4 6 8
                   --dslash-type staggered
                   --solve-type direct-pc
                   --gtest_output=xml:blas_test_parity.xml)
  add_test(NAME blas_test_full_staggered
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_test> ${MPIEXEC_POSTFLAGS}
                   --dim 2 4 6 8
                   --dslash-type staggered
                   --solve-type direct
                   --gtest_output=xml:blas_test_full.xml)
endif()

# loop over Dslash policies
if(QUDA_CTEST_SEP_DSLASH_POLICIES)
  set(DSLASH_POLICIES 0 1 6 7 8 9 12 13 -1)
  if(DEFINED ENV{QUDA_ENABLE_GDR})
    if($ENV{QUDA_ENABLE_GDR} EQUAL 1)
      set(DSLASH_POLICIES 0 1 2 3 4 5 6 7 8 9 10 11 12 13 -1)
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
    add_test(NAME dslash_wilson-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type wilson
                     --test MatPCDagMatPC
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_wilson_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_wilson-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_CLOVER)
    # symmetric preconditioning
    add_test(NAME dslash_clover-sym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type clover
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_clover_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_clover-sym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_clover-asym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type clover
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_clover_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_clover-asym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

 if(QUDA_DIRAC_CLOVER_HASENBUSCH_TWIST)
    # symmetric preconditioning
    add_test(NAME dslash_clover_hasenbusch_twist-sym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type clover-hasenbusch-twist
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_clover_test_sym_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_clover_hasenbusch_twist-sym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_clover_hasenbusch_twist-asym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type clover-hasenbusch-twist
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_clover_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_clover_hasenbusch_twist-asym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_TWISTED_MASS)
    add_test(NAME dslash_twisted-mass-sym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type twisted-mass
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_twisted-mass_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_twisted-mass-sym-policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_twisted-mass-asym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type twisted-mass
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_twisted-mass_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_twisted-mass-asym-policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_NDEG_TWISTED_MASS)
    # symmetric preconditioning
    add_test(NAME dslash_ndeg-twisted-mass-sym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type twisted-mass
                     --test MatPCDagMatPC
                     --matpc even-even
                     --flavor nondeg-doublet
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_ndeg-twisted-mass_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_ndeg-twisted-mass-sym-policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_ndeg-twisted-mass-asym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type twisted-mass
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --flavor nondeg-doublet
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_ndeg-twisted-mass_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_ndeg-twisted-mass-asym-policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_TWISTED_CLOVER)
    # symmetric preconditioning
    add_test(NAME dslash_twisted-clover-sym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type twisted-clover
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_twisted-clover_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_twisted-clover-sym-policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_twisted-clover-asym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type twisted-clover
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_twisted-clover_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_twisted-clover-asym-policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_DOMAIN_WALL)
    add_test(NAME dslash_domain-wall-sym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type domain-wall
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --Lsdim 4
                     --gtest_output=xml:dslash_domain-wall_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_domain-wall-sym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # symmetric 4-d preconditioning
    add_test(NAME dslash_domain-wall-4d-sym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type domain-wall-4d
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --Lsdim 4
                     --gtest_output=xml:dslash_domain-wall-4d_test_pol${pol2}.xml)
    add_test(NAME dslash_mobius-sym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type mobius
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --Lsdim 4
                     --gtest_output=xml:dslash_mobius_test_pol${pol2}.xml)
    add_test(NAME dslash_mobius_eofa-sym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type mobius-eofa
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --Lsdim 4
                     --gtest_output=xml:dslash_mobius_eofa_test_pol${pol2}.xml)

    if(polenv)
      set_tests_properties(dslash_domain-wall-4d-sym-policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
      set_tests_properties(dslash_mobius-sym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric 4-d preconditioning
    add_test(NAME dslash_domain-wall-4d-asym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type domain-wall-4d
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --Lsdim 4
                     --gtest_output=xml:dslash_domain-wall-4d_test_pol${pol2}.xml)
    add_test(NAME dslash_mobius-asym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type mobius
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --Lsdim 4
                     --gtest_output=xml:dslash_mobius_test_pol${pol2}.xml)
    add_test(NAME dslash_mobius_eofa-asym-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type mobius-eofa
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --Lsdim 4
                     --gtest_output=xml:dslash_mobius_eofa_test_pol${pol2}.xml)

    # MdagM local operator
    add_test(NAME dslash_mobius_local-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type mobius
                     --test MatPCDagMatPCLocal
                     --matpc even-even
                     --dim 2 4 6 8
                     --Lsdim 4
                     --gtest_output=xml:dslash_mobius_local{pol2}.xml)

     if(polenv)
      set_tests_properties(dslash_domain-wall-4d-asym-policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
      set_tests_properties(dslash_mobius-asym-policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_STAGGERED)
    add_test(NAME dslash_improved_staggered-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type asqtad
                     --test MatPC
                     --dim 6 8 10 12
                     --gtest_output=xml:dslash_improved_staggered_test_pol${pol2}.xml)
    add_test(NAME dslash_naive_staggered-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type staggered
                     --test MatPC
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_naive_staggered_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_improved_staggered-policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol2})
      set_tests_properties(dslash_naive_staggered-policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol2})
    endif()

    add_test(NAME dslash_improved_staggered_build-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type asqtad
                     --test MatPC
                     --dim 6 8 10 12
                     --compute-fat-long true
                     --epsilon-naik -0.01
                     --tadpole-coeff 0.9
                     --gtest_output=xml:dslash_improved_staggered_build_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_improved_staggered_build-policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol2})
    endif()
  endif()

  if(QUDA_COVDEV)
    add_test(NAME covdev_-policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:covdev_test> ${MPIEXEEC_POSTFLAGS}
                     --dim 6 8 10 12
                     --gtest_output=xml:covdev_test_pol${pol2}.xml)
  endif()

endforeach(pol)

if(QUDA_FORCE_GAUGE)
  add_test(NAME gauge_force
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:gauge_force_test> ${MPIEXEEC_POSTFLAGS}
                   --dim 2 4 6 8
                   --gtest_output=xml:gauge_force_test.xml)
endif()
