# enable tests build a common library for all test utilities
add_library(quda_test googletest/src/gtest-all.cc)
target_include_directories(quda_test SYSTEM PUBLIC  googletest/include googletest)
target_link_libraries(quda_test PUBLIC Eigen)
target_link_libraries(quda_test PUBLIC QUDA::quda)
set_target_properties(quda_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(
  quda_test PUBLIC
  $<IF:$<CONFIG:RELEASE>,-w,-Wall -Wextra>
  $<$<CONFIG:STRICT>:-Werror>
)
if(BUILD_SHARED_LIBS)
  install(TARGETS quda_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

add_subdirectory(utils)
add_subdirectory(host_reference)
enable_language(Fortran)

if(QUDA_NVSHMEM AND QUDA_DOWNLOAD_NVSHMEM)
  add_dependencies(quda_test NVSHMEM)
endif()

set(TEST_LIBS quda_test)

# custom target for building all tests (useful if QUDA_BUILD_ALL_TESTS is OFF)
add_custom_target(all_tests)

macro(QUDA_CHECKBUILDTEST mytarget qudabuildtests)
  if(NOT ${qudabuildtests})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
    set(QUDA_EXCLUDE_FROM_INSTALL "EXCLUDE_FROM_ALL")
  endif()
  add_dependencies(all_tests ${mytarget})
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
add_executable(c_interface_test c_interface_test.c)
target_link_libraries(c_interface_test ${TEST_LIBS} "-lstdc++")
quda_checkbuildtest(c_interface_test QUDA_BUILD_ALL_TESTS)

# if we build with QDP JIT the tests cannot run anyway
if(QUDA_QDPJIT)
  set(QUDA_BUILD_ALL_TESTS OFF)
endif()

if(QUDA_DIRAC_WILSON
   OR QUDA_DIRAC_CLOVER
   OR QUDA_DIRAC_TWISTED_MASS
   OR QUDA_DIRAC_TWISTED_CLOVER
   OR QUDA_DIRAC_NDEG_TWISTED_MASS
   OR QUDA_DIRAC_NDEG_TWISTED_CLOVER
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

endif()

if(QUDA_DIRAC_WILSON
   OR QUDA_DIRAC_CLOVER
   OR QUDA_DIRAC_TWISTED_MASS
   OR QUDA_DIRAC_TWISTED_CLOVER
   OR QUDA_DIRAC_NDEG_TWISTED_MASS
   OR QUDA_DIRAC_NDEG_TWISTED_CLOVER
   OR QUDA_DIRAC_DOMAIN_WALL
   OR QUDA_DIRAC_STAGGERED)
  add_executable(blas_test blas_test.cpp)
  target_link_libraries(blas_test ${TEST_LIBS})

  quda_checkbuildtest(blas_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS blas_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(dilution_test dilution_test.cpp)
  target_link_libraries(dilution_test ${TEST_LIBS})
  quda_checkbuildtest(dilution_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS dilution_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(QUDA_MULTIGRID)
  add_executable(multigrid_benchmark_test multigrid_benchmark_test.cpp)
  target_link_libraries(multigrid_benchmark_test ${TEST_LIBS})

  quda_checkbuildtest(multigrid_benchmark_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS multigrid_benchmark_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(multigrid_evolve_test multigrid_evolve_test.cpp)
  target_link_libraries(multigrid_evolve_test ${TEST_LIBS})
  quda_checkbuildtest(multigrid_evolve_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS multigrid_evolve_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

endif()

if(QUDA_BUILD_NATIVE_LAPACK)
  add_executable(blas_interface_test blas_interface_test.cpp)
  target_link_libraries(blas_interface_test ${TEST_LIBS})
  quda_checkbuildtest(blas_interface_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS blas_interface_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(QUDA_QIO)
  add_executable(io_test io_test.cpp)
  target_link_libraries(io_test ${TEST_LIBS})
  quda_checkbuildtest(io_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS io_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

add_executable(tune_test tune_test.cpp)
target_link_libraries(tune_test ${TEST_LIBS})
quda_checkbuildtest(tune_test QUDA_BUILD_ALL_TESTS)
install(TARGETS tune_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

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

if(QUDA_SMEAR_GAUSS_TWOLINK)
  add_executable(staggered_gsmear_test staggered_gsmear_test.cpp)
  target_link_libraries(staggered_gsmear_test ${TEST_LIBS})
  quda_checkbuildtest(staggered_gsmear_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS staggered_gsmear_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR}) 	
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

  add_executable(hisq_paths_force_test hisq_paths_force_test.cpp)
  target_link_libraries(hisq_paths_force_test ${TEST_LIBS})
  quda_checkbuildtest(hisq_paths_force_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS hisq_paths_force_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(hisq_unitarize_force_test hisq_unitarize_force_test.cpp)
  target_link_libraries(hisq_unitarize_force_test ${TEST_LIBS})
  quda_checkbuildtest(hisq_unitarize_force_test QUDA_BUILD_ALL_TESTS)
  install(TARGETS hisq_unitarize_force_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

add_executable(gauge_path_test gauge_path_test.cpp)
target_link_libraries(gauge_path_test ${TEST_LIBS})
quda_checkbuildtest(gauge_path_test QUDA_BUILD_ALL_TESTS)
install(TARGETS gauge_path_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(gauge_alg_test gauge_alg_test.cpp)
target_link_libraries(gauge_alg_test ${TEST_LIBS})
quda_checkbuildtest(gauge_alg_test QUDA_BUILD_ALL_TESTS)
install(TARGETS gauge_alg_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(heatbath_test heatbath_test.cpp)
target_link_libraries(heatbath_test ${TEST_LIBS})
quda_checkbuildtest(heatbath_test QUDA_BUILD_ALL_TESTS)
install(TARGETS heatbath_test ${QUDA_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

if(QUDA_MPI OR QUDA_QMP)
  if(DEFINED ENV{QUDA_TEST_NUMPROCS})
    # user is setting number of processes to use through the QUDA_TEST_NUMPROCS env
    set(QUDA_CTEST_LAUNCH ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
                          $ENV{QUDA_TEST_NUMPROCS} ${MPIEXEC_PREFLAGS})
  else()
    # use FindMPI variables for QUDA_CTEST_LAUNCH set MPIEXEC_MAX_NUMPROCS to the
    # number of ranks you want to launch
    set(QUDA_CTEST_LAUNCH ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
                          ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS})
  endif()
endif()

# BLAS tests
if(QUDA_DIRAC_WILSON
   OR QUDA_DIRAC_CLOVER
   OR QUDA_DIRAC_TWISTED_MASS
   OR QUDA_DIRAC_NDEG_TWISTED_MASS
   OR QUDA_DIRAC_NDEG_TWISTED_CLOVER
   OR QUDA_DIRAC_TWISTED_CLOVER
   OR QUDA_DIRAC_DOMAIN_WALL)
  add_test(NAME blas_test_parity_wilson
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_test> ${MPIEXEC_POSTFLAGS}
                   --dim 2 4 6 8 
                   --nsrc 8 --msrc 9
                   --solve-type direct-pc
                   --gtest_output=xml:blas_test_parity.xml)
  add_test(NAME blas_test_full_wilson
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_test> ${MPIEXEC_POSTFLAGS}
                   --dim 2 4 6 8
                   --nsrc 8 --msrc 9
                   --solve-type direct
                   --gtest_output=xml:blas_test_full.xml)
endif()

if(QUDA_DIRAC_STAGGERED)
  add_test(NAME blas_test_parity_staggered
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_test> ${MPIEXEC_POSTFLAGS}
                   --dim 2 4 6 8
                   --nsrc 8 --msrc 9
                   --dslash-type staggered
                   --solve-type direct-pc
                   --gtest_output=xml:blas_test_parity.xml)
  add_test(NAME blas_test_full_staggered
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_test> ${MPIEXEC_POSTFLAGS}
                   --dim 2 4 6 8
                   --nsrc 8 --msrc 9
                   --dslash-type staggered
                   --solve-type direct
                   --gtest_output=xml:blas_test_full.xml)
endif()

#BLAS interface test
if(QUDA_BUILD_NATIVE_LAPACK)
  add_test(NAME blas_interface_test
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:blas_interface_test> ${MPIEXEC_POSTFLAGS}
    --blas-gemm-mnk 64 64 64
    --blas-gemm-leading-dims 128 128 128
    --blas-gemm-offsets 16 16 16
    --blas-gemm-alpha 1.0 2.0
    --blas-gemm-beta -3.0 1.5
    --blas-gemm-trans-a T
    --blas-gemm-trans-b C
    --blas-lu-inv-mat-size 96
    --blas-data-type Z
    --blas-data-order row
    --blas-batch 20
    --enable-testing true
    --gtest_output=xml:blas_interface_test.xml)
endif()

#Contraction test
if(QUDA_CONTRACT)
  add_test(NAME contract_test
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:contract_test> ${MPIEXEC_POSTFLAGS}
    --dim 2 4 6 8
    --gtest_output=xml:contract_test.xml)
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
    set(DIRAC_NAME wilson)
    add_test(NAME dslash_${DIRAC_NAME}_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --all-partitions 1
                     --test MatPCDagMatPC
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_${DIRAC_NAME}_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME benchmark_dslash_${DIRAC_NAME}_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --all-partitions 1
                     --test 0
                     --dim 20 20 20 20
                     --gtest_output=json:dslash_${DIRAC_NAME}_benchmark_pol${pol2}.json
                     --gtest_filter=*benchmark/*n0)
    set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES DISABLED ${QUDA_CTEST_DISABLE_BENCHMARKS})
    if(polenv)
      set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
endif()

  if(QUDA_DIRAC_CLOVER)
    set(DIRAC_NAME clover)

    # symmetric preconditioning
    add_test(NAME dslash_${DIRAC_NAME}_sym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_${DIRAC_NAME}_sym_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_sym_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    set(DIRAC_NAME clover)
    add_test(NAME dslash_${DIRAC_NAME}_asym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_${DIRAC_NAME}_asym_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_asym_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME benchmark_dslash_${DIRAC_NAME}_policy${pol2}
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
            --dslash-type ${DIRAC_NAME}
            --test 0
            --dim 20 20 20 20
            --gtest_output=json:dslash_${DIRAC_NAME}_benchmark_pol${pol2}.json
            --gtest_filter=*benchmark/*n0)
    set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES DISABLED ${QUDA_CTEST_DISABLE_BENCHMARKS})
    if(polenv)
      set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

  endif()

  if(QUDA_DIRAC_CLOVER_HASENBUSCH)
    set(DIRAC_NAME clover-hasenbusch-twist)

    # symmetric preconditioning
    add_test(NAME dslash_${DIRAC_NAME}_sym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_${DIRAC_NAME}_sym_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_sym_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_${DIRAC_NAME}_asym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_${DIRAC_NAME}_asym_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_asym_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME benchmark_dslash_${DIRAC_NAME}_policy${pol2}
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
            --dslash-type ${DIRAC_NAME}
            --test 0
            --dim 20 20 20 20
            --gtest_output=json:dslash_${DIRAC_NAME}_benchmark_pol${pol2}.json
            --gtest_filter=*benchmark/*n0)
    set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES DISABLED ${QUDA_CTEST_DISABLE_BENCHMARKS})
    if(polenv)
      set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

  endif()

  if(QUDA_DIRAC_TWISTED_MASS)
    set(DIRAC_NAME twisted-mass)

    # symmetric preconditioning
    add_test(NAME dslash_${DIRAC_NAME}_sym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_${DIRAC_NAME}_sym_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_sym_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_${DIRAC_NAME}_asym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_${DIRAC_NAME}_asym_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_asym_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME benchmark_dslash_${DIRAC_NAME}_policy${pol2}
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
            --dslash-type ${DIRAC_NAME}
            --test 0
            --dim 20 20 20 20
            --gtest_output=json:dslash_${DIRAC_NAME}_benchmark_pol${pol2}.json
            --gtest_filter=*benchmark/*n0)
    set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES DISABLED ${QUDA_CTEST_DISABLE_BENCHMARKS})
    if(polenv)
      set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

  endif()

  if(QUDA_DIRAC_NDEG_TWISTED_MASS)
    # the ndeg-twisted-mas operator does not fit the pattern used for most other operators
    # since it is defined via 'dslash-type = twisted-mass` and 'flavor = nondeg-doublet'
    set(DIRAC_NAME twisted-mass)

    # symmetric preconditioning
    add_test(NAME dslash_ndeg_${DIRAC_NAME}_sym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even
                     --flavor nondeg-doublet
                     --dim 2 4 6 8
        --gtest_output=xml:dslash_ndeg_${DIRAC_NAME}_sym_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_ndeg_${DIRAC_NAME}_sym_policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_ndeg_${DIRAC_NAME}_asym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --flavor nondeg-doublet
                     --dim 2 4 6 8
        --gtest_output=xml:dslash_ndeg_${DIRAC_NAME}_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_ndeg_${DIRAC_NAME}_asym_policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME benchmark_dslash_ndeg_${DIRAC_NAME}_policy${pol2}
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
            --dslash-type ${DIRAC_NAME}
            --test 0
            --dim 20 20 20 20
            --gtest_output=json:dslash_${DIRAC_NAME}_benchmark_pol${pol2}.json
            --gtest_filter=*benchmark/*n0)
    set_tests_properties(benchmark_dslash_ndeg_${DIRAC_NAME}_policy${pol2} PROPERTIES DISABLED ${QUDA_CTEST_DISABLE_BENCHMARKS})
    if(polenv)
      set_tests_properties(benchmark_dslash_ndeg_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_TWISTED_CLOVER)
    set(DIRAC_NAME twisted-clover)

    # symmetric preconditioning
    add_test(NAME dslash_${DIRAC_NAME}_sym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_${DIRAC_NAME}_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_sym_policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_${DIRAC_NAME}_asym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_${DIRAC_NAME}_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_asym_policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME benchmark_dslash_${DIRAC_NAME}_policy${pol2}
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
            --dslash-type ${DIRAC_NAME}
            --test 0
            --dim 20 20 20 20
            --gtest_output=json:dslash_${DIRAC_NAME}_benchmark_pol${pol2}.json
            --gtest_filter=*benchmark/*n0)
    set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES DISABLED ${QUDA_CTEST_DISABLE_BENCHMARKS})
    if(polenv)
      set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

  endif()

  if(QUDA_DIRAC_NDEG_TWISTED_CLOVER)
    # the ndeg-twisted-clover operator does not fit the pattern used for most other operators
    # since it is defined via 'dslash-type = twisted-clover` and 'flavor = nondeg-doublet'
    set(DIRAC_NAME twisted-clover)

    # symmetric preconditioning
    add_test(NAME dslash_ndeg_${DIRAC_NAME}_sym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --flavor nondeg-doublet
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_ndeg_${DIRAC_NAME}_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_ndeg_${DIRAC_NAME}_sym_policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric preconditioning
    add_test(NAME dslash_ndeg_${DIRAC_NAME}_asym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --flavor nondeg-doublet
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_ndeg_${DIRAC_NAME}_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_ndeg_${DIRAC_NAME}_asym_policy${pol2}
                           PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME benchmark_dslash_ndeg_${DIRAC_NAME}_policy${pol2}
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
            --dslash-type ${DIRAC_NAME}
            --flavor nondeg-doublet
            --test 0
            --dim 20 20 20 20
            --gtest_output=json:dslash_${DIRAC_NAME}_benchmark_pol${pol2}.json
            --gtest_filter=*benchmark/*n0)
    set_tests_properties(benchmark_dslash_ndeg_${DIRAC_NAME}_policy${pol2} PROPERTIES DISABLED ${QUDA_CTEST_DISABLE_BENCHMARKS})
    if(polenv)
      set_tests_properties(benchmark_dslash_ndeg_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

  endif()

  if(QUDA_DIRAC_DOMAIN_WALL)
    # 5-d preconditioned
    set(DIRAC_NAME domain-wall)

    add_test(NAME dslash_${DIRAC_NAME}_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --Lsdim 4
        --gtest_output=xml:dslash_${DIRAC_NAME}_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME benchmark_dslash_${DIRAC_NAME}_policy${pol2}
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
            --dslash-type ${DIRAC_NAME}
            --test 0
            --dim 20 20 20 20
            --Lsdim 12
            --gtest_output=json:dslash_${DIRAC_NAME}_benchmark_pol${pol2}.json
            --gtest_filter=*benchmark/*n0)
    set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES DISABLED ${QUDA_CTEST_DISABLE_BENCHMARKS})
    if(polenv)
      set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # 4-d preconditioned
    set(DIRAC_NAME domain-wall-4d)

    # symmetric 4-d preconditioning
    add_test(NAME dslash_${DIRAC_NAME}_sym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --Lsdim 4
        --gtest_output=xml:dslash_${DIRAC_NAME}_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_asym_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # asymmetric 4-d preconditioning
    add_test(NAME dslash_${DIRAC_NAME}_asym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --Lsdim 4
        --gtest_output=xml:dslash_${DIRAC_NAME}_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_asym_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME benchmark_dslash_${DIRAC_NAME}_policy${pol2}
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
            --dslash-type ${DIRAC_NAME}
            --test 0
            --dim 20 20 20 20
            --Lsdim 12
            --gtest_output=json:dslash_${DIRAC_NAME}_benchmark_pol${pol2}.json
            --gtest_filter=*benchmark/*n0)
    set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES DISABLED ${QUDA_CTEST_DISABLE_BENCHMARKS})
    if(polenv)
      set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # Mobius fermions
    set(DIRAC_NAME mobius)

    # 4-d preconditioning
    add_test(NAME dslash_${DIRAC_NAME}_sym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --Lsdim 4
                     --gtest_output=xml:dslash_${DIRAC_NAME}_sym_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_sym_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME dslash_${DIRAC_NAME}_asym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --Lsdim 4
                     --gtest_output=xml:dslash_${DIRAC_NAME}_asym_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_asym_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME benchmark_dslash_${DIRAC_NAME}_policy${pol2}
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
            --dslash-type ${DIRAC_NAME}
            --test 0
            --dim 20 20 20 20
            --Lsdim 12
            --gtest_output=json:dslash_${DIRAC_NAME}_benchmark_pol${pol2}.json
            --gtest_filter=*benchmark/*n0)
    set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES DISABLED ${QUDA_CTEST_DISABLE_BENCHMARKS})
    if(polenv)
      set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    # MdagM local operator
    # This requires tensor cores so only enable NVIDIA targets
    # FIXME: Ideally we should have a more sensitive guard -- e.g. QUDA_TENSOR_CORE_AVAILABLE 
    # or some such so that this would not be an architecture specific guard, but a feature
    # specific guard
    # 
    if( QUDA_TARGET_CUDA )
      add_test(NAME dslash_${DIRAC_NAME}_local_policy${pol2}
               COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                       --dslash-type ${DIRAC_NAME}
                       --test MatPCDagMatPCLocal
                       --matpc even-even
                       --dim 2 4 6 8
                       --Lsdim 4
                       --gtest_output=xml:dslash_${DIRAC_NAME}_local{pol2}.xml)
    endif()

    # Mobius-EOFA fermions
    set(DIRAC_NAME mobius-eofa)

    add_test(NAME dslash_${DIRAC_NAME}_sym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even
                     --dim 2 4 6 8
                     --Lsdim 4
        --gtest_output=xml:dslash_${DIRAC_NAME}_sym_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_sym_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME dslash_${DIRAC_NAME}_asym_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPCDagMatPC
                     --matpc even-even-asym
                     --dim 2 4 6 8
                     --Lsdim 4
                     --gtest_output=xml:dslash_${DIRAC_NAME}_asym_test_pol${pol2}.xml)
 
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_asym_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()
  endif()

  if(QUDA_DIRAC_STAGGERED)
    set(DIRAC_NAME staggered)
    add_test(NAME dslash_${DIRAC_NAME}_matpc_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test MatPC
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_${DIRAC_NAME}_matpc_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_matpc_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol2})
    endif()

    add_test(NAME dslash_${DIRAC_NAME}_mat_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --test Mat
                     --dim 2 4 6 8
                     --gtest_output=xml:dslash_${DIRAC_NAME}_mat_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_mat_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol2})
    endif()

    add_test(NAME benchmark_dslash_${DIRAC_NAME}_policy${pol2}
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS}
            --dslash-type ${DIRAC_NAME}
            --test 0
            --dim 20 20 20 20
            --gtest_output=json:dslash_${DIRAC_NAME}_benchmark_pol${pol2}.json
            --gtest_filter=*benchmark/*n0)
    set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES DISABLED ${QUDA_CTEST_DISABLE_BENCHMARKS})
    if(polenv)
      set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    set(DIRAC_NAME asqtad)
    add_test(NAME dslash_${DIRAC_NAME}_matpc_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --all-partitions 1
                     --test MatPC
                     --dim 6 8 10 12
                     --gtest_output=xml:dslash_${DIRAC_NAME}_matpc_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_matpc_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol2})
    endif()

    add_test(NAME dslash_${DIRAC_NAME}_mat_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type ${DIRAC_NAME}
                     --all-partitions 1
                     --test Mat
                     --dim 6 8 10 12
                     --gtest_output=xml:dslash_${DIRAC_NAME}_mat_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_mat_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol2})
    endif()

    add_test(NAME benchmark_dslash_${DIRAC_NAME}_policy${pol2}
    COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS}
            --dslash-type ${DIRAC_NAME}
            --all-partitions 1
            --test 0
            --dim 20 20 20 20
            --gtest_output=json:dslash_${DIRAC_NAME}_benchmark_pol${pol2}.json
            --gtest_filter=*benchmark/*n0)
    set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES DISABLED ${QUDA_CTEST_DISABLE_BENCHMARKS})
    if(polenv)
      set_tests_properties(benchmark_dslash_${DIRAC_NAME}_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol})
    endif()

    add_test(NAME dslash_${DIRAC_NAME}_build_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_dslash_ctest> ${MPIEXEC_POSTFLAGS}
                     --dslash-type asqtad
                     --test MatPC
                     --dim 6 8 10 12
                     --compute-fat-long true
                     --epsilon-naik -0.01
                     --tadpole-coeff 0.9
        --gtest_output=xml:dslash_${DIRAC_NAME}_build_test_pol${pol2}.xml)
    if(polenv)
      set_tests_properties(dslash_${DIRAC_NAME}_build_policy${pol2} PROPERTIES ENVIRONMENT QUDA_ENABLE_DSLASH_POLICY=${pol2})
    endif()
  endif()

  if(QUDA_COVDEV)
    add_test(NAME covdev_policy${pol2}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:covdev_test> ${MPIEXEC_POSTFLAGS}
                     --dim 6 8 10 12
                     --gtest_output=xml:covdev_test_pol${pol2}.xml)
  endif()

endforeach(pol)

# enable the precisions that are compiled
math(EXPR double_prec  "${QUDA_PRECISION} & 8")
math(EXPR single_prec  "${QUDA_PRECISION} & 4")
math(EXPR half_prec    "${QUDA_PRECISION} & 2")
math(EXPR quarter_prec "${QUDA_PRECISION} & 1")

if(double_prec AND single_prec)
  set(TEST_PRECS single double)
elseif(double_prec)
  set(TEST_PRECS double)
elseif(single_prec)
  set(TEST_PRECS single)
endif()

# Inversions
foreach(prec IN LISTS TEST_PRECS)

  if(${prec} STREQUAL "double")
    set(tol 1e-12)
  elseif(${prec} STREQUAL "single")
    set(tol 1e-6)
  endif()

  if(QUDA_DIRAC_WILSON)    
    add_test(NAME invert_test_wilson_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type wilson --ngcrkrylov 8
      --dim 2 4 6 8 --prec ${prec} --tol ${tol} --niter 1000
      --enable-testing true
      --gtest_output=xml:invert_test_wilson_${prec}.xml)
  endif()
  
  if(QUDA_DIRAC_TWISTED_MASS)
    add_test(NAME invert_test_twisted_mass_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-mass
      --dim 2 4 6 8 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even --enable-testing true
      --gtest_output=xml:invert_test_twisted_mass_sym_${prec}.xml)

    add_test(NAME invert_test_twisted_mass_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-mass
      --dim 2 4 6 8 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even-asym --enable-testing true
      --gtest_output=xml:invert_test_twisted_mass_asym_${prec}.xml)    
  endif()

  if(QUDA_DIRAC_NDEG_TWISTED_MASS)
    add_test(NAME invert_test_ndeg_twisted_mass_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-mass
      --dim 2 4 6 8 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even --flavor nondeg-doublet
      --enable-testing true
      --gtest_output=xml:invert_test_ndeg_twisted_mass_sym_${prec}.xml)

    add_test(NAME invert_test_ndeg_twisted_mass_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-mass
      --dim 2 4 6 8 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even-asym --flavor nondeg-doublet
      --enable-testing true
      --gtest_output=xml:invert_test_ndeg_twisted_mass_asym_${prec}.xml)
  endif()

  if(QUDA_DIRAC_CLOVER)
    add_test(NAME invert_test_clover_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type clover --compute-clover true
      --dim 2 4 6 8 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even --enable-testing true
      --gtest_output=xml:invert_test_clover_sym_${prec}.xml)

    add_test(NAME invert_test_clover_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type clover --compute-clover true
      --dim 2 4 6 8 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even-asym --enable-testing true
      --gtest_output=xml:invert_test_clover_asym_${prec}.xml)
  endif()

  if(QUDA_DIRAC_TWISTED_CLOVER)
    add_test(NAME invert_test_twisted_clover_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-clover --compute-clover true
      --dim 2 4 6 8 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even --enable-testing true
      --gtest_output=xml:invert_test_twisted_clover_sym_${prec}.xml)

    add_test(NAME invert_test_twisted_clover_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-clover --compute-clover true
      --dim 2 4 6 8 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even-asym --enable-testing true
      --gtest_output=xml:invert_test_twisted_clover_asym_${prec}.xml)
  endif()

  if(QUDA_DIRAC_NDEG_TWISTED_CLOVER)
    add_test(NAME invert_test_ndeg_twisted_clover_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-clover --compute-clover true
      --dim 2 4 6 8 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even --flavor nondeg-doublet
      --enable-testing true
      --gtest_output=xml:invert_test_ndeg_twisted_clover_sym_${prec}.xml)

    add_test(NAME invert_test_ndeg_twisted_clover_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-clover --compute-clover true
      --dim 2 4 6 8 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even-asym --flavor nondeg-doublet
      --enable-testing true
      --gtest_output=xml:invert_test_ndeg_twisted_clover_asym_${prec}.xml)
  endif()

  if(QUDA_DIRAC_DOMAIN_WALL)
    add_test(NAME invert_test_domain_wall_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type domain-wall
      --dim 2 4 6 8 --Lsdim 4 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even
      --enable-testing true
      --gtest_output=xml:invert_test_domain_wall_${prec}.xml)

    add_test(NAME invert_test_domain_wall_4d_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type domain-wall-4d
      --dim 2 4 6 8 --Lsdim 4 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even
      --enable-testing true
      --gtest_output=xml:invert_test_domain_wall_4d_sym_${prec}.xml)

    add_test(NAME invert_test_domain_wall_4d_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type domain-wall-4d
      --dim 2 4 6 8 --Lsdim 4 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even-asym
      --enable-testing true
      --gtest_output=xml:invert_test_domain_wall_4d_asym_${prec}.xml)

    add_test(NAME invert_test_mobius_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type mobius
      --dim 2 4 6 8 --Lsdim 4 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even
      --enable-testing true
      --gtest_output=xml:invert_test_mobius_sym_${prec}.xml)

    add_test(NAME invert_test_mobius_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type mobius
      --dim 2 4 6 8 --Lsdim 4 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even-asym
      --enable-testing true
      --gtest_output=xml:invert_test_mobius_asym_${prec}.xml)

    add_test(NAME invert_test_mobius_eofa_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type mobius-eofa
      --dim 2 4 6 8 --Lsdim 4 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even
      --enable-testing true
      --gtest_output=xml:invert_test_mobius_eofa_sym_${prec}.xml)

    add_test(NAME invert_test_mobius_eofa_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:invert_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type mobius-eofa
      --dim 2 4 6 8 --Lsdim 4 --prec ${prec} --tol ${tol} --niter 1000 --ngcrkrylov 8
      --matpc even-even-asym
      --enable-testing true
      --gtest_output=xml:invert_test_mobius_eofa_asym_${prec}.xml)
  endif()
endforeach(prec)

# Eigensolves
foreach(prec IN LISTS TEST_PRECS)

  if(${prec} STREQUAL "double")
    set(tol 1e-12)
  elseif(${prec} STREQUAL "single")
    set(tol 1e-6)
  endif()

  if(QUDA_DIRAC_WILSON)    
    add_test(NAME eigensolve_test_wilson_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type wilson --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --dim 2 4 6 8 --prec ${prec} --eig-tol ${tol} --eig-max-restarts 1000
      --enable-testing true 
      --gtest_output=xml:eigensolve_test_wilson_${prec}.xml)
  endif()
  
  if(QUDA_DIRAC_TWISTED_MASS)
    add_test(NAME eigensolve_test_twisted_mass_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-mass --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --dim 2 4 6 8 --prec ${prec} --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even --enable-testing true
      --gtest_output=xml:eigensolve_test_twisted_mass_sym_${prec}.xml)
    
    add_test(NAME eigensolve_test_twisted_mass_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-mass --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --dim 2 4 6 8 --prec ${prec} --eig-tol ${tol} --eig-max-restarts 1000      
      --matpc even-even-asym --enable-testing true 
      --gtest_output=xml:eigensolve_test_twisted_mass_asym_${prec}.xml)
  endif()
  
  if(QUDA_DIRAC_NDEG_TWISTED_MASS)
    add_test(NAME eigensolve_test_ndeg_twisted_mass_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-mass --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --dim 2 4 6 8 --prec ${prec} --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even --flavor nondeg-doublet 
      --enable-testing true --eig-use-eigen-qr false --eig-qr-tol ${tol}
      --gtest_output=xml:eigensolve_test_ndeg_twisted_mass_sym_${prec}.xml)

    add_test(NAME eigensolve_test_ndeg_twisted_mass_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-mass --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --dim 2 4 6 8 --prec ${prec} --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even-asym --flavor nondeg-doublet
      --enable-testing true --eig-use-eigen-qr false --eig-qr-tol ${tol}
      --gtest_output=xml:eigensolve_test_ndeg_twisted_mass_asym_${prec}.xml)
  endif()
  
  if(QUDA_DIRAC_CLOVER)
    add_test(NAME eigensolve_test_clover_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type clover --compute-clover true      
      --dim 2 4 6 8 --prec ${prec} --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000 
      --matpc even-even --enable-testing true 
      --gtest_output=xml:eigensolve_test_clover_sym_${prec}.xml)

    add_test(NAME eigensolve_test_clover_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type clover --compute-clover true      
      --dim 2 4 6 8 --prec ${prec} --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even-asym --enable-testing true
      --gtest_output=xml:eigensolve_test_clover_asym_${prec}.xml)
  endif()

  if(QUDA_DIRAC_TWISTED_CLOVER)
    add_test(NAME eigensolve_test_twisted_clover_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-clover --compute-clover true
      --dim 2 4 6 8 --prec ${prec} --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even --enable-testing true 
      --gtest_output=xml:eigensolve_test_twisted_clover_sym_${prec}.xml)
    
    add_test(NAME eigensolve_test_twisted_clover_asym_${prec}      
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-clover --compute-clover true
      --dim 2 4 6 8 --prec ${prec} --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even-asym --enable-testing true 
      --gtest_output=xml:eigensolve_test_twisted_clover_asym_${prec}.xml)
  endif()
  
  if(QUDA_DIRAC_NDEG_TWISTED_CLOVER)
    add_test(NAME eigensolve_test_ndeg_twisted_clover_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-clover --compute-clover true
      --dim 2 4 6 8 --prec ${prec} --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even --flavor nondeg-doublet
      --enable-testing true 
      --gtest_output=xml:eigensolve_test_ndeg_twisted_clover_sym_${prec}.xml)
    
    add_test(NAME eigensolve_test_ndeg_twisted_clover_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type twisted-clover --compute-clover true
      --dim 2 4 6 8 --prec ${prec} --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even-asym --flavor nondeg-doublet
      --enable-testing true 
      --gtest_output=xml:eigensolve_test_ndeg_twisted_clover_asym_${prec}.xml)
  endif()
  
  if(QUDA_DIRAC_DOMAIN_WALL)
    add_test(NAME eigensolve_test_domain_wall_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type domain-wall --prec ${prec}
      --dim 2 4 6 8 --Lsdim 4 --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even --enable-testing true 
      --gtest_output=xml:eigensolve_test_domain_wall_${prec}.xml)

    add_test(NAME eigensolve_test_domain_wall_4d_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type domain-wall-4d --prec ${prec}
      --dim 2 4 6 8 --Lsdim 4 --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even --enable-testing true 
      --gtest_output=xml:eigensolve_test_domain_wall_4d_sym_${prec}.xml)

    add_test(NAME eigensolve_test_domain_wall_4d_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type domain-wall-4d --prec ${prec}
      --dim 2 4 6 8 --Lsdim 4 --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even-asym --enable-testing true 
      --gtest_output=xml:eigensolve_test_domain_wall_4d_asym_${prec}.xml)

    add_test(NAME eigensolve_test_mobius_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type mobius --prec ${prec}
      --dim 2 4 6 8 --Lsdim 4 --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even --enable-testing true 
      --gtest_output=xml:eigensolve_test_mobius_sym_${prec}.xml)

    add_test(NAME eigensolve_test_mobius_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type mobius --prec ${prec}
      --dim 2 4 6 8 --Lsdim 4 --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even-asym --enable-testing true 
      --gtest_output=xml:eigensolve_test_mobius_asym_${prec}.xml)

    add_test(NAME eigensolve_test_mobius_eofa_sym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type mobius-eofa --prec ${prec}
      --dim 2 4 6 8 --Lsdim 4 --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even --enable-testing true 
      --gtest_output=xml:eigensolve_test_mobius_eofa_sym_${prec}.xml)

    add_test(NAME eigensolve_test_mobius_eofa_asym_${prec}
      COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:eigensolve_test> ${MPIEXEC_POSTFLAGS}
      --dslash-type mobius-eofa --prec ${prec}
      --dim 2 4 6 8 --Lsdim 4 --eig-n-conv 24 --eig-n-ev 24 --eig-n-kr 128
      --eig-tol ${tol} --eig-max-restarts 1000
      --matpc even-even-asym --enable-testing true 
      --gtest_output=xml:eigensolve_test_mobius_eofa_asym_${prec}.xml)
  endif()
endforeach(prec)
  
foreach(prec IN LISTS TEST_PRECS)

  add_test(NAME gauge_path_${prec}
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:gauge_path_test> ${MPIEXEC_POSTFLAGS}
                   --dim 2 4 6 8 --prec ${prec}
                   --gtest_output=xml:gauge_path_test_${prec}.xml)

  if(QUDA_DIRAC_STAGGERED)
    add_test(NAME unitarize_link_${prec}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:unitarize_link_test> ${MPIEXEC_POSTFLAGS}
                     --dim 2 4 6 8 --prec ${prec}
                     --gtest_output=xml:unitarize_link_test_${prec}.xml)
  endif()

  if(QUDA_FORCE_HISQ)
    add_test(NAME hisq_paths_force_${prec}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:hisq_paths_force_test> ${MPIEXEC_POSTFLAGS}
                     --dim 2 4 6 8 --prec ${prec}
                     --gtest_output=xml:hisq_paths_force_test_${prec}.xml)

    add_test(NAME hisq_unitarize_force_${prec}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:hisq_unitarize_force_test> ${MPIEXEC_POSTFLAGS}
                     --dim 2 4 6 8 --prec ${prec}
                     --gtest_output=xml:hisq_unitarize_force_test_${prec}.xml)
  endif()

  add_test(NAME gauge_alg_${prec}
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:gauge_alg_test> ${MPIEXEC_POSTFLAGS}
                   --dim 4 6 8 10 --prec ${prec}
                   --gtest_output=xml:gauge_alg_test_${prec}.xml)

  if (TARGET dilution_test)
    add_test(NAME dilution_test_${prec}
             COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:dilution_test> ${MPIEXEC_POSTFLAGS}
                     --dim 4 6 8 10 --prec ${prec}
                     --gtest_output=xml:dilution_test_${prec}.xml)
  endif()

  if(QUDA_SMEAR_GAUSS_TWOLINK)
    set(KERNEL_TYPE TwoLink GaussianSmear)
    foreach(kerneltp IN LISTS KERNEL_TYPE)
      if(${kerneltp} STREQUAL "TwoLink")
	set(laplace3D 4)
	set(smear_t0 -1)
      elseif(${kerneltp} STREQUAL "GaussianSmear")
	set(laplace3D 3)
	set(smear_t0 1)
      endif()	    
      add_test(NAME staggered_gsmear_${kerneltp}_${prec}
        COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:staggered_gsmear_test> ${MPIEXEC_POSTFLAGS}
        --dim 6 8 10 12 --prec ${prec}
        --test ${kerneltp}
	--laplace3D ${laplace3D}
	--smear-t0 ${smear_t0}
        --gtest_output=xml:staggered_gsmear_test_${kerneltp}_${prec}.xml)
    endforeach(kerneltp)
  endif()

endforeach(prec)

if(QUDA_QIO)
  add_test(NAME io_test
           COMMAND ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:io_test> ${MPIEXEC_POSTFLAGS}
                   --dim 4 6 8 10
                   --gtest_output=xml:io_test.xml)
endif()

add_test(NAME tune_test
         COMMAND  ${QUDA_CTEST_LAUNCH} $<TARGET_FILE:tune_test> ${MPIEXEC_POSTFLAGS}
                   --gtest_output=xml:tune_test.xml)
