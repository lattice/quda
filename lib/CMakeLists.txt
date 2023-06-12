# cmake-format: off

# this allows simplified running of clang-tidy
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# build up git version add -debug to GITVERSION if we build with debug options enabled
string(REGEX MATCH [Dd][Ee][Bb][Uu][Gg] DEBUG_BUILD ${CMAKE_BUILD_TYPE})
if(DEBUG_BUILD)
  if(GITVERSION)
    set(GITVERSION ${GITVERSION}-debug)
  else()
    set(GITVERSION debug)
  endif()
endif()

set (QUDA_OBJS
  # cmake-format: sortable
  dirac_coarse.cpp dslash_coarse.cpp
  coarse_op.cpp coarsecoarse_op.cpp
  coarse_op_preconditioned.cpp staggered_coarse_op.cpp
  eig_iram.cpp eig_trlm.cpp eig_block_trlm.cpp vector_io.cpp
  eigensolve_quda.cpp quda_arpack_interface.cpp
  multigrid.cpp transfer.cpp block_orthogonalize.cpp
  prolongator.cpp restrictor.cpp staggered_prolong_restrict.cu
  gauge_phase.cu timer.cpp
  solver.cpp inv_bicgstab_quda.cpp inv_cg_quda.cpp inv_bicgstabl_quda.cpp
  inv_multi_cg_quda.cpp inv_eigcg_quda.cpp gauge_ape.cu
  gauge_stout.cu gauge_wilson_flow.cu gauge_plaq.cu
  gauge_laplace.cpp gauge_observable.cpp
  inv_cg3_quda.cpp inv_ca_gcr.cpp inv_ca_cg.cpp
  inv_gcr_quda.cpp inv_mr_quda.cpp inv_sd_quda.cpp
  inv_pcg_quda.cpp inv_mre.cpp interface_quda.cpp util_quda.cpp
  color_spinor_field.cpp color_spinor_util.cu
  field_cache.cpp
  gauge_covdev.cpp dirac.cpp
  clover_field.cpp lattice_field.cpp gauge_field.cpp
  cpu_gauge_field.cpp cuda_gauge_field.cpp extract_gauge_ghost.cu
  gauge_norm.cu gauge_update_quda.cu
  max_clover.cu dirac_clover.cpp dirac_wilson.cpp dirac_staggered.cpp
  dirac_staggered_kd.cpp dirac_clover_hasenbusch_twist.cpp
  dirac_improved_staggered.cpp dirac_improved_staggered_kd.cpp dirac_domain_wall.cpp
  dirac_domain_wall_4d.cpp dirac_mobius.cpp dirac_twisted_clover.cpp
  dirac_twisted_mass.cpp 
  llfat_quda.cu staggered_two_link_quda.cu gauge_force.cu gauge_loop_trace.cu gauge_polyakov_loop.cu
  gauge_random.cu gauge_noise.cu
  gauge_field_strength_tensor.cu clover_quda.cu 
  dslash_gamma_helper.cu dslash_clover_helper.cu
  staggered_kd_build_xinv.cu staggered_kd_reorder_xinv.cu staggered_kd_apply_xinv.cu
  madwf_transfer.cu madwf_tensor.cu
  blas_quda.cu multi_blas_quda.cu reduce_quda.cu
  multi_reduce_quda.cu reduce_helper.cu
  contract.cu comm_common.cpp communicator_stack.cpp
  clover_deriv_quda.cu clover_invert.cu copy_gauge_extended.cu
  extract_gauge_ghost_extended.cu copy_color_spinor.cpp
  spinor_noise.cu spinor_dilute.cu
  copy_color_spinor_dd.cu copy_color_spinor_ds.cu
  copy_color_spinor_dh.cu copy_color_spinor_dq.cu
  copy_color_spinor_ss.cu copy_color_spinor_sd.cu
  copy_color_spinor_sh.cu copy_color_spinor_sq.cu
  copy_color_spinor_hd.cu copy_color_spinor_hs.cu
  copy_color_spinor_hh.cu copy_color_spinor_hq.cu
  copy_color_spinor_qd.cu copy_color_spinor_qs.cu
  copy_color_spinor_qh.cu copy_color_spinor_qq.cu
  copy_color_spinor_mg_dd.cu copy_color_spinor_mg_ds.cu
  copy_color_spinor_mg_sd.cu copy_color_spinor_mg_ss.cu
  copy_color_spinor_mg_sh.cu copy_color_spinor_mg_sq.cu
  copy_color_spinor_mg_hs.cu copy_color_spinor_mg_hh.cu
  copy_color_spinor_mg_hq.cu copy_color_spinor_mg_qs.cu
  copy_color_spinor_mg_qh.cu copy_color_spinor_mg_qq.cu
  copy_gauge_double.cu copy_gauge_single.cu
  copy_gauge_half.cu copy_gauge_quarter.cu
  copy_gauge.cpp copy_clover.cu
  copy_gauge_offset.cu copy_color_spinor_offset.cu copy_clover_offset.cu
  staggered_oprod.cu clover_trace_quda.cu
  hisq_paths_force_quda.cu
  unitarize_force_quda.cu unitarize_links_quda.cu milc_interface.cpp
  tune.cpp
  device_vector.cu
  inv_gmresdr_quda.cpp
  pgauge_exchange.cu pgauge_init.cu pgauge_heatbath.cu random.cu
  gauge_fix_fft.cu gauge_fix_ovr.cu pgauge_det_trace.cu clover_outer_product.cu
  clover_sigma_outer_product.cu momentum.cu gauge_qcharge.cu
  deflation.cpp checksum.cu transform_reduce.cu
  dslash5_mobius_eofa.cu
  madwf_ml.cpp
  instantiate.cpp version.cpp
  block_transpose.cu )
# cmake-format: on

# current workaround for nvshmem
set(QUDA_DSLASH_OBJS 
  dslash_quda.cu color_spinor_pack.cu
  dslash_staggered.cu dslash_improved_staggered.cu
  dslash_wilson.cu dslash_wilson_clover.cu dslash5_domain_wall.cu
  dslash_wilson_clover_preconditioned.cu 
  dslash_twisted_mass.cu dslash_twisted_mass_preconditioned.cu
  dslash_ndeg_twisted_mass.cu dslash_ndeg_twisted_mass_preconditioned.cu
  dslash_twisted_clover.cu dslash_twisted_clover_preconditioned.cu
  dslash_ndeg_twisted_clover.cu dslash_ndeg_twisted_clover_preconditioned.cu
  dslash_wilson_clover_hasenbusch_twist.cu
  dslash_wilson_clover_hasenbusch_twist_preconditioned.cu
  dslash_domain_wall_4d.cu  dslash_domain_wall_5d.cu
  dslash_domain_wall_4d_m5inv.cu
  dslash_domain_wall_4d_m5pre.cu
  dslash_domain_wall_4d_m5pre_m5inv.cu
  dslash_domain_wall_4d_m5inv_m5pre.cu
  dslash_domain_wall_4d_m5inv_m5inv.cu
  dslash_domain_wall_4d_m5mob.cu
  dslash_domain_wall_4d_m5pre_m5mob.cu
  dslash_pack2.cu laplace.cu covDev.cu staggered_quark_smearing.cu)

if(QUDA_NVSHMEM)
  list(APPEND QUDA_DSLASH_OBJS dslash_constant_arg.cu)
endif()

if(NOT QUDA_NVSHMEM)
  list(APPEND QUDA_OBJS ${QUDA_DSLASH_OBJS})
endif()

# split source into cu and cpp files
foreach(item ${QUDA_OBJS})
  string(REGEX MATCH ".+\\.cu$" item_match ${item})
  if(item_match)
    list(APPEND QUDA_CU_OBJS ${item})
  endif(item_match)
endforeach(item ${QUDA_OBJS})

list(REMOVE_ITEM QUDA_OBJS ${QUDA_CU_OBJS})

set(QUDA_MDW_FUSED_LS_LIST "4,8,12,16,20" CACHE STRING "The list of Ls that get compiled")
mark_as_advanced(QUDA_MDW_FUSED_LS_LIST)

message(STATUS "QUDA_mdw_fused_Ls=${QUDA_MDW_FUSED_LS_LIST}")
configure_file(dslash_mdw_fused.in.hpp dslash_mdw_fused.hpp @ONLY)

set(QUDA_MDW_FUSED_ALLOWED_LS_LIST 4 8 12 16 20)
if(QUDA_DIRAC_DOMAIN_WALL)
  string(REPLACE "," ";" QUDA_MDW_FUSED_LS_LIST_SEMICOLON "${QUDA_MDW_FUSED_LS_LIST}")
  foreach(QUDA_MDW_FUSED_LS ${QUDA_MDW_FUSED_LS_LIST_SEMICOLON})
    if(NOT ${QUDA_MDW_FUSED_LS} IN_LIST QUDA_MDW_FUSED_ALLOWED_LS_LIST)
      string(REPLACE ";" "," QUDA_MDW_FUSED_ALLOWED_LS_LIST_COMMA "${QUDA_MDW_FUSED_ALLOWED_LS_LIST}")
      message(FATAL_ERROR "'${QUDA_MDW_FUSED_LS}' is not a valid Ls: allowed ones are '${QUDA_MDW_FUSED_ALLOWED_LS_LIST_COMMA}'")
    endif()
    configure_file(dslash_mdw_fused.in.cu "dslash_mdw_fused_ls${QUDA_MDW_FUSED_LS}.cu" @ONLY)
    list(PREPEND QUDA_CU_OBJS "dslash_mdw_fused_ls${QUDA_MDW_FUSED_LS}.cu")
  endforeach()
endif()


set(QUDA_MULTIGRID_NVEC_LIST "6,24,32" CACHE STRING "The list of null space vector sizes that get compiled")
set(QUDA_MULTIGRID_NC_NVEC_LIST "3,${QUDA_MULTIGRID_NVEC_LIST}")
mark_as_advanced(QUDA_MULTIGRID_NVEC_LIST)

set(QUDA_MULTIGRID_MRHS_LIST "16" CACHE STRING "The list of multi-rhs sizes that get compiled")
mark_as_advanced(QUDA_MULTIGRID_MRHS_LIST)

option(QUDA_MULTIGRID_SETUP_USE_SMMA ON)
mark_as_advanced(QUDA_MULTIGRID_SETUP_USE_SMMA)

message(STATUS "QUDA_MULTIGRID_NVEC_LIST=${QUDA_MULTIGRID_NVEC_LIST}")
message(STATUS "QUDA_MULTIGRID_MRHS_LIST=${QUDA_MULTIGRID_MRHS_LIST}")
configure_file(dslash_coarse.in.cpp dslash_coarse.cpp @ONLY)
configure_file(coarse_op.in.cpp coarse_op.cpp @ONLY)
configure_file(coarsecoarse_op.in.cpp coarsecoarse_op.cpp @ONLY)
configure_file(coarse_op_preconditioned.in.cpp coarse_op_preconditioned.cpp @ONLY)
configure_file(staggered_coarse_op.in.cpp staggered_coarse_op.cpp @ONLY)
configure_file(prolongator.in.cpp prolongator.cpp @ONLY)
configure_file(restrictor.in.cpp restrictor.cpp @ONLY)
configure_file(block_orthogonalize.in.cpp block_orthogonalize.cpp @ONLY)
configure_file(copy_gauge.in.cpp copy_gauge.cpp @ONLY)
configure_file(extract_gauge_ghost.in.cu extract_gauge_ghost.cu @ONLY)
configure_file(gauge_noise.in.cu gauge_noise.cu @ONLY)
configure_file(gauge_norm.in.cu gauge_norm.cu @ONLY)
configure_file(spinor_noise.in.cu spinor_noise.cu @ONLY)
configure_file(copy_color_spinor_mg.in.hpp copy_color_spinor_mg.hpp @ONLY)
configure_file(color_spinor_pack.in.cu color_spinor_pack.cu @ONLY)
configure_file(color_spinor_util.in.cu color_spinor_util.cu @ONLY)
configure_file(dslash_coarse_mma.in.hpp dslash_coarse_mma.hpp @ONLY)
configure_file(block_transpose.in.cu block_transpose.cu @ONLY)

if(QUDA_MULTIGRID)
  string(REPLACE "," ";" QUDA_MULTIGRID_NVEC_LIST_SEMICOLON "${QUDA_MULTIGRID_NVEC_LIST}")
  string(REPLACE "," ";" QUDA_MULTIGRID_MRHS_LIST_SEMICOLON "${QUDA_MULTIGRID_MRHS_LIST}")
  foreach(QUDA_MULTIGRID_NVEC ${QUDA_MULTIGRID_NVEC_LIST_SEMICOLON})
    configure_file(copy_gauge_mg.in.cu "copy_gauge_mg_${QUDA_MULTIGRID_NVEC}.cu" @ONLY)
    configure_file(extract_gauge_ghost_mg.in.cu "extract_gauge_ghost_mg_${QUDA_MULTIGRID_NVEC}.cu" @ONLY)
    list(PREPEND QUDA_CU_OBJS "copy_gauge_mg_${QUDA_MULTIGRID_NVEC}.cu")
    list(PREPEND QUDA_CU_OBJS "extract_gauge_ghost_mg_${QUDA_MULTIGRID_NVEC}.cu")

    configure_file(coarse_op.in.cu "coarse_op_${QUDA_MULTIGRID_NVEC}.cu" @ONLY)
    configure_file(staggered_coarse_op.in.cu "staggered_coarse_op_${QUDA_MULTIGRID_NVEC}.cu" @ONLY)
    list(PREPEND QUDA_CU_OBJS "coarse_op_${QUDA_MULTIGRID_NVEC}.cu")
    list(PREPEND QUDA_CU_OBJS "staggered_coarse_op_${QUDA_MULTIGRID_NVEC}.cu")

    configure_file(coarse_op_preconditioned.in.cu "coarse_op_preconditioned_${QUDA_MULTIGRID_NVEC}.cu" @ONLY)
    list(PREPEND QUDA_CU_OBJS "coarse_op_preconditioned_${QUDA_MULTIGRID_NVEC}.cu")

    foreach(QUDA_MULTIGRID_NVEC2 ${QUDA_MULTIGRID_NVEC_LIST_SEMICOLON})
      if(${QUDA_MULTIGRID_NVEC2} GREATER_EQUAL ${QUDA_MULTIGRID_NVEC})
         configure_file(coarsecoarse_op.in.cu "coarsecoarse_op_${QUDA_MULTIGRID_NVEC}_${QUDA_MULTIGRID_NVEC2}.cu" @ONLY)
         configure_file(coarsecoarse_op_mma.in.cu "coarsecoarse_op_mma_${QUDA_MULTIGRID_NVEC}_${QUDA_MULTIGRID_NVEC2}.cu" @ONLY)
         list(PREPEND QUDA_CU_OBJS "coarsecoarse_op_${QUDA_MULTIGRID_NVEC}_${QUDA_MULTIGRID_NVEC2}.cu")
         list(PREPEND QUDA_CU_OBJS "coarsecoarse_op_mma_${QUDA_MULTIGRID_NVEC}_${QUDA_MULTIGRID_NVEC2}.cu")
      endif()
    endforeach()

  endforeach()

  string(REPLACE "," ";" QUDA_MULTIGRID_NC_NVEC_LIST_SEMICOLON "${QUDA_MULTIGRID_NC_NVEC_LIST}")
  foreach(QUDA_MULTIGRID_NC_NVEC ${QUDA_MULTIGRID_NC_NVEC_LIST_SEMICOLON})
    foreach(QUDA_MULTIGRID_NVEC2 ${QUDA_MULTIGRID_NVEC_LIST_SEMICOLON})
      if(${QUDA_MULTIGRID_NVEC2} GREATER_EQUAL ${QUDA_MULTIGRID_NC_NVEC})
         configure_file(prolongator.in.cu "prolongator_${QUDA_MULTIGRID_NC_NVEC}_${QUDA_MULTIGRID_NVEC2}.cu" @ONLY)
         configure_file(restrictor.in.cu "restrictor_${QUDA_MULTIGRID_NC_NVEC}_${QUDA_MULTIGRID_NVEC2}.cu" @ONLY)
         configure_file(block_orthogonalize.in.cu "block_orthogonalize_${QUDA_MULTIGRID_NC_NVEC}_${QUDA_MULTIGRID_NVEC2}.cu" @ONLY)
         list(PREPEND QUDA_CU_OBJS "prolongator_${QUDA_MULTIGRID_NC_NVEC}_${QUDA_MULTIGRID_NVEC2}.cu")
         list(PREPEND QUDA_CU_OBJS "restrictor_${QUDA_MULTIGRID_NC_NVEC}_${QUDA_MULTIGRID_NVEC2}.cu")
         list(PREPEND QUDA_CU_OBJS "block_orthogonalize_${QUDA_MULTIGRID_NC_NVEC}_${QUDA_MULTIGRID_NVEC2}.cu")
      endif()
    endforeach()
  endforeach()

  # add dslash_coarse last to the list so it is compiled first
  foreach(QUDA_MULTIGRID_NVEC ${QUDA_MULTIGRID_NVEC_LIST_SEMICOLON})
    set(QUDA_MULTIGRID_DAGGER "false")
    configure_file(dslash_coarse.in.cu "dslash_coarse_${QUDA_MULTIGRID_NVEC}.cu" @ONLY)
    set(QUDA_MULTIGRID_DAGGER "true")
    configure_file(dslash_coarse.in.cu "dslash_coarse_dagger_${QUDA_MULTIGRID_NVEC}.cu" @ONLY)
    list(PREPEND QUDA_CU_OBJS "dslash_coarse_${QUDA_MULTIGRID_NVEC}.cu")
    list(PREPEND QUDA_CU_OBJS "dslash_coarse_dagger_${QUDA_MULTIGRID_NVEC}.cu")
  endforeach()

endif()

if(BUILD_FORTRAN_INTERFACE)
  list(APPEND QUDA_OBJS quda_fortran.F90)

  # set_source_files_properties(quda_fortran.F90 PROPERTIES OBJECT_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/quda_fortran.mod)
endif()

# QUDA_CU_OBJS should contain all cuda files now QUDA_OBJS all c, cpp, fortran sources
# if we have a git version make version.cpp depend on git head so that it is rebuild if the git sha changed
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  if(GITVERSION)
    find_path(
      QUDA_GITDIR NAME HEAD
      PATHS ${CMAKE_SOURCE_DIR}/.git/logs
      NO_DEFAULT_PATH)
    include(AddFileDependencies)
    if(QUDA_GITDIR)
      add_file_dependencies(version.cpp ${QUDA_GITDIR}/HEAD)
    endif()
  endif()
  mark_as_advanced(QUDA_GITDIR)
endif()

# generate a cmake object library for all cpp files first
add_library(quda_cpp OBJECT ${QUDA_OBJS})

# add comms and QIO
target_sources(
  quda_cpp
  PRIVATE
    $<IF:$<BOOL:${QUDA_MPI}>,communicator_mpi.cpp,$<IF:$<BOOL:${QUDA_QMP}>,communicator_qmp.cpp,communicator_single.cpp>>
)

target_sources(quda_cpp PRIVATE $<$<BOOL:${QUDA_QIO}>:qio_field.cpp layout_hyper.cpp>)

if(QUDA_BUILD_SHAREDLIB)
  set_target_properties(quda_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  add_library(quda SHARED)
else()
  add_library(quda STATIC)
endif()
add_library(QUDA::quda ALIAS quda)

# add target specific files
if(${QUDA_TARGET_TYPE} STREQUAL "CUDA")
  include(targets/cuda/target_cuda.cmake)
endif()
if(${QUDA_TARGET_TYPE} STREQUAL "HIP")
  include(targets/hip/target_hip.cmake)
endif()
if(${QUDA_TARGET_TYPE} STREQUAL "SYCL")
  include(targets/sycl/target_sycl.cmake)
endif()

# make one library
target_sources(quda PRIVATE $<TARGET_OBJECTS:quda_cpp> $<$<TARGET_EXISTS:quda_pack>:$<TARGET_OBJECTS:quda_pack>>
                            ${QUDA_CU_OBJS})

# for a develop build reduce the size by compressing the debug information
include(CheckLinkerFlag)
check_linker_flag(CXX "-Wl,--compress-debug-sections=zlib" QUDA_LINKER_COMPRESS)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-gz QUDA_COMPRESS_DEBUG)

if(NOT QUDA_BACKWARDS)
  if(${QUDA_LINKER_COMPRESS})
    target_link_options(quda PRIVATE $<$<CONFIG:DEVEL>:$<${QUDA_LINKER_COMPRESS}:-Wl,--compress-debug-sections=zlib>>)
  endif()
  if(${QUDA_COMPRESS_DEBUG})
    target_compile_options(quda_cpp PRIVATE $<$<CONFIG:DEVEL>:$<${QUDA_COMPRESS_DEBUG}:-gz>>)
    target_compile_options(quda PRIVATE $<$<CONFIG:DEVEL>:$<${QUDA_COMPRESS_DEBUG}:-gz>>)
  endif()
endif()

# set up QUDA compile options, put them before target specific options
target_compile_options(
  quda BEFORE PRIVATE $<$<COMPILE_LANGUAGE:CXX,C,Fortran>:
               $<IF:$<CONFIG:RELEASE>,-w,-Wall -Wextra>
               $<$<CONFIG:STRICT>:-Werror>
  >)

target_compile_definitions(
  quda PRIVATE $<$<CONFIG:DEVEL>:DEVEL> $<$<CONFIG:HOSTDEBUG>:HOST_DEBUG> $<$<CONFIG:DEBUG>:HOST_DEBUG>
               $<$<CONFIG:DEBUG>:DEVICE_DEBUG> $<$<CONFIG:SANITIZE>:HOST_DEBUG>)
target_include_directories(quda PRIVATE .)
target_include_directories(quda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(quda SYSTEM PRIVATE ../include/externals)

target_link_libraries(quda PRIVATE $<BUILD_INTERFACE:Eigen>)
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
                                       $<INSTALL_INTERFACE:include/>)

target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)

target_include_directories(quda_cpp SYSTEM PRIVATE ../include/externals)

target_link_libraries(quda_cpp PRIVATE $<BUILD_INTERFACE:Eigen>)
target_compile_definitions(quda_cpp PRIVATE $<TARGET_PROPERTY:quda,COMPILE_DEFINITIONS>)
target_include_directories(quda_cpp PRIVATE $<TARGET_PROPERTY:quda,INCLUDE_DIRECTORIES>)
target_compile_options(quda_cpp PRIVATE $<TARGET_PROPERTY:quda,COMPILE_OPTIONS>)

add_subdirectory(targets/generic)
target_include_directories(quda PRIVATE ../include/targets/generic)
add_subdirectory(interface)

# ignore any unkown pragmas
target_compile_options(quda_cpp PUBLIC $<$<COMPILE_LANGUAGE:CXX>:
  $<$<CXX_COMPILER_ID:Clang>:-Wno-unknown-pragmas>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>
  >)

# add some deifnitions that cause issues with cmake 3.7 and nvcc only to cpp files
target_compile_definitions(quda_cpp PUBLIC -DQUDA_HASH="${HASH}")
if(GITVERSION)
  target_compile_definitions(quda_cpp PUBLIC -DGITVERSION="${GITVERSION}")
endif()

# some clang warnings should be warning even when turning warnings into errors
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(quda_cpp PUBLIC $<$<COMPILE_LANGUAGE:CXX>: -Wno-error=unused-function>)
  # this is a hack to get colored diagnostics back when using Ninja and clang
  if(CMAKE_GENERATOR MATCHES "Ninja")
    target_compile_options(quda PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
  endif()
endif()

target_link_libraries(quda PUBLIC Threads::Threads)

if(${QUDA_BUILD_NATIVE_LAPACK} STREQUAL "ON")
  target_compile_definitions(quda PRIVATE NATIVE_LAPACK_LIB)
endif()

if(${QUDA_BUILD_NATIVE_FFT} STREQUAL "ON")
  target_compile_definitions(quda PRIVATE NATIVE_FFT_LIB)
endif()

if(QUDA_BLOCKSOLVER)
  target_compile_definitions(quda PRIVATE BLOCKSOLVER)
endif()

if(QUDA_INTERFACE_QDP OR QUDA_INTERFACE_ALL)
  target_compile_definitions(quda PUBLIC BUILD_QDP_INTERFACE)
endif(QUDA_INTERFACE_QDP OR QUDA_INTERFACE_ALL)

if(QUDA_INTERFACE_MILC OR QUDA_INTERFACE_ALL)
  target_compile_definitions(quda PUBLIC BUILD_MILC_INTERFACE)
endif(QUDA_INTERFACE_MILC OR QUDA_INTERFACE_ALL)

if(QUDA_INTERFACE_CPS OR QUDA_INTERFACE_ALL)
  target_compile_definitions(quda PUBLIC BUILD_CPS_INTERFACE)
endif(QUDA_INTERFACE_CPS OR QUDA_INTERFACE_ALL)

if(QUDA_INTERFACE_QDPJIT OR QUDA_INTERFACE_ALL)
  target_compile_definitions(quda PUBLIC BUILD_QDPJIT_INTERFACE)
endif(QUDA_INTERFACE_QDPJIT OR QUDA_INTERFACE_ALL)

if(QUDA_INTERFACE_BQCD OR QUDA_INTERFACE_ALL)
  target_compile_definitions(quda PUBLIC BUILD_BQCD_INTERFACE)
endif(QUDA_INTERFACE_BQCD OR QUDA_INTERFACE_ALL)

if(QUDA_INTERFACE_TIFR OR QUDA_INTERFACE_ALL)
  target_compile_definitions(quda PUBLIC BUILD_TIFR_INTERFACE)
endif(QUDA_INTERFACE_TIFR OR QUDA_INTERFACE_ALL)

# MULTI GPU AND USQCD
if(QUDA_MPI OR QUDA_QMP)
  target_compile_definitions(quda PUBLIC MULTI_GPU)
endif()

if(QUDA_MPI)
  target_compile_definitions(quda PUBLIC MPI_COMMS)
  target_link_libraries(quda PUBLIC MPI::MPI_CXX)
endif()

if(QUDA_QMP)
  target_compile_definitions(quda PUBLIC QMP_COMMS)
  target_link_libraries(quda PUBLIC QMP::qmp)
  target_link_libraries(quda PUBLIC MPI::MPI_CXX)
endif()

if(QUDA_QIO)
  target_compile_definitions(quda PUBLIC HAVE_QIO)
  target_link_libraries(quda PUBLIC QIO::qio)
endif()

if(QUDA_QDPJIT)
  target_compile_definitions(quda PRIVATE USE_QDPJIT)
  # this avoids issues when qdpjit is build as a static library we should eventually switch to
  # target_link_libraries(quda PRIVATE QDPXX::qdp)
  target_include_directories(quda PRIVATE $<TARGET_PROPERTY:QDPXX::qdp,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

if(QUDA_ARPACK)
  target_compile_definitions(quda PRIVATE ARPACK_LIB)

  if(QUDA_ARPACK_LOGGING)
    # ARPACK-NG does not suppport logging - we must warn the user
    message(
      WARNING
      "Specifying QUDA_ARPACK_LOGGING with ARPACK-NG package will cause link failures. Please ensure that QUDA_ARPACK_LOGGING=OFF if downloading ARPACK-NG or using system installed ARPACK-NG"
    )
    target_compile_definitions(QUDA_ARPACK_LOGGING ON)
  endif(QUDA_ARPACK_LOGGING)

  if(QUDA_DOWNLOAD_ARPACK)
    target_link_libraries(quda PUBLIC $<BUILD_INTERFACE:ARPACK::arpack>)

    if(QUDA_MPI OR QUDA_QMP)
      target_link_libraries(quda PUBLIC $<BUILD_INTERFACE:ARPACK::parpack>)
    endif()
  else(QUDA_DOWNLOAD_ARPACK)
    target_link_libraries(quda INTERFACE ${ARPACK})

    if(QUDA_MPI OR QUDA_QMP)
      target_link_libraries(quda INTERFACE ${PARPACK} MPI::MPI_Fortran)
    endif()
  endif(QUDA_DOWNLOAD_ARPACK)
endif(QUDA_ARPACK)

if(QUDA_OPENBLAS)
  target_compile_definitions(quda PRIVATE OPENBLAS_LIB)

  if(QUDA_DOWNLOAD_OPENBLAS)
    target_link_libraries(quda PUBLIC openblas)
    target_link_libraries(quda_cpp PUBLIC openblas)
  else()
    target_link_libraries(quda INTERFACE ${OPENBLAS})
  endif()
endif()

if(QUDA_USE_EIGEN)
  target_compile_definitions(quda PRIVATE EIGEN)
endif()

if(QUDA_OPENMP)
  target_link_libraries(quda PUBLIC OpenMP::OpenMP_CXX)
endif()

# set which precisions to enable
target_compile_definitions(quda PUBLIC QUDA_PRECISION=${QUDA_PRECISION})
target_compile_definitions(quda PUBLIC QUDA_RECONSTRUCT=${QUDA_RECONSTRUCT})

if(QUDA_ENABLE_MMA)
  target_compile_definitions(quda PUBLIC QUDA_ENABLE_MMA)
endif()

if(QUDA_MULTIGRID_SETUP_USE_SMMA)
  target_compile_definitions(quda PUBLIC QUDA_MULTIGRID_SETUP_USE_SMMA)
endif()

if(QUDA_FAST_COMPILE_REDUCE)
  target_compile_definitions(quda PRIVATE QUDA_FAST_COMPILE_REDUCE)
endif()

if(QUDA_FAST_COMPILE_DSLASH)
  target_compile_definitions(quda PRIVATE QUDA_FAST_COMPILE_DSLASH)
endif()

if(QUDA_ALTERNATIVE_I_TO_F)
  target_compile_definitions(quda PRIVATE QUDA_ALTERNATIVE_I_TO_F)
endif()

if(QUDA_BACKWARDS)
  target_include_directories(quda_cpp SYSTEM PRIVATE ${backward-cpp_SOURCE_DIR})

  set_property(
    SOURCE comm_common.cpp
    APPEND
    PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(
    SOURCE comm_common.cpp
    APPEND
    PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
  target_link_libraries(quda PUBLIC ${BACKWARD_LIBRARIES})
endif()

configure_file(../include/quda_define.h.in ../include/quda_define.h @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/include/quda_define.h" DESTINATION include/)

# until we define an install step copy the include directory to the build directory
add_custom_command(
  TARGET quda
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)

# some hackery to prevent having old shared / static builds of quda messing with the current build
add_custom_command(
  TARGET quda
  PRE_LINK
  COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/libquda.a ${CMAKE_CURRENT_BINARY_DIR}/libquda.so)

install(
  TARGETS quda
  EXPORT QUDATargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

install(
  EXPORT QUDATargets
  FILE QUDATargets.cmake
  NAMESPACE QUDA::
  DESTINATION lib/cmake/QUDA)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/qudaConfigVersion.cmake"
  VERSION ${QUDA_VERSION}
  COMPATIBILITY AnyNewerVersion)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_SOURCE_DIR}/QUDAConfig.cmake.in QUDAConfig.cmake INSTALL_DESTINATION lib/cmake/QUDA)

write_basic_package_version_file(
  QUDAConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/QUDAConfigVersion.cmake ${CMAKE_CURRENT_BINARY_DIR}/QUDAConfig.cmake
	DESTINATION lib/cmake/QUDA)
