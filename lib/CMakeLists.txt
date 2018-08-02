# all files for quda -- needs some cleanup
set (QUDA_OBJS
  dirac_coarse.cpp dslash_coarse.cu coarse_op.cu coarsecoarse_op.cu
  coarse_op_preconditioned.cu staggered_coarse_op.cu
  multigrid.cpp transfer.cpp block_orthogonalize.cu inv_bicgstab_quda.cpp
  prolongator.cu restrictor.cu gauge_phase.cu timer.cpp malloc.cpp
  solver.cpp inv_bicgstab_quda.cpp inv_cg_quda.cpp inv_bicgstabl_quda.cpp
  inv_multi_cg_quda.cpp inv_eigcg_quda.cpp gauge_ape.cu
  gauge_stout.cu gauge_plaq.cu laplace.cu gauge_laplace.cpp
  inv_cg3_quda.cpp inv_cg3ne_quda.cpp inv_ca_gcr.cpp inv_ca_cg.cpp
  inv_gcr_quda.cpp inv_mr_quda.cpp inv_sd_quda.cpp inv_xsd_quda.cpp
  inv_pcg_quda.cpp inv_mre.cpp interface_quda.cpp util_quda.cpp
  color_spinor_field.cpp color_spinor_util.cu color_spinor_pack.cu
  color_spinor_wuppertal.cu covDev.cu gauge_covdev.cpp 
  cpu_color_spinor_field.cpp cuda_color_spinor_field.cu dirac.cpp
  clover_field.cpp lattice_field.cpp gauge_field.cpp
  cpu_gauge_field.cpp cuda_gauge_field.cu extract_gauge_ghost.cu
  extract_gauge_ghost_mg.cu max_gauge.cu gauge_update_quda.cu
  dirac_clover.cpp dirac_wilson.cpp dirac_staggered.cpp
  dirac_improved_staggered.cpp dirac_domain_wall.cpp
  dirac_domain_wall_4d.cpp dirac_mobius.cpp dirac_twisted_clover.cpp
  dirac_twisted_mass.cpp tune.cpp
  llfat_quda.cu gauge_force.cu gauge_random.cu
  field_strength_tensor.cu clover_quda.cu dslash_quda.cu
  dslash_wilson.cu dslash_clover.cu dslash_clover_asym.cu
  dslash_twisted_mass.cu dslash_ndeg_twisted_mass.cu
  dslash_twisted_clover.cu dslash_domain_wall.cu
  dslash_domain_wall_4d.cu dslash_mobius.cu dslash_staggered.cu
  dslash_improved_staggered.cu dslash_pack.cu blas_quda.cu
  multi_blas_quda.cu copy_quda.cu reduce_quda.cu
  multi_reduce_quda.cu
  comm_common.cpp ${COMM_OBJS} ${NUMA_AFFINITY_OBJS} ${QIO_UTIL}
  clover_deriv_quda.cu clover_invert.cu copy_gauge_extended.cu
  extract_gauge_ghost_extended.cu copy_color_spinor.cu spinor_noise.cu
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
  copy_gauge_half.cu copy_gauge.cu copy_gauge_mg.cu copy_clover.cu
  staggered_oprod.cu clover_trace_quda.cu ks_force_quda.cu
  hisq_paths_force_quda.cu
  unitarize_force_quda.cu unitarize_links_quda.cu milc_interface.cpp
  extended_color_spinor_utilities.cu eig_lanczos_quda.cpp
  ritz_quda.cpp eig_solver.cpp blas_cublas.cu blas_magma.cu
  inv_mpcg_quda.cpp inv_mpbicgstab_quda.cpp inv_gmresdr_quda.cpp
  pgauge_exchange.cu pgauge_init.cu pgauge_heatbath.cu random.cu
  gauge_fix_ovr_extra.cu gauge_fix_fft.cu gauge_fix_ovr.cu
  pgauge_det_trace.cu clover_outer_product.cu
  clover_sigma_outer_product.cu momentum.cu qcharge_quda.cu
  quda_cuda_api.cpp quda_arpack_interface.cpp deflation.cpp checksum.cu version.cpp )

## split source into cu and cpp files
FOREACH(item ${QUDA_OBJS})
  STRING(REGEX MATCH ".+\\.cu$" item_match ${item})
  IF(item_match)
    LIST(APPEND QUDA_CU_OBJS ${item})
  ENDIF(item_match)
ENDFOREACH(item ${QUDA_OBJS})

LIST(REMOVE_ITEM QUDA_OBJS ${QUDA_CU_OBJS})

if(BUILD_FORTRAN_INTERFACE)
  LIST(APPEND QUDA_OBJS quda_fortran.F90)
  set_source_files_properties(quda_fortran.F90 PROPERTIES OBJECT_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/quda_fortran.mod)
endif()

# QUDA_CU_OBJS shoudl contain all cuda files now
# QUDA_OBJS all c, cpp, fortran sources

# if we have a git version make version.cpp depend on git head so that it is rebuild if the git sha changed
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  if(GITVERSION)
    find_path(QUDA_GITDIR NAME HEAD PATHS ${CMAKE_SOURCE_DIR}/.git/logs NO_DEFAULT_PATH)
    if(QUDA_GITDIR)
      set_property(SOURCE version.cpp PROPERTY OBJECT_DEPENDS  ${QUDA_GITDIR}/HEAD)
    endif()
  endif()
  mark_as_advanced(QUDA_GITDIR)
endif()

# include_directories
include_directories(dslash_core)
include_directories(.)

# generate a cmake object library for all cpp files first
add_library(quda_cpp OBJECT ${QUDA_OBJS})

# add some deifnitions that cause issues with cmake 3.7 and nvcc only to cpp files
target_compile_definitions(quda_cpp PUBLIC -DQUDA_HASH="${HASH}")
IF(GITVERSION)
  target_compile_definitions(quda_cpp PUBLIC -DGITVERSION="${GITVERSION}")
ENDIF()

# make one library
if(QUDA_BUILD_SHAREDLIB)
    set_target_properties(quda_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    cuda_add_library(quda SHARED $<TARGET_OBJECTS:quda_cpp> ${QUDA_CU_OBJS} )
else()
  cuda_add_library(quda STATIC $<TARGET_OBJECTS:quda_cpp> ${QUDA_CU_OBJS})
endif()
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# propagate CXX flags to CUDA host compiler
if(USING_CUDA_LANG_SUPPORT AND ${QUDA_PROPAGATE_CXX_FLAGS})
  target_compile_options(quda PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    $<$<CONFIG:DEVEL>:-Xcompiler ${CMAKE_CXX_FLAGS_DEVEL}>
    $<$<CONFIG:STRICT>:-Xcompiler ${CMAKE_CXX_FLAGS_STRICT}>
    $<$<CONFIG:RELEASE>:-Xcompiler ${CMAKE_CXX_FLAGS_RELEASE}>
    $<$<CONFIG:DEBUG>:-Xcompiler ${CMAKE_CXX_FLAGS_DEBUG}>  
    $<$<CONFIG:HOSTDEBUG>:-Xcompiler ${CMAKE_CXX_FLAGS_HOSTDEBUG}>
    $<$<CONFIG:DEVICEDEBUG>:-Xcompiler ${CMAKE_CXX_FLAGS_DEVICEDEBUG}> 
    >)
endif()

# some clang warnings shouds be warning even when turning warnings into errors
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(quda_cpp PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=unused-private-field>)
# this is a hack to get colored diagnostics back when using Ninja and clang
    if(CMAKE_GENERATOR MATCHES "Ninja")
      target_compile_options(quda_cpp PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
    endif()
endif()

target_link_libraries(quda ${QUDA_SHAREDLIB_INTERFACE} ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIBS})

if(QUDA_QIO)
  target_link_libraries(quda ${QUDA_SHAREDLIB_INTERFACE} ${QIO_LIB} ${LIME_LIB})
endif()

if(QUDA_QDPJIT)
  target_link_libraries(quda ${QUDA_SHAREDLIB_INTERFACE} ${QDP_LDFLAGS} ${QDP_LIB} ${QDP_LIBS} ${QIO_LIB} ${LIME_LIB} ${QUDA_QMP_LDFLAGS} ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_QMP)
  target_link_libraries(quda ${QUDA_SHAREDLIB_INTERFACE} ${QUDA_QMP_LDFLAGS} ${QUDA_QMP_LIBS} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MPI)
  target_link_libraries(quda ${QUDA_SHAREDLIB_INTERFACE} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MAGMA)
  target_link_libraries(quda ${QUDA_SHAREDLIB_PRIVATE} ${MAGMA})
endif()

if(QUDA_ARPACK)
  target_link_libraries(quda ${QUDA_SHAREDLIB_PRIVATE} ${PARPACK_OR_ARPACK})
endif()

if(QUDA_NVML)
  target_link_libraries(quda ${QUDA_SHAREDLIB_PRIVATE} ${NVML_LIBRARY})
endif()

if(QUDA_NUMA_NVML)
  target_link_libraries(quda ${QUDA_SHAREDLIB_PRIVATE} ${NVML_LIBRARY})
endif()


# malloc.cpp uses both the driver and runtime api
# So we need to find the CUDA_CUDA_LIBRARY (driver api) or the stub version
# for cmake 3.8 and later this has been integrated into  FindCUDALibs.cmake 
if(${CMAKE_VERSION} VERSION_LESS 3.8)
  find_library(CUDA_cuda_LIBRARY cuda HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64 PATH_SUFFIXES stubs )
endif()
target_link_libraries(quda ${QUDA_SHAREDLIB_PRIVATE} ${CUDA_cuda_LIBRARY})

# if we did not find Eigen but downloaded it we need to add it as dependency so the download is done first
if (QUDA_DOWNLOAD_EIGEN)
  add_dependencies(quda_cpp Eigen)
  add_dependencies(quda Eigen)
endif()

# until we define an install step copy the include directory to the build directory
ADD_CUSTOM_COMMAND(TARGET quda POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
# some hackery to prevent having old shared / static builds of quda messing with the current build
ADD_CUSTOM_COMMAND(TARGET quda PRE_LINK COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/libquda.a ${CMAKE_CURRENT_BINARY_DIR}/libquda.so)

install(TARGETS quda EXPORT qudaTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/qudaConfigVersion.cmake"
  VERSION ${QUDA_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(EXPORT qudaTargets FILE "${CMAKE_CURRENT_BINARY_DIR}/qudaTargets.cmake" NAMESPACE quda::)
set(ConfigPackageLocation lib/cmake/quda/)
install(EXPORT qudaTargets NAMESPACE quda:: DESTINATION ${ConfigPackageLocation})


add_custom_target(gen ${PYTHON_EXECUTABLE} generate/gen.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating dslash_core files" )

add_custom_target(mpi_nvtx ${PYTHON_EXECUTABLE} generate/wrap.py -g -o nvtx_pmpi.c generate/nvtx.w
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating mpi_nvtx wrapper" )
