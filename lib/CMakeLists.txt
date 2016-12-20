# all files for quda -- needs some cleanup
set (QUDA_OBJS
  dirac_coarse.cpp dslash_coarse.cu coarse_op.cu coarsecoarse_op.cu
  multigrid.cpp transfer.cpp transfer_util.cu inv_bicgstab_quda.cpp
  prolongator.cu restrictor.cu gauge_phase.cu timer.cpp malloc.cpp
  solver.cpp inv_bicgstab_quda.cpp inv_cg_quda.cpp inv_bicgstabl_quda.cpp
  inv_multi_cg_quda.cpp inv_eigcg_quda.cpp gauge_ape.cu
  gauge_stout.cu gauge_plaq.cu
  inv_gcr_quda.cpp inv_mr_quda.cpp inv_sd_quda.cpp inv_xsd_quda.cpp
  inv_pcg_quda.cpp inv_mre.cpp interface_quda.cpp util_quda.cpp
  color_spinor_field.cpp color_spinor_util.cu color_spinor_pack.cu
  cpu_color_spinor_field.cpp cuda_color_spinor_field.cu dirac.cpp
  clover_field.cpp covd.cpp lattice_field.cpp gauge_field.cpp
  cpu_gauge_field.cpp cuda_gauge_field.cu extract_gauge_ghost.cu
  extract_gauge_ghost_mg.cu max_gauge.cu gauge_update_quda.cu
  dirac_clover.cpp dirac_wilson.cpp dirac_staggered.cpp
  dirac_improved_staggered.cpp dirac_domain_wall.cpp
  dirac_domain_wall_4d.cpp dirac_mobius.cpp dirac_twisted_clover.cpp
  dirac_twisted_mass.cpp tune.cpp fat_force_quda.cpp
  llfat_quda_itf.cpp llfat_quda.cu gauge_force.cu
  field_strength_tensor.cu clover_quda.cu dslash_quda.cu covDev.cu
  dslash_wilson.cu dslash_clover.cu dslash_clover_asym.cu
  dslash_twisted_mass.cu dslash_ndeg_twisted_mass.cu
  dslash_twisted_clover.cu dslash_domain_wall.cu
  dslash_domain_wall_4d.cu dslash_mobius.cu dslash_staggered.cu
  dslash_improved_staggered.cu dslash_pack.cu blas_quda.cu
  multi_blas_quda.cu copy_quda.cu reduce_quda.cu
  multi_reduce_quda.cu face_buffer.cpp face_gauge.cpp
  comm_common.cpp ${COMM_OBJS} ${NUMA_AFFINITY_OBJS} ${QIO_UTIL}
  clover_deriv_quda.cu clover_invert.cu copy_gauge_extended.cu
  extract_gauge_ghost_extended.cu copy_color_spinor.cu
  copy_color_spinor_dd.cu copy_color_spinor_ds.cu
  copy_color_spinor_dh.cu copy_color_spinor_ss.cu
  copy_color_spinor_sd.cu copy_color_spinor_sh.cu
  copy_color_spinor_hd.cu copy_color_spinor_hs.cu
  copy_color_spinor_hh.cu copy_color_spinor_mg_dd.cu
  copy_color_spinor_mg_ds.cu copy_color_spinor_mg_sd.cu
  copy_color_spinor_mg_ss.cu copy_gauge_double.cu copy_gauge_single.cu
  copy_gauge_half.cu copy_gauge.cu copy_gauge_mg.cu copy_clover.cu
  staggered_oprod.cu clover_trace_quda.cu ks_force_quda.cu
  hisq_paths_force_quda.cu fermion_force_quda.cu
  unitarize_force_quda.cu unitarize_links_quda.cu milc_interface.cpp
  extended_color_spinor_utilities.cu eig_lanczos_quda.cpp
  ritz_quda.cpp eig_solver.cpp blas_magma.cu misc_helpers.cu
  inv_mpcg_quda.cpp inv_mpbicgstab_quda.cpp inv_gmresdr_quda.cpp
  pgauge_exchange.cu pgauge_init.cu pgauge_heatbath.cu random.cu
  gauge_fix_ovr_extra.cu gauge_fix_fft.cu gauge_fix_ovr.cu
  pgauge_det_trace.cu clover_outer_product.cu
  clover_sigma_outer_product.cu momentum.cu qcharge_quda.cu
  quda_memcpy.cpp version.cpp )

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
endif()

# QUDA_CU_OBJS shoudl contain all cuda files now
# QUDA_OBJS all c, cpp, fortran sources

# if we have a git version make version.cpp depend on git head so that it is rebuild if the git sha changed
if(GITVERSION)
  find_file(QUDA_GITDIR NAMES .git PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
  set_property(SOURCE version.cpp PROPERTY OBJECT_DEPENDS  ${QUDA_GITDIR}/logs/HEAD)
endif()
mark_as_advanced(QUDA_GITDIR)



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
    add_library(quda SHARED $<TARGET_OBJECTS:quda_cpp> ${QUDA_CU_OBJS} )
else()
  add_library(quda STATIC $<TARGET_OBJECTS:quda_cpp> ${QUDA_CU_OBJS})
endif()

# malloc.cpp uses both the driver and runtime api
target_link_libraries(quda LINK_PRIVATE cuda)

# until we define an install step copy the include directory to the build directory
ADD_CUSTOM_COMMAND(TARGET quda POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)


add_custom_target(gen ${PYTHON_EXECUTABLE} generate/gen.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating dslash_core files" )

add_custom_target(mpi_nvtx ${PYTHON_EXECUTABLE} generate/wrap.py -g -o nvtx_pmpi.c generate/nvtx.w
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating mpi_nvtx wrapper" )
