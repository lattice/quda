# all files for quda -- needs some cleanup
#  dirac_coarse.cpp dslash_coarse.cu coarse_op.cu coarsecoarse_op.cu multigrid.cpp transfer.cpp transfer_util.cpp prolongator.cu restrictor.cu
set (QUDA_OBJS
  gauge_phase.cu timer.cpp malloc.cpp
  solver.cpp inv_cg_quda.cpp 
  gauge_stout.cu gauge_plaq.cu
  inv_gcr_quda.cpp inv_mr_quda.cpp 
  interface_quda.cpp util_quda.cpp
  color_spinor_field.cpp color_spinor_util.cu color_spinor_pack.cu
  cpu_color_spinor_field.cpp cuda_color_spinor_field.cu dirac.cpp
  lattice_field.cpp gauge_field.cpp
  cpu_gauge_field.cpp cuda_gauge_field.cu extract_gauge_ghost.cu
  extract_gauge_ghost_mg.cu
  dirac_wilson.cpp dirac_staggered.cpp
  dirac_improved_staggered.cpp dirac_domain_wall.cpp
  dirac_domain_wall_4d.cpp tune.cpp dslash_quda.cu 
  dslash_wilson.cu dslash_domain_wall.cu
  dslash_domain_wall_4d.cu dslash_staggered.cu
  dslash_improved_staggered.cu dslash_pack.cu blas_quda.cu
  multi_blas_quda.cu copy_quda.cu reduce_quda.cu
  multi_reduce_quda.cu face_buffer.cpp face_gauge.cpp
  comm_common.cpp ${COMM_OBJS} ${NUMA_AFFINITY_OBJS} ${QIO_UTIL}
  copy_gauge_extended.cu 
  extract_gauge_ghost_extended.cu copy_color_spinor.cu
  copy_color_spinor_dd.cu copy_color_spinor_ds.cu
  copy_color_spinor_dh.cu copy_color_spinor_ss.cu
  copy_color_spinor_sd.cu copy_color_spinor_sh.cu
  copy_color_spinor_hd.cu copy_color_spinor_hs.cu
  copy_color_spinor_hh.cu copy_color_spinor_mg_dd.cu
  copy_color_spinor_mg_ds.cu copy_color_spinor_mg_sd.cu
  copy_color_spinor_mg_ss.cu copy_gauge_double.cu copy_gauge_single.cu
  copy_gauge_half.cu copy_gauge.cu copy_gauge_mg.cu 
  staggered_oprod.cu staggered_oprod_new.cu
  extended_color_spinor_utilities.cu 
  misc_helpers.cu
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
  set_source_files_properties(quda_fortran.F90 PROPERTIES OBJECT_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/quda_fortran.mod)
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
    cuda_add_library(quda SHARED $<TARGET_OBJECTS:quda_cpp> ${QUDA_CU_OBJS} )
else()
  cuda_add_library(quda STATIC $<TARGET_OBJECTS:quda_cpp> ${QUDA_CU_OBJS})
endif()

# until we define an install step copy the include directory to the build directory
ADD_CUSTOM_COMMAND(TARGET quda POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)


add_custom_target(gen ${PYTHON_EXECUTABLE} generate/gen.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating dslash_core files" )

add_custom_target(mpi_nvtx ${PYTHON_EXECUTABLE} generate/wrap.py -g -o nvtx_pmpi.c generate/nvtx.w
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating mpi_nvtx wrapper" )
