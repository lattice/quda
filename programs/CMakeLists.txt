# all files for quda -- needs some cleanup
# cmake-format: off
set (PROGRAMS_OBJS
  # cmake-format: sortable
  stoch_laph_quark_smear_utils.cpp
  spin_dilute.cu
  evec_project.cu
  )
# cmake-format: on

#add_library(quda_progs MODULE ${PROGRAMS_OBJS})
#add_library(quda_progs OBJECT ${PROGRAMS_OBJS})
#add_library(quda_progs SHARED ${PROGRAMS_OBJS}${CUDA_cuda_LIBRARY})
add_library(quda_progs STATIC ${PROGRAMS_OBJS})

if(BUILD_FORTRAN_INTERFACE)
  list(APPEND PROGRAMS_OBJS quda_fortran.F90)
  set_source_files_properties(quda_fortran.F90 PROPERTIES OBJECT_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/quda_fortran.mod)
endif()

# if we have a git version make version.cpp depend on git head so that it is rebuild if the git sha changed
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  if(GITVERSION)
    find_path(QUDA_GITDIR NAME HEAD PATHS ${CMAKE_SOURCE_DIR}/.git/logs NO_DEFAULT_PATH)
    include(AddFileDependencies)
    if(QUDA_GITDIR)
      add_file_dependencies(version.cpp ${QUDA_GITDIR}/HEAD)
    endif()
  endif()
  mark_as_advanced(QUDA_GITDIR)
endif()

# include_directories
target_include_directories(quda_progs SYSTEM PRIVATE ../include/externals)
target_include_directories(quda_progs PRIVATE .)

if(QUDA_MULTIGRID)
  target_link_libraries(quda_progs PUBLIC ${CUDA_cublas_LIBRARY})
endif(QUDA_MULTIGRID)

if(QUDA_GAUGE_ALG)
  add_definitions(-DGPU_GAUGE_ALG)
  add_definitions(-DGPU_GAUGE_TOOLS)
  add_definitions(-DGPU_UNITARIZE)
  target_link_libraries(quda_progs PUBLIC ${CUDA_cufft_LIBRARY})
endif(QUDA_GAUGE_ALG)

if(QUDA_JITIFY)
  target_link_libraries(quda_progs PUBLIC ${CUDA_nvrtc_LIBRARY})
  target_link_libraries(quda_progs PUBLIC ${LIBDL_LIBRARIES})
endif()

if(QUDA_NVTX)
  if(NOT NVTX3)
    target_link_libraries(quda_progs PUBLIC ${CUDA_nvToolsExt_LIBRARY})
  endif()
endif(QUDA_NVTX)

if(QUDA_BACKWARDS)
  set_property(SOURCE comm_common.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE comm_common.cpp APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
  set_property(SOURCE malloc.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE malloc.cpp APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
  target_link_libraries(quda_progs PUBLIC ${BACKWARD_LIBRARIES})
endif()

if(QUDA_QIO)
  if(QUDA_DOWNLOAD_USQCD AND NOT QIO_FOUND)
    add_dependencies(quda_progs QIO)
  endif()
  target_link_libraries(quda_progs INTERFACE ${QUDA_QIO_LDFLAGS} ${QUDA_QIO_LIBS})
endif()

if(QUDA_QDPJIT)
  target_link_libraries(quda_progs
                        INTERFACE ${QDP_LDFLAGS} ${QDP_LIB} ${QDP_LIBS} ${QIO_LIB} ${LIME_LIB} ${QUDA_QMP_LDFLAGS}
                                  ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_QMP)
  if(QUDA_DOWNLOAD_USQCD AND NOT QMP_FOUND)
    add_dependencies(quda_progs QMP)
  endif()
  target_link_libraries(quda_progs INTERFACE ${QUDA_QMP_LDFLAGS} ${QUDA_QMP_LIBS} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MPI)
  target_link_libraries(quda_progs INTERFACE ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MAGMA)
  target_link_libraries(quda_progs PRIVATE ${MAGMA})
endif()

if(QUDA_ARPACK)
  if(QUDA_DOWNLOAD_ARPACK)
    target_link_libraries(quda_progs PUBLIC arpack-ng)
    if(QUDA_MPI OR QUDA_QMP)
      target_link_libraries(quda_progs PUBLIC parpack-ng)
    endif()
  else()
    target_link_libraries(quda_progs INTERFACE ${ARPACK})
    if(QUDA_MPI OR QUDA_QMP)
      target_link_libraries(quda_progs INTERFACE ${PARPACK} MPI::MPI_Fortran)
    endif()
  endif()
endif()

if(QUDA_NVML)
  target_link_libraries(quda_progs PRIVATE ${NVML_LIBRARY})
endif()

if(QUDA_NUMA_NVML)
  target_link_libraries(quda_progs PRIVATE ${NVML_LIBRARY})
endif()


# malloc.cpp uses both the driver and runtime api So we need to find the CUDA_CUDA_LIBRARY (driver api) or the stub
# version for cmake 3.8 and later this has been integrated into  FindCUDALibs.cmake
target_link_libraries(quda_progs INTERFACE ${CUDA_cuda_LIBRARY})

# if we did not find Eigen but downloaded it we need to add it as dependency so the download is done first
if(QUDA_DOWNLOAD_EIGEN)
  add_dependencies(quda_progs Eigen)
endif()

if(QUDA_JITIFY)
  configure_file(../include/jitify_options.hpp.in ../include/jitify_options.hpp)
  install(FILES "${CMAKE_BINARY_DIR}/include/jitify_options.hpp" DESTINATION include/)
endif()

# Compile and link the executable
include_directories( ../utils)
target_include_directories(quda_progs PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)


set(PROGRAMS_LIBS quda quda_utils quda_progs)

add_executable(stoch_laph_quark_smear stoch_laph_quark_smear_driver.cpp)
target_link_libraries(stoch_laph_quark_smear ${PROGRAMS_LIBS})
