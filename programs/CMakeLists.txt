# all files for quda -- needs some cleanup
# cmake-format: off
set (PROGRAMS_OBJS
  # cmake-format: sortable
  stoch_laph_quark_smear_utils.cpp
  spin_dilute.cu
  evec_project.cu
  )
# cmake-format: on

# split source into cu and cpp files
foreach(item ${PROGRAMS_OBJS})
  string(REGEX MATCH ".+\\.cu$" item_match ${item})
  if(item_match)
    list(APPEND PROGRAMS_CU_OBJS ${item})
  endif(item_match)
endforeach(item ${PROGRAMS_OBJS})

list(REMOVE_ITEM PROGRAMS_OBJS ${PROGRAMS_CU_OBJS})

if(BUILD_FORTRAN_INTERFACE)
  list(APPEND PROGRAMS_OBJS quda_fortran.F90)
  set_source_files_properties(quda_fortran.F90 PROPERTIES OBJECT_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/quda_fortran.mod)
endif()

# PROGRAMS_CU_OBJS should contain all cuda files now PROGRAMS_OBJS all c, cpp, fortran sources

# generate a cmake object library for all cpp files first
add_library(programs_cpp OBJECT ${PROGRAMS_OBJS})

# add some definitions that cause issues with cmake 3.7 and nvcc only to cpp files
target_compile_definitions(programs_cpp PUBLIC -DQUDA_HASH="${HASH}")
if(GITVERSION)
  target_compile_definitions(programs_cpp PUBLIC -DGITVERSION="${GITVERSION}")
endif()

# make one library
if(PROGRAMS_BUILD_SHAREDLIB)
  set_target_properties(programs_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  cuda_add_library(quda_progs SHARED $<TARGET_OBJECTS:programs_cpp> ${PROGRAMS_CU_OBJS})
else()
  cuda_add_library(quda_progs STATIC $<TARGET_OBJECTS:programs_cpp> ${PROGRAMS_CU_OBJS})
endif()

# include_directories
target_include_directories(quda_progs SYSTEM PRIVATE ../include/externals)
target_include_directories(quda_progs PRIVATE .)
target_include_directories(programs_cpp SYSTEM PRIVATE ../include/externals)
target_include_directories(programs_cpp PRIVATE .)

target_include_directories(quda_progs PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(quda_progs PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/.> $<INSTALL_INTERFACE:.>)

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
  target_include_directories(programs_cpp SYSTEM PRIVATE ${backward-cpp_SOURCE_DIR})
  set_property(SOURCE comm_common.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE comm_common.cpp APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
  set_property(SOURCE malloc.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE malloc.cpp APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
  target_link_libraries(quda_progs PUBLIC ${BACKWARD_LIBRARIES})
endif()

if(QUDA_QIO)
  if(QUDA_DOWNLOAD_USQCD AND NOT QIO_FOUND)
    add_dependencies(quda_progs QIO)
    add_dependencies(programs_cpp QIO)
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
    add_dependencies(programs_cpp QMP)
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
    target_link_libraries(programs_cpp PUBLIC arpack-ng)
    if(QUDA_MPI OR QUDA_QMP)
      target_link_libraries(quda_progs PUBLIC parpack-ng)
      target_link_libraries(programs_cpp PUBLIC parpack-ng)
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
target_link_libraries(quda_progs PUBLIC ${CUDA_cuda_LIBRARY})

# if we did not find Eigen but downloaded it we need to add it as dependency so the download is done first
if(QUDA_DOWNLOAD_EIGEN)
  add_dependencies(programs_cpp Eigen)
  add_dependencies(quda_progs Eigen)
endif()

if(QUDA_JITIFY)
  configure_file(../include/jitify_options.hpp.in ../include/jitify_options.hpp)
  install(FILES "${CMAKE_BINARY_DIR}/include/jitify_options.hpp" DESTINATION include/)
endif()

# Compile and link the executable
include_directories( ../utils)

set(PROGRAMS_LIBS quda quda_progs quda_utils)

cuda_add_executable(stoch_laph_quark_smear stoch_laph_quark_smear_driver.cpp)
target_link_libraries(stoch_laph_quark_smear ${PROGRAMS_LIBS})
