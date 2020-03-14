include_directories( ../utils)

# enable programs build a common library for all programs
if(QUDA_QDPJIT)
  target_link_libraries(quda
    INTERFACE ${QDP_LDFLAGS} ${QDP_LIB} ${QDP_LIBS} ${QIO_LIB} ${LIME_LIB} ${QUDA_QMP_LDFLAGS}
    ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

# all files for quda -- needs some cleanup
# cmake-format: off
set (PROGRAMS_OBJS
  # cmake-format: sortable
  stoch_laph_quark_smear.cpp
  test.cu
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

# PROGRAMS_CU_OBJS should contain all cuda files now QUDA_OBJS all c, cpp, fortran sources

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

# generate a cmake object library for all cpp files first
add_library(programs_cpp OBJECT ${PROGRAMS_OBJS})

# add some deifnitions that cause issues with cmake 3.7 and nvcc only to cpp files
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

# propagate CXX flags to CUDA host compiler
if(${QUDA_PROPAGATE_CXX_FLAGS})
  target_compile_options(quda_progs
                         PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: $<$<CONFIG:DEVEL>:-Xcompiler ${CMAKE_CXX_FLAGS_DEVEL}>
                                $<$<CONFIG:STRICT>:-Xcompiler ${CMAKE_CXX_FLAGS_STRICT}> $<$<CONFIG:RELEASE>:-Xcompiler
                                ${CMAKE_CXX_FLAGS_RELEASE}> $<$<CONFIG:DEBUG>:-Xcompiler ${CMAKE_CXX_FLAGS_DEBUG}>
                                $<$<CONFIG:HOSTDEBUG>:-Xcompiler ${CMAKE_CXX_FLAGS_HOSTDEBUG}>
                                $<$<CONFIG:DEVICEDEBUG>:-Xcompiler ${CMAKE_CXX_FLAGS_DEVICEDEBUG}> >)
endif()

# some clang warnings should be warning even when turning warnings into errors
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(programs_cpp
                         PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=unused-private-field -Wno-error=unused-function>)
  target_compile_options(quda_progs
                         PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: "SHELL:-Xcompiler -Wno-error=unused-private-field"
                                "SHELL:-Xcompiler -Wno-error=unused-function" >)

  # this is a hack to get colored diagnostics back when using Ninja and clang
  if(CMAKE_GENERATOR MATCHES "Ninja")
    target_compile_options(quda PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
  endif()
endif()

target_link_libraries(quda_progs INTERFACE ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIBS})

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

configure_file(../include/quda_define.h.in ../include/quda_define.h @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/include/quda_define.h" DESTINATION include/)

if(QUDA_JITIFY)
  configure_file(../include/jitify_options.hpp.in ../include/jitify_options.hpp)
  install(FILES "${CMAKE_BINARY_DIR}/include/jitify_options.hpp" DESTINATION include/)
endif()

# until we define an install step copy the include directory to the build directory
add_custom_command(TARGET quda_progs POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)

# some hackery to prevent having old shared / static builds of quda messing with the current build
add_custom_command(TARGET quda_progs PRE_LINK
                   COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/libquda.a
                                               ${CMAKE_CURRENT_BINARY_DIR}/libquda.so)

install(TARGETS quda_progs EXPORT qudaProgsTargets LIBRARY DESTINATION programs ARCHIVE DESTINATION programs INCLUDES DESTINATION include)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/qudaConfigVersion.cmake"
                                 VERSION ${QUDA_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

export(EXPORT qudaProgsTargets FILE "${CMAKE_CURRENT_BINARY_DIR}/qudaProgsTargets.cmake" NAMESPACE quda::)
set(ConfigPackageLocation programs/cmake/quda_progs/)
install(EXPORT qudaProgsTargets NAMESPACE quda:: DESTINATION ${ConfigPackageLocation})

add_custom_target(mpi_nvtx_progs ${PYTHON_EXECUTABLE} generate/wrap.py -g -o nvtx_pmpi.c generate/nvtx.w
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Generating mpi_nvtx_progs wrapper")

set(PROGRAMS_LIBS quda quda_utils quda_progs)

cuda_add_executable(stoch_laph_quark_smear stoch_laph_quark_smear.cpp)
target_link_libraries(stoch_laph_quark_smear ${PROGRAMS_LIBS})
