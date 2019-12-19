# all files for quda -- needs some cleanup
# cmake-format: off
set (QUDA_OBJS
  # cmake-format: sortable
  timer.cpp malloc.cpp
  interface_quda.cpp util_quda.cpp
  color_spinor_field.cpp color_spinor_util.cu
  cpu_color_spinor_field.cpp cuda_color_spinor_field.cpp
  lattice_field.cpp
  tune.cpp
  blas_quda.cu reduce_quda.cu
  random.cu
  comm_common.cpp ${COMM_OBJS} ${NUMA_AFFINITY_OBJS} ${QIO_UTIL}
  copy_color_spinor.cu spinor_noise.cu
  copy_color_spinor_dd.cu copy_color_spinor_ds.cu
  copy_color_spinor_dh.cu copy_color_spinor_dq.cu
  copy_color_spinor_ss.cu copy_color_spinor_sd.cu
  copy_color_spinor_sh.cu copy_color_spinor_sq.cu
  copy_color_spinor_hd.cu copy_color_spinor_hs.cu
  copy_color_spinor_hh.cu copy_color_spinor_hq.cu
  copy_color_spinor_qd.cu copy_color_spinor_qs.cu
  copy_color_spinor_qh.cu copy_color_spinor_qq.cu
  quda_cuda_api.cpp
  version.cpp )
# cmake-format: on

# split source into cu and cpp files
foreach(item ${QUDA_OBJS})
  string(REGEX MATCH ".+\\.cu$" item_match ${item})
  if(item_match)
    list(APPEND QUDA_CU_OBJS ${item})
  endif(item_match)
endforeach(item ${QUDA_OBJS})

list(REMOVE_ITEM QUDA_OBJS ${QUDA_CU_OBJS})

if(BUILD_FORTRAN_INTERFACE)
  list(APPEND QUDA_OBJS quda_fortran.F90)
  set_source_files_properties(quda_fortran.F90 PROPERTIES OBJECT_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/quda_fortran.mod)
endif()

# QUDA_CU_OBJS shoudl contain all cuda files now QUDA_OBJS all c, cpp, fortran sources

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
add_library(quda_cpp OBJECT ${QUDA_OBJS})

# add some deifnitions that cause issues with cmake 3.7 and nvcc only to cpp files
target_compile_definitions(quda_cpp PUBLIC -DQUDA_HASH="${HASH}")
if(GITVERSION)
  target_compile_definitions(quda_cpp PUBLIC -DGITVERSION="${GITVERSION}")
endif()

# make one library
if(QUDA_BUILD_SHAREDLIB)
  set_target_properties(quda_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  cuda_add_library(quda SHARED $<TARGET_OBJECTS:quda_cpp> ${QUDA_CU_OBJS})
else()
  cuda_add_library(quda STATIC $<TARGET_OBJECTS:quda_cpp> ${QUDA_CU_OBJS})
endif()

# include_directories
target_include_directories(quda SYSTEM PRIVATE ../include/externals)
target_include_directories(quda PRIVATE .)
target_include_directories(quda_cpp SYSTEM PRIVATE ../include/externals)
target_include_directories(quda_cpp PRIVATE .)

target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)

# propagate CXX flags to CUDA host compiler
if(${QUDA_PROPAGATE_CXX_FLAGS})
  target_compile_options(quda
                         PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: $<$<CONFIG:DEVEL>:-Xcompiler ${CMAKE_CXX_FLAGS_DEVEL}>
                                $<$<CONFIG:STRICT>:-Xcompiler ${CMAKE_CXX_FLAGS_STRICT}> $<$<CONFIG:RELEASE>:-Xcompiler
                                ${CMAKE_CXX_FLAGS_RELEASE}> $<$<CONFIG:DEBUG>:-Xcompiler ${CMAKE_CXX_FLAGS_DEBUG}>
                                $<$<CONFIG:HOSTDEBUG>:-Xcompiler ${CMAKE_CXX_FLAGS_HOSTDEBUG}>
                                $<$<CONFIG:DEVICEDEBUG>:-Xcompiler ${CMAKE_CXX_FLAGS_DEVICEDEBUG}> >)
endif()

# some clang warnings shouds be warning even when turning warnings into errors
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(quda_cpp
                         PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=unused-private-field -Wno-error=unused-function>)
  target_compile_options(quda
                         PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: "SHELL:-Xcompiler -Wno-error=unused-private-field"
                                "SHELL:-Xcompiler -Wno-error=unused-function" >)

  # this is a hack to get colored diagnostics back when using Ninja and clang
  if(CMAKE_GENERATOR MATCHES "Ninja")
    target_compile_options(quda PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
  endif()
endif()

target_link_libraries(quda INTERFACE ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIBS})

if(QUDA_MULTIGRID)
  target_link_libraries(quda PUBLIC ${CUDA_cublas_LIBRARY})
endif(QUDA_MULTIGRID)

if(QUDA_JITIFY)
  target_link_libraries(quda PUBLIC ${CUDA_nvrtc_LIBRARY})
  target_link_libraries(quda PUBLIC ${LIBDL_LIBRARIES})
endif()

if(QUDA_BACKWARDS)
  target_include_directories(quda_cpp SYSTEM PRIVATE ${backward-cpp_SOURCE_DIR})
  set_property(SOURCE comm_common.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE comm_common.cpp APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
  set_property(SOURCE malloc.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE malloc.cpp APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
  target_link_libraries(quda PUBLIC ${BACKWARD_LIBRARIES})
endif()

if(QUDA_QIO)
  if(QUDA_DOWNLOAD_USQCD AND NOT QIO_FOUND)
    add_dependencies(quda QIO)
    add_dependencies(quda_cpp QIO)
  endif()
  target_link_libraries(quda INTERFACE ${QUDA_QIO_LDFLAGS} ${QUDA_QIO_LIBS})
endif()

if(QUDA_QDPJIT)
  target_link_libraries(quda
                        INTERFACE ${QDP_LDFLAGS} ${QDP_LIB} ${QDP_LIBS} ${QIO_LIB} ${LIME_LIB} ${QUDA_QMP_LDFLAGS}
                                  ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_QMP)
  if(QUDA_DOWNLOAD_USQCD AND NOT QMP_FOUND)
    add_dependencies(quda QMP)
    add_dependencies(quda_cpp QMP)
  endif()
  target_link_libraries(quda INTERFACE ${QUDA_QMP_LDFLAGS} ${QUDA_QMP_LIBS} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MPI)
  target_link_libraries(quda INTERFACE ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MAGMA)
  target_link_libraries(quda PRIVATE ${MAGMA})
endif()

if(QUDA_ARPACK)
  if(QUDA_DOWNLOAD_ARPACK)
    target_link_libraries(quda PUBLIC arpack-ng)
    target_link_libraries(quda_cpp PUBLIC arpack-ng)
    if(QUDA_MPI OR QUDA_QMP)
      target_link_libraries(quda PUBLIC parpack-ng)
      target_link_libraries(quda_cpp PUBLIC parpack-ng)
    endif()
  else()
    target_link_libraries(quda INTERFACE ${ARPACK})
    if(QUDA_MPI OR QUDA_QMP)
      target_link_libraries(quda INTERFACE ${PARPACK} MPI::MPI_Fortran)
    endif()
  endif()
endif()

if(QUDA_NVML)
  target_link_libraries(quda PRIVATE ${NVML_LIBRARY})
endif()

if(QUDA_NUMA_NVML)
  target_link_libraries(quda PRIVATE ${NVML_LIBRARY})
endif()

# malloc.cpp uses both the driver and runtime api So we need to find the CUDA_CUDA_LIBRARY (driver api) or the stub
# version for cmake 3.8 and later this has been integrated into  FindCUDALibs.cmake
target_link_libraries(quda PUBLIC ${CUDA_cuda_LIBRARY})

# if we did not find Eigen but downloaded it we need to add it as dependency so the download is done first
if(QUDA_DOWNLOAD_EIGEN)
  add_dependencies(quda_cpp Eigen)
  add_dependencies(quda Eigen)
endif()

configure_file(../include/quda_define.h.in ../include/quda_define.h @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/include/quda_define.h" DESTINATION include/)

if(QUDA_JITIFY)
  configure_file(../include/jitify_options.hpp.in ../include/jitify_options.hpp)
  install(FILES "${CMAKE_BINARY_DIR}/include/jitify_options.hpp" DESTINATION include/)
endif()

# until we define an install step copy the include directory to the build directory
add_custom_command(TARGET quda POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)

# some hackery to prevent having old shared / static builds of quda messing with the current build
add_custom_command(TARGET quda PRE_LINK
                   COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/libquda.a
                                               ${CMAKE_CURRENT_BINARY_DIR}/libquda.so)

install(TARGETS quda EXPORT qudaTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib INCLUDES DESTINATION include)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/qudaConfigVersion.cmake"
                                 VERSION ${QUDA_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

export(EXPORT qudaTargets FILE "${CMAKE_CURRENT_BINARY_DIR}/qudaTargets.cmake" NAMESPACE quda::)
set(ConfigPackageLocation lib/cmake/quda/)
install(EXPORT qudaTargets NAMESPACE quda:: DESTINATION ${ConfigPackageLocation})

add_custom_target(mpi_nvtx ${PYTHON_EXECUTABLE} generate/wrap.py -g -o nvtx_pmpi.c generate/nvtx.w
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Generating mpi_nvtx wrapper")
