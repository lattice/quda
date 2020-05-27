# all files for quda -- needs some cleanup
# cmake-format: off

# QUDA_HASH for tunecache
if(NOT GITVERSION)
  set(GITVERSION ${PROJECT_VERSION})
endif()
set(HASH cpu_arch=${CPU_ARCH},gpu_arch=${QUDA_GPU_ARCH},cuda_version=${CMAKE_CUDA_COMPILER_VERSION})

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
  vector_io.cpp
  timer.cpp
  interface_quda.cpp util_quda.cpp
  color_spinor_field.cpp color_spinor_util.cu color_spinor_pack.cu
  cpu_color_spinor_field.cpp cuda_color_spinor_field.cpp
  lattice_field.cpp
  blas_quda.cu multi_blas_quda.cu reduce_quda.cu
  multi_reduce_quda.cu
  comm_common.cpp 
  copy_color_spinor.cu
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
  extended_color_spinor_utilities.cu
  random.cu
  quda_cuda_api.cpp checksum.cu
  instantiate.cpp version.cpp )
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
target_sources(quda_cpp PRIVATE $<IF:$<BOOL:${QUDA_MPI}>,comm_mpi.cpp,$<IF:$<BOOL:${QUDA_QMP}>,comm_qmp.cpp,comm_single.cpp>>)

target_sources(quda_cpp PRIVATE $<$<BOOL:${QUDA_QIO}>:qio_field.cpp layout_hyper.cpp>)



# add some deifnitions that cause issues with cmake 3.7 and nvcc only to cpp files
target_compile_definitions(quda_cpp PUBLIC -DQUDA_HASH="${HASH}")
if(GITVERSION)
  target_compile_definitions(quda_cpp PUBLIC -DGITVERSION="${GITVERSION}")
endif()


# make one library
if(QUDA_BUILD_SHAREDLIB)
  set_target_properties(quda_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  add_library(quda SHARED $<TARGET_OBJECTS:quda_cpp> ${QUDA_CU_OBJS})
  if(CUDAToolkit_FOUND)
    target_link_libraries(quda INTERFACE CUDA::cudart_static)
  endif()
else()
  add_library(quda STATIC $<TARGET_OBJECTS:quda_cpp> ${QUDA_CU_OBJS})
endif()

# malloc.cpp uses both the driver and runtime api So we need to find the CUDA_CUDA_LIBRARY (driver api) or the stub
# version for cmake 3.8 and later this has been integrated into  FindCUDALibs.cmake
target_link_libraries(quda PUBLIC ${CUDA_cuda_driver_LIBRARY})

# set up QUDA compile options
target_compile_definitions(
  quda
  PRIVATE
  $<$<CONFIG:DEVEL>:DEVEL>
  $<$<CONFIG:HOSTDEBUG>:HOST_DEBUG>
  $<$<CONFIG:DEVICEDEBUG>:DEVICE_DEBUG>
  $<$<CONFIG:DEBUG>:HOST_DEBUGDEVICE_DEBUG>
  $<$<CONFIG:SANITIZE>:HOST_DEBUG>)

target_compile_options(quda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -ftz=true -prec-div=false -prec-sqrt=false>)
target_compile_options(quda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Wno-deprecated-gpu-targets -arch=${QUDA_GPU_ARCH}
                                    --expt-relaxed-constexpr>)
if(QUDA_VERBOSE_BUILD)
  target_compile_options(quda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif(QUDA_VERBOSE_BUILD)

# workaround for 10.2
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "10.2" AND ${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "10.3")
  target_compile_options(quda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL: -Xcicc \"--Xllc -dag-vectorize-ops=1\" " >)
endif()

# include_directories
target_include_directories(quda SYSTEM PRIVATE ../include/externals)
target_include_directories(quda PRIVATE .)
target_include_directories(quda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(quda SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
                                       $<INSTALL_INTERFACE:include/>)

target_include_directories(quda SYSTEM PRIVATE ${EIGEN_INCLUDE_DIRS})
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)

target_compile_definitions(quda_cpp PRIVATE $<TARGET_PROPERTY:quda,COMPILE_DEFINITIONS>)
target_include_directories(quda_cpp PRIVATE $<TARGET_PROPERTY:quda,INCLUDE_DIRECTORIES>)
target_compile_options(quda_cpp PRIVATE $<TARGET_PROPERTY:quda,COMPILE_OPTIONS>)

# add target specific files
if(${QUDA_TARGET_TYPE} STREQUAL "CUDA")
  add_subdirectory(targets/cuda)
endif()
if(${QUDA_TARGET_TYPE} STREQUAL "HIP")
  #set(QUDA_TARGET_DIR "hip")
endif()

target_sources(quda PRIVATE $<TARGET_OBJECTS:quda_target>)

# propagate CXX flags to CUDA host compiler
if(${QUDA_PROPAGATE_CXX_FLAGS})
  target_compile_options(
    quda
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
           $<$<CONFIG:DEVEL>:-Xcompiler
           ${CMAKE_CXX_FLAGS_DEVEL}>
           $<$<CONFIG:STRICT>:-Xcompiler
           ${CMAKE_CXX_FLAGS_STRICT}>
           $<$<CONFIG:RELEASE>:-Xcompiler
           ${CMAKE_CXX_FLAGS_RELEASE}>
           $<$<CONFIG:DEBUG>:-Xcompiler
           ${CMAKE_CXX_FLAGS_DEBUG}>
           $<$<CONFIG:HOSTDEBUG>:-Xcompiler
           ${CMAKE_CXX_FLAGS_HOSTDEBUG}>
           $<$<CONFIG:DEVICEDEBUG>:-Xcompiler
           ${CMAKE_CXX_FLAGS_DEVICEDEBUG}>
           >)
endif()

# some clang warnings should be warning even when turning warnings into errors
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(quda_cpp PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=unused-private-field
                                         -Wno-error=unused-function>)
  target_compile_options(quda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: "SHELL:-Xcompiler -Wno-error=unused-private-field"
                                     "SHELL:-Xcompiler -Wno-error=unused-function" >)

  # this is a hack to get colored diagnostics back when using Ninja and clang
  if(CMAKE_GENERATOR MATCHES "Ninja")
    target_compile_options(quda PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
  endif()
endif()

# QUDA FEATURES
if(QUDA_DIRAC_WILSON)
  target_compile_definitions(quda PUBLIC GPU_WILSON_DIRAC)
endif(QUDA_DIRAC_WILSON)

if(QUDA_DIRAC_DOMAIN_WALL)
  target_compile_definitions(quda PUBLIC GPU_DOMAIN_WALL_DIRAC)
endif(QUDA_DIRAC_DOMAIN_WALL)

if(QUDA_DIRAC_STAGGERED)
  target_compile_definitions(quda PUBLIC GPU_STAGGERED_DIRAC GPU_FATLINK GPU_UNITARIZE)
endif(QUDA_DIRAC_STAGGERED)

if(QUDA_DIRAC_CLOVER)
  target_compile_definitions(quda PUBLIC GPU_CLOVER_DIRAC GPU_WILSON_DIRAC GPU_GAUGE_TOOLS)
endif(QUDA_DIRAC_CLOVER)

if(QUDA_DIRAC_TWISTED_MASS)
  target_compile_definitions(quda PUBLIC GPU_TWISTED_MASS_DIRAC GPU_WILSON_DIRAC)
endif(QUDA_DIRAC_TWISTED_MASS)

if(QUDA_DIRAC_TWISTED_CLOVER)
  target_compile_definitions(
    quda
    PUBLIC
    GPU_TWISTED_CLOVER_DIRAC
    GPU_CLOVER_DIRAC
    GPU_TWISTED_MASS_DIRAC
    GPU_WILSON_DIRAC
    GPU_GAUGE_TOOLS)
endif(QUDA_DIRAC_TWISTED_CLOVER)

if(QUDA_DIRAC_CLOVER_HASENBUSCH)
  target_compile_definitions(
    quda
    PUBLIC
    GPU_CLOVER_HASENBUSCH_TWIST
    GPU_TWISTED_CLOVER_DIRAC
    GPU_CLOVER_DIRAC
    GPU_WILSON_DIRAC
    GPU_GAUGE_TOOLS)
endif(QUDA_DIRAC_CLOVER_HASENBUSCH)

if(QUDA_DIRAC_NDEG_TWISTED_MASS)
  target_compile_definitions(quda PUBLIC GPU_NDEG_TWISTED_MASS_DIRAC)
endif(QUDA_DIRAC_NDEG_TWISTED_MASS)

if(QUDA_MULTIGRID)
  target_link_libraries(quda PUBLIC ${CUDA_cublas_LIBRARY})
  target_compile_definitions(quda PRIVATE CUBLAS_LIB)
  target_compile_definitions(quda PRIVATE GPU_MULTIGRID)
endif(QUDA_MULTIGRID)

if(QUDA_GAUGE_ALG)
  target_compile_definitions(quda PUBLIC GPU_GAUGE_ALG GPU_GAUGE_TOOLS GPU_UNITARIZE)
  target_link_libraries(quda PUBLIC ${CUDA_cufft_LIBRARY})
endif(QUDA_GAUGE_ALG)

if(QUDA_SSTEP)
  target_compile_definitions(quda PRIVATE SSTEP)
endif()

if(QUDA_BLOCKSOLVER)
  target_compile_definitions(quda PRIVATE BLOCKSOLVER)
endif()

if(QUDA_FORCE_GAUGE)
  target_compile_definitions(quda PUBLIC GPU_GAUGE_FORCE GPU_GAUGE_TOOLS)
endif(QUDA_FORCE_GAUGE)

if(QUDA_FORCE_HISQ)
  target_compile_definitions(quda PUBLIC GPU_HISQ_FORCE GPU_STAGGERED_OPROD GPU_GAUGE_TOOLS)
endif(QUDA_FORCE_HISQ)

if(QUDA_GAUGE_TOOLS)
  target_compile_definitions(quda PUBLIC GPU_GAUGE_TOOLS)
endif(QUDA_GAUGE_TOOLS)

if(QUDA_GAUGE_ALG)
  target_compile_definitions(quda PUBLIC GPU_GAUGE_ALG GPU_GAUGE_TOOLS GPU_UNITARIZE)
endif(QUDA_GAUGE_ALG)

if(QUDA_INTERFACE_QDP)
  target_compile_definitions(quda PUBLIC BUILD_QDP_INTERFACE)
endif(QUDA_INTERFACE_QDP)

if(QUDA_INTERFACE_MILC)
  target_compile_definitions(quda PUBLIC BUILD_MILC_INTERFACE)
endif(QUDA_INTERFACE_MILC)

if(QUDA_INTERFACE_CPS)
  target_compile_definitions(quda PUBLIC BUILD_CPS_INTERFACE)
endif(QUDA_INTERFACE_CPS)

if(QUDA_INTERFACE_QDPJIT)
  target_compile_definitions(quda PUBLIC BUILD_QDPJIT_INTERFACE)
endif(QUDA_INTERFACE_QDPJIT)

if(QUDA_INTERFACE_BQCD)
  target_compile_definitions(quda PUBLIC BUILD_BQCD_INTERFACE)
endif(QUDA_INTERFACE_BQCD)

if(QUDA_INTERFACE_TIFR)
  target_compile_definitions(quda PUBLIC BUILD_TIFR_INTERFACE)
endif(QUDA_INTERFACE_TIFR)

if(QUDA_CONTRACT)
  target_compile_definitions(quda PUBLIC GPU_CONTRACT)
endif(QUDA_CONTRACT)

if(QUDA_COVDEV)
  target_compile_definitions(quda PUBLIC GPU_COVDEV)
endif(QUDA_COVDEV)

# MULTI GPU AND USQCD
if(QUDA_MPI OR QUDA_QMP)
  target_compile_definitions(quda PUBLIC MULTI_GPU)
endif()

if(QUDA_MPI)
  target_link_libraries(quda PUBLIC MPI::MPI_C)
  target_compile_definitions(quda PUBLIC MPI_COMMS)
endif()

if(QUDA_QMP)
  if(QUDA_DOWNLOAD_USQCD AND NOT QMP_FOUND)
    add_dependencies(quda QMP)
    add_dependencies(quda_cpp QMP)
    #TODO: need to figure out how to propagate that cleanly
    add_dependencies(quda_cuda_target QMP)
  endif()
  target_include_directories(quda SYSTEM PUBLIC $<BUILD_INTERFACE:${QUDA_QMPHOME}/include>)
  target_compile_definitions(quda PUBLIC QMP_COMMS)
  target_link_libraries(quda INTERFACE ${QUDA_QMP_LDFLAGS} ${QUDA_QMP_LIBS})
  target_link_libraries(quda PUBLIC MPI::MPI_C)
endif()

if(QUDA_QIO)
  if(QUDA_DOWNLOAD_USQCD AND NOT QIO_FOUND)
    add_dependencies(quda QIO)
    add_dependencies(quda_cpp QIO)
    #TODO: need to figure out how to propagate that cleanly
    add_dependencies(quda_cuda_target QMP)
  endif()
  target_compile_definitions(quda PUBLIC HAVE_QIO)
  target_include_directories(quda SYSTEM PUBLIC $<BUILD_INTERFACE:${QUDA_QIOHOME}/include>)
  target_include_directories(quda SYSTEM PUBLIC $<BUILD_INTERFACE:${QUDA_LIMEHOME}/include>)
  target_link_libraries(quda INTERFACE ${QUDA_QIO_LDFLAGS} ${QUDA_QIO_LIBS})
endif()

if(QUDA_QDPJIT)
  target_compile_definitions(quda PUBLIC USE_QDPJIT)
  target_include_directories(quda SYSTEM PUBLIC $<BUILD_INTERFACE:${QUDA_QDPJITHOME}/include>)
  target_link_libraries(
    quda
    INTERFACE ${QDP_LDFLAGS}
              ${QDP_LIB}
              ${QDP_LIBS}
              ${QIO_LIB}
              ${LIME_LIB}
              ${QUDA_QMP_LDFLAGS}
              ${QMP_LIB}
              MPI::MPI_C)
endif()

if(QUDA_ARPACK)
  target_compile_definitions(quda PRIVATE ARPACK_LIB)
  if(QUDA_ARPACK_LOGGING)
    # ARPACK-NG does not suppport logging - we must warn the user
    message(
      WARNING
        "Specifying QUDA_ARPACK_LOGGING with ARPACK-NG package will cause link failures. Please ensure that QUDA_ARPACK_LOGGING=OFF if downloading ARPACK-NG or using system installed ARPACK-NG"
    )
    target_compile_definitions(ARPACK_LOGGING)
  endif()
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

if(QUDA_USE_EIGEN)
  target_compile_definitions(quda PRIVATE EIGEN)
endif()

if(QUDA_OPENMP)
  target_link_libraries(quda PUBLIC OpenMP::OpenMP_CXX)
endif()

if(QUDA_MAGMA)
  target_link_libraries(quda PUBLIC MAGMA::MAGMA)
endif()

# set which precisions to enable
target_compile_definitions(quda PUBLIC QUDA_PRECISION=${QUDA_PRECISION})
target_compile_definitions(quda PUBLIC QUDA_RECONSTRUCT=${QUDA_RECONSTRUCT})

if(QUDA_FAST_COMPILE_REDUCE)
  target_compile_definitions(quda PRIVATE QUDA_FAST_COMPILE_REDUCE)
endif()

if(QUDA_FAST_COMPILE_DSLASH)
  target_compile_definitions(quda PRIVATE QUDA_FAST_COMPILE_DSLASH)
endif()

if(QUDA_JITIFY)
  target_compile_definitions(quda PRIVATE JITIFY)
  find_package(LibDL)
  target_link_libraries(quda PUBLIC ${CUDA_nvrtc_LIBRARY})
  target_link_libraries(quda PUBLIC ${LIBDL_LIBRARIES})
  target_include_directories(quda PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)
endif()

if(QUDA_MPI_NVTX)
  target_sources(quda_cpp PRIVATE nvtx_pmpi.c)
  set(QUDA_NVTX ON)
endif(QUDA_MPI_NVTX)

if(QUDA_INTERFACE_NVTX)
  target_compile_definitions(quda PRIVATE INTERFACE_NVTX)
  set(QUDA_NVTX ON)
endif(QUDA_INTERFACE_NVTX)

if(QUDA_NVTX)
  find_path(
    NVTX3 "nvtx3/nvToolsExt.h"
    PATHS ${CUDA_TOOLKIT_INCLUDE}
    NO_DEFAULT_PATH)
  if(NVTX3)
    target_compile_definitions(quda PRIVATE QUDA_NVTX_VERSION=3)
  else()
    target_link_libraries(quda PUBLIC ${CUDA_nvToolsExt_LIBRARY})
  endif(NVTX3)
endif(QUDA_NVTX)

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
  set_property(
    SOURCE malloc.cpp
    APPEND
    PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(
    SOURCE malloc.cpp
    APPEND
    PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
  target_link_libraries(quda PUBLIC ${BACKWARD_LIBRARIES})
endif()

if(QUDA_NUMA_NVML)
  target_compile_definitions(quda PRIVATE NUMA_NVML)
  target_sources(quda_cpp PRIVATE numa_affinity.cpp)
  find_package(NVML REQUIRED)
  target_include_directories(quda PRIVATE SYSTEM NVML_INCLUDE_DIR)
  target_link_libraries(quda PUBLIC ${NVML_LIBRARY})
endif(QUDA_NUMA_NVML)

if(QUDA_NVML)
  target_link_libraries(quda PUBLIC ${NVML_LIBRARY})
endif()

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
  EXPORT qudaTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES
  DESTINATION include)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/qudaConfigVersion.cmake"
  VERSION ${QUDA_VERSION}
  COMPATIBILITY AnyNewerVersion)

export(
  EXPORT qudaTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/qudaTargets.cmake"
  NAMESPACE quda::)

set(ConfigPackageLocation lib/cmake/quda/)

install(
  EXPORT qudaTargets
  NAMESPACE quda::
  DESTINATION ${ConfigPackageLocation})

add_custom_target(
  mpi_nvtx
  ${PYTHON_EXECUTABLE} generate/wrap.py -g -o nvtx_pmpi.c generate/nvtx.w
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating mpi_nvtx wrapper")
