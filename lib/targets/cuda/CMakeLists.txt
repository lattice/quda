
set(QUDA_TARGET_CUDA ON)
# add target specific files / options
include(CMakeDependentOption)

if(DEFINED ENV{QUDA_GPU_ARCH})
  set(QUDA_DEFAULT_GPU_ARCH $ENV{QUDA_GPU_ARCH})
else()
  set(QUDA_DEFAULT_GPU_ARCH sm_70)
endif()
if(NOT QUDA_GPU_ARCH)
  message(STATUS "Building QUDA for GPU ARCH " "${QUDA_DEFAULT_GPU_ARCH}")
endif()

set(QUDA_GPU_ARCH
    ${QUDA_DEFAULT_GPU_ARCH}
    CACHE STRING "set the GPU architecture (sm_60, sm_70, sm_80)")
set_property(CACHE QUDA_GPU_ARCH PROPERTY STRINGS sm_60 sm_70 sm_80)


####
# CUDA specific options
option(QUDA_NVML "use NVML to report CUDA graphics driver version" OFF)
option(QUDA_NUMA_NVML "experimental use of NVML to set numa affinity" OFF)
option(QUDA_VERBOSE_BUILD "display kernel register usage" OFF)
option(QUDA_JITIFY "build QUDA using Jitify" OFF)
option(QUDA_DOWNLOAD_NVSHMEM "Download NVSHMEM" OFF)
set(QUDA_NVSHMEM OFF CACHE BOOL "set to 'yes' to build the NVSHMEM multi-GPU code")
set(QUDA_NVSHMEM_HOME $ENV{NVSHMEM_HOME} CACHE PATH "path to NVSHMEM")

set(QUDA_GDRCOPY_HOME "/usr/local/gdrcopy" CACHE STRING "path to gdrcopy used when QUDA_DOWNLOAD_NVSHMEM is enabled")
# NVTX options
option(QUDA_INTERFACE_NVTX "add NVTC markup to interface calls" OFF)

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0" AND CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
  set(QUDA_HETEROGENEOUS_ATOMIC_SUPPORT ON)
  message(STATUS "Heterogeneous atomics supported: ${QUDA_HETEROGENEOUS_ATOMIC_SUPPORT}")
endif()
CMAKE_DEPENDENT_OPTION(QUDA_HETEROGENEOUS_ATOMIC "enable heterogeneous atomic support (CUDA >= 11.0)?" ON "QUDA_HETEROGENEOUS_ATOMIC_SUPPORT" OFF)

if((QUDA_HETEROGENEOUS_ATOMIC OR QUDA_NVSHMEM) AND ${CMAKE_BUILD_TYPE} STREQUAL "SANITIZE")
  message(SEND_ERROR "QUDA_HETEROGENEOUS_ATOMIC=ON AND/OR QUDA_NVSHMEM=ON do not support SANITIZE build)")
endif()
if(QUDA_HETEROGENEOUS_ATOMIC AND QUDA_JITIFY)
  message(SEND_ERROR "QUDA_HETEROGENEOUS_ATOMIC=ON does not support JITIFY)")
endif()
mark_as_advanced(QUDA_HETEROGENEOUS_ATOMIC)
mark_as_advanced(QUDA_JITIFY)

mark_as_advanced(QUDA_DOWNLOAD_NVSHMEM)
mark_as_advanced(QUDA_DOWNLOAD_NVSHMEM_TAR)
mark_as_advanced(QUDA_GDRCOPY_HOME)
mark_as_advanced(QUDA_NVML)
mark_as_advanced(QUDA_NUMA_NVML)
mark_as_advanced(QUDA_VERBOSE_BUILD)
mark_as_advanced(QUDA_INTERFACE_NVTX)

####################################
# NVSHMEM
####################################
if(QUDA_NVSHMEM)
  if(QUDA_DOWNLOAD_NVSHMEM)
  # workaround potential UCX interaction issue with CUDA 11.3+ and UCX in NVSHMEM 2.1.2
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "11.3")
    set(QUDA_DOWNLOAD_NVSHMEM_TAR "https://developer.download.nvidia.com/compute/redist/nvshmem/2.1.2/source/nvshmem_src_2.1.2-0.txz" CACHE STRING "location of NVSHMEM tarball")
  else()
    set(QUDA_DOWNLOAD_NVSHMEM_TAR "https://developer.download.nvidia.com/compute/redist/nvshmem/2.2.1/source/nvshmem_src_2.2.1-0.txz" CACHE STRING "location of NVSHMEM tarball")
  endif()
    get_filename_component(NVSHMEM_CUDA_HOME ${CUDAToolkit_INCLUDE_DIRS} DIRECTORY)
    find_path(GDRCOPY_HOME NAME gdrapi.h PATHS "/usr/local/gdrcopy" ${QUDA_GDRCOPY_HOME} PATH_SUFFIXES "include")
    mark_as_advanced(GDRCOPY_HOME)
    if(NOT GDRCOPY_HOME)
      message(SEND_ERROR "QUDA_DOWNLOAD_NVSHMEM requires gdrcopy to be installed. Please set QUDA_GDRCOPY_HOME to the location of your gdrcopy installation.")
    endif()
    get_filename_component(NVSHMEM_GDRCOPY_HOME ${GDRCOPY_HOME} DIRECTORY)
    ExternalProject_Add(NVSHMEM
                        URL ${QUDA_DOWNLOAD_NVSHMEM_TAR}
                        PREFIX nvshmem
                        CONFIGURE_COMMAND ""
                        BUILD_IN_SOURCE ON
                        BUILD_COMMAND  make -j8 MPICC=${MPI_C_COMPILER} CUDA_HOME=${NVSHMEM_CUDA_HOME} NVSHMEM_PREFIX=<INSTALL_DIR> NVSHMEM_MPI_SUPPORT=1 GDRCOPY_HOME=${NVSHMEM_GDRCOPY_HOME} install
                        INSTALL_COMMAND ""
                        LOG_INSTALL ON
                        LOG_BUILD ON
                        LOG_DOWNLOAD ON
                        )
    ExternalProject_Get_Property(NVSHMEM INSTALL_DIR)
    set(QUDA_NVSHMEM_HOME ${INSTALL_DIR} CACHE PATH "path to NVSHMEM" FORCE)
    set(NVSHMEM_LIBS  ${INSTALL_DIR}/lib/libnvshmem.a)
    set(NVSHMEM_INCLUDE ${INSTALL_DIR}/include/)   
  else()
    if("${QUDA_NVSHMEM_HOME}" STREQUAL "")
      message( FATAL_ERROR "QUDA_NVSHMEM_HOME must be defined if QUDA_NVSHMEM is set" )
    endif()
      find_library(NVSHMEM_LIBS NAMES nvshmem PATHS "${QUDA_NVSHMEM_HOME}/lib/"  )
      find_path(NVSHMEM_INCLUDE NAMES nvshmem.h PATHS "${QUDA_NVSHMEM_HOME}/include/" )
  endif()
    
  mark_as_advanced(NVSHMEM_LIBS)
  mark_as_advanced(NVSHMEM_INCLUDE)
  add_library(nvshmem_lib STATIC IMPORTED)
  set_target_properties(nvshmem_lib PROPERTIES IMPORTED_LOCATION ${NVSHMEM_LIBS})
  set_target_properties(nvshmem_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(nvshmem_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  set_target_properties(nvshmem_lib PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES CUDA)
endif()


# This is needed now GPU ARCH
string(REGEX REPLACE sm_ "" COMP_CAP ${QUDA_GPU_ARCH})
set(CMAKE_CUDA_ARCHITECTURES ${COMP_CAP})
set(COMP_CAP "${COMP_CAP}0")
# QUDA_HASH for tunecache
set(HASH cpu_arch=${CPU_ARCH},gpu_arch=${QUDA_GPU_ARCH},cuda_version=${CMAKE_CUDA_COMPILER_VERSION})
set(GITVERSION "${PROJECT_VERSION}-${QUDA_GPU_ARCH} ${GITVERSION}")

target_sources(quda_cpp PRIVATE quda_api.cpp device.cpp malloc.cpp blas_lapack_cublas.cpp comm_target.cpp)

if(QUDA_JITIFY)
  target_sources(quda_cpp PRIVATE jitify_helper.cpp)
endif()

if(QUDA_BACKWARDS)
  set_property(SOURCE malloc.cpp DIRECTORY ${CMAKE_SOURCE_DIR}/lib APPEND PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE malloc.cpp DIRECTORY ${CMAKE_SOURCE_DIR}/lib APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
endif()

# cuda specific compile options
target_compile_options(
  quda
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
          -ftz=true
          -prec-div=false
          -prec-sqrt=false>
          $<$<COMPILE_LANG_AND_ID:CUDA,NVHPC>:
          -Mflushz
          -Mfpapprox=div
          -Mfpapprox=sqrt>
          $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:
          -fcuda-flush-denormals-to-zero
          -fcuda-approx-transcendentals
          -Xclang -fcuda-allow-variadic-functions>)
target_compile_options(
  quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-Wno-unknown-cuda-version> $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
               -Wno-deprecated-gpu-targets -arch=${QUDA_GPU_ARCH} --expt-relaxed-constexpr>)

target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -ftz=true -prec-div=false -prec-sqrt=false>)
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -Wno-deprecated-gpu-targets
                                    -arch=${QUDA_GPU_ARCH} --expt-relaxed-constexpr>)
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}
                                    --cuda-gpu-arch=${QUDA_GPU_ARCH}>)
target_link_options(quda PUBLIC $<$<CUDA_COMPILER_ID:Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}>)

if(QUDA_VERBOSE_BUILD)
  target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--ptxas-options=-v>)
endif(QUDA_VERBOSE_BUILD)

if (${CMAKE_CUDA_COMPILER_ID} MATCHES "NVHPC" AND NOT ${CMAKE_BUILD_TYPE} MATCHES "DEBUG")
  target_compile_options(quda PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVHPC>:SHELL: -gpu=nodebug" >)
endif()

target_include_directories(quda SYSTEM PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDAToolkit_INCLUDE_DIRS}>)
target_include_directories(quda_cpp SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

# workaround for 10.2
if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA"
   AND ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "10.2"
   AND ${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "10.3")
  target_compile_options(
    quda PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL: -Xcicc \"--Xllc -dag-vectorize-ops=1\" " >)
endif()
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:--cuda-path=${CUDAToolkit_TARGET_DIR}>)
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xfatbin=-compress-all>)
target_include_directories(quda PRIVATE ${CMAKE_SOURCE_DIR}/include/targets/cuda)
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/targets/cuda> $<INSTALL_INTERFACE:include/targets/cuda>)
target_include_directories(quda SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/include/targets/cuda/externals)

# Specific config dependent warning suppressions and lineinfo forwarding
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
                            -Wreorder
                            $<$<CXX_COMPILER_ID:Clang>: -Xcompiler=-Wno-unused-function -Xcompiler=-Wno-unknown-pragmas>
                            $<$<CXX_COMPILER_ID:GNU>: -Xcompiler=-Wno-unknown-pragmas>
                            $<$<CONFIG:DEVEL>:-Xptxas -warn-lmem-usage,-warn-spills -lineinfo>
                            $<$<CONFIG:STRICT>: -Werror=all-warnings -lineinfo>
                            $<$<CONFIG:HOSTDEBUG>:-lineinfo>
                            $<$<CONFIG:SANITIZE>:-lineinfo>
                            >)

target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVHPC>:
                            -gpu=lineinfo
                            >)

target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:
                            -Wall -Wextra -Wno-unknown-pragmas
                            $<$<CONFIG:STRICT>:-Werror -Wno-error=pass-failed>
                            $<$<CONFIG:SANITIZE>:-fsanitize=address -fsanitize=undefined>
          >)

# malloc.cpp uses both the driver and runtime api So we need to find the CUDA_CUDA_LIBRARY (driver api) or the stub
# version for cmake 3.8 and later this has been integrated into  FindCUDALibs.cmake
target_link_libraries(quda PUBLIC ${CUDA_cuda_driver_LIBRARY})

if(QUDA_NVSHMEM)
  if((${COMP_CAP} LESS "700"))
    message(SEND_ERROR "QUDA_NVSHMEM=ON requires at least QUDA_GPU_ARCH=sm_70")
  endif()
  set_target_properties(quda_pack PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET quda PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(quda PUBLIC MPI::MPI_C)
  target_compile_definitions(quda PUBLIC NVSHMEM_COMMS)
  if(QUDA_DOWNLOAD_NVSHMEM)
    add_dependencies(quda NVSHMEM)
    add_dependencies(quda_cpp NVSHMEM)
    add_dependencies(quda_pack NVSHMEM)
  endif()
  get_filename_component(NVSHMEM_LIBPATH ${NVSHMEM_LIBS} DIRECTORY)
  target_link_libraries(quda PUBLIC -L${NVSHMEM_LIBPATH} -lnvshmem)
  target_include_directories(quda SYSTEM PUBLIC $<BUILD_INTERFACE:${NVSHMEM_INCLUDE}>)
endif()

if(${QUDA_BUILD_NATIVE_LAPACK} STREQUAL "ON")
  target_link_libraries(quda PUBLIC ${CUDA_cublas_LIBRARY})
endif()

if(QUDA_GAUGE_ALG)
  target_link_libraries(quda PUBLIC ${CUDA_cufft_LIBRARY})
endif(QUDA_GAUGE_ALG)

if(QUDA_JITIFY)
  target_compile_definitions(quda PRIVATE JITIFY)
  find_package(LibDL)
  target_link_libraries(quda PUBLIC ${CUDA_nvrtc_LIBRARY})
  target_link_libraries(quda PUBLIC ${LIBDL_LIBRARIES})
  target_include_directories(quda PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

  configure_file(${CMAKE_SOURCE_DIR}/include/targets/cuda/jitify_options.hpp.in ${${CMAKE_BINARY_DIR}}/include/targets/cuda/jitify_options.hpp)
  install(FILES "${CMAKE_BINARY_DIR}/include/targets/cuda/jitify_options.hpp" DESTINATION include/)
endif()


if(QUDA_INTERFACE_NVTX)
  target_compile_definitions(quda PRIVATE INTERFACE_NVTX)
  set(QUDA_NVTX ON)
endif(QUDA_INTERFACE_NVTX)

if(QUDA_NVTX)
  find_path(NVTX3 "nvtx3/nvToolsExt.h" PATHS ${CUDA_TOOLKIT_INCLUDE} NO_DEFAULT_PATH)
  if(NVTX3)
    target_compile_definitions(quda PRIVATE QUDA_NVTX_VERSION=3)
  else()
    target_link_libraries(quda PUBLIC ${CUDA_nvToolsExt_LIBRARY})
  endif(NVTX3)
endif(QUDA_NVTX)

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