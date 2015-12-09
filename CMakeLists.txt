# basic setup for cmake
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)

# disable in source builds
# this is only a temporary fix, but for now we need it as cmake will
# otherwise overwrite the existing makefiles
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)


find_package(Git)
SET(DEFBUILD "RELEASE")
IF(GIT_FOUND)
execute_process(COMMAND ${GIT_EXECUTABLE} show WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} RESULT_VARIABLE IS_GIT_REPOSIITORY OUTPUT_QUIET ERROR_QUIET)
  IF(${IS_GIT_REPOSIITORY} EQUAL 0)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --abbrev=0 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE GITTAG OUTPUT_STRIP_TRAILING_WHITESPACE)
# we use git rev-list and pipe that through wc here. Newer git versions support --count as option to rev-list but that might not always be available
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-list ${GITTAG}..HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} COMMAND wc -l OUTPUT_VARIABLE GITCOUNT OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GIT_EXECUTABLE}  describe --long --dirty WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE GITVERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    IF(GITCOUNT EQUAL 0)
      SET(DEFBUILD "RELEASE")
    ELSE()
      SET(DEFBUILD "DEVEL")
    ENDIF()
  ENDIF()
ENDIF(GIT_FOUND)

SET(CMAKE_BUILD_TYPE "${DEFBUILD}" CACHE STRING  "Choose the type of build, options are: Devel Release Debug Release HostDebug DeviceDebug")

set(VALID_BUILD_TYPES DEVEL RELEASE DEBUG HOSTDEBUG DEVICEDEBUG )
string(TOUPPER ${CMAKE_BUILD_TYPE} CHECK_BUILD_TYPE)
LIST(FIND VALID_BUILD_TYPES ${CHECK_BUILD_TYPE} BUILD_TYPE_VALID)

if(BUILD_TYPE_VALID LESS 0)
  message(SEND_ERROR "Please specify a valid CMAKE_BUILD_TYPE type! Valid build types are:" "${VALID_BUILD_TYPES}")
endif()


#
# PROJECT is QUDA
#
project("QUDA")

###### Local configuration:

##CMAKE we need to support this
set(GPU_ARCH sm_35 CACHE STRING "set the GPU architecture (sm_20, sm_21, sm_30, sm_35, sm_50, sm_52)")
set(CPU_ARCH x86_64 CACHE STRING "set the CPU architecture (currently only x86_64 supported)")

# we need to check for some packages
find_package(PythonInterp)

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)


set(QUDA_BLAS_TEX ON CACHE BOOL "enable texture reads in BLAS?")
set(QUDA_FERMI_DBLE_TEX ON CACHE BOOL "enable double-precision texture reads on Fermi?")
set(QUDA_NUMA_AFFINITY ON CACHE BOOL "enable NUMA affinity")
set(QUDA_VERBOSE_BUILD OFF CACHE BOOL "display kernel register useage")

# build options
set(BUILD_WILSON_DIRAC ON CACHE BOOL "build Wilson Dirac operators")
set(BUILD_CLOVER_DIRAC ON CACHE BOOL "build clover Dirac operators")
set(BUILD_DOMAIN_WALL_DIRAC ON CACHE BOOL "build domain wall Dirac operators")
set(BUILD_STAGGERED_DIRAC ON CACHE BOOL "build staggered Dirac operators")
set(BUILD_TWISTED_MASS_DIRAC ON CACHE BOOL "build twisted mass Dirac operators")
set(BUILD_TWISTED_CLOVER_DIRAC ON CACHE BOOL "build twisted clover Dirac operators")
set(BUILD_NDEG_TWISTED_MASS_DIRAC OFF CACHE BOOL "build non-degenerate twisted mass Dirac operators")
set(BUILD_FATLINK OFF CACHE BOOL "build code for computing asqtad fat links")
set(BUILD_HISQLINK OFF CACHE BOOL "build code for computing hisq fat links")
set(BUILD_GAUGE_FORCE OFF CACHE BOOL "build code for (1-loop Symanzik) gauge force")
set(BUILD_FERMION_FORCE OFF CACHE BOOL "build code for asqtad fermion force")
set(BUILD_HISQ_FORCE OFF CACHE BOOL "build code for hisq fermion force")
set(BUILD_GAUGE_TOOLS OFF CACHE BOOL "build auxilary gauge-field tools")
set(BUILD_GAUGE_ALG OFF CACHE BOOL "build gauge-fixing and pure-gauge algorithms")
set(BUILD_SSTEP OFF CACHE BOOL "build s-step linear solvers")
set(BUILD_CONTRACT OFF CACHE BOOL "build code for bilinear contraction")
# Dynamic inversion saves memory but decreases the flops
set(BUILD_DYNAMIC_CLOVER OFF CACHE BOOL "Dynamically invert the clover term for twisted-clover")
set(BUILD_QIO OFF CACHE BOOL "build QIO code for binary I/O")

# Multi-GPU options
set(BUILD_MULTI_GPU OFF CACHE BOOL "set to 'yes' to build the multi-GPU code")
set(BUILD_QMP OFF CACHE BOOL "set to 'yes' to build the QMP multi-GPU code")
set(BUILD_MPI OFF CACHE BOOL "set to 'yes' to build the MPI multi-GPU code")
set(POSIX_THREADS OFF CACHE BOOL "set to 'yes' to build pthread-enabled dslash")

#BLAS library
set(BUILD_MAGMA OFF CACHE BOOL "build magma interface")

# GPUdirect options
set(QUDA_GPU_DIRECT ON CACHE BOOL "set to 'yes' to allow GPU and NIC to shared pinned buffers")
set(QUDA_GPU_COMMS OFF CACHE BOOL "set to 'yes' to allow GPU and NIC to communicate using RDMA")

# NVTX options
set(QUDA_MPI_NVTX OFF CACHE BOOL "add nvtx markup to MPI API calls for the visual profiler")
set(QUDA_INTERFACE_NVTX OFF CACHE BOOL "add nvtx markup to interface calls for the visual profiler")

# Interface options
set(BUILD_QDP_INTERFACE ON CACHE BOOL "build qdp interface")
set(BUILD_MILC_INTERFACE ON CACHE BOOL "build milc interface")
set(BUILD_CPS_INTERFACE OFF CACHE BOOL "build cps interface")
set(BUILD_QDPJIT_INTERFACE OFF CACHE BOOL "build qdpjit interface")
set(BUILD_BQCD_INTERFACE OFF CACHE BOOL "build bqcd interface")
set(BUILD_TIFR_INTERFACE OFF CACHE BOOL "build tifr interface")

# Packing option
set(QUDA_DEVICE_PACK ON CACHE BOOL "set to 'yes' to enable packing and unpacking on the device")

# QDPJIT
set(QUDA_USE_QDPJIT OFF CACHE BOOL "build QDP-JIT support?")

set(QUDA_QIOHOME "" CACHE PATH "path to QIO")
set(QUDA_QMPHOME "" CACHE PATH "path to QMP")

#########################################################
# everything below here is processing the setup
#


# do all the build definitions
#

if(${BUILD_MULTI_GPU})
  add_definitions(-DMULTI_GPU)
  if(NOT (${BUILD_MPI} OR ${BUILD_QMP}) )
    message(FATAL_ERROR "Multi GPU build requires either MPI or QMP")
  endif()
  find_package(MPI)
else(${BUILD_MULTI_GPU})
  set(COMM_OBJS comm_single.cpp)
endif()

if(${BUILD_MPI})
  add_definitions(-DMPI_COMMS)
  set(COMM_OBJS comm_mpi.cpp)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  #message(WARNING "MPI not yet fully implemented in CMakeLists")
endif()

if (${BUILD_QMP})
  add_definitions(-DQMP_COMMS)
  execute_process(COMMAND ${QUDA_QMPHOME}/bin/qmp-config --cflags OUTPUT_VARIABLE QMP_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${QUDA_QMPHOME}/bin/qmp-config --ldflags OUTPUT_VARIABLE QMP_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${QUDA_QMPHOME}/bin/qmp-config --libs OUTPUT_VARIABLE QMP_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
  FIND_LIBRARY(QMP_LIB qmp ${QUDA_QMPHOME}/lib)
  include_directories(${QUDA_QMPHOME}/include)
  set(COMM_OBJS comm_qmp.cpp)
endif()

if (${BUILD_QIO})
  add_definitions(-DHAVE_QIO)
  set(QIO_UTIL qio_util.cpp layout_hyper.c gauge_qio.cpp)
  FIND_LIBRARY(QIO_LIB qio ${QUDA_QIOHOME}/lib/)
  FIND_LIBRARY(LIME_LIB lime ${QUDA_QIOHOME}/lib/)
  include_directories(${QUDA_QIOHOME}/include)
endif()

if(BUILD_MAGMA)
  message(WARNING "Magma support not yet implemented in CMakeLists")
else(BUILD_MAGMA)
  #do we need these
  # MAGMA_INCLUDE
  # MAGMA_FLAGS
endif(BUILD_MAGMA)

if(BUILD_SSTEP)
  add_definitions(-DSSTEP)
endif()

if(POSIX_THREADS)
  add_definitions(-DPTHREADS)
endif()

if(BUILD_WILSON_DIRAC)
  add_definitions(-DGPU_WILSON_DIRAC)
endif(BUILD_WILSON_DIRAC)

if(BUILD_DOMAIN_WALL_DIRAC)
  add_definitions(-DDGPU_DOMAIN_WALL_DIRAC)
endif(BUILD_DOMAIN_WALL_DIRAC)

if (BUILD_STAGGERED_DIRAC)
  add_definitions(-DGPU_STAGGERED_DIRAC)
endif(BUILD_STAGGERED_DIRAC)

if(BUILD_CLOVER_DIRAC)
  add_definitions(-DGPU_CLOVER_DIRAC -DGPU_WILSON_DIRAC -DGPU_GAUGE_TOOLS)
endif(BUILD_CLOVER_DIRAC)

if(BUILD_TWISTED_MASS_DIRAC)
  add_definitions(-DGPU_TWISTED_MASS_DIRAC)
endif(BUILD_TWISTED_MASS_DIRAC)

if(BUILD_TWISTED_CLOVER_DIRAC)
  add_definitions(-DGPU_TWISTED_CLOVER_DIRAC -DGPU_CLOVER_DIRAC -DGPU_TWISTED_MASS_DIRAC -DGPU_WILSON_DIRAC -DGPU_GAUGE_TOOLS)
endif(BUILD_TWISTED_CLOVER_DIRAC)

if(BUILD_NDEG_TWISTED_MASS_DIRAC)
  add_definitions(-DGPU_NDEG_TWISTED_MASS_DIRAC -DGPU_TWISTED_MASS_DIRAC -DGPU_WILSON_DIRAC)
endif(BUILD_NDEG_TWISTED_MASS_DIRAC)

if(BUILD_FATLINK)
  add_definitions(-DGPU_FATLINK -DGPU_GAUGE_TOOLS)
endif(BUILD_FATLINK)

if(BUILD_HISQLINK)
  add_definitions(-DGPU_FATLINK -DGPU_UNITARIZE -DGPU_GAUGE_TOOLS)
endif(BUILD_HISQLINK)

if(BUILD_GAUGE_FORCE)
  add_definitions(-DGPU_GAUGE_FORCE -DGPU_GAUGE_TOOLS)
endif(BUILD_GAUGE_FORCE)

if(BUILD_FERMION_FORCE)
  add_definitions(-DGPU_FERMION_FORCE -DGPU_GAUGE_TOOLS)
endif(BUILD_FERMION_FORCE)

if(BUILD_HISQ_FORCE)
  add_definitions(-DGPU_HISQ_FORCE -DGPU_STAGGERED_OPROD -DGPU_GAUGE_TOOLS)
endif(BUILD_HISQ_FORCE)

if(BUILD_GAUGE_TOOLS)
  add_definitions(-DGPU_GAUGE_TOOLS)
endif(BUILD_GAUGE_TOOLS)

if(BUILD_GAUGE_ALG)
  add_definitions(-DGPU_GAUGE_ALG)
  add_definitions(-DGPU_GAUGE_TOOLS)
  add_definitions(-DGPU_UNITARIZE)
  LIST(APPEND QUDA_LIBS ${CUDA_cufft_LIBRARY} ${CUDA_curand_LIBRARY})
endif(BUILD_GAUGE_ALG)

if(BUILD_DYNAMIC_CLOVER)
  add_definitions(-DDYNAMIC_CLOVER)
endif(BUILD_DYNAMIC_CLOVER)

if(QUDA_MPI_NVTX)
  LIST(APPEND COMM_OBJS nvtx_pmpi.c)
  set(QUDA_NVTX ON)
endif(QUDA_MPI_NVTX)

if(QUDA_INTERFACE_NVTX)
  add_definitions(-DINTERFACE_NVTX)
  set(QUDA_NVTX ON)
endif(QUDA_INTERFACE_NVTX)

#TODO: move to tests, check lib64
if(QUDA_NVTX)
  FIND_LIBRARY(CUDA_NVTX_LIB nvToolsExt ${CUDA_TOOLKIT_ROOT_DIR}/lib/)
  LIST(APPEND QUDA_LIBS ${CUDA_NVTX_LIB})
endif(QUDA_NVTX)

if(NOT QUDA_BLAS_TEX)
  add_definitions(-DDIRECT_ACCESS_BLAS)
endif(NOT QUDA_BLAS_TEX)

if(NOT QUDA_FERMI_DBLE_TEX)
  add_definitions(-DFERMI_NO_DBLE_TEX)
endif(NOT QUDA_FERMI_DBLE_TEX)

if(GPU_DIRECT)
  add_definitions(-DGPU_DIRECT)
endif(GPU_DIRECT)

if(GPU_COMMS)
  add_definitions(-DGPU_COMMS)
endif(GPU_COMMS)

if(BUILD_QDP_INTERFACE)
  add_definitions(-DBUILD_QDP_INTERFACE)
endif(BUILD_QDP_INTERFACE)

if(BUILD_MILC_INTERFACE)
  add_definitions(-DBUILD_MILC_INTERFACE)
endif(BUILD_MILC_INTERFACE)

if(BUILD_CPS_INTERFACE)
  add_definitions(-DBUILD_CPS_INTERFACE)
endif(BUILD_CPS_INTERFACE)

if(BUILD_QDPJIT_INTERFACE)
add_definitions(-DBUILD_QDPJIT_INTERFACE)
endif(BUILD_QDPJIT_INTERFACE)

if(BUILD_BQCD_INTERFACE)
  add_definitions(-DBUILD_BQCD_INTERFACE)
endif(BUILD_BQCD_INTERFACE)

if(BUILD_TIFR_INTERFACE)
  add_definitions(-DBUILD_TIFR_INTERFACE)
endif(BUILD_TIFR_INTERFACE)

# derive whether we need to build the fortran interface
if(BUILD_TIFR_INTERFACE OR BUILD_BQCD_INTERFACE)
  SET(BUILD_FORTRAN_INTERFACE ON)
  enable_language(Fortran)
endif()

if(DEVICE_PACK)
  add_definitions(-DDEVICE_PACK)
endif(DEVICE_PACK)

if(NOT ${APPLE})
  if(QUDA_NUMA_AFFINITY)
    add_definitions(-DNUMA_AFFINITY)
    set(NUMA_AFFINITY_OBJS numa_affinity.cpp)
  endif(QUDA_NUMA_AFFINITY)
endif(NOT ${APPLE})

if(BUILD_CONTRACT)
  add_definitions(-DGPU_CONTRACT)
endif(BUILD_CONTRACT)


# COMPILER OPTIONS and BUILD types
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(include)
include_directories(lib)

# QUDA_HASH for tunecache
file(STRINGS ${CUDA_TOOLKIT_INCLUDE}/cuda.h  CUDA_VERSIONLONG REGEX "\#define CUDA_VERSION" )
STRING(REPLACE "\#define CUDA_VERSION " ""  CUDA_VERSIONLONG ${CUDA_VERSIONLONG})
STRING(STRIP CUDA_VERSIONLONG ${CUDA_VERSIONLONG} )
set(HASH \"cpu_arch=${CPU_ARCH},gpu_arch=${GPU_ARCH},cuda_version=${CUDA_VERSIONLONG}\")
add_definitions(-DQUDA_HASH=${HASH})

string(REGEX MATCH [Dd][Ee][Bb][Uu][Gg] DEBUG_BUILD ${CMAKE_BUILD_TYPE})


# add -debug to GITVERSION if we build with debug options enabled
if(DEBUG_BUILD)
  if(GITVERSION)
    set(GITVERSION ${GITVERSION}-debug)
  else()
    set(GITVERSION debug)
  endif()
endif()

IF(GITVERSION)
  add_definitions(-DGITVERSION="${GITVERSION}")
ENDIF()

# GPU ARCH
STRING(REGEX REPLACE sm_ "" COMP_CAP ${GPU_ARCH})
SET(COMP_CAP "${COMP_CAP}0")
add_definitions(-D__COMPUTE_CAPABILITY__=${COMP_CAP})


# NVCC FLAGS independet off build type
set(QUDA_NVCC_FLAGS -arch=${GPU_ARCH} -ftz=true -prec-div=false -prec-sqrt=false)

if(QUDA_VERBOSE_BUILD)
  LIST(APPEND QUDA_NVCC_FLAGS --ptxas-options=-v)
endif(QUDA_VERBOSE_BUILD)


set(CUDA_NVCC_FLAGS_DEVEL ${QUDA_NVCC_FLAGS} -O3 -lineinfo CACHE STRING
    "Flags used by the CUDA compiler during regular development builds."
    FORCE )
set(CUDA_NVCC_FLAGS_RELEASE ${QUDA_NVCC_FLAGS} -O3 -w CACHE STRING
    "Flags used by the C++ compiler during release builds."
    FORCE )
set(CUDA_NVCC_FLAGS_HOSTDEBUG ${QUDA_NVCC_FLAGS} -g -DHOST_DEBUG CACHE STRING
    "Flags used by the C++ compiler during host-debug builds."
    FORCE )
set(CUDA_NVCC_FLAGS_DEVICEDEBUG ${QUDA_NVCC_FLAGS} -G CACHE STRING
    "Flags used by the C++ compiler during device-debug builds."
    FORCE )
set(CUDA_NVCC_FLAGS_DEBUG ${QUDA_NVCC_FLAGS} -g -DHOST_DEBUG -G CACHE STRING
    "Flags used by the C++ compiler during full (host+device) debug builds."
    FORCE )


set(CMAKE_CXX_FLAGS_DEVEL  "-O3 -Wall" CACHE STRING
    "Flags used by the CUDA compiler during regular development builds."
    FORCE )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -w" CACHE STRING
    "Flags used by the C++ compiler during release builds."
    FORCE )
set(CMAKE_CXX_FLAGS_HOSTDEBUG "-Wall -g -fno-inline -DHOST_DEBUG" CACHE STRING
    "Flags used by the C++ compiler during host-debug builds."
    FORCE )
set(CMAKE_CXX_FLAGS_DEVICEDEBUG "-Wall" CACHE STRING
    "Flags used by the C++ compiler during device-debug builds."
    FORCE )
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -fno-inline -DHOST_DEBUG" CACHE STRING
    "Flags used by the C++ compiler during full (host+device) debug builds."
    FORCE )

set(CMAKE_F_FLAGS -std=c99 CACHE STRING "Fortran Flags" FORCE)

set(CMAKE_C_FLAGS_DEVEL "-Wall -O3"  CACHE STRING
    "Flags used by the CUDA compiler during regular development builds."
    FORCE )
set(CMAKE_C_FLAGS_RELEASE "-Wall -O3 -w" CACHE STRING
    "Flags used by the C++ compiler during release builds."
    FORCE )
set(CMAKE_C_FLAGS_HOSTDEBUG "-Wall -g -fno-inline -DHOST_DEBUG" CACHE STRING
    "Flags used by the C++ compiler during host-debug builds."
    FORCE )
set(CMAKE_C_FLAGS_DEVICEDEBUG "-Wall" CACHE STRING
    "Flags used by the C++ compiler during device-debug builds."
    FORCE )
set(CMAKE_C_FLAGS_DEBUG "-Wall -g -fno-inline -DHOST_DEBUG" CACHE STRING
    "Flags used by the C++ compiler during full (host+device) debug builds."
    FORCE )

### add tests and quda library
add_subdirectory(lib)

add_subdirectory(tests)
