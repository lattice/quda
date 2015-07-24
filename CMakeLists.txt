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

#
# PROJECT is QUDA
#
project("QUDA")

###### Local configuration:

#CUDA_INSTALL_PATH = /usr/local/cuda-7.0/
#QDP_INSTALL_PATH =
#QDPXX_CXXFLAGS =
#QDPXX_LDFLAGS =
#QDPXX_LIBS =

##CMAKE no longer needed
#CPU_ARCH = x86_64  	  # x86 or x86_64

##CMAKE we need to support this
set(GPU_ARCH sm_35 CACHE STRING "set the GPU architecture (sm_20, sm_21, sm_30, sm_35, sm_50, sm_52)")




message ("os is ${OS}")

##CMAKE use Find
#GIT      = git
find_package(Git)

##CMAKE use Find
#PYTHON = python	  # python 2.5 or later required for 'make gen'
find_package(PythonInterp)

# we need to check for some packages
find_package(CUDA REQUIRED)

# compilation options
# HOST_DEBUG = no			# compile host debug code
# DEVICE_DEBUG = no		# compile device debug code for cuda-gdb
# VERBOSE = no			# display kernel register useage


set(QUDA_BLAS_TEX ON CACHE BOOL "enable texture reads in BLAS?")
set(QUDA_FERMI_DBLE_TEX ON CACHE BOOL "enable double-precision texture reads on Fermi?")
set(QUDA_NUMA_AFFINITY ON CACHE BOOL "enable NUMA affinity")

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
set(BUILD_SSTEP OFF CACHE BOOL "build s-step linear solvers")
set(BUILD_CONTRACT OFF CACHE BOOL "build code for bilinear contraction")
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



include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(include)
include_directories(lib)

add_definitions(-D__COMPUTE_CAPABILITY__=350)
add_definitions(-DQUDA_HASH="123")

STRING(REGEX REPLACE sm_ "" COMP_CAP ${GPU_ARCH})
SET(COMP_CAP "${COMP_CAP}0")



message("gpu arch " ${GPU_ARCH})
message("comp cap " ${COMP_CAP})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_35,code=sm_35 )

add_definitions(-D__COMPUTE_CAPABILITY__=${COMP_CAP})


# do all the build definitions
#

if (${BUILD_QIO})
  add_definitions(-DHAVE_QIO)
  set(QIO_UTIL qio_util.cpp layout_hyper.c gauge_qio.cpp)
  message("BUILD with QIO")
  FIND_LIBRARY(QIO_LIB qio ${QUDA_QIOHOME}/lib/)
  FIND_LIBRARY(LIME_LIB lime ${QUDA_QIOHOME}/lib/)
endif()
  #TARGET_LINK_LIBRARIES(testOVAS ${TOURTRE_LIBRARY})



if (BUILD_MILC_INTERFACE)
  add_definitions(-DBUILD_MILC_INTERFACE)
endif(BUILD_MILC_INTERFACE)

if (BUILD_QDP_INTERFACE)
  add_definitions(-DBUILD_QDP_INTERFACE)
endif(BUILD_QDP_INTERFACE)

if (BUILD_STAGGERED_DIRAC)
  add_definitions(-DGPU_STAGGERED_DIRAC)
endif(BUILD_STAGGERED_DIRAC)

### add tests and quda library
add_subdirectory(lib)

add_subdirectory(tests)
