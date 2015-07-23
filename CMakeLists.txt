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
#GPU_ARCH = sm_35	  # sm_13, sm_20, sm_21, sm_30, or sm_35


##CMAKE We can do it this way but also use ${APPLE} right away
#OS       = linux	  # linux or osx
if(${UNIX})
  if(${APPLE})
    set(OS "osx")
  else(${APPLE})
    set(OS "linux")
  endif(${APPLE})
endif(${UNIX})

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
#BUILD_MULTI_GPU = yes  # set to 'yes' to build the multi-GPU code
#BUILD_QMP = no              # set to 'yes' to build the QMP multi-GPU code
#BUILD_MPI = yes              # set to 'yes' to build the MPI multi-GPU code
#POSIX_THREADS = no     # set to 'yes' to build pthread-enabled dslash

#BLAS library
#BUILD_MAGMA = no 	# build magma interface

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


include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(include)
include_directories(lib)

add_definitions(-D__COMPUTE_CAPABILITY__=350)
add_definitions(-DQUDA_HASH="123")



set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_35,code=sm_35 )


# do all the build definitions
#
if (BUILD_MILC_INTERFACE)
  add_definitions(-DBUILD_MILC_INTERFACE)
endif(BUILD_MILC_INTERFACE)

if (BUILD_STAGGERED_DIRAC)
  add_definitions(-DGPU_STAGGERED_DIRAC)
endif(BUILD_STAGGERED_DIRAC)


# build quda in the lib subdirectory
add_subdirectory(lib)
