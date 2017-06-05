# basic setup for cmake
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 11)
# disable in source builds
# this is only a temporary fix, but for now we need it as cmake will
# otherwise overwrite the existing makefiles
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
# add a directory for cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")



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

set(VALID_BUILD_TYPES DEVEL RELEASE STRICT DEBUG HOSTDEBUG DEVICEDEBUG )
SET(CMAKE_BUILD_TYPE "${DEFBUILD}" CACHE STRING  "Choose the type of build, options are: ${VALID_BUILD_TYPES}")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS DEVEL RELEASE STRICT DEBUG HOSTDEBUG DEVICEDEBUG)

string(TOUPPER ${CMAKE_BUILD_TYPE} CHECK_BUILD_TYPE)
LIST(FIND VALID_BUILD_TYPES ${CHECK_BUILD_TYPE} BUILD_TYPE_VALID)

if(BUILD_TYPE_VALID LESS 0)
  message(SEND_ERROR "Please specify a valid CMAKE_BUILD_TYPE type! Valid build types are:" "${VALID_BUILD_TYPES}")
endif()


#
# PROJECT is QUDA
#
project("QUDA" VERSION 0.9.0)

if(NOT GITVERSION)
  set(GITVERSION ${CMAKE_VERSION})
endif()


#######################################################################
# QUDA OPTIONS
# likely to be changed by users
#######################################################################
if(DEFINED ENV{QUDA_GPU_ARCH})
  set(QUDA_DEFAULT_GPU_ARCH $ENV{QUDA_GPU_ARCH})
else()
  set(QUDA_DEFAULT_GPU_ARCH sm_35)
endif()
if(NOT QUDA_GPU_ARCH)
  message ("-- Building QUDA for arch " "${QUDA_DEFAULT_GPU_ARCH}")
endif()

set(QUDA_GPU_ARCH ${QUDA_DEFAULT_GPU_ARCH} CACHE STRING "set the GPU architecture (sm_20, sm_21, sm_30, sm_35, sm_37, sm_50, sm_52, sm_60)")
set_property(CACHE QUDA_GPU_ARCH PROPERTY STRINGS sm_20 sm_21 sm_30 sm_35 sm_37 sm_50 sm_52 sm_60)
# build options
set(QUDA_DIRAC_WILSON ON CACHE BOOL "build Wilson Dirac operators")
set(QUDA_DIRAC_CLOVER ON CACHE BOOL "build clover Dirac operators")
set(QUDA_DIRAC_DOMAIN_WALL ON CACHE BOOL "build domain wall Dirac operators")
set(QUDA_DIRAC_STAGGERED ON CACHE BOOL "build staggered Dirac operators")
set(QUDA_DIRAC_TWISTED_MASS ON CACHE BOOL "build twisted mass Dirac operators")
set(QUDA_DIRAC_TWISTED_CLOVER ON CACHE BOOL "build twisted clover Dirac operators")
set(QUDA_DIRAC_NDEG_TWISTED_MASS OFF CACHE BOOL "build non-degenerate twisted mass Dirac operators")
set(QUDA_LINK_ASQTAD OFF CACHE BOOL "build code for computing asqtad fat links")
set(QUDA_LINK_HISQ OFF CACHE BOOL "build code for computing hisq fat links")
set(QUDA_FORCE_GAUGE OFF CACHE BOOL "build code for (1-loop Symanzik) gauge force")
set(QUDA_FORCE_ASQTAD OFF CACHE BOOL "build code for asqtad fermion force")
set(QUDA_FORCE_HISQ OFF CACHE BOOL "build code for hisq fermion force")
set(QUDA_GAUGE_TOOLS OFF CACHE BOOL "build auxiliary gauge-field tools")
set(QUDA_GAUGE_ALG OFF CACHE BOOL "build gauge-fixing and pure-gauge algorithms")
set(QUDA_CONTRACT OFF CACHE BOOL "build code for bilinear contraction")
# Dynamic inversion saves memory but decreases the flops
set(QUDA_DYNAMIC_CLOVER OFF CACHE BOOL "Dynamically invert the clover term for twisted-clover")
set(QUDA_QIO OFF CACHE BOOL "build QIO code for binary I/O")

# Multi-GPU options
set(QUDA_QMP OFF CACHE BOOL "set to 'yes' to build the QMP multi-GPU code")
set(QUDA_MPI OFF CACHE BOOL "set to 'yes' to build the MPI multi-GPU code")
set(QUDA_POSIX_THREADS OFF CACHE BOOL "set to 'yes' to build pthread-enabled dslash")

#BLAS library
set(QUDA_MAGMA OFF CACHE BOOL "build magma interface")

#ARPACK
set(QUDA_ARPACK OFF CACHE BOOL "build arpack interface")

# Interface options
set(QUDA_INTERFACE_QDP ON CACHE BOOL "build qdp interface")
set(QUDA_INTERFACE_MILC ON CACHE BOOL "build milc interface")
set(QUDA_INTERFACE_CPS OFF CACHE BOOL "build cps interface")
set(QUDA_INTERFACE_QDPJIT OFF CACHE BOOL "build qdpjit interface")
set(QUDA_INTERFACE_BQCD OFF CACHE BOOL "build bqcd interface")
set(QUDA_INTERFACE_TIFR OFF CACHE BOOL "build tifr interface")

# QDPJIT
set(QUDA_QDPJIT OFF CACHE BOOL "build QDP-JIT support?")
set(QUDA_QDPJITHOME "" CACHE PATH "path to QDPJIT installation")

# Locations for QIO / QMP
set(QUDA_QIOHOME "" CACHE PATH "path to QIO")
set(QUDA_QMPHOME "" CACHE PATH "path to QMP")
set(QUDA_LIMEHOME "" CACHE PATH "path to LIME")
set(QUDA_ARPACK_HOME "" CACHE PATH "path to arpack / parpack")
set(QUDA_MAGMAHOME "" CACHE PATH "path to MAGMA, if not set, pkg-config will be attempted")
set(QUDA_MAGMA_LIBS "" CACHE STRING "additional linker flags required to link against magma")

#######################################################################
#QUDA ADVANCED OPTIONS
# that ususally should not be changed by users
#######################################################################
set(QUDA_BUILD_ALL_TESTS ON CACHE BOOL "build tests by default")
set(QUDA_BUILD_SHAREDLIB OFF CACHE BOOL "build quda as a shared lib")
set(QUDA_BLAS_TEX ON CACHE BOOL "enable texture reads in BLAS?")
set(QUDA_FERMI_DBLE_TEX ON CACHE BOOL "enable double-precision texture reads on Fermi?")
set(QUDA_NUMA_NVML OFF CACHE BOOL "experimental use of NVML to set numa affinity" )
set(QUDA_VERBOSE_BUILD OFF CACHE BOOL "display kernel register useage")
set(QUDA_MAX_MULTI_BLAS_N  "4" CACHE STRING "maximum value to intizlize template for multi-blas /-reduce")


# NVTX options
set(QUDA_MPI_NVTX OFF CACHE BOOL "add nvtx markup to MPI API calls for the visual profiler")
set(QUDA_INTERFACE_NVTX OFF CACHE BOOL "add nvtx markup to interface calls for the visual profiler")

# GPUdirect options
set(QUDA_GPU_DIRECT ON CACHE BOOL "set to 'yes' to allow GPU and NIC to shared pinned buffers")

# features in development
set(QUDA_SSTEP OFF CACHE BOOL "build s-step linear solvers")
set(QUDA_MULTIGRID OFF CACHE BOOL "build multigrid solvers")
set(QUDA_BLOCKSOLVER OFF CACHE BOOL "build block solvers")
set(QUDA_DEFLATEDSOLVER OFF CACHE BOOL "build deflated solvers")
set(QUDA_ALTRELIABLE OFF CACHE BOOL "use alternative reliable updates in CG")
set(QUDA_EIGEN OFF CACHE BOOL "build with EIGEN library")

mark_as_advanced(QUDA_BUILD_ALL_TESTS)
mark_as_advanced(QUDA_BUILD_SHAREDLIB)
mark_as_advanced(QUDA_BLAS_TEX)
mark_as_advanced(QUDA_FERMI_DBLE_TEX)
mark_as_advanced(QUDA_NUMA_NVML)
mark_as_advanced(QUDA_VERBOSE_BUILD)
mark_as_advanced(QUDA_MAX_MULTI_BLAS_N)

mark_as_advanced(QUDA_MPI_NVTX)
mark_as_advanced(QUDA_INTERFACE_NVTX)

mark_as_advanced(QUDA_GPU_DIRECT)

mark_as_advanced(QUDA_SSTEP)
mark_as_advanced(QUDA_EIGEN)
mark_as_advanced(QUDA_ALTRELIABLE)

#######################################################################
# options that are not exposed at all because only one option exists
set(CPU_ARCH x86_64)


#######################################################################
# everything below here is processing the setup
#######################################################################

# we need to check for some packages
find_package(PythonInterp)

if(${CMAKE_VERSION} VERSION_GREATER 3.7.99)
  find_package(CUDAWrapper)
  set(USING_CUDA_LANG_SUPPORT True)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
else()
  set(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Host side compiler used by NVCC")
  mark_as_advanced(CUDA_HOST_COMPILER)
  find_package(CUDA REQUIRED)
  set(USING_CUDA_LANG_SUPPORT False)
endif()

find_package(Threads REQUIRED)

# do all the build definitions
#

if(QUDA_MPI OR QUDA_QMP)
  add_definitions(-DMULTI_GPU)
  find_package(MPI)
else()
  set(COMM_OBJS comm_single.cpp)
endif()

if(QUDA_QDPJIT)
  add_definitions(-DUSE_QDPJIT)
  include_directories(SYSTEM ${QUDA_QDPJITHOME}/include)
  execute_process(COMMAND ${QUDA_QDPJITHOME}/bin/qdp\+\+-config --ldflags OUTPUT_VARIABLE QDP_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${QUDA_QDPJITHOME}/bin/qdp\+\+-config --libs OUTPUT_VARIABLE QDP_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
  FIND_LIBRARY(QDP_LIB qdp PATH ${QUDA_QDPJITHOME}/lib)
  FIND_LIBRARY(QIO_LIB qio ${QUDA_QDPJITHOME}/lib/)
  FIND_LIBRARY(LIME_LIB lime ${QUDA_QDPJITHOME}/lib/)
endif()

if(QUDA_MPI AND QUDA_QMP)
  message(WARNING "Specifying QUDA_QMP and QUDA_MPI might result in undefined behavior. If you intend to use QMP set QUDA_MPI=OFF.")
endif()

if(QUDA_MPI)
  add_definitions(-DMPI_COMMS)
  set(COMM_OBJS comm_mpi.cpp)
  include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
endif()

if(QUDA_QMP)
  if("${QUDA_QMPHOME}" STREQUAL "")
    message( FATAL_ERROR "QUDA_QMPHOME must be defined if QUDA_QMP is set" )
  endif()
  add_definitions(-DQMP_COMMS)
  execute_process(COMMAND ${QUDA_QMPHOME}/bin/qmp-config --cflags OUTPUT_VARIABLE QMP_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${QUDA_QMPHOME}/bin/qmp-config --ldflags OUTPUT_VARIABLE QMP_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${QUDA_QMPHOME}/bin/qmp-config --libs OUTPUT_VARIABLE QMP_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
  FIND_LIBRARY(QMP_LIB qmp ${QUDA_QMPHOME}/lib)
  include_directories(SYSTEM ${QUDA_QMPHOME}/include)
  set(COMM_OBJS comm_qmp.cpp)
endif()

if(QUDA_QIO)
  if("${QUDA_QIOHOME}" STREQUAL "" OR "${QUDA_LIMEHOME}" STREQUAL "")
    message( FATAL_ERROR "QUDA_QIOHOME and QUDA_LIMEHOME must be defined when QUDA_QIO is set" )
  endif()
  add_definitions(-DHAVE_QIO)
  set(QIO_UTIL qio_util.cpp qio_field.cpp layout_hyper.c)
  FIND_LIBRARY(QIO_LIB qio ${QUDA_QIOHOME}/lib/)
  FIND_LIBRARY(LIME_LIB lime ${QUDA_LIMEHOME}/lib/)
  include_directories(SYSTEM ${QUDA_QIOHOME}/include)
  include_directories(SYSTEM ${QUDA_LIMEHOME}/include)
endif()

if(QUDA_MAGMA)
  add_definitions(-DMAGMA_LIB -DADD_ -DMAGMA_SETAFFINITY -DGPUSHMEM=300 -DHAVE_CUBLAS -DMAGMA_LIB)
  find_package(OpenMP)
  if("${QUDA_MAGMAHOME}" STREQUAL "")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(MAGMA magma)
    include_directories(SYSTEM ${MAGMA_INCLUDEDIR})
    message("${MAGMA_INCLUDEDIR}")
    find_library(MAGMA ${MAGMA_LIBRARIES} PATH ${MAGMA_LIBRARY_DIRS})
  else()
    # prefer static library
    find_library(MAGMA libmagma.a magma ${QUDA_MAGMAHOME}/lib/)
    # append additional libraries required by magma
    LIST(APPEND MAGMA ${CUDA_cublas_LIBRARY})
    LIST(APPEND MAGMA ${CUDA_cusparse_LIBRARY})
    LIST(APPEND MAGMA ${QUDA_MAGMA_LIBS})
    # and any additional OpenMP linker flags
    SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}" )
    include_directories(SYSTEM ${QUDA_MAGMAHOME}/include)
  endif()  
endif(QUDA_MAGMA)

# This selects arpack or parpack for Multi GPU
if(QUDA_ARPACK)
  add_definitions(-DARPACK_LIB)
  find_package(PkgConfig REQUIRED)
  if(QUDA_QMP OR QUDA_MPI)
    pkg_check_modules(PARPACK QUIET parpack)
    if(NOT PARPACK_FOUND OR QUDA_ARPACK_HOME)
      find_library(PARPACK parpack PATH ${QUDA_ARPACK_HOME})
    else()
      find_library(PARPACK ${PARPACK_LIBRARIES} PATH ${PARPACK_LIBRARY_DIRS})
    endif()
  set(PARPACK_OR_ARPACK ${PARPACK})
  else()
    pkg_check_modules(ARPACK QUIET arpack)
    if(NOT ARPACK_FOUND OR QUDA_ARPACK_HOME)
      find_library(ARPACK arpack PATH ${QUDA_ARPACK_HOME})
    else()
      find_library(ARPACK ${ARPACK_LIBRARIES} PATH ${ARPACK_LIBRARY_DIRS})
    endif()
  set(PARPACK_OR_ARPACK ${ARPACK})
  endif()
endif(QUDA_ARPACK)


# MAX_MULTI_BLAS_N
add_definitions(-DMAX_MULTI_BLAS_N=${QUDA_MAX_MULTI_BLAS_N})

if(QUDA_SSTEP)
  add_definitions(-DSSTEP)
endif()

if (QUDA_MULTIGRID)
  add_definitions(-DCUBLAS_LIB)
  add_definitions(-DGPU_MULTIGRID)
  LIST(APPEND QUDA_LIBS ${CUDA_cublas_LIBRARY})
endif(QUDA_MULTIGRID)

if(QUDA_BLOCKSOLVER)
  add_definitions(-DBLOCKSOLVER)
  find_package(Eigen REQUIRED)
  include_directories(${EIGEN_INCLUDE_DIRS})
endif()

if(QUDA_DEFLATEDSOLVER)
  add_definitions(-DDEFLATEDSOLVER)
  find_package(Eigen REQUIRED)
  include_directories(${EIGEN_INCLUDE_DIRS})
endif()

if(QUDA_ALTRELIABLE)
  add_definitions(-DALTRELIABE)
endif()

if(QUDA_EIGEN)
  add_definitions(-DEIGEN)
  find_package(Eigen REQUIRED)
  include_directories(${EIGEN_INCLUDE_DIRS})
endif()

if(QUDA_POSIX_THREADS)
  add_definitions(-DPTHREADS)
endif()

if(QUDA_DIRAC_WILSON)
  add_definitions(-DGPU_WILSON_DIRAC)
endif(QUDA_DIRAC_WILSON)

if(QUDA_DIRAC_DOMAIN_WALL)
  add_definitions(-DGPU_DOMAIN_WALL_DIRAC)
endif(QUDA_DIRAC_DOMAIN_WALL)

if (QUDA_DIRAC_STAGGERED)
  add_definitions(-DGPU_STAGGERED_DIRAC)
endif(QUDA_DIRAC_STAGGERED)

if(QUDA_DIRAC_CLOVER)
  add_definitions(-DGPU_CLOVER_DIRAC -DGPU_WILSON_DIRAC -DGPU_GAUGE_TOOLS)
endif(QUDA_DIRAC_CLOVER)

if(QUDA_DIRAC_TWISTED_MASS)
  add_definitions(-DGPU_TWISTED_MASS_DIRAC -DGPU_WILSON_DIRAC)
endif(QUDA_DIRAC_TWISTED_MASS)

if(QUDA_DIRAC_TWISTED_CLOVER)
  add_definitions(-DGPU_TWISTED_CLOVER_DIRAC -DGPU_CLOVER_DIRAC -DGPU_TWISTED_MASS_DIRAC -DGPU_WILSON_DIRAC -DGPU_GAUGE_TOOLS)
endif(QUDA_DIRAC_TWISTED_CLOVER)

if(QUDA_DIRAC_NDEG_TWISTED_MASS)
  add_definitions(-DGPU_NDEG_TWISTED_MASS_DIRAC -DGPU_TWISTED_MASS_DIRAC -DGPU_WILSON_DIRAC)
endif(QUDA_DIRAC_NDEG_TWISTED_MASS)

if(QUDA_LINK_ASQTAD)
  add_definitions(-DGPU_FATLINK -DGPU_GAUGE_TOOLS)
endif(QUDA_LINK_ASQTAD)

if(QUDA_LINK_HISQ)
  add_definitions(-DGPU_FATLINK -DGPU_UNITARIZE -DGPU_GAUGE_TOOLS)
endif(QUDA_LINK_HISQ)

if(QUDA_FORCE_GAUGE)
  add_definitions(-DGPU_GAUGE_FORCE -DGPU_GAUGE_TOOLS)
endif(QUDA_FORCE_GAUGE)

if(QUDA_FORCE_ASQTAD)
  add_definitions(-DGPU_FERMION_FORCE -DGPU_GAUGE_TOOLS)
endif(QUDA_FORCE_ASQTAD)

if(QUDA_FORCE_HISQ)
  add_definitions(-DGPU_HISQ_FORCE -DGPU_STAGGERED_OPROD -DGPU_GAUGE_TOOLS)
endif(QUDA_FORCE_HISQ)

if(QUDA_GAUGE_TOOLS)
  add_definitions(-DGPU_GAUGE_TOOLS)
endif(QUDA_GAUGE_TOOLS)

if(QUDA_GAUGE_ALG)
  add_definitions(-DGPU_GAUGE_ALG)
  add_definitions(-DGPU_GAUGE_TOOLS)
  add_definitions(-DGPU_UNITARIZE)
  LIST(APPEND QUDA_LIBS ${CUDA_cufft_LIBRARY} ${CUDA_curand_LIBRARY})
endif(QUDA_GAUGE_ALG)

if(QUDA_DYNAMIC_CLOVER)
  add_definitions(-DDYNAMIC_CLOVER)
endif(QUDA_DYNAMIC_CLOVER)

if(QUDA_MPI_NVTX)
  LIST(APPEND COMM_OBJS nvtx_pmpi.c)
  set(QUDA_NVTX ON)
endif(QUDA_MPI_NVTX)

if(QUDA_INTERFACE_NVTX)
  add_definitions(-DINTERFACE_NVTX)
  set(QUDA_NVTX ON)
endif(QUDA_INTERFACE_NVTX)

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

if(QUDA_INTERFACE_QDP)
  add_definitions(-DBUILD_QDP_INTERFACE)
endif(QUDA_INTERFACE_QDP)

if(QUDA_INTERFACE_MILC)
  add_definitions(-DBUILD_MILC_INTERFACE)
endif(QUDA_INTERFACE_MILC)

if(QUDA_INTERFACE_CPS)
  add_definitions(-DBUILD_CPS_INTERFACE)
endif(QUDA_INTERFACE_CPS)

if(QUDA_INTERFACE_QDPJIT)
add_definitions(-DBUILD_QDPJIT_INTERFACE)
endif(QUDA_INTERFACE_QDPJIT)

if(QUDA_INTERFACE_BQCD)
  add_definitions(-DBUILD_BQCD_INTERFACE)
endif(QUDA_INTERFACE_BQCD)

if(QUDA_INTERFACE_TIFR)
  add_definitions(-DBUILD_TIFR_INTERFACE)
endif(QUDA_INTERFACE_TIFR)

# derive whether we need to build the fortran interface
if(QUDA_INTERFACE_TIFR OR QUDA_INTERFACE_BQCD)
  SET(BUILD_FORTRAN_INTERFACE ON)
  enable_language(Fortran)
endif()



if(QUDA_NUMA_NVML)
  add_definitions(-DNUMA_NVML)
  set(NUMA_AFFINITY_OBJS numa_affinity.cpp)
  find_package(NVML REQUIRED)
  include_directories(NVML_INCLUDE_DIR)
endif(QUDA_NUMA_NVML)


if(QUDA_CONTRACT)
  add_definitions(-DGPU_CONTRACT)
endif(QUDA_CONTRACT)


# COMPILER OPTIONS and BUILD types
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(include)
include_directories(lib)

# QUDA_HASH for tunecache
file(STRINGS ${CUDA_TOOLKIT_INCLUDE}/cuda.h  CUDA_VERSIONLONG REGEX "\#define CUDA_VERSION" )
STRING(REPLACE "\#define CUDA_VERSION " ""  CUDA_VERSIONLONG ${CUDA_VERSIONLONG})
STRING(STRIP CUDA_VERSIONLONG ${CUDA_VERSIONLONG} )
set(HASH cpu_arch=${CPU_ARCH},gpu_arch=${QUDA_GPU_ARCH},cuda_version=${CUDA_VERSIONLONG})


string(REGEX MATCH [Dd][Ee][Bb][Uu][Gg] DEBUG_BUILD ${CMAKE_BUILD_TYPE})

# build up git version
# add -debug to GITVERSION if we build with debug options enabled
if(DEBUG_BUILD)
  if(GITVERSION)
    set(GITVERSION ${GITVERSION}-debug)
  else()
    set(GITVERSION debug)
  endif()
endif()

set(GITVERSION ${GITVERSION}-${QUDA_GPU_ARCH})


# GPU ARCH
STRING(REGEX REPLACE sm_ "" COMP_CAP ${QUDA_GPU_ARCH})
SET(COMP_CAP "${COMP_CAP}0")
add_definitions(-D__COMPUTE_CAPABILITY__=${COMP_CAP})


# NVCC FLAGS independet off build type
if(NOT USING_CUDA_LANG_SUPPORT)
  set(QUDA_NVCC_FLAGS -std c++11 -arch=${QUDA_GPU_ARCH} -ftz=true -prec-div=false -prec-sqrt=false)
else()
  set(QUDA_NVCC_FLAGS "-ftz=true -prec-div=false -prec-sqrt=false")
  set(CMAKE_CUDA_FLAGS "-arch=${QUDA_GPU_ARCH}" CACHE STRING "Flags used by the CUDA compiler" FORCE)
endif()


if(QUDA_VERBOSE_BUILD)
  LIST(APPEND QUDA_NVCC_FLAGS --ptxas-options=-v)
endif(QUDA_VERBOSE_BUILD)

# some clang warnings shouds be warning even when turning warnings into errors
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CLANG_NOERROR "-Wno-error=unused-private-field")
# this is a hack to get colored diagnostics back when using Ninja and clang
    if(CMAKE_GENERATOR MATCHES "Ninja")
      set(CLANG_FORCE_COLOR "-fcolor-diagnostics")
    endif()
endif()

## define CUDA flags when CMake < 3.8
set(CUDA_NVCC_FLAGS_DEVEL ${QUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets -Xcompiler -Wno-unknown-pragmas,-Wno-unused-function,-Wno-unused-local-typedef,-Wno-unused-private-field -O3 -lineinfo CACHE STRING
    "Flags used by the CUDA compiler during regular development builds."
    FORCE )
set(CUDA_NVCC_FLAGS_STRICT ${CUDA_NVCC_FLAGS_DEVEL} CACHE STRING
    "Flags used by the CUDA compiler during strict jenkins builds."
    FORCE )
set(CUDA_NVCC_FLAGS_RELEASE ${QUDA_NVCC_FLAGS} -O3 -w CACHE STRING
    "Flags used by the C++ compiler during release builds."
    FORCE )
set(CUDA_NVCC_FLAGS_HOSTDEBUG ${QUDA_NVCC_FLAGS} -g -lineinfo -DHOST_DEBUG CACHE STRING
    "Flags used by the C++ compiler during host-debug builds."
    FORCE )
set(CUDA_NVCC_FLAGS_DEVICEDEBUG ${QUDA_NVCC_FLAGS} -G CACHE STRING
    "Flags used by the C++ compiler during device-debug builds."
    FORCE )
set(CUDA_NVCC_FLAGS_DEBUG ${QUDA_NVCC_FLAGS} -g -DHOST_DEBUG -G CACHE STRING
    "Flags used by the C++ compiler during full (host+device) debug builds."
    FORCE )

## define CUDA flags when CMake >= 3.8
set(CMAKE_CUDA_FLAGS_DEVEL "${QUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets -Xcompiler -Wno-unknown-pragmas,-Wno-unused-function,-Wno-unused-local-typedef,-Wno-unused-private-field -O3 -lineinfo" CACHE STRING
    "Flags used by the CUDA compiler during regular development builds."
    FORCE )
set(CMAKE_CUDA_FLAGS_STRICT "${CMAKE_CUDA_FLAGS_DEVEL}" CACHE STRING
    "Flags used by the CUDA compiler during strict jenkins builds."
    FORCE )
set(CMAKE_CUDA_FLAGS_RELEASE "${QUDA_NVCC_FLAGS} -O3 -w" CACHE STRING
    "Flags used by the CUDA compiler during release builds."
    FORCE )
set(CMAKE_CUDA_FLAGS_HOSTDEBUG "${QUDA_NVCC_FLAGS} -g -lineinfo -DHOST_DEBUG" CACHE STRING
    "Flags used by the C++ compiler during host-debug builds."
    FORCE )
set(CMAKE_CUDA_FLAGS_DEVICEDEBUG "${QUDA_NVCC_FLAGS} -G" CACHE STRING
    "Flags used by the C++ compiler during device-debug builds."
    FORCE )
set(CMAKE_CUDA_FLAGS_DEBUG "${QUDA_NVCC_FLAGS} -g -DHOST_DEBUG -G" CACHE STRING
    "Flags used by the C++ compiler during full (host+device) debug builds."
    FORCE )

#define CXX FLAGS
set(CMAKE_CXX_FLAGS_DEVEL  "${OpenMP_CXX_FLAGS} -O3 -Wall ${CLANG_FORCE_COLOR}" CACHE STRING
"Flags used by the C++ compiler during regular development builds.")
set(CMAKE_CXX_FLAGS_STRICT  "${OpenMP_CXX_FLAGS} -O3 -Wall -Werror ${CLANG_NOERROR}" CACHE STRING
"Flags used by the C++ compiler during strict jenkins builds.")
set(CMAKE_CXX_FLAGS_RELEASE "${OpenMP_CXX_FLAGS} -O3 -w" CACHE STRING
    "Flags used by the C++ compiler during release builds.")
set(CMAKE_CXX_FLAGS_HOSTDEBUG "${OpenMP_CXX_FLAGS} -Wall -Wno-unknown-pragmas -g -fno-inline -DHOST_DEBUG ${CLANG_FORCE_COLOR}" CACHE STRING
    "Flags used by the C++ compiler during host-debug builds.")
set(CMAKE_CXX_FLAGS_DEVICEDEBUG "${OpenMP_CXX_FLAGS} -Wall -Wno-unknown-pragmas ${CLANG_FORCE_COLOR}" CACHE STRING
    "Flags used by the C++ compiler during device-debug builds.")
set(CMAKE_CXX_FLAGS_DEBUG "${OpenMP_CXX_FLAGS} -Wall -Wno-unknown-pragmas -g -fno-inline -DHOST_DEBUG ${CLANG_FORCE_COLOR}" CACHE STRING
    "Flags used by the C++ compiler during full (host+device) debug builds.")

#define C FLAGS
set(CMAKE_C_FLAGS_DEVEL "-Wall -O3"  CACHE STRING
    "Flags used by the C compiler during regular development builds.")
set(CMAKE_C_FLAGS_STRICT "-Wall -O3 -Werror -Wno-error=unused-private-field"  CACHE STRING
    "Flags used by the C compiler during strict jenkins builds.")
set(CMAKE_C_FLAGS_RELEASE "-Wall -O3 -w" CACHE STRING
    "Flags used by the C compiler during release builds.")
set(CMAKE_C_FLAGS_HOSTDEBUG "-Wall -Wno-unknown-pragmas -g -fno-inline -DHOST_DEBUG" CACHE STRING
    "Flags used by the C compiler during host-debug builds.")
set(CMAKE_C_FLAGS_DEVICEDEBUG "-Wall" CACHE STRING
"Flags used by the C compiler during device-debug builds.")
set(CMAKE_C_FLAGS_DEBUG "-Wall -g -fno-inline -DHOST_DEBUG" CACHE STRING
    "Flags used by the C compiler during full (host+device) debug builds.")

#define FORTRAN FLAGS
set(CMAKE_F_FLAGS -std=c99 CACHE STRING "Fortran Flags")

# make the compiler flags an advanced option for all user defined build types (cmake defined build types are advanced by default )
mark_as_advanced(CMAKE_CXX_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CXX_FLAGS_STRICT)
mark_as_advanced(CMAKE_CXX_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CXX_FLAGS_DEVICEDEBUG)

mark_as_advanced(CUDA_NVCC_FLAGS_DEVEL)
mark_as_advanced(CUDA_NVCC_FLAGS_STRICT)
mark_as_advanced(CUDA_NVCC_FLAGS_HOSTDEBUG)
mark_as_advanced(CUDA_NVCC_FLAGS_DEVICEDEBUG)
mark_as_advanced(CMAKE_CUDA_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CUDA_FLAGS_STRICT)
mark_as_advanced(CMAKE_CUDA_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CUDA_FLAGS_DEVICEDEBUG)

mark_as_advanced(CMAKE_C_FLAGS_DEVEL)
mark_as_advanced(CMAKE_C_FLAGS_STRICT)
mark_as_advanced(CMAKE_C_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_C_FLAGS_DEVICEDEBUG)

mark_as_advanced(CMAKE_F_FLAGS)

### add tests and quda library
add_subdirectory(lib)
add_subdirectory(tests)
