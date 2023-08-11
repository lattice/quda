# This file is divided into 5 sections:
# 1. Basic setup for cmake 
# 2. QUDA Physics options 
# 3. QUDA Dependency options
# 4. QUDA Advanced options
# 5. QUDA CMake configuration set up
#
# Each section is delineated by START and END. If you make additions to this file,
# please add an appropriate comment and, if necessary, document the addition
# in the QUDA Wiki page: https://github.com/lattice/quda/wiki/QUDA-Build-With-CMake

####################################################################################
# START 1. Basic setup for cmake
####################################################################################
# basic setup for cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Disable gnu exentions
set(CMAKE_CXX_EXTENSIONS OFF)

# Disable in source builds this is only a temporary fix, but for now we need it as
# cmake will otherwise overwrite the existing makefiles.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
# add a directory for cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(cmake/CPM.cmake)

find_package(Git)

# By default we will build DEVEL. The different build types will pass different
# flags to the compiler which may be strict or permissive on warnings, or
# very verbose at run time and/or compile time.
if(DEFINED ENV{QUDA_BUILD_TYPE})
  set(DEFBUILD $ENV{QUDA_BUILD_TYPE})
else()
  set(DEFBUILD "DEVEL")
endif()

set(VALID_BUILD_TYPES
    DEVEL
    RELEASE
    STRICT
    DEBUG
    HOSTDEBUG
    SANITIZE)
set(CMAKE_BUILD_TYPE
  "${DEFBUILD}"
  CACHE STRING "Choose the type of build, options are: ${VALID_BUILD_TYPES}")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${VALID_BUILD_TYPES})

string(TOUPPER ${CMAKE_BUILD_TYPE} CHECK_BUILD_TYPE)
list(FIND VALID_BUILD_TYPES ${CHECK_BUILD_TYPE} BUILD_TYPE_VALID)

if(BUILD_TYPE_VALID LESS 0)
  message(SEND_ERROR "Please specify a valid CMAKE_BUILD_TYPE type! Valid build types are:" "${VALID_BUILD_TYPES}")
endif()

# QUDA may be built to run using CUDA, HIP or SYCL, which we call the
# Target type. By default, the target is CUDA.
if(DEFINED ENV{QUDA_TARGET})
  set(DEFTARGET $ENV{QUDA_TARGET})
else()
  set(DEFTARGET "CUDA")
endif()

set(VALID_TARGET_TYPES CUDA HIP SYCL)
set(QUDA_TARGET_TYPE
  "${DEFTARGET}"
  CACHE STRING "Choose the type of target, options are: ${VALID_TARGET_TYPES}")
set_property(CACHE QUDA_TARGET_TYPE PROPERTY STRINGS CUDA HIP SYCL)

string(TOUPPER ${QUDA_TARGET_TYPE} CHECK_TARGET_TYPE)
list(FIND VALID_TARGET_TYPES ${CHECK_TARGET_TYPE} TARGET_TYPE_VALID)

if(TARGET_TYPE_VALID LESS 0)
  message(SEND_ERROR "Please specify a valid QUDA_TARGET_TYPE type! Valid target types are:" "${VALID_TARGET_TYPES}")
endif()

if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} show
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE IS_GIT_REPOSIITORY
    OUTPUT_QUIET ERROR_QUIET)
  if(${IS_GIT_REPOSIITORY} EQUAL 0)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --abbrev=0
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GITTAG
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    # we use git rev-list and pipe that through wc here. Newer git versions support --count as option to rev-list but
    # that might not always be available
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-list ${GITTAG}..HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMAND wc -l
      OUTPUT_VARIABLE GITCOUNT
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --match 1 --always  --long --dirty
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GITVERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
endif(GIT_FOUND)

# Define the project
project(
  "QUDA"
  VERSION 1.1.0
  LANGUAGES)

# Print the configuration details to stdout
message(STATUS "")
message(STATUS "${PROJECT_NAME} ${PROJECT_VERSION} (${GITVERSION}) **")
message(STATUS "cmake version: ${CMAKE_VERSION}")
message(STATUS "Source location: ${CMAKE_SOURCE_DIR}")
message(STATUS "Build location: ${CMAKE_BINARY_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "QUDA target: ${QUDA_TARGET_TYPE}")

# ######################################################################################################################
# QUDA OPTIONS likely to be changed by users
# ######################################################################################################################

# build options
option(QUDA_DIRAC_DEFAULT_OFF "default value for QUDA_DIRAC_<TYPE> setting" $ENV{QUDA_DIRAC_DEFAULT_OFF})
mark_as_advanced(QUDA_DIRAC_DEFAULT_OFF)
if(QUDA_DIRAC_DEFAULT_OFF)
  set(QUDA_DIRAC_DEFAULT OFF)
else()
  set(QUDA_DIRAC_DEFAULT ON)
endif()

option(QUDA_DIRAC_WILSON "build Wilson Dirac operators" ${QUDA_DIRAC_DEFAULT})
option(QUDA_DIRAC_CLOVER "build clover Dirac operators" ${QUDA_DIRAC_DEFAULT})
option(QUDA_DIRAC_DOMAIN_WALL "build domain wall Dirac operators" ${QUDA_DIRAC_DEFAULT})
option(QUDA_DIRAC_STAGGERED "build staggered Dirac operators" ${QUDA_DIRAC_DEFAULT})
option(QUDA_DIRAC_TWISTED_MASS "build twisted mass Dirac operators" ${QUDA_DIRAC_DEFAULT})
option(QUDA_DIRAC_TWISTED_CLOVER "build twisted clover Dirac operators" ${QUDA_DIRAC_DEFAULT})
option(QUDA_DIRAC_CLOVER_HASENBUSCH "build clover Hasenbusch twist operators" ${QUDA_DIRAC_DEFAULT})
option(QUDA_DIRAC_NDEG_TWISTED_MASS "build non-degenerate twisted mass Dirac operators" ${QUDA_DIRAC_DEFAULT})
option(QUDA_DIRAC_NDEG_TWISTED_CLOVER "build non-degenerate twisted clover Dirac operators" ${QUDA_DIRAC_DEFAULT})

option(QUDA_CONTRACT "build code for bilinear contraction" OFF)
option(QUDA_COVDEV "build code for covariant derivative" OFF)

option(QUDA_LAPLACE "build laplace operator" OFF)
option(QUDA_QIO "build QIO code for binary I/O" OFF)

option(QUDA_SMEAR_GAUSS_TWOLINK "build code for two-link Gaussian smearing" OFF)

# Multi-GPU options
option(QUDA_QMP "build the QMP multi-GPU code" OFF)
option(QUDA_MPI "build the MPI multi-GPU code" OFF)

# ARPACK
option(QUDA_ARPACK "build arpack interface" OFF)
option(QUDA_ARPACK_LOGGING "enable ARPACK logging (not availible for NG)" OFF)

# OpenBLAS
option(QUDA_OPENBLAS "enable OpenBLAS" OFF)

# Interface options
option(QUDA_INTERFACE_QDP "build qdp interface" ON)
option(QUDA_INTERFACE_MILC "build milc interface" ON)
option(QUDA_INTERFACE_CPS "build cps interface" OFF)
option(QUDA_INTERFACE_QDPJIT "build qdpjit interface" OFF)
option(QUDA_INTERFACE_BQCD "build bqcd interface" OFF)
option(QUDA_INTERFACE_TIFR "build tifr interface" OFF)
option(QUDA_INTERFACE_ALL "enable all data-orders triggered by the various interfaces" OFF)

# QDPJIT
option(QUDA_QDPJIT "build QDP-JIT support?" OFF)

set(QUDA_ARPACK_HOME
  ""
  CACHE PATH "path to arpack / parpack")
set(QUDA_OPENBLAS_HOME
    ""
    CACHE PATH "path to OpenBLAS")

# ######################################################################################################################
# QUDA ADVANCED OPTIONS that usually should not be changed by users
# ######################################################################################################################
option(QUDA_BUILD_ALL_TESTS "build tests by default" ON)
option(QUDA_INSTALL_ALL_TESTS "install tests by default" ON)
option(QUDA_BUILD_SHAREDLIB "build quda as a shared lib" ON)

set(QUDA_ORDER_FP "8" CACHE STRING "which data order to use for Wilson, gauge (recon-8/9) and clover fixed-point fields (4, 8)")
set(QUDA_ORDER_SP_MG "2" CACHE STRING "which data order to use for fp32 MG fields (2, 4)")
set(QUDA_ORDER_FP_MG "2" CACHE STRING "which data order to use for fixed-point MG fields (2, 4, 8)")

set_property(CACHE QUDA_ORDER_FP PROPERTY STRINGS 4 8)
set_property(CACHE QUDA_ORDER_SP_MG PROPERTY STRINGS 2 4)
set_property(CACHE QUDA_ORDER_FP_MG PROPERTY STRINGS 2 4 8)

option(QUDA_BUILD_NATIVE_LAPACK "build the native blas/lapack library according to QUDA_TARGET" ON)
option(QUDA_BUILD_NATIVE_FFT "build the native FFT library according to QUDA_TARGET" ON)

# QUDA uses tiling routines to compute certain BLAS routines. The maximum allowable
# tile size is governed by this number. The larger the number, the faster the routines
# will compute, but it will take longer to compile.
set(QUDA_MAX_MULTI_BLAS_N
  "4"
  CACHE STRING "maximum value to initialize template for multi-blas /-reduce")
if(QUDA_MAX_MULTI_BLAS_N GREATER 32)
  message(SEND_ERROR "Maximum QUDA_MAX_MULTI_BLAS_N is 32.")
endif()

set(QUDA_PRECISION
  "14"
  CACHE STRING "which precisions to instantiate in QUDA (4-bit number - double, single, half, quarter)")
set(QUDA_RECONSTRUCT
  "7"
  CACHE STRING "which reconstructs to instantiate in QUDA (3-bit number - 18, 13/12, 9/8)")

option(QUDA_CLOVER_DYNAMIC "Dynamically invert the clover term" ON)
option(QUDA_CLOVER_RECONSTRUCT "set to ON to enable compressed clover storage (requires QUDA_CLOVER_DYNAMIC)" ON)
option(QUDA_CLOVER_CHOLESKY_PROMOTE "Whether to promote the internal precision when inverting the clover term" ON)

# Set CTest options
option(QUDA_CTEST_SEP_DSLASH_POLICIES "Test Dslash policies separately in ctest instead of only autotuning them." OFF)
option(QUDA_CTEST_DISABLE_BENCHMARKS "Disable benchmark test" ON)

option(QUDA_FAST_COMPILE_REDUCE "enable fast compilation in blas and reduction kernels (single warp per reduction)" OFF)
option(QUDA_FAST_COMPILE_DSLASH "enable fast compilation in dslash kernels (~20% perf impact)" OFF)

option(QUDA_ALTERNATIVE_I_TO_F "enable using alternative integer-to-float conversion" OFF)

option(QUDA_OPENMP "enable OpenMP" OFF)
set(QUDA_CXX_STANDARD
    17
    CACHE STRING "set the CXX Standard (14 or 17)")

set_property(CACHE QUDA_CXX_STANDARD PROPERTY STRINGS 14 17)

option(QUDA_BACKWARDS "Enable stacktrace generation using backwards-cpp")

option(QUDA_MULTIGRID "build multigrid solvers" OFF)
option(QUDA_MULTIGRID_DSLASH_PROMOTE "Whether to promote the internal precision when computing the coarse dslash" OFF)

# features in development
option(QUDA_SSTEP "build s-step linear solvers" OFF)
option(QUDA_BLOCKSOLVER "build block solvers" OFF)
option(QUDA_USE_EIGEN "use EIGEN library (where optional)" ON)
option(QUDA_DOWNLOAD_EIGEN "Download Eigen" ON)
option(QUDA_DOWNLOAD_USQCD "Download USQCD software as requested by QUDA_QMP / QUDA_QIO" OFF)
option(QUDA_DOWNLOAD_ARPACK "Download ARPACK-NG software as requested by QUDA_ARPACK" OFF)
option(QUDA_DOWNLOAD_OPENBLAS "Download OpenBLAS software as requested by QUDA_OPENBLAS" OFF)

option(QUDA_GENERATE_DOXYGEN "generate doxygen documentation")

# mark as advanced
mark_as_advanced(QUDA_BUILD_ALL_TESTS)
mark_as_advanced(QUDA_INSTALL_ALL_TESTS)

mark_as_advanced(QUDA_ORDER_FP)
mark_as_advanced(QUDA_ORDER_SP_MG)
mark_as_advanced(QUDA_ORDER_FP_MG)
mark_as_advanced(QUDA_FAST_COMPILE_REDUCE)
mark_as_advanced(QUDA_FAST_COMPILE_DSLASH)

mark_as_advanced(QUDA_ALTERNATIVE_I_TO_F)

mark_as_advanced(QUDA_MAX_MULTI_BLAS_N)
mark_as_advanced(QUDA_PRECISION)
mark_as_advanced(QUDA_RECONSTRUCT)
mark_as_advanced(QUDA_CLOVER_CHOLESKY_PROMOTE)
mark_as_advanced(QUDA_MULTIGRID_DSLASH_PROMOTE)
mark_as_advanced(QUDA_CTEST_SEP_DSLASH_POLICIES)
mark_as_advanced(QUDA_CTEST_LAUNCH)
mark_as_advanced(QUDA_CTEST_LAUNCH_ARGS)
mark_as_advanced(QUDA_OPENMP)

mark_as_advanced(QUDA_BACKWARDS)

mark_as_advanced(QUDA_INTERFACE_ALL)

mark_as_advanced(QUDA_SSTEP)
mark_as_advanced(QUDA_USE_EIGEN)
mark_as_advanced(QUDA_BLOCKSOLVER)
mark_as_advanced(QUDA_CXX_STANDARD)

mark_as_advanced(QUDA_ARPACK_LOGGING)
####################################################################################
# END 4. QUDA advanced options that usually should not be changed by users
####################################################################################

####################################################################################
# START 5. QUDA CMake configuration set up
####################################################################################
# All the CMake code from here and below is not exposed to the user. Its purpose
# is to configure the build given the options above

# Some checks for invalid combinations
if(QUDA_MPI AND QUDA_QMP)
  message(
    SEND_ERROR
    "Specifying QUDA_QMP and QUDA_MPI might result in undefined behavior. If you intend to use QMP set QUDA_MPI=OFF.")
endif()


if(QUDA_NVSHMEM AND NOT (QUDA_QMP OR QUDA_MPI))
  message(SEND_ERROR "Specifying QUDA_NVSHMEM requires either QUDA_QMP or QUDA_MPI.")
endif()

if(QUDA_SMEAR_GAUSS_TWOLINK AND NOT QUDA_DIRAC_STAGGERED)
  message(SEND_ERROR "Specifying QUDA_SMEAR_GAUSS_TWOLINK requires QUDA_DIRAC_STAGGERED.")
endif()

# COMPILER FLAGS Linux: CMAKE_HOST_SYSTEM_PROCESSOR "x86_64" Mac: CMAKE_HOST_SYSTEM_PROCESSOR "x86_64" Power:
# CMAKE_HOST_SYSTEM_PROCESSOR "ppc64le"

# We need to use different optimization flags depending on whether we are on x86 or
# power Note: This only applies to the RELEASE build type this is just a quick fix
# and we should probably use
# https://cmake.org/cmake/help/latest/module/CheckCXXCompilerFlag.html

set(CPU_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
if(${CPU_ARCH} STREQUAL "x86_64")
  set(CXX_OPT "-mtune=native")
elseif(${CPU_ARCH} STREQUAL "ppc64le")
  set(CXX_OPT "-Ofast -mcpu=native -mtune=native")
endif()

set(CMAKE_CXX_STANDARD ${QUDA_CXX_STANDARD})

set(CMAKE_CXX_FLAGS_DEVEL
  "-g -O3"
  CACHE STRING "Flags used by the C++ compiler during regular development builds.")
set(CMAKE_CXX_FLAGS_STRICT
  "-O3"
  CACHE STRING "Flags used by the C++ compiler during strict jenkins builds.")
set(CMAKE_CXX_FLAGS_RELEASE
  "-O3 ${CXX_OPT}"
  CACHE STRING "Flags used by the C++ compiler during release builds.")
set(CMAKE_CXX_FLAGS_HOSTDEBUG
    "-g"
    CACHE STRING "Flags used by the C++ compiler during host-debug builds.")
set(CMAKE_CXX_FLAGS_DEBUG
    "-g -fno-inline"
    CACHE STRING "Flags used by the C++ compiler during full (host+device) debug builds.")
set(CMAKE_CXX_FLAGS_SANITIZE
    "-g -fno-inline \"-fsanitize=address,undefined\""
    CACHE STRING "Flags used by the C++ compiler during santizer debug builds.")

enable_language(CXX)

# define C FLAGS
set(CMAKE_C_FLAGS_DEVEL
  "-g -O3"
  CACHE STRING "Flags used by the C compiler during regular development builds.")
set(CMAKE_C_FLAGS_STRICT
  "-O3"
  CACHE STRING "Flags used by the C compiler during strict jenkins builds.")
set(CMAKE_C_FLAGS_RELEASE
  "-O3"
  CACHE STRING "Flags used by the C compiler during release builds.")
set(CMAKE_C_FLAGS_HOSTDEBUG
    "-g"
    CACHE STRING "Flags used by the C compiler during host-debug builds.")
set(CMAKE_C_FLAGS_DEBUG
  "-g -fno-inline"
  CACHE STRING "Flags used by the C compiler during full (host+device) debug builds.")
set(CMAKE_C_FLAGS_SANITIZE
    "-g -fno-inline \"-fsanitize=address,undefined\""
    CACHE STRING "Flags used by the C compiler during sanitizer debug builds.")


enable_language(C)

if(QUDA_INTERFACE_TIFR
    OR QUDA_INTERFACE_BQCD
    OR QUDA_ARPACK
    OR QUDA_OPENBLAS)
  set(BUILD_FORTRAN_INTERFACE ON)
  enable_language(Fortran)
endif()

# define LINKER FLAGS
set(CMAKE_EXE_LINKER_FLAGS_SANITIZE
    "-fsanitize=address,undefined"
    CACHE STRING "Flags used by the linker during sanitizer debug builds.")

if(QUDA_CLOVER_RECONSTRUCT AND NOT QUDA_CLOVER_DYNAMIC)
  message(SEND_ERROR "QUDA_CLOVER_RECONSTRUCT requires QUDA_CLOVER_DYNAMIC)")
endif()

find_package(Threads REQUIRED)
if(QUDA_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# ######################################################################################################################
# Handle dependencies
# ######################################################################################################################
if(QUDA_BUILD_SHAREDLIB)
  set(BUILD_SHARED_LIBS ON)
endif()

# ######################################################################################################################
# Eigen
# ######################################################################################################################
add_library(Eigen INTERFACE IMPORTED)
# set(CPM_USE_LOCAL_PACKAGES TRUE)
if(QUDA_DOWNLOAD_EIGEN)
  set(QUDA_EIGEN_VERSION
      "3.4.0"
      CACHE STRING "Eigen use for QUDA_DOWNLOAD_EIGEN")
  mark_as_advanced(QUDA_EIGEN_VERSION)
  CPMAddPackage(
    NAME
    Eigen
    VERSION
    ${QUDA_EIGEN_VERSION}
    URL
    https://gitlab.com/libeigen/eigen/-/archive/${QUDA_EIGEN_VERSION}/eigen-${QUDA_EIGEN_VERSION}.tar.bz2
    DOWNLOAD_ONLY
    YES)
  target_include_directories(Eigen SYSTEM INTERFACE ${Eigen_SOURCE_DIR})
  install(DIRECTORY ${Eigen_SOURCE_DIR}/Eigen TYPE INCLUDE)
else()
  # fall back to using find_package
  find_package(Eigen QUIET)
  if(NOT EIGEN_FOUND)
    message(
      FATAL_ERROR
      "QUDA requires Eigen (http://eigen.tuxfamily.org). Please either set EIGEN_INCLUDE_DIRS to path to eigen3 include directory, e.g. /usr/local/include/eigen3 or set QUDA_DOWNLOAD_EIGEN to ON to enable automatic download of the necessary components."
      )
  endif()
  target_include_directories(Eigen SYSTEM INTERFACE ${EIGEN_INCLUDE_DIRS})
endif()

# ######################################################################################################################
# MPI
# ######################################################################################################################
# we need to enable Fortran if we want to detect MPI_Fortran_COMPILER
if(QUDA_ARPACK OR QUDA_OPENBLAS)
  enable_language(Fortran)
endif()

if(QUDA_MPI OR QUDA_QMP)
  # if we are using MPI and no MPI_<LANG>_COMPILER was specified on the command line
  # check for MPICXX and MPICC environment variables
  if( NOT QUDA_SPACK_BUILD )
    if((NOT MPI_CXX_COMPILER) AND DEFINED ENV{MPICXX})
      set(MPI_CXX_COMPILER $ENV{MPICXX})
      set(mpimessage True)
      message(STATUS "Found environment variable MPICXX. Using it for MPI detection: $ENV{MPICXX}")
    endif()
    if((NOT MPI_C_COMPILER) AND DEFINED ENV{MPICC})
      message(STATUS "Found environment variable MPICC. Using it for MPI detection: $ENV{MPICC}")
      set(MPI_C_COMPILER $ENV{MPICC})
      set(mpimessage True)
    endif()
    # I think we don't use that at all but
    if((NOT MPI_Fortran_COMPILER) AND DEFINED ENV{MPIFORT})
      message(STATUS "Found environment variable MPIFORT. Using it for MPI detection: $ENV{MPIFORT}")
      set(MPI_Fortran_COMPILER $ENV{MPIFORT})
      set(mpimessage True)
    endif()
    if(mpimessage)
      message(
        "Found MPIFORT/MPICC/MPICXX environment variables. If this is not what you want please use -DMPI_<LANG>_COMPILER and consult the cmake FindMPI documentation."
        )
    endif()
  endif()
  find_package(MPI REQUIRED)
endif()

# ######################################################################################################################
# USQCD 
# ######################################################################################################################
# We might only want to do that if using QUDA_DOWNLOAD_USQCD, but this does not work if not set on the initial run
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      ${CMAKE_BINARY_DIR}/usqcd
      CACHE PATH "..." FORCE)
endif()

# ######################################################################################################################
# rpath handling before adding dependencies
# ######################################################################################################################

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")

# ######################################################################################################################
# QMP
# ######################################################################################################################
if(QUDA_QMP)
  if(QUDA_DOWNLOAD_USQCD)
    set(QUDA_QMP_TAG
        "3010fef"
        CACHE STRING "Git tag to use for QMP when using QUDA_DOWNLOAD_USQCD")
    mark_as_advanced(QUDA_QMP_TAG)  
    CPMAddPackage(
      NAME
      QMP
      GITHUB_REPOSITORY
      usqcd-software/qmp
      GIT_TAG
      ${QUDA_QMP_TAG}
      OPTIONS
      "QMP_MPI ON"
      "QMP_TESTING OFF"
      "QMP_BUILD_DOCS OFF")
  else()
    find_package(QMP REQUIRED) 
  endif()
endif()

# ######################################################################################################################
# QIO
# ######################################################################################################################
if(QUDA_QIO)
  if(NOT QUDA_QMP)
    message(FATAL_ERROR "Use of QIO (via QUDA_QIO=ON) requires QMP. Please set QUDA_QMP=ON.")
  endif()
  if(QUDA_DOWNLOAD_USQCD)
    set(QUDA_QIO_TAG
        "qio3-0-0"
        CACHE STRING "Git tag to use for QIO when using QUDA_DOWNLOAD_USQCD")
    mark_as_advanced(QUDA_QIO_TAG)
    CPMAddPackage(
      NAME
      QIO
      GITHUB_REPOSITORY
      usqcd-software/qio
      GIT_TAG
      ${QUDA_QIO_TAG}
      OPTIONS 
        "QIO_ENABLE_PARALLEL_BUILD ON"
        "QIO_ENABLE_QMP_ROUTE OFF" 
        "QIO_ENABLE_OUTPUT_BUFFERING ON" 
      "QIO_DML_BUF_BYTES 33554432")
  else()
    find_package(QIO REQUIRED)
  endif()
endif()

# ######################################################################################################################
# USQCD QDPJIT
# ######################################################################################################################
 if(QUDA_QDPJIT)
   if(NOT QUDA_QMP)
     message(SEND_ERROR "Specifying QUDA_QDPJIT requires use of QUDA_QMP. Please set QUDA_QMP=ON and set QMP_DIR to the location of  the QMPConfig.cmake file ( usually <qmp-install-dir>/lib/cmake/QMP ) or add the QMP installation directory to your CMAKE_PREFIX_PATH")
   endif()
   find_package(QDPXX REQUIRED)
   if( NOT ${QDP_IS_QDPJIT} EQUAL 1 )
     message(FATAL_ERROR "Found QDPXX but it is not QDP-JIT. Please specify with QDPXX_DIR pointing to the right path or add to the CMAKE_PREFIX_PATH")
   endif()
   set(QUDA_INTERFACE_QDPJIT ON)
 endif()


# ######################################################################################################################
# ARPACK
# ######################################################################################################################
if(QUDA_ARPACK)

  if(QUDA_MPI OR QUDA_QMP)
    set(ARPACK_MPI ON)
  else()
    set(ARPACK_MPI OFF)
  endif()

  if(QUDA_DOWNLOAD_ARPACK)
    CPMAddPackage(
      NAME
      ARPACK-NG
      GITHUB_REPOSITORY
      opencollab/arpack-ng
      GIT_TAG 3.8.0
      OPTIONS
      "EXAMPLES OFF ON"
      "MPI ${ARPACK_MPI}"
      "BLAS_openblas_LIBRARY ${OpenBLAS_LIBS}"
    )
    add_library(ARPACK::arpack ALIAS arpack)
    if(ARPACK_MPI)
      add_library(ARPACK::parpack ALIAS parpack)
    endif()
else(QUDA_DOWNLOAD_ARPACK)
    find_package(PkgConfig REQUIRED)
    # We always need the serial library
    pkg_check_modules(ARPACK QUIET arpack)
    if(NOT ARPACK_FOUND OR QUDA_ARPACK_HOME)
      find_library(ARPACK arpack PATH ${QUDA_ARPACK_HOME})
    else()
      find_library(ARPACK ${ARPACK_LIBRARIES} PATH ${ARPACK_LIBRARY_DIRS})
    endif()
    # Link the parallel library if required
    if(QUDA_MPI OR QUDA_QMP)
      pkg_check_modules(PARPACK QUIET parpack)
      if(NOT PARPACK_FOUND OR QUDA_ARPACK_HOME)
        find_library(PARPACK parpack PATH ${QUDA_ARPACK_HOME})
      else()
        find_library(PARPACK ${PARPACK_LIBRARIES} PATH ${PARPACK_LIBRARY_DIRS})
      endif()
    endif()
  endif(QUDA_DOWNLOAD_ARPACK)
endif(QUDA_ARPACK)

# ######################################################################################################################
# OPENBLAS
# ######################################################################################################################
if(QUDA_OPENBLAS)
  enable_language(Fortran)

  if(QUDA_DOWNLOAD_OPENBLAS)
  #TODO: switch to CPM
    include(GNUInstallDirs)
    include(ExternalProject)
    ExternalProject_Add(
      OPENBLAS
      GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
      GIT_TAG v0.3.10
      GIT_SHALLOW YES
      PREFIX openblas
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> 
      CMAKE_GENERATOR "Unix Makefiles")
    ExternalProject_Get_Property(OPENBLAS INSTALL_DIR)
    set(QUDA_OPENBLAS_HOME ${INSTALL_DIR})
    add_library(openblas STATIC IMPORTED)
    add_dependencies(openblas OPENBLAS)
    set_target_properties(openblas PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES Fortran)
    set_target_properties(openblas PROPERTIES IMPORTED_LOCATION
                                               ${QUDA_OPENBLAS_HOME}/${CMAKE_INSTALL_LIBDIR}/libopenblas.a)
  else(QUDA_DOWNLOAD_OPENBLAS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENBLAS QUIET openblas)
    if(NOT OPENBLAS_FOUND OR QUDA_OPENBLAS_HOME)
      find_library(OPENBLAS openblas PATH ${QUDA_OPENBLAS_HOME})
    else()
      find_library(OPENBLAS ${OPENBLAS_LIBRARIES} PATH ${OPENBLAS_LIBRARY_DIRS})
    endif()    
  endif(QUDA_DOWNLOAD_OPENBLAS)
endif(QUDA_OPENBLAS)

# ######################################################################################################################
# BACKWARDS
# ######################################################################################################################
if(QUDA_BACKWARDS)
  include(FetchContent)
  FetchContent_Declare(
    backward-cpp
    GIT_REPOSITORY https://github.com/bombela/backward-cpp.git
    GIT_TAG v1.6
    GIT_SHALLOW ON)
  FetchContent_GetProperties(backward-cpp)
  if(NOT backward-cpp_POPULATED)
    FetchContent_Populate(backward-cpp)
  endif()
  include(${backward-cpp_SOURCE_DIR}/BackwardConfig.cmake)
endif()

# ######################################################################################################################
# ADVANCED setup 
# ######################################################################################################################

# this allows simplified running of clang-tidy
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()


# make the compiler flags an advanced option for all user defined build types (cmake defined build types are advanced by
# default )
mark_as_advanced(CMAKE_CXX_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CXX_FLAGS_STRICT)
mark_as_advanced(CMAKE_CXX_FLAGS_RELEASE)
mark_as_advanced(CMAKE_CXX_FLAGS_DEBUG)
mark_as_advanced(CMAKE_CXX_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CXX_FLAGS_SANITIZE)

mark_as_advanced(CMAKE_C_FLAGS_DEVEL)
mark_as_advanced(CMAKE_C_FLAGS_STRICT)
mark_as_advanced(CMAKE_C_FLAGS_RELEASE)
mark_as_advanced(CMAKE_C_FLAGS_DEBUG)
mark_as_advanced(CMAKE_C_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_C_FLAGS_SANITIZE)
mark_as_advanced(CMAKE_F_FLAGS)

mark_as_advanced(CMAKE_EXE_LINKER_FLAGS_SANITIZE)

# enable ctest
include(CTest)

# add tests, utils, reference, and quda library
add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(doc)
