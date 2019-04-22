# basic setup for cmake
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
project("QUDA" VERSION 0.9.0 LANGUAGES)

message(STATUS "")
message(STATUS "${PROJECT_NAME} ${PROJECT_VERSION} (${GITVERSION}) **")
message(STATUS "cmake version: ${CMAKE_VERSION}")
message(STATUS "Source location: ${CMAKE_SOURCE_DIR}")
message(STATUS "Build location: ${CMAKE_BINARY_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


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
  message (STATUS "Building QUDA for GPU ARCH " "${QUDA_DEFAULT_GPU_ARCH}")
endif()
message(STATUS "")


set(QUDA_GPU_ARCH ${QUDA_DEFAULT_GPU_ARCH} CACHE STRING "set the GPU architecture (sm_20, sm_21, sm_30, sm_35, sm_37, sm_50, sm_52, sm_60, sm_70, sm_75)")
set_property(CACHE QUDA_GPU_ARCH PROPERTY STRINGS sm_20 sm_21 sm_30 sm_35 sm_37 sm_50 sm_52 sm_60 sm_70 sm_75)
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
set(QUDA_PROPAGATE_CXX_FLAGS ON CACHE BOOL "propagate CXX_FLAGS to CUDA host compiler (for cmake >= 3.8)")
set(QUDA_LEGACY_DSLASH OFF CACHE BOOL "enable legacy dslash kernels?")
set(QUDA_TEX ON CACHE BOOL "enable texture reads?")
set(QUDA_NVML OFF CACHE BOOL "use NVML to report CUDA graphics driver version" )
set(QUDA_NUMA_NVML OFF CACHE BOOL "experimental use of NVML to set numa affinity" )
set(QUDA_VERBOSE_BUILD OFF CACHE BOOL "display kernel register useage")
set(QUDA_MAX_MULTI_BLAS_N  "4" CACHE STRING "maximum value to initialize template for multi-blas /-reduce")

set(QUDA_PRECISION "14" CACHE STRING "which precisions to instantiate in QUDA (4-bit number - double, single, half, char)")
set(QUDA_RECONSTRUCT "7" CACHE STRING "which reconstructs to instantiate in QUDA (3-bit number - 18, 13/12, 9/8)")

# Set CTest options
set(QUDA_CTEST_SEP_DSLASH_POLICIES OFF CACHE BOOL "Test Dslash policies separately in ctest instead of only autotuning them.")

set(QUDA_OPENMP OFF CACHE BOOL "enable OpenMP")
set(QUDA_CXX_STANDARD 11 CACHE STRING "set the CXX Standard (11 or 14)")
set_property(CACHE QUDA_CXX_STANDARD PROPERTY STRINGS 11 14)


# NVTX options
set(QUDA_MPI_NVTX OFF CACHE BOOL "add nvtx markup to MPI API calls for the visual profiler")
set(QUDA_INTERFACE_NVTX OFF CACHE BOOL "add nvtx markup to interface calls for the visual profiler")

# features in development
set(QUDA_SSTEP OFF CACHE BOOL "build s-step linear solvers")
set(QUDA_MULTIGRID OFF CACHE BOOL "build multigrid solvers")
set(QUDA_BLOCKSOLVER OFF CACHE BOOL "build block solvers")
set(QUDA_USE_EIGEN OFF CACHE BOOL "use EIGEN library (where optional)")
set(QUDA_DOWNLOAD_EIGEN ON CACHE BOOL "Download Eigen")


option(QUDA_GENERATE_DOXYGEN "generate doxygen documentation")

set(QUDA_JITIFY OFF CACHE BOOL "build QUDA using Jitify")

mark_as_advanced(QUDA_BUILD_ALL_TESTS)
mark_as_advanced(QUDA_BUILD_SHAREDLIB)
mark_as_advanced(QUDA_PROPAGATE_CXX_FLAGS)
mark_as_advanced(QUDA_LEGACY_DSLASH)
mark_as_advanced(QUDA_TEX)
mark_as_advanced(QUDA_NVML)
mark_as_advanced(QUDA_NUMA_NVML)
mark_as_advanced(QUDA_VERBOSE_BUILD)
mark_as_advanced(QUDA_MAX_MULTI_BLAS_N)
mark_as_advanced(QUDA_PRECISION)
mark_as_advanced(QUDA_RECONSTRUCT)
mark_as_advanced(QUDA_CTEST_SEP_DSLASH_POLICIES)
mark_as_advanced(QUDA_CTEST_LAUNCH)
mark_as_advanced(QUDA_CTEST_LAUNCH_ARGS)
mark_as_advanced(QUDA_OPENMP)

mark_as_advanced(QUDA_MPI_NVTX)
mark_as_advanced(QUDA_INTERFACE_NVTX)

mark_as_advanced(QUDA_SSTEP)
mark_as_advanced(QUDA_USE_EIGEN)
mark_as_advanced(QUDA_BLOCKSOVER)
mark_as_advanced(QUDA_CXX_STANDARD)

mark_as_advanced(QUDA_JITIFY)

#######################################################################
# options that are not exposed at all because only one option exists
set(CPU_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})


#######################################################################
# everything below here is processing the setup
#######################################################################
# we need to check for some packages
find_package(PythonInterp)


#######################################################################
## QUDA depends on Eigen
## this part makes sure we can download eigen if it is not found
if (QUDA_DOWNLOAD_EIGEN)
  set(EIGEN_VERSION 3.3.5)

  set(EIGEN_DOWNLOAD_LOCATION ${CMAKE_SOURCE_DIR}/externals/eigen/${EIGEN_VERSION}.tar.bz2)
  set(EIGEN_URL http://bitbucket.org/eigen/eigen/get/${EIGEN_VERSION}.tar.bz2)
  set(EIGEN_SHA 7352bff3ea299e4c7d7fbe31c504f8eb9149d7e685dec5a12fbaa26379f603e2)
  if(NOT EXISTS ${EIGEN_DOWNLOAD_LOCATION} )
    message(STATUS "Checking for Eigen tarball and downloading if necessary.")
  endif()
  FILE(DOWNLOAD ${EIGEN_URL} ${EIGEN_DOWNLOAD_LOCATION} EXPECTED_HASH SHA256=${EIGEN_SHA} STATUS EIGEN_DOWNLOADED)
  list(GET EIGEN_DOWNLOADED 0 EIGEN_DOWNLOADED_CODE)
  list(GET EIGEN_DOWNLOADED 1 EIGEN_DOWNLOADED_MSG)
  if(${EIGEN_DOWNLOADED_CODE})
    message(SEND_ERROR "Could not download Eigen automatically (${EIGEN_DOWNLOADED_MSG}). Please download eigen from ${EIGEN_URL} and save it to ${EIGEN_DOWNLOAD_LOCATION} and try running cmake again.")
  endif()

  include(ExternalProject)
  ExternalProject_Add(Eigen
    URL ${CMAKE_SOURCE_DIR}/externals/eigen/${EIGEN_VERSION}.tar.bz2
    URL_HASH SHA256=${EIGEN_SHA}
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/externals/eigen/
    CONFIGURE_COMMAND ""
    BUILD_COMMAND COMMAND ""
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(Eigen source_dir)
  set(EIGEN_INCLUDE_DIRS ${source_dir})
else()
  # fall back to using find_package
  find_package(Eigen QUIET)
  if(NOT EIGEN_FOUND)
    message(FATAL_ERROR "QUDA requires Eigen (http://eigen.tuxfamily.org). Please either set EIGEN_INCLUDE_DIRS to path to eigen3 include directory, e.g. /usr/local/include/eigen3 or set QUDA_DOWNLOAD_EIGEN to ON to enable automatic download of the necessary components.")
  endif()
endif()
include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})
## Now we hopefully found some way to get eigen to work



# Linux: CMAKE_HOST_SYSTEM_PROCESSOR "x86_64"
# Mac: CMAKE_HOST_SYSTEM_PROCESSOR "x86_64"
# Power: CMAKE_HOST_SYSTEM_PROCESSOR "ppc64le"

## We need to use different optimization flags depending on whether we are on x86 or power
## Note: This only applies to the RELASE build type
## this is just a quick fix and we should probably use https://cmake.org/cmake/help/latest/module/CheckCXXCompilerFlag.html

if(${CPU_ARCH} STREQUAL "x86_64")
  set(CXX_OPT "-march=native")
elseif(${CPU_ARCH} STREQUAL "ppc64le")
  set(CXX_OPT "-Ofast -mcpu=native -mtune=native")
endif()

set(CMAKE_CXX_STANDARD ${QUDA_CXX_STANDARD})
#define CXX FLAGS
set(CMAKE_CXX_FLAGS_DEVEL  "${OpenMP_CXX_FLAGS} -O3 -Wall ${CLANG_FORCE_COLOR}" CACHE STRING
"Flags used by the C++ compiler during regular development builds.")
set(CMAKE_CXX_FLAGS_STRICT  "${OpenMP_CXX_FLAGS} -O3 -Wall -Werror ${CLANG_NOERROR}" CACHE STRING
"Flags used by the C++ compiler during strict jenkins builds.")
set(CMAKE_CXX_FLAGS_RELEASE "${OpenMP_CXX_FLAGS} -O3 -w ${CXX_OPT} " CACHE STRING
    "Flags used by the C++ compiler during release builds.")
set(CMAKE_CXX_FLAGS_HOSTDEBUG "${OpenMP_CXX_FLAGS} -Wall -Wno-unknown-pragmas -g -fno-inline -DHOST_DEBUG ${CLANG_FORCE_COLOR}" CACHE STRING
    "Flags used by the C++ compiler during host-debug builds.")
set(CMAKE_CXX_FLAGS_DEVICEDEBUG "${OpenMP_CXX_FLAGS} -Wall -Wno-unknown-pragmas ${CLANG_FORCE_COLOR}" CACHE STRING
    "Flags used by the C++ compiler during device-debug builds.")
set(CMAKE_CXX_FLAGS_DEBUG "${OpenMP_CXX_FLAGS} -Wall -Wno-unknown-pragmas -g -fno-inline -DHOST_DEBUG ${CLANG_FORCE_COLOR}" CACHE STRING
    "Flags used by the C++ compiler during full (host+device) debug builds.")
enable_language(CXX)




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


enable_language(C)
# do all the build definitions
#

if(QUDA_MPI OR QUDA_QMP)
  add_definitions(-DMULTI_GPU)
  # if we are using MPI and no MPI_<LANG>_COMPILER was specified on the command line check for MPICXX and MPICC environment variables
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
  if(mpimessage)
    message("Found MPICC/MPICXX environment variables. If this is not what you want please use -DMPI_<LANG>_COMPILER and consult the cmake FindMPI documentation.")
  endif()
  find_package(MPI)
else()
  set(COMM_OBJS comm_single.cpp)
endif()

if(QUDA_QDPJIT)
  if(NOT QUDA_QMP)
    message(WARNING "Specifying QUDA_QDPJIT requires use of QUDA_QMP. Please set QUDA_QMP=ON and set QUDA_QMPHOME.")
  endif()
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

  execute_process(COMMAND ${QUDA_QMPHOME}/bin/qmp-config --cflags OUTPUT_VARIABLE QUDA_QMP_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${QUDA_QMPHOME}/bin/qmp-config --ldflags OUTPUT_VARIABLE QUDA_QMP_LDFLAGS_INTERNAL OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${QUDA_QMPHOME}/bin/qmp-config --libs OUTPUT_VARIABLE QUDA_QMP_LIBS_INTERNAL OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(QUDA_QMP_LDFLAGS  ${QUDA_QMP_LDFLAGS_INTERNAL} CACHE STRING "LDFLAGS for QMP - should be derived from qmp-config --ldflags")
  set(QUDA_QMP_LIBS  ${QUDA_QMP_LIBS_INTERNAL} CACHE STRING "LIBS for QMP - should be derived from qmp-config --libs")
  #FIND_LIBRARY(QMP_LIB qmp ${QUDA_QMPHOME}/lib)
  include_directories(SYSTEM ${QUDA_QMPHOME}/include)
  include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
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


# set which precisions to enable
add_definitions(-DQUDA_PRECISION=${QUDA_PRECISION})

# set which precisions to enable
add_definitions(-DQUDA_RECONSTRUCT=${QUDA_RECONSTRUCT})

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
endif()

if(QUDA_JITIFY)
  add_definitions(-DJITIFY)
  find_package(LibDL)
#  find_library(LibDL_Libraries nvToolsExt ${CUDA_TOOLKIT_ROOT_DIR}/lib/)
  LIST(APPEND QUDA_LIBS ${CUDA_nvrtc_LIBRARY})
  LIST(APPEND QUDA_LIBS ${LIBDL_LIBRARIES})
  configure_file(include/jitify_options.hpp.in include/jitify_options.hpp)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/jitify_options.hpp" DESTINATION include/)
endif()

if(QUDA_USE_EIGEN)
  add_definitions(-DEIGEN)
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

if(QUDA_MPI_NVTX)
  LIST(APPEND COMM_OBJS nvtx_pmpi.c)
  set(QUDA_NVTX ON)
endif(QUDA_MPI_NVTX)

if(QUDA_INTERFACE_NVTX)
  add_definitions(-DINTERFACE_NVTX)
  set(QUDA_NVTX ON)
endif(QUDA_INTERFACE_NVTX)

if(QUDA_NVTX)
  LIST(APPEND QUDA_LIBS ${CUDA_nvToolsExt_LIBRARY})
endif(QUDA_NVTX)

if(QUDA_LEGACY_DSLASH)
  add_definitions(-DUSE_LEGACY_DSLASH)
endif(QUDA_LEGACY_DSLASH)

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

if(QUDA_NUMA_NVML)
  add_definitions(-DNUMA_NVML)
  set(NUMA_AFFINITY_OBJS numa_affinity.cpp)
  find_package(NVML REQUIRED)
  include_directories(SYSTEM NVML_INCLUDE_DIR)
endif(QUDA_NUMA_NVML)

if(QUDA_CONTRACT)
  add_definitions(-DGPU_CONTRACT)
endif(QUDA_CONTRACT)


#define FORTRAN FLAGS
set(CMAKE_F_FLAGS -std=c99 CACHE STRING "Fortran Flags")

# derive whether we need to build the fortran interface
if(QUDA_INTERFACE_TIFR OR QUDA_INTERFACE_BQCD)
  SET(BUILD_FORTRAN_INTERFACE ON)
  enable_language(Fortran)
endif()

# CUDA stuff

set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Host compiler to be used by nvcc")
set(CMAKE_CUDA_STANDARD ${QUDA_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED True)
mark_as_advanced(CMAKE_CUDA_HOST_COMPILER)


# NVCC FLAGS independent off build type

set(QUDA_NVCC_FLAGS "-ftz=true -prec-div=false -prec-sqrt=false")
set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets -arch=${QUDA_GPU_ARCH}" CACHE STRING "Flags used by the CUDA compiler" FORCE)
if(QUDA_VERBOSE_BUILD)
  set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets -arch=${QUDA_GPU_ARCH} --ptxas-options=-v" CACHE STRING "Flags used by the CUDA compiler" FORCE)
endif(QUDA_VERBOSE_BUILD)

## define CUDA flags when CMake >= 3.8
set(CMAKE_CUDA_DISABLE_XCOMPILER_WARNINGS "-Wno-unknown-pragmas,-Wno-unused-function,-Wno-unused-local-typedef,-Wno-unused-private-field")
set(CMAKE_CUDA_FLAGS_DEVEL "${QUDA_NVCC_FLAGS} -lineinfo -O3 -Xcompiler ${CMAKE_CUDA_DISABLE_XCOMPILER_WARNINGS}" CACHE STRING
  "Flags used by the CUDA compiler during regular development builds.")
set(CMAKE_CUDA_FLAGS_STRICT "${CMAKE_CUDA_FLAGS_DEVEL}" CACHE STRING
  "Flags used by the CUDA compiler during strict jenkins builds.")
set(CMAKE_CUDA_FLAGS_RELEASE "${QUDA_NVCC_FLAGS} -O3 -w" CACHE STRING
  "Flags used by the CUDA compiler during release builds.")
set(CMAKE_CUDA_FLAGS_HOSTDEBUG "${QUDA_NVCC_FLAGS} -g -lineinfo -DHOST_DEBUG" CACHE STRING
  "Flags used by the C++ compiler during host-debug builds.")
set(CMAKE_CUDA_FLAGS_DEVICEDEBUG "${QUDA_NVCC_FLAGS} -G" CACHE STRING
  "Flags used by the C++ compiler during device-debug builds.")
set(CMAKE_CUDA_FLAGS_DEBUG "${QUDA_NVCC_FLAGS} -g -DHOST_DEBUG -G" CACHE STRING
  "Flags used by the C++ compiler during full (host+device) debug builds." )

find_package(CUDAWrapper)


# We need threads
find_package(Threads REQUIRED)


# COMPILER OPTIONS and BUILD types
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(include)
include_directories(lib)
#if(QUDA_JITIFY)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
#endif()

# QUDA_HASH for tunecache
if(NOT GITVERSION)
  set(GITVERSION ${PROJECT_VERSION})
endif()
file(STRINGS ${CUDA_TOOLKIT_INCLUDE}/cuda.h  CUDA_VERSIONLONG REGEX "\#define CUDA_VERSION" )
STRING(REPLACE "\#define CUDA_VERSION " ""  CUDA_VERSIONLONG ${CUDA_VERSIONLONG})
STRING(STRIP CUDA_VERSIONLONG ${CUDA_VERSIONLONG} )
set(HASH cpu_arch=${CPU_ARCH},gpu_arch=${QUDA_GPU_ARCH},cuda_version=${CUDA_VERSIONLONG})

# this allows simplified running of clang-tidy 
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
   set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()
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

# GPU ARCH
set(GITVERSION ${GITVERSION}-${QUDA_GPU_ARCH})
STRING(REGEX REPLACE sm_ "" COMP_CAP ${QUDA_GPU_ARCH})
SET(COMP_CAP "${COMP_CAP}0")

# make the compiler flags an advanced option for all user defined build types (cmake defined build types are advanced by default )
mark_as_advanced(CMAKE_CUDA_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CUDA_FLAGS_STRICT)
mark_as_advanced(CMAKE_CUDA_FLAGS_RELEASE)
mark_as_advanced(CMAKE_CUDA_FLAGS_DEBUG)
mark_as_advanced(CMAKE_CUDA_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CUDA_FLAGS_DEVICEDEBUG)

mark_as_advanced(CMAKE_CXX_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CXX_FLAGS_STRICT)
mark_as_advanced(CMAKE_CXX_FLAGS_RELEASE)
mark_as_advanced(CMAKE_CXX_FLAGS_DEBUG)
mark_as_advanced(CMAKE_CXX_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CXX_FLAGS_DEVICEDEBUG)

mark_as_advanced(CMAKE_C_FLAGS_DEVEL)
mark_as_advanced(CMAKE_C_FLAGS_STRICT)
mark_as_advanced(CMAKE_C_FLAGS_RELEASE)
mark_as_advanced(CMAKE_C_FLAGS_DEBUG)
mark_as_advanced(CMAKE_C_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_C_FLAGS_DEVICEDEBUG)

mark_as_advanced(CMAKE_F_FLAGS)

configure_file(include/quda_define.h.in include/quda_define.h @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/quda_define.h" DESTINATION include/)

set(BUILDNAME ${HASH})
include( CTest )
### add tests and quda library
add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(doc)
