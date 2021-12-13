cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# set the project name and version
project(QUDAexamples VERSION 1.0.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
message(STATUS "CUDA Compiler is" ${CMAKE_CUDA_COMPILER})
message(STATUS "Compiler ID is " ${CMAKE_CUDA_COMPILER_ID})

# CUDA Wrapper for finding libs etc
if(CMAKE_VERSION VERSION_LESS 3.17)
  find_package(CUDAWrapper)
else()
  # for cmake 3.17+ we rely on
  find_package(CUDAToolkit)
endif()

#Path to QUDA
set(QUDA_BUILD_HOME "" CACHE PATH "Path to QUDA installation")
set(QUDA_SOURCE_HOME "" CACHE PATH "Path to QUDA source")
find_library(QUDA quda ${QUDA_BUILD_HOME}/lib)
find_library(QUDA_TEST quda_test ${QUDA_BUILD_HOME}/tests)

# Multi-GPU options
option(USE_QIO "link to the QIO code (requires QMP)" OFF)
option(USE_QMP "link to the QMP multi-GPU code" OFF)
option(USE_MPI "link to the MPI multi-GPU code" OFF)

if(USE_MPI AND USE_QMP)
  message(SEND_ERROR "Specifying USE_QMP and USE_MPI might result in undefined behavior. If you intend to use QMP set USE_MPI=OFF.")
endif()

if(USE_MPI OR USE_QMP)
  # if we are using MPI and no MPI_<LANG>_COMPILER was specified on the command line check for MPICXX and MPICC
  # environment variables
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
  # we need to enable Fortran if we want to detect MPI_Fortran_COMPILER
  find_package(MPI)
  set(MPI_LIBS MPI::MPI_CXX)
else()
  set(MPI_LIBS "")
endif()

if(USE_QIO)
  find_library(QIO qio ${QUDA_BUILD_HOME}/usqcd/lib)
  find_library(LIME lime ${QUDA_BUILD_HOME}/usqcd/lib)
endif()
if(USE_QMP)
  find_library(QMP qmp ${QUDA_BUILD_HOME}/usqcd/lib)
endif()

# Create library dependency
set(EXAMPLES_LIBS ${QUDA} ${QUDA_TEST})
if(USE_QIO)
  list(APPEND EXAMPLES_LIBS ${QIO} ${LIME})
endif()
if(USE_QMP)
  list(APPEND EXAMPLES_LIBS ${QMP} ${MPI_LIBS})
elseif(USE_MPI)
  list(APPEND EXAMPLES_LIBS ${MPI_LIBS})
endif()

# Create includes
set(EXAMPLES_INC ${QUDA_BUILD_HOME}/include ${QUDA_BUILD_HOME}/include/externals ${QUDA_SOURCE_HOME}/tests/utils ${QUDA_SOURCE_HOME}/tests/host_reference ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Set install path
include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_FULL_LIBDIR})

add_executable(laph_contractions_quda laph_contractions_quda.cpp)
target_link_libraries(laph_contractions_quda ${EXAMPLES_LIBS})
target_include_directories(laph_contractions_quda PUBLIC ${EXAMPLES_INC})

add_executable(correlator_quda correlator_quda.cpp)
target_link_libraries(correlator_quda ${EXAMPLES_LIBS})
target_include_directories(correlator_quda PUBLIC ${EXAMPLES_INC})

add_executable(wilson_flow_quda wilson_flow_quda.cpp)
target_link_libraries(wilson_flow_quda ${EXAMPLES_LIBS})
target_include_directories(wilson_flow_quda PUBLIC ${EXAMPLES_INC})

add_executable(heatbath_quda heatbath_quda.cpp)
target_link_libraries(heatbath_quda ${EXAMPLES_LIBS})
target_include_directories(heatbath_quda PUBLIC ${EXAMPLES_INC})
