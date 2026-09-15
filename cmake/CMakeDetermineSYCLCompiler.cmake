if(NOT CMAKE_SYCL_COMPILER)
  set(CMAKE_SYCL_COMPILER ${CMAKE_CXX_COMPILER})
endif()
mark_as_advanced(CMAKE_SYCL_COMPILER)
message(STATUS "The SYCL compiler is " ${CMAKE_SYCL_COMPILER})

if(NOT CMAKE_SYCL_COMPILER_ID_RUN)
  set(CMAKE_SYCL_COMPILER_ID_RUN 1)

  # Try to identify the compiler.
  set(CMAKE_SYCL_COMPILER_ID)
  set(CMAKE_SYCL_PLATFORM_ID)
  file(READ ${CMAKE_ROOT}/Modules/CMakePlatformId.h.in CMAKE_SYCL_COMPILER_ID_PLATFORM_CONTENT)

  set(CMAKE_SYCL_COMPILER_ID_TEST_FLAGS_FIRST)
  set(CMAKE_SYCL_COMPILER_ID_TEST_FLAGS)

  set(CMAKE_CXX_COMPILER_ID_CONTENT "#if defined(__INTEL_LLVM_COMPILER)\n# define COMPILER_ID \"IntelLLVM\"\n")
  string(APPEND CMAKE_CXX_COMPILER_ID_CONTENT "#elif defined(__clang__)\n# define COMPILER_ID \"Clang\"\n")
  string(APPEND CMAKE_CXX_COMPILER_ID_CONTENT "#endif\n")
  include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerId.cmake)
  CMAKE_DETERMINE_COMPILER_ID(SYCL SYCLFLAGS CMakeCXXCompilerId.cpp)

  _cmake_find_compiler_sysroot(SYCL)
endif()


#set(CMAKE_SYCL_COMPILER_ID_TEST_FLAGS_FIRST)
#set(CMAKESYCL_COMPILER_ID_TEST_FLAGS "-c")
#include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerId.cmake)
#CMAKE_DETERMINE_COMPILER_ID(SYCL SYCLFLAGS CMakeCXXCompilerId.cpp)

configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeSYCLCompiler.cmake.in
               ${CMAKE_PLATFORM_INFO_DIR}/CMakeSYCLCompiler.cmake)

set(CMAKE_SYCL_COMPILER_ENV_VAR "SYCL")
