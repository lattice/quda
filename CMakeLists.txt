cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project("QUDA")

find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(include)
include_directories(lib)

add_definitions(-D__COMPUTE_CAPABILITY__=350)
add_definitions(-DQUDA_HASH="123")

set(QUDA_STAGGERED_DSLASH ON CACHE BOOL "build staggered Dslash")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_35,code=sm_35 )
add_subdirectory(lib)
