# ######################################################################################################################
# SYCL target
set(QUDA_TARGET_SYCL ON)

# ######################################################################################################################
# SYCL specific part of CMakeLists

if(DEFINED ENV{QUDA_WARP_SIZE})
  set(QUDA_DEFAULT_WARP_SIZE $ENV{QUDA_WARP_SIZE})
else()
  set(QUDA_DEFAULT_WARP_SIZE 16)
endif()
set(QUDA_WARP_SIZE
    ${QUDA_DEFAULT_WARP_SIZE}
    CACHE STRING "Sycl subgroup size (warp size)")
set_property(CACHE QUDA_WARP_SIZE PROPERTY STRINGS 8 16 32)
target_compile_definitions(quda PUBLIC QUDA_WARP_SIZE=${QUDA_WARP_SIZE})
message(STATUS "Using subgroup (warp) size " "${QUDA_WARP_SIZE}")
mark_as_advanced(QUDA_WARP_SIZE)




# ######################################################################################################################
# define SYCL flags

set(CMAKE_CXX_FLAGS_DEVEL
    " -O3 "
    CACHE STRING "Flags used by the C++ compiler during regular development builds.")
set(CMAKE_CXX_FLAGS_STRICT
    " -O3"
    CACHE STRING "Flags used by the C++ compiler during strict jenkins builds.")
set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 -w"
    CACHE STRING "Flags used by the C++ compiler during release builds.")
set(CMAKE_CXX_FLAGS_HOSTDEBUG
    ""
    CACHE STRING "Flags used by the C++ compiler during host-debug builds.")
set(CMAKE_CXX_FLAGS_DEBUG
    " -G"
    CACHE STRING "Flags used by the C++ compiler during full (host+device) debug builds.")
set(CMAKE_CXX_FLAGS_SANITIZE
    " "
    CACHE STRING "Flags used by the C++ compiler during sanitizer debug builds.")

mark_as_advanced(CMAKE_CXX_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CXX_FLAGS_STRICT)
mark_as_advanced(CMAKE_CXX_FLAGS_RELEASE)
mark_as_advanced(CMAKE_CXX_FLAGS_DEBUG)
mark_as_advanced(CMAKE_CXX_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CXX_FLAGS_SANITIZE)
message(STATUS "Sycl compiler is" ${CMAKE_CXX_COMPILER})
message(STATUS "Compiler ID is " ${CMAKE_CXX_COMPILER_ID})

# ######################################################################################################################
# SYCL specific QUDA options
#include(CMakeDependentOption)


# ######################################################################################################################
# SYCL specific variables


# QUDA_HASH for tunecache
#set(HASH cpu_arch=${CPU_ARCH},gpu_arch=${QUDA_GPU_ARCH},sycl_version=${CMAKE_SYCL_COMPILER_VERSION})
set(HASH cpu_arch=${CPU_ARCH},Sycl)
#set(GITVERSION "${PROJECT_VERSION}-${GITVERSION}-${QUDA_GPU_ARCH}")
set(GITVERSION "${PROJECT_VERSION}-${GITVERSION}-Sycl")

# ######################################################################################################################
# sycl specific compile options
#target_compile_options(quda PRIVATE $<$<CXX_COMPILER_ID:Clang>:-DClang>)
#target_compile_options(quda PRIVATE $<$<CXX_COMPILER_ID:IntelLLVM>:-DIntelLLVM>)

target_compile_options(quda PRIVATE -fhonor-nan-compares)
target_compile_options(quda PRIVATE -Wno-sign-compare)
target_compile_options(quda PRIVATE -mllvm -pragma-unroll-threshold=16)
target_compile_options(quda PRIVATE -Wno-pass-failed)
#target_link_options(quda PUBLIC -fsycl-device-code-split=per_kernel)
target_link_options(quda INTERFACE -fsycl-device-code-split=per_kernel)

set(SYCL_MKL_LIBRARY "-lmkl_sycl -lmkl_intel_ilp64 -lmkl_core -lmkl_tbb_thread")

#target_link_options(quda PUBLIC $<$<SYCL_COMPILER_ID:Clang>: --sycl-path=${SYCLToolkit_TARGET_DIR}>)

#if(QUDA_VERBOSE_BUILD)
#  target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:SYCL,NVIDIA>:--ptxas-options=-v>)
#endif(QUDA_VERBOSE_BUILD)

#if(${CMAKE_SYCL_COMPILER_ID} MATCHES "NVHPC" AND NOT ${CMAKE_BUILD_TYPE} MATCHES "DEBUG")
#  target_compile_options(quda PRIVATE "$<$<COMPILE_LANG_AND_ID:SYCL,NVHPC>:SHELL: -gpu=nodebug" >)
#endif()

#target_include_directories(quda SYSTEM PUBLIC $<$<COMPILE_LANGUAGE:SYCL>:${SYCLToolkit_INCLUDE_DIRS}>)
#target_include_directories(quda_cpp SYSTEM PUBLIC ${SYCLToolkit_INCLUDE_DIRS})

target_include_directories(quda PRIVATE ${CMAKE_SOURCE_DIR}/include/targets/sycl)
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/targets/sycl>
                                       $<INSTALL_INTERFACE:include/targets/sycl>)
#target_include_directories(quda SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/include/targets/sycl/externals)

# Specific config dependent warning suppressions and lineinfo forwarding
#target_compile_options(
#  quda
#  PRIVATE $<$<COMPILE_LANG_AND_ID:SYCL,NVIDIA>:
#          -Wreorder
#          $<$<CXX_COMPILER_ID:Clang>:
#          -Xcompiler=-Wno-unused-function
#          -Xcompiler=-Wno-unknown-pragmas>
#          $<$<CXX_COMPILER_ID:GNU>:
#          -Xcompiler=-Wno-unknown-pragmas>
#          $<$<CONFIG:DEVEL>:-Xptxas
#          -warn-lmem-usage,-warn-spills
#          -lineinfo>
#          $<$<CONFIG:STRICT>:
#          -Werror=all-warnings
#          -lineinfo>
#          $<$<CONFIG:HOSTDEBUG>:-lineinfo>
#          $<$<CONFIG:SANITIZE>:-lineinfo>
#          >)

#target_compile_options(
#  quda
#  PRIVATE $<$<COMPILE_LANG_AND_ID:SYCL,Clang>:
#          -Wall
#          -Wextra
#          -Wno-unknown-pragmas
#          $<$<CONFIG:STRICT>:-Werror
#          -Wno-error=pass-failed>
#          $<$<CONFIG:SANITIZE>:-fsanitize=address
#          -fsanitize=undefined>
#          >)

set_source_files_properties(${QUDA_CU_OBJS} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${QUDA_CU_OBJS} PROPERTIES COMPILE_FLAGS "-x c++")

if(${QUDA_BUILD_NATIVE_LAPACK} STREQUAL "ON")
  target_link_libraries(quda PUBLIC ${SYCL_MKL_LIBRARY})
endif()

if(QUDA_GAUGE_ALG)
  target_link_libraries(quda PUBLIC ${SYCL_MKL_LIBRARY})
endif(QUDA_GAUGE_ALG)

add_subdirectory(targets/sycl)
