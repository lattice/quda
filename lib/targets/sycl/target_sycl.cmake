# ######################################################################################################################
# SYCL target
set(QUDA_TARGET_SYCL ON)

# ######################################################################################################################
# SYCL specific part of CMakeLists

if(DEFINED ENV{QUDA_WARP_SIZE})
  set(QUDA_WARP_SIZE_DEFAULT $ENV{QUDA_WARP_SIZE})
else()
  set(QUDA_WARP_SIZE_DEFAULT 16)
endif()
set(QUDA_WARP_SIZE
    ${QUDA_WARP_SIZE_DEFAULT}
    CACHE STRING "SYCL subgroup size (warp size)")
set_property(CACHE QUDA_WARP_SIZE PROPERTY STRINGS 8 16 32)
target_compile_definitions(quda PUBLIC QUDA_WARP_SIZE=${QUDA_WARP_SIZE})
message(STATUS "Using subgroup (warp) size " "${QUDA_WARP_SIZE}")
mark_as_advanced(QUDA_WARP_SIZE)

if(DEFINED ENV{QUDA_MAX_BLOCK_SIZE})
  set(QUDA_MAX_BLOCK_SIZE_DEFAULT $ENV{QUDA_MAX_BLOCK_SIZE})
else()
  set(QUDA_MAX_BLOCK_SIZE_DEFAULT 512)
endif()
set(QUDA_MAX_BLOCK_SIZE
    ${QUDA_MAX_BLOCK_SIZE_DEFAULT}
    CACHE STRING "SYCL max group size (max block size)")
#set_property(CACHE QUDA_MAX_BLOCK_SIZE PROPERTY STRINGS 8 16 32)
target_compile_definitions(quda PUBLIC QUDA_MAX_BLOCK_SIZE=${QUDA_MAX_BLOCK_SIZE})
message(STATUS "Using max group (block) size " "${QUDA_MAX_BLOCK_SIZE}")
mark_as_advanced(QUDA_MAX_BLOCK_SIZE)

if(DEFINED ENV{QUDA_MAX_ARGUMENT_SIZE})
  set(QUDA_MAX_ARGUMENT_SIZE_DEFAULT $ENV{QUDA_MAX_ARGUMENT_SIZE})
else()
  set(QUDA_MAX_ARGUMENT_SIZE_DEFAULT 2048)
endif()
set(QUDA_MAX_ARGUMENT_SIZE
    ${QUDA_MAX_ARGUMENT_SIZE_DEFAULT}
    CACHE STRING "SYCL max argument size")
#set_property(CACHE QUDA_MAX_ARGUMENT_SIZE PROPERTY STRINGS 8 16 32)
target_compile_definitions(quda PUBLIC QUDA_MAX_ARGUMENT_SIZE=${QUDA_MAX_ARGUMENT_SIZE})
message(STATUS "Using max argument size " "${QUDA_MAX_ARGUMENT_SIZE}")
mark_as_advanced(QUDA_MAX_ARGUMENT_SIZE)

option(QUDA_BUILD_NATIVE_FFT "build the native FFT library according to QUDA_TARGET" ON)
if(${QUDA_BUILD_NATIVE_FFT} STREQUAL "ON")
  target_compile_definitions(quda PRIVATE NATIVE_FFT_LIB)
endif()


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
message(STATUS "SYCL compiler is " ${CMAKE_CXX_COMPILER})
message(STATUS "Compiler ID is " ${CMAKE_CXX_COMPILER_ID})

# ######################################################################################################################
# SYCL specific QUDA options
#include(CMakeDependentOption)


# ######################################################################################################################
# SYCL specific variables


# QUDA_HASH for tunecache
#set(HASH cpu_arch=${CPU_ARCH},gpu_arch=${QUDA_GPU_ARCH},sycl_version=${CMAKE_SYCL_COMPILER_VERSION})
set(HASH cpu_arch=${CPU_ARCH},SYCL)
#set(GITVERSION "${PROJECT_VERSION}-${GITVERSION}-${QUDA_GPU_ARCH}")
set(GITVERSION "${PROJECT_VERSION}-${GITVERSION}-SYCL")

# ######################################################################################################################
# sycl specific compile options

if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
  #target_compile_options(quda PUBLIC -fhonor-nan-compares)
  target_compile_options(quda PUBLIC -Wno-tautological-constant-compare)
  target_compile_options(quda PRIVATE -Wno-division-by-zero)
  target_compile_options(quda PRIVATE -Wno-sign-compare)
  target_compile_options(quda PRIVATE -Wno-pass-failed)
  target_compile_options(quda PRIVATE -Wno-unused-parameter)
  #target_compile_options(quda PRIVATE -Wno-unused-but-set-variable)
  #target_compile_options(quda PRIVATE -Wno-error)
  target_compile_options(quda PUBLIC -fsycl)

  set(SYCL_FLAGS "-mllvm -pragma-unroll-threshold=16")

  set(SYCL_LINK_FLAGS -fsycl -fsycl-device-code-split=per_kernel)
endif()

if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xIntelLLVM")
  target_compile_options(quda PUBLIC -fhonor-nan-compares)
  target_compile_options(quda PUBLIC -Wno-tautological-constant-compare)
  target_compile_options(quda PRIVATE -Wno-division-by-zero)
  target_compile_options(quda PRIVATE -Wno-sign-compare)
  target_compile_options(quda PRIVATE -Wno-pass-failed)
  target_compile_options(quda PRIVATE -Wno-unused-parameter)
  #target_compile_options(quda PRIVATE -Wno-unused-but-set-variable)
  #target_compile_options(quda PRIVATE -Wno-error)
  target_compile_options(quda PUBLIC -fsycl)

  set(SYCL_FLAGS "-mllvm -pragma-unroll-threshold=16")

  set(SYCL_LINK_FLAGS -fsycl -fsycl-device-code-split=per_kernel)
endif()

if(DEFINED ENV{SYCL_FLAGS})
  set(SYCL_FLAGS $ENV{SYCL_FLAGS})
endif()

if(DEFINED ENV{SYCL_LINK_FLAGS})
  separate_arguments(SYCL_LINK_FLAGS NATIVE_COMMAND $ENV{SYCL_LINK_FLAGS})
endif()

target_include_directories(quda PRIVATE ${CMAKE_SOURCE_DIR}/include/targets/sycl)
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/targets/sycl>
                                       $<INSTALL_INTERFACE:include/targets/sycl>)

set(SYCL_FLAGS "-x c++ ${SYCL_FLAGS}")
set_source_files_properties(${QUDA_CU_OBJS} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${QUDA_CU_OBJS} PROPERTIES COMPILE_FLAGS ${SYCL_FLAGS})
target_link_options(quda PUBLIC ${SYCL_LINK_FLAGS})

set(SYCL_MKL_LIBRARY "-lmkl_sycl -lmkl_intel_ilp64 -lmkl_core -lmkl_tbb_thread")

if(${QUDA_BUILD_NATIVE_LAPACK} STREQUAL "ON")
  target_link_libraries(quda PUBLIC ${SYCL_MKL_LIBRARY})
endif()

if(${QUDA_BUILD_NATIVE_FFT} STREQUAL "ON")
  target_link_libraries(quda PUBLIC ${SYCL_MKL_LIBRARY})
endif()

add_subdirectory(targets/sycl)
