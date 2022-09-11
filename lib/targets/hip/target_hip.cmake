# ######################################################################################################################
# HIP specific part of CMakeLists
include(CheckLanguage)
check_language(HIP)

set(QUDA_TARGET_HIP ON)

if(DEFINED ENV{QUDA_GPU_ARCH})
  set(QUDA_DEFAULT_GPU_ARCH $ENV{QUDA_GPU_ARCH})
else()
  set(QUDA_DEFAULT_GPU_ARCH gfx908)
endif()

set(QUDA_GPU_ARCH
    ${QUDA_DEFAULT_GPU_ARCH}
    CACHE STRING "set the GPU architecture (gfx906 gfx908 gfx90a)")
set_property(CACHE QUDA_GPU_ARCH PROPERTY STRINGS gfx906 gfx908 gfx90a)

set(CMAKE_HIP_ARCHITECTURES "${QUDA_GPU_ARCH}")
set(GPU_TARGETS "${QUDA_GPU_ARCH}")

mark_as_advanced(GPU_TARGETS)
mark_as_advanced(CMAKE_HIP_ARCHITECTURES)
message(STATUS "Building for GPU Architectures: ${QUDA_GPU_ARCH}")

find_package(HIP)
find_package(hipfft REQUIRED)
find_package(hiprand REQUIRED)
find_package(rocrand REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)
find_package(hipcub REQUIRED)
find_package(rocprim REQUIRED)


# ######################################################################################################################
# define CUDA flags
set(CMAKE_HIP_HOST_COMPILER
    "${CMAKE_CXX_COMPILER}"
    CACHE FILEPATH "Host compiler to be used by hip")
set(CMAKE_HIP_STANDARD ${QUDA_CXX_STANDARD})
set(CMAKE_HIP_STANDARD_REQUIRED True)
mark_as_advanced(CMAKE_HIP_HOST_COMPILER)

set(CMAKE_HIP_FLAGS_DEVEL
    "-g -O3 "
    CACHE STRING "Flags used by the CUDA compiler during regular development builds.")
set(CMAKE_HIP_FLAGS_STRICT
    "-g -O3"
    CACHE STRING "Flags used by the CUDA compiler during strict jenkins builds.")
set(CMAKE_HIP_FLAGS_RELEASE
    "-O3 -w"
    CACHE STRING "Flags used by the CUDA compiler during release builds.")
set(CMAKE_HIP_FLAGS_HOSTDEBUG
    "-g"
    CACHE STRING "Flags used by the C++ compiler during host-debug builds.")
set(CMAKE_HIP_FLAGS_DEBUG
    "-g -G"
    CACHE STRING "Flags used by the C++ compiler during full (host+device) debug builds.")
set(CMAKE_HIP_FLAGS_SANITIZE
    "-g "
    CACHE STRING "Flags used by the C++ compiler during sanitizer debug builds.")

mark_as_advanced(CMAKE_HIP_FLAGS_DEVEL)
mark_as_advanced(CMAKE_HIP_FLAGS_STRICT)
mark_as_advanced(CMAKE_HIP_FLAGS_RELEASE)
mark_as_advanced(CMAKE_HIP_FLAGS_DEBUG)
mark_as_advanced(CMAKE_HIP_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_HIP_FLAGS_SANITIZE)
enable_language(HIP)
message(STATUS "HIP Compiler is" ${CMAKE_HIP_COMPILER})
message(STATUS "Compiler ID is " ${CMAKE_HIP_COMPILER_ID})

# ######################################################################################################################
# CUDA specific QUDA options options
set(QUDA_HETEROGENEOUS_ATOMIC OFF)
mark_as_advanced(QUDA_HETEROGENEOUS_ATOMIC)

# ######################################################################################################################
# CUDA specific variables
set_target_properties(quda PROPERTIES HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})

# QUDA_HASH for tunecache
set(HASH cpu_arch=${CPU_ARCH},gpu_arch=${QUDA_GPU_ARCH},hip_version=${CMAKE_HIP_COMPILER_VERSION})
set(GITVERSION "${PROJECT_VERSION}-${GITVERSION}-${QUDA_GPU_ARCH}")



# ######################################################################################################################
# cuda specific compile options

target_include_directories(quda PRIVATE ${CMAKE_SOURCE_DIR}/include/targets/hip)
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/targets/hip>
                                       $<INSTALL_INTERFACE:include/targets/hip>)


set_source_files_properties(block_orthogonalize.cu PROPERTIES COMPILE_OPTIONS "-mllvm;-pragma-unroll-threshold=4096")

target_compile_options(
  quda
  PRIVATE -Wall
          -Wextra
          -Wno-unknown-pragmas
				  -Wno-unused-result
          $<$<CONFIG:STRICT>:-Werror
          -Wno-error=pass-failed>
          $<$<CONFIG:SANITIZE>:-fsanitize=address
          -fsanitize=undefined>)

set_source_files_properties( ${QUDA_CU_OBJS} PROPERTIES LANGUAGE HIP)
# malloc.cpp uses both the driver and runtime api So we need to find the CUDA_CUDA_LIBRARY (driver api) or the stub
# version for cmake 3.8 and later this has been integrated into  FindCUDALibs.cmake
target_link_libraries(quda PUBLIC hip::hiprand roc::rocrand hip::hipcub roc::rocprim_hip)
target_link_libraries(quda PUBLIC roc::hipblas roc::rocblas)

target_include_directories(quda PUBLIC ${ROCM_PATH}/hipfft/include)
target_link_libraries(quda PUBLIC hip::hipfft)

add_subdirectory(targets/hip)

install(FILES ${CMAKE_SOURCE_DIR}/cmake/find_target_hip_dependencies.cmake DESTINATION lib/cmake/QUDA)
