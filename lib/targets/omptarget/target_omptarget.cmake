# ######################################################################################################################
# OMPTARGET specific part of CMakeLists
set(QUDA_TARGET_OMPTARGET ON)

if(DEFINED ENV{QUDA_GPU_ARCH})
  set(QUDA_DEFAULT_GPU_ARCH $ENV{QUDA_GPU_ARCH})
else()
  set(QUDA_DEFAULT_GPU_ARCH xehp)
endif()

set(QUDA_GPU_ARCH
    ${QUDA_DEFAULT_GPU_ARCH}
    CACHE STRING "set the GPU architecture (xehp)")
set_property(CACHE QUDA_GPU_ARCH PROPERTY STRINGS xehp)

message(STATUS "Building for GPU Architectures: ${QUDA_GPU_ARCH}")

set(QUDA_WARP_SIZE 16 CACHE STRING "OpenMP target logical warp size")
set_property(CACHE QUDA_WARP_SIZE PROPERTY STRINGS 8 16 32 64)
target_compile_definitions(quda PUBLIC QUDA_WARP_SIZE=${QUDA_WARP_SIZE})
message(STATUS "Using logical warp size: ${QUDA_WARP_SIZE}")

set(QUDA_MAX_BLOCK_SIZE 1024 CACHE STRING "OpenMP target maximum team size (number of threads per team)")
set_property(CACHE QUDA_MAX_BLOCK_SIZE PROPERTY STRINGS 256 512 768 1024)
target_compile_definitions(quda PUBLIC QUDA_MAX_BLOCK_SIZE=${QUDA_MAX_BLOCK_SIZE})
message(STATUS "Using maximum team size: ${QUDA_MAX_BLOCK_SIZE}")

set(QUDA_MAX_SHARED_MEMORY_SIZE 393216 CACHE STRING "OpenMP target maximum shared memory size (among threads in a team)")
set_property(CACHE QUDA_MAX_SHARED_MEMORY_SIZE PROPERTY STRINGS 36864 40960 45056 49152 65536 98304 131072 196608 262144 393216 524288)
target_compile_definitions(quda PUBLIC QUDA_MAX_SHARED_MEMORY_SIZE=${QUDA_MAX_SHARED_MEMORY_SIZE})
message(STATUS "Using maximum shared memory size: ${QUDA_MAX_SHARED_MEMORY_SIZE}")

# ######################################################################################################################
# define omptarget flags
set(QUDA_OMPTARGET_FLAGS -fiopenmp -fopenmp-targets=spir64_gen -fopenmp-version=51 -mllvm -vpo-paropt-simulate-get-num-threads-in-target=false)
message(STATUS "Using OpenMP target flags: ${QUDA_OMPTARGET_FLAGS}")

# QUDA_HASH for tunecache
set(HASH cpu_arch=${CPU_ARCH},gpu_arch=${QUDA_GPU_ARCH},cxx_version=${CMAKE_CXX_COMPILER_VERSION})
set(GITVERSION "${PROJECT_VERSION}-${GITVERSION}-omptarget:${QUDA_GPU_ARCH}")

# ######################################################################################################################
# omptarget specific compile options

target_include_directories(quda PRIVATE
  ${CMAKE_SOURCE_DIR}/include/targets/omptarget)

# We need to overwrite some cuda-ism
target_include_directories(quda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/targets/omptarget>
  $<INSTALL_INTERFACE:include/targets/omptarget>)
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

target_compile_options(quda PRIVATE ${QUDA_OMPTARGET_FLAGS})

target_link_options(quda PRIVATE ${QUDA_OMPTARGET_FLAGS} -Xopenmp-target-backend "-device ${QUDA_GPU_ARCH}")

set_source_files_properties( ${QUDA_CU_OBJS} PROPERTIES LANGUAGE CXX)
set_source_files_properties( ${QUDA_CU_OBJS} PROPERTIES COMPILE_FLAGS "-x c++")

# target_link_libraries(quda PUBLIC hip::hiprand roc::rocrand hip::hipcub roc::rocprim_hip)
# target_link_libraries(quda PUBLIC roc::hipblas roc::rocblas)

# if(QUDA_GAUGE_ALG)
#   target_include_directories(quda PUBLIC ${ROCM_PATH}/hipfft/include)
#   target_link_libraries(quda PUBLIC hip::hipfft)
# endif(QUDA_GAUGE_ALG)

add_subdirectory(targets/omptarget)