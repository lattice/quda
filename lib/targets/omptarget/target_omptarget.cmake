# ######################################################################################################################
# OMPTARGET specific part of CMakeLists
set(QUDA_TARGET_OMPTARGET ON)

# Field order defaults are now supplied by each target.
set(QUDA_ORDER_DOUBLE
    "2"
    CACHE STRING "data order for double precision fields (2 = default, 0 = legacy)")
set(QUDA_ORDER_SINGLE
    "4"
    CACHE STRING "data order for single precision fields (4 = default, 0 = legacy)")
set(QUDA_ORDER_HALF
    "8"
    CACHE STRING "data order for half precision fields (8 = default, 0 = legacy)")
set(QUDA_ORDER_QUARTER
    "8"
    CACHE STRING "data order for quarter precision fields (8 = default, 0 = legacy)")

set(QUDA_HETEROGENEOUS_ATOMIC OFF)
set(QUDA_LARGE_KERNEL_ARG OFF)
set(QUDA_SHARED_MEMORY_SPILL OFF)
set(QUDA_ENABLE_MMA OFF)

if(QUDA_FLUSH_DENORMALS)
  message(FATAL_ERROR "QUDA_FLUSH_DENORMALS is not supported on the OpenMP target")
endif()

if(QUDA_REDUCTION_ALGORITHM STREQUAL "REPRODUCIBLE")
  message(FATAL_ERROR "QUDA_REDUCTION_ALGORITHM=REPRODUCIBLE requires the CUDA accumulator implementation")
endif()

if(QUDA_BUILD_NATIVE_FFT)
  message(STATUS "Native FFT is unavailable on the OpenMP target; using the generic FFT interface")
endif()
set(QUDA_BUILD_NATIVE_FFT OFF)

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

set(QUDA_WARP_SIZE
    32
    CACHE STRING "OpenMP target logical warp size")
set_property(CACHE QUDA_WARP_SIZE PROPERTY STRINGS 8 16 32 64)
target_compile_definitions(quda PUBLIC QUDA_WARP_SIZE=${QUDA_WARP_SIZE})
message(STATUS "Using logical warp size: ${QUDA_WARP_SIZE}")

set(QUDA_MAX_BLOCK_SIZE
    1024
    CACHE STRING "OpenMP target maximum team size (number of threads per team)")
set_property(CACHE QUDA_MAX_BLOCK_SIZE PROPERTY STRINGS 256 512 768 1024)
target_compile_definitions(quda PUBLIC QUDA_MAX_BLOCK_SIZE=${QUDA_MAX_BLOCK_SIZE})
message(STATUS "Using maximum team size: ${QUDA_MAX_BLOCK_SIZE}")

set(QUDA_MAX_SHARED_MEMORY_SIZE
    114688
    CACHE STRING "OpenMP target maximum shared memory size (among threads in a team)")
set_property(
  CACHE QUDA_MAX_SHARED_MEMORY_SIZE
  PROPERTY STRINGS
           32768
           49152
           65536
           81920
           98304
           114688
           131072)
target_compile_definitions(quda PUBLIC QUDA_MAX_SHARED_MEMORY_SIZE=${QUDA_MAX_SHARED_MEMORY_SIZE})
message(STATUS "Using maximum shared memory size: ${QUDA_MAX_SHARED_MEMORY_SIZE}")

option(QUDA_OMPTARGET_THREAD_ARRAY_SLM "Build OpenMP target backend with thread_array on SLM" ON)
mark_as_advanced(QUDA_OMPTARGET_THREAD_ARRAY_SLM)

option(QUDA_OMPTARGET_JIT "Build OpenMP target backend with JIT" OFF)
mark_as_advanced(QUDA_OMPTARGET_JIT)

option(QUDA_OMPTARGET_SIMD16 "Build OpenMP target backend with simd width 16" OFF)
mark_as_advanced(QUDA_OMPTARGET_SIMD16)

string(REPLACE " " "" QUDA_GPU_ARCH_TAG "${QUDA_GPU_ARCH}")

# ######################################################################################################################
# define omptarget flags

if(QUDA_OMPTARGET_JIT)
  set(QUDA_OMPTARGET_FLAGS -fiopenmp -fopenmp-targets=spir64 -fopenmp-version=60
                           "SHELL:-mllvm -vpo-paropt-simulate-get-num-threads-in-target=false")
else()
  # set(QUDA_OMPTARGET_FLAGS -fiopenmp -fopenmp-targets=spir64_gen -fopenmp-version=60 "SHELL:-mllvm
  # -pragma-unroll-threshold=16" "SHELL:-mllvm -vpo-paropt-simulate-get-num-threads-in-target=false")
  # set(QUDA_OMPTARGET_FLAGS -fiopenmp -fopenmp-targets=spir64_gen -fopenmp-device-code-split=per_kernel
  # -fopenmp-max-parallel-link-jobs=64 -fopenmp-version=60 "SHELL:-mllvm
  # -vpo-paropt-simulate-get-num-threads-in-target=false")
  set(QUDA_OMPTARGET_FLAGS -fiopenmp -fopenmp-version=60
                           "SHELL:-mllvm -vpo-paropt-simulate-get-num-threads-in-target=false")
  if(QUDA_OMPTARGET_DEBUG)
    set(QUDA_OMPTARGET_FLAGS ${QUDA_OMPTARGET_FLAGS} -g -fopenmp-targets=spir64_gen=-g0)
  else()
    set(QUDA_OMPTARGET_FLAGS ${QUDA_OMPTARGET_FLAGS} -fopenmp-targets=spir64_gen)
  endif()
endif()

if(QUDA_OMPTARGET_SIMD16)
  set(QUDA_OMPTARGET_FLAGS ${QUDA_OMPTARGET_FLAGS} "SHELL:-mllvm -vpo-paropt-fixed-simd-width=16")
  set(QUDA_GPU_ARCH_TAG "${QUDA_GPU_ARCH_TAG}-SIMD16")
endif()

message(STATUS "Using OpenMP target flags: ${QUDA_OMPTARGET_FLAGS}")

# QUDA_HASH for tunecache
if(QUDA_OMPTARGET_JIT)
  set(HASH cpu_arch=${CPU_ARCH},gpu_arch=JIT,cxx_version=${CMAKE_CXX_COMPILER_VERSION})
  set(GITVERSION "${PROJECT_VERSION}-${GITVERSION}-omptarget:JIT")
else()
  set(HASH cpu_arch=${CPU_ARCH},gpu_arch=${QUDA_GPU_ARCH_TAG},cxx_version=${CMAKE_CXX_COMPILER_VERSION})
  set(GITVERSION "${PROJECT_VERSION}-${GITVERSION}-omptarget")
endif()

# ######################################################################################################################
# omptarget specific compile options

if(NOT QUDA_OMPTARGET_THREAD_ARRAY_SLM)
  target_compile_definitions(quda PUBLIC QUDA_OMPTARGET_THREAD_ARRAY_SIMPLE)
endif()

target_include_directories(quda PRIVATE ${CMAKE_SOURCE_DIR}/include/targets/omptarget)

# We need to overwrite some cuda-ism
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/targets/omptarget>
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

if(QUDA_OMPTARGET_JIT)
  target_link_options(quda PRIVATE ${QUDA_OMPTARGET_FLAGS} -flink-huge-device-code)
else()
  target_link_options(quda PRIVATE ${QUDA_OMPTARGET_FLAGS} -Xopenmp-target-backend "-device ${QUDA_GPU_ARCH}"
                      -flink-huge-device-code)
endif()

set_source_files_properties(${QUDA_CU_OBJS} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${QUDA_CU_OBJS} PROPERTIES COMPILE_FLAGS "-x c++")

# target_link_libraries(quda PUBLIC hip::hiprand roc::rocrand hip::hipcub roc::rocprim_hip) target_link_libraries(quda
# PUBLIC roc::hipblas roc::rocblas)

set(OMPTARGET_MKL_LIBRARY "-lmkl_sycl -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread")

if(QUDA_BUILD_NATIVE_LAPACK)
  target_compile_definitions(quda PRIVATE MKL_ILP64)
  target_link_libraries(quda PUBLIC ${OMPTARGET_MKL_LIBRARY})
endif()

# if(QUDA_GAUGE_ALG) target_include_directories(quda PUBLIC ${ROCM_PATH}/hipfft/include) target_link_libraries(quda
# PUBLIC hip::hipfft) endif(QUDA_GAUGE_ALG)

add_subdirectory(targets/omptarget)
