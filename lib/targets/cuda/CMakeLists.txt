# add target specific files / options



target_sources(quda_cpp PRIVATE quda_api.cpp device.cpp malloc.cpp blas_lapack_cublas.cpp comm_target.cpp)

if(QUDA_JITIFY)
  target_sources(quda_cpp PRIVATE jitify_helper.cpp)
endif()

if(QUDA_BACKWARDS)
  set_property(SOURCE malloc.cpp DIRECTORY ${CMAKE_SOURCE_DIR}/lib APPEND PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(SOURCE malloc.cpp DIRECTORY ${CMAKE_SOURCE_DIR}/lib APPEND PROPERTY COMPILE_DEFINITIONS QUDA_BACKWARDSCPP)
endif()

# cuda specific compile options
target_compile_options(
  quda
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
          -ftz=true
          -prec-div=false
          -prec-sqrt=false>
          $<$<COMPILE_LANG_AND_ID:CUDA,NVHPC>:
          -Mflushz
          -Mfpapprox=div
          -Mfpapprox=sqrt>
          $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:
          -fcuda-flush-denormals-to-zero
          -fcuda-approx-transcendentals
          -Xclang -fcuda-allow-variadic-functions>)
target_compile_options(
  quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-Wno-unknown-cuda-version> $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
               -Wno-deprecated-gpu-targets -arch=${QUDA_GPU_ARCH} --expt-relaxed-constexpr>)

target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -ftz=true -prec-div=false -prec-sqrt=false>)
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -Wno-deprecated-gpu-targets
                                    -arch=${QUDA_GPU_ARCH} --expt-relaxed-constexpr>)
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}
                                    --cuda-gpu-arch=${QUDA_GPU_ARCH}>)
target_link_options(quda PUBLIC $<$<CUDA_COMPILER_ID:Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}>)

if(QUDA_VERBOSE_BUILD)
  target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--ptxas-options=-v>)
endif(QUDA_VERBOSE_BUILD)

if (${CMAKE_CUDA_COMPILER_ID} MATCHES "NVHPC" AND NOT ${CMAKE_BUILD_TYPE} MATCHES "DEBUG")
  target_compile_options(quda PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVHPC>:SHELL: -gpu=nodebug" >)
endif()

target_include_directories(quda SYSTEM PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDAToolkit_INCLUDE_DIRS}>)
target_include_directories(quda_cpp SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

# workaround for 10.2
if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA"
   AND ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "10.2"
   AND ${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "10.3")
  target_compile_options(
    quda PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL: -Xcicc \"--Xllc -dag-vectorize-ops=1\" " >)
endif()
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:--cuda-path=${CUDAToolkit_TARGET_DIR}>)
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xfatbin=-compress-all>)
target_include_directories(quda PRIVATE ${CMAKE_SOURCE_DIR}/include/targets/cuda)
target_include_directories(quda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/targets/cuda> $<INSTALL_INTERFACE:include/targets/cuda>)
target_include_directories(quda SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/include/targets/cuda/externals)

# Specific config dependent warning suppressions and lineinfo forwarding
target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
                            -Wreorder
                            $<$<CXX_COMPILER_ID:Clang>: -Xcompiler=-Wno-unused-function -Xcompiler=-Wno-unknown-pragmas>
                            $<$<CXX_COMPILER_ID:GNU>: -Xcompiler=-Wno-unknown-pragmas>
                            $<$<CONFIG:DEVEL>:-Xptxas -warn-lmem-usage,-warn-spills -lineinfo>
                            $<$<CONFIG:STRICT>: -Werror=all-warnings -lineinfo>
                            $<$<CONFIG:HOSTDEBUG>:-lineinfo>
                            $<$<CONFIG:SANITIZE>:-lineinfo>
                            >)

target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVHPC>:
                            -gpu=lineinfo
                            >)

target_compile_options(quda PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:
                            -Wall -Wextra -Wno-unknown-pragmas
                            $<$<CONFIG:STRICT>:-Werror -Wno-error=pass-failed>
                            $<$<CONFIG:SANITIZE>:-fsanitize=address -fsanitize=undefined>
          >)

# malloc.cpp uses both the driver and runtime api So we need to find the CUDA_CUDA_LIBRARY (driver api) or the stub
# version for cmake 3.8 and later this has been integrated into  FindCUDALibs.cmake
target_link_libraries(quda PUBLIC ${CUDA_cuda_driver_LIBRARY})

if(QUDA_NVSHMEM)
set_target_properties(quda_pack PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET quda PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if(QUDA_NVSHMEM)
  target_link_libraries(quda PUBLIC MPI::MPI_C)
  target_compile_definitions(quda PUBLIC NVSHMEM_COMMS)
  if(QUDA_DOWNLOAD_NVSHMEM)
    add_dependencies(quda NVSHMEM)
    add_dependencies(quda_cpp NVSHMEM)
    add_dependencies(quda_pack NVSHMEM)
  endif()
  get_filename_component(NVSHMEM_LIBPATH ${NVSHMEM_LIBS} DIRECTORY)
  target_link_libraries(quda PUBLIC -L${NVSHMEM_LIBPATH} -lnvshmem)
  target_include_directories(quda SYSTEM PUBLIC $<BUILD_INTERFACE:${NVSHMEM_INCLUDE}>)
endif()

if(${QUDA_BUILD_NATIVE_LAPACK} STREQUAL "ON")
  target_link_libraries(quda PUBLIC ${CUDA_cublas_LIBRARY})
endif()

if(QUDA_GAUGE_ALG)
  target_link_libraries(quda PUBLIC ${CUDA_cufft_LIBRARY})
endif(QUDA_GAUGE_ALG)

if(QUDA_JITIFY)
  target_compile_definitions(quda PRIVATE JITIFY)
  find_package(LibDL)
  target_link_libraries(quda PUBLIC ${CUDA_nvrtc_LIBRARY})
  target_link_libraries(quda PUBLIC ${LIBDL_LIBRARIES})
  target_include_directories(quda PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

  configure_file(${CMAKE_SOURCE_DIR}/include/targets/cuda/jitify_options.hpp.in ${${CMAKE_BINARY_DIR}}/include/targets/cuda/jitify_options.hpp)
  install(FILES "${CMAKE_BINARY_DIR}/include/targets/cuda/jitify_options.hpp" DESTINATION include/)
endif()


if(QUDA_INTERFACE_NVTX)
  target_compile_definitions(quda PRIVATE INTERFACE_NVTX)
  set(QUDA_NVTX ON)
endif(QUDA_INTERFACE_NVTX)

if(QUDA_NVTX)
  find_path(NVTX3 "nvtx3/nvToolsExt.h" PATHS ${CUDA_TOOLKIT_INCLUDE} NO_DEFAULT_PATH)
  if(NVTX3)
    target_compile_definitions(quda PRIVATE QUDA_NVTX_VERSION=3)
  else()
    target_link_libraries(quda PUBLIC ${CUDA_nvToolsExt_LIBRARY})
  endif(NVTX3)
endif(QUDA_NVTX)

if(QUDA_NUMA_NVML)
  target_compile_definitions(quda PRIVATE NUMA_NVML)
  target_sources(quda_cpp PRIVATE numa_affinity.cpp)
  find_package(NVML REQUIRED)
  target_include_directories(quda PRIVATE SYSTEM NVML_INCLUDE_DIR)
  target_link_libraries(quda PUBLIC ${NVML_LIBRARY})
endif(QUDA_NUMA_NVML)

if(QUDA_NVML)
  target_link_libraries(quda PUBLIC ${NVML_LIBRARY})
endif()