# HIP Specific CMake
# Enable the language first so the packages come from the compiler's ROCm
# root rather than ROCM_PATH or /opt/rocm.
enable_language(HIP)

if(CMAKE_HIP_COMPILER_ROCM_ROOT)
  set(_quda_rocm_path "${CMAKE_HIP_COMPILER_ROCM_ROOT}")
elseif(DEFINED ROCM_PATH)
  set(_quda_rocm_path "${ROCM_PATH}")
elseif(DEFINED ENV{ROCM_PATH})
  set(_quda_rocm_path "$ENV{ROCM_PATH}")
else()
  set(_quda_rocm_path "/opt/rocm")
endif()

set(CMAKE_MODULE_PATH "${_quda_rocm_path}/lib/cmake" ${CMAKE_MODULE_PATH})
find_dependency(HIP REQUIRED HINTS "${_quda_rocm_path}")
find_dependency(hipfft REQUIRED HINTS "${_quda_rocm_path}")
find_dependency(hiprand REQUIRED HINTS "${_quda_rocm_path}")
find_dependency(rocrand REQUIRED HINTS "${_quda_rocm_path}")
find_dependency(hipblas REQUIRED HINTS "${_quda_rocm_path}")
find_dependency(rocblas REQUIRED HINTS "${_quda_rocm_path}")
find_dependency(hipcub REQUIRED HINTS "${_quda_rocm_path}")
find_dependency(rocprim REQUIRED HINTS "${_quda_rocm_path}")
unset(_quda_rocm_path)
