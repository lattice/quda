enable_language(CUDA)

find_dependency(CUDAToolkit REQUIRED)

#if( QUDA_NVSHMEM )
#  set(NVSHMEM_LIBS @NVSHMEM_LIBS@)
#  set(NVSHMEM_INCLUDE @NVSHMEM_INCLUDE@)
#  add_library(nvshmem_lib STATIC IMPORTED)
#  set_target_properties(nvshmem_lib PROPERTIES IMPORTED_LOCATION ${NVSHMEM_LIBS})
#  set_target_properties(nvshmem_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#  set_target_properties(nvshmem_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
#  set_target_properties(nvshmem_lib PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES CUDA)
#endif()
