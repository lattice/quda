#pragma once

#include <malloc_quda.h>

#undef device_malloc
#undef device_free

// this is temporary addition until we remove CUB from QUDA and rely on the CUDA toolkit version
#define THRUST_IGNORE_CUB_VERSION_CHECK

// ensures we use shfl_sync and not shfl when compiling with clang
#if defined(__clang__) && defined(__CUDA__) && CUDA_VERSION >= 9000
#define CUB_USE_COOPERATIVE_GROUPS
#endif

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define device_malloc(size) quda::device_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define device_free(ptr) quda::device_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)

/**
   Allocator helper class which allows us to redirect thrust temporary
   alocations to use QUDA's pool memory
 */
class thrust_allocator
{
public:
  // just allocate bytes
  typedef char value_type;

  thrust_allocator() {}
  ~thrust_allocator() { }

  char *allocate(std::ptrdiff_t num_bytes) { return reinterpret_cast<char*>(pool_device_malloc(num_bytes)); }
  void deallocate(char *ptr, size_t n) { pool_device_free(ptr); }

};
