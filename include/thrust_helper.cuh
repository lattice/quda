#pragma once

#include <malloc_quda.h>

#undef device_malloc
#undef device_free

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/detail/retag.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define device_malloc(size) quda::device_malloc_(__func__, quda::file_name(__FILE__), __LINE__, size)
#define device_free(ptr) quda::device_free_(__func__, quda::file_name(__FILE__), __LINE__, ptr)

// create a tag derived from system::cuda::tag for distinguishing
// our overloads of get_temporary_buffer and return_temporary_buffer
struct my_tag : thrust::system::cuda::tag {};

// overload get_temporary_buffer on my_tag
// its job is to forward allocation requests to g_allocator
template<typename T>
thrust::pair<T*, std::ptrdiff_t> get_temporary_buffer(my_tag, std::ptrdiff_t n)
{
  // ask the allocator for sizeof(T) * n bytes
  T* result = reinterpret_cast<T*>(device_malloc(sizeof(T) * n));

  // return the pointer and the number of elements allocated
  return thrust::make_pair(result,n);
}


// overload return_temporary_buffer on my_tag
// its job is to forward deallocations to g_allocator
// an overloaded return_temporary_buffer should always accompany
// an overloaded get_temporary_buffer
template<typename Pointer>
void return_temporary_buffer(my_tag, Pointer p)
{
  // return the pointer to the allocator
  device_free(thrust::raw_pointer_cast(p));
}
