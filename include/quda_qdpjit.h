// -*- c++ -*-

//
// QDP-JIT -- support
// ------------------
//
// This installs C macros for
//
// cudaMalloc(dst, size)
// cudaFree(dst)
//
// which call qdp-jit device memory routines.
//

#ifndef QUDA_QDPJIT
#define QUDA_QDPJIT

#ifndef __CUDACC__

#warning "Using QDP-JIT macros"

#include <qdp_init.h>
#include <qdp_debugmacro.h>
#include <qdp_devicestats.h>
#include <qdp_cuda.h>
#include <qdp_singleton.h>
#include <qdp_cache.h>
#include <qdp_pool_allocator.h>
#include <qdp_dynamic_allocator.h>
#include <qdp_allocators.h>

#define cudaMalloc(dst, size) QDP_allocate(dst, size , __FILE__ , __LINE__ )
#define cudaFree(dst) QDP_free(dst)

inline cudaError_t QDP_allocate(void **dst, size_t size, char * cstrFile , int intLine )
{
  if (!QDP::Allocator::theQDPDeviceAllocator::Instance().allocate_CACHE_spilling( dst , size , cstrFile , intLine ))
    return cudaErrorMemoryAllocation;
  else
    return cudaSuccess;
}

inline void QDP_free(void *dst) 
{
  QDP::Allocator::theQDPDeviceAllocator::Instance().free( dst );
}

#endif // __CUDACC__

#endif // QUDA_MEM
