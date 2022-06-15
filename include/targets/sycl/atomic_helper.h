#pragma once

#include <array.h>

/**
   @file atomic_helper.h

   @section Provides definitions of atomic functions that are used in QUDA.
 */

//inline constexpr auto mo = sycl::memory_order::relaxed;
//inline constexpr auto mo = sycl::ext::oneapi::memory_order::acq_rel;
//inline constexpr auto mo = memory_order::seq_cst;
inline constexpr auto mo = sycl::memory_order::acq_rel;

//inline constexpr auto ms = sycl::ext::oneapi::memory_scope::system;
//inline constexpr auto ms = sycl::memory_scope::system;
//inline constexpr auto ms = sycl::ext::oneapi::memory_scope::device;
inline constexpr auto ms = sycl::memory_scope::device;

//inline constexpr auto as = sycl::access::address_space::generic_space;
inline constexpr auto as = sycl::access::address_space::global_space;

template <typename T>
//using atomicRef = sycl::ext::oneapi::atomic_ref<T,mo,ms,as>;
using atomicRef = sycl::atomic_ref<T,mo,ms,as>;

template <typename T>
static inline atomicRef<T> makeAtomicRef(T *address) {
  return atomicRef<T>(*address);
}

static inline uint __float_as_uint(float x) {
  return *reinterpret_cast<uint*>(&x);
}

static inline int atomicAdd(int *address, int val)
{
  auto ar = makeAtomicRef(address);
  auto old = ar.fetch_add(val);
  return old;
}

static inline unsigned int atomicAdd(unsigned int *address, unsigned int val)
{
  auto ar = makeAtomicRef(address);
  auto old = ar.fetch_add(val);
  return old;
}

static inline float atomicAdd(float *address, float val)
{
  auto ar = makeAtomicRef(address);
  auto old = ar.fetch_add(val);
  return old;
}

static inline double atomicAdd(double *address, double val)
{
  auto ar = makeAtomicRef(address);
  auto old = ar.fetch_add(val);
  return old;
}

static inline uint32_t atomicMax(uint32_t *address, uint32_t val)
{
  auto ar = makeAtomicRef(address);
  auto old = ar.fetch_max(val);
  return old;
}

#include <../cuda/atomic_helper.h>
