#pragma once

#include <array.h>

/**
   @file atomic_helper.h

   @section Provides definitions of atomic functions that are used in QUDA.
 */

inline constexpr auto mo = sycl::memory_order::relaxed;
//inline constexpr auto mo = sycl::ext::oneapi::memory_order::acq_rel;
//inline constexpr auto mo = memory_order::seq_cst;
//inline constexpr auto mo = sycl::memory_order::acq_rel;

//inline constexpr auto ms = sycl::memory_scope::system;
inline constexpr auto ms = sycl::memory_scope::device;
inline constexpr auto msg = sycl::memory_scope::work_group;

//inline constexpr auto as = sycl::access::address_space::generic_space;
inline constexpr auto as = sycl::access::address_space::global_space;
inline constexpr auto asl = sycl::access::address_space::local_space;

//using atomicRef = sycl::ext::oneapi::atomic_ref<T,mo,ms,as>;
template <typename T>
using atomicRef = sycl::atomic_ref<T,mo,ms,as>;
template <typename T>
using atomicRefL = sycl::atomic_ref<T,mo,msg,asl>;

template <typename T>
static inline atomicRef<T> makeAtomicRef(T *address) {
  return atomicRef<T>(*address);
}

template <typename T>
static inline atomicRefL<T> makeAtomicRefL(T *address) {
  return atomicRefL<T>(*address);
}

#if 0
using lfloat = std::remove_pointer_t<decltype(std::declval<sycl::local_ptr<float>>().get())>;
using ldouble = std::remove_pointer_t<decltype(std::declval<sycl::local_ptr<double>>().get())>;

static inline atomicRefL<float> makeAtomicRef(lfloat *address) {
  return atomicRefL<float>(*address);
}

static inline atomicRefL<double> makeAtomicRef(ldouble *address) {
  return atomicRefL<double>(*address);
}

static inline atomicRefL<float> makeAtomicRefL(lfloat *address) {
  return atomicRefL<float>(*address);
}

static inline atomicRefL<double> makeAtomicRefL(ldouble *address) {
  return atomicRefL<double>(*address);
}
#endif

static inline uint __float_as_uint(float x) {
  return *reinterpret_cast<uint*>(&x);
}

#if 0
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
#endif
template <typename T, typename U>
static inline int atomicAdd(T *address, U val)
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

template <typename T, typename U>
__device__ __host__ inline void atomic_fetch_add(T *addr, U val)
{
  atomicAdd(addr, val);
}

template <typename T, typename U>
__device__ __host__ inline void atomic_add_local(T *addr, U val)
{
  auto ar = makeAtomicRefL(addr);
  ar += val;
}

#include <../cuda/atomic_helper.h>
