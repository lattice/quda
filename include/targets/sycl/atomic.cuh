#pragma once

/**
   @file atomic.cuh

   @section Description

   Provides definitions of atomic functions that are not native to
   CUDA.  These are intentionally not declared in the namespace to
   avoid confusion when resolving the native atomicAdd functions.
 */

//inline constexpr auto mo = sycl::memory_order::relaxed;
inline constexpr auto mo = sycl::ext::oneapi::memory_order::acq_rel;
//inline constexpr auto mo = memory_order::seq_cst;
//inline constexpr auto mo = sycl::memory_order::acq_rel;

inline constexpr auto ms = sycl::ext::oneapi::memory_scope::system;
//inline constexpr auto ms = sycl::memory_scope::system;

//inline constexpr auto as = sycl::access::address_space::generic_space;
inline constexpr auto as = sycl::access::address_space::global_space;

template <typename T>
using atomicRef = sycl::ext::oneapi::atomic_ref<T,mo,ms,as>;
//using atomicRef = sycl::atomic_ref<T,mo,ms,as>;

template <typename T>
static inline atomicRef<T> makeAtomicRef(T *address) {
  return atomicRef<T>(*address);
}

static inline uint __float_as_uint(float x) {
  return *reinterpret_cast<uint*>(&x);
}

static inline float __uint_as_float(uint x) {
  return *reinterpret_cast<float*>(&x);
}

static inline unsigned int atomicMax(unsigned int* address, unsigned int val)
{
  auto ar = makeAtomicRef(address);
  auto old = ar.fetch_max(val);
  return old;
}

static inline int atomicCAS(int* address, int compare, int val)
{
  auto ar = makeAtomicRef(address);
  auto old = ar.compare_exchange_strong(compare, val);
  return old;
}
static inline unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val)
{
  auto ar = makeAtomicRef(address);
  auto old = ar.compare_exchange_strong(compare, val);
  return old;
}

/**
   @brief Implementation of double-precision atomic addition using compare
   and swap. Taken from the CUDA programming guide.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline int atomicAdd(int* address, int val)
{
  auto ar = makeAtomicRef(address);
  auto old = ar.fetch_add(val);
  return old;
}
static inline float atomicAdd(float* address, float val)
{
  auto ar = makeAtomicRef(address);
  auto old = ar.fetch_add(val);
  return old;
}
static inline double atomicAdd(double* address, double val)
{
  auto ar = makeAtomicRef(address);
  auto old = ar.fetch_add(val);
  return old;
}

/**
   @brief Implementation of double2 atomic addition using two
   double-precision additions.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline double2 atomicAdd(double2 *addr, double2 val)
{
  double2 old = *addr;
  // This is a necessary evil to avoid conflicts between the atomicAdd
  // declared in the CUDA headers which are visible for host
  // compilation, which cause a conflict when compiled on clang-cuda.
  // As a result we do not support any architecture without native
  // double precision atomics on clang-cuda.
  old.x = atomicAdd((double*)addr, val.x);
  old.y = atomicAdd((double*)addr + 1, val.y);
  return old;
}

/**
   @brief Implementation of float2 atomic addition using two
   single-precision additions.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline float2 atomicAdd(float2 *addr, float2 val){
  float2 old = *addr;
  old.x = atomicAdd((float*)addr, val.x);
  old.y = atomicAdd((float*)addr + 1, val.y);
  return old;
}

/**
   @brief Implementation of int2 atomic addition using two
   int additions.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline int2 atomicAdd(int2 *addr, int2 val){
  int2 old = *addr;
  old.x = atomicAdd((int*)addr, val.x);
  old.y = atomicAdd((int*)addr + 1, val.y);
  return old;
}

union uint32_short2 { unsigned int i; short2 s; };

/**
   @brief Implementation of short2 atomic addition using compare
   and swap.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline short2 atomicAdd(short2 *addr, short2 val){
  uint32_short2 old, assumed, incremented;
  old.s = *addr;
  do {
    assumed.s = old.s;
    incremented.s = make_short2(val.x + assumed.s.x, val.y + assumed.s.y);
    old.i =  atomicCAS((unsigned int*)addr, assumed.i, incremented.i);
  } while ( assumed.i != old.i );

  return old.s;
}

union uint32_char2 { unsigned short i; char2 s; };

/**
   @brief Implementation of char2 atomic addition using compare
   and swap.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline char2 atomicAdd(char2 *addr, char2 val){
  uint32_char2 old, assumed, incremented;
  old.s = *addr;
  do {
    assumed.s = old.s;
    incremented.s = make_char2(val.x + assumed.s.x, val.y + assumed.s.y);
    old.i =  atomicCAS((unsigned int*)addr, assumed.i, incremented.i);
  } while ( assumed.i != old.i );

  return old.s;
}

/**
   @brief Implementation of single-precision atomic max using compare
   and swap. May not support NaNs properly...

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline float atomicMax(float *addr, float val){
  unsigned int old = __float_as_uint(*addr), assumed;
  do {
    assumed = old;
    if (__uint_as_float(old) >= val) break;

    old = atomicCAS((unsigned int*)addr,
           assumed,
           __float_as_uint(val));
  } while ( assumed != old );

  return __uint_as_float(old);
}

/**
   @brief Implementation of single-precision atomic max specialized
   for positive-definite numbers.  Here we take advantage of the
   property that when positive floating point numbers are
   reinterpretted as unsigned integers, they have the same unique
   sorted order.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline float atomicAbsMax(float *addr, float val){
  uint32_t val_ = __float_as_uint(val);
  uint32_t *addr_ = reinterpret_cast<uint32_t*>(addr);
  return atomicMax(addr_, val_);
}

/**
   @brief atomic_fetch_add function performs similarly as atomic_ref::fetch_add
   @param[in,out] addr The memory address of the variable we are
   updating atomically
   @param[in] val The value we summing to the value at addr
 */
template <typename T> inline void atomic_fetch_add(T *addr, T val)
{
  atomicAdd(addr, val);
}

template <typename T, int n> void atomic_fetch_add(vector_type<T, n> *addr, vector_type<T, n> val)
{
  for (int i = 0; i < n; i++) atomic_fetch_add(&(*addr)[i], val[i]);
}

/**
   @brief atomic_fetch_max function that does an atomic max.
   @param[in,out] addr The memory address of the variable we are
   updating atomically
   @param[in] val The value we are comparing against.  Must be
   positive valued else result is undefined.
 */
template <typename T> inline void atomic_fetch_abs_max(T *addr, T val)
{
  atomicAbsMax(addr, val);
}
