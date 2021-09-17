#pragma once

/**
   @file atomic.cuh

   @section Description
 
   Provides definitions of atomic functions that are not native to
   CUDA.  These are intentionally not declared in the namespace to
   avoid confusion when resolving the native atomicAdd functions.
 */

#ifdef QUDA_BACKEND_OMPTARGET
static inline unsigned int
__float_as_uint(float x)
{
  return reinterpret_cast<unsigned int &>(x);  // FIXME UB?
}
static inline float
__uint_as_float(unsigned int x)
{
  return reinterpret_cast<float &>(x);  // FIXME UB?
}

static inline int atomicAdd(int *x, int v)
{
  int z;
#pragma omp atomic capture
  {z=*x; *x+=v;}
  return z;
}
static inline unsigned int atomicAdd(unsigned int *x, unsigned int v)
{
  unsigned int z;
#pragma omp atomic capture
  {z=*x; *x+=v;}
  return z;
}
static inline float atomicAdd(float *x, float v)
{
  float z;
#pragma omp atomic capture
  {z=*x; *x+=v;}
  return z;
}
static inline double atomicAdd(double *x, double v)
{
  double z;
#pragma omp atomic capture
  {z=*x; *x+=v;}
  return z;
}
static inline unsigned int atomicCAS(unsigned int*x, unsigned int c, unsigned int v)
{
  unsigned int z;
#pragma omp atomic capture
  {z=*x; *x=z==c?v:z;}
  return z;
}
static inline uint32_t atomicMax(uint32_t *x, uint32_t v)
{
  uint32_t z;
#pragma omp atomic capture
  {z=*x; *x=z<v?v:z;}
  return z;
}
#endif

/**
   @brief Implementation of double2 atomic addition using two
   double-precision additions.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline __device__ double2 atomicAdd(double2 *addr, double2 val)
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
static inline __device__ float2 atomicAdd(float2 *addr, float2 val){
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
static inline __device__ int2 atomicAdd(int2 *addr, int2 val){
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
static inline __device__ short2 atomicAdd(short2 *addr, short2 val){
  uint32_short2 old, assumed, incremented;
  old.s = *addr;
  do {
    assumed.s = old.s;
    incremented.s = make_short2(val.x + assumed.s.x, val.y + assumed.s.y);
    old.i =  atomicCAS((unsigned int*)addr, assumed.i, incremented.i);
  } while ( assumed.i != old.i );

  return old.s;
}

union uint32_char2 { unsigned int i; char2 s; };

/**
   @brief Implementation of char2 atomic addition using compare
   and swap.

   @param addr Address that stores the atomic variable to be updated
   @param val Value to be added to the atomic
*/
static inline __device__ char2 atomicAdd(char2 *addr, char2 val){
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
static inline __device__ float atomicMax(float *addr, float val){
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
static inline __device__ float atomicAbsMax(float *addr, float val){
  uint32_t val_ = __float_as_uint(val);
  uint32_t *addr_ = reinterpret_cast<uint32_t*>(addr);
  return atomicMax(addr_, val_);
}

template <bool is_device> struct atomic_fetch_add_impl {
  template <typename T> inline void operator()(T *addr, T val)
  {
#pragma omp atomic update
    *addr += val;
  }
};

template <> struct atomic_fetch_add_impl<true> {
  template <typename T> __device__ inline void operator()(T *addr, T val) { atomicAdd(addr, val); }
};

/**
   @brief atomic_fetch_add function performs similarly as atomic_ref::fetch_add
   @param[in,out] addr The memory address of the variable we are
   updating atomically
   @param[in] val The value we summing to the value at addr
 */
template <typename T> __device__ __host__ inline void atomic_fetch_add(T *addr, T val)
{
  target::dispatch<atomic_fetch_add_impl>(addr, val);
}

template <typename T, int n> __device__ __host__ void atomic_fetch_add(vector_type<T, n> *addr, vector_type<T, n> val)
{
  for (int i = 0; i < n; i++) atomic_fetch_add(&(*addr)[i], val[i]);
}

template <bool is_device> struct atomic_fetch_abs_max_impl {
  template <typename T> inline void operator()(T *addr, T val)
  {
    T z;
#pragma omp atomic capture
    {z=*addr; *addr=z<val?val:z;}
  }
};

template <> struct atomic_fetch_abs_max_impl<true> {
  template <typename T> __device__ inline void operator()(T *addr, T val) { atomicAbsMax(addr, val); }
};

/**
   @brief atomic_fetch_max function that does an atomic max.
   @param[in,out] addr The memory address of the variable we are
   updating atomically
   @param[in] val The value we are comparing against.  Must be
   positive valued else result is undefined.
 */
template <typename T> __device__ __host__ inline void atomic_fetch_abs_max(T *addr, T val)
{
  target::dispatch<atomic_fetch_abs_max_impl>(addr, val);
}
