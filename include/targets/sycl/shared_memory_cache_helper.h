#pragma once

#if 1

#include <../generic/shared_memory_cache_helper.h>

namespace quda {
  template <typename T, typename D, typename O> static constexpr bool needsFullBlock<SharedMemoryCache<T,D,O>> = true;
}

#else

#include <target_device.h>
#include <tunable_kernel.h>

#define DYNAMIC_SLM

/**
   @file shared_memory_cache_helper.h

   Helper functionality for aiding the use of the shared memory for
   sharing data between threads in a thread block.
 */

namespace quda
{
  struct offNone {};

  /**
     @brief Class which wraps around a shared memory cache for type T,
     where each thread in the thread block stores a unique value in
     the cache which any other thread can access.

     This accessor supports both explicit run-time block size and
     compile-time sizing.

     * For run-time block size, the constructor should be initialized
       with the desired block size.

     * For compile-time block size, no arguments should be passed to
       the constructor, and then the second and third template
       parameters correspond to the y and z dimensions of the block,
       respectively.  The x dimension of the block will be set
       according the maximum number of threads possible, given these
       dimensions.
   */
  //template <typename T, int block_size_y = 1, int block_size_z = 1, bool dynamic = true>
  template <typename T, typename D = opDimsBlock, typename O = offNone>
  class SharedMemoryCache
  {
    using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;
    static_assert(sizeof(T) % 4 == 0, "Shared memory cache does not support sub-word size types");

    // The number of elements of type atom_t that we break T into for optimal shared-memory access
    static constexpr int n_element = sizeof(T) / sizeof(atom_t);

    // used to avoid instantiation of load functions if unused, in case T is not a valid return type (e.g. C array)
    template <typename dummy = void> using maybeT = std::conditional_t<std::is_same_v<dummy,void>,T,void>;

    const dim3 block;
    const int stride;
    sycl::local_ptr<atom_t> cache_ptr;

#ifdef DYNAMIC_SLM
    using opSmem = op_SharedMemory<T, opSizeDims<D>>;
    using deps = op_Sequential<op_blockSync,opSmem>;
    using depOps = SpecialOps<op_blockSync,opSmem>;
#else
#endif
    using SharedMemoryCache_t = SharedMemoryCache<T, D, O>;
    //dependentOps ops;

    __device__ __host__ inline void save_detail(const T &a, int x, int y, int z)
    {
      atom_t tmp[n_element];
      memcpy(tmp, (void *)&a, sizeof(T));
      int j = (z * block.y + y) * block.x + x;
#pragma unroll
      for (int i = 0; i < n_element; i++) cache_ptr[i * stride + j] = tmp[i];
    }

    template <typename dummy = void>
    __device__ __host__ inline maybeT<dummy> load_detail(int x, int y, int z)
    {
      atom_t tmp[n_element];
      int j = (z * block.y + y) * block.x + x;
#pragma unroll
      for (int i = 0; i < n_element; i++) tmp[i] = cache_ptr[i * stride + j];
      T a;
      memcpy((void *)&a, tmp, sizeof(T));
      return a;
    }

  public:
    using dependencies = deps;
    using dependentOps = depOps;
    /**
       @brief constructor for SharedMemory cache.  If no arguments are
       pass, then the dimensions are set according to the templates
       block_size_y and block_size_z, together with the derived
       block_size_x.  Otherwise use the block sizes passed into the
       constructor.

       @param[in] block Block dimensions for the 3-d shared memory object
    */
#if 0
    template <typename dummy = void>
    constexpr SharedMemoryCache(dim3 block = dim3(block_size_x, block_size_y, block_size_z)) :
      block(block), stride(block.x * block.y * block.z)
    {
      static_assert(dynamic==false, "SYCL target requires SpecialOps parameter for dynamic shared memory");
      using atype = atom_t[n_element * block_size_x * block_size_y * block_size_z];
      if constexpr (std::is_same_v<dummy,void>) {
	auto mem = sycl::ext::oneapi::group_local_memory_for_overwrite<atype>(getGroup());
	//cache_ptr = *mem.get();
	cache_ptr = &((*mem)[0]);
      }
    }
    template <typename S, typename ...Arg>
    inline SharedMemoryCacheImpl(const only_SharedMemoryCache<T,S> &op, const Arg &...arg) :
      block(S::dims(target::block_dim(), arg...)), stride(block.x * block.y * block.z)
    {
      sycl::local_ptr<void> v(op.smem);
      sycl::local_ptr<atom_t> p(v);
      cache_ptr = p;
    }
#endif
    template <typename ...U, typename ...Arg>
    inline SharedMemoryCache(const SpecialOps<U...> &ops, const Arg &...arg) :
      block(D::dims(target::block_dim(), arg...)),
      stride(block.x * block.y * block.z)
    {
      auto op = getDependentOps<SharedMemoryCache_t>(ops);
      sycl::local_ptr<void> v(op.smem);
      sycl::local_ptr<atom_t> p(v);
      cache_ptr = p;
    }

    /**
       @brief Grab the raw base address to shared memory.
    */
    inline auto data() {
      sycl::local_ptr<void> v(cache_ptr);
      sycl::local_ptr<T> p(v);
      return p.get();
    }

    /**
       @brief Save the value into the 3-d shared memory cache.
       @param[in] a The value to store in the shared memory cache
       @param[in] x The x index to use
       @param[in] y The y index to use
       @param[in] z The z index to use
     */
    __device__ __host__ inline void save(const T &a, int x = -1, int y = -1, int z = -1)
    {
      auto tid = target::thread_idx();
      x = (x == -1) ? tid.x : x;
      y = (y == -1) ? tid.y : y;
      z = (z == -1) ? tid.z : z;
      save_detail(a, x, y, z);
    }

    /**
       @brief Save the value into the 3-d shared memory cache.
       @param[in] a The value to store in the shared memory cache
       @param[in] x The x index to use
     */
    __device__ __host__ inline void save_x(const T &a, int x = -1)
    {
      auto tid = target::thread_idx();
      x = (x == -1) ? tid.x : x;
      save_detail(a, x, tid.y, tid.z);
    }

    /**
       @brief Save the value into the 3-d shared memory cache.
       @param[in] a The value to store in the shared memory cache
       @param[in] y The y index to use
     */
    __device__ __host__ inline void save_y(const T &a, int y = -1)
    {
      auto tid = target::thread_idx();
      y = (y == -1) ? tid.y : y;
      save_detail(a, tid.x, y, tid.z);
    }

    /**
       @brief Save the value into the 3-d shared memory cache.
       @param[in] a The value to store in the shared memory cache
       @param[in] z The z index to use
     */
    __device__ __host__ inline void save_z(const T &a, int z = -1)
    {
      auto tid = target::thread_idx();
      z = (z == -1) ? tid.z : z;
      save_detail(a, tid.x, tid.y, z);
    }

    /**
       @brief Load a value from the shared memory cache
       @param[in] x The x index to use
       @param[in] y The y index to use
       @param[in] z The z index to use
       @return The value at coordinates (x,y,z)
     */
    template <typename dummy = void>
    __device__ __host__ inline maybeT<dummy> load(int x = -1, int y = -1, int z = -1)
    {
      auto tid = target::thread_idx();
      x = (x == -1) ? tid.x : x;
      y = (y == -1) ? tid.y : y;
      z = (z == -1) ? tid.z : z;
      return load_detail(x, y, z);
    }

    /**
       @brief Load a vector from the shared memory cache
       @param[in] x The x index to use
       @return The value at coordinates (x,y,z)
    */
    template <typename dummy = void>
    __device__ __host__ inline maybeT<dummy> load_x(int x = -1)
    {
      auto tid = target::thread_idx();
      x = (x == -1) ? tid.x : x;
      return load_detail(x, tid.y, tid.z);
    }

    /**
       @brief Load a vector from the shared memory cache
       @param[in] y The y index to use
       @return The value at coordinates (x,y,z)
    */
    template <typename dummy = void>
    __device__ __host__ inline maybeT<dummy> load_y(int y = -1)
    {
      auto tid = target::thread_idx();
      y = (y == -1) ? tid.y : y;
      return load_detail(tid.x, y, tid.z);
    }

    /**
       @brief Load a vector from the shared memory cache
       @param[in] z The z index to use
       @return The value at coordinates (x,y,z)
    */
    template <typename dummy = void>
    __device__ __host__ inline maybeT<dummy> load_z(int z = -1)
    {
      auto tid = target::thread_idx();
      z = (z == -1) ? tid.z : z;
      return load_detail(tid.x, tid.y, z);
    }

    /**
       @brief Synchronize the cache
    */
    __device__ __host__ inline void sync() { __syncthreads(); }
  };

  template <typename T, typename O = offNone>
  using SharedMemoryCacheOffset = SharedMemoryCache<T,opDimsBlock,O>;

#if 0
  template <typename T, int block_size_y = 1, int block_size_z = 1, bool dynamic = true>
  class SharedMemoryCache : public SharedMemoryCacheImpl<T, block_size_y, block_size_z, dynamic> {
    using SharedMemoryCacheImpl<T, block_size_y, block_size_z, dynamic>::SharedMemoryCacheImpl;
  };

  template <typename O>
  class SharedMemoryCache<O, (int)(isOpSharedMemoryCache<O> ? 1 : 0), 1, true> : public SharedMemoryCacheImpl<typename O::ElemT> {
  public:
    template <typename ...U> inline SharedMemoryCache(const SpecialOps<U...> *ops) :
      SharedMemoryCacheImpl<typename O::ElemT>(getSpecialOp<SpecialOps<O>>(ops)) {}
    template <typename ...U, typename Arg> inline SharedMemoryCache(const SpecialOps<U...> *ops, const Arg &arg) :
      SharedMemoryCacheImpl<typename O::ElemT>(getSpecialOp<SpecialOps<O>>(ops), arg) {}
  };
#endif

} // namespace quda

#endif
