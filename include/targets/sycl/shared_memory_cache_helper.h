#pragma once

#include <target_device.h>
#include <tunable_kernel.h>

/**
   @file shared_memory_cache_helper.cuh

   Helper functionality for aiding the use of the shared memory for
   sharing data between threads in a thread block.
 */

namespace quda
{

  /**
     @brief Class which wraps around a shared memory cache for type T,
     where each thread in the thread block stores a unique value in
     the cache which any other thread can access.

     This accessor supports both explicit run-time block size and
     compile-time sizing.

     * For run-time block size, the constructor should be initialied
       with the desired block size.

     * For compile-time block size, no arguments should be passed to
       the constructor, and then the second and third template
       parameters correspond to the y and z dimensions of the block,
       respectively.  The x dimension of the block will be set
       according the maximum number of threads possible, given these
       dimensions.  To prevent shared-memory bank conflicts this width
       is optionally padded to allow for access along the y and z dimensions.
   */
  template <typename T>
  class SharedMemoryCacheD : only_SharedMemoryCache<T>
  {
    static_assert(sizeof(T) % 4 == 0, "Shared memory cache does not support sub-word size types");
    const dim3 block;
    const int stride;
    using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;
    static constexpr int n_element = sizeof(T) / sizeof(atom_t);

    /**
       @brief This is a dummy instantiation for the host compiler
    */
    template <bool, typename dummy = void> struct cache_dynamic {
      atom_t *operator()(SharedMemoryCacheD<T> *)
      {
        static T cache_;
        return reinterpret_cast<atom_t *>(&cache_);
      }
    };

    /**
       @brief This is the handle to the shared memory, dynamic specialization
       @return Shared memory pointer
     */
    template <typename dummy> struct cache_dynamic<true, dummy> {
      //__device__ inline atom_t *operator()(SharedMemoryCacheD<T> *t)
      __device__ inline sycl::local_ptr<atom_t> operator()(SharedMemoryCacheD<T> *t)
      {
        //return reinterpret_cast<atom_t *>(getSharedMemPtr(t).get());
	sycl::local_ptr<void> v(t->smem);
	sycl::local_ptr<atom_t> p(v);
	return p;
        //return reinterpret_cast<atom_t *>(0);
      }
    };

    //__device__ __host__ inline atom_t *cache()
    __device__ __host__ inline sycl::local_ptr<atom_t> cache()
    {
      return target::dispatch<cache_dynamic>(this);
    }

    __device__ __host__ inline void save_detail(const T &a, int x, int y, int z)
    {
      atom_t tmp[n_element];
      memcpy(tmp, (void*)&a, sizeof(T));
      int j = (z * block.y + y) * block.x + x;
#pragma unroll
      for (int i = 0; i < n_element; i++) cache()[i * stride + j] = tmp[i];
    }

    __device__ __host__ inline T load_detail(int x, int y, int z)
    {
      atom_t tmp[n_element];
      int j = (z * block.y + y) * block.x + x;
#pragma unroll
      for (int i = 0; i < n_element; i++) tmp[i] = cache()[i * stride + j];
      T a;
      memcpy((void*)&a, tmp, sizeof(T));
      return a;
    }

  public:
    /**
       @brief constructor for SharedMemory cache.  If no arguments are
       pass, then the dimensions are set according to the templates
       block_size_y and block_size_z, together with the derived
       block_size_x.  Otherwise use the block sizes passed into the
       constructor.

       @param[in] block Block dimensions for the 3-d shared memory object
    */
    template <typename O>
    SharedMemoryCacheD(O *ops, dim3 block = target::block_dim()) :
      only_SharedMemoryCache<T>(getSpecialOp<only_SharedMemoryCache<T>>(ops)),
      block(block),
      stride(block.x * block.y * block.z)
    {
      //cache_ = reinterpret_cast<atom_t*>(this->smem.get());
      //cache_ = (atom_t*)0;
    }

    /**
       @brief Grab the raw base address to shared memory.
    */
    //inline T* data() { return reinterpret_cast<T*>(cache()); }
    //inline T* data() { return reinterpret_cast<T*>(cache().get()); }
    //inline T* data0() { return reinterpret_cast<T*>(0); }
    //inline T* dataX() {
    auto data() {
    //inline sycl::local_ptr<T> data() {
      sycl::local_ptr<void> v(this->smem);
      sycl::local_ptr<T> p(v);
      //T *g = p.get();
      //sycl::generic_ptr<T> g(p);
      return p.get();
      //return 0;
      //return reinterpret_cast<T*>(cache_);
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
    __device__ __host__ inline T load(int x = -1, int y = -1, int z = -1)
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
    __device__ __host__ inline T load_x(int x = -1)
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
    __device__ __host__ inline T load_y(int y = -1)
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
    __device__ __host__ inline T load_z(int z = -1)
    {
      auto tid = target::thread_idx();
      z = (z == -1) ? tid.z : z;
      return load_detail(tid.x, tid.y, z);
    }

    /**
       @brief Synchronize the cache
    */
    //__device__ __host__ void sync() { this->blockSync(); }
    __device__ __host__ void sync() { __syncthreads(); }
  };

  template <typename T, int block_size_y = 1, int block_size_z = 1, bool pad = false, bool dynamic = true>
  class SharedMemoryCache
  {
    /** maximum number of threads in x given the y and z block sizes */
    static constexpr int max_block_size_x = device::max_block_size<block_size_y, block_size_z>();

    /** pad in the x dimension width if requested to ensure that it isn't a multiple of the bank width */
    static constexpr int block_size_x = !pad ? max_block_size_x :
      ((max_block_size_x + device::shared_memory_bank_width() - 1) /
       device::shared_memory_bank_width()) * device::shared_memory_bank_width();

    //using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;
    using atom_t = T;
    static_assert(sizeof(T) % 4 == 0, "Shared memory cache does not support sub-word size types");

    // The number of elements of type atom_t that we break T into for optimal shared-memory access
    static constexpr int n_element = sizeof(T) / sizeof(atom_t);
    static constexpr int block_size = block_size_x * block_size_y * block_size_z;
    static constexpr int max_array_len = device::shared_memory_size() / sizeof(T);
    static constexpr int array_len = std::min(block_size, max_array_len);
    using atype = atom_t[n_element * array_len];

    const dim3 block;
    const int stride;
    sycl::multi_ptr<atype, sycl::access::address_space::local_space> mem;
    atom_t *cache_;

    __device__ __host__ inline void save_detail(const T &a, int x, int y, int z)
    {
      atom_t tmp[n_element];
      memcpy(tmp, (void*)&a, sizeof(T));
      int j = (z * block.y + y) * block.x + x;
#pragma unroll
      for (int i = 0; i < n_element; i++) cache_[i * stride + j] = tmp[i];
    }

    __device__ __host__ inline T load_detail(int x, int y, int z)
    {
      atom_t tmp[n_element];
      int j = (z * block.y + y) * block.x + x;
#pragma unroll
      for (int i = 0; i < n_element; i++) tmp[i] = cache_[i * stride + j];
      T a;
      memcpy((void*)&a, tmp, sizeof(T));
      return a;
    }

    /**
       @brief Dummy instantiation for the host compiler
    */
    template <bool is_device, typename dummy = void> struct sync_impl {
      void operator()() { }
    };

    /**
       @brief Synchronize the cache when on the device
    */
    template <typename dummy> struct sync_impl<true, dummy> {
      inline void operator()() { __syncthreads(); }
      //inline void operator()() { this->blockSync(); }
    };

  public:
    /**
       @brief constructor for SharedMemory cache.  If no arguments are
       pass, then the dimensions are set according to the templates
       block_size_y and block_size_z, together with the derived
       block_size_x.  Otherwise use the block sizes passed into the
       constructor.

       @param[in] block Block dimensions for the 3-d shared memory object
    */
    SharedMemoryCache(dim3 block = dim3(block_size_x, block_size_y, block_size_z)) :
      block(block),
      stride(block.x * block.y * block.z)
    {
      auto g = getGroup();
      auto len = g.get_local_linear_range();
      if(len<=array_len) {
	//auto cache = sycl::group_local_memory_for_overwrite<atype>(g);
	//return reinterpret_cast<atom_t*>(cache_.get());
	mem = sycl::ext::oneapi::group_local_memory<atype>(g);
	cache_ = *mem.get();
	//cache_ = smem->getMem();
      } else {
	cache_ = nullptr;
      }
    }
    template <typename O>
    SharedMemoryCache(O *ops, dim3 block = target::block_dim()) :
      block(block),
      stride(block.x * block.y * block.z)
    {
      auto op = getSpecialOp<only_SharedMemoryCache<T>>(ops);
      sycl::multi_ptr<void, sycl::access::address_space::local_space> v(op.smem);
      decltype(mem) p(v);
      //mem = reinterpret_cast<decltype(mem)>(v);
      mem = p;
      cache_ = *mem.get();
    }

    /**
       @brief Grab the raw base address to shared memory.
    */
    inline T* data() { return reinterpret_cast<T*>(cache_); }

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
    __device__ __host__ inline T load(int x = -1, int y = -1, int z = -1)
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
    __device__ __host__ inline T load_x(int x = -1)
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
    __device__ __host__ inline T load_y(int y = -1)
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
    __device__ __host__ inline T load_z(int z = -1)
    {
      auto tid = target::thread_idx();
      z = (z == -1) ? tid.z : z;
      return load_detail(tid.x, tid.y, z);
    }

    /**
       @brief Synchronize the cache
    */
    __device__ __host__ void sync() { target::dispatch<sync_impl>(); }
    //__device__ __host__ void sync() { this->blockSync(); }
  };

} // namespace quda
