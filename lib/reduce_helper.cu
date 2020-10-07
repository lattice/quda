#include <quda_internal.h>
#include <malloc_quda.h>
#include <reduce_helper.h>
#include <tune_quda.h>

// These are used for reduction kernels
static device_reduce_t *d_reduce = nullptr;
static device_reduce_t *h_reduce = nullptr;
static device_reduce_t *hd_reduce = nullptr;

static count_t *reduce_count = nullptr;
static cudaEvent_t reduceEnd;

namespace quda
{

  namespace reducer
  {

    // FIXME need to dynamically resize these
    void *get_device_buffer() { return d_reduce; }
    void *get_mapped_buffer() { return hd_reduce; }
    void *get_host_buffer() { return h_reduce; }
    count_t *get_count() { return reduce_count; }
    cudaEvent_t &get_event() { return reduceEnd; }

    size_t buffer_size()
    {
      /* we have these different reductions to cater for:

         - regular reductions (reduce_quda.cu) where are reducing to a
           single vector type (max length 4 presently), and a
           grid-stride loop with max number of blocks = 2 x SM count

         - multi-reductions where we are reducing to a matrix of size
           of size QUDA_MAX_MULTI_REDUCE of vectors (max length 4),
           and a grid-stride loop with maximum number of blocks = 2 x
           SM count
      */

      int reduce_size = 4 * sizeof(device_reduce_t);
      int max_reduce = reduce_size;
      int max_multi_reduce = max_n_reduce() * reduce_size;
      int max_reduce_blocks = 2 * deviceProp.multiProcessorCount;

      // reduction buffer size
      size_t bytes = max_reduce_blocks * std::max(max_reduce, max_multi_reduce);
      return bytes;
    }

    // need to use placement new constructor to initialize the atomic counters
    template <typename T> __global__ void init_count(T *counter)
    {
      for (int i = 0; i < max_n_reduce(); i++) new (counter + i) T {0};
    }

    void init()
    {
      auto bytes = buffer_size();
      if (!d_reduce) d_reduce = (device_reduce_t *)device_malloc(bytes);

      // these arrays are actually oversized currently (only needs to be device_reduce_t x 3)

      // if the device supports host-mapped memory then use a host-mapped array for the reduction
      if (!h_reduce) {
        h_reduce = (device_reduce_t *)mapped_malloc(bytes);
        hd_reduce = (device_reduce_t *)get_mapped_device_pointer(h_reduce); // set the matching device pointer

#ifdef HETEROGENEOUS_ATOMIC
        using system_atomic_t = device_reduce_t;
        size_t n_reduce = bytes / sizeof(system_atomic_t);
        auto *atomic_buf = reinterpret_cast<system_atomic_t *>(h_reduce);               // FIXME
        for (size_t i = 0; i < n_reduce; i++) new (atomic_buf + i) system_atomic_t {0}; // placement new constructor
#else
        memset(h_reduce, 0, bytes); // added to ensure that valgrind doesn't report h_reduce is unitialised
#endif
      }

      if (!reduce_count) {
        reduce_count = static_cast<count_t *>(device_malloc(max_n_reduce() * sizeof(decltype(*reduce_count))));
        TuneParam tp;
        tp.grid = dim3(1, 1, 1);
        tp.block = dim3(1, 1, 1);

        qudaLaunchKernel(init_count<count_t>, tp, 0, reduce_count);
      }

      cudaEventCreateWithFlags(&reduceEnd, cudaEventDisableTiming);

      checkCudaError();
    }

    void destroy()
    {
      cudaEventDestroy(reduceEnd);

      if (reduce_count) {
        device_free(reduce_count);
        reduce_count = nullptr;
      }
      if (d_reduce) {
        device_free(d_reduce);
        d_reduce = 0;
      }
      if (h_reduce) {
        host_free(h_reduce);
        h_reduce = 0;
      }
      hd_reduce = 0;
    }

  } // namespace reducer
} // namespace quda
