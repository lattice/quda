#include <quda_internal.h>
#include <malloc_quda.h>
#include <reduce_helper.h>
#include <tunable_nd.h>
#include <kernels/reduce_init.cuh>

// These are used for reduction kernels
static device_reduce_t *d_reduce = nullptr;
static device_reduce_t *h_reduce = nullptr;
static device_reduce_t *hd_reduce = nullptr;

static count_t *reduce_count = nullptr;
static qudaEvent_t reduceEnd;

namespace quda
{

  namespace reducer
  {

    void *get_device_buffer() { return d_reduce; }
    void *get_mapped_buffer() { return hd_reduce; }
    void *get_host_buffer() { return h_reduce; }
    template <> count_t *get_count() { return reduce_count; }
    qudaEvent_t &get_event() { return reduceEnd; }

    static size_t allocated_bytes = 0;
    static int allocated_n_reduce = 0;
    static bool init_event = false;

    template <typename T>
    struct init_reduce : public TunableKernel1D {
      T *reduce_count;
      const int n_reduce;
      long long bytes() const { return n_reduce * sizeof(T); }
      unsigned int minThreads() const { return n_reduce; }

      init_reduce(T *reduce_count, int n_reduce) :
        TunableKernel1D(n_reduce, QUDA_CUDA_FIELD_LOCATION),
        reduce_count(reduce_count),
        n_reduce(n_reduce)
      { apply(device::get_default_stream()); }

      void apply(const qudaStream_t &stream)
      {
        // intentionally do not autotune, since this can be called inside a tuning region
        auto tp = tuneLaunch(*this, QUDA_TUNE_NO, getVerbosity());
        launch_device<init_count>(tp, stream, init_arg<T>(reduce_count, n_reduce));
      }
    };

    void init(int n_reduce, size_t reduce_size)
    {
      auto max_reduce_blocks = 2 * device::processor_count();
      auto bytes = max_reduce_blocks * n_reduce * reduce_size;

      if (allocated_bytes < bytes) {
        if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
          printfQuda("reducer::init buffer resizing for n_reduce = %d, reduce_size = %lu, bytes = %lu\n",
                     n_reduce, reduce_size, bytes);
        if (d_reduce) device_free(d_reduce);
        d_reduce = static_cast<device_reduce_t *>(device_malloc(bytes));

        if (h_reduce) host_free(h_reduce);
        h_reduce = static_cast<device_reduce_t *>(mapped_malloc(bytes));
        hd_reduce = static_cast<device_reduce_t *>(get_mapped_device_pointer(h_reduce)); // set the matching device pointer

        using system_atomic_t = device_reduce_t;
        size_t n_reduce = bytes / sizeof(system_atomic_t);
        auto *atomic_buf = reinterpret_cast<system_atomic_t *>(h_reduce);
        for (size_t i = 0; i < n_reduce; i++) new (atomic_buf + i) system_atomic_t {0}; // placement new constructor

        allocated_bytes = bytes;
      }

      if (allocated_n_reduce < n_reduce) {
        if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
          printfQuda("reducer::init count resizing for n_reduce = %d\n", n_reduce);
        if (reduce_count) device_free(reduce_count);
        reduce_count = static_cast<count_t *>(device_malloc(n_reduce * sizeof(count_t)));
        init_reduce<count_t> init(reduce_count, n_reduce);

        allocated_n_reduce = n_reduce;
      }

      if (!init_event) {
        reduceEnd = qudaEventCreate();
        init_event = true;
      }
    }

    void destroy()
    {
      if (init_event) qudaEventDestroy(reduceEnd);

      if (reduce_count) {
        device_free(reduce_count);
        reduce_count = nullptr;
      }
      if (d_reduce) {
        device_free(d_reduce);
        d_reduce = nullptr;
      }
      if (h_reduce) {
        host_free(h_reduce);
        h_reduce = nullptr;
      }
      hd_reduce = nullptr;
    }

  } // namespace reducer
} // namespace quda
