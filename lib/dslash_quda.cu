#include "color_spinor_field.h"
#include "dslash_quda.h"
#include "dslash_shmem.h"
#include <dslash_policy.cuh>
#include "tunable_nd.h"
#include "instantiate.h"
#include <shmem_helper.cuh>
#include "kernels/dslash_shmem_helper.cuh"

namespace quda {

  namespace dslash {
    int it = 0;

    static constexpr int nDim = 4;
    static constexpr int nDir = 2;
    static constexpr int nStream = nDim * nDir + 1;

    qudaEvent_t packEnd[2];
    qudaEvent_t gatherEnd[nStream];
    qudaEvent_t scatterEnd[nStream];
    qudaEvent_t dslashStart[2];

    // for shmem lightweight sync
    shmem_sync_t sync_counter = 10;
    shmem_sync_t sync_counter2 = 10;
    shmem_sync_t get_dslash_shmem_sync_counter() { return sync_counter; }
    shmem_sync_t set_dslash_shmem_sync_counter(shmem_sync_t count) { return sync_counter = count; }
    shmem_sync_t inc_dslash_shmem_sync_counter() { return ++sync_counter; }
    shmem_sync_t get_exchangeghost_shmem_sync_counter() { return sync_counter2; }
    shmem_sync_t set_exchangeghost_shmem_sync_counter(shmem_sync_t count) { return sync_counter2 = count; }
    shmem_sync_t inc_exchangeghost_shmem_sync_counter() { return ++sync_counter2; }

#ifdef NVSHMEM_COMMS
    shmem_sync_t *sync_arr = nullptr;
    shmem_retcount_intra_t *_retcount_intra = nullptr;
    shmem_retcount_inter_t *_retcount_inter = nullptr;
    shmem_interior_done_t *_interior_done = nullptr;
    shmem_interior_count_t *_interior_count = nullptr;
    shmem_sync_t *get_dslash_shmem_sync_arr() { return sync_arr; }
    shmem_sync_t *get_exchangeghost_shmem_sync_arr() { return sync_arr + 2 * QUDA_MAX_DIM; }
    shmem_retcount_intra_t *get_shmem_retcount_intra() { return _retcount_intra; }
    shmem_retcount_inter_t *get_shmem_retcount_inter() { return _retcount_inter; }
    shmem_interior_done_t *get_shmem_interior_done() { return _interior_done; }
    shmem_interior_count_t *get_shmem_interior_count() { return _interior_count; }
#endif

    // these variables are used for benchmarking the dslash components in isolation
    bool dslash_pack_compute;
    bool dslash_interior_compute;
    bool dslash_exterior_compute;
    bool dslash_comms;
    bool dslash_copy;

    // whether the dslash policy tuner has been enabled
    bool dslash_policy_init;

    // used to keep track of which policy to start the autotuning
    int first_active_policy;
    int first_active_p2p_policy;

    // list of dslash policies that are enabled
    std::vector<QudaDslashPolicy> policies;

    // list of p2p policies that are enabled
    std::vector<QudaP2PPolicy> p2p_policies;

    // string used as a tunekey to ensure we retune if the dslash policy env changes
    char policy_string[TuneKey::aux_n];

    // FIX this is a hack from hell
    // Auxiliary work that can be done while waiting on comms to finis
    Worker *aux_worker;
  }

  template <typename T>
  struct init_dslash : public TunableKernel1D {
    T *counter;
    unsigned int size;
    long long bytes() const { return size * sizeof(T); }
    unsigned int minThreads() const { return size; }

    init_dslash(T *counter, unsigned int size) :
      TunableKernel1D(size, QUDA_CUDA_FIELD_LOCATION),
      counter(counter),
      size(size)
    { apply(device::get_default_stream()); }

    void apply(const qudaStream_t &stream)
    {
      auto tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch_device<init_dslash_atomic>(tp, stream, init_dslash_atomic_arg<T>(counter, size));
    }
  };

  template <typename T>
  struct init_arr : public TunableKernel1D {
    T *counter;
    T val;
    unsigned int size;
    long long bytes() const { return size * sizeof(T); }
    unsigned int minThreads() const { return size; }

    init_arr(T *counter, T val, unsigned int size) :
      TunableKernel1D(size, QUDA_CUDA_FIELD_LOCATION),
      counter(counter),
      val(val),
      size(size)
    { apply(device::get_default_stream()); }

    void apply(const qudaStream_t &stream)
    {
      auto tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch_device<init_sync_arr>(tp, stream, init_arr_arg<T>(counter, val, size));
    }
  };

#ifdef NVSHMEM_COMMS
  template <typename T> struct dslash_shmem_signal_wait : public TunableKernel1D {

    long long bytes() const { return 0; }
    unsigned int minThreads() const { return 8; }

    dslash_shmem_signal_wait() : TunableKernel1D(8, QUDA_CUDA_FIELD_LOCATION) { }

    void apply(const qudaStream_t &stream)
    {
      auto tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch_device<shmem_signal_wait>(tp, stream, shmem_signal_wait_arg<T>());
    }
  };

  namespace dslash
  {
    void shmem_signal_wait_all()
    {
      dslash_shmem_signal_wait<quda::dslash::shmem_sync_t> d;
      d.apply(device::get_default_stream());
    }
  } // namespace dslash
#endif

  void createDslashEvents()
  {
    using namespace dslash;
    for (int i=0; i<nStream; i++) {
      gatherEnd[i] = qudaEventCreate();
      scatterEnd[i] = qudaEventCreate();
    }
    for (int i=0; i<2; i++) {
      packEnd[i] = qudaEventCreate();
      dslashStart[i] = qudaEventCreate();
    }
#ifdef NVSHMEM_COMMS
    sync_arr = static_cast<shmem_sync_t *>(device_comms_pinned_malloc(4 * QUDA_MAX_DIM * sizeof(shmem_sync_t)));

    // initialize to 9 here so where we need to do tuning we can skip
    // the wait if necessary by using smaller values
    init_arr<shmem_sync_t>(sync_arr, static_cast<shmem_sync_t>(9), 4 * QUDA_MAX_DIM);
    sync_counter = 10;

    // atomic for controlling signaling in nvshmem packing
    _retcount_intra
      = static_cast<shmem_retcount_intra_t *>(device_pinned_malloc(2 * QUDA_MAX_DIM * sizeof(shmem_retcount_intra_t)));
    init_dslash<shmem_retcount_intra_t>(_retcount_intra, 2 * QUDA_MAX_DIM);
    _retcount_inter
      = static_cast<shmem_retcount_inter_t *>(device_pinned_malloc(2 * QUDA_MAX_DIM * sizeof(shmem_retcount_inter_t)));
    init_dslash<shmem_retcount_inter_t>(_retcount_inter, 2 * QUDA_MAX_DIM);

    // workspace for interior done sync in uber kernel
    _interior_done = static_cast<shmem_interior_done_t *>(device_pinned_malloc(sizeof(shmem_interior_done_t)));
    init_dslash<shmem_interior_done_t>(_interior_done, 1);
    _interior_count = static_cast<shmem_interior_count_t *>(device_pinned_malloc(sizeof(shmem_interior_count_t)));
    init_dslash<shmem_interior_count_t>(_interior_count, 1);
#endif

    aux_worker = NULL;

    dslash_pack_compute = true;
    dslash_interior_compute = true;
    dslash_exterior_compute = true;
    dslash_comms = true;
    dslash_copy = true;

    dslash_policy_init = false;
    first_active_policy = 0;
    first_active_p2p_policy = 0;

    // list of dslash policies that are enabled
    policies = std::vector<QudaDslashPolicy>(
        static_cast<int>(QudaDslashPolicy::QUDA_DSLASH_POLICY_DISABLED), QudaDslashPolicy::QUDA_DSLASH_POLICY_DISABLED);

    // list of p2p policies that are enabled
    p2p_policies = std::vector<QudaP2PPolicy>(
        static_cast<int>(QudaP2PPolicy::QUDA_P2P_POLICY_DISABLED), QudaP2PPolicy::QUDA_P2P_POLICY_DISABLED);

    strcat(policy_string, ",pol=");
  }

  void destroyDslashEvents()
  {
    using namespace dslash;

    for (int i=0; i<nStream; i++) {
      qudaEventDestroy(gatherEnd[i]);
      qudaEventDestroy(scatterEnd[i]);
    }

    for (int i=0; i<2; i++) {
      qudaEventDestroy(packEnd[i]);
      qudaEventDestroy(dslashStart[i]);
    }
#ifdef NVSHMEM_COMMS
    device_comms_pinned_free(sync_arr);
    device_pinned_free(_retcount_intra);
    device_pinned_free(_retcount_inter);
    device_pinned_free(_interior_done);
    device_pinned_free(_interior_count);
#endif
  }

} // namespace quda
