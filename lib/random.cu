#include <util_quda.h>
#include <random_quda.h>
#include <malloc_quda.h>
#include <tunable_nd.h>
#include <kernels/random_init.cuh>

namespace quda {

  class RNGInit : public TunableKernel2D {

    RNGState *state;
    unsigned long long seed;
    unsigned int minThreads() const { return field.VolumeCB(); }
    bool tuneSharedBytes() const { return false; }

  public:
    RNGInit(const LatticeField &meta, RNGState *state, unsigned long long seed) :
      TunableKernel2D(meta, meta.SiteSubset()),
      state(state),
      seed(seed)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch_device<init_random>(tp, stream, rngArg(state, seed, field));
    }

    long long flops() const { return 0; }
    long long bytes() const { return 0; }
  };

  RNG::RNG(const LatticeField &meta, unsigned long long seedin) :
    meta(meta),
    size(meta.Volume()),
    state((RNGState *)device_malloc(size * sizeof(RNGState))),
    seed(seedin),
    master(true)
  {
#if defined(XORWOW)
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Using randStateXORWOW\n");
#elif defined(RG32k3a)
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Using randStateMRG32k3a\n");
#else
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Using randStateMRG32k3a\n");
#endif

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
      printfQuda("Allocated array of random numbers with size: %.2f MB\n",
                 size * sizeof(RNGState) / (float)(1048576));

    RNGInit(meta, state, seed);
  }

  RNG::RNG(const RNG &rng) :
    meta(rng.meta),
    size(rng.size),
    state(rng.state),
    seed(rng.seed),
    master(false) {}

  RNG::~RNG()
  {
    if (master) {
      device_free(state);
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
        printfQuda("Free array of random numbers with size: %.2f MB\n", size * sizeof(RNGState) / (float)(1048576));
    }
  }

  /*! @brief Backup CURAND array states initialization */
  void RNG::backup()
  {
    backup_state = (RNGState *)safe_malloc(size * sizeof(RNGState));
    qudaMemcpy(backup_state, state, size * sizeof(RNGState), qudaMemcpyDeviceToHost);
  }

  /*! @brief Restore CURAND array states initialization */
  void RNG::restore()
  {
    qudaMemcpy(state, backup_state, size * sizeof(RNGState), qudaMemcpyHostToDevice);
    host_free(backup_state);
  }

} // namespace quda
