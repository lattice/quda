#include <util_quda.h>
#include <random_quda.h>
#include <malloc_quda.h>
#include <tunable_nd.h>
#include <kernels/random_init.cuh>

namespace quda {

  class RNGInit : public TunableKernel2D {

    RNG &rng;
    const LatticeField &meta;
    unsigned long long seed;
    unsigned int minThreads() const { return meta.VolumeCB(); }
    bool tuneSharedBytes() const { return false; }

  public:
    RNGInit(RNG &rng, const LatticeField &meta, unsigned long long seed) :
      TunableKernel2D(meta, meta.SiteSubset()),
      rng(rng),
      meta(meta),
      seed(seed)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch_device<init_random>(tp, stream, rngArg(rng.State(), seed, meta));
    }

    long long flops() const { return 0; }
    long long bytes() const { return 0; }
  };

  RNG::RNG(const LatticeField &meta, unsigned long long seedin) :
    size(meta.LocalVolume()),
    state((RNGState *)device_malloc(size * sizeof(RNGState)), [](RNGState *ptr){ device_free(ptr); } ),
    seed(seedin)
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

    RNGInit(*this, meta, seed);
  }

  /*! @brief Backup CURAND array states initialization */
  void RNG::backup()
  {
    backup_state = (RNGState *)safe_malloc(size * sizeof(RNGState));
    qudaMemcpy(backup_state, state.get(), size * sizeof(RNGState), qudaMemcpyDeviceToHost);
  }

  /*! @brief Restore CURAND array states initialization */
  void RNG::restore()
  {
    qudaMemcpy(state.get(), backup_state, size * sizeof(RNGState), qudaMemcpyHostToDevice);
    host_free(backup_state);
  }

} // namespace quda
