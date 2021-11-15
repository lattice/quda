#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/copy_clover.cuh>

namespace quda {

  using namespace clover;

  template <typename OutOrder, typename InOrder, typename FloatOut, typename FloatIn>
  class CopyClover : TunableKernel2D {
    using Arg = CopyCloverArg<FloatOut, FloatIn, OutOrder, InOrder>;
    using real = typename mapper<FloatOut>::type;
    bool compute_diagonal;
    real *diagonal_d;
    real *diagonal_h;
    CloverField &out;
    const CloverField &in;
    bool inverse;
    FloatOut *Out;
    const FloatIn *In;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    CopyClover(CloverField &out, const CloverField &in, bool inverse, QudaFieldLocation location,
               void *Out, const void *In) :
      TunableKernel2D(in, 2, location),
      compute_diagonal(out.Reconstruct() && !in.Reconstruct()), // if writing to a compressed field, we need to compute the diagonal
      diagonal_d(compute_diagonal ? static_cast<real*>(pool_device_malloc(sizeof(real))) : nullptr),
      diagonal_h(compute_diagonal ? static_cast<real*>(pool_pinned_malloc(sizeof(real))) : nullptr),
      out(out),
      in(in),
      inverse(inverse),
      Out(static_cast<FloatOut*>(Out)),
      In(static_cast<const FloatIn*>(In))
    {
      if (compute_diagonal) {
        char aux2[TuneKey::aux_n];
        strcpy(aux2, aux);
        strcat(aux,",compute_diagonal");
        apply(device::get_default_stream());
        strcpy(aux, aux2);
        compute_diagonal = false;
      }

      apply(device::get_default_stream());
    }

    virtual ~CopyClover()
    {
      if (diagonal_h) pool_pinned_free(diagonal_h);
      if (diagonal_d) pool_device_free(diagonal_d);
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      auto diagonal = location == QUDA_CUDA_FIELD_LOCATION ? diagonal_d : diagonal_h;
      if constexpr (OutOrder::enable_reconstruct && InOrder::enable_reconstruct) {
        launch<CompressedCloverCopy, true>(tp, stream, Arg(out, in, inverse, Out, In, compute_diagonal, diagonal));
      } else {
        launch<CloverCopy, true>(tp, stream, Arg(out, in, inverse, Out, In, compute_diagonal, diagonal));
      }

      if (compute_diagonal) {
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          qudaMemcpyAsync(diagonal_h, diagonal_d, sizeof(real), qudaMemcpyDeviceToHost, stream);
          qudaDeviceSynchronize();
        }
        double diagonal = 0.0;
        // only use the result from node 0 (site 0) for multi-node determinism
        if (comm_rank() == 0) diagonal = *diagonal_h / 2; // factor of two for native normalization
        if (!activeTuning()) comm_broadcast(&diagonal, sizeof(double));
        out.Diagonal(diagonal);
      }
    }

    long long bytes() const { return (out.Bytes() + in.Bytes()) / (compute_diagonal ? in.Volume() : 1); }
  };

  template <typename InOrder, typename FloatOut, typename FloatIn>
  void copyClover(CloverField &out, const CloverField &in, bool inverse, QudaFieldLocation location,
                  void *Out, const void *In)
  {
    if (out.isNative()) {
      if (out.Reconstruct()) {
        using C = typename clover_mapper<FloatOut, 72, false, true>::type;
        CopyClover<C, InOrder, FloatOut, FloatIn>(out, in, inverse, location, Out, In);
      } else {
        using C = typename clover_mapper<FloatOut, 72, false, false>::type;
        CopyClover<C, InOrder, FloatOut, FloatIn>(out, in, inverse, location, Out, In);
      }
    } else if (out.Order() == QUDA_PACKED_CLOVER_ORDER) {
      CopyClover<QDPOrder<FloatOut>, InOrder, FloatOut, FloatIn>(out, in, inverse, location, Out, In);
    } else if (out.Order() == QUDA_QDPJIT_CLOVER_ORDER) {
#ifdef BUILD_QDPJIT_INTERFACE
      CopyClover<QDPJITOrder<FloatOut>, InOrder, FloatOut, FloatIn>(out, in, inverse, location, Out, In);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif
    } else if (out.Order() == QUDA_BQCD_CLOVER_ORDER) {
      errorQuda("BQCD output not supported");
    } else {
      errorQuda("Clover field %d order not supported", out.Order());
    }
  }

  template <typename FloatOut, typename FloatIn> struct CloverCopyOut {
    CloverCopyOut(CloverField &out, const CloverField &in, bool inverse, QudaFieldLocation location,
                  void *Out, const void *In)
    {
      if (in.isNative()) {
        if (in.Reconstruct()) {
          using C = typename clover_mapper<FloatIn, 72, false, true>::type;
          copyClover<C, FloatOut, FloatIn>(out, in, inverse, location, Out, In);
        } else {
          using C = typename clover_mapper<FloatIn, 72, false, false>::type;
          copyClover<C, FloatOut, FloatIn>(out, in, inverse, location, Out, In);
        }
      } else if (in.Order() == QUDA_PACKED_CLOVER_ORDER) {
        copyClover<QDPOrder<FloatIn>, FloatOut, FloatIn>(out, in, inverse, location, Out, In);
      } else if (in.Order() == QUDA_QDPJIT_CLOVER_ORDER) {
#ifdef BUILD_QDPJIT_INTERFACE
        copyClover<QDPJITOrder<FloatIn>, FloatOut, FloatIn>(out, in, inverse, location, Out, In);
#else
        errorQuda("QDPJIT interface has not been built\n");
#endif
      } else if (in.Order() == QUDA_BQCD_CLOVER_ORDER) {
#ifdef BUILD_BQCD_INTERFACE
        copyClover<BQCDOrder<FloatIn>, FloatOut, FloatIn>(out, in, inverse, location, Out, In);
#else
        errorQuda("BQCD interface has not been built\n");
#endif
      } else {
        errorQuda("Clover field %d order not supported", in.Order());
      }
    }
  };

  template <typename FloatIn> struct CloverCopyIn {
    CloverCopyIn(const CloverField &in, CloverField &out, bool inverse, QudaFieldLocation location,
                 void *Out, const void *In)
    {
      // swizzle in/out back to instantiate out precision
      instantiatePrecision2<CloverCopyOut, FloatIn>(out, in, inverse, location, Out, In);
    }
  };

#ifdef GPU_CLOVER_DIRAC
  void copyGenericClover(CloverField &out, const CloverField &in, bool inverse, QudaFieldLocation location,
                         void *Out, const void *In)
  {
    // swizzle in/out since we first want to instantiate precision
    instantiatePrecision<CloverCopyIn>(in, out, inverse, location, Out, In);
  }
#else
  void copyGenericClover(CloverField &, const CloverField &, bool, QudaFieldLocation, void *, const void *)
  {
    errorQuda("Clover has not been built");
  }
#endif

} // namespace quda
