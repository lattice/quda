#include <gauge_field.h>
#include <blas_quda.h>
#include <blas_lapack.h>
#include <tunable_nd.h>

#include <staggered_kd_build_xinv.h>
#include <kernels/staggered_kd_reorder_xinv_kernel.cuh>

namespace quda {

  template <typename Float, int fineColor, int coarseSpin, int coarseColor, bool dagger_approximation, typename Arg>
  class CalculateStaggeredGeometryReorder : public TunableKernel3D {

    Arg &arg;
    const GaugeField &xInvCoarse;
    GaugeField &meta;

    long long flops() const {
      if (dagger_approximation) {
        // rescale of all values
        return meta.Volume() * Arg::kdBlockSize * fineColor * fineColor * 2ll;
      } else {
        // just a permutation
        return 0ll;
      }
    }

    long long bytes() const
    {
      // 1. Loading xInvCoarse, the coarse KD inverse field
      // 2. Storing meta, the reordered fine KD inverse field
      return xInvCoarse.Bytes() + meta.Bytes();
    }

    unsigned int minThreads() const { return arg.fineVolumeCB; }

  public:
    CalculateStaggeredGeometryReorder(Arg &arg, GaugeField &meta, const GaugeField &xInvCoarse) :
      TunableKernel3D(meta, Arg::kdBlockSize, 2),
      arg(arg),
      xInvCoarse(xInvCoarse),
      meta(meta)
    {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
      strcat(aux,",computeStaggeredGeometryReorder");
      strcat(aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && xInvCoarse.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped," :
             meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU-device," : ",CPU,");
      if (dagger_approximation) strcat(aux, "dagger_approximation");
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        launch_host<ComputeStaggeredGeometryReorder>(tp, stream, arg);
      } else {
        launch_device<ComputeStaggeredGeometryReorder>(tp, stream, arg);
      }
    }

    bool advanceTuneParam(TuneParam &param) const {
      // only do autotuning if we have device fields
      if (xInvCoarse.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
      else return false;
    }
  };

  /**
     @brief Reorder the staggered Kahler-Dirac inverse from a coarse scalar layout to a fine KD geometry

     @param xInvFine[out] KD inverse fine gauge in KD geometry accessor
     @param xInvCoarse[in] KD inverse coarse lattice field accessor
     @param xInvFine_[out] KD inverse fine gauge in KD geometry
     @param xInvCoarse_[in] KD inverse coarse lattice field
     @param scale[in] Scaling factor for the reorder
   */
  template<typename Float, int fineColor, int coarseSpin, int coarseColor, bool dagger_approximation, typename fineXinv, typename coarseXinv>
  void calculateStaggeredGeometryReorder(fineXinv &xInvFine, coarseXinv &xInvCoarse, GaugeField &xInvFine_, const GaugeField &xInvCoarse_, const Float scale)
  {
    // sanity checks
    if (fineColor != 3)
      errorQuda("Input gauge field should have nColor=3, not nColor=%d\n", fineColor);

    if (xInvFine.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    if (fineColor * 16 != coarseColor*coarseSpin)
      errorQuda("Fine nColor=%d is not consistent with KD dof %d", fineColor, coarseColor*coarseSpin);

    int x_size[QUDA_MAX_DIM] = { };
    int xc_size[QUDA_MAX_DIM] = { };
    for (int i = 0; i < nDim; i++) {
      x_size[i] = xInvFine_.X()[i];
      xc_size[i] = xInvCoarse_.X()[i];
      // check that local volumes are consistent
      if (2 * xc_size[i] != x_size[i]) {
        errorQuda("Inconsistent fine dimension %d and coarse KD dimension %d", x_size[i], xc_size[i]);
      }
    }
    x_size[4] = xc_size[4] = 1;

    // Calculate X (KD block), which is really just a permutation of the gauge fields w/in a KD block
    using Arg = CalculateStaggeredGeometryReorderArg<Float,coarseSpin,fineColor,coarseColor,dagger_approximation,fineXinv,coarseXinv>;
    Arg arg(xInvFine, xInvCoarse, x_size, xc_size, scale);
    CalculateStaggeredGeometryReorder<Float, fineColor, coarseSpin, coarseColor, dagger_approximation, Arg> y(arg, xInvFine_, xInvCoarse_);

    QudaFieldLocation location = checkLocation(xInvFine_, xInvCoarse_);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Permuting KD block on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // We know exactly what the scale should be: the max of the input inverse clover
    double max_scale = xInvCoarse_.abs_max();
    if (dagger_approximation) max_scale *= (1.01 * abs(scale));
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Global xInv_max = %e\n", max_scale);

    if (fineXinv::fixedPoint()) {
      arg.fineXinv.resetScale(max_scale);
      xInvFine_.Scale(max_scale);
    }

    y.apply(device::get_default_stream());

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("XInvFine2 = %e\n", xInvFine_.norm2());

  }

  template <typename Float, typename vFloat, int fineColor, bool dagger_approximation>
  void calculateStaggeredGeometryReorder(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout, const Float scale)
  {
    constexpr int coarseSpin = 2;
    constexpr int coarseColor = 8 * fineColor;

    QudaFieldLocation location = xInvFineLayout.Location();

    if (location == QUDA_CPU_FIELD_LOCATION) {

      constexpr QudaGaugeFieldOrder xOrder = QUDA_QDP_GAUGE_ORDER;

      if (xInvFineLayout.FieldOrder() != xOrder) errorQuda("Unsupported field order %d\n", xInvFineLayout.FieldOrder());
      if (xInvCoarseLayout.FieldOrder() != xOrder) errorQuda("Unsupported field order %d\n", xInvCoarseLayout.FieldOrder());

      using xInvFine = typename gauge::FieldOrder<Float,fineColor,1,xOrder,true,vFloat>;
      using xInvCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,xOrder,true,vFloat>;

      xInvFine xInvFineAccessor(xInvFineLayout);
      xInvCoarse xInvCoarseAccessor(const_cast<GaugeField&>(xInvCoarseLayout));

      calculateStaggeredGeometryReorder<Float,fineColor,coarseSpin,coarseColor,dagger_approximation>(xInvFineAccessor, xInvCoarseAccessor, xInvFineLayout, xInvCoarseLayout, scale);

    } else {

      constexpr QudaGaugeFieldOrder xFineOrder = QUDA_FLOAT2_GAUGE_ORDER;
      constexpr QudaGaugeFieldOrder xCoarseOrder = QUDA_MILC_GAUGE_ORDER;

      if (xInvFineLayout.FieldOrder() != xFineOrder) errorQuda("Unsupported field order %d\n", xInvFineLayout.FieldOrder());
      if (xInvCoarseLayout.FieldOrder() != xCoarseOrder) errorQuda("Unsupported field order %d\n", xInvCoarseLayout.FieldOrder());

      using xInvFine = typename gauge::FieldOrder<Float,fineColor,1,xFineOrder,true,vFloat>;
      using xInvCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,xCoarseOrder,true,vFloat>;

      xInvFine xInvFineAccessor(const_cast<GaugeField&>(xInvFineLayout));
      xInvCoarse xInvCoarseAccessor(const_cast<GaugeField&>(xInvCoarseLayout));

      calculateStaggeredGeometryReorder<Float,fineColor,coarseSpin,coarseColor,dagger_approximation>(xInvFineAccessor, xInvCoarseAccessor, xInvFineLayout, xInvCoarseLayout, scale);
    }

  }

  // template on dagger approximation
  template <typename Float, typename vFloat, int fineColor>
  void calculateStaggeredGeometryReorder(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout, const bool dagger_approximation, const double mass)
  {
    if (dagger_approximation) {
      // approximate the inverse with the dagger: the free field for staggered, 
      // B^-1 = 1 / (4 * (d + mass^2)), where the 4 is due to the factor of 2 convention
      Float scale = static_cast<Float>(1. / (4. * (xInvFineLayout.Ndim() + mass * mass)));
      calculateStaggeredGeometryReorder<Float,vFloat,fineColor,true>(xInvFineLayout, xInvCoarseLayout, scale);
    } else {
      Float scale = static_cast<Float>(1);
      calculateStaggeredGeometryReorder<Float,vFloat,fineColor,false>(xInvFineLayout, xInvCoarseLayout, scale);
    }
  }

  // template on "fine", consistent coarse colors
  template <typename Float, typename vFloat>
  void calculateStaggeredGeometryReorder(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout, const bool dagger_approximation, const double mass)
  {
    const int fineColor = xInvFineLayout.Ncolor();
    constexpr int coarseSpin = 2;
    const int coarseColor = xInvCoarseLayout.Ncolor() / coarseSpin;

    if (coarseColor != 8 * fineColor)
      errorQuda("Inconsistent fine color %d and coarse color %d", fineColor, coarseColor);

    if (xInvFineLayout.Ncolor() == 3) {
      calculateStaggeredGeometryReorder<Float,vFloat,3>(xInvFineLayout, xInvCoarseLayout, dagger_approximation, mass);
    } else {
      errorQuda("Unsupported number of colors %d\n", xInvFineLayout.Ncolor());
    }
  }

#if defined(GPU_STAGGERED_DIRAC) && defined(GPU_MULTIGRID)
  void ReorderStaggeredKahlerDiracInverse(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout, const bool dagger_approximation, const double mass) {

    QudaFieldLocation location = checkLocation(xInvFineLayout, xInvCoarseLayout);
    QudaPrecision precision = checkPrecision(xInvFineLayout, xInvCoarseLayout);

    if (xInvFineLayout.Geometry() != QUDA_KDINVERSE_GEOMETRY)
      errorQuda("Unsupported geometry %d", xInvFineLayout.Geometry());

    if (xInvCoarseLayout.Geometry() != QUDA_SCALAR_GEOMETRY)
      errorQuda("Unsupported geometry %d", xInvCoarseLayout.Geometry());

#if QUDA_PRECISION & 4
    if (xInvFineLayout.Precision() == QUDA_SINGLE_PRECISION) {
      calculateStaggeredGeometryReorder<float,float>(xInvFineLayout, xInvCoarseLayout, dagger_approximation, mass);
    } else
#endif
#if QUDA_PRECISION & 2
    if (xInvFineLayout.Precision() == QUDA_HALF_PRECISION) {
      calculateStaggeredGeometryReorder<float, short>(xInvFineLayout, xInvCoarseLayout, dagger_approximation, mass);
    } else
#endif
    {
      errorQuda("Unsupported precision %d", xInvFineLayout.Precision());
    }
  }
#else
  void ReorderStaggeredKahlerDiracInverse(GaugeField &, const GaugeField &, const bool, const double) {
    errorQuda("Staggered fermion support has not been built");
  }
#endif

} //namespace quda
