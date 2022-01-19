#include <gauge_field.h>
#include <blas_quda.h>
#include <blas_lapack.h>
#include <tunable_nd.h>
#include <instantiate.h>

#include <staggered_kd_build_xinv.h>
#include <kernels/staggered_kd_reorder_xinv_kernel.cuh>

namespace quda {

  template <typename Float, int fineColor, bool dagger_approximation>
  class CalculateStaggeredGeometryReorder : public TunableKernel3D {

    GaugeField &fineXinv;
    const GaugeField &coarseXinv;
    double scale;

    long long flops() const {
      if (dagger_approximation) {
        // rescale of all values
        return fineXinv.Volume() * fineXinv.Geometry() * fineXinv.Ncolor() * fineXinv.Ncolor() * 2ll;
      } else {
        // just a permutation
        return 0ll;
      }
    }

    long long bytes() const
    {
      // 1. Loading coarseXinv, the coarse KD inverse field
      // 2. Storing fineXinv, the reordered fine KD inverse field
      return coarseXinv.Bytes() + fineXinv.Bytes();
    }

    unsigned int minThreads() const { return fineXinv.VolumeCB(); }

  public:
    CalculateStaggeredGeometryReorder(GaugeField& fineXinv, const GaugeField& coarseXinv, const double scale) :
      TunableKernel3D(fineXinv, QUDA_KDINVERSE_GEOMETRY, 2),
      fineXinv(fineXinv),
      coarseXinv(coarseXinv),
      scale(scale)
    {
      checkPrecision(fineXinv, coarseXinv);
      checkLocation(fineXinv, coarseXinv);
      if (fineXinv.Geometry() != QUDA_KDINVERSE_GEOMETRY)
        errorQuda("Unsupported geometry %d", fineXinv.Geometry());
      if (fineXinv.Ndim() != 4)
        errorQuda("Number of dimensions %d is not supported", fineXinv.Ndim());
      if (coarseXinv.Geometry() != QUDA_SCALAR_GEOMETRY)
        errorQuda("Unsupported geometry %d", coarseXinv.Geometry());

      strcat(aux,",computeStaggeredGeometryReorder");
      if (dagger_approximation) strcat(aux, ",dagger_approximation");

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (fineXinv.Location() == QUDA_CPU_FIELD_LOCATION) {
        constexpr QudaGaugeFieldOrder fineOrder = QUDA_QDP_GAUGE_ORDER;
        constexpr QudaGaugeFieldOrder coarseOrder = QUDA_QDP_GAUGE_ORDER;
        CalculateStaggeredGeometryReorderArg<Float,fineColor,fineOrder,coarseOrder,dagger_approximation> arg(fineXinv, coarseXinv, scale);
        launch_host<ComputeStaggeredGeometryReorder>(tp, stream, arg);

      } else if (fineXinv.Location() == QUDA_CUDA_FIELD_LOCATION) {
        constexpr QudaGaugeFieldOrder fineOrder = QUDA_FLOAT2_GAUGE_ORDER;
        constexpr QudaGaugeFieldOrder coarseOrder = QUDA_MILC_GAUGE_ORDER;
        CalculateStaggeredGeometryReorderArg<Float,fineColor,fineOrder,coarseOrder,dagger_approximation> arg(fineXinv, coarseXinv, scale);
        launch_device<ComputeStaggeredGeometryReorder>(tp, stream, arg);

      }
    }
  };

  template<typename Float, int fineColor>
  struct calculateStaggeredGeometryReorder {
    calculateStaggeredGeometryReorder(GaugeField &fineXinv, const GaugeField &coarseXinv, const bool dagger_approximation, const double mass) {
      // template on dagger approximation
      if (dagger_approximation)  {
        // Approximate the inverse with the dagger, multiplied by a rescale factor which
        // makes the norm of the dagger max equal the norm of the inverse in the free field
        double xinv_max_abs = coarseXinv.abs_max(0);
        double scale = 1. / (4. * (xinv_max_abs * xinv_max_abs + mass * mass));

        // reset scales as appropriate
        if constexpr (sizeof(Float) < QUDA_SINGLE_PRECISION) {
          double max_scale = xinv_max_abs * scale;
          if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Global xInv_max = %e\n", max_scale);

          fineXinv.Scale(max_scale);
        }
        CalculateStaggeredGeometryReorder<Float,fineColor,true>(fineXinv, coarseXinv, scale);
      } else {
        double scale = 1.;
        CalculateStaggeredGeometryReorder<Float,fineColor,false>(fineXinv, coarseXinv, scale);
      }
    }
  };

#if defined(GPU_STAGGERED_DIRAC) && defined(GPU_MULTIGRID)
  /**
     @brief Reorder the staggered Kahler-Dirac inverse from a coarse scalar layout to a fine KD geometry

     @param fineXinv[out] KD inverse fine gauge in KD geometry
     @param coarseXinv[in] KD inverse coarse lattice field
     @param dagger_approximation[in] Whether or not to apply the dagger approximation
     @param mass[in] Mass of staggered fermion (used for dagger approximation only)
   */
  void ReorderStaggeredKahlerDiracInverse(GaugeField &fineXinv, const GaugeField &coarseXinv, const bool dagger_approximation, const double mass) {
    // Instantiate based on precision, number of colors
    instantiate<calculateStaggeredGeometryReorder>(fineXinv, coarseXinv, dagger_approximation, mass);
  }
#else
  void ReorderStaggeredKahlerDiracInverse(GaugeField &, const GaugeField &, const bool, const double) {
    errorQuda("Staggered fermion support has not been built");
  }
#endif

} //namespace quda
