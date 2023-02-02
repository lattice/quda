#include <cstdlib>
#include <cstdio>

#include <gauge_field.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/unitarize_force.cuh>

namespace quda {

  namespace fermion_force {

    void setUnitarizeForceConstants(double unitarize_eps_, double force_filter_,
				    double max_det_error_, bool allow_svd_, bool svd_only_,
				    double svd_rel_error_, double svd_abs_error_)
    {
      unitarize_eps = unitarize_eps_;
      force_filter = force_filter_;
      max_det_error = max_det_error_;
      allow_svd = allow_svd_;
      svd_only = svd_only_;
      svd_rel_error = svd_rel_error_;
      svd_abs_error = svd_abs_error_;
    }

    template <typename Float, int nColor, QudaReconstructType recon> class ForceUnitarize : TunableKernel2D
    {
      UnitarizeForceArg<Float, nColor, recon> arg;
      const GaugeField &meta;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      ForceUnitarize(GaugeField &newForce, const GaugeField &oldForce, const GaugeField &u, int* fails) :
        TunableKernel2D(u, 2),
        arg(newForce, oldForce, u, fails, unitarize_eps, force_filter,
            max_det_error, allow_svd, svd_only, svd_rel_error, svd_abs_error),
        meta(u)
      {
        apply(device::get_default_stream());
        qudaDeviceSynchronize(); // need to synchronize to ensure failure write has completed
      }

      void apply(const qudaStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	launch<UnitarizeForce>(tp, stream, arg);
      }

      void preTune() { ; }
      void postTune() { qudaMemset(arg.fails, 0, sizeof(int)); } // reset fails counter

      long long flops() const { return 4ll*4528*meta.Volume(); }
      long long bytes() const { return 4ll * meta.Volume() * (arg.force.Bytes() + arg.force_old.Bytes() + arg.u.Bytes()); }
    }; // UnitarizeForce

#ifdef GPU_STAGGERED_DIRAC
    void unitarizeForce(GaugeField &newForce, const GaugeField &oldForce, const GaugeField &u,
			int* fails)
    {
      checkReconstruct(u, oldForce, newForce);
      checkPrecision(u, oldForce, newForce);

      if (!u.isNative() || !oldForce.isNative() || !newForce.isNative())
        errorQuda("Only native order supported");

      instantiate<ForceUnitarize, ReconstructNone>(newForce, oldForce, u, fails);
    }
#else
    void unitarizeForce(GaugeField &, const GaugeField &, const GaugeField &, int*)
    {
      errorQuda("HISQ force has not been built");
    }
#endif

    template <typename Float, typename Arg> void unitarizeForceCPU(Arg &arg)
    {
      Matrix<complex<double>, 3> v, result, oprod;
      Matrix<complex<Float>, 3> v_tmp, result_tmp, oprod_tmp;

      for (int parity = 0; parity < 2; parity++) {
        for (unsigned int i = 0; i < arg.threads.x; i++) {
          for (int dir = 0; dir < 4; dir++) {
            oprod_tmp = arg.force_old(dir, i, parity);
            v_tmp = arg.u(dir, i, parity);
            v = v_tmp;
            oprod = oprod_tmp;

            getUnitarizeForceSite<double>(result, v, oprod, arg);

            result_tmp = result;
            arg.force(dir, i, parity) = result_tmp;
          }
        }
      }
    }

#ifdef GPU_STAGGERED_DIRAC
    void unitarizeForceCPU(GaugeField &newForce, const GaugeField &oldForce, const GaugeField &u)
    {
      if (checkLocation(newForce, oldForce, u) != QUDA_CPU_FIELD_LOCATION) errorQuda("Location must be CPU");
      int num_failures = 0;
      constexpr int nColor = 3;
      if (u.Order() == QUDA_MILC_GAUGE_ORDER) {
        if (u.Precision() == QUDA_DOUBLE_PRECISION) {
          UnitarizeForceArg<double, nColor, QUDA_RECONSTRUCT_NO, QUDA_MILC_GAUGE_ORDER> arg(
            newForce, oldForce, u, &num_failures, unitarize_eps, force_filter, max_det_error, allow_svd, svd_only,
            svd_rel_error, svd_abs_error);
          unitarizeForceCPU<double>(arg);
        } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
          UnitarizeForceArg<float, nColor, QUDA_RECONSTRUCT_NO, QUDA_MILC_GAUGE_ORDER> arg(
            newForce, oldForce, u, &num_failures, unitarize_eps, force_filter, max_det_error, allow_svd, svd_only,
            svd_rel_error, svd_abs_error);
          unitarizeForceCPU<float>(arg);
        } else {
          errorQuda("Precision = %d not supported", u.Precision());
        }
      } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
        if (u.Precision() == QUDA_DOUBLE_PRECISION) {
          UnitarizeForceArg<double, nColor, QUDA_RECONSTRUCT_NO, QUDA_QDP_GAUGE_ORDER> arg(
            newForce, oldForce, u, &num_failures, unitarize_eps, force_filter, max_det_error, allow_svd, svd_only,
            svd_rel_error, svd_abs_error);
          unitarizeForceCPU<double>(arg);
        } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
          UnitarizeForceArg<float, nColor, QUDA_RECONSTRUCT_NO, QUDA_QDP_GAUGE_ORDER> arg(
            newForce, oldForce, u, &num_failures, unitarize_eps, force_filter, max_det_error, allow_svd, svd_only,
            svd_rel_error, svd_abs_error);
          unitarizeForceCPU<float>(arg);
        } else {
          errorQuda("Precision = %d not supported", u.Precision());
        }
      } else {
        errorQuda("Only MILC and QDP gauge orders supported\n");
      }
      if (num_failures) errorQuda("Unitarization failed, failures = %d", num_failures);
    } // unitarize_force_cpu
#else
    void unitarizeForceCPU(GaugeField &, const GaugeField &, const GaugeField &)
    {
      errorQuda("HISQ force has not been built");
    }
#endif

  } // namespace fermion_force

} // namespace quda
