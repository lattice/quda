#include <color_spinor_field.h>
#include <contract_quda.h>
#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <kernels/evec_project.cuh>
#include <comm_quda.h>

namespace quda
{

  template <typename Float, int nColor> class EvecProjectLaplace3D : TunableMultiReduction
  {
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    std::vector<Complex> result;

  public:
    EvecProjectLaplace3D(const ColorSpinorField &x, const ColorSpinorField &y, std::vector<Complex> &result) :
      TunableMultiReduction(x, 1, x.X()[3]), x(x), y(y), result(result)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      std::vector<double> result_local(2 * x.Nspin() * x.X()[3]);

      EvecProjectionArg<Float, nColor> arg(x, y);
      launch<EvecProjection>(result_local, tp, stream, arg);

      // now reassemble global array
      if (!activeTuning()) {
        for (int t = 0; t < x.X()[3]; t++) {
          for (int s = 0; s < 4; s++) {
            result[(comm_coord(3) * comm_dim(3) + t) * 4 + s]
              = {result_local[(t * 4 + s) * 2 + 0], result_local[(t * 4 + s) * 2 + 1]};
          }
        }
      }
    }

    // 4 prop spins, 1 evec spin, 3 color, 6 complex, lattice volume
    long long flops() const { return 4 * 3 * 6ll * x.Volume(); }
    long long bytes() const { return x.Bytes() + y.Bytes(); }
  };

  void evecProjectLaplace3D(std::vector<Complex> &result, const ColorSpinorField &x, const ColorSpinorField &y)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    checkPrecision(x, y);
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (x.Nspin() != 4 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());

    instantiate<EvecProjectLaplace3D>(x, y, result);

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
