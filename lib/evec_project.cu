#include <color_spinor_field.h>
#include <contract_quda.h>
#include <kernels/evec_project.cuh>
#include <tunable_reduction.h>
#include <instantiate.h>

namespace quda
{

  template <typename Float, int nColor> class EvecProjectLaplace3D : TunableMultiReduction
  {
    cvector_ref<const ColorSpinorField> &x;
    cvector_ref<const ColorSpinorField> &y;
    std::vector<double> &result;
    bool tuneSharedBytes() const override { return false; }

  public:
    EvecProjectLaplace3D(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y,
                         std::vector<double> &result) :
      TunableMultiReduction(x[0], 1, x.size() * y.size() * x.X(3), 8), x(x), y(y), result(result)
    {
      strcat(aux, ",nx=");
      char rhs_str[16];
      i32toa(rhs_str, x.size());
      strcat(aux, rhs_str);
      strcat(aux, ",ny=");
      i32toa(rhs_str, y.size());
      strcat(aux, rhs_str);
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      EvecProjectionArg<Float, nColor> arg(x, y);
      launch<EvecProjection>(result, tp, stream, arg);
    }

    long long flops() const override { return 8 * x.size() * y.size() * x.Nspin() * x.Ncolor() * x.Volume(); }
    long long bytes() const override { return x.Bytes() * y.size() + y.Bytes() * x.size(); }
  };

  void evecProjectLaplace3D(std::vector<Complex> &result, cvector_ref<const ColorSpinorField> &x,
                            cvector_ref<const ColorSpinorField> &y)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    checkNative(x[0], y[0]);
    if (x.Nspin() != 4 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());

    // deploy as a tile computation
    auto Lt = x.X(3);

    for (auto tx = 0u; tx < x.size(); tx += max_nx) {
      for (auto ty = 0u; ty < y.size(); ty += max_ny) {
        // compute remainder here
        auto tile_x = std::min(max_nx, x.size() - tx);
        auto tile_y = std::min(max_ny, y.size() - ty);
        std::vector<double> result_tile(2 * x.Nspin() * Lt * tile_x * tile_y);

        instantiate<EvecProjectLaplace3D>(cvector_ref<const ColorSpinorField> {x.begin() + tx, x.begin() + tx + tile_x},
                                          cvector_ref<const ColorSpinorField> {y.begin() + ty, y.begin() + ty + tile_y},
                                          result_tile);

        for (auto i = 0u; i < tile_x; i++) {
          for (auto j = 0u; j < tile_y; j++) {
            for (auto t = 0; t < Lt; t++) {
              for (auto s = 0u; s < 4; s++) {
                result[(((tx + i) * y.size() + (ty + j)) * Lt + t) * 4 + s]
                  = {result_tile[(((t * tile_y + j) * tile_x + i) * 4 + s) * 2 + 0],
                     result_tile[(((t * tile_y + j) * tile_x + i) * 4 + s) * 2 + 1]};
              }
            }
          }
        }
      }
    }

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
