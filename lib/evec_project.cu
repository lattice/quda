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
    std::vector<Complex> &result;
    bool tuneSharedBytes() const override { return false; }

  public:
    EvecProjectLaplace3D(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y,
                         std::vector<Complex> &result) :
      TunableMultiReduction(x[0], 1, x.size() * y.size() * x[0].X()[3], 8), x(x), y(y), result(result)
    {
      assert(result.size() == x.Nspin() * x[0].X()[3] * x.size() * y.size());
      strcat(aux, ",nx=");
      char rhs_str[16];
      i32toa(rhs_str, x.size());
      strcat(aux, rhs_str);
      strcat(aux, ",ny=");
      i32toa(rhs_str, y.size());
      strcat(aux, rhs_str);
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      auto Lt = x[0].X()[3];
      std::vector<double> result_local(2 * x.Nspin() * Lt * x.size() * y.size());

      EvecProjectionArg<Float, nColor> arg(x, y);
      launch<EvecProjection>(result_local, tp, stream, arg);

      // now reassemble global array
      if (!activeTuning()) {
        for (auto i = 0u; i < x.size(); i++) {
          for (auto j = 0u; j < y.size(); j++) {
            for (int t = 0; t < Lt; t++) {
              for (int s = 0; s < 4; s++) {
                result[((i * y.size() + j) * Lt + t) * 4 + s]
                  = {result_local[(((t * y.size() + j) * x.size() + i) * 4 + s) * 2 + 0],
                     result_local[(((t * y.size() + j) * x.size() + i) * 4 + s) * 2 + 1]};
              }
            }
          }
        }
      }
    }

    long long flops() const { return 8 * x.size() * y.size() * x.Nspin() * x.Ncolor() * x.Volume(); }
    long long bytes() const { return x.Bytes() * y.size() + y.Bytes() * x.size(); }
  };

  void evecProjectLaplace3D(std::vector<Complex> &result, cvector_ref<const ColorSpinorField> &x,
                            cvector_ref<const ColorSpinorField> &y)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    checkNative(x[0], y[0]);
    if (x.Nspin() != 4 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());

    // deploy as a tile computation
    auto Lt = x[0].X()[3];

    for (auto tx = 0u; tx < x.size(); tx += max_nx) {
      for (auto ty = 0u; ty < y.size(); ty += max_ny) {
        // compute remainder here
        auto tile_x = std::min(max_nx, x.size() - tx);
        auto tile_y = std::min(max_ny, y.size() - ty);
        std::vector<Complex> result_tile(x.Nspin() * Lt * tile_x * tile_y);

        instantiate<EvecProjectLaplace3D>(cvector_ref<const ColorSpinorField> {x.begin() + tx, x.begin() + tx + tile_x},
                                          cvector_ref<const ColorSpinorField> {y.begin() + ty, y.begin() + ty + tile_y},
                                          result_tile);

        for (auto i = 0u; i < tile_x; i++) {
          for (auto j = 0u; j < tile_y; j++) {
            for (auto t = 0; t < Lt; t++) {
              for (auto s = 0u; s < 4; s++) {
                result[(((tx + i) * y.size() + (ty + j)) * Lt + t) * 4 + s]
                  = result_tile[((i * tile_y + j) * Lt + t) * 4 + s];
              }
            }
          }
        }
      }
    }

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
