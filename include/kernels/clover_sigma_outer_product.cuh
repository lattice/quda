#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <color_spinor.h>
#include <kernel.h>

namespace quda
{

  template <typename Float, int nColor_, bool doublet_> struct CloverSigmaOprodArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr bool doublet = doublet_; // whether we applying the operator to a doublet
    static constexpr int n_flavor = doublet ? 2 : 1;
    using Oprod = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO, 18>::type;
    using F = typename colorspinor_mapper<Float, nSpin, nColor, false, false, true>::type;

    static constexpr int max_n_rhs = MAX_MULTI_RHS;
    const unsigned int n_rhs;

    Oprod oprod;
    const unsigned int volume_4d_cb;
    F inA[max_n_rhs];
    F inB[max_n_rhs];
    array_2d<real, max_n_rhs, 2> coeff;

    CloverSigmaOprodArg(GaugeField &oprod, cvector_ref<const ColorSpinorField> &inA,
                        cvector_ref<const ColorSpinorField> &inB, const std::vector<array<double, 2>> &coeff_) :
      kernel_param(dim3(oprod.VolumeCB(), 2, 6)), n_rhs(inA.size()), oprod(oprod), volume_4d_cb(inA.VolumeCB() / 2)
    {
      for (auto i = 0u; i < n_rhs; i++) {
        this->inA[i] = inA[i];
        this->inB[i] = inB[i];
        coeff[i] = {static_cast<real>(coeff_[i][0]), static_cast<real>(coeff_[i][1])};
      }
    }
  };

  template <int mu, int nu, typename Arg>
  inline __device__ void sigmaOprod(const Arg &arg, int x_cb, int parity)
  {
    using Spinor = ColorSpinor<typename Arg::real, Arg::nColor, 4>;
    using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
    Link result = {};

    for (unsigned int i = 0; i < arg.n_rhs; i++) {
#pragma unroll
      for (int flavor = 0; flavor < Arg::n_flavor; flavor++) {
        const int flavor_offset_idx = flavor * (arg.volume_4d_cb);
        const Spinor A = arg.inA[i](x_cb + flavor_offset_idx, parity);
        const Spinor B = arg.inB[i](x_cb + flavor_offset_idx, parity);
        Spinor C = A.sigma(nu, mu); // multiply by sigma_mu_nu
        result += arg.coeff[i][parity] * outerProdSpinTrace(C, B);
      }
    }

    result -= conj(result);

    Link temp = arg.oprod((mu - 1) * mu / 2 + nu, x_cb, parity);
    arg.oprod((mu - 1) * mu / 2 + nu, x_cb, parity) = result + temp;
  }

  template <typename Arg> struct SigmaOprod {
    const Arg &arg;
    constexpr SigmaOprod(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int mu_nu)
    {
      switch (mu_nu) {
      case 0: sigmaOprod<1, 0>(arg, x_cb, parity); break;
      case 1: sigmaOprod<2, 0>(arg, x_cb, parity); break;
      case 2: sigmaOprod<2, 1>(arg, x_cb, parity); break;
      case 3: sigmaOprod<3, 0>(arg, x_cb, parity); break;
      case 4: sigmaOprod<3, 1>(arg, x_cb, parity); break;
      case 5: sigmaOprod<3, 2>(arg, x_cb, parity); break;
      }
    } // sigmaOprodKernel
  };

} // namespace quda
