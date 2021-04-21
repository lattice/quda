#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <color_spinor.h>
#include <kernel.h>

namespace quda
{

  // This is the maximum number of color spinors we can process in a single kernel
  // FIXME - make this multi-RHS once we have the multi-RHS framework developed
#define MAX_NVECTOR 1

  template <typename Float, int nColor_, int nvector_>
  struct CloverSigmaOprodArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr int nvector = nvector_;
    using Oprod = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO, 18>::type;
    using F = typename colorspinor_mapper<Float, nSpin, nColor>::type;

    Oprod oprod;
    const F inA[nvector];
    const F inB[nvector];
    real coeff[nvector][2];

    CloverSigmaOprodArg(GaugeField &oprod, const std::vector<ColorSpinorField*> &inA,
                        const std::vector<ColorSpinorField*> &inB,
                        const std::vector<std::vector<double>> &coeff_) :
      kernel_param(dim3(oprod.VolumeCB(), 2, 6)),
      oprod(oprod),
      inA{*inA[0]},
      inB{*inB[0]}
    {
      for (int i = 0; i < nvector; i++) {
        coeff[i][0] = coeff_[i][0];
        coeff[i][1] = coeff_[i][1];
      }
    }
  };

  template <int mu, int nu, typename Arg>
  inline __device__ void sigmaOprod(const Arg &arg, int x_cb, int parity)
  {
    using Spinor = ColorSpinor<typename Arg::real, Arg::nColor, 4>;
    using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
    Link result;

#pragma unroll
    for (int i = 0; i < Arg::nvector; i++) {
      const Spinor A = arg.inA[i](x_cb, parity);
      const Spinor B = arg.inB[i](x_cb, parity);
      Spinor C = A.sigma(nu, mu); // multiply by sigma_mu_nu
      result += arg.coeff[i][parity] * outerProdSpinTrace(C, B);
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
