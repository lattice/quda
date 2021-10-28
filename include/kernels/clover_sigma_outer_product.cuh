#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <color_spinor.h>

namespace quda
{

  // This is the maximum number of color spinors we can process in a single kernel
  // FIXME - make this multi-RHS once we have the multi-RHS framework developed
#define MAX_NVECTOR 1

  template <typename Float, int nColor_> struct CloverSigmaOprodArg {
    typedef typename mapper<Float>::type real;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    using Oprod = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO, 18>::type;
    using F = typename colorspinor_mapper<double, nSpin, nColor>::type;

    Oprod oprod;
    const F inA[MAX_NVECTOR];
    const F inB[MAX_NVECTOR];
    Float coeff[MAX_NVECTOR][2];
    unsigned int length;
    int nvector;

    CloverSigmaOprodArg(GaugeField &oprod, const std::vector<ColorSpinorField*> &inA, const std::vector<ColorSpinorField*> &inB,
                        const std::vector<std::vector<double>> &coeff_, int nvector) :
      oprod(oprod),
      inA{*inA[0]},
      inB{*inB[0]},
      length(oprod.VolumeCB()),
      nvector(nvector)
    {
      for (int i = 0; i < nvector; i++) {
        coeff[i][0] = coeff_[i][0];
        coeff[i][1] = coeff_[i][1];
      }
    }
  };

  template <typename real, int nvector, int mu, int nu, int parity, typename Arg>
  inline __device__ void sigmaOprod(Arg &arg, int idx)
  {
    typedef complex<real> Complex;
    Matrix<Complex, 3> result;

#pragma unroll
    for (int i = 0; i < nvector; i++) {
      ColorSpinor<real, Arg::nColor, 4> A = arg.inA[i](idx, parity);
      ColorSpinor<real, Arg::nColor, 4> B = arg.inB[i](idx, parity);

      // multiply by sigma_mu_nu
      ColorSpinor<real, 3, 4> C = A.sigma(nu, mu);
      result += arg.coeff[i][parity] * outerProdSpinTrace(C, B);
    }

    result -= conj(result);

    Matrix<Complex, 3> temp = arg.oprod((mu - 1) * mu / 2 + nu, idx, parity);
    temp = result + temp;
    arg.oprod((mu - 1) * mu / 2 + nu, idx, parity) = temp;
  }

  template <int nvector, typename real, typename Arg> __global__ void sigmaOprodKernel(Arg arg)
  {
    typedef complex<real> Complex;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int parity = blockIdx.y * blockDim.y + threadIdx.y;
    int mu_nu = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= arg.length) return;
    if (mu_nu >= 6) return;

    switch (parity) {
    case 0:
      switch (mu_nu) {
      case 0: sigmaOprod<real, nvector, 1, 0, 0>(arg, idx); break;
      case 1: sigmaOprod<real, nvector, 2, 0, 0>(arg, idx); break;
      case 2: sigmaOprod<real, nvector, 2, 1, 0>(arg, idx); break;
      case 3: sigmaOprod<real, nvector, 3, 0, 0>(arg, idx); break;
      case 4: sigmaOprod<real, nvector, 3, 1, 0>(arg, idx); break;
      case 5: sigmaOprod<real, nvector, 3, 2, 0>(arg, idx); break;
      }
      break;
    case 1:
      switch (mu_nu) {
      case 0: sigmaOprod<real, nvector, 1, 0, 1>(arg, idx); break;
      case 1: sigmaOprod<real, nvector, 2, 0, 1>(arg, idx); break;
      case 2: sigmaOprod<real, nvector, 2, 1, 1>(arg, idx); break;
      case 3: sigmaOprod<real, nvector, 3, 0, 1>(arg, idx); break;
      case 4: sigmaOprod<real, nvector, 3, 1, 1>(arg, idx); break;
      case 5: sigmaOprod<real, nvector, 3, 2, 1>(arg, idx); break;
      }
      break;
    }

  } // sigmaOprodKernel

} // namespace quda
