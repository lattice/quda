#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <color_spinor.h>

namespace quda
{

#include <texture.h> // we need to convert this kernel to using colorspinor accessors

  // This is the maximum number of color spinors we can process in a single kernel
#if (CUDA_VERSION < 8000)
#define MAX_NVECTOR 1 // multi-vector code doesn't seem to work well with CUDA 7.x
#else
#define MAX_NVECTOR 9
#endif

  template <typename Float, typename Output, typename InputA, typename InputB> struct CloverSigmaOprodArg {
    Output oprod;
    InputA inA[MAX_NVECTOR];
    InputB inB[MAX_NVECTOR];
    Float coeff[MAX_NVECTOR][2];
    unsigned int length;
    int nvector;

    CloverSigmaOprodArg(Output &oprod, InputA *inA_, InputB *inB_, const std::vector<std::vector<double>> &coeff_,
        const GaugeField &meta, int nvector) :
        oprod(oprod),
        length(meta.VolumeCB()),
        nvector(nvector)
    {
      for (int i = 0; i < nvector; i++) {
        inA[i] = inA_[i];
        inB[i] = inB_[i];
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
      ColorSpinor<real, 3, 4> A, B;

      arg.inA[i].load(static_cast<Complex *>(A.data), idx, parity);
      arg.inB[i].load(static_cast<Complex *>(B.data), idx, parity);

      // multiply by sigma_mu_nu
      ColorSpinor<real, 3, 4> C = A.sigma(nu, mu);
      result += arg.coeff[i][parity] * outerProdSpinTrace(C, B);
    }

    result -= conj(result);

    Matrix<Complex, 3> temp;
    arg.oprod.load(reinterpret_cast<real *>(temp.data), idx, (mu - 1) * mu / 2 + nu, parity);
    temp = result + temp;
    arg.oprod.save(reinterpret_cast<real *>(temp.data), idx, (mu - 1) * mu / 2 + nu, parity);
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
