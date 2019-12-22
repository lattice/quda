#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_, int wflowDim_> struct GaugeWFlowArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int wflowDim = wflowDim_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;

    Gauge out;
    Gauge temp;
    Gauge in;

    int threads; // number of active threads required
    int X[4];    // grid dimensions
    int border[4];
    const Float epsilon;    
    
    GaugeWFlowArg(GaugeField &out, GaugeField &temp, GaugeField &in, Float epsilon) :
      out(out),
      in(in),
      temp(temp),
      threads(1),
      epsilon(epsilon)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = in.R()[dir];
        X[dir] = in.X()[dir] - border[dir] * 2;
        threads *= X[dir];
      }
      threads /= 2;
    }
  };

  // Wilson Flow as defined in https://arxiv.org/abs/1006.4518v3 
  template <typename Arg> __global__ void computeWFlowStepW1(Arg arg)
  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    int dir = threadIdx.z + blockIdx.z * blockDim.z;
    if (idx >= arg.threads) return;
    if (dir >= Arg::wflowDim) return;
    using real = typename Arg::Float;
    typedef complex<real> Complex;
    typedef Matrix<complex<real>, Arg::nColor> Link;

    int X[4];
    for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords(x, idx, X, parity);
    for (int dr = 0; dr < 4; ++dr) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }

    int dx[4] = {0, 0, 0, 0};
    // Only spatial dimensions are flowed
    {
      Link U, UDag, Stap, Omega, OmegaDiff, ODT, Q, exp_iQ;
      Complex OmegaDiffTr;
      Complex i_2(0, 0.5);

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, idx, parity, dir, Stap);

      // Store Staple in temp
      arg.temp(dir, linkIndexShift(x, dx, X), parity) = Stap;
      
      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);

      // Get U^{\dagger}
      UDag = inverse(U);

      // Compute \Omega = \rho * S * U^{\dagger}
      Omega = (arg.epsilon / 4.0) * Stap * UDag;

      // Compute anti-hermitian part, exponentiate, update U
      anti_herm_part(Omega, &Q);      
      exponentiate_iQ(Q, &exp_iQ);
      U = exp_iQ * U;
      arg.out(dir, linkIndexShift(x, dx, X), parity) = U;
    }
  }

  // Wilson Flow as defined in https://arxiv.org/abs/1006.4518v3 
  template <typename Arg> __global__ void computeWFlowStepW2(Arg arg)
  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    int dir = threadIdx.z + blockIdx.z * blockDim.z;
    if (idx >= arg.threads) return;
    if (dir >= Arg::wflowDim) return;
    using real = typename Arg::Float;
    typedef complex<real> Complex;
    typedef Matrix<complex<real>, Arg::nColor> Link;

    int X[4];
    for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords(x, idx, X, parity);
    for (int dr = 0; dr < 4; ++dr) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }

    int dx[4] = {0, 0, 0, 0};
    // Only spatial dimensions are flowed
    {
      /*
      Link U, UDag, Stap, temp1, temp2, ODT, Q, exp_iQ;
      Complex OmegaDiffTr;
      Complex i_2(0, 0.5);

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, idx, parity, dir, Stap);

      // Get updated U stored in arg.out
      U = arg.out(dir, linkIndexShift(x, dx, X), parity);

      // Get U^{\dagger}
      UDag = inverse(U);

      // Compute \Omega = c * S * U^{\dagger}
      temp1 = Stap * UDag;
      temp2 = (8.0 / 9.0) * temp1;

      // Use staple computed from W1 step
      temp1 = arg.temp(dir, linkIndexShift(x, dx, X), parity);
      temp1 *= (-17.0 / 36.0);
      
      // Save for Vt step
      arg.temp(dir, linkIndexShift(x, dx, X), parity) = temp1 + temp2;

      // Get traceless anti-Hermitian part of temp1 + temp2;
      
      
      //exponentiate_iQ((temp1 + temp2), &exp_iQ);

      //temp1 = exp_iQ * arg.epsilon;


      
      arg.out(dir, linkIndexShift(x, dx, X), parity) = temp1;
      */
    }
  }

  // Wilson Flow as defined in https://arxiv.org/abs/1006.4518v3 
  template <typename Arg> __global__ void computeWFlowStepVt(Arg arg)
  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    int dir = threadIdx.z + blockIdx.z * blockDim.z;
    if (idx >= arg.threads) return;
    if (dir >= Arg::wflowDim) return;
    using real = typename Arg::Float;
    typedef complex<real> Complex;
    typedef Matrix<complex<real>, Arg::nColor> Link;

    int X[4];
    for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords(x, idx, X, parity);
    for (int dr = 0; dr < 4; ++dr) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }

    int dx[4] = {0, 0, 0, 0};
    // Only spatial dimensions are flowed
    {
    }
  }  
} // namespace quda
