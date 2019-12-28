#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_, int wFlowDim_> struct GaugeWFlowArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int wFlowDim = wFlowDim_;
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
    if (dir >= Arg::wFlowDim) return;
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
    {
      Link U, Stap, Z0, exp_Z0;
      Complex im(0.0,-1.0);
            
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, idx, parity, dir, Stap, Arg::wFlowDim);

      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);
      
      // Compute Z0, store in temp
      Z0 = Stap * conj(U);
      arg.temp(dir, linkIndexShift(x, dx, X), parity) = Z0;
      Z0 *= (1.0 / 4.0) * arg.epsilon;
      
      // Compute anti-hermitian projection of Z0, exponentiate, update U
      makeAntiHerm(Z0);
      //exponentiate_iQ assumes hermitian Q, rescale Z by -i
      Z0 = im * Z0; 
      exponentiate_iQ(Z0, &exp_Z0);
      
      U = exp_Z0 * U;
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
    if (dir >= Arg::wFlowDim) return;
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
    {
      Link U, Stap, Z1, Z0, exp_Z1;
      Complex im(0.0,-1.0);
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, idx, parity, dir, Stap, Arg::wFlowDim);

      // Get updated U 
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);

      // Compute Z1.
      Z1 = (8.0 / 9.0) * Stap * conj(U);
      // Retrieve Z0, (8/9 Z1 - 17/36 Z0) stored in temp
      Z0 = arg.temp(dir, linkIndexShift(x, dx, X), parity);
      Z0 *= (17.0 / 36.0);
      Z1 = Z1 - Z0;
      arg.temp(dir, linkIndexShift(x, dx, X), parity) = Z1;
      Z1 *= arg.epsilon;
      
      // Compute anti-hermitian projection of Z1, exponentiate, update U
      makeAntiHerm(Z1);
      //exponentiate_iQ assumes hermitian Q, rescale Z by -i
      Z1 = im * Z1; 
      exponentiate_iQ(Z1, &exp_Z1);
      
      U = exp_Z1 * U;
      arg.out(dir, linkIndexShift(x, dx, X), parity) = U;
    }
  }

  // Wilson Flow as defined in https://arxiv.org/abs/1006.4518v3 
  template <typename Arg> __global__ void computeWFlowStepVt(Arg arg)
  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    int dir = threadIdx.z + blockIdx.z * blockDim.z;
    if (idx >= arg.threads) return;
    if (dir >= Arg::wFlowDim) return;
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
     {
      Link U, Stap, Z2, Z1, exp_Z2;
      Complex im(0.0,-1.0);
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, idx, parity, dir, Stap, Arg::wFlowDim);

      // Get updated U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);

      // Compute Z2, construct (3/4 Z2 - 8/9 Z1 + 17/36 Z0)
      Z2 = (3.0 / 4.0) * Stap * conj(U);
      // Use (8/9 Z1 - 17/36 Z0) computed from W2 step
      Z1 = arg.temp(dir, linkIndexShift(x, dx, X), parity);
      Z2 = Z2 - Z1;
      Z2 *= arg.epsilon;

      // Compute anti-hermitian projection of Z2, exponentiate, update U
      makeAntiHerm(Z2);
      //exponentiate_iQ assumes hermitian Q, rescale Z by -i
      Z2 = im * Z2; 
      exponentiate_iQ(Z2, &exp_Z2);
      
      U = exp_Z2 * U;
      arg.out(dir, linkIndexShift(x, dx, X), parity) = U;      
    }
  }  
} // namespace quda
