#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_> struct GaugeWFlowArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
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
      Link U, Stap, temp1, Q, exp_iQ;
            
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, idx, parity, dir, Stap, 3);

      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);
      
      // Compute Z0, store in temp
      temp1 = Stap * conj(U);
      arg.temp(dir, linkIndexShift(x, dx, X), parity) = temp1;
      
      // Compute hermitian projection of temp1, exponentiate, update U
      herm_proj(temp1, &Q);
      Q *= (1.0 / 4.0) * arg.epsilon;
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
      Link U, Stap, temp1, temp2, Q, exp_iQ;
            
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, idx, parity, dir, Stap);

      // Get updated U 
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);

      // Compute Z1, store (8/9 Z1 - 17/36 Z0) in temp
      temp1 = (8.0 / 9.0) * Stap * conj(U);
      // Z0 stored in temp
      temp2 = arg.temp(dir, linkIndexShift(x, dx, X), parity);
      temp2 *= (17.0 / 36.0);
      temp1 -= temp2;
      arg.temp(dir, linkIndexShift(x, dx, X), parity) = temp1;
      //temp1 *= arg.epsilon;
      
      // Compute hermitian projection of temp1, exponentiate, update U
      herm_proj(temp1, &Q);
      Q *= arg.epsilon;
      exponentiate_iQ(Q, &exp_iQ);
      U = exp_iQ * U;
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
      Link U, Stap, temp1, temp2, Q, exp_iQ;
            
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, idx, parity, dir, Stap);

      // Get updated U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);

      // Compute Z2, construct (3/4 Z2 - 8/9 Z1 + 17/36 Z0)
      temp1 = (3.0 / 4.0) * Stap * conj(U);
      // Use (8/9 Z1 - 17/36 Z0) computed from W2 step
      temp2 = arg.temp(dir, linkIndexShift(x, dx, X), parity);
      temp1 -= temp2;
      //temp1 *= arg.epsilon;
      
      // Compute hermitian proj of temp1, exponentiate, update U
      herm_proj(temp1, &Q);
      Q *= arg.epsilon;
      exponentiate_iQ(Q, &exp_iQ);
      U = exp_iQ * U;
      arg.out(dir, linkIndexShift(x, dx, X), parity) = U;      
    }
  }  
} // namespace quda
