#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>
#include <kernels/gauge_utils.cuh>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_, int stoutDim_> struct GaugeSTOUTArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int stoutDim = stoutDim_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;

    Gauge out;
    const Gauge in;

    int threads; // number of active threads required
    int X[4];    // grid dimensions
    int border[4];
    const Float rho;
    const Float epsilon;

    GaugeSTOUTArg(GaugeField &out, const GaugeField &in, Float rho, Float epsilon=0) :
      out(out),
      in(in),
      threads(1),
      rho(rho),
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

  template <typename Arg> __global__ void computeSTOUTStep(Arg arg)
  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    int dir = threadIdx.z + blockIdx.z * blockDim.z;
    if (idx >= arg.threads) return;
    if (dir >= Arg::stoutDim) return;
    using real = typename Arg::Float;
    typedef complex<real> Complex;
    typedef Matrix<complex<real>, Arg::nColor> Link;

    // Compute spacetime and local coords
    int X[4];
    for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
    int x[4];
    getCoords(x, idx, X, parity);
    for (int dr = 0; dr < 4; ++dr) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }

    int dx[4] = {0, 0, 0, 0};
    Link U, Stap, Omega, OmegaDiff, Q;
    Complex i_2(0, 0.5);

    // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
    computeStaple(arg, x, X, parity, dir, Stap, Arg::stoutDim);
    
    // Get link U
    U = arg.in(dir, linkIndexShift(x, dx, X), parity);
    
    // Compute Omega_{mu}=[Sum_{mu neq nu}rho_{mu,nu}C_{mu,nu}]*U_{mu}^dag
    //--------------------------------------------------------------------
    // Compute \Omega = \rho * S * U^{\dagger}
    Q = (arg.rho * Stap) * conj(U);    
    // Compute \Q_{mu} = i/2[Omega_{mu}^dag - Omega_{mu}
    //                      - 1/3 Tr(Omega_{mu}^dag - Omega_{mu})]
    makeHerm(Q);
    // Q is now defined.

    Link exp_iQ = exponentiate_iQ(Q);
    U = exp_iQ * U;
    arg.out(dir, linkIndexShift(x, dx, X), parity) = U;

    // Debug tools
#if 0
    //Test for Tracless:
    double error = getTrace(Q).real();
    printf("Trace test %d %d %.15e\n", idx, dir, error);    
    //Test for hermiticity:
    Link Q_diff = conj(Q) - Q; //This should be the zero matrix. Test by ReTr(Q_diff^2);
    Q_diff *= Q_diff;
    error = getTrace(Q_diff).real();
    printf("Herm test %d %d %.15e\n", idx, dir, error);
    //Test for expiQ unitarity:
    error = ErrorSU3(exp_iQ);
    printf("expiQ test %d %d %.15e\n", idx, dir, error);
    //Test for expiQ*U unitarity:
    error = ErrorSU3(U);
    printf("expiQ*u test %d %d %.15e\n", idx, dir, error);    
#endif
  }
  
  
  //------------------------//
  // Over-Improved routines //
  //------------------------//
  template <typename Arg> __global__ void computeOvrImpSTOUTStep(Arg arg)
  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    int dir = threadIdx.z + blockIdx.z * blockDim.z;
    if (idx >= arg.threads) return;
    if (dir >= Arg::stoutDim) return;

    using real = typename Arg::Float;
    typedef complex<real> Complex;
    typedef Matrix<complex<real>, Arg::nColor> Link;

    // Compute spacetime and local coords
    int X[4];
    for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
    int x[4];
    getCoords(x, idx, X, parity);
    for (int dr = 0; dr < 4; ++dr) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }

    double staple_coeff = (5.0 - 2.0 * arg.epsilon) / 3.0;
    double rectangle_coeff = (1.0 - arg.epsilon) / 12.0;

    int dx[4] = {0, 0, 0, 0};
    Link U, UDag, Stap, Rect, Omega, OmegaDiff, ODT, Q;
    Complex OmegaDiffTr;
    Complex i_2(0, 0.5);
    
    // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
    // and the 1x2 and 2x1 rectangles of length 5. From the following paper:
    // https://arxiv.org/abs/0801.1165
    computeStapleRectangle(arg, x, X, parity, dir, Stap, Rect, Arg::stoutDim);
    
    // Get link U
    U = arg.in(dir, linkIndexShift(x, dx, X), parity);

    // Compute Omega_{mu}=[Sum_{mu neq nu}rho_{mu,nu}C_{mu,nu}]*U_{mu}^dag
    //-------------------------------------------------------------------
    // Compute \rho * staple_coeff * S - \rho * rectangle_coeff * R
    Q = ((arg.rho * staple_coeff) * (Stap) - (arg.rho * rectangle_coeff) * (Rect)) * conj(U);
    // Compute \Q_{mu} = i/2[Omega_{mu}^dag - Omega_{mu}
    //                      - 1/3 Tr(Omega_{mu}^dag - Omega_{mu})]
    makeHerm(Q);
    // Q is now defined.

    Link exp_iQ = exponentiate_iQ(Q);
    U = exp_iQ * U;
    arg.out(dir, linkIndexShift(x, dx, X), parity) = U;

    // Debug tools
#if 0
    //Test for Tracless:
    double error = getTrace(Q).real();
    printf("Trace test %d %d %.15e\n", idx, dir, error);    
    //Test for hermiticity:
    Link Q_diff = conj(Q) - Q; //This should be the zero matrix. Test by ReTr(Q_diff^2);
    Q_diff *= Q_diff;
    error = getTrace(Q_diff).real();
    printf("Herm test %d %d %.15e\n", idx, dir, error);
    //Test for expiQ unitarity:
    error = ErrorSU3(exp_iQ);
    printf("expiQ test %d %d %.15e\n", idx, dir, error);
    //Test for expiQ*U unitarity:
    error = ErrorSU3(U);
    printf("expiQ*u test %d %d %.15e\n", idx, dir, error);    
#endif
  }
} // namespace quda
