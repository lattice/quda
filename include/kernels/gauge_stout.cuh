#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>

namespace quda
{

#define  DOUBLE_TOL	1e-15
#define  SINGLE_TOL	2e-6

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
    const Float tolerance;

    GaugeSTOUTArg(GaugeField &out, const GaugeField &in, Float rho, Float epsilon=0) :
      out(out),
      in(in),
      threads(1),
      rho(rho),
      epsilon(epsilon),
      tolerance(in.Precision() == QUDA_DOUBLE_PRECISION ? DOUBLE_TOL : SINGLE_TOL)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = in.R()[dir];
        X[dir] = in.X()[dir] - border[dir] * 2;
        threads *= X[dir];
      }
      threads /= 2;
    }
  };

  template <typename Arg, typename Link>
  __host__ __device__ void computeStaple(Arg &arg, int idx, int parity, int dir, Link &staple)
  {
    // compute spacetime dimensions and parity
    int X[4];
    for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords(x, idx, X, parity);
    for (int dr = 0; dr < 4; ++dr) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }

    setZero(&staple);

    // I believe most users won't want to include time staples in smearing
    for (int mu = 0; mu < 3; mu++) {

      // identify directions orthogonal to the link.
      if (mu != dir) {

        int nu = dir;
        {
          int dx[4] = {0, 0, 0, 0};
          Link U1, U2, U3;

          // Get link U_{\mu}(x)
          U1 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          dx[mu]++;
          // Get link U_{\nu}(x+\mu)
          U2 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          dx[mu]--;
          dx[nu]++;
          // Get link U_{\mu}(x+\nu)
          U3 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          // staple += U_{\mu}(x) * U_{\nu}(x+\mu) * U^\dag_{\mu}(x+\nu)
          staple = staple + U1 * U2 * conj(U3);

          dx[mu]--;
          dx[nu]--;
          // Get link U_{\mu}(x-\mu)
          U1 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);
          // Get link U_{\nu}(x-\mu)
          U2 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          dx[nu]++;
          // Get link U_{\mu}(x-\mu+\nu)
          U3 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          // staple += U^\dag_{\mu}(x-\mu) * U_{\nu}(x-\mu) * U_{\mu}(x-\mu+\nu)
          staple = staple + conj(U1) * U2 * U3;
        }
      }
    }
  }

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

    int X[4];
    for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords(x, idx, X, parity);
    for (int dr = 0; dr < 4; ++dr) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }

    int dx[4] = {0, 0, 0, 0};
    // Only spatial dimensions are smeared
    {
      Link U, UDag, Stap, Omega, OmegaDiff, ODT, Q, exp_iQ;
      Complex OmegaDiffTr;
      Complex i_2(0, 0.5);

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, idx, parity, dir, Stap);
      //
      // |- > -|                /- > -/                /- > -
      // ^     v               ^     v                ^
      // |     |              /     /                /- < -
      //         + |     |  +         +  /     /  +         +  - > -/
      //           v     ^              v     ^                    v
      //           |- > -|             /- > -/                - < -/

      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);

      // Compute Omega_{mu}=[Sum_{mu neq nu}rho_{mu,nu}C_{mu,nu}]*U_{mu}^dag

      // Get U^{\dagger}
      UDag = inverse(U);

      // Compute \Omega = \rho * S * U^{\dagger}
      Omega = (arg.rho * Stap) * UDag;

      // Compute \Q_{mu} = i/2[Omega_{mu}^dag - Omega_{mu}
      //                      - 1/3 Tr(Omega_{mu}^dag - Omega_{mu})]

      OmegaDiff = conj(Omega) - Omega;

      Q = OmegaDiff;
      OmegaDiffTr = getTrace(OmegaDiff);
      OmegaDiffTr = (1.0 / 3.0) * OmegaDiffTr;

      // Matrix proportional to OmegaDiffTr
      setIdentity(&ODT);

      Q = Q - OmegaDiffTr * ODT;
      Q = i_2 * Q;
      // Q is now defined.

#if 0
      //Test for Tracless:
      //reuse OmegaDiffTr
      OmegaDiffTr = getTrace(Q);
      double error;
      error = OmegaDiffTr.real();
      printf("Trace test %d %d %.15e\n", idx, dir, error);

      //Test for hemiticity:
      Link Q_diff = conj(Q);
      Q_diff -= Q; //This should be the zero matrix. Test by ReTr(Q_diff^2);
      Q_diff *= Q_diff;
      //reuse OmegaDiffTr
      OmegaDiffTr = getTrace(Q_diff);
      error = OmegaDiffTr.real();
      printf("Herm test %d %d %.15e\n", idx, dir, error);
#endif

      exponentiate_iQ(Q, &exp_iQ);

#if 0
      //Test for expiQ unitarity:
      error = ErrorSU3(exp_iQ);
      printf("expiQ test %d %d %.15e\n", idx, dir, error);
#endif

      U = exp_iQ * U;
#if 0
      //Test for expiQ*U unitarity:
      error = ErrorSU3(U);
      printf("expiQ*u test %d %d %.15e\n", idx, dir, error);
#endif

      arg.out(dir, linkIndexShift(x, dx, X), parity) = U;
    }
  }

  //------------------------//
  // Over-Improved routines //
  //------------------------//
  template <typename Arg, typename Link>
  __host__ __device__ void computeStapleRectangle(Arg &arg, int idx, int parity, int dir, Link &staple, Link &rectangle)
  {
    // compute spacetime dimensions and parity
    int X[4];
    for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords(x, idx, X, parity);
    for (int dr = 0; dr < 4; ++dr) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }

    setZero(&staple);
    setZero(&rectangle);

    // Over-Improved stout is usually done for topological
    // measuremnts, so we include the temporal direction.
    for (int mu = 0; mu < 4; mu++) {

      // identify directions orthogonal to the link.
      if (mu != dir) {
        int nu = dir;

        // RECTANGLE calculation
        // This is done in three parts. For some link U_nu(x) there are
        // 1x2 rectangles (R12) and two sets of 2x1 rectangles, defined as
        // 'forward' (R21f)  and 'backward' (R21b).

        // STAPLE calculation
        // This is done part way through the computation of (R21f) as the
        // First two links of the staple are already in memory.

        // Memory usage and communications.
        // There are 10 unique links to be fetched per direction. 3 of these
        // links (the ones that form the simple staple) can be recycled on
        // the fly. The two links immediately succeeding and preceding
        // U_nu(x) in the nu directon are also reused when changing from
        // +ve to -ve mu.

        {
          int dx[4] = {0, 0, 0, 0};
          Link U1, U2, U3, U4, U5, U6, U7;

          //--------//
          // +ve mu //
          //--------//

          //----------------------------------------------------------------
          // R12 = U_mu(x)*U_mu(x+mu)*U_nu(x+2mu)*U^d_mu(x+nu+mu)*U^d_mu(x+nu)
          // Get link U_mu(x)
          U1 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          dx[mu]++;
          // Get link U_mu(x+mu)
          U2 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          dx[mu]++;
          // Get link U_nu(x+2mu)
          U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

          dx[mu]--;
          dx[nu]++;
          // Get link U_mu(x+nu+mu)
          U4 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          dx[mu]--;
          // Get link U_mu(x+nu)
          U5 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          rectangle = rectangle + U1 * U2 * U3 * conj(U4) * conj(U5);
          //---------------------------------------------------------------

          // reset dx
          dx[nu]--;
          //---------------------------------------------------------------
          // R21f=U_mu(x)*U_nu(x+mu)*U_nu(x+nu+mu)*U^d_mu(x+2nu)*U^d_nu(x+nu)
          // Get link U_mu(x)
          // Same as U1 from R12

          dx[mu]++;
          // Get link U_nu(x+mu)
          U2 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          ///////////////////////////////////////////////////////
          // Here we get the third link in the staple and compute.
          // Get U_mu(x+nu)
          // Same as U5 from R12
          staple = staple + U1 * U2 * conj(U5);
          ///////////////////////////////////////////////////////

          dx[nu]++;
          // Get link U_nu(x+nu+mu)
          U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

          dx[mu]--;
          dx[nu]++;
          // Get link U_mu(x+2nu)
          U4 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          dx[nu]--;
          // Get link U_nu(x+nu)
          U6 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          rectangle = rectangle + U1 * U2 * U3 * conj(U4) * conj(U6);
          //---------------------------------------------------------------

          // reset dx
          dx[nu]--;
          //---------------------------------------------------------------
          // R21b=U^d_nu(x-nu)*U_mu(x-nu)*U_nu(x+nu+mu)*U^d_mu(x+2nu)*U^dag_nu(x+nu)

          // Get link U_nu(x-nu)
          dx[nu]--;
          U7 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          // Get link U_mu(x-nu)
          U4 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          // Get link U_nu(x-nu+mu)
          dx[mu]++;
          U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

          // Get link U_nu(x+mu)
          // Same as U2 from R21f

          // Get link U_mu(x+nu)
          // Same as U5 from R12

          rectangle = rectangle + conj(U7) * U4 * U3 * U2 * conj(U5);
          //---------------------------------------------------------------

          //--------//
          // -ve mu //
          //--------//

          // reset dx
          dx[mu]--;
          dx[nu]++;
          //---------------------------------------------------------------
          // R12 = U^dag_mu(x-mu) * U^dag_mu(x-2mu) * U_nu(x-2mu) * U_mu(x-2mu+nu) * U_mu(x-mu+nu)

          dx[mu]--;
          // Get link U_mu(x-mu)
          U1 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          dx[mu]--;
          // Get link U_mu(x-2mu)
          U2 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          // Get link U_nu(x-2mu)
          U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

          dx[nu]++;
          // Get link U_mu(x-2mu+nu)
          U4 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          dx[mu]++;
          // Get link U_mu(x-mu+nu)
          U5 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          rectangle = rectangle + conj(U1) * conj(U2) * U3 * U4 * U5;
          //---------------------------------------------------------------

          // reset dx
          dx[mu]++;
          dx[nu]--;
          //---------------------------------------------------------------
          // R21f = U^dag_mu(x-mu) * U_nu(x-mu) * U_nu(x-mu+nu) * U_mu(x-mu+2nu) * U^dag_nu(x+nu)

          // Get link U_mu(x-mu)
          // Same as U1 from R12

          dx[mu]--;
          // Get link U_nu(x-mu)
          U2 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          ///////////////////////////////////////////////////////
          // Here we get the third link in the staple and compute.
          // Get U_mu(x-mu+nu)
          // Same as U5 from R12
          staple = staple + conj(U1) * U2 * U5;
          ///////////////////////////////////////////////////////

          dx[nu]++;
          // Get link U_nu(x-mu+nu)
          U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

          dx[nu]++;
          // Get link U_mu(x-mu+2nu)
          U4 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          // Get link U_nu(x+nu)
          // Same as U6 from +ve R21f

          rectangle = rectangle + conj(U1) * U2 * U3 * U4 * conj(U6);
          //---------------------------------------------------------------

          // reset dx
          dx[nu]--;
          dx[nu]--;
          dx[mu]++;
          //---------------------------------------------------------------
          // R21b= U^dag_nu(x-nu) * U^dag_mu(x-mu-nu) * U_nu(x-mu-nu) * U_nu(x-mu) * U_mu(x-mu+nu)

          // Get link U_nu(x-nu)
          // Same as U7 from +ve R21b

          // Get link U_mu(x-mu-nu)
          dx[nu]--;
          dx[mu]--;
          U4 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          // Get link U_nu(x-nu-mu)
          U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

          // Get link U_nu(x-mu)
          // Same as U2 from R21f

          // Get link U_mu(x-mu+nu)
          // Same as U5 from R12

          rectangle = rectangle + conj(U7) * conj(U4) * U3 * U2 * U5;
          //---------------------------------------------------------------
        }
      }
    }
  }

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
    // All dimensions are smeared
    {
      Link U, UDag, Stap, Rect, Omega, OmegaDiff, ODT, Q, exp_iQ;
      Complex OmegaDiffTr;
      Complex i_2(0, 0.5);

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      // and the 1x2 and 2x1 rectangles of length 5. From the following paper:
      // https://arxiv.org/abs/0801.1165
      computeStapleRectangle(arg, idx, parity, dir, Stap, Rect);

      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);

      // Compute Omega_{mu}=[Sum_{mu neq nu}rho_{mu,nu}C_{mu,nu}]*U_{mu}^dag
      //-------------------------------------------------------------------

      // Get U^{\dagger}
      UDag = inverse(U);

      // Compute \rho * staple_coeff * S
      Omega = (arg.rho * staple_coeff) * (Stap);

      // Compute \rho * rectangle_coeff * R
      Omega = Omega - (arg.rho * rectangle_coeff) * (Rect);
      Omega = Omega * UDag;

      // Compute \Q_{mu} = i/2[Omega_{mu}^dag - Omega_{mu}
      //                      - 1/3 Tr(Omega_{mu}^dag - Omega_{mu})]

      OmegaDiff = conj(Omega) - Omega;

      Q = OmegaDiff;
      OmegaDiffTr = getTrace(OmegaDiff);
      OmegaDiffTr = (1.0 / 3.0) * OmegaDiffTr;

      // Matrix proportional to OmegaDiffTr
      setIdentity(&ODT);

      Q = Q - OmegaDiffTr * ODT;
      Q = i_2 * Q;
      // Q is now defined.

#if 0
      //Test for Tracless:
      //reuse OmegaDiffTr
      OmegaDiffTr = getTrace(Q);
      double error;
      error = OmegaDiffTr.real();
      printf("Trace test %d %d %.15e\n", idx, dir, error);

      //Test for hemiticity:
      Link Q_diff = conj(Q);
      Q_diff -= Q; //This should be the zero matrix. Test by ReTr(Q_diff^2);
      Q_diff *= Q_diff;
      //reuse OmegaDiffTr
      OmegaDiffTr = getTrace(Q_diff);
      error = OmegaDiffTr.real();
      printf("Herm test %d %d %.15e\n", idx, dir, error);
#endif

      exponentiate_iQ(Q, &exp_iQ);

#if 0
      //Test for expiQ unitarity:
      error = ErrorSU3(exp_iQ);
      printf("expiQ test %d %d %.15e\n", idx, dir, error);
#endif

      U = exp_iQ * U;
#if 0
      //Test for expiQ*U unitarity:
      error = ErrorSU3(U);
      printf("expiQ*u test %d %d %.15e\n", idx, dir, error);
#endif

      arg.out(dir, linkIndexShift(x, dx, X), parity) = U;
    }
  }

} // namespace quda
