#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>

namespace quda
{

#define  DOUBLE_TOL	1e-15
#define  SINGLE_TOL	2e-6

  template <typename Float_, int nColor_, QudaReconstructType recon_> struct GaugeAPEArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;

    Gauge out;
    const Gauge in;

    int threads; // number of active threads required
    int X[4];    // grid dimensions
    int border[4];
    const Float alpha;
    const Float tolerance;

    GaugeAPEArg(GaugeField &out, const GaugeField &in, double alpha) :
      out(out),
      in(in),
      threads(1),
      alpha(alpha),
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

  template <typename Arg> __global__ void computeAPEStep(Arg arg)
  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    int dir = threadIdx.z + blockIdx.z * blockDim.z;
    if (idx >= arg.threads) return;
    if (dir >= 3) return;
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
      Link U, S, TestU, I;
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, idx, parity, dir, S);
      //
      // |- > -|                /- > -/                /- > -
      // ^     v               ^     v                ^
      // |     |              /     /                /- < -
      //         + |     |  +         +  /     /  +         +  - > -/
      //           v     ^              v     ^                    v
      //           |- > -|             /- > -/                - < -/

      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);

      S = S * (arg.alpha / ((real)(2. * (3. - 1.))));
      setIdentity(&I);

      TestU = I * (1. - arg.alpha) + S * conj(U);
      polarSu3<real>(TestU, arg.tolerance);
      U = TestU * U;

      arg.out(dir, linkIndexShift(x, dx, X), parity) = U;
    }
  }

} // namespace quda
