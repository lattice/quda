#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_> struct FmunuArg
  {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float,recon>::type G;
    typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type F;

    G u;
    F f;

    int threads; // number of active threads required
    int X[4];    // grid dimensions
    int border[4];

    FmunuArg(GaugeField &f, const GaugeField &u) :
      threads(f.VolumeCB()),
      f(f),
      u(u)
    {
      for (int dir = 0; dir < 4; ++dir) {
        X[dir] = f.X()[dir];
        border[dir] = (u.X()[dir] - X[dir]) / 2;
      }
    }
  };

  template <int mu, int nu, typename Arg>
  __device__ __forceinline__ void computeFmunuCore(Arg &arg, int idx, int parity)
  {
    typedef Matrix<complex<typename Arg::Float>, 3> Link;

    int x[4];
    auto &X = arg.X;

    getCoords(x, idx, X, parity);
    for (int dir = 0; dir < 4; ++dir) {
      x[dir] += arg.border[dir];
      X[dir] += 2 * arg.border[dir];
    }

    Link F;
    { // U(x,mu) U(x+mu,nu) U[dagger](x+nu,mu) U[dagger](x,nu)

      // load U(x)_(+mu)
      int dx[4] = {0, 0, 0, 0};
      Link U1 = arg.u(mu, linkIndexShift(x, dx, X), parity);

      // load U(x+mu)_(+nu)
      dx[mu]++;
      Link U2 = arg.u(nu, linkIndexShift(x, dx, X), 1 - parity);
      dx[mu]--;

      // load U(x+nu)_(+mu)
      dx[nu]++;
      Link U3 = arg.u(mu, linkIndexShift(x, dx, X), 1 - parity);
      dx[nu]--;

      // load U(x)_(+nu)
      Link U4 = arg.u(nu, linkIndexShift(x, dx, X), parity);

      // compute plaquette
      F = U1 * U2 * conj(U3) * conj(U4);
    }

    { // U(x,nu) U[dagger](x+nu-mu,mu) U[dagger](x-mu,nu) U(x-mu, mu)

      // load U(x)_(+nu)
      int dx[4] = {0, 0, 0, 0};
      Link U1 = arg.u(nu, linkIndexShift(x, dx, X), parity);

      // load U(x+nu)_(-mu) = U(x+nu-mu)_(+mu)
      dx[nu]++;
      dx[mu]--;
      Link U2 = arg.u(mu, linkIndexShift(x, dx, X), parity);
      dx[mu]++;
      dx[nu]--;

      // load U(x-mu)_nu
      dx[mu]--;
      Link U3 = arg.u(nu, linkIndexShift(x, dx, X), 1 - parity);
      dx[mu]++;

      // load U(x)_(-mu) = U(x-mu)_(+mu)
      dx[mu]--;
      Link U4 = arg.u(mu, linkIndexShift(x, dx, X), 1 - parity);
      dx[mu]++;

      // sum this contribution to Fmunu
      F += U1 * conj(U2) * conj(U3) * U4;
    }

    { // U[dagger](x-nu,nu) U(x-nu,mu) U(x+mu-nu,nu) U[dagger](x,mu)

      // load U(x)_(-nu)
      int dx[4] = {0, 0, 0, 0};
      dx[nu]--;
      Link U1 = arg.u(nu, linkIndexShift(x, dx, X), 1 - parity);
      dx[nu]++;

      // load U(x-nu)_(+mu)
      dx[nu]--;
      Link U2 = arg.u(mu, linkIndexShift(x, dx, X), 1 - parity);
      dx[nu]++;

      // load U(x+mu-nu)_(+nu)
      dx[mu]++;
      dx[nu]--;
      Link U3 = arg.u(nu, linkIndexShift(x, dx, X), parity);
      dx[nu]++;
      dx[mu]--;

      // load U(x)_(+mu)
      Link U4 = arg.u(mu, linkIndexShift(x, dx, X), parity);

      // sum this contribution to Fmunu
      F += conj(U1) * U2 * U3 * conj(U4);
    }

    { // U[dagger](x-mu,mu) U[dagger](x-mu-nu,nu) U(x-mu-nu,mu) U(x-nu,nu)

      // load U(x)_(-mu)
      int dx[4] = {0, 0, 0, 0};
      dx[mu]--;
      Link U1 = arg.u(mu, linkIndexShift(x, dx, X), 1 - parity);
      dx[mu]++;

      // load U(x-mu)_(-nu) = U(x-mu-nu)_(+nu)
      dx[mu]--;
      dx[nu]--;
      Link U2 = arg.u(nu, linkIndexShift(x, dx, X), parity);
      dx[nu]++;
      dx[mu]++;

      // load U(x-nu)_mu
      dx[mu]--;
      dx[nu]--;
      Link U3 = arg.u(mu, linkIndexShift(x, dx, X), parity);
      dx[nu]++;
      dx[mu]++;

      // load U(x)_(-nu) = U(x-nu)_(+nu)
      dx[nu]--;
      Link U4 = arg.u(nu, linkIndexShift(x, dx, X), 1 - parity);
      dx[nu]++;

      // sum this contribution to Fmunu
      F += conj(U1) * conj(U2) * U3 * U4;
    }
    // 3 matrix additions, 12 matrix-matrix multiplications, 8 matrix conjugations
    // Each matrix conjugation involves 9 unary minus operations but these ar not included in the operation count
    // Each matrix addition involves 18 real additions
    // Each matrix-matrix multiplication involves 9*3 complex multiplications and 9*2 complex additions
    // = 9*3*6 + 9*2*2 = 198 floating-point ops
    // => Total number of floating point ops per site above is
    // 3*18 + 12*198 =  54 + 2376 = 2430
    {
      F -= conj(F);                   // 18 real subtractions + one matrix conjugation
      F *= static_cast<typename Arg::Float>(0.125); // 18 real multiplications
                                      // 36 floating point operations here
    }

    constexpr int munu_idx = (mu * (mu - 1)) / 2 + nu; // lower-triangular indexing
    arg.f(munu_idx, idx, parity) = F;
  }

  template <typename Arg> __global__ void computeFmunuKernel(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    int mu_nu = threadIdx.z + blockIdx.z * blockDim.z;
    if (x_cb >= arg.threads) return;
    if (mu_nu >= 6) return;

    switch (mu_nu) { // F[1,0], F[2,0], F[2,1], F[3,0], F[3,1], F[3,2]
    case 0: computeFmunuCore<1, 0>(arg, x_cb, parity); break;
    case 1: computeFmunuCore<2, 0>(arg, x_cb, parity); break;
    case 2: computeFmunuCore<2, 1>(arg, x_cb, parity); break;
    case 3: computeFmunuCore<3, 0>(arg, x_cb, parity); break;
    case 4: computeFmunuCore<3, 1>(arg, x_cb, parity); break;
    case 5: computeFmunuCore<3, 2>(arg, x_cb, parity); break;
    }
  }

  template <typename Arg> void computeFmunuCPU(Arg &arg)
  {
    for (int parity = 0; parity < 2; parity++) {
      for (int x_cb = 0; x_cb < arg.threads; x_cb++) {
        for (int mu = 0; mu < 4; mu++) {
          for (int nu = 0; nu < mu; nu++) {
            int mu_nu = (mu * (mu - 1)) / 2 + nu;
            switch (mu_nu) { // F[1,0], F[2,0], F[2,1], F[3,0], F[3,1], F[3,2]
            case 0: computeFmunuCore<1, 0>(arg, x_cb, parity); break;
            case 1: computeFmunuCore<2, 0>(arg, x_cb, parity); break;
            case 2: computeFmunuCore<2, 1>(arg, x_cb, parity); break;
            case 3: computeFmunuCore<3, 0>(arg, x_cb, parity); break;
            case 4: computeFmunuCore<3, 1>(arg, x_cb, parity); break;
            case 5: computeFmunuCore<3, 2>(arg, x_cb, parity); break;
            }
          }
        }
      }
    }
  }

} // namespace quda
