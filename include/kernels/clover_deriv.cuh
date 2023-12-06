#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <thread_array.h>
#include <kernel.h>

namespace quda
{

  template <typename Float, int nColor_, QudaReconstructType recon> struct CloverDerivArg : kernel_param<> {
    static constexpr int nColor = nColor_;
    using Force = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
    using Oprod = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
    using Gauge = typename gauge_mapper<Float, recon>::type;
    using real = typename mapper<Float>::type;
    int X[4];
    int E[4];
    int border[4];
    real coeff;

    Force force;
    Gauge gauge;
    Oprod oprod;

    CloverDerivArg(const GaugeField &force, const GaugeField &gauge, const GaugeField &oprod, double coeff) :
      kernel_param(dim3(force.VolumeCB(), 2, 4)),
      coeff(coeff),
      force(force),
      gauge(gauge),
      oprod(oprod)
    {
      for (int dir = 0; dir < 4; ++dir) {
        X[dir] = force.X()[dir];
        E[dir] = oprod.X()[dir];
        border[dir] = (E[dir] - X[dir]) / 2;
      }
    }
  };

  template <typename Link, typename Force, typename Arg>
  __device__ __host__ void computeForce(Force &force_total, const Arg &arg, int xIndex, int parity, int mu, int nu)
  {
    const int otherparity = (1 - parity);
    const int tidx = mu > nu ? (mu - 1) * mu / 2 + nu : (nu - 1) * nu / 2 + mu;

    int x[4];
    getCoordsExtended(x, xIndex, arg.X, parity, arg.border);

    // U[mu](x) U[nu](x+mu) U[*mu](x+nu) U[*nu](x) Oprod(x)
    {
      thread_array<int, 4> d = { };

      // load U(x)_(+mu)
      Link U1 = arg.gauge(mu, linkIndexShift(x, d, arg.E), parity);

      // load U(x+mu)_(+nu)
      d[mu]++;
      Link U2 = arg.gauge(nu, linkIndexShift(x, d, arg.E), otherparity);
      d[mu]--;

      // load U(x+nu)_(+mu)
      d[nu]++;
      Link U3 = arg.gauge(mu, linkIndexShift(x, d, arg.E), otherparity);
      d[nu]--;

      // load U(x)_(+nu)
      Link U4 = arg.gauge(nu, linkIndexShift(x, d, arg.E), parity);

      // load Oprod
      Link Oprod1 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), parity);
      Link force = U1 * U2 * conj(U3) * conj(U4) * Oprod1;

      d[mu]++;
      d[nu]++;
      Link Oprod2 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), parity);
      force += U1 * U2 * Oprod2 * conj(U3) * conj(U4);

      if (nu < mu) force_total -= force;
      else force_total += force;
    }

    {
      thread_array<int, 4> d = { };

      // load U(x-nu)(+nu)
      d[nu]--;
      Link U1 = arg.gauge(nu, linkIndexShift(x, d, arg.E), otherparity);
      d[nu]++;

      // load U(x-nu)(+mu)
      d[nu]--;
      Link U2 = arg.gauge(mu, linkIndexShift(x, d, arg.E), otherparity);
      d[nu]++;

      // load U(x+mu-nu)(nu)
      d[mu]++;
      d[nu]--;
      Link U3 = arg.gauge(nu, linkIndexShift(x, d, arg.E), parity);
      d[mu]--;
      d[nu]++;

      // load U(x)_(+mu)
      Link U4 = arg.gauge(mu, linkIndexShift(x, d, arg.E), parity);

      d[mu]++;
      d[nu]--;
      Link Oprod1 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), parity);
      Link force = conj(U1) * U2 * Oprod1 * U3 * conj(U4);

      d[mu]--;
      d[nu]++;
      Link Oprod4 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), parity);
      force += Oprod4 * conj(U1) * U2 * U3 * conj(U4);

      if (nu < mu) force_total += force;
      else force_total -= force;
    }

    {
      thread_array<int, 4> d = { };

      // load U(x)_(+mu)
      Link U1 = arg.gauge(mu, linkIndexShift(x, d, arg.E), parity);

      // load U(x+mu)_(+nu)
      d[mu]++;
      Link U2 = arg.gauge(nu, linkIndexShift(x, d, arg.E), otherparity);
      d[mu]--;

      // load U(x+nu)_(+mu)
      d[nu]++;
      Link U3 = arg.gauge(mu, linkIndexShift(x, d, arg.E), otherparity);
      d[nu]--;

      // load U(x)_(+nu)
      Link U4 = arg.gauge(nu, linkIndexShift(x, d, arg.E), parity);

      // load opposite parity Oprod
      d[nu]++;
      Link Oprod3 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), otherparity);
      Link force = U1 * U2 * conj(U3) * Oprod3 * conj(U4);

      // load Oprod(x+mu)
      d[nu]--;
      d[mu]++;
      Link Oprod4 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), otherparity);
      force += U1 * Oprod4 * U2 * conj(U3) * conj(U4);

      if (nu < mu) force_total -= force;
      else force_total += force;
    }

    // Lower leaf
    // U[nu*](x-nu) U[mu](x-nu) U[nu](x+mu-nu) Oprod(x+mu) U[*mu](x)
    {
      thread_array<int, 4> d = { };

      // load U(x-nu)(+nu)
      d[nu]--;
      Link U1 = arg.gauge(nu, linkIndexShift(x, d, arg.E), otherparity);
      d[nu]++;

      // load U(x-nu)(+mu)
      d[nu]--;
      Link U2 = arg.gauge(mu, linkIndexShift(x, d, arg.E), otherparity);
      d[nu]++;

      // load U(x+mu-nu)(nu)
      d[mu]++;
      d[nu]--;
      Link U3 = arg.gauge(nu, linkIndexShift(x, d, arg.E), parity);
      d[mu]--;
      d[nu]++;

      // load U(x)_(+mu)
      Link U4 = arg.gauge(mu, linkIndexShift(x, d, arg.E), parity);

      // load Oprod(x+mu)
      d[mu]++;
      Link Oprod1 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), otherparity);
      Link force = conj(U1) * U2 * U3 * Oprod1 * conj(U4);

      d[mu]--;
      d[nu]--;
      Link Oprod2 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), otherparity);
      force += conj(U1) * Oprod2 * U2 * U3 * conj(U4);

      if (nu < mu) force_total += force;
      else force_total -= force;
    }
  }

  template <typename Arg> struct CloverDerivative
  {
    const Arg &arg;
    constexpr CloverDerivative(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __host__ __device__ void operator()(int x_cb, int parity, int mu)
    {
      using real = typename Arg::real;
      using Complex = complex<real>;
      using Link = Matrix<Complex, Arg::nColor>;

      Link force;

      for (int nu = 0; nu < 4; nu++) {
        if (nu == mu) continue;
        computeForce<Link>(force, arg, x_cb, parity, mu, nu);
      }

      // Write to array
      Link F = arg.force(mu, x_cb, parity);
      F += arg.coeff * static_cast<Link>(force);
      arg.force(mu, x_cb, parity) = F;
    }
  };

} // namespace quda
