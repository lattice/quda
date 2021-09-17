#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <kernel.h>

namespace quda
{

  template <typename Float, QudaReconstructType recon> struct CloverDerivArg : kernel_param<> {
    using Force = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
    using Oprod = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
    using Gauge = typename gauge_mapper<Float, recon>::type;
    using real = typename mapper<Float>::type;
    int X[4];
    int E[4];
    int border[4];
    real coeff;
    int parity;

    Force force;
    Gauge gauge;
    Oprod oprod;

    CloverDerivArg(const GaugeField &force, const GaugeField &gauge, const GaugeField &oprod, double coeff, int parity) :
      kernel_param(dim3(force.VolumeCB(), 2, 4)),
      coeff(coeff),
      parity(parity),
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

  template <typename Arg, int mu, int nu, typename Link>
  __device__ __forceinline__ void computeForce(Link &force, const Arg &arg, int xIndex, int yIndex)
  {
    int otherparity = (1 - arg.parity);

    const int tidx = mu > nu ? (mu - 1) * mu / 2 + nu : (nu - 1) * nu / 2 + mu;

    if (yIndex == 0) { // do "this" force

      int x[4];
      getCoordsExtended(x, xIndex, arg.X, arg.parity, arg.border);

      // U[mu](x) U[nu](x+mu) U[*mu](x+nu) U[*nu](x) Oprod(x)
      {
        int d[4] = { };

        // load U(x)_(+mu)
        Link U1 = arg.gauge(mu, linkIndexShift(x, d, arg.E), arg.parity);

        // load U(x+mu)_(+nu)
        d[mu]++;
        Link U2 = arg.gauge(nu, linkIndexShift(x, d, arg.E), otherparity);
        d[mu]--;

        // load U(x+nu)_(+mu)
        d[nu]++;
        Link U3 = arg.gauge(mu, linkIndexShift(x, d, arg.E), otherparity);
        d[nu]--;

        // load U(x)_(+nu)
        Link U4 = arg.gauge(nu, linkIndexShift(x, d, arg.E), arg.parity);

        // load Oprod
        Link Oprod1 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), arg.parity);

        if (nu < mu)
          force -= U1 * U2 * conj(U3) * conj(U4) * Oprod1;
        else
          force += U1 * U2 * conj(U3) * conj(U4) * Oprod1;

        d[mu]++;
        d[nu]++;
        Link Oprod2 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), arg.parity);
        d[mu]--;
        d[nu]--;

        if (nu < mu)
          force -= U1 * U2 * Oprod2 * conj(U3) * conj(U4);
        else
          force += U1 * U2 * Oprod2 * conj(U3) * conj(U4);
      }

      {
        int d[4] = { };

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
        Link U3 = arg.gauge(nu, linkIndexShift(x, d, arg.E), arg.parity);
        d[mu]--;
        d[nu]++;

        // load U(x)_(+mu)
        Link U4 = arg.gauge(mu, linkIndexShift(x, d, arg.E), arg.parity);

        d[mu]++;
        d[nu]--;
        Link Oprod1 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), arg.parity);
        d[mu]--;
        d[nu]++;

        if (nu < mu)
          force += conj(U1) * U2 * Oprod1 * U3 * conj(U4);
        else
          force -= conj(U1) * U2 * Oprod1 * U3 * conj(U4);

        Link Oprod4 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), arg.parity);

        if (nu < mu)
          force += Oprod4 * conj(U1) * U2 * U3 * conj(U4);
        else
          force -= Oprod4 * conj(U1) * U2 * U3 * conj(U4);
      }

    } else { // else do other force

      int y[4] = { };
      getCoordsExtended(y, xIndex, arg.X, otherparity, arg.border);

      {
        int d[4] = { };

        // load U(x)_(+mu)
        Link U1 = arg.gauge(mu, linkIndexShift(y, d, arg.E), otherparity);

        // load U(x+mu)_(+nu)
        d[mu]++;
        Link U2 = arg.gauge(nu, linkIndexShift(y, d, arg.E), arg.parity);
        d[mu]--;

        // load U(x+nu)_(+mu)
        d[nu]++;
        Link U3 = arg.gauge(mu, linkIndexShift(y, d, arg.E), arg.parity);
        d[nu]--;

        // load U(x)_(+nu)
        Link U4 = arg.gauge(nu, linkIndexShift(y, d, arg.E), otherparity);

        // load opposite parity Oprod
        d[nu]++;
        Link Oprod3 = arg.oprod(tidx, linkIndexShift(y, d, arg.E), arg.parity);
        d[nu]--;

        if (nu < mu)
          force -= U1 * U2 * conj(U3) * Oprod3 * conj(U4);
        else
          force += U1 * U2 * conj(U3) * Oprod3 * conj(U4);

        // load Oprod(x+mu)
        d[mu]++;
        Link Oprod4 = arg.oprod(tidx, linkIndexShift(y, d, arg.E), arg.parity);
        d[mu]--;

        if (nu < mu)
          force -= U1 * Oprod4 * U2 * conj(U3) * conj(U4);
        else
          force += U1 * Oprod4 * U2 * conj(U3) * conj(U4);
      }

      // Lower leaf
      // U[nu*](x-nu) U[mu](x-nu) U[nu](x+mu-nu) Oprod(x+mu) U[*mu](x)
      {
        int d[4] = { };

        // load U(x-nu)(+nu)
        d[nu]--;
        Link U1 = arg.gauge(nu, linkIndexShift(y, d, arg.E), arg.parity);
        d[nu]++;

        // load U(x-nu)(+mu)
        d[nu]--;
        Link U2 = arg.gauge(mu, linkIndexShift(y, d, arg.E), arg.parity);
        d[nu]++;

        // load U(x+mu-nu)(nu)
        d[mu]++;
        d[nu]--;
        Link U3 = arg.gauge(nu, linkIndexShift(y, d, arg.E), otherparity);
        d[mu]--;
        d[nu]++;

        // load U(x)_(+mu)
        Link U4 = arg.gauge(mu, linkIndexShift(y, d, arg.E), otherparity);

        // load Oprod(x+mu)
        d[mu]++;
        Link Oprod1 = arg.oprod(tidx, linkIndexShift(y, d, arg.E), arg.parity);
        d[mu]--;

        if (nu < mu)
          force += conj(U1) * U2 * U3 * Oprod1 * conj(U4);
        else
          force -= conj(U1) * U2 * U3 * Oprod1 * conj(U4);

        d[nu]--;
        Link Oprod2 = arg.oprod(tidx, linkIndexShift(y, d, arg.E), arg.parity);
        d[nu]++;

        if (nu < mu)
          force += conj(U1) * Oprod2 * U2 * U3 * conj(U4);
        else
          force -= conj(U1) * Oprod2 * U2 * U3 * conj(U4);
      }
    }

  } // namespace quda

  template <typename Arg> struct CloverDerivative
  {
    const Arg &arg;
    constexpr CloverDerivative(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __host__ __device__ void operator()(int x_cb, int parity, int mu)
    {
      using real = typename Arg::real;
      using Complex = complex<real>;
      using Link = Matrix<Complex, 3>;

      Link force;

      switch (mu) {
      case 0:
        computeForce<Arg, 0, 1, Link>(force, arg, x_cb, parity);
        computeForce<Arg, 0, 2, Link>(force, arg, x_cb, parity);
        computeForce<Arg, 0, 3, Link>(force, arg, x_cb, parity);
        break;
      case 1:
        computeForce<Arg, 1, 0, Link>(force, arg, x_cb, parity);
        computeForce<Arg, 1, 3, Link>(force, arg, x_cb, parity);
        computeForce<Arg, 1, 2, Link>(force, arg, x_cb, parity);
        break;
      case 2:
        computeForce<Arg, 2, 3, Link>(force, arg, x_cb, parity);
        computeForce<Arg, 2, 0, Link>(force, arg, x_cb, parity);
        computeForce<Arg, 2, 1, Link>(force, arg, x_cb, parity);
        break;
      case 3:
        computeForce<Arg, 3, 2, Link>(force, arg, x_cb, parity);
        computeForce<Arg, 3, 1, Link>(force, arg, x_cb, parity);
        computeForce<Arg, 3, 0, Link>(force, arg, x_cb, parity);
        break;
      }

      // Write to array
      Link F = arg.force(mu, x_cb, parity == 0 ? arg.parity : 1 - arg.parity);
      F += arg.coeff * force;
      arg.force(mu, x_cb, parity == 0 ? arg.parity : 1 - arg.parity) = F;
    }
  };

} // namespace quda
