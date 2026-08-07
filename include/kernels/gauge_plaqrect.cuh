#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <array.h>
#include <packed_array.h>
#include <reduction_kernel.h>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_>
  struct GaugePlaqRectArg : public ReduceArg<array<double, 4>> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float, recon>::type Gauge;

    int E[4]; // extended grid dimensions
    int X[4]; // true grid dimensions
    int border[4];
    Gauge U;

    GaugePlaqRectArg(const GaugeField &U_) : ReduceArg<reduce_t>(dim3(U_.LocalVolumeCB(), 2, 1)), U(U_)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = U_.R()[dir];
        E[dir] = U_.X()[dir];
        X[dir] = U_.X()[dir] - border[dir] * 2;
      }
    }
  };

  // This function computes the 2 rectangles and 1 plaquette in the mu-nu plane
  // associated with the site specified by x and parity.
  //
  // Site diagram:
  //
  // x+2nu--x+mu+2nu
  // |         |
  // x+nu---x+mu+nu----x+2mu+nu
  // |         |          |
  // x--------x+mu------x+2mu
  //
  template <typename Arg>
  __device__ inline double2 plaquetteRectangle(const Arg &arg, int x[], int parity, int mu, int nu)
  {
    using Link = Matrix<complex<typename Arg::Float>, 3>;
    // There are 10 unique links to be fetched, with two of the links
    // being common to all three objects.
    double plaq, rect;
    packed_array<int8_t, 4> dx = {};

    // Accumulate the two common links U_mu(x) and U_nu(x) in U1
    Link U1 = arg.U(mu, linkIndexShift(x, dx, arg.E), parity);                          // U_mu(x)
    U1 = conj(static_cast<Link>(arg.U(nu, linkIndexShift(x, dx, arg.E), parity))) * U1; // conj(U_nu(x))U_mu(x)

    // Accumulate a third link U_nu(x+mu) to form a staple used by the plaquette and one rectangle
    dx[mu]++; // Now at x+mu
    Link U2 = U1 * static_cast<Link>(arg.U(nu, linkIndexShift(x, dx, arg.E), 1 - parity));

    // Get fourth link U_mu(x+nu)
    dx[mu]--;
    dx[nu]++; // Now at x+nu
    Link U3 = conj(static_cast<Link>(arg.U(mu, linkIndexShift(x, dx, arg.E), 1 - parity)));

    // Finish plaquette
    plaq = getTrace(U2 * U3).real();

    // Finish first rectangle, accumulate into U4
    dx[mu]++; // Now at x+mu+nu
    Link U4 = U2 * static_cast<Link>(arg.U(nu, linkIndexShift(x, dx, arg.E), parity));
    dx[mu]--;
    dx[nu]++; // Now at x+2nu
    U4 = U4 * conj(static_cast<Link>(arg.U(mu, linkIndexShift(x, dx, arg.E), parity)));
    dx[nu]--; // Now at x+nu
    U4 = U4 * conj(static_cast<Link>(arg.U(nu, linkIndexShift(x, dx, arg.E), 1 - parity)));

    // Finish second rectangle, partially constructed by U1 and U3
    // Accumulate into U3
    U3 = U3 * U1;
    dx[nu]--;
    dx[mu]++; // Now at x+mu
    U3 = U3 * static_cast<Link>(arg.U(mu, linkIndexShift(x, dx, arg.E), 1 - parity));
    dx[mu]++; // Now at x+2mu
    U3 = U3 * static_cast<Link>(arg.U(nu, linkIndexShift(x, dx, arg.E), parity));
    dx[mu]--;
    dx[nu]++; // Now at x+mu+nu
    U3 = U3 * conj(static_cast<Link>(arg.U(mu, linkIndexShift(x, dx, arg.E), parity)));

    // Sum of the two rectangles
    rect = getTrace(U4 + U3).real();

    return {plaq, rect};
  }

  template <typename Arg> struct PlaquetteRectangle : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y
    const Arg &arg;
    constexpr PlaquetteRectangle(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    // return the rectangle and plaquette at site (x_cb, parity)
    __device__ __host__ inline reduce_t operator()(reduce_t &value, int x_cb, int parity)
    {
      reduce_t plaqRect {0, 0, 0, 0};
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

      for (int mu = 0; mu < 3; mu++) {
        for (int nu = 0; nu < 3; nu++) {
          if (nu >= mu + 1) {
            auto tmp = plaquetteRectangle(arg, x, parity, mu, nu);
            plaqRect[0] += tmp.x; // Spatial plaquette
            plaqRect[2] += tmp.y; // Spatial rectangle
          }
        }
        auto tmp = plaquetteRectangle(arg, x, parity, mu, 3);
        plaqRect[1] += tmp.x; // Temporal plaquette
        plaqRect[3] += tmp.y; // Temporal rectangle
      }
      return operator()(plaqRect, value);
    }
  };
} // namespace quda
