#pragma once

#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <kernel.h>
#include <math_helper.h>

namespace quda
{

  /**
     @brief Where the sites of a gauge field are on the global lattice, halo included, and which t links cross the
     t boundary
   */
  struct GaugePhaseGeometry {
    int_fastdiv E[4]; // extended grid dims
    int_fastdiv X[4]; // regular grid dims
    int border[4];
    int offset[4];       // global coordinates of the rank's first regular site
    int_fastdiv G[4];    // global grid dims
    int parity_shift;    // sites are labelled by their regular parity: this changes it to that of their coordinates
    int boundary_slices; // the last global time slices whose t links cross the boundary: three for long links

    GaugePhaseGeometry(const GaugeField &u)
    {
      parity_shift = 0;
      for (int d = 0; d < 4; d++) {
        E[d] = u.X()[d];
        border[d] = u.R()[d];
        X[d] = E[d] - 2 * border[d];
        offset[d] = comm_coord(d) * X[d];
        G[d] = comm_dim(d) * X[d];
        parity_shift += border[d];
      }
      parity_shift &= 1;
      boundary_slices = u.LinkType() == QUDA_THREE_LINKS ? 3 : 1;
    }
  };

  /**
     @brief The global coordinates of a site of a gauge field, halo included, which wraps around the global lattice
     @param[out] x The coordinates
     @param[in] x_cb The site's checkerboard index in the field
     @param[in] parity The site's parity label
     @param[in] arg The field's geometry
   */
  template <typename Arg>
  __device__ __host__ inline void getGlobalCoords(int x[4], int x_cb, int parity, const Arg &arg)
  {
    getCoords(x, x_cb, arg.E, (parity + arg.parity_shift) & 1);
#pragma unroll
    for (int d = 0; d < 4; d++) x[d] = (x[d] - arg.border[d] + arg.offset[d] + arg.G[d]) % arg.G[d];
  }

  template <typename Float_, int nColor_, QudaReconstructType recon_, QudaStaggeredPhase phase_>
  struct GaugePhaseArg : kernel_param<>, GaugePhaseGeometry {
    using Float = Float_;
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr QudaStaggeredPhase phase = phase_;
    typedef typename gauge_mapper<Float, recon>::type Gauge;

    Gauge u;
    real tBoundary;
    real i_mu;
    complex<real> i_mu_phase;
    GaugePhaseArg(GaugeField &u) :
      kernel_param(dim3(u.VolumeCB(), 2, 1)), GaugePhaseGeometry(u), u(u), i_mu(static_cast<real>(u.iMu()))
    {
      // if staggered phases are applied, then we are removing them
      // else we are applying them
      real dir = u.StaggeredPhaseApplied() ? real(-1.0) : real(1.0);

      const int T = G[3];
      i_mu_phase = complex<real>(static_cast<real>(cos(M_PI * double(u.iMu()) / T)),
                                 dir * static_cast<real>(sin(M_PI * double(u.iMu()) / T)));

      tBoundary = static_cast<real>(static_cast<int>(u.TBoundary()));
    }
  };

  // x, y, z and t are a site's global coordinates
  template <int dim, typename Arg> constexpr auto getPhase(int x, int y, int z, int t, const Arg &arg)
  {
    typename Arg::real phase = typename Arg::real(1.0);
    if (Arg::phase == QUDA_STAGGERED_PHASE_MILC) {
      if (dim == 0) {
        phase = (1 - 2 * (t % 2));
      } else if (dim == 1) {
        phase = (1 - 2 * ((t + x) % 2));
      } else if (dim == 2) {
        phase = (1 - 2 * ((t + x + y) % 2));
      } else if (dim == 3) { // also apply boundary condition
        phase = (t >= arg.G[3] - arg.boundary_slices) ? arg.tBoundary : typename Arg::real(1.0);
      }
    } else if (Arg::phase == QUDA_STAGGERED_PHASE_TIFR) {
      if (dim == 0) {
        phase = (1 - 2 * ((3 + t + z + y) % 2));
      } else if (dim == 1) {
        phase = (1 - 2 * ((2 + t + z) % 2));
      } else if (dim == 2) {
        phase = (1 - 2 * ((1 + t) % 2));
      } else if (dim == 3) { // also apply boundary condition
        phase = (t >= arg.G[3] - arg.boundary_slices) ? arg.tBoundary : typename Arg::real(1.0);
      }
    } else if (Arg::phase == QUDA_STAGGERED_PHASE_CHROMA) {
      // Chroma follows CPS convention, but uses -Dslash instead of Dslash compared to QUDA
      if (dim == 0) {
        phase = -1.0;
      } else if (dim == 1) {
        phase = (1 - 2 * ((1 + x) % 2));
      } else if (dim == 2) {
        phase = (1 - 2 * ((1 + x + y) % 2));
      } else if (dim == 3) { // also apply boundary condition
        phase = ((t >= arg.G[3] - arg.boundary_slices) ? arg.tBoundary : typename Arg::real(1.0))
          * (1 - 2 * ((1 + x + y + z) % 2));
      }
    }
    return phase;
  }

  template <int dim, typename Arg>
  __device__ __host__ void gaugePhase(const int x[4], int indexCB, int parity, const Arg &arg)
  {
    using real = typename Arg::real;

    real phase = getPhase<dim>(x[0], x[1], x[2], x[3], arg);
    Matrix<complex<real>, Arg::nColor> u = arg.u(dim, indexCB, parity);
    u *= phase;

    // apply imaginary chemical potential if needed
    if (dim == 3 && arg.i_mu != real(0.0)) u *= arg.i_mu_phase;

    arg.u(dim, indexCB, parity) = u;
  }

  template <typename Arg> struct GaugePhase {
    const Arg &arg;
    constexpr GaugePhase(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      int x[4];
      getGlobalCoords(x, x_cb, parity, arg);
      gaugePhase<0>(x, x_cb, parity, arg);
      gaugePhase<1>(x, x_cb, parity, arg);
      gaugePhase<2>(x, x_cb, parity, arg);
      gaugePhase<3>(x, x_cb, parity, arg);
    }
  };

  template <typename Float_, int nColor_, QudaReconstructType recon_>
  struct GaugeTBoundaryArg : kernel_param<>, GaugePhaseGeometry {
    using Float = Float_;
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float, recon>::type Gauge;

    Gauge u;
    GaugeTBoundaryArg(GaugeField &u) : kernel_param(dim3(u.VolumeCB(), 2, 1)), GaugePhaseGeometry(u), u(u) { }
  };

  /**
     @brief Negate the t links that cross the t boundary.  They are loaded under the field's t boundary label and
     stored without it: the reconstructions pack a link without the label, which only the unpacking uses.
   */
  template <typename Arg> struct GaugeTBoundary {
    const Arg &arg;
    constexpr GaugeTBoundary(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      int x[4];
      getGlobalCoords(x, x_cb, parity, arg);
      if (x[3] < arg.G[3] - arg.boundary_slices) return;
      Matrix<complex<typename Arg::real>, Arg::nColor> u = arg.u(3, x_cb, parity);
      u *= static_cast<typename Arg::real>(-1.0);
      arg.u(3, x_cb, parity) = u;
    }
  };

} // namespace quda
