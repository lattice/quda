#pragma once

#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <kernel.h>

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon_, QudaStaggeredPhase phase_>
  struct GaugePhaseArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr QudaStaggeredPhase phase = phase_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;

    Gauge u;
    int X[4];
    Float tBoundary;
    Float i_mu;
    complex<Float> i_mu_phase;
    GaugePhaseArg(GaugeField &u) :
      kernel_param(dim3(u.VolumeCB(), 2, 1)),
      u(u),
      i_mu(u.iMu())
    {
      // if staggered phases are applied, then we are removing them
      // else we are applying them
      Float dir = u.StaggeredPhaseApplied() ? -1.0 : 1.0;

      i_mu_phase = complex<Float>( cos(M_PI * u.iMu() / (u.X()[3]*comm_dim(3)) ),
				   dir * sin(M_PI * u.iMu() / (u.X()[3]*comm_dim(3))) );

      for (int d=0; d<4; d++) X[d] = u.X()[d];

      // only set the boundary condition on the last time slice of nodes
      bool last_node_in_t = (commCoords(3) == commDim(3)-1);
      tBoundary = (Float)(last_node_in_t ? u.TBoundary() : QUDA_PERIODIC_T);
    }
  };

  // FIXME need to check this with odd local volumes
  template <int dim, typename Arg> constexpr auto getPhase(int x, int y, int z, int t, const Arg &arg) {
    typename Arg::Float phase = 1.0;
    if (Arg::phase == QUDA_STAGGERED_PHASE_MILC) {
      if (dim==0) {
	phase = (1.0 - 2.0 * (t % 2) );
      } else if (dim == 1) {
	phase = (1.0 - 2.0 * ((t + x) % 2) );
      } else if (dim == 2) {
	phase = (1.0 - 2.0 * ((t + x + y) % 2) );
      } else if (dim == 3) { // also apply boundary condition
	phase = (t == arg.X[3]-1) ? arg.tBoundary : 1.0;
      }
    } else if (Arg::phase == QUDA_STAGGERED_PHASE_TIFR) {
      if (dim==0) {
	phase = (1.0 - 2.0 * ((3 + t + z + y) % 2) );
      } else if (dim == 1) {
	phase = (1.0 - 2.0 * ((2 + t + z) % 2) );
      } else if (dim == 2) {
	phase = (1.0 - 2.0 * ((1 + t) % 2) );
      } else if (dim == 3) { // also apply boundary condition
	phase = (t == arg.X[3]-1) ? arg.tBoundary : 1.0;
      }
    } else if (Arg::phase == QUDA_STAGGERED_PHASE_CPS) {
      if (dim==0) {
	phase = 1.0;
      } else if (dim == 1) {
	phase = (1.0 - 2.0 * ((1 + x) % 2) );
      } else if (dim == 2) {
	phase = (1.0 - 2.0 * ((1 + x + y) % 2) );
      } else if (dim == 3) { // also apply boundary condition
	phase = ((t == arg.X[3]-1) ? arg.tBoundary : 1.0) *
	  (1.0 - 2 * ((1 + x + y + z) % 2) );
      }
    }
    return phase;
  }

  template <int dim, typename Arg>
  __device__ __host__ void gaugePhase(int indexCB, int parity, const Arg &arg) {
    typedef typename mapper<typename Arg::Float>::type real;

    int x[4];
    getCoords(x, indexCB, arg.X, parity);

    real phase = getPhase<dim>(x[0], x[1], x[2], x[3], arg);
    Matrix<complex<real>,Arg::nColor> u = arg.u(dim, indexCB, parity);
    u *= phase;

    // apply imaginary chemical potential if needed
    if (dim==3 && arg.i_mu != 0.0) u *= arg.i_mu_phase;

    arg.u(dim, indexCB, parity) = u;
  }

  template <typename Arg> struct GaugePhase
  {
    const Arg &arg;
    constexpr GaugePhase(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      gaugePhase<0>(x_cb, parity, arg);
      gaugePhase<1>(x_cb, parity, arg);
      gaugePhase<2>(x_cb, parity, arg);
      gaugePhase<3>(x_cb, parity, arg);
    }
  };

}
