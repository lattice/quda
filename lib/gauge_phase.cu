#include <gauge_field_order.h>
#include <comm_quda.h>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <tune_quda.h>
#include <instantiate.h>

/**
   This code has not been checked.  In particular, I suspect it is
 erroneous in multi-GPU since it looks like the halo ghost region
 isn't being treated here.
 */

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon_, QudaStaggeredPhase phase_>
  struct GaugePhaseArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr QudaStaggeredPhase phase = phase_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;

    Gauge u;
    int X[4];
    int threads;
    Float tBoundary;
    Float i_mu;
    complex<Float> i_mu_phase;
    GaugePhaseArg(GaugeField &u) :
      u(u),
      threads(u.VolumeCB()),
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
  template <int dim, typename Arg> constexpr auto getPhase(int x, int y, int z, int t, Arg &arg) {
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
  __device__ __host__ void gaugePhase(int indexCB, int parity, Arg &arg) {
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

  /**
     Generic GPU staggered phase application
  */
  template <typename Arg>
  __global__ void gaugePhaseKernel(Arg arg) {
    int indexCB = blockIdx.x * blockDim.x + threadIdx.x;
    if (indexCB >= arg.threads) return;
    int parity = blockIdx.y * blockDim.y + threadIdx.y;
    gaugePhase<0>(indexCB, parity, arg);
    gaugePhase<1>(indexCB, parity, arg);
    gaugePhase<2>(indexCB, parity, arg);
    gaugePhase<3>(indexCB, parity, arg);
  }

  template <typename Arg>
  class GaugePhase : TunableVectorY {
    Arg &arg;
    const GaugeField &meta; // used for meta data only

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

  public:
    GaugePhase(Arg &arg, const GaugeField &meta)
      : TunableVectorY(2), arg(arg), meta(meta) { }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      gaugePhaseKernel<Arg> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    void preTune() { arg.u.save(); }
    void postTune() { arg.u.load(); }

    long long flops() const { return 0; }
    long long bytes() const { return 2 * meta.Bytes(); } // 2 from i/o
  };


  template <typename Float, int nColor, QudaReconstructType recon>
  struct GaugePhase_ {
    GaugePhase_(GaugeField &u) {
      if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC) {
        GaugePhaseArg<Float, nColor, recon, QUDA_STAGGERED_PHASE_MILC> arg(u);
        GaugePhase<decltype(arg)> phase(arg, u);
        phase.apply(0);
      } else if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_CPS) {
        GaugePhaseArg<Float, nColor, recon, QUDA_STAGGERED_PHASE_CPS> arg(u);
        GaugePhase<decltype(arg)> phase(arg, u);
        phase.apply(0);
      } else if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_TIFR) {
        GaugePhaseArg<Float, nColor, recon, QUDA_STAGGERED_PHASE_TIFR> arg(u);
        GaugePhase<decltype(arg)> phase(arg, u);
        phase.apply(0);
      } else {
        errorQuda("Undefined phase type");
      }
      checkCudaError();
    }
  };

  void applyGaugePhase(GaugeField &u) {
#ifdef GPU_GAUGE_TOOLS
    instantiate<GaugePhase_, ReconstructNone>(u);
    // ensure that ghosts are updated if needed
    if (u.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD) u.exchangeGhost();
#else
    errorQuda("Gauge tools are not build");
#endif
  }

} // namespace quda
