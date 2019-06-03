#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <atomic.cuh>
#include <cub_helper.cuh>
#include <index_helper.cuh>
#include <random_quda.h>

namespace quda {

  template <typename Gauge>
  struct GaugeGaussArg {
    int threads; // number of active threads required
    int E[4]; // extended grid dimensions
    int X[4]; // true grid dimensions
    int border[4];
    Gauge U;
    RNG rngstate;

    GaugeGaussArg(const Gauge &U, const GaugeField &data, RNG &rngstate)
      : U(U), rngstate(rngstate)
    {
      int R = 0;
      for (int dir=0; dir<4; ++dir) {
	border[dir] = data.R()[dir];
	E[dir] = data.X()[dir];
	X[dir] = data.X()[dir] - border[dir]*2;
	R += border[dir];
      }
      threads = X[0]*X[1]*X[2]*X[3]/2;
    }
  };

  template<typename real, typename Link>
  __device__ __host__  Link genGaussSU3(cuRNGState &localState) {
    Link ret;
    real rand1[4], rand2[4], phi[4], radius[4], temp1[4], temp2[4];

    for (int i=0; i<4; ++i) {
      rand1[i] = Random<real>(localState);
      rand2[i] = Random<real>(localState);
    }

    for (int i=0; i<4; ++i) {
      phi[i] = 2.0*M_PI*rand1[i];
      rand2[i] = rand2[i];
      radius[i] = sqrt( -log(rand2[i]) );

      sincos(phi[i], &temp2[i], &temp1[i]);
      temp1[i] *= radius[i];
      temp2[i] *= radius[i];
    }

    ret(0,0) = complex<real>( temp1[2] + rsqrt(3.0)*temp2[3], 0.0);
    ret(0,1) = complex<real>( temp1[0], -temp1[1]);
    ret(0,2) = complex<real>( temp1[3], -temp2[0]);
    ret(1,0) = complex<real>( temp1[0], temp1[1] );
    ret(1,1) = complex<real>( -temp1[2] + rsqrt(3.0) * temp2[3], 0.0 );
    ret(1,2) = complex<real>( temp2[1], -temp2[2] );
    ret(2,0) = complex<real>( temp1[3], temp2[0] );
    ret(2,1) = complex<real>( temp2[1], temp2[2] );
    ret(2,2) = complex<real>( - 2.0*rsqrt(3.0) * temp2[3], 0.0 );

    return ret;
  }


  template<typename Float, typename Gauge>
  __global__ void computeGenGauss(GaugeGaussArg<Gauge> arg)
  {
    using real = typename mapper<Float>::type;
    typedef Matrix<complex<real>,3> Link;
    int x_cb = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y + blockIdx.y*blockDim.y;
    if (x_cb >= arg.threads) return;

    int x[4];
    getCoords(x, x_cb, arg.X, parity);
    for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

    for (int mu = 0; mu < 4; mu++) {
      cuRNGState localState = arg.rngstate.State()[parity * arg.threads + x_cb];

      Link U = genGaussSU3<real,Link>(localState);

      arg.rngstate.State()[parity * arg.threads + x_cb] = localState;
      arg.U(mu, linkIndex(x,arg.E), parity) = U;
    }
  }

  template<typename Float, typename Gauge> class GaugeGauss : TunableVectorY {
    GaugeGaussArg<Gauge> arg;
    const GaugeField &meta;

  private:
    unsigned int minThreads() const { return arg.threads; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.

  public:
    GaugeGauss(GaugeGaussArg<Gauge> &arg, GaugeField &meta)
      : TunableVectorY(2), arg(arg), meta(meta) {}
    ~GaugeGauss () { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        computeGenGauss<Float><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      } else {
        errorQuda("Randomize GaugeFields on CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString());
    }

    long long flops() const { return 0; }
    long long bytes() const { return meta.Bytes(); }

    void preTune() { arg.rngstate.backup(); }
    void postTune() { arg.rngstate.restore(); }
  };

  template<typename Float, typename Gauge> void genGauss(const Gauge &U, GaugeField& data, RNG &rngstate)
  {
    GaugeGaussArg<Gauge> arg(U, data, rngstate);
    GaugeGauss<Float,Gauge> gaugeGauss(arg, data);
    gaugeGauss.apply(0);
  }

  template<typename Float> void gaugeGauss(GaugeField &U, RNG &rngstate)
  {
    switch (U.Reconstruct()) {
    case QUDA_RECONSTRUCT_NO:
      {
        typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type Gauge;
        genGauss<Float>(Gauge(U), U, rngstate);
        break;
      }
    case QUDA_RECONSTRUCT_12:
      {
        typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type Gauge;
        genGauss<Float>(Gauge(U), U, rngstate);
        break;
      }
    case QUDA_RECONSTRUCT_8:
      {
        typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type Gauge;
        genGauss<Float>(Gauge(U), U, rngstate);
      }
    default: errorQuda("Reconstruction type %d of origin gauge field not supported", U.Reconstruct());
    }
  }

  void gaugeGauss(GaugeField &U, RNG &rngstate)
  {
    if (!U.isNative()) errorQuda("Order %d with %d reconstruct not supported", U.Order(), U.Reconstruct());
    if (U.Ncolor() != 3) errorQuda("Nc = %d not supported", U.Ncolor());

    switch (U.Precision()) {
    case QUDA_DOUBLE_PRECISION: gaugeGauss<double>(U, rngstate); break;
    case QUDA_SINGLE_PRECISION: gaugeGauss<float>(U, rngstate); break;
    default: errorQuda("Precision %d not supported", U.Precision());
    }

    // ensure multi-gpu consistency if required
    if (U.GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) {
      U.exchangeExtendedGhost(U.R());
    } else if (U.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD) {
      U.exchangeGhost();
    }
  }

}
