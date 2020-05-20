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
#include <instantiate.h>

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon_, bool group_> struct GaugeGaussArg {
    using Float = Float_;
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    static constexpr bool group = group_;

    using Gauge = typename gauge_mapper<Float, recon>::type;

    int threads; // number of active threads required
    int E[4]; // extended grid dimensions
    int X[4]; // true grid dimensions
    int border[4];
    Gauge U;
    RNG rngstate;
    real sigma; // where U = exp(sigma * H)

    GaugeGaussArg(const GaugeField &U, RNG &rngstate, double sigma) : U(U), rngstate(rngstate), sigma(sigma)
    {
      int R = 0;
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = U.R()[dir];
        E[dir] = U.X()[dir];
        X[dir] = U.X()[dir] - border[dir] * 2;
        R += border[dir];
      }
      threads = X[0]*X[1]*X[2]*X[3]/2;
    }
  };

  template <typename real, typename Link> __device__ __host__ Link gauss_su3(cuRNGState &localState)
  {
    Link ret;
    real rand1[4], rand2[4], phi[4], radius[4], temp1[4], temp2[4];

    for (int i = 0; i < 4; ++i) {
      rand1[i] = Random<real>(localState);
      rand2[i] = Random<real>(localState);
    }

    for (int i = 0; i < 4; ++i) {
      phi[i] = 2.0 * M_PI * rand1[i];
      radius[i] = sqrt(-log(rand2[i]));
      sincos(phi[i], &temp2[i], &temp1[i]);
      temp1[i] *= radius[i];
      temp2[i] *= radius[i];
    }

    // construct Anti-Hermitian matrix
    ret(0, 0) = complex<real>(0.0, temp1[2] + rsqrt(3.0) * temp2[3]);
    ret(1, 1) = complex<real>(0.0, -temp1[2] + rsqrt(3.0) * temp2[3]);
    ret(2, 2) = complex<real>(0.0, -2.0 * rsqrt(3.0) * temp2[3]);
    ret(0, 1) = complex<real>(temp1[0], temp1[1]);
    ret(1, 0) = complex<real>(-temp1[0], temp1[1]);
    ret(0, 2) = complex<real>(temp1[3], temp2[0]);
    ret(2, 0) = complex<real>(-temp1[3], temp2[0]);
    ret(1, 2) = complex<real>(temp2[1], temp2[2]);
    ret(2, 1) = complex<real>(-temp2[1], temp2[2]);

    return ret;
  }

  template <typename Arg> __global__ void computeGenGauss(Arg arg)
  {
    using real = typename mapper<typename Arg::Float>::type;
    using Link = Matrix<complex<real>, Arg::nColor>;
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    if (x_cb >= arg.threads) return;

    int x[4];
    getCoords(x, x_cb, arg.X, parity);
    for (int dr = 0; dr < 4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

    if (arg.group && arg.sigma == 0.0) {
      // if sigma = 0 then we just set the output matrix to the identity and finish
      Link I;
      setIdentity(&I);
      for (int mu = 0; mu < 4; mu++) arg.U(mu, linkIndex(x, arg.E), parity) = I;
    } else {
      for (int mu = 0; mu < 4; mu++) {
        cuRNGState localState = arg.rngstate.State()[parity * arg.threads + x_cb];

        // generate Gaussian distributed su(n) fiueld
        Link u = gauss_su3<real, Link>(localState);
        if (arg.group) {
          u = arg.sigma * u;
          expsu3<real>(u);
        }
        arg.U(mu, linkIndex(x, arg.E), parity) = u;

        arg.rngstate.State()[parity * arg.threads + x_cb] = localState;
      }
    }
  }

  template <typename Arg> class GaugeGauss : TunableVectorY
  {
    Arg &arg;
    const GaugeField &meta;

    unsigned int minThreads() const { return arg.threads; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.

  public:
    GaugeGauss(Arg &arg, GaugeField &meta) :
      TunableVectorY(2),
      arg(arg),
      meta(meta) {}

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      computeGenGauss<<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }

    long long flops() const { return 0; }
    long long bytes() const { return meta.Bytes(); }

    void preTune() { arg.rngstate.backup(); }
    void postTune() { arg.rngstate.restore(); }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  struct GenGaussGroup {
    GenGaussGroup(GaugeField &U, RNG &rngstate, double sigma)
    {
      constexpr bool group = true;
      GaugeGaussArg<Float, nColor, recon, group> arg(U, rngstate, sigma);
      GaugeGauss<decltype(arg)> gaugeGauss(arg, U);
      gaugeGauss.apply(0);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  struct GenGaussAlgebra {
    GenGaussAlgebra(GaugeField &U, RNG &rngstate, double sigma)
    {
      constexpr bool group = false;
      GaugeGaussArg<Float, nColor, recon, group> arg(U, rngstate, sigma);
      GaugeGauss<decltype(arg)> gaugeGauss(arg, U);
      gaugeGauss.apply(0);
    }
  };

  void gaugeGauss(GaugeField &U, RNG &rng, double sigma)
  {
    if (!U.isNative()) errorQuda("Order %d with %d reconstruct not supported", U.Order(), U.Reconstruct());

    if (U.LinkType() == QUDA_SU3_LINKS) {

      if (getVerbosity() >= QUDA_SUMMARIZE)
        printfQuda("Creating Gaussian distrbuted gauge field with sigma = %e\n", sigma);
      instantiate<GenGaussGroup, ReconstructFull>(U, rng, sigma);

    } else if (U.LinkType() == QUDA_MOMENTUM_LINKS) {

      if (getVerbosity() >= QUDA_SUMMARIZE)
        printfQuda("Creating Gaussian distrbuted momentum field\n");
      instantiate<GenGaussAlgebra, ReconstructMom>(U, rng, sigma);

    } else {
      errorQuda("Unexpected linkt type %d", U.LinkType());
    }

    // ensure multi-gpu consistency if required
    if (U.GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) {
      U.exchangeExtendedGhost(U.R());
    } else if (U.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD) {
      U.exchangeGhost();
    }
  }

  void gaugeGauss(GaugeField &U, unsigned long long seed, double sigma)
  {
    RNG *randstates = new RNG(U, seed);
    randstates->Init();
    quda::gaugeGauss(U, *randstates, sigma);
    randstates->Release();
    delete randstates;
  }
}
