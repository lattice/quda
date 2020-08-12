#include <cstdio>
#include <cstdlib>
#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <float_vector.h>
#include <complex_quda.h>
#include <instantiate.h>

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon_u, QudaReconstructType recon_m, int N_>
  struct UpdateGaugeArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static constexpr int N = N_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    typedef typename gauge_mapper<Float,recon_u>::type Gauge;
    typedef typename gauge_mapper<Float,recon_m>::type Mom;
    Gauge out;
    Gauge in;
    Mom mom;
    Float dt;
    int nDim;
    UpdateGaugeArg(GaugeField &out, const GaugeField &in, const GaugeField &mom, Float dt, int nDim)
      : out(out), in(in), mom(mom), dt(dt), nDim(nDim) { }
  };

  template <bool conj_mom, bool exact, typename Arg>
  __device__ __host__  void compute(Arg &arg, int x, int parity)
  {
    using Float = typename Arg::Float;
    typedef complex<Float> Complex;
    Matrix<Complex, Arg::nColor> link, result, mom;

    for (int dir=0; dir<arg.nDim; ++dir) {
      link = arg.in(dir, x, parity);
      mom = arg.mom(dir, x, parity);

      Complex trace = getTrace(mom);
      for (int c=0; c<Arg::nColor; c++) mom(c,c) -= trace/static_cast<Float>(Arg::nColor);

      if (!exact) {
	result = link;

	// Nth order expansion of exponential
	if (!conj_mom) {
	  for (int r= Arg::N; r>0; r--)
	    result = (arg.dt/r)*mom*result + link;
	} else {
	  for (int r= Arg::N; r>0; r--)
	    result = (arg.dt/r)*conj(mom)*result + link;
	}
      } else {
	mom = arg.dt * mom;
        expsu3<Float>(mom);

        if (!conj_mom) {
          link = mom * link;
        } else {
          link = conj(mom) * link;
        }

        result = link;
      }

      arg.out(dir, x, parity) = result;
    } // dir
  }

  template <bool conj_mom, bool exact, typename Arg>
  __global__ void updateGaugeFieldKernel(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    if (x_cb >= arg.out.volumeCB) return;
    int parity = blockIdx.y*blockDim.y + threadIdx.y;
    compute<conj_mom,exact>(arg, x_cb, parity);
  }

  template <typename Arg, bool conj_mom, bool exact>
   class UpdateGaugeField : public TunableVectorY {
    Arg &arg;
    const GaugeField &meta; // meta data

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.in.volumeCB; }

  public:
    UpdateGaugeField(Arg &arg, const GaugeField &meta) :
      TunableVectorY(2),
      arg(arg),
      meta(meta) {}

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      updateGaugeFieldKernel<conj_mom,exact><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
    } // apply

    long long flops() const {
      const int Nc = Arg::nColor;
      return arg.nDim*2*arg.in.volumeCB*Arg::N*(Nc*Nc*2 +                 // scalar-matrix multiply
                                                (8*Nc*Nc*Nc - 2*Nc*Nc) +  // matrix-matrix multiply
                                                Nc*Nc*2);                 // matrix-matrix addition
    }

    long long bytes() const { return arg.nDim*2*arg.in.volumeCB*(arg.in.Bytes() + arg.out.Bytes() + arg.mom.Bytes()); }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }
  };

  template <typename Float, int nColor, QudaReconstructType recon_u> struct UpdateGauge
  {
    UpdateGauge(GaugeField &out, const GaugeField &in, const GaugeField &mom, double dt, bool conj_mom, bool exact)
    {
      if (mom.Reconstruct() != QUDA_RECONSTRUCT_10) errorQuda("Reconstruction type %d not supported", mom.Reconstruct());
      constexpr QudaReconstructType recon_m = QUDA_RECONSTRUCT_10;
      constexpr int N = 8; // degree of exponential expansion
      UpdateGaugeArg<Float, nColor, recon_u, recon_m, N> arg(out, in, mom, dt, 4);
      if (conj_mom) {
        if (exact) {
          UpdateGaugeField<decltype(arg),true,true> updateGauge(arg, in);
          updateGauge.apply(0);
        } else {
          UpdateGaugeField<decltype(arg),true,false> updateGauge(arg, in);
          updateGauge.apply(0);
        }
      } else {
        if (exact) {
          UpdateGaugeField<decltype(arg),false,true> updateGauge(arg, in);
          updateGauge.apply(0);
        } else {
          UpdateGaugeField<decltype(arg),false,false> updateGauge(arg, in);
          updateGauge.apply(0);
        }
      }
      checkCudaError();
    }
  };

  void updateGaugeField(GaugeField &out, double dt, const GaugeField& in, const GaugeField& mom, bool conj_mom, bool exact)
  {
#ifdef GPU_GAUGE_TOOLS
    checkPrecision(out, in, mom);
    checkLocation(out, in, mom);
    checkReconstruct(out, in);
    instantiate<UpdateGauge,ReconstructNo12>(out, in, mom, dt, conj_mom, exact);
#else
    errorQuda("Gauge tools are not build");
#endif
  }

} // namespace quda
