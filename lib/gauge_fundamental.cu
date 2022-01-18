#include <quda_internal.h>
#include <gauge_field.h>
#include <tunable_nd.h>
#include <kernels/gauge_fundamental.cuh>
#include <instantiate.h>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeFundamental : TunableKernel3D
  {
    using real = typename mapper<Float>::type;
    const GaugeField &in;
    GaugeField &out;
    const double qr_tol;
    const int qr_max_iter;
    const int taylor_N;
    
    bool tuneSharedBytes() const { return false; }
    unsigned int minThreads() const { return in.LocalVolumeCB(); }
    unsigned int maxBlockSize(const TuneParam &) const { return 32; }
    int blockStep() const { return 8; }
    int blockMin() const { return 8; }

  public:
    GaugeFundamental(const GaugeField &in, GaugeField &out, const double qr_tol, const int qr_max_iter, const int taylor_N) :
      TunableKernel3D(in, 2, 4),
      in(in),
      out(out),
      qr_tol(qr_tol),
      qr_max_iter(qr_max_iter),
      taylor_N(taylor_N)
    {
      strcat(aux, comm_dim_partitioned_string());
      strcat(aux,",computeGaugeFundamental");
      apply(device::get_default_stream());
    }
    
     void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugeFundamentalArg<Float, nColor, recon> arg(in, out, qr_tol, qr_max_iter, taylor_N);
      launch<Fundamental>(tp, stream, arg);
    }
    
    void preTune() { out.backup(); }
    void postTune() { out.restore(); }
    
    long long flops() const
    {
      // only counts number of mat-muls per thread
      long long threads = in.LocalVolume() * 4;
      long long mat_flops = nColor * nColor * (8 * nColor - 2);
      long long mat_muls = 1; // 1 comes from Z * conj(U) term
      return mat_muls * mat_flops * threads;
    }
    
    long long bytes() const
    {
      return 0;
    }
  }; // GaugeFundamental
  
#ifdef GPU_GAUGE_TOOLS
  void gaugeFundamentalRep(GaugeField &out, const GaugeField &in, const double qr_tol, const int qr_max_iter, const int taylor_N)
  {
    checkPrecision(out, in);
    checkNative(out, in);
    if(out.Reconstruct() != QUDA_RECONSTRUCT_NO) {
      //errorQuda("The fundamental representation in expressed as a non-unitary, hermitian matrix, so we may not perform reconstruction.");
    }
    
    instantiate<GaugeFundamental>(in, out, qr_tol, qr_max_iter, taylor_N);
  }
#else
  void gaugeFundamentalRep(GaugeField &, const GaugeField &, const double, const int, const int)
  {
    errorQuda("Gauge tools are not built");
  }
#endif

}
