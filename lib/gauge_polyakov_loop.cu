#include <gauge_field.h>
#include <gauge_tools.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <kernels/gauge_polyakov_loop.cuh>

namespace quda {

  template<typename Float, int nColor, QudaReconstructType recon>
  class GaugePolyakovLoopSplit : public TunableKernel2D {
    GaugeField &product_field;
    const GaugeField &u;
    unsigned int minThreads() const { return product_field.LocalVolumeCB(); }

  public:
    GaugePolyakovLoopSplit(const GaugeField &u, GaugeField &product_field) :
      TunableKernel2D(u, 2),
      product_field(product_field),
      u(u)
    {
      strcat(aux, ",4d_vol=");
      strcat(aux, u.VolString());
      strcat(aux, ",3d_vol=");
      strcat(aux, product_field.VolString());

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugePolyakovLoopSplitArg<Float, nColor, recon> arg(product_field, u);
      launch<PolyakovLoopSplit>(tp, stream, arg);
    }

    long long flops() const
    {
      auto Nc = u.Ncolor();
      auto mat_mul_flops = 8ll * Nc * Nc * Nc - 2 * Nc * Nc;
      // multiplies for each loop
      return mat_mul_flops * u.Volume() / 4;
    }

    long long bytes() const {
      // links * one LatticeColorMatrix worth of data
      return u.Bytes() / 4 + product_field.Bytes();
    }
  };

  template<typename Float, int nColor, QudaReconstructType recon>
  class GaugePolyakovLoopTrace : public TunableReduction2D {
    const GaugeField &u;
    using reduce_t = array<double, 2>;
    reduce_t &ploop;
    bool compute_loop;

  public:
    GaugePolyakovLoopTrace(const GaugeField &u, array<double, 2> &ploop, bool compute_loop) :
      TunableReduction2D(u),
      u(u),
      ploop(ploop),
      compute_loop(compute_loop)
    {
      strcat(aux, ",4d_vol=");
      strcat(aux, u.VolString());
      if (compute_loop) strcat(aux, ",loop");

      apply(device::get_default_stream());

      // We normalize by the 3-d volume
      long vol3d = u.Volume() * comm_dim(0) * comm_dim(1) * comm_dim(2) / u.X()[3];
      ploop[0] /= vol3d;
      ploop[1] /= vol3d;
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (compute_loop) {
        GaugePolyakovLoopTraceArg<Float, nColor, recon, true> arg(u);
        launch<PolyakovLoopTrace>(ploop, tp, stream, arg);
      } else {
        GaugePolyakovLoopTraceArg<Float, nColor, recon, false> arg(u);
        launch<PolyakovLoopTrace>(ploop, tp, stream, arg);
      }
    }

    long long flops() const
    {
      auto Nc = u.Ncolor();
      auto mat_mul_flops = 8ll * Nc * Nc * Nc - 2 * Nc * Nc;
      // multiplies for each loop plus traces
      return ((compute_loop) ? (mat_mul_flops * u.Volume() / 4) : 0) + 2 * Nc * u.Volume() / u.X()[3];
    }

    long long bytes() const {
      // links * one LatticeColorMatrix
      return compute_loop ? (u.Bytes()) / 4 : u.Bytes();
    }
  };

  void gaugePolyakovLoop(double ploop[2], const GaugeField& u, int dir) {

    if (dir != 3) errorQuda("Unsupported direction %d", dir);

    // output array
    array<double, 2> loop;

    // If the dir dimension isn't partitioned, we can just do a quick compute + reduce
    if (commDimPartitioned(dir)) {
      // comms aren't actually supported yet
      if (comm_dim(dir) > 1) errorQuda("Splitting in the %d direction is not supported yet", dir);

      // Form a usable 3-d gauge slice
      GaugeFieldParam gParam(u);
      lat_dim_t x;
      for (int d = 0; d < 3; d++) x[d] = u.X()[d];
      x[3] = 1;
      gParam.x = x;
      gParam.create = QUDA_NULL_FIELD_CREATE;
      gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      gParam.location = QUDA_CUDA_FIELD_LOCATION;
      gParam.geometry = QUDA_SCALAR_GEOMETRY;

      std::unique_ptr<GaugeField> product_field_ = std::make_unique<cudaGaugeField>(gParam);
      GaugeField& product_field = reinterpret_cast<GaugeField&>(*product_field_.get());

      instantiate<GaugePolyakovLoopSplit, ReconstructNo12>(u, product_field);

      // Combine (tbd)

      // Trace only
      instantiate<GaugePolyakovLoopTrace, ReconstructNo12>(product_field, loop, false);

    } else {
      // construct loop and trace
      instantiate<GaugePolyakovLoopTrace, ReconstructNo12>(u, loop, true);
    }

    ploop[0] = loop[0];
    ploop[1] = loop[1];

  }

} // namespace quda
