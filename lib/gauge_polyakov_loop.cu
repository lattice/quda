#include <gauge_field.h>
#include <gauge_tools.h>
#include <instantiate.h>
#include <tunable_reduction.h>
#include <kernels/gauge_polyakov_loop.cuh>

namespace quda {

  template<typename Float, int nColor, QudaReconstructType recon>
  class GaugePolyakovLoop : public TunableReduction2D {
    GaugeField &product_field;
    const GaugeField &u;
    using reduce_t = array<double, 2>;
    reduce_t &ploop;

  public:
    GaugePolyakovLoop(GaugeField &product_field, const GaugeField &u, array<double, 2> &ploop) :
      TunableReduction2D(u),
      product_field(product_field),
      u(u),
      ploop(ploop)
    {
      strcat(aux, ",3d_vol=");
      strcat(aux, product_field.VolString());

      apply(device::get_default_stream());

      // We normalize by the 3-d volume
      long vol3d = product_field.Volume() * comm_dim(0) * comm_dim(1) * comm_dim(2);
      ploop[0] /= vol3d;
      ploop[1] /= vol3d;
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugePolyakovLoopArg<Float, nColor, recon> arg(product_field, u);
      launch<PolyakovLoop>(ploop, tp, stream, arg);
    }

    long long flops() const
    {
      auto Nc = u.Ncolor();
      auto mat_mul_flops = 8ll * Nc * Nc * Nc - 2 * Nc * Nc;
      // multiplies for each loop plus traces
      return mat_mul_flops * u.Volume() / 4 + 2 * Nc * product_field.Volume();
    }

    long long bytes() const {
      // links * one LatticeColorMatrix worth of data
      return u.Bytes() / 4 + product_field.Bytes();
    }
  };

  void gaugePolyakovLoop(double ploop[2], const GaugeField& u, int dir) {

    if (dir != 3) errorQuda("Unsupported direction %d", dir);
    if (comm_dim(dir) != 1) errorQuda("Splitting in the %d direction is not supported yet", dir);

    // Form a usable (or metadata) 3-d gauge "container"
    GaugeFieldParam gParam(u);
    lat_dim_t x;
    for (int d = 0; d < 3; d++) x[d] = u.X()[d];
    x[3] = 1;
    gParam.x = x;
    gParam.create = QUDA_NULL_FIELD_CREATE; // not needed in case where decomp isn't split in `t` direction
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.location = QUDA_CUDA_FIELD_LOCATION;
    gParam.geometry = QUDA_SCALAR_GEOMETRY;

    std::unique_ptr<GaugeField> product_field_ = std::make_unique<cudaGaugeField>(gParam);
    GaugeField& product_field = reinterpret_cast<GaugeField&>(*product_field_.get());

    // output array
    array<double, 2> loop;

    instantiate<GaugePolyakovLoop, ReconstructNo12>(product_field, u, loop);

    ploop[0] = loop[0];
    ploop[1] = loop[1];

  }

} // namespace quda
