#include <gauge_field.h>
#include <gauge_path_quda.h>
#include <instantiate.h>
#include <tunable_reduction.h>
#include <kernels/gauge_loop_trace.cuh>

namespace quda {

  template<typename Float, int nColor, QudaReconstructType recon>
  class GaugeLoopTrace : public TunableMultiReduction {
    const GaugeField &u;
    using reduce_t = array<double, 2>;
    std::vector<reduce_t>& loop_traces;
    double factor;
    const paths<1> p;

  public:
    // max block size of 8 is arbitrary for now
    GaugeLoopTrace(const GaugeField &u, std::vector<reduce_t> &loop_traces, double factor, const paths<1>& p) :
      TunableMultiReduction(u, 2u, p.num_paths, 8),
      u(u),
      loop_traces(loop_traces),
      factor(factor),
      p(p)
    {
      if (p.num_paths != static_cast<int>(loop_traces.size())) errorQuda("Loop traces size %lu != number of paths %d", loop_traces.size(), p.num_paths);

      strcat(aux, "num_paths=");
      u32toa(aux + strlen(aux), p.num_paths);

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugeLoopTraceArg<Float, nColor, recon> arg(u, factor, p);
      launch<GaugeLoop>(loop_traces, tp, stream, arg);
    }

    long long flops() const
    {
      auto Nc = u.Ncolor();
      auto mat_mul_flops = 8ll * Nc * Nc * Nc - 2 * Nc * Nc;
      // matrix multiplies + traces + rescale
      return (p.count * mat_mul_flops + p.num_paths * (2 * Nc + 2)) * u.Volume();
    }

    long long bytes() const {
      // links * one LatticeColorMatrix worth of data
      return p.count * u.Bytes() / 4;
    }
  };

  void gaugeLoopTrace(const GaugeField& u, std::vector<Complex>& loop_traces, double factor, std::vector<int**>& input_path,
		 std::vector<int>& length, std::vector<double>& path_coeff, int num_paths, int path_max_length)
  {
    paths<1> p(input_path, length, path_coeff, num_paths, path_max_length);

    std::vector<array<double, 2>> tr_array(loop_traces.size());

    // gauge field must be passed as first argument so we peel off its reconstruct type
    instantiate<GaugeLoopTrace, ReconstructNo12>(u, tr_array, factor, p);

    for (auto i = 0u; i < tr_array.size(); i++) { loop_traces[i] = Complex(tr_array[i][0], tr_array[i][1]); }

    p.free();
  }

} // namespace quda
