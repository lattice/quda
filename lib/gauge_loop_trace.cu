#include <gauge_field.h>
#include <gauge_tools.h>
#include <instantiate.h>
#include <tunable_reduction.h>
#include <kernels/gauge_loop_trace.cuh>

namespace quda {

  template<typename Float, int nColor, QudaReconstructType recon>
  class GaugeLoopTrace : public TunableMultiReduction {
    const GaugeField &u;
    using reduce_t = array<double, 2>;
    std::vector<reduce_t>& loop_traces;
    const paths p;

  public:
    // max block size of 8 is arbitrary for now, u needs to be local volume??
    GaugeLoopTrace(const GaugeField &u, std::vector<reduce_t> &loop_traces, const paths& p) :
      TunableMultiReduction(u, 1u, p.num_paths, 8),
      u(u),
      loop_traces(loop_traces),
      p(p)
    {
      if (p.num_paths != loop_traces.size()) errorQuda("Loop traces size %lu != number of paths %d", loop_traces.size(), p.num_paths);

      strcat(aux, "num_paths=");
      char loop_str[4];
      u32toa(loop_str, p.num_paths);
      strcat(aux, loop_str);

      // we should add some hash of lengths, paths, etc...

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugeLoopTraceArg<Float, nColor, recon> arg(u, p);
      launch<GaugeLoop>(loop_traces, tp, stream, arg);
      for (auto v : loop_traces) {
        v[0] /= (2. * u.LocalVolumeCB() * comm_size()); 
        v[1] /= (2. * u.LocalVolumeCB() * comm_size());
      }
    }

    long long flops() const
    {
      // UPDATE ME
      auto Nc = u.Ncolor();
      return 6ll*u.Volume()*(3 * (8 * Nc * Nc * Nc - 2 * Nc * Nc) + Nc);
    }

    long long bytes() const {
      // UPDATE ME
      return u.Bytes();
    }
  };

  void gaugeLoopTrace(const GaugeField& u, std::vector<Complex>& loop_traces, int ***input_path,
		 int *length_h, double *path_coeff_h, int num_paths, int path_max_length)
  {
    paths p(input_path, length_h, path_coeff_h, num_paths, path_max_length);

    std::vector<array<double, 2>> tr_array(loop_traces.size());

    // gauge field must be passed as first argument so we peel off its reconstruct type
    instantiate<GaugeLoopTrace, ReconstructGauge>(u, tr_array, p);

    for (auto i = 0u; i < tr_array.size(); i++) { loop_traces[i] = Complex(tr_array[i][0], tr_array[i][1]); }

    p.free();
  }

} // namespace quda
