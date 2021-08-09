#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <float_vector.h>
#include <kernel.h>

namespace quda {

  template <typename Float, int nColor_, QudaReconstructType recon_u, QudaReconstructType recon_m,
            int N_, bool conj_mom_, bool exact_>
  struct UpdateGaugeArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int N = N_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr bool conj_mom = conj_mom_;
    static constexpr bool exact = exact_;
    typedef typename gauge_mapper<Float,recon_u>::type Gauge;
    typedef typename gauge_mapper<Float,recon_m>::type Mom;
    Gauge out;
    Gauge in;
    Mom mom;
    real dt;
    UpdateGaugeArg(GaugeField &out, const GaugeField &in, const GaugeField &mom, real dt) :
      kernel_param(dim3(in.VolumeCB(), 2, in.Geometry())),
      out(out),
      in(in),
      mom(mom),
      dt(dt) { }
  };

  template <typename Arg> struct UpdateGauge {
    const Arg &arg;
    constexpr UpdateGauge(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x, int parity, int dir)
    {
      using real = typename Arg::real;
      using Complex = complex<real>;
      Matrix<Complex, Arg::nColor> link, result, mom;

      link = arg.in(dir, x, parity);
      mom = arg.mom(dir, x, parity);

      Complex trace = getTrace(mom);
      for (int c=0; c<Arg::nColor; c++) mom(c,c) -= trace/static_cast<real>(Arg::nColor);

      if (!arg.exact) {
        result = link;

        // Nth order expansion of exponential
        if (!arg.conj_mom) {
          for (int r= Arg::N; r>0; r--)
            result = (arg.dt/r)*mom*result + link;
        } else {
          for (int r= Arg::N; r>0; r--)
            result = (arg.dt/r)*conj(mom)*result + link;
        }
      } else {
        mom = arg.dt * mom;
        expsu3<real>(mom);

        if (!arg.conj_mom) {
          link = mom * link;
        } else {
          link = conj(mom) * link;
        }

        result = link;
      }

      arg.out(dir, x, parity) = result;
    }
  };
  
}
