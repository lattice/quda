#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <comm_quda.h>
#include <pgauge_monte.h>
#include <reduce_helper.h>
#include <index_helper.cuh>
#include <instantiate.h>

namespace quda {

  template <typename Float, int nColor_, QudaReconstructType recon_>
  struct KernelArg : public ReduceArg<double2> {
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    using real = typename mapper<Float>::type;
    using Gauge = typename gauge_mapper<real, recon>::type;
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    int border[4];
    Gauge dataOr;
    double2 result;

    KernelArg(const GaugeField &data) :
      ReduceArg<double2>(),
      dataOr(data)
    {
      for(int dir=0; dir<4; ++dir){
        border[dir] = data.R()[dir];
        X[dir] = data.X()[dir] - border[dir]*2;
      }
      threads = X[0]*X[1]*X[2]*X[3]/2;
    }
    double2 getValue() { return result; }
  };

  template <int blockSize, int type, typename Arg>
  __global__ void compute_Value(Arg arg)
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y;

    complex<double> val(0.0, 0.0);
    while (idx < arg.threads) {
      int X[4];
#pragma unroll
      for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

      int x[4];
      getCoords(x, idx, X, parity);
#pragma unroll
      for(int dr=0; dr<4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2*arg.border[dr];
      }
      idx = linkIndex(x,X);
#pragma unroll
      for (int mu = 0; mu < 4; mu++) {
        Matrix<complex<typename Arg::real>, Arg::nColor> U = arg.dataOr(mu, idx, parity);
        if (type == 0) val += getDeterminant(U);
        else if (type == 1) val += getTrace(U);
      }

      idx += blockDim.x*gridDim.x;
    }

    double2 sum = make_double2(val.real(), val.imag());
    reduce2d<blockSize,2>(arg, sum);
  }

  template <typename Arg, int type>
  class CalcFunc : TunableLocalParity {
    Arg &arg;
    const GaugeField &meta;
    TuneParam tp;
    bool tuneGridDim() const { return true; }

  public:
    CalcFunc(Arg &arg, const GaugeField &meta) :
      arg(arg),
      meta(meta)
    { }

    void apply(const qudaStream_t &stream)
    {
      tp = tuneLaunch(*this, getTuning(), getVerbosity());
      LAUNCH_KERNEL_LOCAL_PARITY(compute_Value, (*this), tp, stream, arg, type);
      arg.complete(&arg.result, stream);
      if (!activeTuning()) {
        comm_allreduce_array((double*)&arg.result, 2);
        arg.result.x /= (double)(4*2*arg.threads*comm_size());
        arg.result.y /= (double)(4*2*arg.threads*comm_size());
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }

    long long flops() const {
      if (meta.Ncolor()==3 && type == 0) return 264LL*meta.Volume();
      else if (type == 1) return 2*meta.Geometry()*meta.Ncolor()*meta.Volume();
      else return 0;
    }

    long long bytes() const { return meta.Bytes(); }
  };

  template <typename Float, int nColor, QudaReconstructType recon, int type>
  double2 computeValue(GaugeField& data)
  {
    TimeProfile profileGenericFunc("GenericFunc", false);
    if (getVerbosity() >= QUDA_SUMMARIZE) profileGenericFunc.TPSTART(QUDA_PROFILE_COMPUTE);

    KernelArg<Float, nColor, recon> arg(data);
    CalcFunc<decltype(arg), type> func(arg, data);
    func.apply(0);
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("%s: %.16e, %.16e\n", type == 0 ? "Determinant" : "Trace", arg.getValue().x, arg.getValue().y);
    checkCudaError();

    if (getVerbosity() >= QUDA_SUMMARIZE){
      profileGenericFunc.TPSTOP(QUDA_PROFILE_COMPUTE);
      double secs = profileGenericFunc.Last(QUDA_PROFILE_COMPUTE);
      double gflops = (func.flops()*1e-9)/(secs);
      double gbytes = func.bytes()/(secs*1e9);
      printfQuda("%s: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n",
                 type == 0 ? "Determinant" : "Trace", secs, gflops*comm_size(), gbytes*comm_size());
    }
    return arg.getValue();
  }

  template <typename Float, int nColor, QudaReconstructType recon> struct computeDeterminant {
    computeDeterminant(GaugeField &data, double2 &det)
    {
      det = computeValue<Float, nColor, recon, 0>(data);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct computeTrace {
    computeTrace(GaugeField &data, double2 &trace)
    {
      trace = computeValue<Float, nColor, recon, 1>(data);
    }
  };

  /**
   * @brief Calculate the Determinant
   *
   * @param[in] data Gauge field
   * @returns double2 complex Determinant value
   */
  double2 getLinkDeterminant(GaugeField& data)
  {
    double2 det = make_double2(0.0,0.0);
#ifdef GPU_GAUGE_ALG
    instantiate<computeDeterminant>(data, det);
#else
    errorQuda("Pure gauge code has not been built");
#endif // GPU_GAUGE_ALG
    return det;
  }

  /**
   * @brief Calculate the Trace
   *
   * @param[in] data Gauge field
   * @returns double2 complex trace value
   */
  double2 getLinkTrace(GaugeField& data)
  {
    double2 det = make_double2(0.0,0.0);
#ifdef GPU_GAUGE_ALG
    instantiate<computeTrace>(data, det);
#else
    errorQuda("Pure gauge code has not been built");
#endif // GPU_GAUGE_ALG
    return det;
  }

} // namespace quda
