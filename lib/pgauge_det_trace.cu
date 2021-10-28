#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <comm_quda.h>
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

    KernelArg(const GaugeField &data) :
      ReduceArg<double2>(),
      dataOr(data),
      threads(data.LocalVolumeCB())
    {
      for (int dir=0; dir<4; ++dir) {
        border[dir] = data.R()[dir];
        X[dir] = data.X()[dir] - border[dir]*2;
      }
    }
  };

  template <int blockSize, int type, typename Arg>
  __global__ void compute(Arg arg)
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

    arg.template reduce2d<blockSize,2>(val);
  }

  template <typename Float, int nColor, QudaReconstructType recon, int type>
  class CalcFunc : TunableLocalParityReduction {
    double2 &result;
    const GaugeField &u;

  public:
    CalcFunc(double2 &result, const GaugeField &u) :
      result(result),
      u(u)
    {
      apply(0);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      KernelArg<Float, nColor, recon> arg(u);
      LAUNCH_KERNEL_LOCAL_PARITY(compute, (*this), tp, stream, arg, type, decltype(arg));
      arg.complete(result, stream);
      if (!activeTuning()) {
        comm_allreduce_array((double*)&result, 2);
        result.x /= (double)(4*u.LocalVolume()*comm_size());
        result.y /= (double)(4*u.LocalVolume()*comm_size());
      }
    }

    TuneKey tuneKey() const { return TuneKey(u.VolString(), typeid(*this).name(), u.AuxString()); }

    long long flops() const {
      if (u.Ncolor()==3 && type == 0) return 264LL*u.LocalVolume();
      else if (type == 1) return 2*u.Geometry()*u.Ncolor()*u.LocalVolume();
      else return 0;
    }

    long long bytes() const { return u.Bytes(); }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct computeDeterminant {
    computeDeterminant(GaugeField &data, double2 &det)
    {
      CalcFunc<Float, nColor, recon, 0>(det, data);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct computeTrace {
    computeTrace(GaugeField &data, double2 &trace)
    {
      CalcFunc<Float, nColor, recon, 1>(trace, data);
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
