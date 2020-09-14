#pragma once

#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <kernel.h>
#include <reduction_kernel.h>

namespace quda {

  /**
   * @brief container to pass parameters for the gauge fixing quality kernel
   */
  template <typename store_t, QudaReconstructType recon_, int gauge_dir_>
  struct GaugeFixQualityOVRArg : public ReduceArg<double2> {
    using real = typename mapper<store_t>::type;
    static constexpr QudaReconstructType recon = recon_;
    using Gauge = typename gauge_mapper<store_t, recon>::type;
    static constexpr int gauge_dir = gauge_dir_;

    dim3 threads; // number of active threads required
    int X[4]; // grid dimensions
    int border[4];
    Gauge data;
    double2 result;
    GaugeFixQualityOVRArg(const GaugeField &data) :
      ReduceArg<double2>(),
      threads(1, 2, 1),
      data(data)
    {
      for ( int dir = 0; dir < 4; ++dir ) {
        X[dir] = data.X()[dir] - data.R()[dir] * 2;
        border[dir] = data.R()[dir];
      }
      threads.x = X[0]*X[1]*X[2]*X[3]/2;
    }

    double getAction(){ return result.x; }
    double getTheta(){ return result.y; }
  };

  template <typename Arg> struct FixQualityOVR {

    using reduce_t = double2;
    Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr FixQualityOVR(Arg &arg) : arg(arg) {}

    /**
     * @brief Measure gauge fixing quality
     */
    __device__ __host__ inline reduce_t operator()(int x_cb, int parity)
    {
      reduce_t data = make_double2(0.0,0.0);
      using Link = Matrix<complex<typename Arg::real>, 3>;

      int X[4];
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];

      int x[4];
      getCoords(x, x_cb, X, parity);
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }
      Link delta;
      setZero(&delta);
      //load upward links
      for ( int mu = 0; mu < Arg::gauge_dir; mu++ ) {
        Link U = arg.data(mu, linkIndex(x, X), parity);
        delta -= U;
      }
      //18*gauge_dir
      data.x += -delta(0, 0).x - delta(1, 1).x - delta(2, 2).x;
      //2
      //load downward links
      for (int mu = 0; mu < Arg::gauge_dir; mu++) {
        Link U = arg.data(mu, linkIndexM1(x,X,mu), 1 - parity);
        delta += U;
      }
      //18*gauge_dir
      delta -= conj(delta);
      //18
      SubTraceUnit(delta);
      //12
      data.y += getRealTraceUVdagger(delta, delta);
      //35
      //T=36*gauge_dir+65

      return data;
    }
  };

}
