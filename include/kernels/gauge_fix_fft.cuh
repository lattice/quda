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
  struct GaugeFixQualityFFTArg : public ReduceArg<double2> {
    using real = typename mapper<store_t>::type;
    static constexpr QudaReconstructType recon = recon_;
    using Gauge = typename gauge_mapper<store_t, recon>::type;
    static constexpr int gauge_dir = gauge_dir_;

    dim3 threads;     // number of active threads required
    int X[4];     // grid dimensions
    Gauge data;
    complex<real> *delta;
    double2 result;

    GaugeFixQualityFFTArg(const GaugeField &data, complex<real> *delta) :
      ReduceArg<double2>(),
      threads(data.VolumeCB(), 2, 1),
      data(data),
      delta(delta)
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = data.X()[dir];
    }
    double getAction() { return result.x; }
    double getTheta() { return result.y; }
  };

  template <typename Arg> struct FixQualityFFT {

    using reduce_t = double2;
    Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr FixQualityFFT(Arg &arg) : arg(arg) {}

    /**
     * @brief Measure gauge fixing quality
     */
    __device__ __host__ inline reduce_t operator()(int x_cb, int parity)
    {
      reduce_t data = make_double2(0.0,0.0);
      using Link = Matrix<complex<typename Arg::real>, 3>;
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      Link delta;
      setZero(&delta);
      //idx = linkIndex(x,X);
      for (int mu = 0; mu < Arg::gauge_dir; mu++) {
        Link U = arg.data(mu, x_cb, parity);
        delta -= U;
      }
      //18*gauge_dir
      data.x += -delta(0, 0).x - delta(1, 1).x - delta(2, 2).x;
      //2
      for (int mu = 0; mu < Arg::gauge_dir; mu++) {
        Link U = arg.data(mu, linkIndexM1(x, arg.X, mu), 1 - parity);
        delta += U;
      }
      //18*gauge_dir
      delta -= conj(delta);
      //18
      //SAVE DELTA!!!!!
      SubTraceUnit(delta);
      int idx = getIndexFull(x_cb, arg.X, parity);

      //Saving Delta
      arg.delta[idx] = delta(0,0);
      arg.delta[idx + 2 * arg.threads.x] = delta(0,1);
      arg.delta[idx + 4 * arg.threads.x] = delta(0,2);
      arg.delta[idx + 6 * arg.threads.x] = delta(1,1);
      arg.delta[idx + 8 * arg.threads.x] = delta(1,2);
      arg.delta[idx + 10 * arg.threads.x] = delta(2,2);
      //12
      data.y += getRealTraceUVdagger(delta, delta);

      //35
      //T=36*gauge_dir+65
      return data;
    }
  };

}
