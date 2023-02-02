#pragma once

#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <array.h>
#include <kernel.h>
#include <reduction_kernel.h>
#include <fast_intdiv.h>

namespace quda {

//UNCOMMENT THIS IF YOU WANT TO USE LESS MEMORY
#define GAUGEFIXING_DONT_USE_GX
//Without using the precalculation of g(x),
//we loose some performance, because Delta(x) is written in normal lattice coordinates need for the FFTs
//and the gauge array in even/odd format

#ifdef HOST_DEBUG
#ifdef GAUGEFIXING_DONT_USE_GX
#warning Not using precalculated g(x)
#else
#warning Using precalculated g(x)
#endif
#endif

  template <typename Float, int dir_>
  struct GaugeFixFFTRotateArg : kernel_param<> {
    static constexpr int dir = dir_;
    int_fastdiv X[4];     // grid dimensions
    complex<Float> *tmp0;
    complex<Float> *tmp1;
    GaugeFixFFTRotateArg(const GaugeField &data, complex<Float> *tmp0, complex<Float> *tmp1) :
      kernel_param(dim3(data.Volume(), 1, 1)),
      tmp0(tmp0),
      tmp1(tmp1)
    {
      for (int d = 0; d < 4; d++) X[d] = data.X()[d];
    }
  };

  template <typename Arg> struct FFTrotate
  {
    const Arg &arg;
    constexpr FFTrotate(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int id)
    {
      if (arg.dir == 0) {
        int x3 = id / (arg.X[0] * arg.X[1] * arg.X[2]);
        int x2 = (id / (arg.X[0] * arg.X[1])) % arg.X[2];
        int x1 = (id / arg.X[0]) % arg.X[1];
        int x0 = id % arg.X[0];

        int id = x0 + (x1 + (x2 + x3 * arg.X[2]) * arg.X[1]) * arg.X[0];
        int id_out = x2 + (x3 + (x0 + x1 * arg.X[0]) * arg.X[3]) * arg.X[2];
        arg.tmp1[id_out] = arg.tmp0[id];
      }

      if (arg.dir == 1) {
        int x1 = id / (arg.X[2] * arg.X[3] * arg.X[0]);
        int x0 = (id / (arg.X[2] * arg.X[3])) % arg.X[0];
        int x3 = (id / arg.X[2]) % arg.X[3];
        int x2 = id % arg.X[2];

        int id = x2 + (x3 + (x0 + x1 * arg.X[0]) * arg.X[3]) * arg.X[2];
        int id_out = x0 + (x1 + (x2 + x3 * arg.X[2]) * arg.X[1]) * arg.X[0];
        arg.tmp1[id_out] = arg.tmp0[id];
      }
    }
  };

  template <typename store_t, QudaReconstructType recon>
  struct GaugeFixArg : kernel_param<> {
    using Float = typename mapper<store_t>::type;
    using Gauge = typename gauge_mapper<store_t, recon>::type;
    static constexpr int elems = recon / 2;
    Gauge data;
    int_fastdiv X[4];     // grid dimensions
    Float *invpsq;
    complex<Float> *delta;
    complex<Float> *gx;
    Float alpha;
    int volume;

    GaugeFixArg(GaugeField &data, double alpha) :
      kernel_param(dim3(data.VolumeCB(), 2, 1)),
      data(data),
      alpha(static_cast<Float>(alpha)),
      volume(data.Volume())
    {
      for (int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir];
      invpsq = (Float*)device_malloc(sizeof(Float) * volume);
      delta = (complex<Float>*)device_malloc(sizeof(complex<Float>) * volume * 6);
#ifdef GAUGEFIXING_DONT_USE_GX
      gx = (complex<Float>*)device_malloc(sizeof(complex<Float>) * volume);
#else
      gx = (complex<Float>*)device_malloc(sizeof(complex<Float>) * volume * elems);
#endif
    }

    void free()
    {
      device_free(invpsq);
      device_free(delta);
      device_free(gx);
    }
  };

  template <typename Arg> struct set_invpsq
  {
    const Arg &arg;
    constexpr set_invpsq(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using Float = typename Arg::Float;
      int id = parity * arg.threads.x + x_cb;
      int x1 = id / (arg.X[2] * arg.X[3] * arg.X[0]);
      int x0 = (id / (arg.X[2] * arg.X[3])) % arg.X[0];
      int x3 = (id / arg.X[2]) % arg.X[3];
      int x2 = id % arg.X[2];
      //id  =  x2 + (x3 +  (x0 + x1 * arg.X[0]) * arg.X[3]) * arg.X[2];
      Float sx = quda::sinpi( (Float)x0 / (Float)arg.X[0]);
      Float sy = quda::sinpi( (Float)x1 / (Float)arg.X[1]);
      Float sz = quda::sinpi( (Float)x2 / (Float)arg.X[2]);
      Float st = quda::sinpi( (Float)x3 / (Float)arg.X[3]);
      Float sinsq = sx * sx + sy * sy + sz * sz + st * st;
      Float prcfact = 0.0;
      //The FFT normalization is done here
      if (sinsq > 0.00001) prcfact = 4.0 / (sinsq * (Float)(arg.volume));
      arg.invpsq[id] = prcfact;
    }
  };

  template <typename Arg> struct mult_norm_2d
  {
    const Arg &arg;
    constexpr mult_norm_2d(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }
    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      int id = parity * arg.threads.x + x_cb;
      arg.gx[id] = arg.gx[id] * arg.invpsq[id];
    }
  };

  /**
   * @brief container to pass parameters for the gauge fixing quality kernel
   */
  template <typename store_t, QudaReconstructType recon_, int gauge_dir_>
  struct GaugeFixQualityFFTArg : public ReduceArg<array<double, 2>> {
    using real = typename mapper<store_t>::type;
    static constexpr QudaReconstructType recon = recon_;
    using Gauge = typename gauge_mapper<store_t, recon>::type;
    static constexpr int gauge_dir = gauge_dir_;

    int_fastdiv X[4];     // grid dimensions
    Gauge data;
    complex<real> *delta;
    reduce_t result;
    int volume;

    GaugeFixQualityFFTArg(const GaugeField &data, complex<real> *delta) :
      ReduceArg<reduce_t>(dim3(data.VolumeCB(), 2, 1), 1, true), // reset = true
      data(data),
      delta(delta),
      result{0, 0},
      volume(data.Volume())
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = data.X()[dir];
    }

    double getAction() { return result[0]; }
    double getTheta() { return result[1]; }
  };

  template <typename Arg> struct FixQualityFFT : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr FixQualityFFT(const Arg &arg) : arg(arg) {}

    /**
     * @brief Measure gauge fixing quality
     */
    __device__ __host__ inline reduce_t operator()(reduce_t &value, int x_cb, int parity)
    {
      reduce_t data{0, 0};
      using matrix = Matrix<complex<typename Arg::real>, 3>;
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      matrix delta;
      setZero(&delta);

      for (int mu = 0; mu < Arg::gauge_dir; mu++) {
        matrix U = arg.data(mu, x_cb, parity);
        delta -= U;
      }
      //18*gauge_dir
      data[0] = -delta(0, 0).real() - delta(1, 1).real() - delta(2, 2).real();
      //2
      for (int mu = 0; mu < Arg::gauge_dir; mu++) {
        matrix U = arg.data(mu, linkIndexM1(x, arg.X, mu), 1 - parity);
        delta += U;
      }
      //18*gauge_dir
      delta -= conj(delta);
      //18
      //SAVE DELTA!!!!!
      SubTraceUnit(delta);
      int idx = getIndexFull(x_cb, arg.X, parity);

      //Saving Delta
      arg.delta[idx + 0 * arg.volume] = delta(0,0);
      arg.delta[idx + 1 * arg.volume] = delta(0,1);
      arg.delta[idx + 2 * arg.volume] = delta(0,2);
      arg.delta[idx + 3 * arg.volume] = delta(1,1);
      arg.delta[idx + 4 * arg.volume] = delta(1,2);
      arg.delta[idx + 5 * arg.volume] = delta(2,2);

      //12
      data[1] = getRealTraceUVdagger(delta, delta);

      //35
      //T=36*gauge_dir+65
      return operator()(data, value);
    }
  };

  template <typename Float>
  __host__ __device__ inline void reunit_link(Matrix<complex<Float>,3> &U)
  {
    complex<Float> t2((Float)0.0, (Float)0.0);
    Float t1 = 0.0;
    //first normalize first row
    //sum of squares of row
#pragma unroll
    for ( int c = 0; c < 3; c++ ) t1 += norm(U(0,c));
    t1 = (Float)1.0 / sqrt(t1);
    //14
    //used to normalize row
#pragma unroll
    for ( int c = 0; c < 3; c++ ) U(0,c) *= t1;
    //6
#pragma unroll
    for ( int c = 0; c < 3; c++ ) t2 += conj(U(0,c)) * U(1,c);
    //24
#pragma unroll
    for ( int c = 0; c < 3; c++ ) U(1,c) -= t2 * U(0,c);
    //24
    //normalize second row
    //sum of squares of row
    t1 = 0.0;
#pragma unroll
    for ( int c = 0; c < 3; c++ ) t1 += norm(U(1,c));
    t1 = (Float)1.0 / sqrt(t1);
    //14
    //used to normalize row
#pragma unroll
    for ( int c = 0; c < 3; c++ ) U(1, c) *= t1;
    //6
    //Reconstruct last row
    U(2,0) = conj(U(0,1) * U(1,2) - U(0,2) * U(1,1));
    U(2,1) = conj(U(0,2) * U(1,0) - U(0,0) * U(1,2));
    U(2,2) = conj(U(0,0) * U(1,1) - U(0,1) * U(1,0));
    //42
    //T=130
  }

  template <typename Arg> struct U_EO_NEW {
    const Arg &arg;
    constexpr U_EO_NEW(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      using Float = typename Arg::Float;
      using complex = complex<Float>;
      using matrix = Matrix<complex, 3>;

      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      int idx = ((x[3] * arg.X[2] + x[2]) * arg.X[1] + x[1]) * arg.X[0] + x[0];
      matrix de;
      //Read Delta
      de(0,0) = arg.delta[idx + 0 * arg.volume];
      de(0,1) = arg.delta[idx + 1 * arg.volume];
      de(0,2) = arg.delta[idx + 2 * arg.volume];
      de(1,1) = arg.delta[idx + 3 * arg.volume];
      de(1,2) = arg.delta[idx + 4 * arg.volume];
      de(2,2) = arg.delta[idx + 5 * arg.volume];

      de(1,0) = complex(-de(0,1).real(), de(0,1).imag());
      de(2,0) = complex(-de(0,2).real(), de(0,2).imag());
      de(2,1) = complex(-de(1,2).real(), de(1,2).imag());

      matrix g;
      setIdentity(&g);
      g += de * (arg.alpha * static_cast<Float>(0.5));
      //36
      reunit_link<Float>(g);
      //130

      for ( int mu = 0; mu < 4; mu++ ) {
        matrix U = arg.data(mu, x_cb, parity);

        matrix g0;
        U = g * U;
        //198
        idx = linkNormalIndexP1(x, arg.X, mu);
        //Read Delta
        de(0,0) = arg.delta[idx + 0 * arg.volume];
        de(0,1) = arg.delta[idx + 1 * arg.volume];
        de(0,2) = arg.delta[idx + 2 * arg.volume];
        de(1,1) = arg.delta[idx + 3 * arg.volume];
        de(1,2) = arg.delta[idx + 4 * arg.volume];
        de(2,2) = arg.delta[idx + 5 * arg.volume];

        de(1,0) = complex(-de(0,1).real(), de(0,1).imag());
        de(2,0) = complex(-de(0,2).real(), de(0,2).imag());
        de(2,1) = complex(-de(1,2).real(), de(1,2).imag());

        setIdentity(&g0);
        g0 += de * (arg.alpha * static_cast<Float>(0.5));
        //36
        reunit_link<Float>(g0);
        //130

        U = U * conj(g0);
        //198
        arg.data(mu, x_cb, parity) = U;
      }
    }
  };

  template <typename Arg> struct GX {
    const Arg &arg;
    constexpr GX(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using complex = complex<typename Arg::Float>;
      using matrix = Matrix<complex, 3>;

      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      int idx = ((x[3] * arg.X[2] + x[2]) * arg.X[1] + x[1]) * arg.X[0] + x[0];

      matrix de;
      //Read Delta
      de(0,0) = arg.delta[idx + 0 * arg.volume];
      de(0,1) = arg.delta[idx + 1 * arg.volume];
      de(0,2) = arg.delta[idx + 2 * arg.volume];
      de(1,1) = arg.delta[idx + 3 * arg.volume];
      de(1,2) = arg.delta[idx + 4 * arg.volume];
      de(2,2) = arg.delta[idx + 5 * arg.volume];

      de(1,0) = complex(-de(0,1).x, de(0,1).y);
      de(2,0) = complex(-de(0,2).x, de(0,2).y);
      de(2,1) = complex(-de(1,2).x, de(1,2).y);

      matrix g;
      setIdentity(&g);
      g += de * (arg.alpha * static_cast<typename Arg::Float>(0.5));

      //36
      reunit_link<typename Arg::Float>(g);

      //130
      //gx is represented in even/odd order
      for (int i = 0; i < Arg::elems; i++) arg.gx[parity*arg.threads.x + x_cb + i * arg.volume] = g.data[i];

      //T=166 for Elems 9
      //T=208 for Elems 6
    }
  };

  template <typename Arg> struct U_EO {
    const Arg &arg;
    constexpr U_EO(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using complex = complex<typename Arg::Float>;
      using matrix = Matrix<complex, 3>;

      int x[4];
      getCoords(x, x_cb, arg.X, parity);

      matrix g;
      for (int i = 0; i < Arg::elems; i++) g.data[i] = arg.gx[parity * arg.threads.x + x_cb + i * arg.volume];

      if (Arg::elems == 6) {
        g(2,0) = conj(g(0,1) * g(1,2) - g(0,2) * g(1,1));
        g(2,1) = conj(g(0,2) * g(1,0) - g(0,0) * g(1,2));
        g(2,2) = conj(g(0,0) * g(1,1) - g(0,1) * g(1,0));
        //42
      }
      for (int mu = 0; mu < 4; mu++) {
        matrix U = arg.data(mu, x_cb, parity);
        matrix g0;
        U = g * U;
        //198
        int idm1 = linkIndexP1(x, arg.X, mu);
        idm1 += (1 - parity) * arg.volume / 2;
        for (int i = 0; i < Arg::elems; i++) g0.data[i] = arg.gx[idm1 + i * arg.volume];

        if (Arg::elems == 6) {
          g0(2,0) = conj(g0(0,1) * g0(1,2) - g0(0,2) * g0(1,1));
          g0(2,1) = conj(g0(0,2) * g0(1,0) - g0(0,0) * g0(1,2));
          g0(2,2) = conj(g0(0,0) * g0(1,1) - g0(0,1) * g0(1,0));
          //42
        }
        U = U * conj(g0);
        //198
        arg.data(mu, x_cb, parity) = U;
      }
      //T=42+4*(198*2+42) Elems=6
      //T=4*(198*2) Elems=9
      //Not accounting here the reconstruction of the gauge if 12 or 8!!!!!!
    }
  };

}
