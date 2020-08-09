#include <clover_field_order.h>
#include <quda_matrix.h>
#include <linalg.cuh>
#include <reduce_helper.h>

namespace quda
{

  template <typename store_t_> struct CloverExponentialArg : public ReduceArg<double2> {
    using store_t = store_t_;
    using real = typename mapper<store_t>::type;
    static constexpr int nColor = 3;
    static constexpr int nSpin = 4;
    using Clover = typename clover_mapper<store_t>::type;

    Clover clover;
    int order;
    real mass;
    real *c;
    bool inverse;

    CloverExponentialArg(CloverField &field, int order, double mass, bool inverse = false) :
      ReduceArg<double2>(),
      clover(field, false),
      order(order),
      mass(static_cast<real>(mass)),
      inverse(inverse),
      c(static_cast<real *>(pool_device_malloc((order+1) * sizeof(real))))
    {
      if (!field.isNative()) errorQuda("Clover field %d order not supported", field.Order());
    }
  };

  template <class T, int N>
  __device__ __host__ inline T getTrace(HMatrix<T, N> &a)
  {
    T trace = static_cast<T>(0.0);
#pragma unroll
    for (int i=0; i<N; i++) trace+=a.data[i];
    return trace;
  }

  /**
     Make exponential clover term based on normal one.
   */
  template <typename Arg, bool inverse>
  __device__ __host__ inline void cloverExponentialCompute(Arg &arg, int x_cb, int parity)
  {
    using real = typename Arg::real;
    constexpr int N = Arg::nColor * Arg::nSpin / 2;
    using Mat = HMatrix<real, N>;

    real mass = static_cast<real>(arg.mass);
    real invMass = static_cast<real>(1.0) / mass;
    constexpr int order = arg.order;
    real *c = arg.c;

    c[0] = static_cast<real>(1.0);
    if (inverse) {
      for (int i=1; i<order; i++) {
        c[i] = c[i-1] / static_cast<real>(i);
      }
    } else {
      for (int i=1; i<order; i++) {
        c[i] = c[i-1] / static_cast<real>(-i);
      }
    }

    for (int ch = 0; ch < 2; ch++) {
      Mat A = arg.clover(x_cb, parity, ch);
      A *= static_cast<real>(2.0); // factor of two is inherent to QUDA clover storage

      A *= invMass;
      A -= static_cast<real>(1.0);

      Mat A2 = A * A;
      Mat A3 = A2 * A;
      real tr[5] = {getTrace(A2), getTrace(A3), getTrace(A2*A2), getTrace(A2*A3), getTrace(A3*A3)};
      real psv[5];
      psv[0] = (1.0 / 144.0) * (8.0 * tr[1] * tr[1] - 24.0 * tr[4] + tr[0] * (18.0 * tr[2] - 3.0 * tr[0] * tr[0]));
      psv[1] = (1.0 / 30.0) * (5.0 * tr[0] * tr[1] - 6.0 * tr[3]);
      psv[2] = (1.0 / 8.0) * (tr[0] * tr[0] - 2.0 * tr[2]);
      psv[3] = (-1.0 / 3.0) * tr[1];
      psv[4] = -0.5 * tr[0];

      real q5;
      real q[6] = {static_cast<real>(0.0)};
      if (order > 5) {
#pragma unroll
        for (int i=0; i<6; i++)
          q[i] = c[order - 5 + i];
#pragma unroll
        for (int i=order-6; i>=0; i--) {
          q5 = q[5];
          q[5] = q[4];
          q[4] = q[3] - q5 * psv[4];
          q[3] = q[2] - q5 * psv[3];
          q[2] = q[1] - q5 * psv[2];
          q[1] = q[0] - q5 * psv[1];
          q[0] = c[0] - q5 * psv[0];
         }
      } else {
#pragma unroll
          for (int i=0 ; i<=order ; i++)
            q[i] = c[i];
      }

#pragma unroll
      for (int i=0; i<6; i++)
        q[i] *= static_cast<real>(mass);
      
      Mat A_exp = q[3] + q[4] * A + q[5] * A2;
      A_exp *= A3;
      A_exp += q[0] + q[1] * A + q[2] * A2;
      A_exp *= static_cast<real>(0.5)

      arg.clover(x_cb, parity, ch) = A_exp;
    }
  }

  template <typename Arg, bool inverse> void cloverExponential(Arg &arg)
  {
    for (int parity = 0; parity < 2; parity++) {
      for (int x = 0; x < arg.clover.volumeCB; x++) {
        // should make this thread safe if we ever apply threads to cpu code
        cloverExponentialCompute<Arg, inverse>(arg, x, parity);
      }
    }
  }

  template <typename Arg, bool inverse>
  __global__ void cloverExponentialKernel(Arg arg)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int parity = threadIdx.y;
    while (idx < arg.clover.volumeCB) {
      cloverExponentialCompute<Arg, inverse>(arg, idx, parity);
      idx += blockDim.x * gridDim.x;
    }
  }

} // namespace quda
