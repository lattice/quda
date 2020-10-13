#include <clover_field_order.h>
#include <quda_matrix.h>
#include <linalg.cuh>
#include <reduce_helper.h>

namespace quda
{

  template <typename store_t_> struct CloverInvertArg : public ReduceArg<double2> {
    using store_t = store_t_;
    using real = typename mapper<store_t>::type;
    static constexpr int nColor = 3;
    static constexpr int nSpin = 4;
    using Clover = typename clover_mapper<store_t>::type;

    Clover inverse;
    const Clover clover;
    bool compute_tr_log;
    bool twist;
    real mu2;

    CloverInvertArg(CloverField &field, bool compute_tr_log = false) :
      ReduceArg<double2>(),
      inverse(field, true),
      clover(field, false),
      compute_tr_log(compute_tr_log),
      twist(field.Twisted()),
      mu2(field.Mu2())
    {
      if (!field.isNative()) errorQuda("Clover field %d order not supported", field.Order());
    }
  };

  /**
     Use a Cholesky decomposition and invert the clover matrix
   */
  template <typename Arg, bool compute_tr_log, bool twist>
  __device__ __host__ inline double cloverInvertCompute(Arg &arg, int x_cb, int parity)
  {
    using real = typename Arg::real;
    constexpr int N = Arg::nColor * Arg::nSpin / 2;
    using Mat = HMatrix<real, N>;
    double trlogA = 0.0;

    for (int ch = 0; ch < 2; ch++) {
      Mat A = arg.clover(x_cb, parity, ch);
      A *= static_cast<real>(2.0); // factor of two is inherent to QUDA clover storage

      if (twist) { // Compute (T^2 + mu2) first, then invert
        A = A.square();
        A += arg.mu2;
      }

      // compute the Cholesky decomposition
      linalg::Cholesky<HMatrix, real, N> cholesky(A);

      // Accumulate trlogA
      if (compute_tr_log)
        for (int j = 0; j < N; j++) trlogA += 2.0 * log(cholesky.D(j));

      Mat Ainv = static_cast<real>(0.5) * cholesky.invert(); // return full inverse
      arg.inverse(x_cb, parity, ch) = Ainv;
    }

    return trlogA;
  }

  template <typename Arg, bool compute_tr_log, bool twist> void cloverInvert(Arg &arg)
  {
    for (int parity = 0; parity < 2; parity++) {
      for (int x = 0; x < arg.clover.volumeCB; x++) {
        // should make this thread safe if we ever apply threads to cpu code
        double trlogA = cloverInvertCompute<Arg, compute_tr_log, twist>(arg, x, parity);
        if (compute_tr_log) {
          if (parity)
            arg.result_h[0].y += trlogA;
          else
            arg.result_h[0].x += trlogA;
        }
      }
    }
  }

  template <int blockSize, typename Arg, bool compute_tr_log, bool twist>
  __global__ void cloverInvertKernel(Arg arg)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int parity = threadIdx.y;
    double2 trlogA = make_double2(0.0, 0.0);
    double trlogA_parity = 0.0;
    while (idx < arg.clover.volumeCB) {
      trlogA_parity = cloverInvertCompute<Arg, compute_tr_log, twist>(arg, idx, parity);
      trlogA = parity ? make_double2(0.0, trlogA.y + trlogA_parity) : make_double2(trlogA.x + trlogA_parity, 0.0);
      idx += blockDim.x * gridDim.x;
    }
    if (compute_tr_log) arg.template reduce2d<blockSize, 2>(trlogA);
  }

} // namespace quda
