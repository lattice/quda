#include <clover_field_order.h>
#include <complex_quda.h>
#include <quda_matrix.h>
#include <linalg.cuh>
#include <cub_helper.cuh>

namespace quda
{

  template <typename Float> struct CloverInvertArg : public ReduceArg<double2> {
    typedef typename clover_mapper<Float>::type C;
    C inverse;
    const C clover;
    bool computeTraceLog;
    bool twist;
    Float mu2;
    CloverInvertArg(CloverField &field, bool computeTraceLog = 0) :
        ReduceArg<double2>(),
        inverse(field, true),
        clover(field, false),
        computeTraceLog(computeTraceLog),
        twist(field.Twisted()),
        mu2(field.Mu2())
    {
      if (!field.isNative()) errorQuda("Clover field %d order not supported", field.Order());
    }
  };

  /**
     Use a Cholesky decomposition and invert the clover matrix
   */
  template <typename Float, typename Arg, bool computeTrLog, bool twist>
  __device__ __host__ inline double cloverInvertCompute(Arg &arg, int x_cb, int parity)
  {

    constexpr int nColor = 3;
    constexpr int nSpin = 4;
    constexpr int N = nColor * nSpin / 2;
    typedef HMatrix<Float, N> Mat;
    double trlogA = 0.0;

    for (int ch = 0; ch < 2; ch++) {
      Mat A = arg.clover(x_cb, parity, ch);
      A *= static_cast<Float>(2.0); // factor of two is inherent to QUDA clover storage

      if (twist) { // Compute (T^2 + mu2) first, then invert
        A = A.square();
        A += arg.mu2;
      }

      // compute the Cholesky decomposition
      linalg::Cholesky<HMatrix, Float, N> cholesky(A);

      // Accumulate trlogA
      if (computeTrLog)
        for (int j = 0; j < N; j++) trlogA += 2.0 * log(cholesky.D(j));

      Mat Ainv = static_cast<Float>(0.5) * cholesky.invert(); // return full inverse
      arg.inverse(x_cb, parity, ch) = Ainv;
    }

    return trlogA;
  }

  template <typename Float, typename Arg, bool computeTrLog, bool twist> void cloverInvert(Arg &arg)
  {
    for (int parity = 0; parity < 2; parity++) {
      for (int x = 0; x < arg.clover.volumeCB; x++) {
        // should make this thread safe if we ever apply threads to cpu code
        double trlogA = cloverInvertCompute<Float, Arg, computeTrLog, twist>(arg, x, parity);
        if (computeTrLog) {
          if (parity)
            arg.result_h[0].y += trlogA;
          else
            arg.result_h[0].x += trlogA;
        }
      }
    }
  }

  template <int blockSize, typename Float, typename Arg, bool computeTrLog, bool twist>
  __global__ void cloverInvertKernel(Arg arg)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int parity = threadIdx.y;
    double2 trlogA = make_double2(0.0, 0.0);
    double trlogA_parity = 0.0;
    while (idx < arg.clover.volumeCB) {
      trlogA_parity = cloverInvertCompute<Float, Arg, computeTrLog, twist>(arg, idx, parity);
      trlogA = parity ? make_double2(0.0, trlogA.y + trlogA_parity) : make_double2(trlogA.x + trlogA_parity, 0.0);
      idx += blockDim.x * gridDim.x;
    }
    if (computeTrLog) reduce2d<blockSize, 2>(arg, trlogA);
  }

} // namespace quda
