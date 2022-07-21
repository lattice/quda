#include <clover_field_order.h>
#include <quda_matrix.h>
#include <linalg.cuh>
#include <array.h>
#include <reduction_kernel.h>

namespace quda
{

  template <typename store_t_, bool twist_> struct CloverInvertArg : public ReduceArg<array<double, 2>> {
    using store_t = store_t_;
    using real = typename mapper<store_t>::type;
    static constexpr bool twist = twist_;
    static constexpr int nColor = 3;
    static constexpr int nSpin = 4;
    // we must disable clover reconstruction when writing the inverse
    using Clover = typename clover_mapper<store_t, 72, false, false>::type;

    Clover inverse;
    const Clover clover;
    bool compute_tr_log;
    real mu2;
    real epsilon2;
    real mu2_minus_epsilon2;

    CloverInvertArg(CloverField &field, bool compute_tr_log) :
      ReduceArg<reduce_t>(dim3(field.VolumeCB(), 2, 1)),
      inverse(field, clover::dynamic_inverse() ? false : true), // if dynamic_inverse, then alias to direct term
      clover(field, false),
      compute_tr_log(compute_tr_log),
      mu2(field.Mu2()),
      epsilon2(field.Epsilon2()),
      mu2_minus_epsilon2(field.Mu2()-field.Epsilon2())
    {
      if (!field.isNative()) errorQuda("Clover field %d order not supported", field.Order());
    }
  };

  template <typename Arg> struct InvertClover : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y
    const Arg &arg;
    constexpr InvertClover(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    /**
       Use a Cholesky decomposition and invert the clover matrix
    */
    __device__ __host__ inline reduce_t operator()(reduce_t &value, int x_cb, int parity)
    {
      using real = typename Arg::real;
      constexpr int N = Arg::nColor * Arg::nSpin / 2;
      using Mat = HMatrix<real, N>;
      double trLogA = 0.0;

#pragma unroll
      for (int ch = 0; ch < 2; ch++) {
        Mat A = arg.clover(x_cb, parity, ch);
        A *= static_cast<real>(2.0); // factor of two is inherent to QUDA clover storage

        if (Arg::twist) { // Compute (T^2 + mu2 - epsilon2) first, then invert
          A = A.square();
          A += arg.mu2_minus_epsilon2;
        }

        // compute the Cholesky decomposition
        linalg::Cholesky<HMatrix, clover::cholesky_t<real>, N> cholesky(A);

        // Accumulate trlogA
        if (arg.compute_tr_log)
          for (int j = 0; j < N; j++) trLogA += 2.0 * log(cholesky.D(j));

        Mat Ainv = static_cast<real>(0.5) * cholesky.template invert<Mat>(); // return full inverse
        arg.inverse(x_cb, parity, ch) = Ainv;
      }

      reduce_t result{0, 0};
      parity ? result[1] = trLogA : result[0] = trLogA;
      return operator()(result, value);
    }

  };

} // namespace quda
