#pragma once

#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <kernel.h>
#include <linalg.cuh>

namespace quda {

  template <typename Float, int nColor_, bool twist_>
  struct CloverTraceArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr bool twist = twist_;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr bool dynamic_clover = clover::dynamic_inverse();
    using C = typename clover_mapper<Float>::type;
    using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
    G output;
    const C clover;
    const C clover_inv;
    real coeff;
    real mu2_minus_epsilon2;
    const int parity;

    CloverTraceArg(GaugeField& output, const CloverField& clover, double coeff, int parity) :
      kernel_param(dim3(output.VolumeCB(), 1, 1)),
      output(output),
      clover(clover, false),
      clover_inv(clover, dynamic_clover ? false : true),
      coeff(coeff),
      mu2_minus_epsilon2(clover.Mu2() - clover.Epsilon2()),
      parity(parity) {}
  };

  template <typename Arg>
  __device__ __host__ inline void cloverSigmaTraceCompute(const Arg &arg, const int x)
  {
    using namespace linalg; // for Cholesky
    using real = typename Arg::real;
    constexpr int N = Arg::nColor * Arg::nSpin / 2;
    using Mat = HMatrix<real, N>;
    Mat A[2];

    // load the clover term into memory
#pragma unroll
    for (int ch = 0; ch < 2; ch++) {
      A[ch] = arg.clover_inv(x, arg.parity, ch);
      A[ch] *= static_cast<real>(2.0); // factor of two is inherent to QUDA clover storage

      if constexpr (Arg::dynamic_clover) {
        if (Arg::twist) { // Compute (T^2 + mu2 - epsilon2) first, then invert
          A[ch] = A[ch].square();
          A[ch] += arg.mu2_minus_epsilon2;
        }

        // compute the Cholesky decomposition
        Cholesky<HMatrix, clover::cholesky_t<real>, N> cholesky(A[ch]);
        A[ch] = cholesky.template invert<Mat>(); // return full inverse
      }

      if constexpr (Arg::twist) {
        Mat A0 = arg.clover(x, arg.parity, ch);
        A[ch] = static_cast<real>(0.5) * (A0 * A[ch]); // (1 + T + imu g_5)^{-1} = (1 + T - imu g_5)/((1 + T)^2 + mu^2)
      }
    }

    const Mat &A0 = A[0];
    const Mat &A1 = A[1];

#pragma unroll
    for (int mu=0; mu<4; mu++) {
#pragma unroll
      for (int nu=0; nu<4; nu++) {
        if (nu >= mu) continue;
        Matrix<complex<real>, Arg::nColor> mat = {};

        // X, Y
        if (nu == 0) {
          if (mu == 1) {
#pragma unroll
            for (int j=0; j<3; ++j) {
              mat(j, j).imag(A0(j + 3, j + 3).real() + A1(j + 3, j + 3).real() - A0(j, j).real() - A1(j, j).real());
            }

            // triangular part
#pragma unroll
            for (int j=1; j<3; ++j) {
#pragma unroll
              for (int k=0; k<j; ++k) {
                auto ctmp = A0(j + 3, k + 3) + A1(j + 3, k + 3) - A0(j, k) - A1(j, k);
                mat(j, k) = i_(ctmp);
                mat(k, j) =  i_(conj(ctmp));
              }
            } // X Y

          } else if (mu == 2) {

#pragma unroll
            for (int j=0; j<3; ++j) {
#pragma unroll
              for (int k=0; k<3; ++k) {
                mat(j,k) = conj(A0(k + 3, j)) - A0(j + 3, k) + conj(A1(k + 3, j)) - A1(j + 3, k);
              }
            } // X Z

          } else if (mu == 3) {
#pragma unroll
            for (int j=0; j<3; ++j) {
#pragma unroll
              for (int k=0; k<3; ++k) {
                mat(j, k) = i_(conj(A0(k + 3, j)) + A0(j + 3, k) - conj(A1(k + 3, j)) - A1(j + 3, k));
              }
            }
          } // mu == 3 // X T

        } else if (nu == 1) {
          if (mu == 2) { // Y Z
#pragma unroll
            for (int j=0; j<3; ++j) {
#pragma unroll
              for (int k=0; k<3; ++k) {
                mat(j, k) = -i_(conj(A0(k + 3, j)) + A0(j + 3, k) + conj(A1(k + 3, j)) + A1(j + 3, k));
              }
            }
          } else if (mu == 3){ // Y T
#pragma unroll
            for (int j=0; j<3; ++j) {
#pragma unroll
              for (int k=0; k<3; ++k) {
                mat(j, k) = conj(A0(k + 3, j)) - A0(j + 3, k) - conj(A1(k + 3, j)) + A1(j + 3, k);
              }
            }
          } // mu == 3
        } // nu == 1
        else if (nu == 2){
          if (mu == 3) {
#pragma unroll
            for (int j=0; j<3; ++j) {
              mat(j, j).imag(A0(j, j).real() - A0(j + 3, j + 3).real() - A1(j, j).real() + A1(j + 3, j + 3).real());
            }
#pragma unroll
            for (int j=1; j<3; ++j) {
#pragma unroll
              for (int k=0; k<j; ++k) {
                auto ctmp = A0(j, k) - A0(j + 3, k + 3) - A1(j, k) + A1(j + 3, k + 3);
                mat(j, k) = i_(ctmp);
                mat(k, j) = i_(conj(ctmp));
              }
            }
          }
        }

        arg.output((mu - 1) * mu / 2 + nu, x, arg.parity) = arg.coeff * mat;
      } // nu
    } // mu
  }

  template <typename Arg> struct CloverSigmaTr
  {
    const Arg &arg;
    constexpr CloverSigmaTr(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb)
    {
      cloverSigmaTraceCompute<Arg>(arg, x_cb);
    }
  };

}
