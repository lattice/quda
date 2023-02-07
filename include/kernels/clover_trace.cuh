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
    const C clover_inv;
    real coeff;
    real mu2_minus_epsilon2;
    const int parity;

    CloverTraceArg(GaugeField& output, const CloverField& clover, double coeff, int parity) :
      kernel_param(dim3(output.VolumeCB(), 1, 1)),
      output(output),
      clover_inv(clover, dynamic_clover ? false : true),
      coeff(coeff),
      mu2_minus_epsilon2(clover.Mu2() - clover.Epsilon2()),
      parity(parity) {}
  };

  template <typename Arg>
  __device__ __host__ void cloverSigmaTraceCompute(const Arg &arg, const int x)
  {
    using namespace linalg; // for Cholesky
    using real = typename Arg::real;
    constexpr int N = Arg::nColor * Arg::nSpin / 2;
    using Mat = HMatrix<real, N>;
    real A_array[72];

#pragma unroll
    for (int chirality = 0; chirality < 2; chirality++) {
      Mat A = arg.clover_inv(x, arg.parity, chirality);

      if (Arg::dynamic_clover) {
        A *= static_cast<real>(2.0); // factor of two is inherent to QUDA clover storage

        if (Arg::twist) { // Compute (T^2 + mu2 - epsilon2) first, then invert
          A = A.square();
          A += arg.mu2_minus_epsilon2;
        }

        // compute the Cholesky decomposition
        Cholesky<HMatrix, clover::cholesky_t<real>, N> cholesky(A);
        A = static_cast<real>(0.5) * cholesky.template invert<Mat>(); // return full inverse 
        if (Arg::twist) {       
          Mat A0 = arg.clover_inv(x, arg.parity, chirality);
          A = static_cast<real>(0.5) * (A0*A); // (1 + T + imu g_5)^{-1} = (1 + T - imu g_5)/((1 + T)^2 + mu^2)
        }
      }

      for (int i = 0; i < 36; ++i) A_array[chirality * 36 + i] = A.data[i];
    }

    // load the clover term into memory
    for (int mu=0; mu<4; mu++) {
      for (int nu=0; nu<mu; nu++) {

        Matrix<complex<real>, Arg::nColor> mat;
        setZero(&mat);

        real diag[2][6];
        complex<real> tri[2][15];
        const int idtab[15]={0,1,3,6,10,2,4,7,11,5,8,12,9,13,14};
        complex<real> ctmp;

        for (int ch=0; ch<2; ++ch) {
          // factor of two is inherent to QUDA clover storage
          for (int i=0; i<6; i++) diag[ch][i] = 2.0*A_array[ch*36+i];
          for (int i=0; i<15; i++) tri[ch][idtab[i]] = complex<real>(2.0*A_array[ch*36+6+2*i], 2.0*A_array[ch*36+6+2*i+1]);
        }

        // X, Y
        if (nu == 0) {
          if (mu == 1) {
            for (int j=0; j<3; ++j) {
              mat(j,j).y = diag[0][j+3] + diag[1][j+3] - diag[0][j] - diag[1][j];
            }

            // triangular part
            int jk=0;
            for (int j=1; j<3; ++j) {
              int jk2 = (j+3)*(j+2)/2 + 3;
              for (int k=0; k<j; ++k) {
                ctmp = tri[0][jk2] + tri[1][jk2] - tri[0][jk] - tri[1][jk];

                mat(j,k).x = -ctmp.imag();
                mat(j,k).y =  ctmp.real();
                mat(k,j).x =  ctmp.imag();
                mat(k,j).y =  ctmp.real();

                jk++; jk2++;
              }
            } // X Y

          } else if (mu == 2) {

            for (int j=0; j<3; ++j) {
              int jk = (j+3)*(j+2)/2;
              for (int k=0; k<3; ++k) {
                int kj = (k+3)*(k+2)/2 + j;
                mat(j,k) = conj(tri[0][kj]) - tri[0][jk] + conj(tri[1][kj]) - tri[1][jk];
                jk++;
              }
            } // X Z

          } else if (mu == 3) {
            for (int j=0; j<3; ++j) {
              int jk = (j+3)*(j+2)/2;
              for (int k=0; k<3; ++k) {
                int kj = (k+3)*(k+2)/2 + j;
                ctmp = conj(tri[0][kj]) + tri[0][jk] - conj(tri[1][kj]) - tri[1][jk];
                mat(j,k).x = -ctmp.imag();
                mat(j,k).y =  ctmp.real();
                jk++;
              }
            }
          } // mu == 3 // X T
        } else if (nu == 1) {
          if (mu == 2) { // Y Z
            for (int j=0; j<3; ++j) {
              int jk = (j+3)*(j+2)/2;
              for (int k=0; k<3; ++k) {
                int kj = (k+3)*(k+2)/2 + j;
                ctmp = conj(tri[0][kj]) + tri[0][jk] + conj(tri[1][kj]) + tri[1][jk];
                mat(j,k).x =  ctmp.imag();
                mat(j,k).y = -ctmp.real();
                jk++;
              }
            }
          } else if (mu == 3){ // Y T
            for (int j=0; j<3; ++j) {
              int jk = (j+3)*(j+2)/2;
              for (int k=0; k<3; ++k) {
                int kj = (k+3)*(k+2)/2 + j;
                mat(j,k) = conj(tri[0][kj]) - tri[0][jk] - conj(tri[1][kj]) + tri[1][jk];
                jk++;
              }
            }
          } // mu == 3
        } // nu == 1
        else if (nu == 2){
          if (mu == 3) {
            for (int j=0; j<3; ++j) {
              mat(j,j).y = diag[0][j] - diag[0][j+3] - diag[1][j] + diag[1][j+3];
            }
            int jk=0;
            for (int j=1; j<3; ++j) {
              int jk2 = (j+3)*(j+2)/2 + 3;
              for (int k=0; k<j; ++k) {
                ctmp = tri[0][jk] - tri[0][jk2] - tri[1][jk] + tri[1][jk2];
                mat(j,k).x = -ctmp.imag();
                mat(j,k).y =  ctmp.real();

                mat(k,j).x = ctmp.imag();
                mat(k,j).y = ctmp.real();
                jk++; jk2++;
              }
            }
          }
        }

        mat *= arg.coeff;
        arg.output((mu-1)*mu/2 + nu, x, arg.parity) = mat;
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
