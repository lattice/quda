#pragma once

#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <kernel.h>

namespace quda {

  template <typename Float, int nColor_>
  struct CloverTraceArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    using C = typename clover_mapper<Float>::type;
    using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
    G output;
    const C clover1;
    const C clover2;
    real coeff;

    CloverTraceArg(GaugeField& output, const CloverField& clover, double coeff) :
      kernel_param(dim3(clover.VolumeCB(), 1, 1)),
      output(output),
      clover1(clover, 0),
      clover2(clover, 1),
      coeff(coeff) {}
  };

  template <typename Arg>
  __device__ __host__ void cloverSigmaTraceCompute(const Arg &arg, const int x, int parity)
  {
    using real = typename Arg::real;
    real A[72];
    if (parity==0) arg.clover1.load(A,x,parity);
    else arg.clover2.load(A,x,parity);

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
          for (int i=0; i<6; i++) diag[ch][i] = 2.0*A[ch*36+i];
          for (int i=0; i<15; i++) tri[ch][idtab[i]] = complex<real>(2.0*A[ch*36+6+2*i], 2.0*A[ch*36+6+2*i+1]);
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
        arg.output((mu-1)*mu/2 + nu, x, parity) = mat;
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
      // odd parity
      cloverSigmaTraceCompute<Arg>(arg, x_cb, 1);
    }
  };

}
