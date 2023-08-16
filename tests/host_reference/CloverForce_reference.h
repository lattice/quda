#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "misc.h"
#include <color_spinor_field.h> // convenient quark field container
#include <command_line_params.h>
#include <dslash_reference.h>
#include <gauge_field.h>
#include <host_utils.h>
#include <quda.h>
// #include <quda_matrix.h>
// #include <complex_quda.h>
#include <Eigen/Dense>

// FIXME: this was copied from  wilson_dslash_reference.cpp maybe it is better to create a separate file with the projection
// clang-format off
static const double projector[8][4][4][2] = {
  {
    {{1,0}, {0,0}, {0,0}, {0,-1}},
    {{0,0}, {1,0}, {0,-1}, {0,0}},
    {{0,0}, {0,1}, {1,0}, {0,0}},
    {{0,1}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {0,1}},
    {{0,0}, {1,0}, {0,1}, {0,0}},
    {{0,0}, {0,-1}, {1,0}, {0,0}},
    {{0,-1}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {1,0}},
    {{0,0}, {1,0}, {-1,0}, {0,0}},
    {{0,0}, {-1,0}, {1,0}, {0,0}},
    {{1,0}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {-1,0}},
    {{0,0}, {1,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {1,0}, {0,0}},
    {{-1,0}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,-1}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {0,1}},
    {{0,1}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {0,-1}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,1}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {0,-1}},
    {{0,-1}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {0,1}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {-1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {-1,0}},
    {{-1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {-1,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {1,0}},
    {{1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {1,0}}
  }
};
// clang-format on

// todo pass projector
template <typename Float> void multiplySpinorByDiracProjector(Float *res, int projIdx, const Float *spinorIn)
{
  for (int i = 0; i < 4 * 3 * 2; i++) res[i] = 0.0;

  for (int s = 0; s < 4; s++) {
    for (int t = 0; t < 4; t++) {
      Float projRe = projector[projIdx][s][t][0];
      Float projIm = projector[projIdx][s][t][1];

      for (int m = 0; m < 3; m++) {
        Float spinorRe = spinorIn[t * (3 * 2) + m * (2) + 0];
        Float spinorIm = spinorIn[t * (3 * 2) + m * (2) + 1];
        res[s * (3 * 2) + m * (2) + 0] += projRe * spinorRe - projIm * spinorIm;
        res[s * (3 * 2) + m * (2) + 1] += projRe * spinorIm + projIm * spinorRe;
      }
    }
  }
}
template <typename sFloat, typename gFloat> void outerProdSpinTrace(gFloat *gauge, sFloat *x, sFloat *y)
{

  // outer product over color

  for (int i = 0; i < 3; i++) {

    for (int j = 0; j < 3; j++) {
      gauge[j * 6 + i * 2 + 0] = x[0 * 6 + j * 2 + 0] * y[0 * 6 + i * 2 + 0];
      gauge[j * 6 + i * 2 + 0] += x[0 * 6 + j * 2 + 1] * y[0 * 6 + i * 2 + 1];
      gauge[j * 6 + i * 2 + 1] = x[0 * 6 + j * 2 + 1] * y[0 * 6 + i * 2 + 0];
      gauge[j * 6 + i * 2 + 1] -= x[0 * 6 + j * 2 + 0] * y[0 * 6 + i * 2 + 1];
      // trace over spin (manual unroll for perf)
      // out(j, i).real(a(0, j).real() * b(0, i).real());
      // out(j, i).real(out(j, i).real() + a(0, j).imag() * b(0, i).imag());
      // out(j, i).imag(a(0, j).imag() * b(0, i).real());
      // out(j, i).imag(out(j, i).imag() - a(0, j).real() * b(0, i).imag());

      for (int s = 1; s < 4; s++) {
        gauge[j * 6 + i * 2 + 0] += x[s * 6 + j * 2 + 0] * y[s * 6 + i * 2 + 0];
        gauge[j * 6 + i * 2 + 0] += x[s * 6 + j * 2 + 1] * y[s * 6 + i * 2 + 1];
        gauge[j * 6 + i * 2 + 1] += x[s * 6 + j * 2 + 1] * y[s * 6 + i * 2 + 0];
        gauge[j * 6 + i * 2 + 1] -= x[s * 6 + j * 2 + 0] * y[s * 6 + i * 2 + 1];
        //   out(j,i).real( out(j,i).real() + a(s,j).real() * b(s,i).real() );
        //   out(j,i).real( out(j,i).real() + a(s,j).imag() * b(s,i).imag() );
        //   out(j,i).imag( out(j,i).imag() + a(s,j).imag() * b(s,i).real() );
        //   out(j,i).imag( out(j,i).imag() - a(s,j).real() * b(s,i).imag() );
      }
    }
  }
}

template <typename gFloat> void accum_su3xsu3(gFloat *mom, gFloat *gauge, gFloat *oprod, double coeff)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        mom[j * 6 + i * 2 + 0] += coeff * gauge[j * 6 + k * 2 + 0] * oprod[k * 6 + i * 2 + 0];
        mom[j * 6 + i * 2 + 0] -= coeff * gauge[j * 6 + k * 2 + 1] * oprod[k * 6 + i * 2 + 1];
        mom[j * 6 + i * 2 + 1] += coeff * gauge[j * 6 + k * 2 + 1] * oprod[k * 6 + i * 2 + 0];
        mom[j * 6 + i * 2 + 1] += coeff * gauge[j * 6 + k * 2 + 0] * oprod[k * 6 + i * 2 + 1];
      }
    }
  }
}

template <typename gFloat> void accum_su3_to_anti_hermitian(gFloat *mom, gFloat *gauge)
{
  auto temp = (gauge[0 * 6 + 0 * 2 + 1] + gauge[1 * 6 + 1 * 2 + 1] + gauge[2 * 6 + 2 * 2 + 1]) * 0.33333333333333333;
  mom[6] += gauge[0 * 6 + 0 * 2 + 1] - temp;
  mom[7] += gauge[1 * 6 + 1 * 2 + 1] - temp;
  mom[8] += gauge[2 * 6 + 2 * 2 + 1] - temp;
  // of diag
  mom[0] += (gauge[0 * 6 + 1 * 2 + 0] - gauge[1 * 6 + 0 * 2 + 0]) * 0.5;
  mom[1] += (gauge[0 * 6 + 1 * 2 + 1] + gauge[1 * 6 + 0 * 2 + 1]) * 0.5;
  mom[2] += (gauge[0 * 6 + 2 * 2 + 0] - gauge[2 * 6 + 0 * 2 + 0]) * 0.5;
  mom[3] += (gauge[0 * 6 + 2 * 2 + 1] + gauge[2 * 6 + 0 * 2 + 1]) * 0.5;
  mom[4] += (gauge[1 * 6 + 2 * 2 + 0] - gauge[2 * 6 + 1 * 2 + 0]) * 0.5;
  mom[5] += (gauge[1 * 6 + 2 * 2 + 1] + gauge[2 * 6 + 1 * 2 + 1]) * 0.5;
}

template <typename sFloat, typename gFloat>
void CloverForce_kernel_host(std::array<void *, 4> gauge, void *h_mom, quda::ColorSpinorField &inA,
                             quda::ColorSpinorField &inB, int projSign, int parity, double force_coeff)
{

  gFloat **gaugeFull = (gFloat **)gauge.data();
  sFloat **backSpinor = (sFloat **)inB.backGhostFaceBuffer;
  sFloat **fwdSpinor = (sFloat **)inB.fwdGhostFaceBuffer;
  sFloat *spinorField = (sFloat *)inB.V();

  gFloat *gaugeEven[4], *gaugeOdd[4];

  sFloat *A = (sFloat *)inA.V();

  for (int dir = 0; dir < 4; dir++) {
    gaugeEven[dir] = gaugeFull[dir];
    gaugeOdd[dir] = gaugeFull[dir] + Vh * gauge_site_size;
  }

  for (int i = 0; i < Vh; i++) {
    // loop over the forward directions
    for (int dir = 0; dir < 8; dir += 2) {
      // load the gauge
      gFloat **gaugeField = (parity ? gaugeOdd : gaugeEven);
      gFloat *gauge = &gaugeField[dir / 2][i * (3 * 3 * 2)];
      // load shifted spinor and project
      const sFloat *spinor = spinorNeighbor_mg4dir(i, dir, parity, spinorField, fwdSpinor, backSpinor, 1, 1);
      sFloat projectedSpinor[spinor_site_size];
      int projIdx = 2 * (dir / 2) + (projSign + 1) / 2; //+ (dir + daggerBit) % 2;
      multiplySpinorByDiracProjector(projectedSpinor, projIdx, spinor);

      gFloat oprod[gauge_site_size];
      outerProdSpinTrace(oprod, projectedSpinor, &A[i * spinor_site_size]);

      gFloat force[gauge_site_size];
      for (size_t j = 0; j < gauge_site_size; j++) force[j] = 0;
      accum_su3xsu3(force, gauge, oprod, force_coeff);
      int mu = (dir / 2);
      gFloat *mom = (gFloat *)h_mom + (4 * (i + Vh * parity) + mu) * mom_site_size;
      accum_su3_to_anti_hermitian(mom, force);
    }
  }
}

void CloverForce_reference(void *h_mom, std::array<void *, 4> gauge, quda::ColorSpinorField &x,
                           quda::ColorSpinorField &p, double force_coeff)
{
  int dag = 1;
  for (int parity = 0; parity < 2; parity++) {
    quda::ColorSpinorField &inA = (parity & 1) ? p.Odd() : p.Even();
    quda::ColorSpinorField &inB = (parity & 1) ? x.Even() : x.Odd();
    quda::ColorSpinorField &inC = (parity & 1) ? x.Odd() : x.Even();
    quda::ColorSpinorField &inD = (parity & 1) ? p.Even() : p.Odd();

    static constexpr int nFace = 1;
    // every time that exchange ghost is called fwdGhostFaceBuffer becomes the Ghost of the last spinor called
    inB.exchangeGhost((QudaParity)(1 - parity), nFace, dag);
    CloverForce_kernel_host<double, double>(gauge, h_mom, inA, inB, 1, parity, force_coeff);
    inD.exchangeGhost((QudaParity)(1 - parity), nFace, 1 - dag);
    CloverForce_kernel_host<double, double>(gauge, h_mom, inC, inD, -1, parity, force_coeff);
  }
}
template <typename cFloat>
void cloverSigmaTraceCompute_host(cFloat **oprod, cFloat *clover, double coeff, int parity, double mu2, double eps2)
{
  int nSpin = 4;
  int nColor = 3;
  int N = nColor * nSpin / 2;
  int chiralBlock = N + 2 * (N - 1) * N / 2;

  cFloat *oprodEven[6], *oprodOdd[6];
  // cFloat *A_array; //[72];
  cFloat A_array[72];
  typedef Eigen::Matrix<std::complex<cFloat>, 3, 3> Matrix3c;
  typedef Eigen::Matrix<std::complex<cFloat>, 6, 6> CloverM;
  // if (dynamic_clover) {
  // }
  for (int i = 0; i < Vh; i++) {
    for (int chirality = 0; chirality < 2; chirality++) {
      // the cover filed for a given chirality is stored as
      // N real numbers: the diagonal part D
      // Then the off diagonal part is stored in in the complex array L
      // (----       L[0]       L[1]        L[2]         L[3]         L[4]       )
      // (           ----       L[5=N-1]    L[6]         L[7]         L[8]       )
      // (                      ----        L[9=2N-3]    L[10]        L[11]      )
      // (                                  ----         L[12=3N-6]   L[13]      )
      // (                                               ----         L[14=4N-10])
      // (                                                            ----       )
      // cFloat *D = &clover[((parity * Vh + i) * 2 + chirality) * chiralBlock];
      // std::complex<cFloat> *L = reinterpret_cast<std::complex<cFloat> *>(&D[N]);

      // A_array = &clover[((parity * Vh + i) * 2 + chirality) * chiralBlock];
      for (int j = 0; j < 36; j++)
        A_array[chirality * 36 + j] = clover[((parity * Vh + i) * 2 + chirality) * chiralBlock + j];

      CloverM A;
      int index = ((parity * Vh + i) * 2 + chirality) * chiralBlock;
      // diag
      for (int j = 0; j < 6; j++) {
        A(j, j).real(clover[index + j]);
        A(j, j).imag(0);
      }
      // off-diag
      for (int row = 0; row < 6; row++) {
        for (int col = (row + 1); col < 6; col++) {
          // int id = N * (N - 1) / 2 - (N - row) * (N - row - 1) / 2 + (col - row - 1);
          int id = N * row - (row * (row + 1)) / 2 + (col - row - 1);
          A(row, col).real(clover[index + 6 + id * 2]);
          A(row, col).imag(clover[index + 6 + id * 2 + 1]);
        }
      }

      for (int row = 0; row < 6; row++) {
        for (int j = 0; j < row; j++) { A(row, j) = conj(A(j, row)); }
      }
      CloverM B = A * A;
      for (int j = 0; j < 6; j++) B(j, j) = B(j, j) + mu2 - eps2;

      B = 0.5 * B.inverse();
      A = 0.25 * A * B;

      for (int row = 0; row < 6; row++) {
        A_array[chirality * 36 + row] = A(row, row).real();
        for (int col = (row + 1); col < 6; col++) {
          int id = N * row - (row * (row + 1)) / 2 + (col - row - 1);
          A_array[chirality * 36 + 6 + id * 2] = A(row, col).real();
          A_array[chirality * 36 + 6 + id * 2 + 1] = A(row, col).imag();
        }
      }

    }
    for (int mu = 0; mu < 4; mu++) {
      for (int nu = 0; nu < mu; nu++) {
        // oprod is stored only for nu<mu (6 indices) as
        // (---                              )
        // (oprod[0]  ---                    )
        // (oprod[1]  oprod[2]  ---          )
        // (oprod[3]  oprod[4]  oprod[5]  ---)
        // the full lexicographic index of oprod is
        // (mu - 1) * mu / 2 + nu + 6*( re/im + (row_color+ col_color*Ncolor)*2 + quda_idx)
        // quda_idx = 18*(oddBit*(T+LX+LY+LZ)/2+x/2)
        // x = x1 + LX*x2 + LY*LX*x3 + LZ*LY*LX*x0;
        oprodEven[(mu - 1) * mu / 2 + nu] = oprod[(mu - 1) * mu / 2 + nu];
        oprodOdd[(mu - 1) * mu / 2 + nu] = oprod[(mu - 1) * mu / 2 + nu] + Vh * gauge_site_size;
        cFloat **oprod_eo = (parity ? oprodOdd : oprodEven);
        cFloat *out_munu = &oprod_eo[(mu - 1) * mu / 2 + nu][i * (3 * 3 * 2)];

        Matrix3c mat = Matrix3c::Zero();
        // quda::Matrix<quda::complex<cFloat>, 3> mat;
        // setZero(&mat);
        cFloat diag[2][6];
        std::complex<cFloat> tri[2][15];
        const int idtab[15] = {0, 1, 3, 6, 10, 2, 4, 7, 11, 5, 8, 12, 9, 13, 14};
        std::complex<cFloat> ctmp;

        for (int ch = 0; ch < 2; ++ch) {
          // factor of two is inherent to QUDA clover storage
          for (int i = 0; i < 6; i++) diag[ch][i] = 2.0 * A_array[ch * 36 + i];
          for (int i = 0; i < 15; i++)
            tri[ch][idtab[i]]
              = std::complex<cFloat>(2.0 * A_array[ch * 36 + 6 + 2 * i], 2.0 * A_array[ch * 36 + 6 + 2 * i + 1]);
        }

        // X, Y
        if (nu == 0) {
          if (mu == 1) {
            for (int j = 0; j < 3; ++j) { mat(j, j).imag(diag[0][j + 3] + diag[1][j + 3] - diag[0][j] - diag[1][j]); }

            // triangular part
            int jk = 0;
            for (int j = 1; j < 3; ++j) {
              int jk2 = (j + 3) * (j + 2) / 2 + 3;
              for (int k = 0; k < j; ++k) {
                ctmp = tri[0][jk2] + tri[1][jk2] - tri[0][jk] - tri[1][jk];

                mat(j, k).real(-ctmp.imag());
                mat(j, k).imag(ctmp.real());
                mat(k, j).real(ctmp.imag());
                mat(k, j).imag(ctmp.real());

                jk++;
                jk2++;
              }
            } // X Y

          } else if (mu == 2) {

            for (int j = 0; j < 3; ++j) {
              int jk = (j + 3) * (j + 2) / 2;
              for (int k = 0; k < 3; ++k) {
                int kj = (k + 3) * (k + 2) / 2 + j;
                mat(j, k) = conj(tri[0][kj]) - tri[0][jk] + conj(tri[1][kj]) - tri[1][jk];
                jk++;
              }
            } // X Z

          } else if (mu == 3) {
            for (int j = 0; j < 3; ++j) {
              int jk = (j + 3) * (j + 2) / 2;
              for (int k = 0; k < 3; ++k) {
                int kj = (k + 3) * (k + 2) / 2 + j;
                ctmp = conj(tri[0][kj]) + tri[0][jk] - conj(tri[1][kj]) - tri[1][jk];
                mat(j, k).real(-ctmp.imag());
                mat(j, k).imag(ctmp.real());
                jk++;
              }
            }
          } // mu == 3 // X T
        } else if (nu == 1) {
          if (mu == 2) { // Y Z
            for (int j = 0; j < 3; ++j) {
              int jk = (j + 3) * (j + 2) / 2;
              for (int k = 0; k < 3; ++k) {
                int kj = (k + 3) * (k + 2) / 2 + j;
                ctmp = conj(tri[0][kj]) + tri[0][jk] + conj(tri[1][kj]) + tri[1][jk];
                mat(j, k).real(ctmp.imag());
                mat(j, k).imag(-ctmp.real());
                jk++;
              }
            }
          } else if (mu == 3) { // Y T
            for (int j = 0; j < 3; ++j) {
              int jk = (j + 3) * (j + 2) / 2;
              for (int k = 0; k < 3; ++k) {
                int kj = (k + 3) * (k + 2) / 2 + j;
                mat(j, k) = conj(tri[0][kj]) - tri[0][jk] - conj(tri[1][kj]) + tri[1][jk];
                jk++;
              }
            }
          } // mu == 3
        }   // nu == 1
        else if (nu == 2) {
          if (mu == 3) {
            for (int j = 0; j < 3; ++j) { mat(j, j).imag(diag[0][j] - diag[0][j + 3] - diag[1][j] + diag[1][j + 3]); }
            int jk = 0;
            for (int j = 1; j < 3; ++j) {
              int jk2 = (j + 3) * (j + 2) / 2 + 3;
              for (int k = 0; k < j; ++k) {
                ctmp = tri[0][jk] - tri[0][jk2] - tri[1][jk] + tri[1][jk2];
                mat(j, k).real(-ctmp.imag());
                mat(j, k).imag(ctmp.real());

                mat(k, j).real(ctmp.imag());
                mat(k, j).imag(ctmp.real());
                jk++;
                jk2++;
              }
            }
          }
        }

        mat *= coeff;
        // arg.output((mu-1)*mu/2 + nu, x, arg.parity) = mat;

        for (int ci = 0; ci < nColor; ci++) {
          for (int cj = 0; cj < nColor; cj++) {
            out_munu[ci * nColor * 2 + cj * 2 + 0] = mat(ci, cj).real();
            out_munu[ci * nColor * 2 + cj * 2 + 1] = mat(ci, cj).imag();
          }
        }

      } // nu
    }   // mu
  }
}

void computeCloverSigmaTrace_reference(void *oprod, void *clover, double coeff, int parity, double mu2, double eps2)
{

  // here call the appropriate template function according to gauge_precision
  cloverSigmaTraceCompute_host((double **)oprod, (double *)clover, coeff, parity, mu2, eps2);
}
