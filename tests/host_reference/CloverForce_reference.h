#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gauge_force_reference.h"
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

static const double local_gamma[4][4][4][2] = {{// x
                                                {{0, 0}, {0, 0}, {0, 0}, {0, -1}},
                                                {{0, 0}, {0, 0}, {0, -1}, {0, 0}},
                                                {{0, 0}, {0, 1}, {0, 0}, {0, 0}},
                                                {{0, 1}, {0, 0}, {0, 0}, {0, 0}}},
                                               {// Y
                                                {{0, 0}, {0, 0}, {0, 0}, {1, 0}},
                                                {{0, 0}, {0, 0}, {-1, 0}, {0, 0}},
                                                {{0, 0}, {-1, 0}, {0, 0}, {0, 0}},
                                                {{1, 0}, {0, 0}, {0, 0}, {0, 0}}},
                                               {// Z
                                                {{0, 0}, {0, 0}, {0, -1}, {0, 0}},
                                                {{0, 0}, {0, 0}, {0, 0}, {0, 1}},
                                                {{0, 1}, {0, 0}, {0, 0}, {0, 0}},
                                                {{0, 0}, {0, -1}, {0, 0}, {0, 0}}},
                                               {// T
                                                {{0, 0}, {0, 0}, {-1, 0}, {0, 0}},
                                                {{0, 0}, {0, 0}, {0, 0}, {-1, 0}},
                                                {{-1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                                {{0, 0}, {-1, 0}, {0, 0}, {0, 0}}}};
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

// todo pass gamma
template <typename Float> void multiplySpinorByDiracgamma(Float *res, int gammaIdx, const Float *spinorIn)
{
  for (int i = 0; i < 4 * 3 * 2; i++) res[i] = 0.0;

  for (int s = 0; s < 4; s++) {
    for (int t = 0; t < 4; t++) {
      Float projRe = local_gamma[gammaIdx][s][t][0];
      Float projIm = local_gamma[gammaIdx][s][t][1];

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

template <typename gFloat> void mult_su3xsu3(gFloat *mom, gFloat *gauge, gFloat *oprod, double coeff)
{
  for (size_t i = 0; i < gauge_site_size; i++) mom[i] = 0;
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

template <typename gFloat> void mult_su3xsu3dag(gFloat *mom, gFloat *gauge, gFloat *oprod, double coeff)
{
  for (size_t i = 0; i < gauge_site_size; i++) mom[i] = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        mom[j * 6 + i * 2 + 0] += coeff * gauge[j * 6 + k * 2 + 0] * oprod[i * 6 + k * 2 + 0];
        mom[j * 6 + i * 2 + 0] += coeff * gauge[j * 6 + k * 2 + 1] * oprod[i * 6 + k * 2 + 1];
        mom[j * 6 + i * 2 + 1] += coeff * gauge[j * 6 + k * 2 + 1] * oprod[i * 6 + k * 2 + 0];
        mom[j * 6 + i * 2 + 1] -= coeff * gauge[j * 6 + k * 2 + 0] * oprod[i * 6 + k * 2 + 1];
      }
    }
  }
}
template <typename gFloat> void mult_dagsu3xsu3(gFloat *mom, gFloat *gauge, gFloat *oprod, double coeff)
{
  for (size_t i = 0; i < gauge_site_size; i++) mom[i] = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        mom[j * 6 + i * 2 + 0] += coeff * gauge[k * 6 + j * 2 + 0] * oprod[k * 6 + i * 2 + 0];
        mom[j * 6 + i * 2 + 0] += coeff * gauge[k * 6 + j * 2 + 1] * oprod[k * 6 + i * 2 + 1];
        mom[j * 6 + i * 2 + 1] -= coeff * gauge[k * 6 + j * 2 + 1] * oprod[k * 6 + i * 2 + 0];
        mom[j * 6 + i * 2 + 1] += coeff * gauge[k * 6 + j * 2 + 0] * oprod[k * 6 + i * 2 + 1];
      }
    }
  }
}

template <typename gFloat> void accum_su3_to_anti_hermitian(gFloat *mom, gFloat *gauge, int sign = 1)
{
  auto temp = (gauge[0 * 6 + 0 * 2 + 1] + gauge[1 * 6 + 1 * 2 + 1] + gauge[2 * 6 + 2 * 2 + 1]) * 0.33333333333333333;
  mom[6] += sign * (gauge[0 * 6 + 0 * 2 + 1] - temp);
  mom[7] += sign * (gauge[1 * 6 + 1 * 2 + 1] - temp);
  mom[8] += sign * (gauge[2 * 6 + 2 * 2 + 1] - temp);
  // of diag
  mom[0] += sign * (gauge[0 * 6 + 1 * 2 + 0] - gauge[1 * 6 + 0 * 2 + 0]) * 0.5;
  mom[1] += sign * (gauge[0 * 6 + 1 * 2 + 1] + gauge[1 * 6 + 0 * 2 + 1]) * 0.5;
  mom[2] += sign * (gauge[0 * 6 + 2 * 2 + 0] - gauge[2 * 6 + 0 * 2 + 0]) * 0.5;
  mom[3] += sign * (gauge[0 * 6 + 2 * 2 + 1] + gauge[2 * 6 + 0 * 2 + 1]) * 0.5;
  mom[4] += sign * (gauge[1 * 6 + 2 * 2 + 0] - gauge[2 * 6 + 1 * 2 + 0]) * 0.5;
  mom[5] += sign * (gauge[1 * 6 + 2 * 2 + 1] + gauge[2 * 6 + 1 * 2 + 1]) * 0.5;
}
// a= b-b^dag
template <typename gFloat> void su3_imagx2(gFloat *a, gFloat *b)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      a[j * 6 + i * 2 + 0] = b[j * 6 + i * 2 + 0] - b[i * 6 + j * 2 + 0];
      a[j * 6 + i * 2 + 1] = b[j * 6 + i * 2 + 1] + b[i * 6 + j * 2 + 1];
    }
  }
}

template <typename sFloat, typename gFloat>
void CloverForce_kernel_host(std::array<void *, 4> gauge, void *h_mom, quda::ColorSpinorField &inA,
                             quda::ColorSpinorField &inB, int projSign, int parity, double force_coeff)
{

  gFloat **gaugeFull = (gFloat **)gauge.data();
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
#ifndef MULTI_GPU
      const sFloat *spinor = spinorNeighbor(i, dir, parity, spinorField, 1);
#else
      sFloat **backSpinor = (sFloat **)inB.backGhostFaceBuffer;
      sFloat **fwdSpinor = (sFloat **)inB.fwdGhostFaceBuffer;
      const sFloat *spinor = spinorNeighbor_mg4dir(i, dir, parity, spinorField, fwdSpinor, backSpinor, 1, 1);
#endif
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
void cloverSigmaTraceCompute_host(cFloat *oprod, cFloat *clover, double coeff, int parity, double mu2, double eps2)
{
  int nSpin = 4;
  int nColor = 3;
  int N = nColor * nSpin / 2;
  int chiralBlock = N + 2 * (N - 1) * N / 2;

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
        // = reim + 2 * (x_eo / 2 + (V/2) * (color + 9 * (munu + parity * 6)))
        // munu = (mu - 1) * mu / 2 + nu
        // color = col_color+ row_color*Ncolor

        Matrix3c mat = Matrix3c::Zero();
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

        int munu = (mu - 1) * mu / 2 + nu;
        for (int ci = 0; ci < nColor; ci++) {   // row
          for (int cj = 0; cj < nColor; cj++) { // col
            int color = ci * nColor + cj;
            int id = 2 * (i + Vh * (color + 9 * (munu + parity * 6)));
            oprod[id + 0] += mat(ci, cj).real();
            oprod[id + 1] += mat(ci, cj).imag();
          }
        }

      } // nu
    }   // mu
  }
}

void computeCloverSigmaTrace_reference(void *oprod, void *clover, double coeff, int parity, double mu2, double eps2)
{

  // FIXME: here call the appropriate template function according to gauge_precision
  cloverSigmaTraceCompute_host((double *)oprod, (double *)clover, coeff, parity, mu2, eps2);
}
template <typename gFloat>
void get_su3FromOprod(gFloat *oprod_out, gFloat *oprod, int munu, size_t nbr_idx, const lattice_t &lat)
{

  int x_cb = nbr_idx % (lat.volume_ex / 2);
  int OddBit = nbr_idx / (lat.volume_ex / 2);

  for (int i = 0; i < 3; i++) {   // col
    for (int j = 0; j < 3; j++) { // row
      int color = i + j * 3;
      int id = 2 * (x_cb + (lat.volume_ex / 2) * (color + 9 * (munu + OddBit * 6)));
      oprod_out[j * 6 + i * 2 + 0] = oprod[id + 0];
      oprod_out[j * 6 + i * 2 + 1] = oprod[id + 1];
    }
  }
}

template <typename gFloat>
void computeForce_reference(void *h_mom_, void **gauge_ex, lattice_t lat, void *oprod_, int i, int yIndex, int parity,
                            int mu, int nu)
{
  gFloat *oprod = (gFloat *)oprod_;

  int acc_parity = yIndex == 0 ? parity : 1 - parity;
  gFloat *mom = (gFloat *)h_mom_ + (4 * (i + Vh * acc_parity) + mu) * mom_site_size;

  gFloat **gaugeFull_ex = (gFloat **)gauge_ex;

  int otherparity = (1 - parity);
  const int tidx = mu > nu ? (mu - 1) * mu / 2 + nu : (nu - 1) * nu / 2 + mu;
  gFloat su3tmp1[gauge_site_size], su3tmp2[gauge_site_size];

  if (yIndex == 0) { // do "this" force

    // U[mu](x) U[nu](x+mu) U[*mu](x+nu) U[*nu](x) Oprod(x)
    {
      int d[4] = {0, 0, 0, 0};
      int nbr_idx;
      int eo_full_id = i + parity * Vh ;
      // load U(x)_(+mu)
      // Link U1 = arg.gauge(mu, linkIndexShift(x, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U1 = gaugeFull_ex[mu] + nbr_idx * (3 * 3 * 2);

      // load U(x+mu)_(+nu)
      d[mu]++;
      // Link U2 = arg.gauge(nu, linkIndexShift(x, d, arg.E), otherparity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U2 = gaugeFull_ex[nu] + nbr_idx * (3 * 3 * 2);
      d[mu]--;

      // load U(x+nu)_(+mu)
      d[nu]++;
      // Link U3 = arg.gauge(mu, linkIndexShift(x, d, arg.E), otherparity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U3 = gaugeFull_ex[mu] + nbr_idx * (3 * 3 * 2);
      d[nu]--;

      // load U(x)_(+nu)
      // Link U4 = arg.gauge(nu, linkIndexShift(x, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U4 = gaugeFull_ex[nu] + nbr_idx * (3 * 3 * 2);

      // load Oprod
      // Link Oprod1 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat Oprod1[gauge_site_size];
      get_su3FromOprod(Oprod1, oprod, tidx, nbr_idx, lat);

      // if (nu < mu)
      //   force -= U1 * U2 * conj(U3) * conj(U4) * Oprod1;
      // else
      //   force += U1 * U2 * conj(U3) * conj(U4) * Oprod1;
      mult_dagsu3xsu3(su3tmp1, U4, Oprod1, 1);
      mult_dagsu3xsu3(su3tmp2, U3, su3tmp1, 1);
      mult_su3xsu3(su3tmp1, U2, su3tmp2, 1);
      mult_su3xsu3(su3tmp2, U1, su3tmp1, 1);
      if (nu < mu)
        accum_su3_to_anti_hermitian(mom, su3tmp2, -1);
      else
        accum_su3_to_anti_hermitian(mom, su3tmp2);

      d[mu]++;
      d[nu]++;
      // Link Oprod2 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat Oprod2[gauge_site_size];
      get_su3FromOprod(Oprod2, oprod, tidx, nbr_idx, lat);
      d[mu]--;
      d[nu]--;

      // if (nu < mu)
      //   force -= U1 * U2 * Oprod2 * conj(U3) * conj(U4);
      // else
      //   force += U1 * U2 * Oprod2 * conj(U3) * conj(U4);
      mult_su3xsu3(su3tmp1, U4, U3, 1);
      mult_su3xsu3dag(su3tmp2, Oprod2, su3tmp1, 1);
      mult_su3xsu3(su3tmp1, U2, su3tmp2, 1);
      mult_su3xsu3(su3tmp2, U1, su3tmp1, 1);
      if (nu < mu)
        accum_su3_to_anti_hermitian(mom, su3tmp2, -1);
      else
        accum_su3_to_anti_hermitian(mom, su3tmp2);
    }

    {
      int d[4] = {0, 0, 0, 0};
      int nbr_idx;
      int eo_full_id = i + parity * Vh ;

      // load U(x-nu)(+nu)
      d[nu]--;
      // Link U1 = arg.gauge(nu, linkIndexShift(x, d, arg.E), otherparity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U1 = gaugeFull_ex[nu] + nbr_idx * (3 * 3 * 2);
      d[nu]++;

      // load U(x-nu)(+mu)
      d[nu]--;
      // Link U2 = arg.gauge(mu, linkIndexShift(x, d, arg.E), otherparity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U2 = gaugeFull_ex[mu] + nbr_idx * (3 * 3 * 2);
      d[nu]++;

      // load U(x+mu-nu)(nu)
      d[mu]++;
      d[nu]--;
      // Link U3 = arg.gauge(nu, linkIndexShift(x, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U3 = gaugeFull_ex[nu] + nbr_idx * (3 * 3 * 2);
      d[mu]--;
      d[nu]++;

      // load U(x)_(+mu)
      // Link U4 = arg.gauge(mu, linkIndexShift(x, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U4 = gaugeFull_ex[mu] + nbr_idx * (3 * 3 * 2);

      d[mu]++;
      d[nu]--;
      // Link Oprod1 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat Oprod1[gauge_site_size];
      get_su3FromOprod(Oprod1, oprod, tidx, nbr_idx, lat);
      d[mu]--;
      d[nu]++;

      // if (nu < mu)
      //   force += conj(U1) * U2 * Oprod1 * U3 * conj(U4);
      // else
      //   force -= conj(U1) * U2 * Oprod1 * U3 * conj(U4);
      mult_su3xsu3dag(su3tmp1, U3, U4, 1);
      mult_su3xsu3(su3tmp2, Oprod1, su3tmp1, 1);
      mult_su3xsu3(su3tmp1, U2, su3tmp2, 1);
      mult_dagsu3xsu3(su3tmp2, U1, su3tmp1, 1);
      if (nu < mu)
        accum_su3_to_anti_hermitian(mom, su3tmp2);
      else
        accum_su3_to_anti_hermitian(mom, su3tmp2, -1);

      // Link Oprod4 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat Oprod4[gauge_site_size];
      get_su3FromOprod(Oprod4, oprod, tidx, nbr_idx, lat);

      // if (nu < mu)
      //   force += Oprod4 * conj(U1) * U2 * U3 * conj(U4);
      // else
      //   force -= Oprod4 * conj(U1) * U2 * U3 * conj(U4);
      mult_su3xsu3dag(su3tmp1, U3, U4, 1);
      mult_su3xsu3(su3tmp2, U2, su3tmp1, 1);
      mult_dagsu3xsu3(su3tmp1, U1, su3tmp2, 1);
      mult_su3xsu3(su3tmp2, Oprod4, su3tmp1, 1);
      if (nu < mu)
        accum_su3_to_anti_hermitian(mom, su3tmp2);
      else
        accum_su3_to_anti_hermitian(mom, su3tmp2, -1);
    }

  } else { // else do other force

    {
      int d[4] = {0, 0, 0, 0};
      int nbr_idx;
      int eo_full_id = i + otherparity * Vh ;
      // load U(x)_(+mu)
      // Link U1 = arg.gauge(mu, linkIndexShift(y, d, arg.E), otherparity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U1 = gaugeFull_ex[mu] + nbr_idx * (3 * 3 * 2);

      // load U(x+mu)_(+nu)
      d[mu]++;
      // Link U2 = arg.gauge(nu, linkIndexShift(y, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U2 = gaugeFull_ex[nu] + nbr_idx * (3 * 3 * 2);
      d[mu]--;

      // // load U(x+nu)_(+mu)
      d[nu]++;
      // Link U3 = arg.gauge(mu, linkIndexShift(y, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U3 = gaugeFull_ex[mu] + nbr_idx * (3 * 3 * 2);
      d[nu]--;

      // // load U(x)_(+nu)
      // Link U4 = arg.gauge(nu, linkIndexShift(y, d, arg.E), otherparity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U4 = gaugeFull_ex[nu] + nbr_idx * (3 * 3 * 2);
      // // load opposite parity Oprod
      d[nu]++;
      // Link Oprod3 = arg.oprod(tidx, linkIndexShift(y, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat Oprod3[gauge_site_size];
      get_su3FromOprod(Oprod3, oprod, tidx, nbr_idx, lat);
      d[nu]--;

      // if (nu < mu)
      //   force -= U1 * U2 * conj(U3) * Oprod3 * conj(U4);
      // else
      //   force += U1 * U2 * conj(U3) * Oprod3 * conj(U4);
      mult_su3xsu3dag(su3tmp1, Oprod3, U4, 1);
      mult_dagsu3xsu3(su3tmp2, U3, su3tmp1, 1);
      mult_su3xsu3(su3tmp1, U2, su3tmp2, 1);
      mult_su3xsu3(su3tmp2, U1, su3tmp1, 1);
      if (nu < mu)
        accum_su3_to_anti_hermitian(mom, su3tmp2, -1);
      else
        accum_su3_to_anti_hermitian(mom, su3tmp2);

      // load Oprod(x+mu)
      d[mu]++;
      // Link Oprod4 = arg.oprod(tidx, linkIndexShift(y, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat Oprod4[gauge_site_size];
      get_su3FromOprod(Oprod4, oprod, tidx, nbr_idx, lat);
      d[mu]--;

      // if (nu < mu)
      //   force -= U1 * Oprod4 * U2 * conj(U3) * conj(U4);
      // else
      //   force += U1 * Oprod4 * U2 * conj(U3) * conj(U4);
      // below we implemented force +-=U1 * Oprod4 * U2 * conj( U4 * U3);
      mult_su3xsu3(su3tmp1, U4, U3, 1);
      mult_su3xsu3dag(su3tmp2, U2, su3tmp1, 1);
      mult_su3xsu3(su3tmp1, Oprod4, su3tmp2, 1);
      mult_su3xsu3(su3tmp2, U1, su3tmp1, 1);
      if (nu < mu)
        accum_su3_to_anti_hermitian(mom, su3tmp2, -1);
      else
        accum_su3_to_anti_hermitian(mom, su3tmp2);
    }

    {
      int d[4] = {0, 0, 0, 0};
      int nbr_idx;
      int eo_full_id = i + otherparity * Vh ;

      // load U(x-nu)(+nu)
      d[nu]--;
      // Link U1 = arg.gauge(nu, linkIndexShift(y, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U1 = gaugeFull_ex[nu] + nbr_idx * (3 * 3 * 2);
      d[nu]++;

      // load U(x-nu)(+mu)
      d[nu]--;
      // Link U2 = arg.gauge(mu, linkIndexShift(y, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U2 = gaugeFull_ex[mu] + nbr_idx * (3 * 3 * 2);
      d[nu]++;

      // load U(x+mu-nu)(nu)
      d[mu]++;
      d[nu]--;
      // Link U3 = arg.gauge(nu, linkIndexShift(y, d, arg.E), otherparity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U3 = gaugeFull_ex[nu] + nbr_idx * (3 * 3 * 2);
      d[mu]--;
      d[nu]++;

      // load U(x)_(+mu)
      // Link U4 = arg.gauge(mu, linkIndexShift(y, d, arg.E), otherparity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat *U4 = gaugeFull_ex[mu] + nbr_idx * (3 * 3 * 2);

      // load Oprod(x+mu)
      d[mu]++;
      // Link Oprod1 = arg.oprod(tidx, linkIndexShift(y, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat Oprod1[gauge_site_size];
      get_su3FromOprod(Oprod1, oprod, tidx, nbr_idx, lat);
      d[mu]--;

      // if (nu < mu)
      //   force += conj(U1) * U2 * U3 * Oprod1 * conj(U4);
      // else
      //   force -= conj(U1) * U2 * U3 * Oprod1 * conj(U4);
      mult_su3xsu3dag(su3tmp1, Oprod1, U4, 1);
      mult_su3xsu3(su3tmp2, U3, su3tmp1, 1);
      mult_su3xsu3(su3tmp1, U2, su3tmp2, 1);
      mult_dagsu3xsu3(su3tmp2, U1, su3tmp1, 1);
      if (nu < mu)
        accum_su3_to_anti_hermitian(mom, su3tmp2);
      else
        accum_su3_to_anti_hermitian(mom, su3tmp2, -1);

      d[nu]--;
      // Link Oprod2 = arg.oprod(tidx, linkIndexShift(y, d, arg.E), arg.parity);
      nbr_idx = gf_neighborIndexFullLattice(eo_full_id, d, lat);
      gFloat Oprod2[gauge_site_size];
      get_su3FromOprod(Oprod2, oprod, tidx, nbr_idx, lat);
      d[nu]++;

      // if (nu < mu)
      //   force += conj(U1) * Oprod2 * U2 * U3 * conj(U4);
      // else
      //   force -= conj(U1) * Oprod2 * U2 * U3 * conj(U4);
      mult_su3xsu3dag(su3tmp1, U3, U4, 1);
      mult_su3xsu3(su3tmp2, U2, su3tmp1, 1);
      mult_su3xsu3(su3tmp1, Oprod2, su3tmp2, 1);
      mult_dagsu3xsu3(su3tmp2, U1, su3tmp1, 1);
      if (nu < mu)
        accum_su3_to_anti_hermitian(mom, su3tmp2);
      else
        accum_su3_to_anti_hermitian(mom, su3tmp2, -1);
    }
  }
}

void cloverDerivative_reference(void *h_mom, void **gauge, void *oprod, int parity, QudaGaugeParam &gauge_param)
{

  // created extended field
  quda::lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = 2 * quda::comm_dim_partitioned(d);
  QudaGaugeParam param = newQudaGaugeParam();
  setGaugeParam(param);
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  param.t_boundary = QUDA_PERIODIC_T;

  auto qdp_ex = quda::createExtendedGauge(gauge, param, R);
  lattice_t lat(*qdp_ex);

  /// the following does not work: segmentation fault
  // param.geometry = QUDA_TENSOR_GEOMETRY;
  // create a qdp gauge
  // std::vector<char> oprod_qdp_;
  // std::array<void *, 6> oprod_qdp;
  // oprod_qdp_.resize(6 * V * gauge_site_size * host_gauge_data_type_size);
  // for (int i = 0; i < 6; i++) oprod_qdp[i] = oprod_qdp_.data() + i * V * gauge_site_size * host_gauge_data_type_size;
  // int T = param.X[3];
  // int LX = param.X[0];
  // int LY = param.X[1];
  // int LZ = param.X[2];

  // for (int x0 = 0; x0 < T; x0++) {
  //   for (int x1 = 0; x1 < LX; x1++) {
  //     for (int x2 = 0; x2 < LY; x2++) {
  //       for (int x3 = 0; x3 < LZ; x3++) {
  //         int j = (x1 + LX * x2 + LY * LX * x3 + LZ * LY * LX * x0) / 2;
  //         int oddBit = (x0 + x1 + x2 + x3) & 1;
  //         for (int munu = 0; munu < 6; munu++) {
  //           double *out = (double *)oprod_qdp[munu];
  //           double *in = (double *)oprod;
  //           for (int i = 0; i < 9; i++) {
  //             for (int reim = 0; reim < 2; reim++) {
  //               out[reim + 2 * (i + 9 * (j + Vh * (oddBit)))]
  //                 = in[reim + 2 * (j / 2 + Vh * (i + 9 * (munu + 6 * (oddBit))))];
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // auto oprod_ex = quda::createExtendedTensorGauge(oprod_qdp.data(), param, R);
  // printf("HERE before oprod_ex created\n");

  for (int i = 0; i < Vh; i++) {
    for (int yIndex = 0; yIndex < 2; yIndex++) {
      for (int mu = 0; mu < 4; mu++) {
        for (int nu = 0; nu < 4; nu++) {
          if (nu == mu)
            continue;
          else if (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION)
            computeForce_reference<double>(h_mom, (void **)qdp_ex->Gauge_p(), lat, oprod, i, yIndex, parity, mu, nu);
          else if (gauge_param.cpu_prec == QUDA_SINGLE_PRECISION)
            computeForce_reference<float>(h_mom, (void **)qdp_ex->Gauge_p(), lat, oprod, i, yIndex, parity, mu, nu);
          else
            errorQuda("Unsupported precision %d", gauge_param.cpu_prec);
        }
      }
    }
  }
}

template <typename sFloat, typename gFloat>
void CloverSigmaOprod_reference(void *oprod_, quda::ColorSpinorField &inp, quda::ColorSpinorField &inx,
                                std::vector<std::vector<double>> &coeff)
{
  int nColor = 3;
  gFloat *oprod = (gFloat *)oprod_;
  sFloat *x = (sFloat *)inx.V();
  sFloat *p = (sFloat *)inp.V();

  gFloat oprod_f[gauge_site_size];
  gFloat oprod_imx2[gauge_site_size];

  for (int parity = 0; parity < 2; parity++) {
    for (int i = 0; i < Vh; i++) {
      for (int mu = 1; mu < 4; mu++) {
        for (int nu = 0; nu < mu; nu++) {

          sFloat temp[spinor_site_size], temp_munu[spinor_site_size], temp_numu[spinor_site_size];
          multiplySpinorByDiracgamma(temp, nu, &p[spinor_site_size * (i + Vh * parity)]);
          multiplySpinorByDiracgamma(temp_munu, mu, temp);

          multiplySpinorByDiracgamma(temp, mu, &p[spinor_site_size * (i + Vh * parity)]);
          multiplySpinorByDiracgamma(temp_numu, nu, temp);
          for (int s = 0; s < 4; s++) {
            for (int t = 0; t < 3; t++) {
              temp[s * (3 * 2) + t * (2) + 0]
                = -temp_munu[s * (3 * 2) + t * (2) + 0] + temp_numu[s * (3 * 2) + t * (2) + 0];
              temp[s * (3 * 2) + t * (2) + 1]
                = -temp_munu[s * (3 * 2) + t * (2) + 1] + temp_numu[s * (3 * 2) + t * (2) + 1];
            }
          }

          outerProdSpinTrace(oprod_f, temp, &x[spinor_site_size * (i + Vh * parity)]);
          su3_imagx2(oprod_imx2, oprod_f);

          int munu = (mu - 1) * mu / 2 + nu;

          for (int ci = 0; ci < nColor; ci++) {   // row
            for (int cj = 0; cj < nColor; cj++) { // col
              int color = ci * nColor + cj;
              int id = 2 * (i + Vh * (color + 9 * (munu + parity * 6)));
              oprod[id + 0] += coeff[0][parity] * oprod_imx2[color * 2 + 0] / 2.0;
              oprod[id + 1] += coeff[0][parity] * oprod_imx2[color * 2 + 1] / 2.0;
            }
          }
        }
      }
    }
  }
}

void computeCloverSigmaOprod_reference(void *oprod, quda::ColorSpinorField &p, quda::ColorSpinorField &x,
                                       std::vector<std::vector<double>> &ferm_epsilon, QudaGaugeParam &gauge_param)
{
  if (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION)
    CloverSigmaOprod_reference<double, double>(oprod, p, x, ferm_epsilon);
  else if (gauge_param.cpu_prec == QUDA_SINGLE_PRECISION)
    CloverSigmaOprod_reference<float, float>(oprod, p, x, ferm_epsilon);
  else
    errorQuda("Unsupported precision %d", gauge_param.cpu_prec);
}
