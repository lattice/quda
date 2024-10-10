#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

// QUDA headers
#include <gauge_field.h>
#include <color_spinor_field.h> // convenient quark field container

#include "clover_force_reference.h"
#include "host_utils.h"
#include "index_utils.hpp"
#include "misc.h"
#include "dslash_reference.h"
#include "wilson_dslash_reference.h"
#include "gauge_force_reference.h"
#include "gamma_reference.h"

#include <Eigen/Dense>

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
  sFloat *spinorField = (sFloat *)inB.data();

  gFloat *gaugeEven[4], *gaugeOdd[4];

  sFloat *A = (sFloat *)inA.data();

  for (int dir = 0; dir < 4; dir++) {
    gaugeEven[dir] = gaugeFull[dir];
    gaugeOdd[dir] = gaugeFull[dir] + Vh * gauge_site_size;
  }

#pragma omp parallel for
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
      const sFloat *spinor = spinorNeighbor(i, dir, parity, spinorField, fwdSpinor, backSpinor, 1, 1);
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

void CloverForce_reference(void *h_mom, std::array<void *, 4> gauge, std::vector<quda::ColorSpinorField> &x,
                           std::vector<quda::ColorSpinorField> &p, std::vector<double> force_coeff)
{
  int dag = 1;
  // Get spinor ghost fields
  // First wrap the input spinor into a ColorSpinorField
  quda::ColorSpinorParam csParam[4];
  for (int i = 0; i < 4; ++i) {
    csParam[i].location = QUDA_CPU_FIELD_LOCATION;
    // csParam[i].v = in;
    csParam[i].nColor = 3;
    csParam[i].nSpin = 4;
    csParam[i].nDim = 4;
    for (int d = 0; d < 4; d++) csParam[i].x[d] = Z[d];
    csParam[i].setPrecision(QUDA_DOUBLE_PRECISION);
    csParam[i].pad = 0;
    csParam[i].siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam[i].x[0] /= 2;
    csParam[i].siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    csParam[i].fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    csParam[i].gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    csParam[i].create = QUDA_REFERENCE_FIELD_CREATE;
    csParam[i].pc_type = QUDA_4D_PC;
  }

  for (auto i = 0u; i < x.size(); i++) {
    for (int parity = 0; parity < 2; parity++) {
      quda::ColorSpinorField &inA = (parity & 1) ? x[i].Odd() : x[i].Even();
      quda::ColorSpinorField &inB = (parity & 1) ? p[i].Even() : p[i].Odd();
      quda::ColorSpinorField &inC = (parity & 1) ? p[i].Odd() : p[i].Even();
      quda::ColorSpinorField &inD = (parity & 1) ? x[i].Even() : x[i].Odd();

      static constexpr int nFace = 1;
      // every time that exchange ghost is called fwdGhostFaceBuffer becomes the Ghost of the last spinor called
      inB.exchangeGhost((QudaParity)(1 - parity), nFace, dag);
      CloverForce_kernel_host<double, double>(gauge, h_mom, inA, inB, 1, parity, force_coeff[i]);
      inD.exchangeGhost((QudaParity)(1 - parity), nFace, 1 - dag);
      CloverForce_kernel_host<double, double>(gauge, h_mom, inC, inD, -1, parity, force_coeff[i]);

      if (x[0].TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET) {
        csParam[0].v = inA.data();
        csParam[1].v = inB.data();
        csParam[2].v = inC.data();
        csParam[3].v = inD.data();

        for (int j = 0; j < 4; ++j) { csParam[j].v = (char *)csParam[j].v + Vh * spinor_site_size * sizeof(double); }
        quda::ColorSpinorField in2A(csParam[0]);
        quda::ColorSpinorField in2B(csParam[1]);
        quda::ColorSpinorField in2C(csParam[2]);
        quda::ColorSpinorField in2D(csParam[3]);

        in2B.exchangeGhost((QudaParity)(1 - parity), nFace, dag);
        CloverForce_kernel_host<double, double>(gauge, h_mom, in2A, in2B, 1, parity, force_coeff[i]);
        in2D.exchangeGhost((QudaParity)(1 - parity), nFace, 1 - dag);
        CloverForce_kernel_host<double, double>(gauge, h_mom, in2C, in2D, -1, parity, force_coeff[i]);
      }
    }
  }
}
template <typename cFloat>
void cloverSigmaTraceCompute_host(cFloat *oprod, cFloat *clover, double coeff, int parity, double mu2, double eps2,
                                  bool twist)
{
  int nSpin = 4;
  int nColor = 3;
  int N = nColor * nSpin / 2;
  int chiralBlock = N + 2 * (N - 1) * N / 2;

  typedef Eigen::Matrix<std::complex<cFloat>, 3, 3> Matrix3c;
  typedef Eigen::Matrix<std::complex<cFloat>, 6, 6> CloverM;

#pragma omp parallel for
  for (int i = 0; i < Vh; i++) {
    cFloat A_array[72];
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
      A = A * B;
      if (twist) { A = 0.25 * A; }

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

void computeCloverSigmaTrace_reference(void *oprod, void *clover, double coeff, int parity, double mu2, double eps2,
                                       bool twist)
{

  // FIXME: here call the appropriate template function according to gauge_precision
  cloverSigmaTraceCompute_host((double *)oprod, (double *)clover, coeff, parity, mu2, eps2, twist);
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
      int eo_full_id = i + parity * Vh;
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
      int eo_full_id = i + parity * Vh;

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
      int eo_full_id = i + otherparity * Vh;
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
      int eo_full_id = i + otherparity * Vh;

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

  quda::GaugeFieldParam gparam(gauge_param, oprod, QUDA_GENERAL_LINKS);
  gparam.create = QUDA_REFERENCE_FIELD_CREATE;
  gparam.order = QUDA_FLOAT2_GAUGE_ORDER;
  gparam.geometry = QUDA_TENSOR_GEOMETRY;
  auto oprod_ex = quda::createExtendedGauge(quda::GaugeField(gparam), R);

#pragma omp parallel for
  for (int i = 0; i < Vh; i++) {
    for (int yIndex = 0; yIndex < 2; yIndex++) {
      for (int mu = 0; mu < 4; mu++) {
        for (int nu = 0; nu < 4; nu++) {
          if (nu == mu)
            continue;
          else if (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION)
            computeForce_reference<double>(h_mom, (void **)qdp_ex->raw_pointer(), lat, oprod_ex->data(), i, yIndex,
                                           parity, mu, nu);
          else if (gauge_param.cpu_prec == QUDA_SINGLE_PRECISION)
            computeForce_reference<float>(h_mom, (void **)qdp_ex->raw_pointer(), lat, oprod_ex->data(), i, yIndex,
                                          parity, mu, nu);
          else
            errorQuda("Unsupported precision %d", gauge_param.cpu_prec);
        }
      }
    }
  }

  delete oprod_ex;
  delete qdp_ex;
}

template <typename sFloat, typename gFloat>
void CloverSigmaOprod_reference(void *oprod_, quda::ColorSpinorField &inp, quda::ColorSpinorField &inx,
                                std::vector<double> &coeff)
{
  int nColor = 3;
  gFloat *oprod = (gFloat *)oprod_;
  sFloat *x = (sFloat *)inx.data();
  sFloat *p = (sFloat *)inp.data();

  int flavors = inx.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET ? inx.TwistFlavor() : 1;

  for (int parity = 0; parity < 2; parity++) {
#pragma omp parallel for
    for (int i = 0; i < Vh; i++) {
      for (int mu = 1; mu < 4; mu++) {
        for (int nu = 0; nu < mu; nu++) {
          for (int flavor = 0; flavor < flavors; ++flavor) {

            sFloat temp[spinor_site_size], temp_munu[spinor_site_size], temp_numu[spinor_site_size];
            multiplySpinorByDiracGamma(temp, nu, &p[spinor_site_size * (i + Vh * flavor + Vh * flavors * parity)]);
            multiplySpinorByDiracGamma(temp_munu, mu, temp);

            multiplySpinorByDiracGamma(temp, mu, &p[spinor_site_size * (i + Vh * flavor + Vh * flavors * parity)]);
            multiplySpinorByDiracGamma(temp_numu, nu, temp);
            for (int s = 0; s < 4; s++) {
              for (int t = 0; t < 3; t++) {
                temp[s * (3 * 2) + t * (2) + 0]
                  = -temp_munu[s * (3 * 2) + t * (2) + 0] + temp_numu[s * (3 * 2) + t * (2) + 0];
                temp[s * (3 * 2) + t * (2) + 1]
                  = -temp_munu[s * (3 * 2) + t * (2) + 1] + temp_numu[s * (3 * 2) + t * (2) + 1];
              }
            }

            gFloat oprod_f[gauge_site_size];
            gFloat oprod_imx2[gauge_site_size];
            outerProdSpinTrace(oprod_f, temp, &x[spinor_site_size * (i + Vh * flavor + Vh * flavors * parity)]);
            su3_imagx2(oprod_imx2, oprod_f);

            int munu = (mu - 1) * mu / 2 + nu;

            for (int ci = 0; ci < nColor; ci++) {   // row
              for (int cj = 0; cj < nColor; cj++) { // col
                int color = ci * nColor + cj;
                int id = 2 * (i + Vh * (color + 9 * (munu + parity * 6)));
                oprod[id + 0] += coeff[parity] * oprod_imx2[color * 2 + 0] / 2.0;
                oprod[id + 1] += coeff[parity] * oprod_imx2[color * 2 + 1] / 2.0;
              }
            }
          }
        }
      }
    }
  }
}

void computeCloverSigmaOprod_reference(void *oprod, std::vector<quda::ColorSpinorField> &p,
                                       std::vector<quda::ColorSpinorField> &x,
                                       std::vector<std::vector<double>> &ferm_epsilon, QudaGaugeParam &gauge_param)
{
  for (auto i = 0u; i < x.size(); i++) {
    if (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION)
      CloverSigmaOprod_reference<double, double>(oprod, p[i], x[i], ferm_epsilon[i]);
    else if (gauge_param.cpu_prec == QUDA_SINGLE_PRECISION)
      CloverSigmaOprod_reference<float, float>(oprod, p[i], x[i], ferm_epsilon[i]);
    else
      errorQuda("Unsupported precision %d", gauge_param.cpu_prec);
  }
}
void Gamma5_host(double *out, double *in, const int V)
{
#pragma omp parallel for
  for (int i = 0; i < V; i++) {
    for (int c = 0; c < 3; c++) {
      for (int reim = 0; reim < 2; reim++) {
        out[i * 24 + 0 * 6 + c * 2 + reim] = in[i * 24 + 0 * 6 + c * 2 + reim];
        out[i * 24 + 1 * 6 + c * 2 + reim] = in[i * 24 + 1 * 6 + c * 2 + reim];
        out[i * 24 + 2 * 6 + c * 2 + reim] = -in[i * 24 + 2 * 6 + c * 2 + reim];
        out[i * 24 + 3 * 6 + c * 2 + reim] = -in[i * 24 + 3 * 6 + c * 2 + reim];
      }
    }
  }
}

void tau1_host(double *out, double *in, const int V)
{
  int V2 = V * 24 / 2;
#pragma omp parallel for
  for (int i = 0; i < V2; i++) {
    double up = in[i];
    double down = in[i + V2];
    out[i] = down;
    out[i + V2] = up;
  }
}

void axpbyz_host(double a, double *x, double b, double *y, double *z, const int V)
{
#pragma omp parallel for
  for (int i = 0; i < V * 24; i++) { z[i] = a * x[i] + b * y[i]; }
}

void caxpy_host(double a_re, double a_im, double *x, double *y, const int V)
{
#pragma omp parallel for
  for (int i = 0; i < V * 12; i++) {
    double re = a_re * x[i * 2 + 0] - a_im * x[i * 2 + 1] + y[i * 2 + 0];
    double im = a_re * x[i * 2 + 1] + a_im * x[i * 2 + 0] + y[i * 2 + 1];
    y[i * 2 + 0] = re;
    y[i * 2 + 1] = im;
  }
}

void Gamma5_host_UKQCD(double *out, double *in, const int V)
{
#pragma omp parallel for
  for (int i = 0; i < V; i++) {
    for (int c = 0; c < 3; c++) {
      for (int reim = 0; reim < 2; reim++) {
        out[i * 24 + 0 * 6 + c * 2 + reim] = in[i * 24 + 2 * 6 + c * 2 + reim];
        out[i * 24 + 1 * 6 + c * 2 + reim] = in[i * 24 + 3 * 6 + c * 2 + reim];
        out[i * 24 + 2 * 6 + c * 2 + reim] = in[i * 24 + 0 * 6 + c * 2 + reim];
        out[i * 24 + 3 * 6 + c * 2 + reim] = in[i * 24 + 1 * 6 + c * 2 + reim];
      }
    }
  }
}
template <typename Float> void add_mom(Float *a, Float *b, int len, double coeff)
{
#pragma omp parallel for
  for (int i = 0; i < len; i++) { a[i] += coeff * b[i]; }
}

template <typename Float> void set_to_zero(void *oprod_)
{
  Float *oprod = (Float *)oprod_;
#pragma omp parallel for
  for (size_t i = 0; i < V * 6 * gauge_site_size; i++) oprod[i] = 0;
}

void TMCloverForce_reference(void *h_mom, void **h_x, void **h_x0, double *coeff, int nvector,
                             std::array<void *, 4> &gauge, std::vector<char> &clover, std::vector<char> &clover_inv,
                             QudaGaugeParam *gauge_param, QudaInvertParam *inv_param, int detratio)
{
  if (inv_param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    errorQuda("Preconditioned operator type %d not supported by test code", inv_param->matpc_type);
  if (inv_param->dagger != QUDA_DAG_YES) errorQuda("Test code presently requires dagger option");

  quda::ColorSpinorParam qParam;
  inv_param->solution_type = QUDA_MATDAG_MAT_SOLUTION; // set to full solution for field creation
  constructWilsonTestSpinorParam(&qParam, inv_param, gauge_param);
  inv_param->solution_type = QUDA_MATPCDAG_MATPC_SOLUTION; // restore to single parity

  std::vector<quda::ColorSpinorField> x(nvector), p(nvector), x0(nvector);
  for (int i = 0; i < nvector; i++) {
    x[i] = quda::ColorSpinorField(qParam);
    p[i] = quda::ColorSpinorField(qParam);
    if (detratio) x0[i] = quda::ColorSpinorField(qParam);
  }

  qParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  qParam.x[0] /= 2;
  quda::ColorSpinorField tmp(qParam);

  qParam.create = QUDA_REFERENCE_FIELD_CREATE;
  qParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  for (int i = 0; i < nvector; i++) {
    qParam.v = h_x[i];
    quda::ColorSpinorField load_half(qParam);
    x[i].Odd() = load_half;

    Gamma5_host(tmp.data<double *>(), x[i].Odd().data<double *>(), x[i].Odd().VolumeCB());

    int parity = 0;
    QudaMatPCType myMatPCType = inv_param->matpc_type;

    if (myMatPCType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || myMatPCType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {

      if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if (inv_param->twist_flavor == QUDA_TWIST_SINGLET) {
          tmc_dslash(x[i].Even().data(), gauge.data(), clover.data(), clover_inv.data(), tmp.data(), inv_param->kappa,
                     inv_param->mu, inv_param->twist_flavor, myMatPCType, parity, QUDA_DAG_YES, inv_param->cpu_prec,
                     *gauge_param);
        } else if (inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
          tmc_ndeg_dslash(x[i].Even().data(), gauge.data(), clover.data(), clover_inv.data(), tmp.data(),
                          inv_param->kappa, inv_param->mu, inv_param->epsilon, myMatPCType, parity, QUDA_DAG_YES,
                          inv_param->cpu_prec, *gauge_param);
        }
      } else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_dslash(x[i].Even().data(), gauge.data(), clover_inv.data(), tmp.data(), parity, QUDA_DAG_YES,
                      inv_param->cpu_prec, *gauge_param);
      } else {
        errorQuda("TMCloverForce_reference: dslash_type not supported\n");
      }

      if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if (inv_param->twist_flavor == QUDA_TWIST_SINGLET) {
          tmc_matpc(p[i].Odd().data(), gauge.data(), clover.data(), clover_inv.data(), tmp.data(), inv_param->kappa,
                    inv_param->mu, inv_param->twist_flavor, myMatPCType, QUDA_DAG_YES, inv_param->cpu_prec, *gauge_param);
        } else if (inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
          tmc_ndeg_matpc(p[i].Odd().data(), gauge.data(), clover.data(), clover_inv.data(), tmp.data(), inv_param->kappa,
                         inv_param->mu, inv_param->epsilon, myMatPCType, QUDA_DAG_YES, inv_param->cpu_prec, *gauge_param);
        }
      } else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_matpc(p[i].Odd().data(), gauge.data(), clover.data(), clover_inv.data(), tmp.data(), inv_param->kappa,
                     myMatPCType, QUDA_DAG_YES, inv_param->cpu_prec, *gauge_param);
      } else {
        errorQuda("TMCloverForce_reference: dslash_type not supported\n");
      }

      if (inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
        axpbyz_host(1.0 / inv_param->evmax, p[i].Odd().data<double *>(), 0, p[i].Odd().data<double *>(),
                    p[i].Odd().data<double *>(), p[i].Odd().VolumeCB());
        tau1_host(x[i].Even().data<double *>(), x[i].Even().data<double *>(), x[i].Even().VolumeCB());
        tau1_host(p[i].Odd().data<double *>(), p[i].Odd().data<double *>(), p[i].Odd().VolumeCB());
        caxpy_host(0.0, -inv_param->offset[i], x[i].Odd().data<double *>(), p[i].Odd().data<double *>(),
                   p[i].Odd().VolumeCB());
      }

      Gamma5_host(x[i].Even().data<double *>(), x[i].Even().data<double *>(), x[i].Even().VolumeCB());

      if (detratio && inv_param->twist_flavor != QUDA_TWIST_NONDEG_DOUBLET) {
        qParam.v = h_x0[i];
        quda::ColorSpinorField load_half(qParam);
        x0[i].Odd() = load_half;
        axpbyz_host(1, p[i].Odd().data<double *>(), 1, x0[i].Odd().data<double *>(), p[i].Odd().data<double *>(),
                    p[i].Odd().VolumeCB());
      }

      if (inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
        Gamma5_host(p[i].Odd().data<double *>(), p[i].Odd().data<double *>(), p[i].Odd().VolumeCB());
      }

      if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if (inv_param->twist_flavor == QUDA_TWIST_SINGLET)
          tmc_dslash(p[i].Even().data(), gauge.data(), clover.data(), clover_inv.data(), p[i].Odd().data(),
                     inv_param->kappa, inv_param->mu, inv_param->twist_flavor, myMatPCType, parity, QUDA_DAG_NO,
                     inv_param->cpu_prec, *gauge_param);
        else if (inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET)
          tmc_ndeg_dslash(p[i].Even().data(), gauge.data(), clover.data(), clover_inv.data(), p[i].Odd().data(),
                          inv_param->kappa, inv_param->mu, inv_param->epsilon, myMatPCType, parity, QUDA_DAG_YES,
                          inv_param->cpu_prec, *gauge_param);
      } else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_dslash(p[i].Even().data(), gauge.data(), clover_inv.data(), p[i].Odd().data(), parity, QUDA_DAG_NO,
                      inv_param->cpu_prec, *gauge_param);
      } else {
        errorQuda("TMCloverForce_reference: dslash_type not supported\n");
      }

      if (inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
        Gamma5_host(p[i].Odd().data<double *>(), p[i].Odd().data<double *>(), p[i].Odd().VolumeCB());
        Gamma5_host(p[i].Even().data<double *>(), p[i].Even().data<double *>(), p[i].Even().VolumeCB());
        tau1_host(p[i].Even().data<double *>(), p[i].Even().data<double *>(), p[i].Even().VolumeCB());
      }

    } else {
      errorQuda("TMCloverForce_reference: MATPC type not supported\n");
    }

    Gamma5_host(p[i].Even().data<double *>(), p[i].Even().data<double *>(), p[i].Even().VolumeCB());
    Gamma5_host(p[i].Odd().data<double *>(), p[i].Odd().data<double *>(), p[i].Odd().VolumeCB());
  }
  std::vector<double> force_coeff(nvector);
  for (int i = 0; i < nvector; i++) { force_coeff[i] = 1.0 * coeff[i]; }
  quda::GaugeFieldParam momparam(*gauge_param);
  // momparam.order = QUDA_QDP_GAUGE_ORDER;
  momparam.location = QUDA_CPU_FIELD_LOCATION;
  momparam.order = QUDA_MILC_GAUGE_ORDER;
  momparam.reconstruct = QUDA_RECONSTRUCT_10;
  momparam.link_type = QUDA_ASQTAD_MOM_LINKS;
  momparam.create = QUDA_ZERO_FIELD_CREATE;
  quda::GaugeField mom(momparam);
  createMomCPU(mom.data(), gauge_param->cpu_prec, 0.0);
  void *refmom = mom.data();

  // derivative of the wilson operator it correspond to deriv_Sb(OE,...) plus  deriv_Sb(EO,...) in tmLQCD
  CloverForce_reference(refmom, gauge, x, p, force_coeff);

  // create oprod and trace field
  std::vector<char> oprod_(V * 6 * gauge_site_size * host_gauge_data_type_size);
  void *oprod = oprod_.data();

  if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION)
    set_to_zero<double>(oprod);
  else if (gauge_param->cpu_prec == QUDA_SINGLE_PRECISION)
    set_to_zero<float>(oprod);
  else
    errorQuda("precision not valid");

  double k_csw_ov_8 = inv_param->kappa * inv_param->clover_csw / 8.0;
  size_t twist_flavor = inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH ? inv_param->twist_flavor : QUDA_TWIST_NO;
  double mu2
    = twist_flavor != QUDA_TWIST_NO ? 4. * inv_param->kappa * inv_param->kappa * inv_param->mu * inv_param->mu : 0.0;
  double eps2 = twist_flavor == QUDA_TWIST_NONDEG_DOUBLET ?
    4.0 * inv_param->kappa * inv_param->kappa * inv_param->epsilon * inv_param->epsilon :
    0.0;

  // derivative of the determinant of the sw term, second term of (A12) in hep-lat/0112051,  sw_deriv(EE, mnl->mu) in tmLQCD
  if (!detratio) computeCloverSigmaTrace_reference(oprod, clover.data(), k_csw_ov_8 * 32.0, 0, mu2, eps2, twist_flavor);

  std::vector<std::vector<double>> ferm_epsilon(nvector);
  for (int i = 0; i < nvector; i++) {
    ferm_epsilon[i].reserve(2);
    ferm_epsilon[i][0] = k_csw_ov_8 * coeff[i];
    ferm_epsilon[i][1] = k_csw_ov_8 * coeff[i] / (inv_param->kappa * inv_param->kappa);
    if (inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      tau1_host(p[i].Even().data<double *>(), p[i].Even().data<double *>(), p[i].Even().VolumeCB());
      tau1_host(p[i].Odd().data<double *>(), p[i].Odd().data<double *>(), p[i].Odd().VolumeCB());
    }
  }
  // derivative of pseudofermion sw term, first term term of (A12) in hep-lat/0112051,  sw_spinor_eo(EE,..) plus
  // sw_spinor_eo(OO,..)  in tmLQCD
  computeCloverSigmaOprod_reference(oprod, p, x, ferm_epsilon, *gauge_param);

  // oprod = (A12) of hep-lat/0112051
  // compute the insertion of oprod in Fig.27 of hep-lat/0112051
  cloverDerivative_reference(refmom, gauge.data(), oprod, QUDA_ODD_PARITY, *gauge_param);
  cloverDerivative_reference(refmom, gauge.data(), oprod, QUDA_EVEN_PARITY, *gauge_param);

  add_mom((double *)h_mom, (double *)mom.data(), 4 * V * mom_site_size, -1.0);
}
