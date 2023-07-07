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
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
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

#pragma unroll
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
                             quda::ColorSpinorField &inB, quda::ColorSpinorField &inC, quda::ColorSpinorField &inD,
                             int parity, double force_coeff)
{

  gFloat **gaugeFull = (gFloat **)gauge.data();
  sFloat **backSpinor = (sFloat **)inB.fwdGhostFaceBuffer;
  sFloat **fwdSpinor = (sFloat **)inB.backGhostFaceBuffer;
  sFloat *spinorField = (sFloat *)inB.V();

  gFloat *gaugeEven[4], *gaugeOdd[4];
  gFloat *ghostGaugeEven[4], *ghostGaugeOdd[4];

  for (int dir = 0; dir < 4; dir++) {
    gaugeEven[dir] = gaugeFull[dir];
    gaugeOdd[dir] = gaugeFull[dir] + Vh * gauge_site_size;
    // we do not need gauge ghost
    // ghostGaugeEven[dir] = ghostGauge[dir];
    // ghostGaugeOdd[dir] = ghostGauge[dir] + (faceVolume[dir] / 2) * gauge_site_size;
  }
  for (int i = 0; i < Vh; i++) {
    for (int dir = 0; dir < 8; dir += 2) { // figer cross that are the forward direction
      // load the gauge
      gFloat **gaugeField = (parity ? gaugeOdd : gaugeEven);
      gFloat *gauge = &gaugeField[dir / 2][i * (3 * 3 * 2)];
      // load spinor and project
      const sFloat *spinor = spinorNeighbor_mg4dir(i, dir, parity, spinorField, fwdSpinor, backSpinor, 1, 1);
      sFloat projectedSpinor[spinor_site_size];
      int projIdx = 2 * (dir / 2) +1; //+ (dir + daggerBit) % 2;
      multiplySpinorByDiracProjector(projectedSpinor, projIdx, spinor);
      // printf("host x=%d d=%d  B=(%g,%g)(%g,%g)(%g,%g)-(%g,%g)(%g,%g)(%g,%g)-(%g,%g)(%g,%g)(%g,%g)-(%g,%g)(%g,%g)(%g,%g) \n",i,dir
      //   ,projectedSpinor[0 * (6) + 0 * (2) + 0],projectedSpinor[0 * (6) + 0 * (2) + 1]
      //   ,projectedSpinor[0 * (6) + 1 * (2) + 0],projectedSpinor[0 * (6) + 1 * (2) + 1]
      //   ,projectedSpinor[0 * (6) + 2 * (2) + 0],projectedSpinor[0 * (6) + 2 * (2) + 1]
      //   ,projectedSpinor[1 * (6) + 0 * (2) + 0],projectedSpinor[1 * (6) + 0 * (2) + 1]
      //   ,projectedSpinor[1 * (6) + 1 * (2) + 0],projectedSpinor[1 * (6) + 1 * (2) + 1]
      //   ,projectedSpinor[1 * (6) + 2 * (2) + 0],projectedSpinor[1 * (6) + 2 * (2) + 1]
      //   ,projectedSpinor[2 * (6) + 0 * (2) + 0],projectedSpinor[2 * (6) + 0 * (2) + 1]
      //   ,projectedSpinor[2 * (6) + 1 * (2) + 0],projectedSpinor[2 * (6) + 1 * (2) + 1]
      //   ,projectedSpinor[2 * (6) + 2 * (2) + 0],projectedSpinor[2 * (6) + 2 * (2) + 1]
      //   ,projectedSpinor[3 * (6) + 0 * (2) + 0],projectedSpinor[3 * (6) + 0 * (2) + 1]
      //   ,projectedSpinor[3 * (6) + 1 * (2) + 0],projectedSpinor[3 * (6) + 1 * (2) + 1]
      //   ,projectedSpinor[3 * (6) + 2 * (2) + 0],projectedSpinor[3 * (6) + 2 * (2) + 1]   
      // );
      gFloat oprod[gauge_site_size];
      sFloat *A = (sFloat *)inA.V();
      // sFloat *B = (sFloat *)inB.V();
      outerProdSpinTrace(oprod, projectedSpinor, &A[i * spinor_site_size]);
      // outerProdSpinTrace(oprod, &B[i * spinor_site_size], &A[i * spinor_site_size]);
      gFloat force[gauge_site_size];
      for (int j = 0; j < gauge_site_size; j++) force[j] = 0;
      accum_su3xsu3(force, gauge, oprod, force_coeff);
      int mu=(dir / 2);
      gFloat *mom = (gFloat *)h_mom + (4 * i + mu) * mom_site_size;
      accum_su3_to_anti_hermitian(mom, force);
    }
  }
}

void CloverForce_reference(void *h_mom, std::array<void *, 4> gauge, quda::ColorSpinorField &x,
                           quda::ColorSpinorField &p, double force_coeff)
{
  int dag = 1;
  for (int parity = 0; parity < 1; parity++) {
    quda::ColorSpinorField &inA = (parity & 1) ? p.Odd() : p.Even();
    quda::ColorSpinorField &inB = (parity & 1) ? x.Even() : x.Odd();
    quda::ColorSpinorField &inC = (parity & 1) ? x.Odd() : x.Even();
    quda::ColorSpinorField &inD = (parity & 1) ? p.Even() : p.Odd();

    static constexpr int nFace = 1;
    inB.exchangeGhost((QudaParity)(1- parity), nFace, dag);
    //   exchangeGhost(inB, parity, dag);
    inD.exchangeGhost((QudaParity)(1 - parity), nFace, 1 - dag);
    //   exchangeGhost(inD, parity, 1 - dag);

    //   instantiate<CloverForce, ReconstructNo12>(U, force, inA, inB, inC, inD, parity, coeff[i]);
    CloverForce_kernel_host<double, double>(gauge, h_mom, inA, inB, inC, inD, parity, force_coeff);
  }
}