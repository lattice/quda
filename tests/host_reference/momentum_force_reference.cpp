#include <complex>
#include <gauge_field.h>
#include "host_utils.h"
#include "index_utils.hpp"
#include "momentum_force_reference.h"

// CPU implementations matching the GPU's two-step complete force -> momentum pipeline.
// The GPU CompleteForce kernel applies makeAntiHerm and stores to an 18-real force field,
// then updateMomentum reads that force, applies makeAntiHerm again, and stores to 10-real
// momentum. The existing hisqCompleteForceCPU goes directly from U*oprod to 10-real
// momentum in one step, which gives slightly different FP results.

namespace {

template <typename real> struct cplx {
  real re, im;
  cplx() : re(0), im(0) { }
  cplx(real r, real i) : re(r), im(i) { }
};

template <typename real> struct Mat3x3 {
  cplx<real> e[3][3];
};

template <typename real> Mat3x3<real> mat_mul(const Mat3x3<real> &a, const Mat3x3<real> &b)
{
  Mat3x3<real> c;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      real re = 0, im = 0;
      for (int k = 0; k < 3; k++) {
        re += a.e[i][k].re * b.e[k][j].re - a.e[i][k].im * b.e[k][j].im;
        im += a.e[i][k].re * b.e[k][j].im + a.e[i][k].im * b.e[k][j].re;
      }
      c.e[i][j] = {re, im};
    }
  return c;
}

template <typename real> void makeAntiHermCPU(Mat3x3<real> &m)
{
  // am = m - m^dagger
  Mat3x3<real> am;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      am.e[i][j].re = m.e[i][j].re - m.e[j][i].re;
      am.e[i][j].im = m.e[i][j].im + m.e[j][i].im;
    }

  // make traceless: subtract Im(trace)/3 from diagonal imaginary parts
  real imag_trace = am.e[0][0].im + am.e[1][1].im + am.e[2][2].im;
  for (int i = 0; i < 3; i++) am.e[i][i].im -= imag_trace / 3.0;

  // m = 0.5 * am
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      m.e[i][j].re = 0.5 * am.e[i][j].re;
      m.e[i][j].im = 0.5 * am.e[i][j].im;
    }
}

template <typename real>
void loadMatQDP(Mat3x3<real> &m, const real *const *field, int dir, int idx, int oddBit, int half_vol)
{
  const real *p = field[dir] + (oddBit * half_vol + idx) * 18;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      m.e[i][j].re = p[(i * 3 + j) * 2 + 0];
      m.e[i][j].im = p[(i * 3 + j) * 2 + 1];
    }
}

template <typename real>
void storeMatQDP(const Mat3x3<real> &m, real *const *field, int dir, int idx, int oddBit, int half_vol)
{
  real *p = field[dir] + (oddBit * half_vol + idx) * 18;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      p[(i * 3 + j) * 2 + 0] = m.e[i][j].re;
      p[(i * 3 + j) * 2 + 1] = m.e[i][j].im;
    }
}

template <typename real>
void storeMomMILC(const Mat3x3<real> &m, real coeff, real *mom, int dir, int idx, int oddBit, int half_vol)
{
  real *p = mom + ((oddBit * half_vol + idx) * 4 + dir) * 10;
  p[0] = (m.e[0][1].re - m.e[1][0].re) * 0.5 * coeff;
  p[1] = (m.e[0][1].im + m.e[1][0].im) * 0.5 * coeff;
  p[2] = (m.e[0][2].re - m.e[2][0].re) * 0.5 * coeff;
  p[3] = (m.e[0][2].im + m.e[2][0].im) * 0.5 * coeff;
  p[4] = (m.e[1][2].re - m.e[2][1].re) * 0.5 * coeff;
  p[5] = (m.e[1][2].im + m.e[2][1].im) * 0.5 * coeff;
  real temp = (m.e[0][0].im + m.e[1][1].im + m.e[2][2].im) / 3.0;
  p[6] = (m.e[0][0].im - temp) * coeff;
  p[7] = (m.e[1][1].im - temp) * coeff;
  p[8] = (m.e[2][2].im - temp) * coeff;
  p[9] = 0.0;
}

template <typename real>
void completeForceToForceT(real *const *force_out, const real *const *oprod, const real *const *link, const int dim[4],
                           const quda::lat_dim_t &R)
{
  int vol = dim[0] * dim[1] * dim[2] * dim[3];
  int half_vol = vol / 2;
  int e_half_vol = (dim[0] + 2 * R[0]) * (dim[1] + 2 * R[1]) * (dim[2] + 2 * R[2]) * (dim[3] + 2 * R[3]) / 2;

  for (int oddBit = 0; oddBit < 2; oddBit++) {
    real parity_coeff = (oddBit == 1) ? -1.0 : 1.0;
#pragma omp parallel for
    for (int idx = 0; idx < half_vol; idx++) {
      int e_idx = getIndexExtended(idx, dim, oddBit, R);

      for (int dir = 0; dir < 4; dir++) {
        Mat3x3<real> U, O;
        loadMatQDP(U, link, dir, e_idx, oddBit, e_half_vol);
        loadMatQDP(O, oprod, dir, e_idx, oddBit, e_half_vol);
        Mat3x3<real> result = mat_mul(U, O);
        makeAntiHermCPU(result);
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) {
            result.e[i][j].re *= parity_coeff;
            result.e[i][j].im *= parity_coeff;
          }
        storeMatQDP(result, force_out, dir, idx, oddBit, half_vol);
      }
    }
  }
}

template <typename real>
void updateMomentumFromForceT(real *mom, double coeff, const real *const *force, const int dim[4])
{
  int vol = dim[0] * dim[1] * dim[2] * dim[3];
  int half_vol = vol / 2;

  for (int oddBit = 0; oddBit < 2; oddBit++) {
#pragma omp parallel for
    for (int idx = 0; idx < half_vol; idx++) {
      for (int dir = 0; dir < 4; dir++) {
        Mat3x3<real> f;
        loadMatQDP(f, force, dir, idx, oddBit, half_vol);
        makeAntiHermCPU(f);
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) {
            f.e[i][j].re *= coeff;
            f.e[i][j].im *= coeff;
          }
        storeMomMILC(f, static_cast<real>(1.0), mom, dir, idx, oddBit, half_vol);
      }
    }
  }
}

} // anonymous namespace

void hisqCompleteForceToForceCPU(quda::GaugeField &force_out, const quda::GaugeField &oprod,
                                 const quda::GaugeField &link)
{
  const auto &R = oprod.R();
  int dim[4];
  for (int d = 0; d < 4; d++) {
    dim[d] = oprod.X()[d] - 2 * R[d];
    if (link.R()[d] != R[d] || link.X()[d] != oprod.X()[d])
      errorQuda("link and oprod must share the same extended geometry (dim %d)", d);
    if (force_out.R()[d] != 0 || force_out.X()[d] != dim[d])
      errorQuda("force_out must be a non-extended field on the local lattice (dim %d)", d);
  }

  if (oprod.Precision() == QUDA_DOUBLE_PRECISION) {
    completeForceToForceT<double>(force_out.data_array<double *>().data, oprod.data_array<const double *>().data,
                                  link.data_array<const double *>().data, dim, R);
  } else {
    completeForceToForceT<float>(force_out.data_array<float *>().data, oprod.data_array<const float *>().data,
                                 link.data_array<const float *>().data, dim, R);
  }
}

void updateMomentumFromForceCPU(quda::GaugeField &mom, double coeff, const quda::GaugeField &force)
{
  int dim[4];
  for (int d = 0; d < 4; d++) dim[d] = force.X()[d];

  if (force.Precision() == QUDA_DOUBLE_PRECISION) {
    updateMomentumFromForceT<double>(reinterpret_cast<double *>(mom.data()), coeff,
                                     force.data_array<const double *>().data, dim);
  } else {
    updateMomentumFromForceT<float>(reinterpret_cast<float *>(mom.data()), coeff,
                                    force.data_array<const float *>().data, dim);
  }
}
