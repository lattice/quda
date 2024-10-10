#pragma once

#include <array>
#include <host_utils.h>
#include <index_utils.hpp>
#include <comm_quda.h>
#include <gauge_field.h>

template <typename Float> static inline void sum(Float *dst, Float *a, Float *b, int cnt)
{
  for (int i = 0; i < cnt; i++) dst[i] = a[i] + b[i];
}

template <typename Float> static inline void sub(Float *dst, Float *a, Float *b, int cnt)
{
  for (int i = 0; i < cnt; i++) dst[i] = a[i] - b[i];
}

template <typename Float> static inline void ax(Float *dst, Float a, Float *x, int cnt)
{
  for (int i = 0; i < cnt; i++) dst[i] = a * x[i];
}

// performs the operation y[i] = a*x[i] + y[i]
template <typename Float> static inline void axpy(Float a, Float *x, Float *y, int len)
{
  for (int i = 0; i < len; i++) y[i] = a * x[i] + y[i];
}

// performs the operation y[i] = a*x[i] + b*y[i]
template <typename Float> static inline void axpby(Float a, Float *x, Float b, Float *y, int len)
{
  for (int i = 0; i < len; i++) y[i] = a * x[i] + b * y[i];
}

// performs the operation y[i] = a*x[i] - y[i]
template <typename Float> static inline void axmy(Float *x, Float a, Float *y, int len)
{
  for (int i = 0; i < len; i++) y[i] = a * x[i] - y[i];
}

template <typename Float> static double norm2(Float *v, int len)
{
  double sum = 0.0;
  for (int i = 0; i < len; i++) sum += v[i] * v[i];
  return sum;
}

template <typename Float> static inline void negx(Float *x, int len)
{
  for (int i = 0; i < len; i++) x[i] = -x[i];
}

template <typename sFloat, typename gFloat> static inline void dot(sFloat *res, const gFloat *a, const sFloat *b)
{
  res[0] = res[1] = 0;
  for (int m = 0; m < 3; m++) {
    sFloat a_re = a[2 * m + 0];
    sFloat a_im = a[2 * m + 1];
    sFloat b_re = b[2 * m + 0];
    sFloat b_im = b[2 * m + 1];
    res[0] += a_re * b_re - a_im * b_im;
    res[1] += a_re * b_im + a_im * b_re;
  }
}

template <typename Float> static inline void su3Transpose(Float *res, const Float *mat)
{
  for (int m = 0; m < 3; m++) {
    for (int n = 0; n < 3; n++) {
      res[m * (3 * 2) + n * (2) + 0] = +mat[n * (3 * 2) + m * (2) + 0];
      res[m * (3 * 2) + n * (2) + 1] = -mat[n * (3 * 2) + m * (2) + 1];
    }
  }
}

template <typename sFloat, typename gFloat> static inline void su3Mul(sFloat *res, const gFloat *mat, const sFloat *vec)
{
  for (int n = 0; n < 3; n++) dot(&res[n * (2)], &mat[n * (3 * 2)], vec);
}

template <typename sFloat, typename gFloat>
static inline void su3Tmul(sFloat *res, const gFloat *mat, const sFloat *vec)
{
  gFloat matT[3 * 3 * 2];
  su3Transpose(matT, mat);
  su3Mul(res, matT, vec);
}

std::array<double, 2> verifyInversion(void *spinorOut, void *spinorIn, void *spinorCheck, QudaGaugeParam &gauge_param,
                                      QudaInvertParam &inv_param, void **gauge, void *clover, void *clover_inv,
                                      int src_idx);

std::array<double, 2> verifyInversion(void *spinorOut, void **spinorOutMulti, void *spinorIn, void *spinorCheck,
                                      QudaGaugeParam &gauge_param, QudaInvertParam &inv_param, void **gauge,
                                      void *clover, void *clover_inv, int src_idx = 0);

std::array<double, 2> verifyDomainWallTypeInversion(void *spinorOut, void **spinorOutMulti, void *spinorIn,
                                                    void *spinorCheck, QudaGaugeParam &gauge_param,
                                                    QudaInvertParam &inv_param, void **gauge, void *clover,
                                                    void *clover_inv, int src_idx);

double verifyWilsonTypeEigenvector(void *spinor, double _Complex lambda, int i, QudaGaugeParam &gauge_param,
                                   QudaEigParam &eig_param, void **gauge, void *clover, void *clover_inv);

double verifyWilsonTypeSingularVector(void *spinor_left, void *spinor_right, double _Complex sigma, int i,
                                      QudaGaugeParam &gauge_param, QudaEigParam &eig_param, void **gauge, void *clover,
                                      void *clover_inv);

std::array<double, 2> verifyWilsonTypeInversion(void *spinorOut, void **spinorOutMulti, void *spinorIn, void *spinorCheck,
                                                QudaGaugeParam &gauge_param, QudaInvertParam &inv_param, void **gauge,
                                                void *clover, void *clover_inv, int src_idx);

/**
 * @brief Verify a staggered inversion on the host. This version is a thin wrapper around a version that takes
 *        an array of outputs as is necessary for handling both single- and multi-shift solves.
 *
 * @param in The initial rhs
 * @param out The solution to A out = in
 * @param fat_link The fat links in the context of an ASQTAD solve; otherwise the base gauge links with phases applied
 * @param long_link The long links; null for naive staggered and Laplace
 * @param inv_param Invert params, used to query the solve type, etc
 * @return The residual and HQ residual (if requested)
 */
std::array<double, 2> verifyStaggeredInversion(quda::ColorSpinorField &in, quda::ColorSpinorField &out,
                                               quda::GaugeField &fat_link, quda::GaugeField &long_link,
                                               QudaInvertParam &inv_param, int src_idx);

/**
 * @brief Verify a single- or multi-shift staggered inversion on the host
 *
 * @param in The initial rhs
 * @param out The solutions to (A + shift) out = in for multiple shifts; shift == 0 for a single shift solve
 * @param fat_link The fat links in the context of an ASQTAD solve; otherwise the base gauge links with phases applied
 * @param long_link The long links; null for naive staggered and Laplace
 * @param inv_param Invert params, used to query the solve type, etc, also includes the shifts
 * @return The residual and HQ residual (if requested)
 */
std::array<double, 2> verifyStaggeredInversion(quda::ColorSpinorField &in,
                                               std::vector<quda::ColorSpinorField> &out_vector,
                                               quda::GaugeField &fat_link, quda::GaugeField &long_link,
                                               QudaInvertParam &inv_param, int src_idx = 0);

/**
 * @brief Verify a staggered-type eigenvector
 *
 * @param spinor The host eigenvector to be verified
 * @param lambda The host eigenvalue to be verified
 * @param i The number of the eigenvalue, only used when printing outputs
 * @param eig_param Eigensolve params, used to query the operator type, etc
 * @param fat_link The fat links in the context of an ASQTAD solve; otherwise the base gauge links with phases applied
 * @param long_link The long links; null for naive staggered and Laplace
 * @return The residual norm
 */
double verifyStaggeredTypeEigenvector(quda::ColorSpinorField &spinor, double _Complex lambda, int i,
                                      QudaEigParam &eig_param, quda::GaugeField &fat_link, quda::GaugeField &long_link);

/**
 * @brief Verify a staggered-type singular vector
 *
 * @param spinor The host left singular vector to be verified
 * @param spinor_right The host right singular vector to be verified
 * @param lambda The host singular value to be verified
 * @param i The number of the singular value, only used when printing outputs
 * @param eig_param Eigensolve params, used to query the operator type, etc
 * @param fat_link The fat links in the context of an ASQTAD solve; otherwise the base gauge links with phases applied
 * @param long_link The long links; null for naive staggered and Laplace
 * @return The residual norm
 */
double verifyStaggeredTypeSingularVector(quda::ColorSpinorField &spinor_left, quda::ColorSpinorField &spinor_right,
                                         double _Complex sigma, int i, QudaEigParam &eig_param,
                                         quda::GaugeField &fat_link, quda::GaugeField &long_link);

/**
 * @brief Verify the spinor distance reweighting
 *
 * @param spinor The spinor used to check the distance reweighting subroutine
 * @param alpha0 The parameter for the distance preconditioning, should always be positive
 * @param t0 The parameter for the distance preconditioning
 */
double verifySpinorDistanceReweight(quda::ColorSpinorField &spinor, double alpha0, int t0);

/**
 * @brief Return the pointer to a gauge link as a function of an origin and an offset
 *
 * @param i The checkerboard index of the site
 * @param dir The displacement direction
 * @param oddBit The parity of the site
 * @param gaugeEven The even gauge fields stored in a QDP layout
 * @param gaugeOdd The odd gauge fields stored in a QDP layout
 * @param ghostGaugeEven The even-parity gauge ghost fields
 * @param ghostGaugeOdd The odd-parity gauge ghost fields
 * @param n_ghost_faces The depth of the ghost fields
 * @param nbr_distance Displacement distance
 * @return A pointer to the offset gauge link
 */
template <typename Float>
static inline Float *gaugeLink(int i, int dir, int oddBit, Float **gaugeEven, Float **gaugeOdd,
                                      Float **ghostGaugeEven, Float **ghostGaugeOdd, int n_ghost_faces, int nbr_distance)
{
  Float **gaugeField;
  int j;
  int d = nbr_distance;
  if (dir % 2 == 0) {
    j = i;
    gaugeField = (oddBit ? gaugeOdd : gaugeEven);
  } else {

    int Y = fullLatticeIndex(i, oddBit);
    int x4 = Y / (Z[2] * Z[1] * Z[0]);
    int x3 = (Y / (Z[1] * Z[0])) % Z[2];
    int x2 = (Y / Z[0]) % Z[1];
    int x1 = Y % Z[0];
    int X1 = Z[0];
    int X2 = Z[1];
    int X3 = Z[2];
    int X4 = Z[3];
    Float *ghostGaugeField;

    switch (dir) {
    case 1: { //-X direction
      int new_x1 = (x1 - d + X1) % X1;
      if (x1 - d < 0 && quda::comm_dim_partitioned(0)) {
        ghostGaugeField = (oddBit ? ghostGaugeEven[0] : ghostGaugeOdd[0]);
        int offset = (n_ghost_faces + x1 - d) * X4 * X3 * X2 / 2 + (x4 * X3 * X2 + x3 * X2 + x2) / 2;
        return &ghostGaugeField[offset * (3 * 3 * 2)];
      }
      j = (x4 * X3 * X2 * X1 + x3 * X2 * X1 + x2 * X1 + new_x1) / 2;
      break;
    }
    case 3: { //-Y direction
      int new_x2 = (x2 - d + X2) % X2;
      if (x2 - d < 0 && quda::comm_dim_partitioned(1)) {
        ghostGaugeField = (oddBit ? ghostGaugeEven[1] : ghostGaugeOdd[1]);
        int offset = (n_ghost_faces + x2 - d) * X4 * X3 * X1 / 2 + (x4 * X3 * X1 + x3 * X1 + x1) / 2;
        return &ghostGaugeField[offset * (3 * 3 * 2)];
      }
      j = (x4 * X3 * X2 * X1 + x3 * X2 * X1 + new_x2 * X1 + x1) / 2;
      break;
    }
    case 5: { //-Z direction
      int new_x3 = (x3 - d + X3) % X3;
      if (x3 - d < 0 && quda::comm_dim_partitioned(2)) {
        ghostGaugeField = (oddBit ? ghostGaugeEven[2] : ghostGaugeOdd[2]);
        int offset = (n_ghost_faces + x3 - d) * X4 * X2 * X1 / 2 + (x4 * X2 * X1 + x2 * X1 + x1) / 2;
        return &ghostGaugeField[offset * (3 * 3 * 2)];
      }
      j = (x4 * X3 * X2 * X1 + new_x3 * X2 * X1 + x2 * X1 + x1) / 2;
      break;
    }
    case 7: { //-T direction
      int new_x4 = (x4 - d + X4) % X4;
      if (x4 - d < 0 && quda::comm_dim_partitioned(3)) {
        ghostGaugeField = (oddBit ? ghostGaugeEven[3] : ghostGaugeOdd[3]);
        int offset = (n_ghost_faces + x4 - d) * X1 * X2 * X3 / 2 + (x3 * X2 * X1 + x2 * X1 + x1) / 2;
        return &ghostGaugeField[offset * (3 * 3 * 2)];
      }
      j = (new_x4 * (X3 * X2 * X1) + x3 * (X2 * X1) + x2 * (X1) + x1) / 2;
      break;
    } // 7

    default:
      j = -1;
      errorQuda("wrong dir");
    }
    gaugeField = (oddBit ? gaugeEven : gaugeOdd);
  }

  return &gaugeField[dir / 2][j * (3 * 3 * 2)];
}


/**
 * @brief Return the pointer to a gauge link as a function of an origin and an offset
 *
 * @param i The checkerboard index of the site
 * @param dir The displacement direction
 * @param oddBit The parity of the site
 * @param gaugeEven The even gauge fields stored in a QDP layout
 * @param gaugeOdd The odd gauge fields stored in a QDP layout
 * @param nbr_distance Displacement distance
 * @return A pointer to the offset gauge link
 */
template <typename Float>
static inline Float *gaugeLink(int i, int dir, int oddBit, Float **gaugeEven, Float **gaugeOdd, int nbr_distance)
{
  return gaugeLink(i, dir, oddBit, gaugeEven, gaugeOdd, static_cast<Float**>(nullptr), static_cast<Float**>(nullptr), 0, nbr_distance);
}

/**
 * @brief Compute the 4th dimension index for a given checkerboard index
 * @param i The checkerboard index
 * @param oddBit The odd/even bit for the index
 * @return The 4th dimension index
 */
inline int x4_mg(int i, int oddBit)
{
  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y / (Z[2] * Z[1] * Z[0]);
  return x4;
}

/**
 * @brief Return the pointer to a fermion field as a function of an origin and an offset
 *
 * @param i The checkerboard index of the site
 * @param dir The displacement direction
 * @param oddBit The parity of the site
 * @param spinorField The spinor field
 * @param fwd_nbr_spinor The forward ghost region for the spinor field
 * @param back_nbr_spinor The backward ghost region for the spinor field
 * @param neighbor_distance Displacement distance
 * @param nFace The depth of the ghost fields
 * @param site_size The number of values in a single spinor (6 for staggered, 24 for Wilson)
 * @return A pointer to the offset fermion field
 */
template <typename Float>
static inline const Float *spinorNeighbor(int i, int dir, int oddBit, const Float *spinorField,
                                          Float **fwd_nbr_spinor, Float **back_nbr_spinor, int neighbor_distance,
                                          int nFace, int site_size = 24)
{
  int j;
  int nb = neighbor_distance;
  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y / (Z[2] * Z[1] * Z[0]);
  int x3 = (Y / (Z[1] * Z[0])) % Z[2];
  int x2 = (Y / Z[0]) % Z[1];
  int x1 = Y % Z[0];
  int X1 = Z[0];
  int X2 = Z[1];
  int X3 = Z[2];
  int X4 = Z[3];

  switch (dir) {
  case 0: //+X
  {
    int new_x1 = (x1 + nb) % X1;
    if (x1 + nb >= X1 && quda::comm_dim_partitioned(0)) {
      int offset = (x1 + nb - X1) * X4 * X3 * X2 / 2 + (x4 * X3 * X2 + x3 * X2 + x2) / 2;
      return fwd_nbr_spinor[0] + offset * site_size;
    }
    j = (x4 * X3 * X2 * X1 + x3 * X2 * X1 + x2 * X1 + new_x1) / 2;
    break;
  }
  case 1: //-X
  {
    int new_x1 = (x1 - nb + X1) % X1;
    if (x1 - nb < 0 && quda::comm_dim_partitioned(0)) {
      int offset = (x1 + nFace - nb) * X4 * X3 * X2 / 2 + (x4 * X3 * X2 + x3 * X2 + x2) / 2;
      return back_nbr_spinor[0] + offset * site_size;
    }
    j = (x4 * X3 * X2 * X1 + x3 * X2 * X1 + x2 * X1 + new_x1) / 2;
    break;
  }
  case 2: //+Y
  {
    int new_x2 = (x2 + nb) % X2;
    if (x2 + nb >= X2 && quda::comm_dim_partitioned(1)) {
      int offset = (x2 + nb - X2) * X4 * X3 * X1 / 2 + (x4 * X3 * X1 + x3 * X1 + x1) / 2;
      return fwd_nbr_spinor[1] + offset * site_size;
    }
    j = (x4 * X3 * X2 * X1 + x3 * X2 * X1 + new_x2 * X1 + x1) / 2;
    break;
  }
  case 3: // -Y
  {
    int new_x2 = (x2 - nb + X2) % X2;
    if (x2 - nb < 0 && quda::comm_dim_partitioned(1)) {
      int offset = (x2 + nFace - nb) * X4 * X3 * X1 / 2 + (x4 * X3 * X1 + x3 * X1 + x1) / 2;
      return back_nbr_spinor[1] + offset * site_size;
    }
    j = (x4 * X3 * X2 * X1 + x3 * X2 * X1 + new_x2 * X1 + x1) / 2;
    break;
  }
  case 4: //+Z
  {
    int new_x3 = (x3 + nb) % X3;
    if (x3 + nb >= X3 && quda::comm_dim_partitioned(2)) {
      int offset = (x3 + nb - X3) * X4 * X2 * X1 / 2 + (x4 * X2 * X1 + x2 * X1 + x1) / 2;
      return fwd_nbr_spinor[2] + offset * site_size;
    }
    j = (x4 * X3 * X2 * X1 + new_x3 * X2 * X1 + x2 * X1 + x1) / 2;
    break;
  }
  case 5: //-Z
  {
    int new_x3 = (x3 - nb + X3) % X3;
    if (x3 - nb < 0 && quda::comm_dim_partitioned(2)) {
      int offset = (x3 + nFace - nb) * X4 * X2 * X1 / 2 + (x4 * X2 * X1 + x2 * X1 + x1) / 2;
      return back_nbr_spinor[2] + offset * site_size;
    }
    j = (x4 * X3 * X2 * X1 + new_x3 * X2 * X1 + x2 * X1 + x1) / 2;
    break;
  }
  case 6: //+T
  {
    j = neighborIndex_mg(i, oddBit, +nb, 0, 0, 0);
    int x4 = x4_mg(i, oddBit);
    if ((x4 + nb) >= Z[3] && quda::comm_dim_partitioned(3)) {
      int offset = (x4 + nb - Z[3]) * Vsh_t;
      return &fwd_nbr_spinor[3][(offset + j) * site_size];
    }
    break;
  }
  case 7: //-T
  {
    j = neighborIndex_mg(i, oddBit, -nb, 0, 0, 0);
    int x4 = x4_mg(i, oddBit);
    if ((x4 - nb) < 0 && quda::comm_dim_partitioned(3)) {
      int offset = (x4 - nb + nFace) * Vsh_t;
      return &back_nbr_spinor[3][(offset + j) * site_size];
    }
    break;
  }
  default:
    j = -1;
    errorQuda("ERROR: wrong dir");
  }

  return &spinorField[j * site_size];
}

/**
 * @brief Return the pointer to a fermion field as a function of an origin and an offset
 *
 * @param i The checkerboard index of the site
 * @param dir The displacement direction
 * @param oddBit The parity of the site
 * @param spinorField The spinor field
 * @param neighbor_distance Displacement distance
 * @param site_size The number of values in a single spinor (6 for staggered, 24 for Wilson)
 * @return A pointer to the offset fermion field
 */
template <typename Float>
static inline const Float *spinorNeighbor(int i, int dir, int oddBit, const Float *spinorField, int neighbor_distance,
                                          int site_size = 24)
{
  return spinorNeighbor(i, dir, oddBit, spinorField, static_cast<Float**>(nullptr),
                        static_cast<Float**>(nullptr), neighbor_distance, 0, site_size);
}

/**
 * @brief Compute the 4th dimension index for a given 5-d index with or without 4D-PC
 * @tparam type The PCType, either QUDA_5D_PC or QUDA_4D_PC
 * @param i The 5-d index
 * @param oddBit The odd/even bit for the index
 * @return The 4th dimension index
 */
template <QudaPCType type> int x4_5d_mgpu(int i, int oddBit)
{
  int Y = (type == QUDA_5D_PC) ? fullLatticeIndex_5d(i, oddBit) : fullLatticeIndex_5d_4dpc(i, oddBit);
  return (Y / (Z[2] * Z[1] * Z[0])) % Z[3];
}

/**
 * @brief Return the pointer to a 5-d fermion field as a function of an origin and an offset
 *
 * @param i The checkerboard index of the site
 * @param dir The displacement direction
 * @param oddBit The parity of the site
 * @param spinorField The spinor field
 * @param fwd_nbr_spinor The forward ghost region for the spinor field
 * @param back_nbr_spinor The backward ghost region for the spinor field
 * @param neighbor_distance Displacement distance
 * @param nFace The depth of the ghost fields
 * @param site_size The number of values in a single spinor (6 for staggered, 24 for Wilson)
 * @return A pointer to the offset fermion field
 */
template <QudaPCType type, typename Float>
Float *spinorNeighbor_5d(int i, int dir, int oddBit, Float *spinorField, Float **fwd_nbr_spinor,
                         Float **back_nbr_spinor, int neighbor_distance, int nFace, int site_size = 24)
{
  int j;
  int nb = neighbor_distance;
  int Y = (type == QUDA_5D_PC) ? fullLatticeIndex_5d(i, oddBit) : fullLatticeIndex_5d_4dpc(i, oddBit);

  int xs = Y / (Z[3] * Z[2] * Z[1] * Z[0]);
  int x4 = (Y / (Z[2] * Z[1] * Z[0])) % Z[3];
  int x3 = (Y / (Z[1] * Z[0])) % Z[2];
  int x2 = (Y / Z[0]) % Z[1];
  int x1 = Y % Z[0];

  int X1 = Z[0];
  int X2 = Z[1];
  int X3 = Z[2];
  int X4 = Z[3];
  switch (dir) {
  case 0: //+X
  {
    int new_x1 = (x1 + nb) % X1;
    if (x1 + nb >= X1 && quda::comm_dim_partitioned(0)) {
      int offset = ((x1 + nb - X1) * Ls * X4 * X3 * X2 + xs * X4 * X3 * X2 + x4 * X3 * X2 + x3 * X2 + x2) >> 1;
      return fwd_nbr_spinor[0] + offset * site_size;
    }
    j = (xs * X4 * X3 * X2 * X1 + x4 * X3 * X2 * X1 + x3 * X2 * X1 + x2 * X1 + new_x1) >> 1;
    break;
  }
  case 1: //-X
  {
    int new_x1 = (x1 - nb + X1) % X1;
    if (x1 - nb < 0 && quda::comm_dim_partitioned(0)) {
      int offset = ((x1 + nFace - nb) * Ls * X4 * X3 * X2 + xs * X4 * X3 * X2 + x4 * X3 * X2 + x3 * X2 + x2) >> 1;
      return back_nbr_spinor[0] + offset * site_size;
    }
    j = (xs * X4 * X3 * X2 * X1 + x4 * X3 * X2 * X1 + x3 * X2 * X1 + x2 * X1 + new_x1) >> 1;
    break;
  }
  case 2: //+Y
  {
    int new_x2 = (x2 + nb) % X2;
    if (x2 + nb >= X2 && quda::comm_dim_partitioned(1)) {
      int offset = ((x2 + nb - X2) * Ls * X4 * X3 * X1 + xs * X4 * X3 * X1 + x4 * X3 * X1 + x3 * X1 + x1) >> 1;
      return fwd_nbr_spinor[1] + offset * site_size;
    }
    j = (xs * X4 * X3 * X2 * X1 + x4 * X3 * X2 * X1 + x3 * X2 * X1 + new_x2 * X1 + x1) >> 1;
    break;
  }
  case 3: // -Y
  {
    int new_x2 = (x2 - nb + X2) % X2;
    if (x2 - nb < 0 && quda::comm_dim_partitioned(1)) {
      int offset = ((x2 + nFace - nb) * Ls * X4 * X3 * X1 + xs * X4 * X3 * X1 + x4 * X3 * X1 + x3 * X1 + x1) >> 1;
      return back_nbr_spinor[1] + offset * site_size;
    }
    j = (xs * X4 * X3 * X2 * X1 + x4 * X3 * X2 * X1 + x3 * X2 * X1 + new_x2 * X1 + x1) >> 1;
    break;
  }
  case 4: //+Z
  {
    int new_x3 = (x3 + nb) % X3;
    if (x3 + nb >= X3 && quda::comm_dim_partitioned(2)) {
      int offset = ((x3 + nb - X3) * Ls * X4 * X2 * X1 + xs * X4 * X2 * X1 + x4 * X2 * X1 + x2 * X1 + x1) >> 1;
      return fwd_nbr_spinor[2] + offset * site_size;
    }
    j = (xs * X4 * X3 * X2 * X1 + x4 * X3 * X2 * X1 + new_x3 * X2 * X1 + x2 * X1 + x1) >> 1;
    break;
  }
  case 5: //-Z
  {
    int new_x3 = (x3 - nb + X3) % X3;
    if (x3 - nb < 0 && quda::comm_dim_partitioned(2)) {
      int offset = ((x3 + nFace - nb) * Ls * X4 * X2 * X1 + xs * X4 * X2 * X1 + x4 * X2 * X1 + x2 * X1 + x1) >> 1;
      return back_nbr_spinor[2] + offset * site_size;
    }
    j = (xs * X4 * X3 * X2 * X1 + x4 * X3 * X2 * X1 + new_x3 * X2 * X1 + x2 * X1 + x1) >> 1;
    break;
  }
  case 6: //+T
  {
    int x4 = x4_5d_mgpu<type>(i, oddBit);
    if ((x4 + nb) >= Z[3] && quda::comm_dim_partitioned(3)) {
      int offset = ((x4 + nb - Z[3]) * Ls * X3 * X2 * X1 + xs * X3 * X2 * X1 + x3 * X2 * X1 + x2 * X1 + x1) >> 1;
      return fwd_nbr_spinor[3] + offset * site_size;
    }
    j = neighborIndex_5d<type>(i, oddBit, 0, +nb, 0, 0, 0);
    break;
  }
  case 7: //-T
  {
    int x4 = x4_5d_mgpu<type>(i, oddBit);
    if ((x4 - nb) < 0 && quda::comm_dim_partitioned(3)) {
      int offset = ((x4 - nb + nFace) * Ls * X3 * X2 * X1 + xs * X3 * X2 * X1 + x3 * X2 * X1 + x2 * X1 + x1) >> 1;
      return back_nbr_spinor[3] + offset * site_size;
    }
    j = neighborIndex_5d<type>(i, oddBit, 0, -nb, 0, 0, 0);
    break;
  }
  case 8: j = neighborIndex_5d<type>(i, oddBit, +nb, 0, 0, 0, 0); break;
  case 9: j = neighborIndex_5d<type>(i, oddBit, -nb, 0, 0, 0, 0); break;
  default:
    j = -1;
    errorQuda("ERROR: wrong dir");
  }

  return &spinorField[j * site_size];
}

/**
 * @brief Return the pointer to a 5-d fermion field as a function of an origin and an offset
 *
 * @param i The checkerboard index of the site
 * @param dir The displacement direction
 * @param oddBit The parity of the site
 * @param spinorField The spinor field
 * @param neighbor_distance Displacement distance
 * @param site_size The number of values in a single spinor (6 for staggered, 24 for Wilson)
 * @return A pointer to the offset fermion field
 */
template <QudaPCType type, typename Float>
Float *spinorNeighbor_5d(int i, int dir, int oddBit, Float *spinorField, int neighbor_distance = 1, int site_size = 24)
{
  return spinorNeighbor_5d<type>(i, dir, oddBit, spinorField, static_cast<Float **>(nullptr),
                         static_cast<Float**>(nullptr), neighbor_distance, 0, site_size);
}


