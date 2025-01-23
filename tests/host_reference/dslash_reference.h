#pragma once

#include <array>
#include <host_utils.h>
#include <comm_quda.h>
#include <gauge_field.h>

#include "index_utils.hpp"

/**
 * @brief Computes the element-wise sum of two arrays, storing it into a third
 *
 * @tparam real_t The floating-point type of the array elements
 * @param[out] dst The output array to store the sum
 * @param[in] a The first input array
 * @param[in] b The second input array
 * @param[in] len The number of elements in the input arrays
 */
template <typename real_t> static inline void sum(real_t *dst, const real_t *a, const real_t *b, int len)
{
  for (int i = 0; i < len; i++) dst[i] = a[i] + b[i];
}

/**
 * @brief Computes the element-wise difference of two arrays, storing it into a third
 *
 * @tparam real_t The floating-point type of the array elements
 * @param[out] dst The output array to store the sum
 * @param[in] a The first input array
 * @param[in] b The second input array
 * @param[in] len The number of elements in the input arrays
 */
template <typename real_t> static inline void sub(real_t *dst, const real_t *a, const real_t *b, int len)
{
  for (int i = 0; i < len; i++) dst[i] = a[i] - b[i];
}

/**
 * @brief Rescale an array by a real scalar, storing it into a separate array
 *
 * @tparam real_t The floating-point type of the array elements
 * @param[out] dst The output array to store the rescaling
 * @param[in] a The rescaling factor
 * @param[in] x The input array
 * @param[in] len The number of elements in the input array
 */
template <typename real_t> static inline void ax(real_t *dst, real_t a, const real_t *x, int len)
{
  for (int i = 0; i < len; i++) dst[i] = a * x[i];
}

/**
 * @brief Perform the operation y[i] = a*x[i] + y[i]
 *
 * @tparam real_t The floating-point type of the array elements
 * @param[in] a The rescaling factor
 * @param[in] x The input array
 * @param[in,out] y An input and output array that is accumulated into
 * @param[in] len The number of elements in the input array
 */
template <typename real_t> static inline void axpy(real_t a, const real_t *x, real_t *y, int len)
{
  for (int i = 0; i < len; i++) y[i] = a * x[i] + y[i];
}

/**
 * @brief Perform the operation y[i] = a*x[i] + b*y[i]
 *
 * @tparam real_t The floating-point type of the array elements
 * @param[in] a One rescaling factor
 * @param[in] x The input array
 * @param[in] b The second rescaling factor
 * @param[in,out] y An input and output array that is accumulated into
 * @param[in] len The number of elements in the input array
 */
template <typename real_t> static inline void axpby(real_t a, const real_t *x, real_t b, real_t *y, int len)
{
  for (int i = 0; i < len; i++) y[i] = a * x[i] + b * y[i];
}

/**
 * @brief Perform the operation y[i] = a*x[i] - y[i]
 *
 * @tparam real_t The floating-point type of the array elements
 * @param[in] a The rescaling factor
 * @param[in] x The input array
 * @param[in,out] y An input and output array that is accumulated into
 * @param[in] len The number of elements in the input array
 */
template <typename real_t> static inline void axmy(const real_t *x, real_t a, real_t *y, int len)
{
  for (int i = 0; i < len; i++) y[i] = a * x[i] - y[i];
}

/**
 * @brief Perform the element-wise norm2 of an array
 *
 * @tparam real_t The floating-point type of the array elements
 * @param[in] x The input array
 * @param[in] len The number of elements in the input array
 */
template <typename real_t> static double norm2(const real_t *v, int len)
{
  double sum = 0.0;
  for (int i = 0; i < len; i++) sum += v[i] * v[i];
  return sum;
}

/**
 * @brief Perform the element-wise negation of an array
 *
 * @tparam real_t The floating-point type of the array elements
 * @param[in,out] x The input array
 * @param[in] len The number of elements in the input array
 */
template <typename real_t> static inline void negx(real_t *x, int len)
{
  for (int i = 0; i < len; i++) x[i] = -x[i];
}

/**
 * @brief Perform the element-wise complex dot product of a 3-component complex array
 *
 * @tparam real_t The floating-point type of the array elements
 * @param[out] res The output values
 * @param[in] a The first input array
 * @param[in] b The first input array
 */
template <typename real_t> static inline void dot(real_t *res, const real_t *a, const real_t *b)
{
  res[0] = res[1] = 0;
  for (int m = 0; m < 3; m++) {
    real_t a_re = a[2 * m + 0];
    real_t a_im = a[2 * m + 1];
    real_t b_re = b[2 * m + 0];
    real_t b_im = b[2 * m + 1];
    res[0] += a_re * b_re - a_im * b_im;
    res[1] += a_re * b_im + a_im * b_re;
  }
}

/**
 * @brief Perform the Hermitian conjugate of an SU(3) matrix, storing it in a second matrix
 *
 * @tparam real_t The floating-point type of the matrix elements
 * @param[out] res The output SU(3) matrix
 * @param[in] mat The input SU(3) matrix
 */
template <typename real_t> static inline void su3Transpose(real_t *res, const real_t *mat)
{
  for (int m = 0; m < 3; m++) {
    for (int n = 0; n < 3; n++) {
      res[m * (3 * 2) + n * (2) + 0] = +mat[n * (3 * 2) + m * (2) + 0];
      res[m * (3 * 2) + n * (2) + 1] = -mat[n * (3 * 2) + m * (2) + 1];
    }
  }
}

/**
 * @brief Perform an SU(3) matrix-vector multiplication
 *
 * @tparam real_t The floating-point type of the matrix and vector elements
 * @param[out] res The output 3-component vector
 * @param[in] mat The input SU(3) matrix
 * @param[in] vec The input 3-component vector
 */
template <typename real_t> static inline void su3Mul(real_t *res, const real_t *mat, const real_t *vec)
{
  for (int n = 0; n < 3; n++) dot(&res[n * (2)], &mat[n * (3 * 2)], vec);
}

/**
 * @brief Perform an SU(3) matrix-vector multiplication, using the Hermitian conjugate of the matrix
 *
 * @tparam real_t The floating-point type of the matrix and vector elements
 * @param[out] res The output 3-component vector
 * @param[in] mat The input SU(3) matrix
 * @param[in] vec The input 3-component vector
 */
template <typename real_t> static inline void su3Tmul(real_t *res, const real_t *mat, const real_t *vec)
{
  real_t matT[3 * 3 * 2];
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
 * @param laplace3D Whether we are working on the 3-d Laplace operator
 * @param src_idx The source index we working on (when doing mutil-RHS)
 * @return The residual and HQ residual (if requested)
 */
std::array<double, 2> verifyStaggeredInversion(quda::ColorSpinorField &in, quda::ColorSpinorField &out,
                                               quda::GaugeField &fat_link, quda::GaugeField &long_link,
                                               QudaInvertParam &inv_param, int laplace3D, int src_idx);

/**
 * @brief Verify a single- or multi-shift staggered inversion on the host
 *
 * @param in The initial rhs
 * @param out The solutions to (A + shift) out = in for multiple shifts; shift == 0 for a single shift solve
 * @param fat_link The fat links in the context of an ASQTAD solve; otherwise the base gauge links with phases applied
 * @param long_link The long links; null for naive staggered and Laplace
 * @param inv_param Invert params, used to query the solve type, etc, also includes the shifts
 * @param laplace3D Whether we are working on the 3-d Laplace operator
 * @param src_idx The source index we working on (when doing mutil-RHS)
 * @return The residual and HQ residual (if requested)
 */
std::array<double, 2> verifyStaggeredInversion(quda::ColorSpinorField &in,
                                               std::vector<quda::ColorSpinorField> &out_vector,
                                               quda::GaugeField &fat_link, quda::GaugeField &long_link,
                                               QudaInvertParam &inv_param, int laplace3D, int src_idx = 0);

/**
 * @brief Verify a staggered-type eigenvector
 *
 * @param spinor The host eigenvector to be verified
 * @param lambda The host eigenvalue(s) to be verified
 * @param i The number of the eigenvalue, only used when printing outputs
 * @param eig_param Eigensolve params, used to query the operator type, etc
 * @param fat_link The fat links in the context of an ASQTAD solve; otherwise the base gauge links with phases applied
 * @param long_link The long links; null for naive staggered and Laplace
 * @param laplace3D Whether we are working on the 3-d Laplace operator
 * @return The residual norm
 */
double verifyStaggeredTypeEigenvector(quda::ColorSpinorField &spinor, const std::vector<double _Complex> &lambda, int i,
                                      QudaEigParam &eig_param, quda::GaugeField &fat_link, quda::GaugeField &long_link,
                                      int laplace3D);

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
 * @param laplace3D Whether we are working on the 3-d Laplace operator
 * @return The residual norm
 */
double verifyStaggeredTypeSingularVector(quda::ColorSpinorField &spinor_left, quda::ColorSpinorField &spinor_right,
                                         const std::vector<double _Complex> &sigma, int i, QudaEigParam &eig_param,
                                         quda::GaugeField &fat_link, quda::GaugeField &long_link, int laplace3D);

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
 * @tparam real_t The data type of the fields (e.g., float or double)
 * @param[in] i The checkerboard index of the site
 * @param[in] dir The displacement direction
 * @param[in] oddBit The parity of the site
 * @param[in] gaugeEven The even gauge fields stored in a QDP layout
 * @param[in] gaugeOdd The odd gauge fields stored in a QDP layout
 * @param[in] ghostGaugeEven The even-parity gauge ghost fields
 * @param[in] ghostGaugeOdd The odd-parity gauge ghost fields
 * @param[in] n_ghost_faces The depth of the ghost fields
 * @param[in] nbr_distance Displacement distance
 * @return A pointer to the offset gauge link
 */
template <typename real_t>
const real_t *gaugeLink(int i, int dir, int oddBit, const real_t *const *gaugeEven, const real_t *const *gaugeOdd,
                        const real_t *const *ghostGaugeEven, const real_t *const *ghostGaugeOdd, int n_ghost_faces,
                        int nbr_distance)
{
  int j;
  int d = nbr_distance;
  const real_t *const *gaugeField = [&]() -> const real_t *const * {
    if (dir % 2 == 0)
      return (oddBit ? gaugeOdd : gaugeEven);
    else
      return (oddBit ? gaugeEven : gaugeOdd);
  }();

  if (dir % 2 == 0) {
    j = i;
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

    switch (dir) {
    case 1: { //-X direction
      int new_x1 = (x1 - d + X1) % X1;
      if (x1 - d < 0 && quda::comm_dim_partitioned(0)) {
        const real_t *ghostGaugeField = (oddBit ? ghostGaugeEven[0] : ghostGaugeOdd[0]);
        int offset = (n_ghost_faces + x1 - d) * X4 * X3 * X2 / 2 + (x4 * X3 * X2 + x3 * X2 + x2) / 2;
        return &ghostGaugeField[offset * (3 * 3 * 2)];
      }
      j = (x4 * X3 * X2 * X1 + x3 * X2 * X1 + x2 * X1 + new_x1) / 2;
      break;
    }
    case 3: { //-Y direction
      int new_x2 = (x2 - d + X2) % X2;
      if (x2 - d < 0 && quda::comm_dim_partitioned(1)) {
        const real_t *ghostGaugeField = (oddBit ? ghostGaugeEven[1] : ghostGaugeOdd[1]);
        int offset = (n_ghost_faces + x2 - d) * X4 * X3 * X1 / 2 + (x4 * X3 * X1 + x3 * X1 + x1) / 2;
        return &ghostGaugeField[offset * (3 * 3 * 2)];
      }
      j = (x4 * X3 * X2 * X1 + x3 * X2 * X1 + new_x2 * X1 + x1) / 2;
      break;
    }
    case 5: { //-Z direction
      int new_x3 = (x3 - d + X3) % X3;
      if (x3 - d < 0 && quda::comm_dim_partitioned(2)) {
        const real_t *ghostGaugeField = (oddBit ? ghostGaugeEven[2] : ghostGaugeOdd[2]);
        int offset = (n_ghost_faces + x3 - d) * X4 * X2 * X1 / 2 + (x4 * X2 * X1 + x2 * X1 + x1) / 2;
        return &ghostGaugeField[offset * (3 * 3 * 2)];
      }
      j = (x4 * X3 * X2 * X1 + new_x3 * X2 * X1 + x2 * X1 + x1) / 2;
      break;
    }
    case 7: { //-T direction
      int new_x4 = (x4 - d + X4) % X4;
      if (x4 - d < 0 && quda::comm_dim_partitioned(3)) {
        const real_t *ghostGaugeField = (oddBit ? ghostGaugeEven[3] : ghostGaugeOdd[3]);
        int offset = (n_ghost_faces + x4 - d) * X1 * X2 * X3 / 2 + (x3 * X2 * X1 + x2 * X1 + x1) / 2;
        return &ghostGaugeField[offset * (3 * 3 * 2)];
      }
      j = (new_x4 * (X3 * X2 * X1) + x3 * (X2 * X1) + x2 * (X1) + x1) / 2;
      break;
    } // 7

    default: j = -1; errorQuda("wrong dir");
    }
  }

  return &gaugeField[dir / 2][j * (3 * 3 * 2)];
}

/**
 * @brief Return the pointer to a gauge link as a function of an origin and an offset
 *
 * @tparam real_t The data type of the fields (e.g., float or double)
 * @param[in] i The checkerboard index of the site
 * @param[in] dir The displacement direction
 * @param[in] oddBit The parity of the site
 * @param[in] gaugeEven The even gauge fields stored in a QDP layout
 * @param[in] gaugeOdd The odd gauge fields stored in a QDP layout
 * @param[in] nbr_distance Displacement distance
 * @return A pointer to the offset gauge link
 */
template <typename real_t>
const real_t *gaugeLink(int i, int dir, int oddBit, real_t **gaugeEven, real_t **gaugeOdd, int nbr_distance)
{
  return gaugeLink(i, dir, oddBit, gaugeEven, gaugeOdd, static_cast<const real_t *const *>(nullptr),
                   static_cast<const real_t *const *>(nullptr), 0, nbr_distance);
}

/**
 * @brief Compute the 4th dimension index for a given checkerboard index
 * @param[in] i The checkerboard index
 * @param[in] oddBit The odd/even bit for the index
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
 * @tparam real_t The data type of the fields (e.g., float or double)
 * @param[in] i The checkerboard index of the site
 * @param[in] dir The displacement direction
 * @param[in] oddBit The parity of the site
 * @param[in] spinorField The spinor field
 * @param[in] fwd_nbr_spinor The forward ghost region for the spinor field
 * @param[in] back_nbr_spinor The backward ghost region for the spinor field
 * @param[in] neighbor_distance Displacement distance
 * @param[in] nFace The depth of the ghost fields
 * @param[in] site_size The number of values in a single spinor (6 for staggered, 24 for Wilson)
 * @return A pointer to the offset fermion field
 */
template <typename real_t>
const real_t *spinorNeighbor(int i, int dir, int oddBit, const real_t *spinorField, const real_t *const *fwd_nbr_spinor,
                             const real_t *const *back_nbr_spinor, int neighbor_distance, int nFace, int site_size = 24)
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
  default: j = -1; errorQuda("ERROR: wrong dir");
  }

  return &spinorField[j * site_size];
}

/**
 * @brief Return the pointer to a fermion field as a function of an origin and an offset
 *
 * @tparam real_t The data type of the fields (e.g., float or double)
 * @param[in] i The checkerboard index of the site
 * @param[in] dir The displacement direction
 * @param[in] oddBit The parity of the site
 * @param[in] spinorField The spinor field
 * @param[in] neighbor_distance Displacement distance
 * @param[in] site_size The number of values in a single spinor (6 for staggered, 24 for Wilson)
 * @return A pointer to the offset fermion field
 */
template <typename real_t>
const real_t *spinorNeighbor(int i, int dir, int oddBit, const real_t *spinorField, int neighbor_distance,
                             int site_size = 24)
{
  return spinorNeighbor(i, dir, oddBit, spinorField, static_cast<const real_t *const *>(nullptr),
                        static_cast<const real_t *const *>(nullptr), neighbor_distance, 0, site_size);
}

/**
 * @brief Compute the 4th dimension index for a given 5-d index with or without 4D-PC
 *
 * @tparam type The PCType, either QUDA_5D_PC or QUDA_4D_PC
 * @param[in] i The 5-d index
 * @param[in] oddBit The odd/even bit for the index
 * @return The 4th dimension index
 */
template <QudaPCType type> int x4_5d_mgpu(int i, int oddBit)
{
  int Y = (type == QUDA_5D_PC) ? fullLatticeIndex_5d(i, oddBit) : fullLatticeIndex_5d_4dpc(i, oddBit);
  return (Y / (Z[2] * Z[1] * Z[0])) % Z[3];
}

/**
 * @brief Return the pointer to a 5-d fermion field as a function of an origin and an offset
 * @tparam type The PCType, either QUDA_5D_PC or QUDA_4D_PC
 * @tparam real_t The data type of the fields (e.g., float or double)
 * @param[in] i The checkerboard index of the site
 * @param[in] dir The displacement direction
 * @param[in] oddBit The parity of the site
 * @param[in] spinorField The spinor field
 * @param[in] fwd_nbr_spinor The forward ghost region for the spinor field
 * @param[in] back_nbr_spinor The backward ghost region for the spinor field
 * @param[in] neighbor_distance Displacement distance
 * @param[in] nFace The depth of the ghost fields
 * @param[in] site_size The number of values in a single spinor (6 for staggered, 24 for Wilson)
 * @return A pointer to the offset fermion field
 */
template <QudaPCType type, typename real_t>
const real_t *spinorNeighbor_5d(int i, int dir, int oddBit, const real_t *spinorField,
                                const real_t *const *fwd_nbr_spinor, const real_t *const *back_nbr_spinor,
                                int neighbor_distance, int nFace, int site_size = 24)
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
  default: j = -1; errorQuda("ERROR: wrong dir");
  }

  return &spinorField[j * site_size];
}

/**
 * @brief Return the pointer to a 5-d fermion field as a function of an origin and an offset
 *
 * @tparam type The PCType, either QUDA_5D_PC or QUDA_4D_PC
 * @tparam real_t The data type of the fields (e.g., float or double)
 * @param i The checkerboard index of the site
 * @param dir The displacement direction
 * @param oddBit The parity of the site
 * @param spinorField The spinor field
 * @param neighbor_distance Displacement distance
 * @param site_size The number of values in a single spinor (6 for staggered, 24 for Wilson)
 * @return A pointer to the offset fermion field
 */
template <QudaPCType type, typename real_t>
real_t *spinorNeighbor_5d(int i, int dir, int oddBit, real_t *spinorField, int neighbor_distance = 1, int site_size = 24)
{
  return spinorNeighbor_5d<type>(i, dir, oddBit, spinorField, static_cast<real_t **>(nullptr),
                                 static_cast<real_t **>(nullptr), neighbor_distance, 0, site_size);
}
