#pragma once

#include "host_utils.h"

// taken from include/index_helper.cuh

/**
 * @brief Compute the 4-d spatial index from the checkerboarded 1-d index at parity parity
 *
 * @param[out] x Computed spatial index
 * @param[in] cb_index 1-d checkerboarded index
 * @param[in] X Full lattice dimensions
 * @param[in] X0h Half of x-dim lattice dimension
 * @param[in] parity Site parity
 * @return Full linear lattice index
 */
template <typename Coord, typename I, typename J>
inline int getCoordsCB(Coord &x, int cb_index, const I &X, J X0h, int parity)
{
  // x[3] = cb_index/(X[2]*X[1]*X[0]/2);
  // x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
  // x[1] = (cb_index/(X[0]/2)) % X[1];
  // x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

  int za = (cb_index / X0h);
  int zb = (za / X[1]);
  x[1] = (za - zb * X[1]);
  x[3] = (zb / X[2]);
  x[2] = (zb - x[3] * X[2]);
  int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
  int x_idx = 2 * cb_index + x1odd;
  x[0] = (x_idx - za * X[0]);
  return x_idx;
}

/**
 * @brief Compute the 4-d spatial index from the checkerboarded 1-d index
 *        at parity parity.  Wrapper around getCoordsCB.
 *
 * @param[out] x Computed spatial index
 * @param[in] cb_index 1-d checkerboarded index
 * @param[in] X Full lattice dimensions
 * @param[in] parity Site parity
 * @return Full linear lattice index
 */
template <typename Coord, typename I> inline int getCoords(Coord &x, int cb_index, const I &X, int parity)
{
  return getCoordsCB(x, cb_index, X, X[0] >> 1, parity);
}

/**
 * @brief Calculate the full index from the checkerboard index and a parity
 *
 * @param i The checkerboard index of the current site
 * @param oddBit The parity of the input index (0 for even, 1 for odd)
 * @return The full index of the current site
 */
int fullLatticeIndex(int i, int oddBit);

/**
 * @brief Calculate the full index from the checkerboard index and a parity relative
 *        to a specified local volume
 *
 * @param dim Local volume
 * @param i The checkerboard index of the current site
 * @param oddBit The parity of the input index (0 for even, 1 for odd)
 * @return The full index of the current site
 */
int fullLatticeIndex(int dim[4], int index, int oddBit);

/**
 * @brief Calculate the full 5-d index from the checkerboard index and a parity
 *
 * @param i The 5-d checkerboard index of the current site
 * @param oddBit The parity of the input index (0 for even, 1 for odd)
 * @return The full index of the current site
 */
int fullLatticeIndex_5d(int i, int oddBit);

/**
 * @brief Calculate the full 4-d preconditioned index from the checkerboard index and a parity
 *
 * @param i The 4-d checkerboard index of the current site
 * @param oddBit The 4-d parity of the input index (0 for even, 1 for odd)
 * @return The full index of the current site
 */
int fullLatticeIndex_5d_4dpc(int i, int oddBit);

/**
 * @brief Calculate the checkerboard neighbor index from a checkerboard index and parity
 *
 * @param i The checkerboard index of the current site
 * @param oddBit The parity of the input index (0 for even, 1 for odd)
 * @param dx4 The displacement in the t direction
 * @param dx3 The displacement in the z direction
 * @param dx2 The displacement in the y direction
 * @param dx1 The displacement in the x direction
 * @return The checkerboard index of the neighbor site
 */
int neighborIndex(int i, int oddBit, int dx4, int dx3, int dx2, int dx1);

/**
 * @brief Calculate the checkerboard neighbor index from a checkerboard index and parity
 *        relative to a specified local volume
 *
 * @param dim Local volume
 * @param i The checkerboard index of the current site
 * @param oddBit The parity of the input index (0 for even, 1 for odd)
 * @param dx Coordinate displacement
 * @return The checkerboard index of the neighbor site
 */
int neighborIndex(int dim[4], int index, int oddBit, int dx[4]);

/**
 * @brief Calculate the full neighbor index from a full index
 *
 * @param i The full-lattice index of the current site
 * @param dx4 The displacement in the t direction
 * @param dx3 The displacement in the z direction
 * @param dx2 The displacement in the y direction
 * @param dx1 The displacement in the x direction
 * @return The full lattice index of the neighbor site
 */
int neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1);

/**
 * @brief Calculate the full neighbor index from a full index and parity
 *        relative to a specified local volume
 *
 * @param dim Local volume
 * @param i The full-lattice index of the current site
 * @param dx Coordinate displacement
 * @return The full lattice index of the neighbor site
 */
int neighborIndexFullLattice(int dim[4], int index, int dx[4]);

/**
 * @brief Calculate the checkerboard neighbor index from a checkerboard index and parity
 *        relative to a specified local volume for a 5-d field
 *
 * @tparam type Domain wall preconditioning type (4 or 5 dimensions)
 * @param i The checkerboard index of the current site
 * @param oddBit The parity of the input index (0 for even, 1 for odd)
 * @param dxs The displacement in the 5th direction
 * @param dx4 The displacement in the t direction
 * @param dx3 The displacement in the z direction
 * @param dx2 The displacement in the y direction
 * @param dx1 The displacement in the x direction
 * @return The checkerboard index of the neighbor site
 */
template <QudaPCType type> int neighborIndex_5d(int i, int oddBit, int dxs, int dx4, int dx3, int dx2, int dx1)
{
  int ret;
  // fullLatticeIndex was modified for fullLatticeIndex_4d.  It is in util_quda.cpp.
  // This code bit may not properly perform 5dPC.
  int X = type == QUDA_5D_PC ? fullLatticeIndex_5d(i, oddBit) : fullLatticeIndex_5d_4dpc(i, oddBit);
  // Checked that this matches code in dslash_core_ante.h.
  int xs = X / (Z[3] * Z[2] * Z[1] * Z[0]);
  int x4 = (X / (Z[2] * Z[1] * Z[0])) % Z[3];
  int x3 = (X / (Z[1] * Z[0])) % Z[2];
  int x2 = (X / Z[0]) % Z[1];
  int x1 = X % Z[0];
  int ghost_x4 = x4 + dx4;

  // Displace and project back into domain 0,...,Ls-1.
  // Note that we add Ls to avoid the negative problem
  // of the C % operator.
  xs = (xs + dxs + Ls) % Ls;
  // Etc.
  x4 = (x4 + dx4 + Z[3]) % Z[3];
  x3 = (x3 + dx3 + Z[2]) % Z[2];
  x2 = (x2 + dx2 + Z[1]) % Z[1];
  x1 = (x1 + dx1 + Z[0]) % Z[0];

  if ((ghost_x4 >= 0 && ghost_x4) < Z[3] || !quda::comm_dim_partitioned(3)) {
    ret = (xs * Z[3] * Z[2] * Z[1] * Z[0] + x4 * Z[2] * Z[1] * Z[0] + x3 * Z[1] * Z[0] + x2 * Z[0] + x1) >> 1;
  } else {
    ret = (xs * Z[2] * Z[1] * Z[0] + x3 * Z[1] * Z[0] + x2 * Z[0] + x1) >> 1;
  }

  return ret;
}
