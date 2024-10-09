#pragma once

#include <host_utils.h>

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


// i represents a "half index" into an even or odd "half lattice".
// when oddBit={0,1} the half lattice is {even,odd}.
//
// the displacements, such as dx, refer to the full lattice coordinates.
//
// neighborIndex() takes a "half index", displaces it, and returns the
// new "half index", which can be an index into either the even or odd lattices.
// displacements of magnitude one always interchange odd and even lattices.
//
//
template <QudaPCType type> int neighborIndex_5d(int i, int oddBit, int dxs, int dx4, int dx3, int dx2, int dx1)
{
  // fullLatticeIndex was modified for fullLatticeIndex_4d.  It is in util_quda.cpp.
  // This code bit may not properly perform 5dPC.
  int X = type == QUDA_5D_PC ? fullLatticeIndex_5d(i, oddBit) : fullLatticeIndex_5d_4dpc(i, oddBit);
  // Checked that this matches code in dslash_core_ante.h.
  int xs = X / (Z[3] * Z[2] * Z[1] * Z[0]);
  int x4 = (X / (Z[2] * Z[1] * Z[0])) % Z[3];
  int x3 = (X / (Z[1] * Z[0])) % Z[2];
  int x2 = (X / Z[0]) % Z[1];
  int x1 = X % Z[0];
  // Displace and project back into domain 0,...,Ls-1.
  // Note that we add Ls to avoid the negative problem
  // of the C % operator.
  xs = (xs + dxs + Ls) % Ls;
  // Etc.
  x4 = (x4 + dx4 + Z[3]) % Z[3];
  x3 = (x3 + dx3 + Z[2]) % Z[2];
  x2 = (x2 + dx2 + Z[1]) % Z[1];
  x1 = (x1 + dx1 + Z[0]) % Z[0];
  // Return linear half index.  Remember that integer division
  // rounds down.
  return (xs * (Z[3] * Z[2] * Z[1] * Z[0]) + x4 * (Z[2] * Z[1] * Z[0]) + x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
}

template <QudaPCType type> int neighborIndex_5d_mgpu(int i, int oddBit, int dxs, int dx4, int dx3, int dx2, int dx1)
{
  int ret;

  int Y = (type == QUDA_5D_PC) ? fullLatticeIndex_5d(i, oddBit) : fullLatticeIndex_5d_4dpc(i, oddBit);

  int xs = Y / (Z[3] * Z[2] * Z[1] * Z[0]);
  int x4 = (Y / (Z[2] * Z[1] * Z[0])) % Z[3];
  int x3 = (Y / (Z[1] * Z[0])) % Z[2];
  int x2 = (Y / Z[0]) % Z[1];
  int x1 = Y % Z[0];
  int ghost_x4 = x4 + dx4;

  xs = (xs + dxs + Ls) % Ls;
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
