#include "host_utils.h"
#include "index_utils.hpp"

int fullLatticeIndex(int i, int oddBit)
{
  /*
    int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2);
    return 2*i + (boundaryCrossings + oddBit) % 2;
  */

  int X1 = Z[0];
  int X2 = Z[1];
  int X3 = Z[2];
  // int X4 = Z[3];
  int X1h = X1 / 2;

  int sid = i;
  int za = sid / X1h;
  // int x1h = sid - za*X1h;
  int zb = za / X2;
  int x2 = za - zb * X2;
  int x4 = zb / X3;
  int x3 = zb - x4 * X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  // int x1 = 2*x1h + x1odd;
  int X = 2 * sid + x1odd;

  return X;
}

int fullLatticeIndex(int dim[4], int index, int oddBit)
{

  int za = index / (dim[0] >> 1);
  int zb = za / dim[1];
  int x2 = za - zb * dim[1];
  int x4 = zb / dim[2];
  int x3 = zb - x4 * dim[2];

  return 2 * index + ((x2 + x3 + x4 + oddBit) & 1);
}

int fullLatticeIndex_5d(int i, int oddBit)
{
  int boundaryCrossings
    = i / (Z[0] / 2) + i / (Z[1] * Z[0] / 2) + i / (Z[2] * Z[1] * Z[0] / 2) + i / (Z[3] * Z[2] * Z[1] * Z[0] / 2);
  return 2 * i + (boundaryCrossings + oddBit) % 2;
}

int fullLatticeIndex_5d_4dpc(int i, int oddBit)
{
  int boundaryCrossings = i / (Z[0] / 2) + i / (Z[1] * Z[0] / 2) + i / (Z[2] * Z[1] * Z[0] / 2);
  return 2 * i + (boundaryCrossings + oddBit) % 2;
}

int neighborIndex(int i, int oddBit, int dx4, int dx3, int dx2, int dx1)
{
  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y / (Z[2] * Z[1] * Z[0]);
  int x3 = (Y / (Z[1] * Z[0])) % Z[2];
  int x2 = (Y / Z[0]) % Z[1];
  int x1 = Y % Z[0];

  // assert (oddBit == (x+y+z+t)%2);

  x4 = (x4 + dx4 + Z[3]) % Z[3];
  x3 = (x3 + dx3 + Z[2]) % Z[2];
  x2 = (x2 + dx2 + Z[1]) % Z[1];
  x1 = (x1 + dx1 + Z[0]) % Z[0];

  return (x4 * (Z[2] * Z[1] * Z[0]) + x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
}

int neighborIndex(int dim[4], int index, int oddBit, int dx[4])
{
  const int fullIndex = fullLatticeIndex(dim, index, oddBit);

  int x[4];
  x[3] = fullIndex / (dim[2] * dim[1] * dim[0]);
  x[2] = (fullIndex / (dim[1] * dim[0])) % dim[2];
  x[1] = (fullIndex / dim[0]) % dim[1];
  x[0] = fullIndex % dim[0];

  for (int dir = 0; dir < 4; ++dir) x[dir] = (x[dir] + dx[dir] + dim[dir]) % dim[dir];

  return (((x[3] * dim[2] + x[2]) * dim[1] + x[1]) * dim[0] + x[0]) / 2;
}

int neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1)
{
  int oddBit = 0;
  int half_idx = i;
  if (i >= Vh) {
    oddBit = 1;
    half_idx = i - Vh;
  }

  int nbr_half_idx = neighborIndex(half_idx, oddBit, dx4, dx3, dx2, dx1);
  int oddBitChanged = (dx4 + dx3 + dx2 + dx1) % 2;
  if (oddBitChanged) { oddBit = 1 - oddBit; }
  int ret = nbr_half_idx;
  if (oddBit) { ret = Vh + nbr_half_idx; }

  return ret;
}

int neighborIndexFullLattice(int dim[4], int index, int dx[4])
{
  const int volume = dim[0] * dim[1] * dim[2] * dim[3];
  const int halfVolume = volume / 2;
  int oddBit = 0;
  int halfIndex = index;

  if (index >= halfVolume) {
    oddBit = 1;
    halfIndex = index - halfVolume;
  }

  int neighborHalfIndex = neighborIndex(dim, halfIndex, oddBit, dx);

  int oddBitChanged = (dx[0] + dx[1] + dx[2] + dx[3]) % 2;
  if (oddBitChanged) { oddBit = 1 - oddBit; }

  return neighborHalfIndex + oddBit * halfVolume;
}
