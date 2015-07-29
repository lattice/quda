#pragma once

namespace quda {
  /**
     Compute the checkerboard 1-d index from the 4-d coordinate x[] + dx[]

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param dx 4-d shift index
     @param X Full lattice dimensions
   */
  static __device__ __host__ inline int linkIndexShift(int x[], int dx[], const int X[4]) {
    int y[4];
    for ( int i = 0; i < 4; i++ ) y[i] = (x[i] + dx[i] + X[i]) % X[i];
    int idx = (((y[3] * X[2] + y[2]) * X[1] + y[1]) * X[0] + y[0]) >> 1;
    return idx;
  }
  
  /**
     Compute the checkerboard 1-d index from the 4-d coordinate x[] in the mu direction

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param X Full lattice dimensions
   */
  static __device__ __host__ inline int linkIndex(int x[], const int X[4]) {
    int idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
    return idx;
  }

  /**
     Compute the checkerboard 1-d index from the 4-d coordinate x[] -1 in the mu direction

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param X Full lattice dimensions
     @param mu direction in which to subtract 1
   */
  static __device__ __host__ inline int linkIndexM1(int x[], const int X[4], const int mu) {
    int y[4];
    for ( int i = 0; i < 4; i++ ) y[i] = x[i];
    y[mu] = (y[mu] - 1 + X[mu]) % X[mu];
    int idx = (((y[3] * X[2] + y[2]) * X[1] + y[1]) * X[0] + y[0]) >> 1;
    return idx;
  }

  /**
     Compute the full 1-d index from the 4-d coordinate x[] +1 in the mu direction

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param X Full lattice dimensions
     @param mu direction in which to add 1
   */
  static __device__ __host__ inline int linkNormalIndexP1(int x[], const int X[4], const int mu) {
    int y[4];
    for ( int i = 0; i < 4; i++ ) y[i] = x[i];
    y[mu] = (y[mu] + 1 + X[mu]) % X[mu];
    int idx = ((y[3] * X[2] + y[2]) * X[1] + y[1]) * X[0] + y[0];
    return idx;
  }
  
  /**
     Compute the checkerboard 1-d index from the 4-d coordinate x[] +1 in the mu direction

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param X Full lattice dimensions
     @param mu direction in which to add 1
   */  static __device__ __host__ inline int linkIndexP1(int x[], const int X[4], const int mu) {
    int y[4];
    for ( int i = 0; i < 4; i++ ) y[i] = x[i];
    y[mu] = (y[mu] + 1 + X[mu]) % X[mu];
    int idx = (((y[3] * X[2] + y[2]) * X[1] + y[1]) * X[0] + y[0]) >> 1;
    return idx;
  }

  /**
     Compute the 4-d spatial index from the checkerboarded 1-d index at parity parity

     @param x Computed spatial index
     @param cb_index 1-d checkerboarded index
     @param X Full lattice dimensions
     @param parity Site parity
   */
  static __device__ __host__ inline void getCoords(int x[4], int cb_index, const int X[4], int parity) {
    //x[3] = cb_index/(X[2]*X[1]*X[0]/2);
    //x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    //x[1] = (cb_index/(X[0]/2)) % X[1];
    //x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    int za = (cb_index / (X[0] / 2));
    int zb =  (za / X[1]);
    x[1] = za - zb * X[1];
    x[3] = (zb / X[2]);
    x[2] = zb - x[3] * X[2];
    int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
    x[0] = (2 * cb_index + x1odd)  - za * X[0];
    return;
  }
  
  /**
     Compute the 1-d global index from 1-d checkerboard index and
     parity.  This should never be used to index into QUDA fields due
     to the potential of padding between even and odd regions.

     @param cb_index 1-d checkerboard index
     @param X lattice dimensions
     @param parity Site parity
   */
  static __device__ __host__ inline int getIndexFull(int cb_index, const int X[4], int parity) {
    int za = (cb_index / (X[0] / 2));
    int zb =  (za / X[1]);
    int x1 = za - zb * X[1];
    int x3 = (zb / X[2]);
    int x2 = zb - x3 * X[2];
    int x1odd = (x1 + x2 + x3 + parity) & 1;
    return 2 * cb_index + x1odd;  
  }
  
} // namespace quda
