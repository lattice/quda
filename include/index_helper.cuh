#pragma once

namespace quda {
  /**
     Compute the checkerboard 1-d index from the 4-d coordinate x[] + dx[]

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param dx 4-d shift index
     @param X Full lattice dimensions
   */
  template <typename I, typename J, typename K>
  static __device__ __host__ inline int linkIndexShift(const I x[], const J dx[], const K X[4]) {
    int y[4];
#pragma unroll
    for ( int i = 0; i < 4; i++ ) y[i] = (x[i] + dx[i] + X[i]) % X[i];
    int idx = (((y[3] * X[2] + y[2]) * X[1] + y[1]) * X[0] + y[0]) >> 1;
    return idx;
  }
  
  /**
     Compute the checkerboard 1-d index from the 4-d coordinate x[] + dx[]

     @return 1-d checkerboard index
     @param y new 4-d lattice index
     @param x original 4-d lattice index
     @param dx 4-d shift index
     @param X Full lattice dimensions
   */
  template <typename I, typename J, typename K>
  static __device__ __host__ inline int linkIndexShift(I y[], const I x[], const J dx[], const K X[4]) {
#pragma unroll
    for ( int i = 0; i < 4; i++ ) y[i] = (x[i] + dx[i] + X[i]) % X[i];
    int idx = (((y[3] * X[2] + y[2]) * X[1] + y[1]) * X[0] + y[0]) >> 1;
    return idx;
  }

  /**
     Compute the checkerboard 1-d index from the 4-d coordinate x[]

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param X Full lattice dimensions
   */
  template <typename I>
  static __device__ __host__ inline int linkIndex(const int x[], const I X[4]) {
    int idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
    return idx;
  }

  /**
     Compute the checkerboard 1-d index from the 4-d coordinate x[]

     @return 1-d checkerboard index
     @param y copy of 4-d lattice index
     @param x 4-d lattice index
     @param X Full lattice dimensions
   */
  template <typename I>
  static __device__ __host__ inline int linkIndex(int y[], const int x[], const I X[4]) {
    int idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
    y[0] = x[0]; y[1] = x[1]; y[2] = x[2]; y[3] = x[3];
    return idx;
  }

/**
     Compute the checkerboard 1-d index from the 4-d coordinate x[] -n in the mu direction

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param X Full lattice dimensions
     @param mu direction in which to subtract 1
   */
  template <typename I, int n>
  static __device__ __host__ inline int linkIndexDn(const int x[], const I X[4], const int mu) {
    int y[4];
#pragma unroll
    for ( int i = 0; i < 4; i++ ) y[i] = x[i];
    y[mu] = (y[mu] +n + X[mu]) % X[mu];
    int idx = (((y[3] * X[2] + y[2]) * X[1] + y[1]) * X[0] + y[0]) >> 1;
    return idx;
  }

  /**
     Compute the checkerboard 1-d index from the 4-d coordinate x[] -1 in the mu direction

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param X Full lattice dimensions
     @param mu direction in which to subtract 1
   */
  template <typename I>
  static __device__ __host__ inline int linkIndexM1(const int x[], const I X[4], const int mu) {
    return linkIndexDn<I,-1>(x, X, mu);
  }

  template <typename I>
  static __device__ __host__ inline int linkIndexM3(const int x[], const I X[4], const int mu) {
    return linkIndexDn<I,-3>(x, X, mu);
  }

  /**
     Compute the full 1-d index from the 4-d coordinate x[] +1 in the mu direction

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param X Full lattice dimensions
     @param mu direction in which to add 1
   */
  template <typename I>
  static __device__ __host__ inline int linkNormalIndexP1(const int x[], const I X[4], const int mu) {
    int y[4];
#pragma unroll
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
   */  
  template <typename I>
  static __device__ __host__ inline int linkIndexP1(const int x[], const I X[4], const int mu) {
    return linkIndexDn<I,1>(x, X, mu);
  }

  template <typename I>
  static __device__ __host__ inline int linkIndexP3(const int x[], const I X[4], const int mu) {
    return linkIndexDn<I,3>(x, X, mu);
  }

  /**
     Compute the 4-d spatial index from the checkerboarded 1-d index at parity parity

     @param x Computed spatial index
     @param cb_index 1-d checkerboarded index
     @param X Full lattice dimensions
     @param parity Site parity
   */
  template <typename I>
  static __device__ __host__ inline void getCoords(int x[], int cb_index, const I X[], int parity) {
    //x[3] = cb_index/(X[2]*X[1]*X[0]/2);
    //x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    //x[1] = (cb_index/(X[0]/2)) % X[1];
    //x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    int za = (cb_index / (X[0] >> 1));
    int zb =  (za / X[1]);
    x[1] = (za - zb * X[1]);
    x[3] = (zb / X[2]);
    x[2] = (zb - x[3] * X[2]);
    int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
    x[0] = (2 * cb_index + x1odd  - za * X[0]);
    return;
  }

  /**
     Compute the 4-d spatial index from the checkerboarded 1-d index at parity parity

     @param x Computed spatial index
     @param cb_index 1-d checkerboarded index
     @param X Full lattice dimensions
     @param parity Site parity
   */
  template <typename I, typename J>
  static __device__ __host__ inline void getCoordsExtended(I x[], int cb_index, const J X[], int parity, const int R[]) {
    //x[3] = cb_index/(X[2]*X[1]*X[0]/2);
    //x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    //x[1] = (cb_index/(X[0]/2)) % X[1];
    //x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    int za = (cb_index / (X[0] >> 1));
    int zb =  (za / X[1]);
    x[1] = (za - zb * X[1]);
    x[3] = (zb / X[2]);
    x[2] = (zb - x[3] * X[2]);
    int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
    x[0] = (2 * cb_index + x1odd  - za * X[0]);
#pragma unroll
    for (int d=0; d<4; d++) x[d] += R[d];
    return;
  }
  
  /**
     Compute the 4-d spatial index from the checkerboarded 1-d index at parity parity

     @param x Computed spatial index
     @param cb_index 1-d checkerboarded index
     @param X Full lattice dimensions
     @param parity Site parity
   */
  template <typename I>
  static __device__ __host__ inline void getCoords5(int x[5], int cb_index, const I X[5],
						    int parity, QudaDWFPCType pc_type) {
    //x[4] = cb_index/(X[3]*X[2]*X[1]*X[0]/2);
    //x[3] = (cb_index/(X[2]*X[1]*X[0]/2) % X[3];
    //x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    //x[1] = (cb_index/(X[0]/2)) % X[1];
    //x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    int za = (cb_index / (X[0] >> 1));
    int zb =  (za / X[1]);
    x[1] = za - zb * X[1];
    int zc = zb / X[2];
    x[2] = zb - zc*X[2];
    x[4] = (zc / X[3]);
    x[3] = zc - x[4] * X[3];
    int x1odd = (x[1] + x[2] + x[3] + (pc_type==QUDA_5D_PC ? x[4] : 0) + parity) & 1;
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
  template <typename I>
  static __device__ __host__ inline int getIndexFull(int cb_index, const I X[4], int parity) {
    int za = (cb_index / (X[0] / 2));
    int zb =  (za / X[1]);
    int x1 = za - zb * X[1];
    int x3 = (zb / X[2]);
    int x2 = zb - x3 * X[2];
    int x1odd = (x1 + x2 + x3 + parity) & 1;
    return 2 * cb_index + x1odd;  
  }
  
  /**
     Compute the checkerboarded index into the ghost field
     corresponding to full (local) site index x[]
     @param x local site
     @param X local lattice dimensions
     @param dim dimension
     @param depth of ghost
  */
  template <int dir, typename I>
  __device__ __host__ inline int ghostFaceIndex(const int x[], const I X[], int dim, int nFace) {
    int index = 0;
    switch(dim) {
    case 0:
      switch(dir) {
      case 0:
	index = (x[0]*X[4]*X[3]*X[2]*X[1] + x[4]*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1])+x[2]*X[1] + x[1])>>1;
	break;
      case 1:
	index = ((x[0]-X[0]+nFace)*X[4]*X[3]*X[2]*X[1] + x[4]*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1]) + x[2]*X[1] + x[1])>>1;
	break;
      }
      break;
    case 1:
      switch(dir) {
      case 0:
	index = (x[1]*X[4]*X[3]*X[2]*X[0] + x[4]*X[3]*X[2]*X[0] + x[3]*X[2]*X[0]+x[2]*X[0]+x[0])>>1;
	break;
      case 1:
	index = ((x[1]-X[1]+nFace)*X[4]*X[3]*X[2]*X[0] +x[4]*X[3]*X[2]*X[0]+ x[3]*X[2]*X[0] + x[2]*X[0] + x[0])>>1;
	break;
      }
      break;
    case 2:
      switch(dir) {
      case 0:
	index = (x[2]*X[4]*X[3]*X[1]*X[0] + x[4]*X[3]*X[1]*X[0] + x[3]*X[1]*X[0]+x[1]*X[0]+x[0])>>1;
	break;
      case 1:
	index = ((x[2]-X[2]+nFace)*X[4]*X[3]*X[1]*X[0] + x[4]*X[3]*X[1]*X[0] + x[3]*X[1]*X[0] + x[1]*X[0] + x[0])>>1;
	break;
      }
      break;
    case 3:
      switch(dir) {
      case 0:
	index = (x[3]*X[4]*X[2]*X[1]*X[0] + x[4]*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0]+x[0])>>1;
	break;
      case 1:
	index  = ((x[3]-X[3]+nFace)*X[4]*X[2]*X[1]*X[0] + x[4]*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0] + x[0])>>1;
	break;
      }
      break;
    }
    return index;
  }

} // namespace quda
