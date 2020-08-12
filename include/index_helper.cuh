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
  __device__ __host__ inline int linkIndexShift(const I x[], const J dx[], const K X[4]) {
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
  __device__ __host__ inline int linkIndexShift(I y[], const I x[], const J dx[], const K X[4]) {
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
  __device__ __host__ inline int linkIndex(const int x[], const I X[4]) {
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
  __device__ __host__ inline int linkIndex(int y[], const int x[], const I X[4]) {
    int idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
    y[0] = x[0]; y[1] = x[1]; y[2] = x[2]; y[3] = x[3];
    return idx;
  }

  /**
       Compute the checkerboard 1-d index from the 4-d coordinate x[] +n in the mu direction

       @return 1-d checkerboard index
       @tparam n number of hops (=/-) in the mu direction
       @param x 4-d lattice index
       @param X Full lattice dimensions
       @param mu direction in which to add n hops
     */
  template <typename I, int n, typename Coord>
  __device__ __host__ inline int linkIndexDn(const Coord &x, const I X[4], const int mu)
  {
    int y[4];
#pragma unroll
    for ( int i = 0; i < 4; i++ ) y[i] = x[i];
    y[mu] = (y[mu] + n + X[mu]) % X[mu];
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
  template <typename I, typename Coord> __device__ __host__ inline int linkIndexM1(const Coord &x, const I X[4], const int mu)
  {
    return linkIndexDn<I, -1>(x, X, mu);
  }

  /**
     Compute the checkerboard 1-d index from the 4-d coordinate x[] -3 in the mu direction

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param X Full lattice dimensions
     @param mu direction in which to subtract 3
   */
  template <typename I, typename Coord> __device__ __host__ inline int linkIndexM3(const Coord &x, const I X[4], const int mu)
  {
    return linkIndexDn<I, -3>(x, X, mu);
  }

  /**
     Compute the full 1-d index from the 4-d coordinate x[] +1 in the mu direction

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param X Full lattice dimensions
     @param mu direction in which to add 1
   */
  template <typename I>
  __device__ __host__ inline int linkNormalIndexP1(const int x[], const I X[4], const int mu) {
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
  template <typename I, typename Coord>
  __device__ __host__ inline int linkIndexP1(const Coord &x, const I X[4], const int mu) {
    return linkIndexDn<I, 1>(x, X, mu);
  }

  /**
     Compute the checkerboard 1-d index from the 4-d coordinate x[] +3 in the mu direction

     @return 1-d checkerboard index
     @param x 4-d lattice index
     @param X Full lattice dimensions
     @param mu direction in which to add 3
   */
  template <typename I, typename Coord> __device__ __host__ inline int linkIndexP3(const Coord &x, const I X[4], const int mu)
  {
    return linkIndexDn<I, 3>(x, X, mu);
  }

  template <int nDim>
  struct Coord {
    int x[nDim]; // nDim lattice coordinates
    int x_cb;    // checkerboard lattice site index
    int s;       // fifth dimension coord
    int X;       // full lattice site index
    constexpr const int& operator[](int i) const { return x[i]; }
    constexpr int& operator[](int i) { return x[i]; }
  };

  /**
     @brief Compute the checkerboard 1-d index for the nearest
     neighbor
     @param[in] lattice coordinates
     @param[in] mu dimension in which to add 1
     @param[in] dir direction (+1 or -1)
     @param[in] arg parameter struct
     @return 1-d checkboard index
   */
  template <typename Coord, typename Arg>
  __device__ __host__ inline int getNeighborIndexCB(const Coord &x, int mu, int dir, const Arg &arg)
  {
    switch (dir) {
    case +1: // positive direction
      switch (mu) {
      case 0: return (x[0] == arg.X[0] - 1 ? x.X - (arg.X[0] - 1) : x.X + 1) >> 1;
      case 1: return (x[1] == arg.X[1] - 1 ? x.X - arg.X2X1mX1 : x.X + arg.X[0]) >> 1;
      case 2: return (x[2] == arg.X[2] - 1 ? x.X - arg.X3X2X1mX2X1 : x.X + arg.X2X1) >> 1;
      case 3: return (x[3] == arg.X[3] - 1 ? x.X - arg.X4X3X2X1mX3X2X1 : x.X + arg.X3X2X1) >> 1;
      case 4: return (x[4] == arg.X[4] - 1 ? x.X - arg.X5X4X3X2X1mX4X3X2X1 : x.X + arg.X4X3X2X1) >> 1;
      }
    case -1:
      switch (mu) {
      case 0: return (x[0] == 0 ? x.X + (arg.X[0] - 1) : x.X - 1) >> 1;
      case 1: return (x[1] == 0 ? x.X + arg.X2X1mX1 : x.X - arg.X[0]) >> 1;
      case 2: return (x[2] == 0 ? x.X + arg.X3X2X1mX2X1 : x.X - arg.X2X1) >> 1;
      case 3: return (x[3] == 0 ? x.X + arg.X4X3X2X1mX3X2X1 : x.X - arg.X3X2X1) >> 1;
      case 4: return (x[4] == 0 ? x.X + arg.X5X4X3X2X1mX4X3X2X1 : x.X - arg.X4X3X2X1) >> 1;
      }
    }
    return 0; // should never reach here
  }

  /**
     Compute the 4-d spatial index from the checkerboarded 1-d index at parity parity

     @param[out] x Computed spatial index
     @param[in] cb_index 1-d checkerboarded index
     @param[in] X Full lattice dimensions
     @param[in] X0h Half of x-dim lattice dimension
     @param[in] parity Site parity
     @return Full linear lattice index
   */
  template <typename Coord, typename I, typename J>
  __device__ __host__ inline int getCoordsCB(Coord &x, int cb_index, const I X[], J X0h, int parity)
  {
    //x[3] = cb_index/(X[2]*X[1]*X[0]/2);
    //x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    //x[1] = (cb_index/(X[0]/2)) % X[1];
    //x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    int za = (cb_index / X0h);
    int zb =  (za / X[1]);
    x[1] = (za - zb * X[1]);
    x[3] = (zb / X[2]);
    x[2] = (zb - x[3] * X[2]);
    int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
    int x_idx = 2 * cb_index + x1odd;
    x[0] = (x_idx  - za * X[0]);
    return x_idx;
  }

  /**
     Compute the 4-d spatial index from the checkerboarded 1-d index
     at parity parity.  Wrapper around getCoordsCB.

     @param[out] x Computed spatial index
     @param[in] cb_index 1-d checkerboarded index
     @param[in] X Full lattice dimensions
     @param[in] X0h Half of x-dim lattice dimension
     @param[in] parity Site parity
     @return Full linear lattice index
   */
  template <typename Coord, typename I> __device__ __host__ inline int getCoords(Coord &x, int cb_index, const I X[], int parity)
  {
    return getCoordsCB(x, cb_index, X, X[0] >> 1, parity);
  }

  /**
     Compute the 4-d spatial index from the checkerboarded 1-d index at parity parity

     @param x Computed spatial index
     @param cb_index 1-d checkerboarded index
     @param X Full lattice dimensions
     @param parity Site parity
   */
  template <typename I, typename J>
  __device__ __host__ inline void getCoordsExtended(I x[], int cb_index, const J X[], int parity, const int R[]) {
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
     Compute the 5-d spatial index from the checkerboarded 1-d index at parity parity

     @param[out] x Computed spatial index
     @param[in] cb_index 1-d checkerboarded index
     @param[in] X Full lattice dimensions
     @param[in] X0h Half of x-dim lattice dimension
     @param[in] parity Site parity
     @return Full linear lattice index
   */
  template <typename Coord, typename I, typename J>
  __device__ __host__ inline int getCoords5CB(Coord &x, int cb_index, const I X[5], J X0h, int parity, QudaPCType pc_type)
  {
    //x[4] = cb_index/(X[3]*X[2]*X[1]*X[0]/2);
    //x[3] = (cb_index/(X[2]*X[1]*X[0]/2) % X[3];
    //x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    //x[1] = (cb_index/(X[0]/2)) % X[1];
    //x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    int za = (cb_index / X0h);
    int zb =  (za / X[1]);
    x[1] = za - zb * X[1];
    int zc = zb / X[2];
    x[2] = zb - zc*X[2];
    x[4] = (zc / X[3]);
    x[3] = zc - x[4] * X[3];
    int x1odd = (x[1] + x[2] + x[3] + (pc_type==QUDA_5D_PC ? x[4] : 0) + parity) & 1;
    int x_idx = 2 * cb_index + x1odd;
    x[0] = x_idx  - za * X[0];
    return x_idx;
  }

  /**
     Compute the 5-d spatial index from the checkerboarded 1-d index
     at parity parity.  Wrapper around getCoords5CB.

     @param[out] x Computed spatial index
     @param[in] cb_index 1-d checkerboarded index
     @param[in] X Full lattice dimensions
     @param[in] parity Site parity
     @return Full linear lattice index
   */
  template <typename I>
  __device__ __host__ inline int getCoords5(int x[5], int cb_index, const I X[5], int parity, QudaPCType pc_type)
  {
    return getCoords5CB(x, cb_index, X, X[0] >> 1, parity, pc_type);
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
  __device__ __host__ inline int getIndexFull(int cb_index, const I X[4], int parity) {
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
     @param x_ local site
     @param X_ local lattice dimensions
     @param dim dimension
     @param nFace depth of ghost
  */
  template <int dir, int nDim = 4, typename Coord, typename I>
  __device__ __host__ inline int ghostFaceIndex(const Coord &x_, const I X_[], int dim, int nFace)
  {
    static_assert((nDim == 4 || nDim == 5), "Number of dimensions must be 4 or 5");
    int index = 0;
    const int x[] = {x_[0], x_[1], x_[2], x_[3], nDim == 5 ? x_[4] : 0};
    const int X[] = {(int)X_[0], (int)X_[1], (int)X_[2], (int)X_[3], nDim == 5 ? (int)X_[4] : 1};

    switch(dim) {
    case 0:
      switch(dir) {
      case 0:
	index = (x[0]*X[4]*X[3]*X[2]*X[1] + x[4]*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1]) + x[2]*X[1] + x[1])>>1;
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
	index = ((x[3]-X[3]+nFace)*X[4]*X[2]*X[1]*X[0] + x[4]*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0] + x[0])>>1;
	break;
      }
      break;
    }
    return index;
  }

  /**
    Compute the checkerboarded index into the ghost field
    corresponding to full (local) site index x[] for staggered
    @param x_ local site
    @param X_ local lattice dimensions
    @param dim dimension
    @param nFace depth of ghost
 */
  template <int dir, int nDim = 4, typename Coord, typename I>
  __device__ __host__ inline int ghostFaceIndexStaggered(const Coord &x_, const I X_[], int dim, int nFace)
  {
    static_assert((nDim == 4 || nDim == 5), "Number of dimensions must be 4 or 5");
    int index = 0;
    const int x[] = {x_[0], x_[1], x_[2], x_[3], nDim == 5 ? x_[4] : 0};
    const int X[] = {(int)X_[0], (int)X_[1], (int)X_[2], (int)X_[3], nDim == 5 ? (int)X_[4] : 1};

    switch (dim) {
    case 0:
      switch (dir) {
      case 0:
        index = ((x[0] + nFace - 1) * X[4] * X[3] * X[2] * X[1] + x[4] * X[3] * X[2] * X[1] + x[3] * (X[2] * X[1])
                    + x[2] * X[1] + x[1])
            >> 1;
        break;
      case 1:
        index = ((x[0] - X[0] + nFace) * X[4] * X[3] * X[2] * X[1] + x[4] * X[3] * X[2] * X[1] + x[3] * (X[2] * X[1])
                    + x[2] * X[1] + x[1])
            >> 1;
        break;
      }
      break;
    case 1:
      switch (dir) {
      case 0:
        index = ((x[1] + nFace - 1) * X[4] * X[3] * X[2] * X[0] + x[4] * X[3] * X[2] * X[0] + x[3] * X[2] * X[0]
                    + x[2] * X[0] + x[0])
            >> 1;
        break;
      case 1:
        index = ((x[1] - X[1] + nFace) * X[4] * X[3] * X[2] * X[0] + x[4] * X[3] * X[2] * X[0] + x[3] * X[2] * X[0]
                    + x[2] * X[0] + x[0])
            >> 1;
        break;
      }
      break;
    case 2:
      switch (dir) {
      case 0:
        index = ((x[2] + nFace - 1) * X[4] * X[3] * X[1] * X[0] + x[4] * X[3] * X[1] * X[0] + x[3] * X[1] * X[0]
                    + x[1] * X[0] + x[0])
            >> 1;
        break;
      case 1:
        index = ((x[2] - X[2] + nFace) * X[4] * X[3] * X[1] * X[0] + x[4] * X[3] * X[1] * X[0] + x[3] * X[1] * X[0]
                    + x[1] * X[0] + x[0])
            >> 1;
        break;
      }
      break;
    case 3:
      switch (dir) {
      case 0:
        index = ((x[3] + nFace - 1) * X[4] * X[2] * X[1] * X[0] + x[4] * X[2] * X[1] * X[0] + x[2] * X[1] * X[0]
                    + x[1] * X[0] + x[0])
            >> 1;
        break;
      case 1:
        index = ((x[3] - X[3] + nFace) * X[4] * X[2] * X[1] * X[0] + x[4] * X[2] * X[1] * X[0] + x[2] * X[1] * X[0]
                    + x[1] * X[0] + x[0])
            >> 1;
        break;
      }
      break;
    }
    return index;
  }

  enum KernelType {
    INTERIOR_KERNEL = 5,
    EXTERIOR_KERNEL_ALL = 6,
    EXTERIOR_KERNEL_X = 0,
    EXTERIOR_KERNEL_Y = 1,
    EXTERIOR_KERNEL_Z = 2,
    EXTERIOR_KERNEL_T = 3,
    KERNEL_POLICY = 7
  };

  /**
     @brief Compute the full-lattice coordinates from the input face
     index.  This is used by the Wilson-like halo update kernels, and
     can deal with 4-d or 5-d field and 4-d or 5-d preconditioning.

     @param[out] idx The full lattice coordinate
     @param[out] cb_idx The checkboarded lattice coordinate
     @param[out] x Coordinates we are computing
     @param[in] face_idx Input checkerboarded face index
     @param[in] face_num Face number
     @param[in] parity Parity index
     @param[in] arg Argument struct with required meta data
  */
  template <int nDim, QudaPCType type, int dim_, int nLayers, typename Coord, typename Arg>
  inline __device__ __host__ void coordsFromFaceIndex(
      int &idx, int &cb_idx, Coord &x, int face_idx, const int &face_num, int parity, const Arg &arg)
  {
    constexpr int dim = (dim_ == INTERIOR_KERNEL || dim_ == EXTERIOR_KERNEL_ALL) ? 0 : dim_; // silence compiler warning

    const auto *X = arg.dc.X;
    const auto &face_X = arg.dc.face_X[dim];
    const auto &face_Y = arg.dc.face_Y[dim];
    const auto &face_Z = arg.dc.face_Z[dim];
    const auto &face_T = arg.dc.face_T[dim];
    const auto &face_XYZT = arg.dc.face_XYZT[dim];
    const auto &face_XYZ = arg.dc.face_XYZ[dim];
    const auto &face_XY = arg.dc.face_XY[dim];

    // intrinsic parity of the face depends on offset of first element
    int face_parity = (parity + face_num * (arg.dc.X[dim] - nLayers)) & 1;

    // compute coordinates from (checkerboard) face index
    face_idx *= 2;

    if (!(face_X & 1)) { // face_X even
      //   s = face_idx / face_XYZT;
      //   t = (face_idx / face_XYZ) % face_T;
      //   z = (face_idx / face_XY) % face_Z;
      //   y = (face_idx / face_X) % face_Y;
      //   face_idx += (face_parity + s + t + z + y) & 1;
      //   x = face_idx % face_X;
      // equivalent to the above, but with fewer divisions/mods:
      int aux1 = face_idx / face_X;
      int aux2 = aux1 / face_Y;
      int aux3 = aux2 / face_Z;
      int aux4 = (nDim == 5 ? aux3 / face_T : 0);

      x[0] = face_idx - aux1 * face_X;
      x[1] = aux1 - aux2 * face_Y;
      x[2] = aux2 - aux3 * face_Z;
      x[3] = aux3 - aux4 * face_T;
      x[4] = aux4;

      if (type == QUDA_5D_PC)
        x[0] += (face_parity + x[4] + x[3] + x[2] + x[1]) & 1;
      else
        x[0] += (face_parity + x[3] + x[2] + x[1]) & 1;

    } else if (!(face_Y & 1)) { // face_Y even

      x[4] = (nDim == 5 ? face_idx / face_XYZT : 0);
      x[3] = (nDim == 5 ? (face_idx / face_XYZ) % face_T : (face_idx / face_XYZ));
      x[2] = (face_idx / face_XY) % face_Z;

      if (type == QUDA_5D_PC)
        face_idx += (face_parity + x[4] + x[3] + x[2]) & 1;
      else
        face_idx += (face_parity + x[3] + x[2]) & 1;

      x[1] = (face_idx / face_X) % face_Y;
      x[0] = face_idx % face_X;

    } else if (!(face_Z & 1)) { // face_Z even

      x[4] = (nDim == 5 ? face_idx / face_XYZT : 0);
      x[3] = (nDim == 5 ? (face_idx / face_XYZ) % face_T : (face_idx / face_XYZ));

      if (type == QUDA_5D_PC)
        face_idx += (face_parity + x[4] + x[3]) & 1;
      else if (type == QUDA_4D_PC)
        face_idx += (face_parity + x[3]) & 1;

      x[2] = (face_idx / face_XY) % face_Z;
      x[1] = (face_idx / face_X) % face_Y;
      x[0] = face_idx % face_X;

    } else {

      x[4] = (nDim == 5 ? face_idx / face_XYZT : 0);
      face_idx += face_parity;
      x[3] = (nDim == 5 ? (face_idx / face_XYZ) % face_T : (face_idx / face_XYZ));
      x[2] = (face_idx / face_XY) % face_Z;
      x[1] = (face_idx / face_X) % face_Y;
      x[0] = face_idx % face_X;
    }

    // need to convert to global coords, not face coords
    x[dim] += face_num * (X[dim] - nLayers);

    // compute index into the full local volume
    idx = X[0] * (X[1] * (X[2] * (X[3] * x[4] + x[3]) + x[2]) + x[1]) + x[0];

    // compute index into the checkerboard
    cb_idx = idx >> 1;
  }

  /**
     @brief Overloaded variant of indexFromFaceIndex where we use the
     parity declared in arg.
   */
  template <int nDim, QudaPCType type, int dim_, int nLayers, typename Coord, typename Arg>
  inline __device__ __host__ void coordsFromFaceIndex(
      int &idx, int &cb_idx, Coord &x, int face_idx, const int &face_num, const Arg &arg)
  {
    coordsFromFaceIndex<nDim, type, dim_, nLayers>(idx, cb_idx, x, face_idx, face_num, arg.parity, arg);
  }

  /**
     @brief Compute the checkerboard lattice index from the input face
     index.  This is used by the Wilson-like halo packing kernels, and
     can deal with 4-d or 5-d field and 4-d or 5-d preconditioning.
     @param[in] face_idx Checkerboard halo index
     @param[in] parity Parity index
     @param[in] arg Argument struct with required meta data
     @return Checkerboard lattice index
  */
  template <int nDim, QudaPCType type, int dim, int nLayers, int face_num, typename Arg>
  inline __device__ __host__ int indexFromFaceIndex(int face_idx, int parity, const Arg &arg)
  {
    // intrinsic parity of the face depends on offset of first element
    int face_parity = (parity + face_num * (arg.dc.X[dim] - nLayers)) & 1;

    // reconstruct full face index from index into the checkerboard
    face_idx *= 2;

    if (!(arg.dc.face_X[dim] & 1)) { // face_X even
      //   int s = face_idx / face_XYZT;
      //   int t = (face_idx / face_XYZ) % face_T;
      //   int z = (face_idx / face_XY) % face_Z;
      //   int y = (face_idx / face_X) % face_Y;
      //   face_idx += (face_parity + s + t + z + y) & 1;
      // equivalent to the above, but with fewer divisions/mods:
      int aux1 = face_idx / arg.dc.face_X[dim];
      int aux2 = aux1 / arg.dc.face_Y[dim];
      int aux3 = aux2 / arg.dc.face_Z[dim];
      int aux4 = (nDim == 5 ? aux3 / arg.dc.face_T[dim] : 0);

      int y = aux1 - aux2 * arg.dc.face_Y[dim];
      int z = aux2 - aux3 * arg.dc.face_Z[dim];
      int t = aux3 - aux4 * arg.dc.face_T[dim];
      int s = aux4;

      if (type == QUDA_5D_PC)
        face_idx += (face_parity + s + t + z + y) & 1;
      else if (type == QUDA_4D_PC)
        face_idx += (face_parity + t + z + y) & 1;

    } else if (!(arg.dc.face_Y[dim] & 1)) { // face_Y even

      int s = face_idx / arg.dc.face_XYZT[dim];
      int t = (nDim == 5 ? (face_idx / arg.dc.face_XYZ[dim]) % arg.dc.face_T[dim] : (face_idx / arg.dc.face_XYZ[dim]));
      int z = (face_idx / arg.dc.face_XY[dim]) % arg.dc.face_Z[dim];
      if (type == QUDA_5D_PC)
        face_idx += (face_parity + s + t + z) & 1;
      else if (type == QUDA_4D_PC)
        face_idx += (face_parity + t + z) & 1;

    } else if (!(arg.dc.face_Z[dim] & 1)) { // face_Z even

      int s = face_idx / arg.dc.face_XYZT[dim];
      int t = (nDim == 5 ? (face_idx / arg.dc.face_XYZ[dim]) % arg.dc.face_T[dim] : (face_idx / arg.dc.face_XYZ[dim]));
      if (type == QUDA_5D_PC)
        face_idx += (face_parity + s + t) & 1;
      else if (type == QUDA_4D_PC)
        face_idx += (face_parity + t) & 1;

    } else if (!(arg.dc.face_T[dim]) && nDim == 5) {

      int s = face_idx / arg.dc.face_XYZT[dim];
      if (type == QUDA_5D_PC)
        face_idx += (face_parity + s) & 1;
      else if (type == QUDA_4D_PC)
        face_idx += face_parity;

    } else {
      face_idx += face_parity;
    }

    // compute index into the full local volume
    int gap = arg.dc.X[dim] - nLayers;
    int idx = face_idx;
    int aux;
    switch (dim) {
    case 0:
      aux = face_idx / arg.dc.face_X[dim];
      idx += (aux + face_num) * gap;
      break;
    case 1:
      aux = face_idx / arg.dc.face_XY[dim];
      idx += (aux + face_num) * gap * arg.dc.face_X[dim];
      break;
    case 2:
      aux = face_idx / arg.dc.face_XYZ[dim];
      idx += (aux + face_num) * gap * arg.dc.face_XY[dim];
      break;
    case 3:
      aux = (nDim == 5 ? face_idx / arg.dc.face_XYZT[dim] : 0);
      idx += (aux + face_num) * gap * arg.dc.face_XYZ[dim];
      break;
    }

    // return index into the checkerboard
    return idx >> 1;
  }

  /**
     @brief Overloaded variant of indexFromFaceIndex where we use the
     parity declared in arg.
   */
  template <int nDim, QudaPCType type, int dim, int nLayers, int face_num, typename Arg>
  inline __device__ __host__ int indexFromFaceIndex(int face_idx, const Arg &arg)
  {
    return indexFromFaceIndex<nDim, type, dim, nLayers, face_num>(face_idx, arg.parity, arg);
  }

  /**
    @brief Compute global checkerboard index from face index.
    The following indexing routines work for arbitrary lattice
    dimensions (though perhaps not odd like thw Wilson variant?)
    Specifically, we compute an index into the local volume from an
    index into the face.  This is used by the staggered-like face
    packing routines, and is different from the Wilson variant since
    here the halo depth is tranversed in a different order - here the
    halo depth is the faster running dimension.

    @param[in] face_idx_in Checkerboarded face index
    @param[in] param Parameter struct with required meta data
    @return Global checkerboard coordinate
  */

  // int idx = indexFromFaceIndex<4,QUDA_4D_PC,dim,nFace,0>(ghost_idx, parity, arg);

  template <int nDim, QudaPCType type, int dim, int nLayers, int face_num, typename Arg>
  inline __device__ int indexFromFaceIndexStaggered(int face_idx_in, int parity, const Arg &arg)
  {
    const auto *X = arg.dc.X;            // grid dimension
    const auto *dims = arg.dc.dims[dim]; // dimensions of the face
    const auto &V4 = arg.dc.volume_4d;   // 4-d volume

    // intrinsic parity of the face depends on offset of first element
    int face_parity = (parity + face_num * (X[dim] - nLayers)) & 1;

    // reconstruct full face index from index into the checkerboard
    face_idx_in *= 2;

    // first compute src index, then find 4-d index from remainder
    int s = face_idx_in / arg.dc.face_XYZT[dim];
    int face_idx = face_idx_in - s * arg.dc.face_XYZT[dim];

    /*y,z,t here are face indexes in new order*/
    int aux1 = face_idx / dims[0];
    int aux2 = aux1 / dims[1];
    int y = aux1 - aux2 * dims[1];
    int t = aux2 / dims[2];
    int z = aux2 - t * dims[2];
    face_idx += (face_parity + t + z + y) & 1;

    // compute index into the full local volume
    int gap = X[dim] - nLayers;
    int idx = face_idx;
    int aux;
    switch (dim) {
    case 0:
      aux = face_idx;
      idx += face_num * gap + aux * (X[0] - 1);
      idx += (idx / V4) * (1 - V4);
      break;
    case 1:
      aux = face_idx / arg.dc.face_X[dim];
      idx += face_num * gap * arg.dc.face_X[dim] + aux * (X[1] - 1) * arg.dc.face_X[dim];
      idx += (idx / V4) * (X[0] - V4);
      break;
    case 2:
      aux = face_idx / arg.dc.face_XY[dim];
      idx += face_num * gap * arg.dc.face_XY[dim] + aux * (X[2] - 1) * arg.dc.face_XY[dim];
      idx += (idx / V4) * ((X[1] * X[0]) - V4);
      break;
    case 3: idx += face_num * gap * arg.dc.face_XYZ[dim]; break;
    }

    // return index into the checkerboard
    return (idx + s * V4) >> 1;
  }

  /**
     @brief Determines which face a given thread is computing.  Also
     rescale face_idx so that is relative to a given dimension.  If 5-d
     variant if called, then it is assumed that arg.threads contains
     only the 3-d surface of threads but face_idx is a 4-d index
     (surface * fifth dimension).  At present multi-src staggered uses
     the 4-d variant since the face_idx that is passed in is the 3-d
     surface not the 4-d one.

     @param[out] face_idx Face index
     @param[in] tid Checkerboard volume index
     @param[in] arg Input parameters
     @return dimension this face_idx corresponds to
  */
  template <int nDim = 4, typename Arg>
  __host__ __device__ inline int dimFromFaceIndex(int &face_idx, int tid, const Arg &arg)
  {

    // s - the coordinate in the fifth dimension - is the slowest-changing coordinate
    const int s = (nDim == 5 ? tid / arg.threads : 0);

    face_idx = tid - s * arg.threads; // face_idx = face_idx % arg.threads

    if (face_idx < arg.threadDimMapUpper[0]) {
      face_idx += s * arg.threadDimMapUpper[0];
      return 0;
    } else if (face_idx < arg.threadDimMapUpper[1]) {
      face_idx -= arg.threadDimMapLower[1];
      face_idx += s * (arg.threadDimMapUpper[1] - arg.threadDimMapLower[1]);
      return 1;
    } else if (face_idx < arg.threadDimMapUpper[2]) {
      face_idx -= arg.threadDimMapLower[2];
      face_idx += s * (arg.threadDimMapUpper[2] - arg.threadDimMapLower[2]);
      return 2;
    } else {
      face_idx -= arg.threadDimMapLower[3];
      face_idx += s * (arg.threadDimMapUpper[3] - arg.threadDimMapLower[3]);
      return 3;
    }
  }

  template <int nDim = 4, typename Arg> __host__ __device__ inline int dimFromFaceIndex(int &face_idx, const Arg &arg)
  {
    return dimFromFaceIndex<nDim>(face_idx, face_idx, arg);
  }

  /**
     @brief Swizzler for reordering the (x) thread block indices - use on
     conjunction with swizzle-factor autotuning to find the optimum
     swizzle factor.  Specifically, the thread block id is remapped by
     transposing its coordinates: if the original order can be
     parametrized by

     blockIdx.x = j * swizzle + i,

     then the new order is

     block_idx = i * (gridDim.x / swizzle) + j

     We need to factor out any remainder and leave this in original
     ordering.

     @param[in] swizzle Swizzle factor to be applied
     @return Swizzled block index
  */
  //#define SWIZZLE
  template <typename T> __device__ inline int block_idx(const T &swizzle)
  {
#ifdef SWIZZLE
    // the portion of the grid that is exactly divisible by the number of SMs
    const int gridp = gridDim.x - gridDim.x % swizzle;

    int block_idx = blockIdx.x;
    if (blockIdx.x < gridp) {
      // this is the portion of the block that we are going to transpose
      const int i = blockIdx.x % swizzle;
      const int j = blockIdx.x / swizzle;

      // transpose the coordinates
      block_idx = i * (gridp / swizzle) + j;
    }
    return block_idx;
#else
    return blockIdx.x;
#endif
  }

  /**
     @brief Compute the staggered phase factor at unit shift from the
     current lattice coordinates.  The routine below optimizes out the
     shift where possible, hence is only visible where we need to
     consider the boundary condition.

     @param[in] coords Lattice coordinates
     @param[in] X Lattice dimensions
     @param[in] dim Dimension we are hopping
     @param[in] dir Direction of the unit hop (+1 or -1)
     @param[in] tboundary Boundary condition
   */
  template <typename Coord, typename Arg>
  __device__ __host__ inline auto StaggeredPhase(const Coord &coords, int dim, int dir, const Arg &arg) -> typename Arg::real
  {
    using real = typename Arg::real;
    constexpr auto phase = Arg::phase;
    static_assert(
        phase == QUDA_STAGGERED_PHASE_MILC || phase == QUDA_STAGGERED_PHASE_TIFR, "Unsupported staggered phase");
    real sign;

    const auto *X = arg.dim;
    if (phase == QUDA_STAGGERED_PHASE_MILC) {
      switch (dim) {
      case 0: sign = (coords[3]) % 2 == 0 ? static_cast<real>(1.0) : static_cast<real>(-1.0); break;
      case 1: sign = (coords[3] + coords[0]) % 2 == 0 ? static_cast<real>(1.0) : static_cast<real>(-1.0); break;
      case 2:
        sign = (coords[3] + coords[1] + coords[0]) % 2 == 0 ? static_cast<real>(1.0) : static_cast<real>(-1.0);
        break;
      case 3: sign = (coords[3] + dir >= X[3] && arg.is_last_time_slice) || (coords[3] + dir < 0 && arg.is_first_time_slice) ?
          arg.tboundary : static_cast<real>(1.0); break;
      default: sign = static_cast<real>(1.0);
      }
    } else if (phase == QUDA_STAGGERED_PHASE_TIFR) {
      switch (dim) {
      case 0: sign = (coords[3] + coords[2] + coords[1]) % 2 == 0 ? -1 : 1; break;
      case 1: sign = ((coords[3] + coords[2]) % 2 == 1) ? -1 : 1; break;
      case 2: sign = (coords[3] % 2 == 0) ? -1 : 1; break;
      case 3: sign = (coords[3] + dir >= X[3] && arg.is_last_time_slice) || (coords[3] + dir < 0 && arg.is_first_time_slice) ?
          arg.tboundary : static_cast<real>(1.0); break;
      default: sign = static_cast<real>(1.0);
      }
    }
    return sign;
  }

  /*
     Indexing functions used by the outer product kernels.  Should be
     reconciled with the above at some point.  These have added
     functionality that may be useful for dealing with odd-sized local
     dimensions.
   */
  enum IndexType {
    EVEN_X = 0,
    EVEN_Y = 1,
    EVEN_Z = 2,
    EVEN_T = 3
  };

  template <IndexType idxType>
  __device__ __host__ inline void coordsFromIndex(int &idx, int c[4], unsigned int cb_idx, int parity, const int X[4])
  {
    const int &LX = X[0];
    const int &LY = X[1];
    const int &LZ = X[2];
    const int XYZ = X[2]*X[1]*X[0];
    const int XY = X[1]*X[0];

    idx = 2*cb_idx;

    int x, y, z, t;

    if (idxType == EVEN_X /*!(LX & 1)*/) { // X even
      //   t = idx / XYZ;
      //   z = (idx / XY) % Z;
      //   y = (idx / X) % Y;
      //   idx += (parity + t + z + y) & 1;
      //   x = idx % X;
      // equivalent to the above, but with fewer divisions/mods:
      int aux1 = idx / LX;
      x = idx - aux1 * LX;
      int aux2 = aux1 / LY;
      y = aux1 - aux2 * LY;
      t = aux2 / LZ;
      z = aux2 - t * LZ;
      aux1 = (parity + t + z + y) & 1;
      x += aux1;
      idx += aux1;
    } else if (idxType == EVEN_Y /*!(LY & 1)*/) { // Y even
      t = idx / XYZ;
      z = (idx / XY) % LZ;
      idx += (parity + t + z) & 1;
      y = (idx / LX) % LY;
      x = idx % LX;
    } else if (idxType == EVEN_Z /*!(LZ & 1)*/) { // Z even
      t = idx / XYZ;
      idx += (parity + t) & 1;
      z = (idx / XY) % LZ;
      y = (idx / LX) % LY;
      x = idx % LX;
    } else {
      idx += parity;
      t = idx / XYZ;
      z = (idx / XY) % LZ;
      y = (idx / LX) % LY;
      x = idx % LX;
    }

    c[0] = x;
    c[1] = y;
    c[2] = z;
    c[3] = t;
  }

  // Get the  coordinates for the exterior kernels
  __device__ __host__ inline void coordsFromIndexExterior(int x[4], const unsigned int cb_idx, const int X[4],
                                                          const unsigned int dir, const int displacement, const unsigned int parity)
  {
    int Xh[2] = {X[0] / 2, X[1] / 2};
    switch (dir) {
    case 0:
      x[2] = cb_idx / Xh[1] % X[2];
      x[3] = cb_idx / (Xh[1] * X[2]) % X[3];
      x[0] = cb_idx / (Xh[1] * X[2] * X[3]);
      x[0] += (X[0] - displacement);
      x[1] = 2 * (cb_idx % Xh[1]) + ((x[0] + x[2] + x[3] + parity) & 1);
      break;

    case 1:
      x[2] = cb_idx / Xh[0] % X[2];
      x[3] = cb_idx / (Xh[0] * X[2]) % X[3];
      x[1] = cb_idx / (Xh[0] * X[2] * X[3]);
      x[1] += (X[1] - displacement);
      x[0] = 2 * (cb_idx % Xh[0]) + ((x[1] + x[2] + x[3] + parity) & 1);
      break;

    case 2:
      x[1] = cb_idx / Xh[0] % X[1];
      x[3] = cb_idx / (Xh[0] * X[1]) % X[3];
      x[2] = cb_idx / (Xh[0] * X[1] * X[3]);
      x[2] += (X[2] - displacement);
      x[0] = 2 * (cb_idx % Xh[0]) + ((x[1] + x[2] + x[3] + parity) & 1);
      break;

    case 3:
      x[1] = cb_idx / Xh[0] % X[1];
      x[2] = cb_idx / (Xh[0] * X[1]) % X[2];
      x[3] = cb_idx / (Xh[0] * X[1] * X[2]);
      x[3] += (X[3] - displacement);
      x[0] = 2 * (cb_idx % Xh[0]) + ((x[1] + x[2] + x[3] + parity) & 1);
      break;
    }
    return;
  }

  __device__ __host__ inline int neighborIndex(unsigned int cb_idx, const int shift[4], const bool partitioned[4], int parity,
                                               const int X[4])
  {
    int full_idx;
    int x[4];

    coordsFromIndex<EVEN_X>(full_idx, x, cb_idx, parity, X);

    for(int dim = 0; dim<4; ++dim){
      if( partitioned[dim] )
	if( (x[dim]+shift[dim])<0 || (x[dim]+shift[dim])>=X[dim]) return -1;
    }

    for(int dim=0; dim<4; ++dim){
      x[dim] = shift[dim] ? (x[dim]+shift[dim] + X[dim]) % X[dim] : x[dim];
    }
    return (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
  }

} // namespace quda
