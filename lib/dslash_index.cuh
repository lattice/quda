/**
  @brief Compute global extended checkerboard index from face index.  The
  following indexing routines work for arbitrary (including odd)
  lattice dimensions.  Specifically, we compute an index into the
  local volume from an index into the face.  This is used by the
  Wilson-like face packing routines.

  @param[in] face_idx Checkerboarded face index
  @param[in] param Parameter struct with required meta data
  @return Global extended checkerboard coordinate
*/
template <int dim, int nLayers, int face_num, typename Param>
static inline __device__ int indexFromFaceIndexExtended(int face_idx, const Param &param)
{
  const auto *X = param.dc.X;
  const auto *R = param.R;

  int face_X = X[0], face_Y = X[1], face_Z = X[2]; // face_T = X[3]
  switch (dim) {
    case 0:
      face_X = nLayers;
      break;
    case 1:
      face_Y = nLayers;
      break;
    case 2:
      face_Z = nLayers;
      break;
    case 3:
      // face_T = nLayers;
      break;
  }
  int face_XYZ = face_X * face_Y * face_Z;
  int face_XY = face_X * face_Y;

  // intrinsic parity of the face depends on offset of first element

  int face_parity = (param.parity + face_num *(X[dim] - nLayers)) & 1;
  // reconstruct full face index from index into the checkerboard

  face_idx *= 2;

  if (!(face_X & 1)) { // face_X even
    //   int t = face_idx / face_XYZ;
    //   int z = (face_idx / face_XY) % face_Z;
    //   int y = (face_idx / face_X) % face_Y;
    //   face_idx += (face_parity + t + z + y) & 1;
    // equivalent to the above, but with fewer divisions/mods:
    int aux1 = face_idx / face_X;
    int aux2 = aux1 / face_Y;
    int y = aux1 - aux2 * face_Y;
    int t = aux2 / face_Z;
    int z = aux2 - t * face_Z;
    face_idx += (face_parity + t + z + y) & 1;
  } else if (!(face_Y & 1)) { // face_Y even
    int t = face_idx / face_XYZ;
    int z = (face_idx / face_XY) % face_Z;
    face_idx += (face_parity + t + z) & 1;
  } else if (!(face_Z & 1)) { // face_Z even
    int t = face_idx / face_XYZ;
    face_idx += (face_parity + t) & 1;
  } else {
    face_idx += face_parity;
  }

  // compute index into the full local volume

  int idx = face_idx;
  int aux;

  int gap = X[dim] - nLayers;
  switch (dim) {
    case 0:
      aux = face_idx / face_X;
      idx += (aux + face_num)*gap + (1 - 2*face_num)*R[0];
      break;
    case 1:
      aux = face_idx / face_XY;
      idx += ((aux + face_num)*gap + (1 - 2*face_num)*R[1])*face_X;
      break;
    case 2:
      aux = face_idx / face_XYZ;
      idx += ((aux + face_num)*gap + (1 - 2*face_num)*R[2])* face_XY;
      break;
    case 3:
      idx += (face_num*gap + (1 - 2*face_num)*R[3])*face_XYZ;
      break;
  }

  // return index into the checkerboard

  return idx >> 1;
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
template <int dim, int nLayers, int face_num, typename Param>
static inline __device__ int indexFromFaceIndexStaggered(int face_idx_in, const Param &param)
{
  const auto *X = param.dc.X;              // grid dimension
  const auto *dims = param.dc.dims[dim];   // dimensions of the face
  const auto &V4 = param.dc.volume_4d;     // 4-d volume

  // intrinsic parity of the face depends on offset of first element
  int face_parity = (param.parity + face_num *(X[dim] - nLayers)) & 1;

  // reconstruct full face index from index into the checkerboard
  face_idx_in *= 2;

  // first compute src index, then find 4-d index from remainder
  int s = face_idx_in / param.dc.face_XYZT[dim];
  int face_idx = face_idx_in - s*param.dc.face_XYZT[dim];

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
      idx += face_num*gap + aux*(X[0]-1);
      idx += (idx/V4)*(1-V4);
      break;
    case 1:
      aux = face_idx / param.dc.face_X[dim];
      idx += face_num * gap * param.dc.face_X[dim] + aux*(X[1]-1)*param.dc.face_X[dim];
      idx += (idx/V4)*(X[0]-V4);
      break;
    case 2:
      aux = face_idx / param.dc.face_XY[dim];
      idx += face_num * gap * param.dc.face_XY[dim] +aux*(X[2]-1)*param.dc.face_XY[dim];
      idx += (idx/V4)*((X[1]*X[0])-V4);
      break;
    case 3:
      idx += face_num * gap * param.dc.face_XYZ[dim];
      break;
  }

  // return index into the checkerboard
  return (idx + s*V4) >> 1;
}


/**
  @brief Compute global extended checkerboard index from face index.
  The following indexing routines work for arbitrary lattice
  dimensions (though perhaps not odd like thw Wilson variant?)
  Specifically, we compute an index into the local volume from an
  index into the face.  This is used by the staggered-like face
  packing routines, and is different from the Wilson variant since
  here the halo depth is tranversed in a different order - here the
  halo depth is the faster running dimension.

  @param[in] face_idx_in Checkerboarded face index
  @param[in] param Parameter struct with required meta data
  @return Global extended checkerboard coordinate
*/
template <int dim, int nLayers, int face_num, typename Param>
static inline __device__ int indexFromFaceIndexExtendedStaggered(int face_idx, const Param &param)
{
  const auto *X = param.dc.X;
  const auto *R = param.R;

  // dimensions of the face
  int dims[3];
  int V = X[0]*X[1]*X[2]*X[3];
  int face_X = X[0], face_Y = X[1], face_Z = X[2]; // face_T = X[3];
  switch (dim) {
    case 0:
      face_X = nLayers;
      dims[0]=X[1]; dims[1]=X[2]; dims[2]=X[3];
      break;
    case 1:
      face_Y = nLayers;
      dims[0]=X[0];dims[1]=X[2]; dims[2]=X[3];
      break;
    case 2:
      face_Z = nLayers;
      dims[0]=X[0]; dims[1]=X[1]; dims[2]=X[3];
      break;
    case 3:
      // face_T = nLayers;
      dims[0]=X[0]; dims[1]=X[1]; dims[2]=X[3];
      break;
  }
  int face_XYZ = face_X * face_Y * face_Z;
  int face_XY = face_X * face_Y;

  // intrinsic parity of the face depends on offset of first element
  int face_parity = (param.parity + face_num *(X[dim] - nLayers)) & 1;

  // reconstruct full face index from index into the checkerboard
  face_idx *= 2;
  /*y,z,t here are face indexes in new order*/
  int aux1 = face_idx / dims[0];
  int aux2 = aux1 / dims[1];
  int y = aux1 - aux2 * dims[1];
  int t = aux2 / dims[2];
  int z = aux2 - t * dims[2];
  face_idx += (face_parity + t + z + y) & 1;

  int idx = face_idx;
  int aux;

  int gap = X[dim] - nLayers - 2*R[dim];
  switch (dim) {
    case 0:
      aux = face_idx;
      idx += face_num*gap + aux*(X[0]-1);
      idx += (idx/V)*(1-V);    
      idx += R[0];
      break;
    case 1:
      aux = face_idx / face_X;
      idx += face_num * gap * face_X + aux*(X[1]-1)*face_X;
      idx += idx/V*(X[0]-V);
      idx += R[1]*X[0];
      break;
    case 2:
      aux = face_idx / face_XY;    
      idx += face_num * gap * face_XY +aux*(X[2]-1)*face_XY;
      idx += idx/V*(face_XY-V);
      idx += R[2]*face_XY;
      break;
    case 3:
      idx += ((face_num*gap) + R[3])*face_XYZ;
      break;
  }

  // return index into the checkerboard

  return idx >> 1;
}


/**
   @brief Compute the full-lattice coordinates from the input face
   index.  This is used by the staggered halo update kernels.

   @param x[out] Coordinates we are computing
   @param idx[in] Input checkerboard face index
   @param[in] param Parameter struct with required meta data
 */
template<KernelType dim, int nLayers, int Dir, typename Param>
static inline __device__ void coordsFromFaceIndexStaggered(int x[], int idx, const Param &param)
{
  const auto *X = param.dc.X;
  const auto *Xh = param.dc.Xh;

  int za, x1h, x0h, zb;
  switch(dim) {
    case EXTERIOR_KERNEL_X:
      za = idx/Xh[1];
      x1h = idx - za*Xh[1];
      zb = za / X[2];
      x[2] = za - zb*X[2];
      x[0] = zb/X[3];
      x[3] = zb - x[0]*X[3];
      if(Dir == 2){
        x[0] += ((x[0] >= nLayers) ? (X[0] - 2*nLayers) : 0);
      }else if(Dir == 1){
       x[0] += (X[0] - nLayers);
      }
      x[1] = 2*x1h + ((x[0] + x[2] + x[3] + param.parity) & 1);
      break;
    case EXTERIOR_KERNEL_Y:
      za = idx/Xh[0];
      x0h = idx - za*Xh[0];
      zb = za / X[2];
      x[2] = za - zb*X[2];
      x[1] = zb/X[3];
      x[3] = zb - x[1]*X[3];
      if(Dir == 2){
        x[1] += ((x[1] >= nLayers) ? (X[1] - 2*nLayers) : 0);
      }else if(Dir == 1){
        x[1] += (X[1] - nLayers);
      }
      x[0] = 2*x0h + ((x[1] + x[2] + x[3] + param.parity) & 1);
      break;
    case EXTERIOR_KERNEL_Z:
      za = idx/Xh[0];
      x0h = idx - za*Xh[0];
      zb = za / X[1];
      x[1] = za - zb*X[1];
      x[2] = zb / X[3];
      x[3] = zb - x[2]*X[3];
      if(Dir == 2){
        x[2] += ((x[2] >= nLayers) ? (X[2] - 2*nLayers) : 0);
      }else if(Dir == 1){
        x[2] += (X[2] - nLayers);
      }
      x[0] = 2*x0h + ((x[1] + x[2] + x[3] + param.parity) & 1);
      break;
    case EXTERIOR_KERNEL_T:
      za = idx/Xh[0];
      x0h = idx - za*Xh[0];
      zb = za / X[1];
      x[1] = za - zb*X[1];
      x[3] = zb / X[2];
      x[2] = zb - x[3]*X[2];
      if(Dir == 2){
        x[3] += ((x[3] >= nLayers) ? (X[3] - 2*nLayers) : 0);
      }else if(Dir == 1){
        x[3] += (X[3] - nLayers);
      }
      x[0] = 2*x0h + ((x[1] + x[2] + x[3] + param.parity) & 1);
      break; 
  }
  return;
}

enum IndexType {
  EVEN_X = 0,
  EVEN_Y = 1,
  EVEN_Z = 2,
  EVEN_T = 3
};

/**
   @brief Compute coordinates from index into the checkerboard (used
   by the interior Dslash kernels).  This is used by the Wilson-like
   interior update kernels, and can deal with 4-d or 5-d field and 4-d or
   5-d preconditioning.

   @param idx[out] The full lattice coordinate
   @param cb_idx[out] The checkboarded lattice coordinate
   @param x[out] Coordinates we are computing
   @param idx[in] Input checkerboarded face index
   @param[in] param Parameter struct with required meta data
 */
template <int nDim, QudaPCType pc_type, IndexType idxType, typename T, typename Param>
static __device__ __forceinline__ void coordsFromIndex(int &idx, T *x, int &cb_idx, const Param &param)
{

  // The full field index is 
  // idx = x + y*X + z*X*Y + t*X*Y*Z
  // The parity of lattice site (x,y,z,t) 
  // is defined to be (x+y+z+t) & 1
  // 0 => even parity 
  // 1 => odd parity
  // cb_idx runs over the half volume
  // cb_idx = iidx/2 = (x + y*X + z*X*Y + t*X*Y*Z)/2
  //
  // We need to obtain idx from cb_idx + parity.
  // 
  // 1)  First, consider the case where X is even.
  // Then, y*X + z*X*Y + t*X*Y*Z is even and
  // 2*cb_idx = 2*(x/2) + y*X + z*X*Y + t*X*Y*Z
  // Since, 2*(x/2) is even, if y+z+t is even
  // (2*(x/2),y,z,t) is an even parity site.
  // Similarly, if y+z+t is odd
  // (2*(x/2),y,z,t) is an odd parity site. 
  // 
  // Note that (1+y+z+t)&1 = 1 for y+z+t even
  //      and  (1+y+z+t)&1 = 0 for y+z+t odd
  // Therefore, 
  // (2*/(x/2) + (1+y+z+t)&1, y, z, t) is odd.
  //
  // 2)  Consider the case where X is odd but Y is even.
  // Calculate 2*cb_idx
  // t = 2*cb_idx/XYZ
  // z = (2*cb_idx/XY) % Z
  //
  // Now, we  need to compute (x,y) for different parities.
  // To select a site with even parity, consider (z+t).
  // If (z+t) is even, this implies that (x+y) must also 
  // be even in order that (x+y+z+t) is even. 
  // Therefore,  x + y*X is even.
  // Thus, 2*cb_idx = idx 
  // and y =  (2*cb_idx/X) % Y
  // and x =  (2*cb_idx) % X;
  // 
  // On the other hand, if (z+t) is odd, (x+y) must be 
  // also be odd in order to get overall even parity. 
  // Then x + y*X is odd (since X is odd and either x or y is odd)
  // and 2*cb_idx = 2*(idx/2) = idx-1 =  x + y*X -1 + z*X*Y + t*X*Y*Z
  // => idx = 2*cb_idx + 1
  // and y = ((2*cb_idx + 1)/X) % Y
  // and x = (2*cb_idx + 1) % X
  //
  // To select a site with odd parity if (z+t) is even,
  // (x+y) must be odd, which, following the discussion above, implies that
  // y = ((2*cb_idx + 1)/X) % Y
  // x = (2*cb_idx + 1) % X
  // Finally, if (z+t) is odd (x+y) must be even to get overall odd parity, 
  // and 
  // y = ((2*cb_idx)/X) % Y
  // x = (2*cb_idx) % X
  // 
  // The code below covers these cases 
  // as well as the cases where X, Y are odd and Z is even,
  // and X,Y,Z are all odd

  const auto *X = param.dc.X;

  int XYZT = param.dc.Vh << 1; // X[3]*X[2]*X[1]*X[0]
  int XYZ = param.dc.X3X2X1; // X[2]*X[1]*X[0]
  int XY = param.dc.X2X1; // X[1]*X[0]

  idx = 2*cb_idx;
  if (idxType == EVEN_X /*!(X[0] & 1)*/) { // X even
    //   t = idx / XYZ;
    //   z = (idx / XY) % Z;
    //   y = (idx / X) % Y;
    //   idx += (parity + t + z + y) & 1;
    //   x = idx % X;
    // equivalent to the above, but with fewer divisions/mods:
#if DSLASH_TUNE_TILE // tiled indexing - experimental and disabled for now
    const auto *block = param.block;
    const auto *grid = param.grid;

    int aux[9];
    aux[0] = idx;
    for (int i=0; i<4; i++) aux[i+1] = aux[i] / block[i];
    for (int i=4; i<8; i++) aux[i+1] = aux[i] / grid[i];

    for (int i=0; i<4; i++) x[i] = aux[i] - aux[i+1] * block[i];
    for (int i=0; i<4; i++) x[i] += block[i]*(aux[i+4] - aux[i+5] * grid[i]);
    x[4] = (nDim == 5) ? aux[8] : 0;

    int oddbit = (pc_type == QUDA_4D_PC ? (param.parity + t + z + y) : (param.parity + s + t + z + y)) & 1;
    x += oddbit;

    // update cb_idx for the swizzled coordinate
    cb_idx = (nDim == 5 ? (((x[4]*X[3]+x[3])*X[2]+x[2])*X[1]+x[1])*X[0]+x[0] : ((x[3]*X[2]+x[2])*X[1]+x[1])*X[0]+x[0]) >> 1;
    idx = 2*cb_idx + oddbit;
#else
    int aux[5];
    aux[0] = idx;
    for (int i=0; i<4; i++) aux[i+1] = aux[i] / X[i];

    x[0] = aux[0] - aux[1] * X[0];
    x[1] = aux[1] - aux[2] * X[1];
    x[2] = aux[2] - aux[3] * X[2];
    x[3] = aux[3] - (nDim == 5 ? aux[4] * X[3] : 0);
    x[4] = (nDim == 5) ? aux[4] : 0;

    int oddbit = (pc_type == QUDA_4D_PC ? (param.parity + x[3] + x[2] + x[1]) : (param.parity + x[4] + x[3] + x[2] + x[1])) & 1;

    x[0] += oddbit;
    idx += oddbit;
#endif
  } else if (idxType == EVEN_Y /*!(X[1] & 1)*/) { // Y even
    x[4] = idx / XYZT;
    x[3] = (idx / XYZ) % X[3];
    x[2] = (idx / XY) % X[2];
    idx += (param.parity + x[3] + x[2]) & 1;
    x[1] = (idx / X[0]) % X[1];
    x[0] = idx % X[0];
  } else if (idxType == EVEN_Z /*!(X[2] & 1)*/) { // Z even
    x[4] = idx / XYZT;
    x[3] = (idx / XYZ) % X[3];
    idx += (param.parity + x[3]) & 1;
    x[2] = (idx / XY) % X[2];
    x[1] = (idx / X[0]) % X[1];
    x[0] = idx % X[0];
  } else {
    x[4] = idx / XYZT;
    idx += (param.parity + x[4]) & 1;
    x[3] = idx / XYZ;
    x[2] = (idx / XY) % X[2];
    x[1] = (idx / X[0]) % X[1];
    x[0] = idx % X[0];
  } // else we do not support all odd local dimensions except fifth dimension
}


/**
  @brief Compute coordinates from index into the checkerboard (used
  by the interior Dslash kernels).  This is the variant used by the
  shared memory wilson dslash.

  @param[out] idx Linear index
  @param[out] x Compute coordinates
  @param[out] ch_idx Linear checkboard index
  @param[in] param Parameter struct with required meta data
*/
template <IndexType idxType, typename Int, typename Param>
static __device__ __forceinline__ void coordsFromIndex3D(int &idx, Int * const x, int &cb_idx, const Param &param) {
  const auto *X = param.x;

  if (idxType == EVEN_X) { // X even
    int xt = blockIdx.x*blockDim.x + threadIdx.x;
    int aux = xt+xt;
    x[3] = aux / X[0];
    x[0] = aux - x[3]*X[0];
    x[1] = blockIdx.y*blockDim.y + threadIdx.y;
    x[2] = blockIdx.z*blockDim.z + threadIdx.z;
    x[0] += (param.parity + x[3] + x[2] + x[1]) &1;
    idx = ((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0];
    cb_idx = idx >> 1; 
  } else {
    // Non-even X is not (yet) supported.
    return;
  }
}


/**
  @brief Compute whether the provided coordinate is within the halo
  region boundary of a given dimension.

  @param[in] depth Depth of halo
  @param[in] coord Coordinates
  @param[in] X Lattice dimensions
  @return True if in boundary, else false
*/
template <int dim, typename T>
static inline __device__ bool inBoundary(const int depth, const int coord[], const T X[]){
  return ((coord[dim] >= X[dim] - depth) || (coord[dim] < depth));
}


/**
  @brief Compute whether this thread should be active for updating the
  a given offsetDim halo.  This is used by the fused halo region
  update kernels: here every thread has a prescribed dimension it is
  tasked with updating, but for the edges and vertices, the thread
  responsible for the entire update is the "greatest" one.  Hence some
  threads may be labelled as a given dimension, but they have to
  update other dimensions too.  Conversely, a given thread may be
  labeled for a given dimension, but if that thread lies at en edge or
  vertex, and we have partitioned a higher dimension, then that thread
  will cede to the higher thread.

  @param[in] threadDim Prescribed dimension of this thread
  @param[in] offsetDim The dimension we are querying whether this thread should be responsible
  @param[in] offset The size of the hop
  @param[in] y Site coordinate
  @param[in] partitioned Array of which dimensions have been partitioned
  @param[in] X Lattice dimensions
  @return True if this thread is active
*/
template <typename T>
static inline __device__ bool isActive(const int threadDim, int offsetDim, int offset, const int y[],  const int partitioned[], const T X[])
{

  // Threads with threadDim = t can handle t,z,y,x offsets
  // Threads with threadDim = z can handle z,y,x offsets
  // Threads with threadDim = y can handle y,x offsets
  // Threads with threadDim = x can handle x offsets 
  if(!partitioned[offsetDim]) return false;
  
  if(threadDim < offsetDim) return false;
  int width = (offset > 0) ? offset : -offset;
 
  switch(threadDim){
    case 3: // threadDim = T
      break;

    case 2: // threadDim = Z
      if(!partitioned[3]) break;
      if(partitioned[3] && inBoundary<3>(width, y, X)) return false;
      break;

    case 1: // threadDim = Y
      if((!partitioned[3]) && (!partitioned[2])) break;
      if(partitioned[3] && inBoundary<3>(width, y, X)) return false;
      if(partitioned[2] && inBoundary<2>(width, y, X)) return false;
      break;

    case 0: // threadDim = X
      if((!partitioned[3]) && (!partitioned[2]) && (!partitioned[1])) break;
      if(partitioned[3] && inBoundary<3>(width, y, X)) return false;
      if(partitioned[2] && inBoundary<2>(width, y, X)) return false;
      if(partitioned[1] && inBoundary<1>(width, y, X)) return false;
      break;

    default:
      break;
  }
  return true;
}


/**
  @brief Compute the face index from the lattice coordinates.

  @param[in] face_idx Face index
  @param[in] x Lattice coordinates
  @param[in] face_dim Which dimension
  @param[in] param Input parameters
  @return dimension this face_idx corresponds to
*/
template<int nDim, int nLayers, typename I, typename Param>
static inline __device__ void faceIndexFromCoords(int &face_idx, I * const x, int face_dim, const Param &param)
{
  int D[4] = {param.dc.X[0], param.dc.X[1], param.dc.X[2], param.dc.X[3]};
  int y[5] = {x[0], x[1], x[2], x[3], (nDim==5 ? x[4] : 0)};

  y[face_dim] = (y[face_dim] < nLayers) ? y[face_dim] : y[face_dim] - (D[face_dim] - nLayers);
  D[face_dim] = nLayers;

  if (nDim == 5)      face_idx = (((((D[3]*y[4] + y[3])*D[2] + y[2])*D[1] + y[1])*D[0] + y[0]) >> 1);
  else if (nDim == 4) face_idx =  ((((D[2]*y[3] + y[2])*D[1] + y[1])*D[0] + y[0]) >> 1);

  return;
}

/*
  @brief Fast power function that works for negative "a" argument
  @param a argument we want to raise to some power
  @param b power that we want to raise a to
  @return pow(a,b)
*/
__device__ inline float __fast_pow(float a, int b) {
  float sign = signbit(a) ? -1.0f : 1.0f;
  float power = __powf(fabsf(a), b);
  return b&1 ? sign * power : power;
}


 
