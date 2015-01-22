// The following indexing routines work for arbitrary (including odd) lattice dimensions.
// compute an index into the local volume from an index into the face (used by the face packing routines)

  template <int dim, int nLayers, int face_num>
static inline __device__ int indexFromFaceIndex(int face_idx, const int &face_volume, const int &parity, const int X[])
{
  // dimensions of the face (FIXME: optimize using constant cache)

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

  int face_parity = (parity + face_num *(X[dim] - nLayers)) & 1;
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
      idx += (aux + face_num) * gap;
      break;
    case 1:
      aux = face_idx / face_XY;
      idx += (aux + face_num) * gap * face_X;
      break;
    case 2:
      aux = face_idx / face_XYZ;
      idx += (aux + face_num) * gap * face_XY;
      break;
    case 3:
      idx += face_num * gap * face_XYZ;
      break;
  }

  // return index into the checkerboard

  return idx >> 1;
}


  template <int dim, int nLayers, int face_num>
static inline __device__ int indexFromFaceIndexExtended(int face_idx, const int &face_volume, const int &parity, const int X[], const int R[])
{

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

  int face_parity = (parity + face_num *(X[dim] - nLayers)) & 1;
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

// compute an index into the local volume from an index into the face (used by the face packing routines)
// G.Shi: the spinor order in ghost region is different between wilson and staggered, thus different index
//	  computing routine.
  template <int dim, int nLayers, int face_num>
static inline __device__ int indexFromFaceIndexStaggered(int face_idx, const int &face_volume,
    const int &parity, const int X[])
{
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
  int face_parity = (parity + face_num *(X[dim] - nLayers)) & 1;

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

  int gap = X[dim] - nLayers;
  switch (dim) {
    case 0:
      aux = face_idx;
      idx += face_num*gap + aux*(X[0]-1);
      idx += (idx/V)*(1-V);    
      break;
    case 1:
      aux = face_idx / face_X;
      idx += face_num * gap * face_X + aux*(X[1]-1)*face_X;
      idx += idx/V*(X[0]-V);
      break;
    case 2:
      aux = face_idx / face_XY;    
      idx += face_num * gap * face_XY +aux*(X[2]-1)*face_XY;
      idx += idx/V*((X[1]*X[0])-V);
      break;
    case 3:
      idx += face_num * gap * face_XYZ;
      break;
  }

  // return index into the checkerboard

  return idx >> 1;
}

  template <int dim, int nLayers, int face_num>
static inline __device__ int indexFromFaceIndexExtendedStaggered(int face_idx, const int &face_volume,
    const int &parity, const int X[], const int R[])
{
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
  int face_parity = (parity + face_num *(X[dim] - nLayers)) & 1;

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


template<int nLayers, int Dir> 
static inline __device__ void coordsFromFaceIndexStaggered(int x[], int idx, const int parity, const enum KernelType dim, const int X[])
{
  int za, x1h, x0h, zb;
  switch(dim) {
    case EXTERIOR_KERNEL_X:
      za = idx/(X[1]>>1); 
      x1h = idx - za*(X[1]>>1);
      zb = za / X[2];
      x[2] = za - zb*X[2];
      x[0] = zb/X[3];
      x[3] = zb - x[0]*X[3];
      if(Dir == 2){
        x[0] += ((x[0] >= nLayers) ? (X[0] - 2*nLayers) : 0);
      }else if(Dir == 1){
       x[0] += (X[0] - nLayers);
      }
      x[1] = 2*x1h + ((x[0] + x[2] + x[3] + parity) & 1);
      break;
    case EXTERIOR_KERNEL_Y:
      za = idx/(X[0]>>1);
      x0h = idx - za*(X[0]>>1);
      zb = za / X[2];
      x[2] = za - zb*X[2];
      x[1] = zb/X[3];
      x[3] = zb - x[1]*X[3];
      if(Dir == 2){
        x[1] += ((x[1] >= nLayers) ? (X[1] - 2*nLayers) : 0);
      }else if(Dir == 1){
        x[1] += (X[1] - nLayers);
      }
      x[0] = 2*x0h + ((x[1] + x[2] + x[3] + parity) & 1); 
      break;
    case EXTERIOR_KERNEL_Z:
      za = idx/(X[0]>>1);
      x0h = idx - za*(X[0]>>1);
      zb = za / X[1];
      x[1] = za - zb*X[1];
      x[2] = zb / X[3];
      x[3] = zb - x[2]*X[3];
      if(Dir == 2){
        x[2] += ((x[2] >= nLayers) ? (X[2] - 2*nLayers) : 0);
      }else if(Dir == 1){
        x[2] += (X[2] - nLayers);
      }
      x[0] = 2*x0h + ((x[1] + x[2] + x[3] + parity) & 1);
      break;
    case EXTERIOR_KERNEL_T:
      za = idx/(X[0]>>1);
      x0h = idx - za*(X[0]>>1);
      zb = za / X[1];
      x[1] = za - zb*X[1];
      x[3] = zb / X[2];
      x[2] = zb - x[3]*X[2];
      if(Dir == 2){
        x[3] += ((x[3] >= nLayers) ? (X[3] - 2*nLayers) : 0);
      }else if(Dir == 1){
        x[3] += (X[3] - nLayers);
      }
      x[0] = 2*x0h + ((x[1] + x[2] + x[3] + parity) & 1);
      break; 
  }
  return;
}


// compute full coordinates from an index into the face (used by the exterior Dslash kernels)
  template <int nLayers, typename Int>
static inline __device__ void coordsFromFaceIndex(int &idx, int &cb_idx, Int &x, Int &y, Int &z, Int &t, int face_idx,
						  const int &face_volume, const int &dim, const int &face_num, const int &parity, const int X[])
{
  // dimensions of the face (FIXME: optimize using constant cache)

  int face_X = X[0], face_Y = X[1], face_Z = X[2];
  int face_parity;
  switch (dim) {
    case 0:
      face_X = nLayers;
      face_parity = (parity + face_num * (X[0] - nLayers)) & 1;
      break;
    case 1:
      face_Y = nLayers;
      face_parity = (parity + face_num * (X[1] - nLayers)) & 1;
      break;
    case 2:
      face_Z = nLayers;
      face_parity = (parity + face_num * (X[2] - nLayers)) & 1;
      break;
    case 3:
      face_parity = (parity + face_num * (X[3] - nLayers)) & 1;
      break;
  }
  int face_XYZ = face_X * face_Y * face_Z;
  int face_XY = face_X * face_Y;

  // compute coordinates from (checkerboard) face index

  face_idx *= 2;

  if (!(face_X & 1)) { // face_X even
    //   t = face_idx / face_XYZ;
    //   z = (face_idx / face_XY) % face_Z;
    //   y = (face_idx / face_X) % face_Y;
    //   face_idx += (face_parity + t + z + y) & 1;
    //   x = face_idx % face_X;
    // equivalent to the above, but with fewer divisions/mods:
    int aux1 = face_idx / face_X;
    x = face_idx - aux1 * face_X;
    int aux2 = aux1 / face_Y;
    y = aux1 - aux2 * face_Y;
    t = aux2 / face_Z;
    z = aux2 - t * face_Z;
    x += (face_parity + t + z + y) & 1;
    // face_idx += (face_parity + t + z + y) & 1;
  } else if (!(face_Y & 1)) { // face_Y even
    t = face_idx / face_XYZ;
    z = (face_idx / face_XY) % face_Z;
    face_idx += (face_parity + t + z) & 1;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  } else if (!(face_Z & 1)) { // face_Z even
    t = face_idx / face_XYZ;
    face_idx += (face_parity + t) & 1;
    z = (face_idx / face_XY) % face_Z;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  } else {
    face_idx += face_parity;
    t = face_idx / face_XYZ; 
    z = (face_idx / face_XY) % face_Z;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  }

  //printf("Local sid %d (%d, %d, %d, %d)\n", cb_int, x, y, z, t);

  // need to convert to global coords, not face coords
  switch(dim) {
    case 0:
      x += face_num * (X[0]-nLayers);
      break;
    case 1:
      y += face_num * (X[1]-nLayers);
      break;
    case 2:
      z += face_num * (X[2]-nLayers);
      break;
    case 3:
      t += face_num * (X[3]-nLayers);
      break;
  }

  // compute index into the full local volume

  idx = X[0]*(X[1]*(X[2]*t + z) + y) + x; 

  // compute index into the checkerboard

  cb_idx = idx >> 1;

  //printf("Global sid %d (%d, %d, %d, %d)\n", cb_int, x, y, z, t);
}

enum IndexType {
  EVEN_X = 0,
  EVEN_Y = 1,
  EVEN_Z = 2,
  EVEN_T = 3
};

// compute coordinates from index into the checkerboard (used by the interior Dslash kernels)
  template <IndexType idxType, typename Int>
static __device__ __forceinline__ void coordsFromIndex(int &idx, Int &x, Int &y, Int &z, Int &t, 
						       const int &cb_idx, const int &parity, const int X[])
{
  int LX = X[0];
  int LY = X[1];
  int LZ = X[2];
  int &XYZ = X3X2X1; // X[2]*X[1]*X[0]
  int &XY = X2X1; // X[1]*X[0]

  idx = 2*cb_idx;

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
}





// compute coordinates from index into the checkerboard (used by the interior Dslash kernels)
// This is the variant used byt the shared memory wilson dslash
  template <IndexType idxType, typename Int>
static __device__ __forceinline__ void coordsFromIndex3D(int &idx, Int &x, Int &y, Int &z, Int &t, 
							 int &cb_idx, const int &parity, const int X[])
{
  int LX = X[0];
  int LY = X[1];
  int LZ = X[2];

  if (idxType == EVEN_X) { // X even
    int xt = blockIdx.x*blockDim.x + threadIdx.x;
    int aux = xt+xt;
    t = aux / LX;
    x = aux - t*LX;
    y = blockIdx.y*blockDim.y + threadIdx.y;
    z = blockIdx.z*blockDim.z + threadIdx.z;
    x += (parity + t + z + y) &1;
    idx = ((t*LZ + z)*LY + y)*LX + x;
    cb_idx = idx >> 1; 
  } else {
    // Non-even X is not (yet) supported.
    return;
  }
}

//Used in DW kernels only:

  template <int dim, int nLayers, int face_num>
static inline __device__ int indexFromDWFaceIndex(int face_idx, const int &face_volume,
						  const int &parity, const int X[])
{
  // dimensions of the face (FIXME: optimize using constant cache)

  //A.S.: Also used for computing offsets in physical lattice
  //A.S.: note that in the case of DW fermions one is dealing with 4d faces

  // intrinsic parity of the face depends on offset of first element, used for MPI DW as well
  int face_X = X[0], face_Y = X[1], face_Z = X[2], face_T = X[3];
  int face_parity;  

  switch (dim) {
    case 0:
      face_X = nLayers;
      face_parity = (parity + face_num * (X[0] - nLayers)) & 1;
      break;
    case 1:
      face_Y = nLayers;
      face_parity = (parity + face_num * (X[1] - nLayers)) & 1;
      break;
    case 2:
      face_Z = nLayers;
      face_parity = (parity + face_num * (X[2] - nLayers)) & 1;
      break;
    case 3:
      face_T = nLayers;    
      face_parity = (parity + face_num * (X[3] - nLayers)) & 1;
      break;
  }

  int face_XYZT = face_X * face_Y * face_Z * face_T;  
  int face_XYZ = face_X * face_Y * face_Z;
  int face_XY = face_X * face_Y;

  // reconstruct full face index from index into the checkerboard

  face_idx *= 2;

  if (!(face_X & 1)) { // face_X even
    //   int s = face_idx / face_XYZT;    
    //   int t = (face_idx / face_XYZ) % face_T;
    //   int z = (face_idx / face_XY) % face_Z;
    //   int y = (face_idx / face_X) % face_Y;
    //   face_idx += (face_parity + s + t + z + y) & 1;
    // equivalent to the above, but with fewer divisions/mods:
    int aux1 = face_idx / face_X;
    int aux2 = aux1 / face_Y;
    int aux3 = aux2 / face_Z;
    int y = aux1 - aux2 * face_Y;
    int z = aux2 - aux3 * face_Z;    
    int s = aux3 / face_T;
    int t = aux3 - s * face_T;
    face_idx += (face_parity + s + t + z + y) & 1;
  } else if (!(face_Y & 1)) { // face_Y even
    int s = face_idx / face_XYZT;    
    int t = (face_idx / face_XYZ) % face_T;
    int z = (face_idx / face_XY) % face_Z;
    face_idx += (face_parity + s + t + z) & 1;
  } else if (!(face_Z & 1)) { // face_Z even
    int s = face_idx / face_XYZT;        
    int t = (face_idx / face_XYZ) % face_T;
    face_idx += (face_parity + s + t) & 1;
  } else if(!(face_T)){
    int s = face_idx / face_XYZT;        
    face_idx += (face_parity + s) & 1;
  }else{    
    face_idx += face_parity;
  }

  // compute index into the full local volume

  int idx = face_idx;
  int gap, aux;

  switch (dim) {
    case 0:
      gap = X[0] - nLayers;
      aux = face_idx / face_X;
      idx += (aux + face_num) * gap;
      break;
    case 1:
      gap = X[1] - nLayers;
      aux = face_idx / face_XY;
      idx += (aux + face_num) * gap * face_X;
      break;
    case 2:
      gap = X[2] - nLayers;
      aux = face_idx / face_XYZ;
      idx += (aux + face_num) * gap * face_XY;
      break;
    case 3:
      gap = X[3] - nLayers;
      aux = face_idx / face_XYZT;
      idx += (aux + face_num) * gap * face_XYZ;
      break;
  }

  // return index into the checkerboard

  return idx >> 1;
}


// compute full coordinates from an index into the face (used by the exterior Dslash kernels)
  template <int nLayers, typename Int>
static inline __device__ void coordsFromDWFaceIndex(int &cb_idx, Int &x, Int &y, Int &z, Int &t, Int &s, int face_idx,
						    const int &face_volume, const int &dim, const int &face_num, const int &parity, const int X[])
{
  // dimensions of the face (FIXME: optimize using constant cache)

  int face_X = X[0], face_Y = X[1], face_Z = X[2], face_T = X[3];
  int face_parity;
  switch (dim) {
    case 0:
      face_X = nLayers;
      face_parity = (parity + face_num * (X[0] - nLayers)) & 1;
      break;
    case 1:
      face_Y = nLayers;
      face_parity = (parity + face_num * (X[1] - nLayers)) & 1;
      break;
    case 2:
      face_Z = nLayers;
      face_parity = (parity + face_num * (X[2] - nLayers)) & 1;
      break;
    case 3:
      face_T = nLayers;    
      face_parity = (parity + face_num * (X[3] - nLayers)) & 1;
      break;
  }
  int face_XYZT = face_X * face_Y * face_Z * face_T;  
  int face_XYZ  = face_X * face_Y * face_Z;
  int face_XY   = face_X * face_Y;

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
    x = face_idx - aux1 * face_X;
    int aux2 = aux1 / face_Y;
    y = aux1 - aux2 * face_Y;
    int aux3 = aux2 / face_Z;
    z = aux2 - aux3 * face_Z;
    s = aux3 / face_T;
    t = aux3 - s * face_T;
    x += (face_parity + s + t + z + y) & 1;
    // face_idx += (face_parity + t + z + y) & 1;
  } else if (!(face_Y & 1)) { // face_Y even
    s = face_idx / face_XYZT;    
    t = (face_idx / face_XYZ) % face_T;
    z = (face_idx / face_XY) % face_Z;
    face_idx += (face_parity + s + t + z) & 1;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  } else if (!(face_Z & 1)) { // face_Z even
    s = face_idx / face_XYZT;    
    t = (face_idx / face_XYZ) % face_T;
    face_idx += (face_parity + s + t) & 1;
    z = (face_idx / face_XY) % face_Z;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  } else {
    s = face_idx / face_XYZT;        
    face_idx += face_parity;
    t = (face_idx / face_XYZ) % face_T;
    z = (face_idx / face_XY) % face_Z;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  }

  //printf("Local sid %d (%d, %d, %d, %d)\n", cb_int, x, y, z, t);

  // need to convert to global coords, not face coords
  switch(dim) {
    case 0:
      x += face_num * (X[0]-nLayers);
      break;
    case 1:
      y += face_num * (X[1]-nLayers);
      break;
    case 2:
      z += face_num * (X[2]-nLayers);
      break;
    case 3:
      t += face_num * (X[3]-nLayers);
      break;
  }

  // compute index into the checkerboard

  cb_idx = (X[0]*(X[1]*(X[2]*(X[3]*s + t) + z) + y) + x) >> 1;

  //printf("Global sid %d (%d, %d, %d, %d)\n", cb_int, x, y, z, t);
}

template <int dim, int nLayers, int face_num>
static inline __device__ int indexFromDW4DFaceIndex(int face_idx, const int &face_volume,
						    const int &parity, const int X[])
{
  // dimensions of the face (FIXME: optimize using constant cache)
  //H.J. Kim.: note that in the case of DW fermions one is dealing with 4d faces
  
  // intrinsic parity of the face depends on offset of first element, used for MPI DW as well
  int face_X = X[0], face_Y = X[1], face_Z = X[2], face_T = X[3];
  int face_parity;  
  
  switch (dim) {
  case 0:
    face_X = nLayers;
    face_parity = (parity + face_num * (X[0] - nLayers)) & 1;
    break;
  case 1:
    face_Y = nLayers;
    face_parity = (parity + face_num * (X[1] - nLayers)) & 1;
    break;
  case 2:
    face_Z = nLayers;
    face_parity = (parity + face_num * (X[2] - nLayers)) & 1;
    break;
  case 3:
    face_T = nLayers;    
    face_parity = (parity + face_num * (X[3] - nLayers)) & 1;
    break;
  }
  
  int face_XYZT = face_X * face_Y * face_Z * face_T;  
  int face_XYZ = face_X * face_Y * face_Z;
  int face_XY = face_X * face_Y;
  // reconstruct full face index from index into the checkerboard

  face_idx *= 2;

  if (!(face_X & 1)) { // face_X even
    //   int s = face_idx / face_XYZT;    
    //   int t = (face_idx / face_XYZ) % face_T;
    //   int z = (face_idx / face_XY) % face_Z;
    //   int y = (face_idx / face_X) % face_Y;
    //   face_idx += (face_parity + s + t + z + y) & 1;
    // equivalent to the above, but with fewer divisions/mods:
    int aux1 = face_idx / face_X;
    int aux2 = aux1 / face_Y;
    int aux3 = aux2 / face_Z;
    int y = aux1 - aux2 * face_Y;
    int z = aux2 - aux3 * face_Z;    
    int s = aux3 / face_T;
    int t = aux3 - s * face_T;
    face_idx += (face_parity + t + z + y) & 1;
  } else if (!(face_Y & 1)) { // face_Y even
  //  int s = face_idx / face_XYZT;    
    int t = (face_idx / face_XYZ) % face_T;
    int z = (face_idx / face_XY) % face_Z;
    face_idx += (face_parity + t + z) & 1;
  } else if (!(face_Z & 1)) { // face_Z even
  //  int s = face_idx / face_XYZT;        
    int t = (face_idx / face_XYZ) % face_T;
    face_idx += (face_parity + t) & 1;
  } else if(!(face_T)){
  //  int s = face_idx / face_XYZT;        
    face_idx += face_parity;
  }else{    
    face_idx += face_parity;
  }

  // compute index into the full local volume

  int idx = face_idx;
  int gap, aux;

  switch (dim) {
  case 0:
    gap = X[0] - nLayers;
    aux = face_idx / face_X;
    idx += (aux + face_num) * gap;
    break;
  case 1:
    gap = X[1] - nLayers;
    aux = face_idx / face_XY;
    idx += (aux + face_num) * gap * face_X;
    break;
  case 2:
    gap = X[2] - nLayers;
    aux = face_idx / face_XYZ;
    idx += (aux + face_num) * gap * face_XY;
    break;
  case 3:
    gap = X[3] - nLayers;
    aux = face_idx / face_XYZT;
    idx += (aux + face_num) * gap * face_XYZ;
    break;
  }

  // return index into the checkerboard

  return idx >> 1;
}

// compute full coordinates from an index into the face (used by the exterior Dslash kernels)
template <int nLayers, typename Int>
static inline __device__ void coordsFromDW4DFaceIndex(int &cb_idx, Int &x, Int &y, Int &z, Int &t, Int &s, int face_idx,
						      const int &face_volume, const int &dim, const int &face_num, const int &parity, const int X[])
{
  // dimensions of the face (FIXME: optimize using constant cache)

  int face_X = X[0], face_Y = X[1], face_Z = X[2], face_T = X[3];
  int face_parity;
  switch (dim) {
  case 0:
    face_X = nLayers;
    face_parity = (parity + face_num * (X[0] - nLayers)) & 1;
    break;
  case 1:
    face_Y = nLayers;
    face_parity = (parity + face_num * (X[1] - nLayers)) & 1;
    break;
  case 2:
    face_Z = nLayers;
    face_parity = (parity + face_num * (X[2] - nLayers)) & 1;
    break;
  case 3:
    face_T = nLayers;    
    face_parity = (parity + face_num * (X[3] - nLayers)) & 1;
    break;
  }
  int face_XYZT = face_X * face_Y * face_Z * face_T;  
  int face_XYZ  = face_X * face_Y * face_Z;
  int face_XY   = face_X * face_Y;

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
    x = face_idx - aux1 * face_X;
    int aux2 = aux1 / face_Y;
    y = aux1 - aux2 * face_Y;
    int aux3 = aux2 / face_Z;
    z = aux2 - aux3 * face_Z;
    s = aux3 / face_T;
    t = aux3 - s * face_T;
    x += (face_parity + t + z + y) & 1;
  } else if (!(face_Y & 1)) { // face_Y even
    s = face_idx / face_XYZT;    
    t = (face_idx / face_XYZ) % face_T;
    z = (face_idx / face_XY) % face_Z;
    face_idx += (face_parity + t + z) & 1;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  } else if (!(face_Z & 1)) { // face_Z even
    s = face_idx / face_XYZT;    
    t = (face_idx / face_XYZ) % face_T;
    face_idx += (face_parity + t) & 1;
    z = (face_idx / face_XY) % face_Z;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  } else {
    s = face_idx / face_XYZT;        
    face_idx += face_parity;
    t = (face_idx / face_XYZ) % face_T;
    z = (face_idx / face_XY) % face_Z;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  }

  //printf("Local sid %d (%d, %d, %d, %d, %d)\n", cb_idx, x, y, z, t, s);

  // need to convert to global coords, not face coords
  switch(dim) {
  case 0:
    x += face_num * (X[0]-nLayers);
    break;
  case 1:
    y += face_num * (X[1]-nLayers);
    break;
  case 2:
    z += face_num * (X[2]-nLayers);
    break;
  case 3:
    t += face_num * (X[3]-nLayers);
    break;
  }

  // compute index into the checkerboard

  cb_idx = (X[0]*(X[1]*(X[2]*(X[3]*s + t) + z) + y) + x) >> 1;

  //printf("Global sid %d (%d, %d, %d, %d, %d)\n", cb_idx, x, y, z, t, s);
}

//!ndeg tm:
  template <int dim, int nLayers, int face_num>
static inline __device__ int indexFromNdegTMFaceIndex(int face_idx, const int &face_volume,
						      const int &parity, const int X[])
{
  // dimensions of the face (FIXME: optimize using constant cache)

  int face_X = X[0], face_Y = X[1], face_Z = X[2], face_T = X[3];
  int face_parity;  

  switch (dim) {
    case 0:
      face_X = nLayers;
      face_parity = (parity + face_num * (X[0] - nLayers)) & 1;
      break;
    case 1:
      face_Y = nLayers;
      face_parity = (parity + face_num * (X[1] - nLayers)) & 1;
      break;
    case 2:
      face_Z = nLayers;
      face_parity = (parity + face_num * (X[2] - nLayers)) & 1;
      break;
    case 3:
      face_T = nLayers;    
      face_parity = (parity + face_num * (X[3] - nLayers)) & 1;
      break;
  }

  int face_XYZT = face_X * face_Y * face_Z * face_T;  
  int face_XYZ  = face_X * face_Y * face_Z;
  int face_XY   = face_X * face_Y;

  // reconstruct full face index from index into the checkerboard

  face_idx *= 2;

  if (!(face_X & 1)) { // face_X even
    //   int t = (face_idx / face_XYZ) % face_T;
    //   int z = (face_idx / face_XY) % face_Z;
    //   int y = (face_idx / face_X) % face_Y;
    //   face_idx += (face_parity + t + z + y) & 1;//the same parity for both flavors 
    // equivalent to the above, but with fewer divisions/mods:
    int aux1 = face_idx / face_X;
    int aux2 = aux1 / face_Y;
    int aux3 = aux2 / face_Z;
    int y = aux1 - aux2 * face_Y;
    int z = aux2 - aux3 * face_Z;    
    int Nf = aux3 / face_T;
    int t  = aux3 - Nf * face_T;
    face_idx += (face_parity + t + z + y) & 1;
  } else if (!(face_Y & 1)) { // face_Y even
    int t  = (face_idx / face_XYZ) % face_T;
    int z  = (face_idx / face_XY) % face_Z;
    face_idx += (face_parity + t + z) & 1;
  } else if (!(face_Z & 1)) { // face_Z even
    int t = (face_idx / face_XYZ) % face_T;
    face_idx += (face_parity + t) & 1;
  } else if(!(face_T)){
    face_idx += face_parity & 1;
  }else{    
    face_idx += face_parity;
  }

  // compute index into the full local volume

  int idx = face_idx;
  int gap, aux;

  switch (dim) {
    case 0:
      gap = X[0] - nLayers;
      aux = face_idx / face_X;
      idx += (aux + face_num) * gap;
      break;
    case 1:
      gap = X[1] - nLayers;
      aux = face_idx / face_XY;
      idx += (aux + face_num) * gap * face_X;
      break;
    case 2:
      gap = X[2] - nLayers;
      aux = face_idx / face_XYZ;
      idx += (aux + face_num) * gap * face_XY;
      break;
    case 3:
      gap = X[3] - nLayers;
      aux = face_idx / face_XYZT;
      idx += (aux + face_num) * gap * face_XYZ;
      break;
  }

  // return index into the checkerboard

  return idx >> 1;
}



template <int dim>
static inline __device__ bool inBoundary(const int width, const int coord[], const int X[]){
  return ((coord[dim] >= X[dim] - width) || (coord[dim] < width));
}


static inline __device__ bool isActive(const int threadDim, int offsetDim, int offset, const int y[],  const int partitioned[], const int X[])
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

static inline __device__ bool isActive(const int threadDim, int offsetDim, int offset, int x1, int x2, int x3, int x4,
                                       const int partitioned[], const int X[])
{
  int y[4] = {x1, x2, x3, x4};
  return isActive(threadDim, offsetDim, offset, y, partitioned, X);
}

/**
 *  * Determines which face a given thread is computing.  Also rescale
 *   * face_idx so that is relative to a given dimension.
 *    */
template <typename Param>
__device__ inline int dimFromFaceIndex (int &face_idx, const Param &param) {
  if (face_idx < param.threadDimMapUpper[0]) {
    return 0;
  } else if (face_idx < param.threadDimMapUpper[1]) {
    face_idx -= param.threadDimMapLower[1];
    return 1;
  } else if (face_idx < param.threadDimMapUpper[2]) {
    face_idx -= param.threadDimMapLower[2];
    return 2;
  } else { // this is only called if we use T kernel packing 
    face_idx -= param.threadDimMapLower[3];
    return 3;
  }
}

template <typename Param> 
__device__ inline int dimFromDWFaceIndex(int &face_idx, const Param &param){

  // s - the coordinate in the fifth dimension - is the slowest-changing coordinate
  const int s = face_idx/param.threads;

  face_idx = face_idx - s*param.threads; // face_idx = face_idx % param.threads

  if (face_idx < param.threadDimMapUpper[0]){
    face_idx += s*param.threadDimMapUpper[0];
    return 0;
  } else if (face_idx < param.threadDimMapUpper[1]){
    face_idx -= param.threadDimMapLower[1];
    face_idx += s*(param.threadDimMapUpper[1] - param.threadDimMapLower[1]);
    return 1;
  } else if (face_idx < param.threadDimMapUpper[2]){
    face_idx -= param.threadDimMapLower[2];
    face_idx += s*(param.threadDimMapUpper[2] - param.threadDimMapLower[2]);
    return 2;
  } else  { 
    face_idx -= param.threadDimMapLower[3];
    face_idx += s*(param.threadDimMapUpper[3] - param.threadDimMapLower[3]);
    return 3;
  }
}

template<int nLayers>
static inline __device__ void faceIndexFromCoords(int &face_idx, int x, int y, int z, int t, int face_dim, const int X[4])
{
  int D[4] = {X[0], X[1], X[2], X[3]};

  switch(face_dim){
    case 0:
      x = (x < nLayers) ? x : x - (X[0] - nLayers);
      break;
    case 1:
      y = (y < nLayers) ? y : y - (X[1] - nLayers);
      break;
    case 2:
      z = (z < nLayers) ? z : z - (X[2] - nLayers);
      break;
    case 3:
      t = (t < nLayers) ? t : t - (X[3] - nLayers);
      break;
  }
  D[face_dim] = nLayers;

  face_idx = ((((D[2]*t + z)*D[1] + y)*D[0] + x) >> 1);

  return;
}

template<int nLayers> 
static inline __device__ void faceIndexFromDWCoords(int &face_idx, int x, int y, int z, int t, int s, int face_dim, const int X[4])
{
  int D[4] = {X[0], X[1], X[2], X[3]};
  
  switch(face_dim){
    case 0:
      x = (x < nLayers) ? x : x - (X[0] - nLayers);
      break;
    case 1:
      y = (y < nLayers) ? y : y - (X[1] - nLayers);
      break;
    case 2:
      z = (z < nLayers) ? z : z - (X[2] - nLayers);
      break;
    case 3:
      t = (t < nLayers) ? t : t - (X[3] - nLayers);
      break;
  }
  D[face_dim] = nLayers;

  face_idx = (((((D[3]*s + t)*D[2] + z)*D[1] + y)*D[0] + x) >> 1);

  return;
}


 
