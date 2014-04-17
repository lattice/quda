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
static inline __device__ void coordsFromFaceIndex(int &idx, int &cb_idx, Int &X, Int &Y, Int &Z, Int &T, int face_idx,
    const int &face_volume, const int &dim, const int &face_num, const int &parity)
{
  // dimensions of the face (FIXME: optimize using constant cache)

  int face_X = X1, face_Y = X2, face_Z = X3;
  int face_parity;
  switch (dim) {
    case 0:
      face_X = nLayers;
      face_parity = (parity + face_num * (X1 - nLayers)) & 1;
      break;
    case 1:
      face_Y = nLayers;
      face_parity = (parity + face_num * (X2 - nLayers)) & 1;
      break;
    case 2:
      face_Z = nLayers;
      face_parity = (parity + face_num * (X3 - nLayers)) & 1;
      break;
    case 3:
      face_parity = (parity + face_num * (X4 - nLayers)) & 1;
      break;
  }
  int face_XYZ = face_X * face_Y * face_Z;
  int face_XY = face_X * face_Y;

  // compute coordinates from (checkerboard) face index

  face_idx *= 2;

  int x, y, z, t;

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
      x += face_num * (X1-nLayers);
      break;
    case 1:
      y += face_num * (X2-nLayers);
      break;
    case 2:
      z += face_num * (X3-nLayers);
      break;
    case 3:
      t += face_num * (X4-nLayers);
      break;
  }

  // compute index into the full local volume

  idx = X1*(X2*(X3*t + z) + y) + x; 

  // compute index into the checkerboard

  cb_idx = idx >> 1;

  X = x;
  Y = y;
  Z = z;
  T = t;  

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
static __device__ __forceinline__ void coordsFromIndex(int &idx, Int &X, Int &Y, Int &Z, Int &T, 
    const int &cb_idx, const int &parity)
{
  int &LX = X1;
  int &LY = X2;
  int &LZ = X3;
  int &XYZ = X3X2X1;
  int &XY = X2X1;

  idx = 2*cb_idx;

  int x, y, z, t;

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

  X = x;
  Y = y;
  Z = z;
  T = t;
}





// compute coordinates from index into the checkerboard (used by the interior Dslash kernels)
// This is the variant used byt the shared memory wilson dslash
  template <IndexType idxType, typename Int>
static __device__ __forceinline__ void coordsFromIndex3D(int &idx, Int &X, Int &Y, Int &Z, Int &T, 
    int &cb_idx, const int &parity)
{
  int &LX = X1;
  int &LY = X2;
  int &LZ = X3;

  int x, y, z, t;

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

  X = x;
  Y = y;
  Z = z;
  T = t;
}

//Used in DW kernels only:

  template <int dim, int nLayers, int face_num>
static inline __device__ int indexFromDWFaceIndex(int face_idx, const int &face_volume,
    const int &parity)
{
  // dimensions of the face (FIXME: optimize using constant cache)

  //A.S.: Also used for computing offsets in physical lattice
  //A.S.: note that in the case of DW fermions one is dealing with 4d faces

  // intrinsic parity of the face depends on offset of first element, used for MPI DW as well
  int face_X = X1, face_Y = X2, face_Z = X3, face_T = X4;
  int face_parity;  

  switch (dim) {
    case 0:
      face_X = nLayers;
      face_parity = (parity + face_num * (X1 - nLayers)) & 1;
      break;
    case 1:
      face_Y = nLayers;
      face_parity = (parity + face_num * (X2 - nLayers)) & 1;
      break;
    case 2:
      face_Z = nLayers;
      face_parity = (parity + face_num * (X3 - nLayers)) & 1;
      break;
    case 3:
      face_T = nLayers;    
      face_parity = (parity + face_num * (X4 - nLayers)) & 1;
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
      gap = X1 - nLayers;
      aux = face_idx / face_X;
      idx += (aux + face_num) * gap;
      break;
    case 1:
      gap = X2 - nLayers;
      aux = face_idx / face_XY;
      idx += (aux + face_num) * gap * face_X;
      break;
    case 2:
      gap = X3 - nLayers;
      aux = face_idx / face_XYZ;
      idx += (aux + face_num) * gap * face_XY;
      break;
    case 3:
      gap = X4 - nLayers;
      aux = face_idx / face_XYZT;
      idx += (aux + face_num) * gap * face_XYZ;
      break;
  }

  // return index into the checkerboard

  return idx >> 1;
}


// compute full coordinates from an index into the face (used by the exterior Dslash kernels)
  template <int nLayers, typename Int>
static inline __device__ void coordsFromDWFaceIndex(int &cb_idx, Int &X, Int &Y, Int &Z, Int &T, Int &S, int face_idx,
    const int &face_volume, const int &dim, const int &face_num, const int &parity)
{
  // dimensions of the face (FIXME: optimize using constant cache)

  int face_X = X1, face_Y = X2, face_Z = X3, face_T = X4;
  int face_parity;
  switch (dim) {
    case 0:
      face_X = nLayers;
      face_parity = (parity + face_num * (X1 - nLayers)) & 1;
      break;
    case 1:
      face_Y = nLayers;
      face_parity = (parity + face_num * (X2 - nLayers)) & 1;
      break;
    case 2:
      face_Z = nLayers;
      face_parity = (parity + face_num * (X3 - nLayers)) & 1;
      break;
    case 3:
      face_T = nLayers;    
      face_parity = (parity + face_num * (X4 - nLayers)) & 1;
      break;
  }
  int face_XYZT = face_X * face_Y * face_Z * face_T;  
  int face_XYZ  = face_X * face_Y * face_Z;
  int face_XY   = face_X * face_Y;

  // compute coordinates from (checkerboard) face index

  face_idx *= 2;

  int x, y, z, t, s;

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
      x += face_num * (X1-nLayers);
      break;
    case 1:
      y += face_num * (X2-nLayers);
      break;
    case 2:
      z += face_num * (X3-nLayers);
      break;
    case 3:
      t += face_num * (X4-nLayers);
      break;
  }

  // compute index into the checkerboard

  cb_idx = (X1*(X2*(X3*(X4*s + t) + z) + y) + x) >> 1;

  X = x;
  Y = y;
  Z = z;
  T = t;  
  S = s;
  //printf("Global sid %d (%d, %d, %d, %d)\n", cb_int, x, y, z, t);
}


//!ndeg tm:
  template <int dim, int nLayers, int face_num>
static inline __device__ int indexFromNdegTMFaceIndex(int face_idx, const int &face_volume,
    const int &parity)
{
  // dimensions of the face (FIXME: optimize using constant cache)

  int face_X = X1, face_Y = X2, face_Z = X3, face_T = X4;
  int face_parity;  

  switch (dim) {
    case 0:
      face_X = nLayers;
      face_parity = (parity + face_num * (X1 - nLayers)) & 1;
      break;
    case 1:
      face_Y = nLayers;
      face_parity = (parity + face_num * (X2 - nLayers)) & 1;
      break;
    case 2:
      face_Z = nLayers;
      face_parity = (parity + face_num * (X3 - nLayers)) & 1;
      break;
    case 3:
      face_T = nLayers;    
      face_parity = (parity + face_num * (X4 - nLayers)) & 1;
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
      gap = X1 - nLayers;
      aux = face_idx / face_X;
      idx += (aux + face_num) * gap;
      break;
    case 1:
      gap = X2 - nLayers;
      aux = face_idx / face_XY;
      idx += (aux + face_num) * gap * face_X;
      break;
    case 2:
      gap = X3 - nLayers;
      aux = face_idx / face_XYZ;
      idx += (aux + face_num) * gap * face_XY;
      break;
    case 3:
      gap = X4 - nLayers;
      aux = face_idx / face_XYZT;
      idx += (aux + face_num) * gap * face_XYZ;
      break;
  }

  // return index into the checkerboard

  return idx >> 1;
}


// routines for packing the ghost zones (multi-GPU only)

#ifdef MULTI_GPU

template <typename FloatN>
struct PackParam {

  FloatN *out[2*QUDA_MAX_DIM];
  float *outNorm[2*QUDA_MAX_DIM];

  FloatN *in;
  float *inNorm;

  FloatN *clover;
  FloatN *cloverInv;
  float *cloverNorm;
  float *cloverInvNorm;

  int threads; // total number of threads

  // offsets which determine thread mapping to dimension
  int threadDimMapLower[QUDA_MAX_DIM]; // lowest thread which maps to dim
  int threadDimMapUpper[QUDA_MAX_DIM]; // greatest thread + 1 which maps to dim

  int parity;
#ifdef USE_TEXTURE_OBJECTS
  cudaTextureObject_t inTex;
  cudaTextureObject_t inTexNorm;
  cudaTextureObject_t cloverTex;
  cudaTextureObject_t cloverNormTex;
  cudaTextureObject_t cloverInvTex;
  cudaTextureObject_t cloverInvNormTex;
#endif

  int dim;
  int face_num;
  int X[QUDA_MAX_DIM]; // lattice dimensions

  int stride;
};


// Extend the PackParam class to PackExtendedParam
template<typename Float>
struct PackExtendedParam : public PackParam<Float>
{
  PackExtendedParam(){}
  PackExtendedParam(const PackParam<Float>& base) : PackParam<Float>(base) {}
  int R[QUDA_MAX_DIM]; // boundary dimensions
};

/**
 * Determines which face a given thread is computing.  Also rescale
 * face_idx so that is relative to a given dimension.
 */
template <typename Param>
__device__ inline int dimFromFaceIndex (int &face_idx, const Param param) {
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

#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)

// double precision
#if (defined DIRECT_ACCESS_WILSON_PACK_SPINOR) || (defined FERMI_NO_DBLE_TEX)
#define READ_SPINOR READ_SPINOR_DOUBLE
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_DOUBLE_TEX
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexDouble
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_DOUBLE2
#define SPINOR_DOUBLE
  template <int dim, int dagger, int face_num>
static inline __device__ void packFaceWilsonCore(double2 *out, float *outNorm, const double2 *in, 
    const float *inNorm, const int &idx, 
    const int &face_idx, const int &face_volume, 
    PackParam<double2> &param)
{
#if (__COMPUTE_CAPABILITY__ >= 130)
  if (dagger) {
#include "wilson_pack_face_dagger_core.h"
  } else {
#include "wilson_pack_face_core.h"
  }
#endif // (__COMPUTE_CAPABILITY__ >= 130)
}
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR
#undef SPINOR_DOUBLE


// single precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_SINGLE_TEX
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexSingle
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_FLOAT4
  template <int dim, int dagger, int face_num>
static inline __device__ void packFaceWilsonCore(float4 *out, float *outNorm, const float4 *in, const float *inNorm,
    const int &idx, const int &face_idx, 
    const int &face_volume, 
    const PackParam<float4> &param)
{
  if (dagger) {
#include "wilson_pack_face_dagger_core.h"
  } else {
#include "wilson_pack_face_core.h"
  }
}
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR


// half precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_HALF_TEX
#define READ_SPINOR_UP READ_SPINOR_HALF_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexHalf
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_SHORT4
  template <int dim, int dagger, int face_num>
static inline __device__ void packFaceWilsonCore(short4 *out, float *outNorm, const short4 *in, const float *inNorm,
    const int &idx, const int &face_idx, 
    const int &face_volume, 
    const PackParam<short4> &param)
{
  if (dagger) {
#include "wilson_pack_face_dagger_core.h"
  } else {
#include "wilson_pack_face_core.h"
  }
}
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR

  template <int dagger, typename FloatN>
__global__ void packFaceWilsonKernel(PackParam<FloatN> param)
{
  const int nFace = 1; // 1 face for Wilson

  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (face_idx >= param.threads) return;

  // determine which dimension we are packing
  const int dim = dimFromFaceIndex(face_idx, param);

  // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
  const int face_num = (face_idx >= nFace*ghostFace[dim]) ? 1 : 0;
  face_idx -= face_num*nFace*ghostFace[dim];

  // compute where the output is located
  // compute an index into the local volume from the index into the face
  // read spinor, spin-project, and write half spinor to face
  if (dim == 0) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<0,nFace,0>(face_idx,ghostFace[0],param.parity,param.X);
      packFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
          param.inNorm,idx, face_idx, ghostFace[0], param);
    } else {
      const int idx = indexFromFaceIndex<0,nFace,1>(face_idx,ghostFace[0],param.parity,param.X);
      packFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
          param.inNorm,idx, face_idx, ghostFace[0], param);
    }
  } else if (dim == 1) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<1,nFace,0>(face_idx,ghostFace[1],param.parity,param.X);
      packFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
          param.inNorm,idx, face_idx, ghostFace[1], param);
    } else {
      const int idx = indexFromFaceIndex<1,nFace,1>(face_idx,ghostFace[1],param.parity,param.X);
      packFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
          param.inNorm,idx, face_idx, ghostFace[1], param);
    }
  } else if (dim == 2) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<2,nFace,0>(face_idx,ghostFace[2],param.parity,param.X);
      packFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
          param.inNorm,idx, face_idx, ghostFace[2], param);
    } else {
      const int idx = indexFromFaceIndex<2,nFace,1>(face_idx,ghostFace[2],param.parity,param.X);
      packFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
          param.inNorm,idx, face_idx, ghostFace[2], param);
    }
  } else {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<3,nFace,0>(face_idx,ghostFace[3],param.parity,param.X);
      packFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
          param.inNorm,idx, face_idx, ghostFace[3], param);
    } else {
      const int idx = indexFromFaceIndex<3,nFace,1>(face_idx,ghostFace[3],param.parity,param.X);
      packFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
          param.inNorm,idx, face_idx, ghostFace[3], param);
    }
  }

}

#endif // GPU_WILSON_DIRAC || GPU_DOMAIN_WALL_DIRAC


#if defined(GPU_WILSON_DIRAC) || defined(GPU_TWISTED_MASS_DIRAC)

// double precision
#if (defined DIRECT_ACCESS_WILSON_PACK_SPINOR) || (defined FERMI_NO_DBLE_TEX)
#define READ_SPINOR READ_SPINOR_DOUBLE
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_DOUBLE_TEX
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexDouble
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_DOUBLE2
#define SPINOR_DOUBLE
  template <int dim, int dagger, int face_num>
static inline __device__ void packTwistedFaceWilsonCore(double2 *out, float *outNorm, const double2 *in, 
    const float *inNorm, double a, double b, const int &idx, 
    const int &face_idx, const int &face_volume, 
    PackParam<double2> &param)
{
#if (__COMPUTE_CAPABILITY__ >= 130)
  if (dagger) {
#include "wilson_pack_twisted_face_dagger_core.h"
  } else {
#include "wilson_pack_twisted_face_core.h"
  }
#endif // (__COMPUTE_CAPABILITY__ >= 130)
}
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR
#undef SPINOR_DOUBLE


// single precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_SINGLE_TEX
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexSingle
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_FLOAT4
  template <int dim, int dagger, int face_num>
static inline __device__ void packTwistedFaceWilsonCore(float4 *out, float *outNorm, const float4 *in, const float *inNorm, float a, float b,
    const int &idx, const int &face_idx, 
    const int &face_volume, 
    const PackParam<float4> &param)
{
  if (dagger) {
#include "wilson_pack_twisted_face_dagger_core.h"
  } else {
#include "wilson_pack_twisted_face_core.h"
  }
}
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR


// half precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_HALF_TEX
#define READ_SPINOR_UP READ_SPINOR_HALF_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexHalf
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_SHORT4
  template <int dim, int dagger, int face_num>
static inline __device__ void packTwistedFaceWilsonCore(short4 *out, float *outNorm, const short4 *in, const float *inNorm, float a, float b,
    const int &idx, const int &face_idx, 
    const int &face_volume, 
    const PackParam<short4> &param)
{
  if (dagger) {
#include "wilson_pack_twisted_face_dagger_core.h"
  } else {
#include "wilson_pack_twisted_face_core.h"
  }
}
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR

  template <int dagger, typename FloatN, typename Float>
__global__ void packTwistedFaceWilsonKernel(Float a, Float b, PackParam<FloatN> param)
{
  const int nFace = 1; // 1 face for Wilson

  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (face_idx >= param.threads) return;

  // determine which dimension we are packing
  const int dim = dimFromFaceIndex(face_idx, param);

  // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
  const int face_num = (face_idx >= nFace*ghostFace[dim]) ? 1 : 0;
  face_idx -= face_num*nFace*ghostFace[dim];

  // compute where the output is located
  // compute an index into the local volume from the index into the face
  // read spinor, spin-project, and write half spinor to face
  if (dim == 0) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<0,nFace,0>(face_idx,ghostFace[0],param.parity,param.X);
      packTwistedFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[0], param);
    } else {
      const int idx = indexFromFaceIndex<0,nFace,1>(face_idx,ghostFace[0],param.parity,param.X);
      packTwistedFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[0], param);
    }
  } else if (dim == 1) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<1,nFace,0>(face_idx,ghostFace[1],param.parity,param.X);
      packTwistedFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[1], param);
    } else {
      const int idx = indexFromFaceIndex<1,nFace,1>(face_idx,ghostFace[1],param.parity,param.X);
      packTwistedFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[1], param);
    }
  } else if (dim == 2) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<2,nFace,0>(face_idx,ghostFace[2],param.parity,param.X);
      packTwistedFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[2], param);
    } else {
      const int idx = indexFromFaceIndex<2,nFace,1>(face_idx,ghostFace[2],param.parity,param.X);
      packTwistedFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[2], param);
    }
  } else {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<3,nFace,0>(face_idx,ghostFace[3],param.parity,param.X);
      packTwistedFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
          param.inNorm, a, b,idx, face_idx, ghostFace[3], param);
    } else {
      const int idx = indexFromFaceIndex<3,nFace,1>(face_idx,ghostFace[3],param.parity,param.X);
      packTwistedFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[3], param);
    }
  }

}

#endif // GPU_TWISTED_MASS_DIRAC

#if defined(GPU_WILSON_DIRAC) || defined(GPU_TWISTED_CLOVER_DIRAC)

// double precision
#if (defined DIRECT_ACCESS_WILSON_PACK_SPINOR) || (defined FERMI_NO_DBLE_TEX)
#define READ_SPINOR READ_SPINOR_DOUBLE
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_DOUBLE_TEX
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexDouble
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_DOUBLE2
#define SPINOR_DOUBLE

	#if (defined DIRECT_ACCESS_CLOVER) || (defined FERMI_NO_DBLE_TEX)
		#define TMCLOVERTEX (param.clover)
		#define TM_INV_CLOVERTEX (param.cloverInv)
		#define ASSN_CLOVER PACK_CLOVER_DOUBLE
	#else
		#ifdef USE_TEXTURE_OBJECTS
			#define TMCLOVERTEX (param.cloverTex)
			#define TM_INV_CLOVERTEX (param.cloverInvTex)
		#else
			#define TMCLOVERTEX cloverTexDouble
			#define TM_INV_CLOVERTEX cloverInvTexDouble
		#endif
		#define ASSN_CLOVER PACK_CLOVER_DOUBLE_TEX
	#endif
	#define CLOVER_DOUBLE

  template <int dim, int dagger, int face_num>
static inline __device__ void packCloverTwistedFaceWilsonCore(double2 *out, float *outNorm, const double2 *in, 
    const float *inNorm, double a, const int &idx, 
    const int &face_idx, const int &face_volume, 
    PackParam<double2> &param)
{
#if (__COMPUTE_CAPABILITY__ >= 130)
  if (dagger) {
#include "wilson_pack_clover_twisted_face_dagger_core.h"
  } else {
#include "wilson_pack_clover_twisted_face_core.h"
  }
#endif // (__COMPUTE_CAPABILITY__ >= 130)
}
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR
#undef SPINOR_DOUBLE

#undef TMCLOVERTEX
#undef TM_INV_CLOVERTEX
#undef READ_CLOVER
#undef ASSN_CLOVER
#undef CLOVER_DOUBLE

// single precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_SINGLE_TEX
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexSingle
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_FLOAT4

	#ifdef DIRECT_ACCESS_CLOVER
		#define TMCLOVERTEX (param.clover)
		#define TM_INV_CLOVERTEX (param.cloverInv)
		#define ASSN_CLOVER PACK_CLOVER_SINGLE
	#else
		#ifdef USE_TEXTURE_OBJECTS
			#define TMCLOVERTEX (param.cloverTex)
			#define TM_INV_CLOVERTEX (param.cloverInvTex)
		#else
			#define TMCLOVERTEX cloverTexSingle
			#define TM_INV_CLOVERTEX cloverInvTexSingle
		#endif
		#define ASSN_CLOVER PACK_CLOVER_SINGLE_TEX
	#endif

  template <int dim, int dagger, int face_num>
static inline __device__ void packCloverTwistedFaceWilsonCore(float4 *out, float *outNorm, const float4 *in, const float *inNorm, float a,
    const int &idx, const int &face_idx, 
    const int &face_volume, 
    const PackParam<float4> &param)
{
  if (dagger) {
#include "wilson_pack_clover_twisted_face_dagger_core.h"
  } else {
#include "wilson_pack_clover_twisted_face_core.h"
  }
}
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR

#undef TMCLOVERTEX
#undef TM_INV_CLOVERTEX
#undef ASSN_CLOVER

// half precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_HALF_TEX
#define READ_SPINOR_UP READ_SPINOR_HALF_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexHalf
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_SHORT4

	#ifdef DIRECT_ACCESS_CLOVER
		#define TMCLOVERTEX (param.clover)
		#define TMCLOVERTEXNORM (param.cloverNorm)
		#define TM_INV_CLOVERTEX (param.cloverInv)
		#define TM_INV_CLOVERTEXNORM (param.cloverInvNorm)
		#define ASSN_CLOVER PACK_CLOVER_HALF
	#else
		#ifdef USE_TEXTURE_OBJECTS
			#define TMCLOVERTEX (param.cloverTex)
			#define TMCLOVERTEXNORM (param.cloverNormTex)
			#define TM_INV_CLOVERTEX (param.cloverInvTex)
			#define TM_INV_CLOVERTEXNORM (param.cloverInvNormTex)
		#else
			#define TMCLOVERTEX cloverTexHalf
			#define TMCLOVERTEXNORM cloverTexNorm
			#define TM_INV_CLOVERTEX cloverInvTexHalf
			#define TM_INV_CLOVERTEXNORM cloverInvTexNorm
		#endif
		#define ASSN_CLOVER PACK_CLOVER_HALF_TEX
	#endif

  template <int dim, int dagger, int face_num>
static inline __device__ void packCloverTwistedFaceWilsonCore(short4 *out, float *outNorm, const short4 *in, const float *inNorm, float a,
    const int &idx, const int &face_idx, 
    const int &face_volume, 
    const PackParam<short4> &param)
{
  if (dagger) {
#include "wilson_pack_clover_twisted_face_dagger_core.h"
  } else {
#include "wilson_pack_clover_twisted_face_core.h"
  }
}
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR

#undef TMCLOVERTEX
#undef TM_INV_CLOVERTEX
#undef TMCLOVERTEXNORM
#undef TM_INV_CLOVERTEXNORM
#undef ASSN_CLOVER

  template <int dagger, typename FloatN, typename Float>
__global__ void packCloverTwistedFaceWilsonKernel(Float a, PackParam<FloatN> param)
{
  const int nFace = 1; // 1 face for Wilson

  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (face_idx >= param.threads) return;

  // determine which dimension we are packing
  const int dim = dimFromFaceIndex(face_idx, param);

  // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
  const int face_num = (face_idx >= nFace*ghostFace[dim]) ? 1 : 0;
  face_idx -= face_num*nFace*ghostFace[dim];

  // compute where the output is located
  // compute an index into the local volume from the index into the face
  // read spinor, spin-project, and write half spinor to face
  if (dim == 0) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<0,nFace,0>(face_idx,ghostFace[0],param.parity,param.X);
      packCloverTwistedFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
          param.inNorm, a, idx, face_idx, ghostFace[0], param);
    } else {
      const int idx = indexFromFaceIndex<0,nFace,1>(face_idx,ghostFace[0],param.parity,param.X);
      packCloverTwistedFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
          param.inNorm, a, idx, face_idx, ghostFace[0], param);
    }
  } else if (dim == 1) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<1,nFace,0>(face_idx,ghostFace[1],param.parity,param.X);
      packCloverTwistedFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
          param.inNorm, a, idx, face_idx, ghostFace[1], param);
    } else {
      const int idx = indexFromFaceIndex<1,nFace,1>(face_idx,ghostFace[1],param.parity,param.X);
      packCloverTwistedFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
          param.inNorm, a, idx, face_idx, ghostFace[1], param);
    }
  } else if (dim == 2) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<2,nFace,0>(face_idx,ghostFace[2],param.parity,param.X);
      packCloverTwistedFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
          param.inNorm, a, idx, face_idx, ghostFace[2], param);
    } else {
      const int idx = indexFromFaceIndex<2,nFace,1>(face_idx,ghostFace[2],param.parity,param.X);
      packCloverTwistedFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
          param.inNorm, a, idx, face_idx, ghostFace[2], param);
    }
  } else {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<3,nFace,0>(face_idx,ghostFace[3],param.parity,param.X);
      packCloverTwistedFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
          param.inNorm, a,idx, face_idx, ghostFace[3], param);
    } else {
      const int idx = indexFromFaceIndex<3,nFace,1>(face_idx,ghostFace[3],param.parity,param.X);
      packCloverTwistedFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
          param.inNorm, a, idx, face_idx, ghostFace[3], param);
    }
  }

}

#endif // GPU_TWISTED_CLOVER_DIRAC



template <typename FloatN, typename Float>
class PackFace : public Tunable {

  protected:
    FloatN *faces;
    const cudaColorSpinorField *in;
    const FullClover *clov;
    const FullClover *clovInv;
    const int dagger;
    const int parity;
    const int nFace;
    const int dim;
    const int face_num;

    // compute how many threads we need in total for the face packing

    unsigned int threads() const {
      unsigned int threads = 0;
      if(dim < 0){ // if dim is negative, pack all dimensions
        for (int i=0; i<4; i++) {
          if (!dslashParam.commDim[i]) continue;
          if ((i==3 && !(getKernelPackT() || getTwistPack()))) continue; 
          threads += 2*nFace*in->GhostFace()[i]; // 2 for forwards and backwards faces
        }
      }else{ // pack only in dim dimension
        if(dslashParam.commDim[dim] && dim!=3 || (getKernelPackT() || getTwistPack())){
          threads = nFace*in->GhostFace()[dim];
          if(face_num==2) threads *= 2; // sending data forwards and backwards
        }
      }
      return threads;
    }

    virtual int inputPerSite() const = 0;
    virtual int outputPerSite() const = 0;

    // prepare the param struct with kernel arguments
    PackParam<FloatN> prepareParam(int dim=-1, int face_num=2) {
      PackParam<FloatN> param;
      param.in = (FloatN*)in->V();
      param.inNorm = (float*)in->Norm();
      param.dim = dim;
      param.face_num = face_num;
      param.parity = parity;
      for(int d=0; d<QUDA_MAX_DIM; d++) param.X[d] = in->X()[d];
      param.X[0] *= 2;

      if (clov != NULL && clovInv != NULL) {
        if (param.parity == QUDA_EVEN_PARITY) {
          param.clover = (FloatN*)clov->even;
          param.cloverNorm = (float*)clov->evenNorm;
          param.cloverInv = (FloatN*)clovInv->even;
          param.cloverInvNorm = (float*)clovInv->evenNorm;
	} else {
          param.clover = (FloatN*)clov->odd;
          param.cloverNorm = (float*)clov->oddNorm;
          param.cloverInv = (FloatN*)clovInv->odd;
          param.cloverInvNorm = (float*)clovInv->oddNorm;
	}	
      }

#ifdef USE_TEXTURE_OBJECTS
      param.inTex = in->Tex();
      param.inTexNorm = in->TexNorm();
      if (clov != NULL && clovInv != NULL) {
        if (param.parity == QUDA_EVEN_PARITY) {
          param.cloverTex = clov->evenTex;
          param.cloverNormTex = clov->evenNormTex;
          param.cloverInvTex = clovInv->evenTex;
          param.cloverInvNormTex = clovInv->evenNormTex;
	} else {
          param.cloverTex = clov->oddTex;
          param.cloverNormTex = clov->oddNormTex;
          param.cloverInvTex = clovInv->oddTex;
          param.cloverInvNormTex = clovInv->oddNormTex;
	}
      }
#endif

      param.threads = threads();
      param.stride = in->Stride();

      int prev = -1; // previous dimension that was partitioned
      for (int i=0; i<4; i++) {
        param.threadDimMapLower[i] = 0;
        param.threadDimMapUpper[i] = 0;
        if (!dslashParam.commDim[i]) continue;
        param.threadDimMapLower[i] = (prev>=0 ? param.threadDimMapUpper[prev] : 0);
        param.threadDimMapUpper[i] = param.threadDimMapLower[i] + 2*nFace*in->GhostFace()[i];

        size_t faceBytes = nFace*outputPerSite()*in->GhostFace()[i]*sizeof(faces->x);

        if (typeid(FloatN) == typeid(short4) || typeid(FloatN) == typeid(short2)) {
          faceBytes += nFace*in->GhostFace()[i]*sizeof(float);
          param.out[2*i] = (FloatN*)((char*)faces + 
              (outputPerSite()*sizeof(faces->x) + sizeof(float))*param.threadDimMapLower[i]);
          param.outNorm[2*i] = (float*)((char*)param.out[2*i] + 
              nFace*outputPerSite()*in->GhostFace()[i]*sizeof(faces->x));
        } else {
          param.out[2*i] = (FloatN*)((char*)faces+outputPerSite()*sizeof(faces->x)*param.threadDimMapLower[i]);
        }

        param.out[2*i+1] = (FloatN*)((char*)param.out[2*i] + faceBytes);
        param.outNorm[2*i+1] = (float*)((char*)param.outNorm[2*i] + faceBytes);

        prev=i;

        //printf("%d: map=%d %d out=%llu %llu outNorm=%llu %llu bytes=%d\n", 
        //     i,param.threadDimMapLower[i],  param.threadDimMapUpper[i], 
        //     param.out[2*i], param.out[2*i+1], param.outNorm[2*i], param.outNorm[2*i+1], faceBytes);
      }

      return param;
    }

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return threads(); }

  public:
    PackFace(FloatN *faces, const cudaColorSpinorField *in, 
        const int dagger, const int parity, const int nFace, const int dim=-1, const int face_num=2)
      : faces(faces), in(in), clov(NULL), clovInv(NULL), dagger(dagger), parity(parity), nFace(nFace), dim(dim), face_num(face_num) { }
    PackFace(FloatN *faces, const cudaColorSpinorField *in, const FullClover *clov, const FullClover *clovInv,
        const int dagger, const int parity, const int nFace, const int dim=-1, const int face_num=2)
      : faces(faces), in(in), clov(clov), clovInv(clovInv), dagger(dagger), parity(parity), nFace(nFace), dim(dim), face_num(face_num) { }
    virtual ~PackFace() { }

    virtual int tuningIter() const { return 100; }

    virtual TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << in->X()[0] << "x";
      vol << in->X()[1] << "x";
      vol << in->X()[2] << "x";
      vol << in->X()[3];    
      aux << "threads=" <<threads() << ",stride=" << in->Stride() << ",prec=" << sizeof(((FloatN*)0)->x);
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }  

    virtual void apply(const cudaStream_t &stream) = 0;
    virtual void apply_twisted(Float a, Float b, const cudaStream_t &stream) = 0;//for twisted mass only
    virtual void apply_twisted_clover(Float a, const cudaStream_t &stream) = 0;//for twisted clover only

    long long bytes() const { 
      size_t faceBytes = (inputPerSite() + outputPerSite())*this->threads()*sizeof(((FloatN*)0)->x);
      if (sizeof(((FloatN*)0)->x) == QUDA_HALF_PRECISION) 
        faceBytes += 2*this->threads()*sizeof(float); // 2 is from input and output
      return faceBytes;
    }
};

template <typename FloatN, typename Float>
class PackFaceWilson : public PackFace<FloatN, Float> {

  private:

    int inputPerSite() const { return 24; } // input is full spinor
    int outputPerSite() const { return 12; } // output is spin projected

  public:
    PackFaceWilson(FloatN *faces, const cudaColorSpinorField *in, 
        const int dagger, const int parity)
      : PackFace<FloatN, Float>(faces, in, dagger, parity, 1) { }
    PackFaceWilson(FloatN *faces, const cudaColorSpinorField *in, 
        const FullClover *clov, const FullClover *clovInv, const int dagger, const int parity)
      : PackFace<FloatN, Float>(faces, in, clov, clovInv, dagger, parity, 1) { }
    virtual ~PackFaceWilson() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_WILSON_DIRAC
      PackParam<FloatN> param = this->prepareParam();
      if (this->dagger) {
        packFaceWilsonKernel<1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      } else {
        packFaceWilsonKernel<0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      }
#else
      errorQuda("Wilson face packing kernel is not built");
#endif  
    }

    void apply_twisted(Float a, Float b, const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_TWISTED_MASS_DIRAC
      PackParam<FloatN> param = this->prepareParam();
      if (this->dagger) {
        packTwistedFaceWilsonKernel<1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(a, b, param);
      } else {
        packTwistedFaceWilsonKernel<0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(a, b, param);
      }
#else
      errorQuda("Twisted face packing kernel is not built");
#endif  
    }

    void apply_twisted_clover(Float a, const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_TWISTED_CLOVER_DIRAC
      PackParam<FloatN> param = this->prepareParam();
      if (this->dagger) {
        packCloverTwistedFaceWilsonKernel<1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(a, param);
      } else {
        packCloverTwistedFaceWilsonKernel<0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(a, param);
      }
#else
      errorQuda("Twisted Clover face packing kernel is not built");
#endif  
    }

    long long flops() const { return outputPerSite()*this->threads(); }
};

void packFaceWilson(void *ghost_buf, cudaColorSpinorField &in, const int dagger, 
    const int parity, const cudaStream_t &stream) {

  switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceWilson<double2, double> pack((double2*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceWilson<float4, float> pack((float4*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceWilson<short4, float> pack((short4*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
  }  
}

//!
void packTwistedFaceWilson(void *ghost_buf, cudaColorSpinorField &in, const int dagger, 
    const int parity, const double a, const double b, const cudaStream_t &stream) {

  switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceWilson<double2, double> pack((double2*)ghost_buf, &in, dagger, parity);
        pack.apply_twisted((double)a, (double)b, stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceWilson<float4, float> pack((float4*)ghost_buf, &in, dagger, parity);
        pack.apply_twisted((float)a, (float)b, stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceWilson<short4, float> pack((short4*)ghost_buf, &in, dagger, parity);
        pack.apply_twisted((float)a, (float)b, stream);
      }
      break;
  }  
}

//!
void packCloverTwistedFaceWilson(void *ghost_buf, cudaColorSpinorField &in, FullClover &clover, FullClover &clovInv,
    const int dagger, const int parity, const double a, const cudaStream_t &stream) {

  switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceWilson<double2, double> pack((double2*)ghost_buf, &in, &clover, &clovInv, dagger, parity);
        pack.apply_twisted_clover((double)a, stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceWilson<float4, float> pack((float4*)ghost_buf, &in, &clover, &clovInv, dagger, parity);
        pack.apply_twisted_clover((float)a, stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceWilson<short4, float> pack((short4*)ghost_buf, &in, &clover, &clovInv, dagger, parity);
        pack.apply_twisted_clover((float)a, stream);
      }
      break;
  }  
}

#ifdef GPU_STAGGERED_DIRAC

#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEXDOUBLE param.inTex
#define SPINORTEXSINGLE param.inTex
#define SPINORTEXHALF param.inTex
#define SPINORTEXHALFNORM param.inTexNorm
#else
#define SPINORTEXDOUBLE spinorTexDouble
#define SPINORTEXSINGLE spinorTexSingle2
#define SPINORTEXHALF spinorTexHalf2
#define SPINORTEXHALFNORM spinorTexHalf2Norm
#endif

template <typename Float2>
__device__ void packFaceStaggeredCore(Float2 *out, float *outNorm, const int out_idx, 
    const int out_stride, const Float2 *in, const float *inNorm, 
    const int in_idx, const int in_stride) {
  out[out_idx + 0*out_stride] = in[in_idx + 0*in_stride];
  out[out_idx + 1*out_stride] = in[in_idx + 1*in_stride];
  out[out_idx + 2*out_stride] = in[in_idx + 2*in_stride];
}	
template<> 
__device__ void packFaceStaggeredCore(short2 *out, float *outNorm, const int out_idx, 
    const int out_stride, const short2 *in, const float *inNorm, 
    const int in_idx, const int in_stride) {
  out[out_idx + 0*out_stride] = in[in_idx + 0*in_stride];
  out[out_idx + 1*out_stride] = in[in_idx + 1*in_stride];
  out[out_idx + 2*out_stride] = in[in_idx + 2*in_stride];
  outNorm[out_idx] = inNorm[in_idx];
}

#if (defined DIRECT_ACCESS_PACK) || (defined FERMI_NO_DBLE_TEX)
template <typename Float2>
__device__ void packFaceStaggeredCore(Float2 *out, float *outNorm, const int out_idx, 
    const int out_stride, const Float2 *in, const float *inNorm, 
    const int in_idx, const PackParam<double2> &param) {
  out[out_idx + 0*out_stride] = in[in_idx + 0*param.stride];
  out[out_idx + 1*out_stride] = in[in_idx + 1*param.stride];
  out[out_idx + 2*out_stride] = in[in_idx + 2*param.stride];
}	
template<> 
__device__ void packFaceStaggeredCore(short2 *out, float *outNorm, const int out_idx, 
    const int out_stride, const short2 *in, const float *inNorm, 
    const int in_idx, const PackParam<double2> &param) {
  out[out_idx + 0*out_stride] = in[in_idx + 0*param.stride];
  out[out_idx + 1*out_stride] = in[in_idx + 1*param.stride];
  out[out_idx + 2*out_stride] = in[in_idx + 2*param.stride];
  outNorm[out_idx] = inNorm[in_idx];
}


#else
#if __COMPUTE_CAPABILITY__ >= 130
__device__ void packFaceStaggeredCore(double2 *out, float *outNorm, const int out_idx, 
    const int out_stride, const double2 *in, const float *inNorm, 
    const int in_idx, const PackParam<double2> &param) {
  out[out_idx + 0*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 0*param.stride);
  out[out_idx + 1*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 1*param.stride);
  out[out_idx + 2*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 2*param.stride);
}	
#endif
__device__ void packFaceStaggeredCore(float2 *out, float *outNorm, const int out_idx, 
    const int out_stride, const float2 *in, 
    const float *inNorm, const int in_idx, 
    const PackParam<float2> &param) {
  out[out_idx + 0*out_stride] = TEX1DFETCH(float2, SPINORTEXSINGLE, in_idx + 0*param.stride);
  out[out_idx + 1*out_stride] = TEX1DFETCH(float2, SPINORTEXSINGLE, in_idx + 1*param.stride);
  out[out_idx + 2*out_stride] = TEX1DFETCH(float2, SPINORTEXSINGLE, in_idx + 2*param.stride);	
}

// this is rather dumb: undoing the texture load because cudaNormalizedReadMode is used
// should really bind to an appropriate texture instead of reusing
static inline __device__ short2 float22short2(float c, float2 a) {
  return make_short2((short)(a.x*c*MAX_SHORT), (short)(a.y*c*MAX_SHORT));
}

__device__ void packFaceStaggeredCore(short2 *out, float *outNorm, const int out_idx, 
    const int out_stride, const short2 *in, 
    const float *inNorm, const int in_idx, 
    const PackParam<short2> &param) {
  out[out_idx + 0*out_stride] = float22short2(1.0f,TEX1DFETCH(float2,SPINORTEXHALF,in_idx+0*param.stride));
  out[out_idx + 1*out_stride] = float22short2(1.0f,TEX1DFETCH(float2,SPINORTEXHALF,in_idx+1*param.stride));
  out[out_idx + 2*out_stride] = float22short2(1.0f,TEX1DFETCH(float2,SPINORTEXHALF,in_idx+2*param.stride));
  outNorm[out_idx] = TEX1DFETCH(float, SPINORTEXHALFNORM, in_idx);
}
#endif


  template <typename FloatN, int nFace>
__global__ void packFaceStaggeredKernel(PackParam<FloatN> param)
{
  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (face_idx >= param.threads) return;

  // determine which dimension we are packing
  const int dim = dimFromFaceIndex(face_idx, param);

  // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
  const int face_num = (param.face_num==2) ? ((face_idx >= nFace*ghostFace[dim]) ? 1 : 0) : param.face_num;
  if(param.face_num==2) face_idx -= face_num*nFace*ghostFace[dim];

  // compute where the output is located
  // compute an index into the local volume from the index into the face
  // read spinor, spin-project, and write half spinor to face
  if (dim == 0) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexStaggered<0,nFace,0>(face_idx,ghostFace[0],param.parity,param.X);
      packFaceStaggeredCore(param.out[0], param.outNorm[0], face_idx, 
          nFace*ghostFace[0], param.in, param.inNorm, idx, param);
    } else {
      const int idx = indexFromFaceIndexStaggered<0,nFace,1>(face_idx,ghostFace[0],param.parity,param.X);
      packFaceStaggeredCore(param.out[1], param.outNorm[1], face_idx,
          nFace*ghostFace[0], param.in, param.inNorm, idx, param);
    }
  } else if (dim == 1) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexStaggered<1,nFace,0>(face_idx,ghostFace[1],param.parity,param.X);
      packFaceStaggeredCore(param.out[2], param.outNorm[2], face_idx, 
          nFace*ghostFace[1], param.in, param.inNorm, idx, param);
    } else {
      const int idx = indexFromFaceIndexStaggered<1,nFace,1>(face_idx,ghostFace[1],param.parity,param.X);
      packFaceStaggeredCore(param.out[3], param.outNorm[3], face_idx, 
          nFace*ghostFace[1], param.in, param.inNorm, idx, param);
    }
  } else if (dim == 2) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexStaggered<2,nFace,0>(face_idx,ghostFace[2],param.parity,param.X);
      packFaceStaggeredCore(param.out[4], param.outNorm[4], face_idx,
          nFace*ghostFace[2], param.in, param.inNorm, idx, param);
    } else {
      const int idx = indexFromFaceIndexStaggered<2,nFace,1>(face_idx,ghostFace[2],param.parity,param.X);
      packFaceStaggeredCore(param.out[5], param.outNorm[5], face_idx,
          nFace*ghostFace[2], param.in, param.inNorm, idx, param);
    }
  } else {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexStaggered<3,nFace,0>(face_idx,ghostFace[3],param.parity,param.X);
      packFaceStaggeredCore(param.out[6], param.outNorm[6], face_idx,
          nFace*ghostFace[3], param.in, param.inNorm,idx, param);
    } else {
      const int idx = indexFromFaceIndexStaggered<3,nFace,1>(face_idx,ghostFace[3],param.parity,param.X);
      packFaceStaggeredCore(param.out[7], param.outNorm[7], face_idx, 
          nFace*ghostFace[3], param.in, param.inNorm, idx, param);
    }
  }

}


  template <typename FloatN, int nFace>
__global__ void packFaceExtendedStaggeredKernel(PackExtendedParam<FloatN> param)
{
  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (face_idx >= param.threads) return;

  // determine which dimension we are packing
  const int dim = dimFromFaceIndex(face_idx, param);

  // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
  // if param.face_num==2 pack both the start and the end, otherwist pack the region of the 
  // lattice specified by param.face_num
  const int face_num = (param.face_num==2) ? ((face_idx >= nFace*ghostFace[dim]) ? 1 : 0) : param.face_num;
  if(param.face_num==2) face_idx -= face_num*nFace*ghostFace[dim];

  // compute where the output is located
  // compute an index into the local volume from the index into the face
  // read spinor, spin-project, and write half spinor to face
  if (dim == 0) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexExtendedStaggered<0,nFace,0>(face_idx,ghostFace[0],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.out[0], param.outNorm[0], face_idx, 
          nFace*ghostFace[0], param.in, param.inNorm, idx, param);
    } else {
      const int idx = indexFromFaceIndexExtendedStaggered<0,nFace,1>(face_idx,ghostFace[0],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.out[1], param.outNorm[1], face_idx,
          nFace*ghostFace[0], param.in, param.inNorm, idx, param);
    }
  } else if (dim == 1) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexExtendedStaggered<1,nFace,0>(face_idx,ghostFace[1],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.out[2], param.outNorm[2], face_idx, 
          nFace*ghostFace[1], param.in, param.inNorm, idx, param);
    } else {
      const int idx = indexFromFaceIndexExtendedStaggered<1,nFace,1>(face_idx,ghostFace[1],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.out[3], param.outNorm[3], face_idx, 
          nFace*ghostFace[1], param.in, param.inNorm, idx, param);
    }
  } else if (dim == 2) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexExtendedStaggered<2,nFace,0>(face_idx,ghostFace[2],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.out[4], param.outNorm[4], face_idx,
          nFace*ghostFace[2], param.in, param.inNorm, idx, param);
    } else {
      const int idx = indexFromFaceIndexExtendedStaggered<2,nFace,1>(face_idx,ghostFace[2],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.out[5], param.outNorm[5], face_idx,
          nFace*ghostFace[2], param.in, param.inNorm, idx, param);
    }
  } else {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexExtendedStaggered<3,nFace,0>(face_idx,ghostFace[3],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.out[6], param.outNorm[6], face_idx,
          nFace*ghostFace[3], param.in, param.inNorm,idx, param);
    } else {
      const int idx = indexFromFaceIndexExtendedStaggered<3,nFace,1>(face_idx,ghostFace[3],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.out[7], param.outNorm[7], face_idx, 
          nFace*ghostFace[3], param.in, param.inNorm, idx, param);
    }
  }

}


  template <typename FloatN, int nFace>
__global__ void unpackFaceExtendedStaggeredKernel(PackExtendedParam<FloatN> param)
{
  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (face_idx >= param.threads) return;

  // determine which dimension we are packing
  const int dim = dimFromFaceIndex(face_idx, param);

  // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
  // if param.face_num==2 pack both the start and the end, otherwist pack the region of the 
  // lattice specified by param.face_num
  const int face_num = (param.face_num==2) ? ((face_idx >= nFace*ghostFace[dim]) ? 1 : 0) : param.face_num;
  if(param.face_num==2) face_idx -= face_num*nFace*ghostFace[dim];

  // compute where the output is located
  // compute an index into the local volume from the index into the face
  // read spinor, spin-project, and write half spinor to face
  if (dim == 0) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexExtendedStaggered<0,nFace,0>(face_idx,ghostFace[0],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.in, param.inNorm, idx, 
          param.stride, param.out[0], param.outNorm[0], face_idx, nFace*ghostFace[0]);
    } else {
      const int idx = indexFromFaceIndexExtendedStaggered<0,nFace,1>(face_idx,ghostFace[0],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.in, param.inNorm, idx, 
          param.stride, param.out[1], param.outNorm[1], face_idx, nFace*ghostFace[0]);
    }
  } else if (dim == 1) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexExtendedStaggered<1,nFace,0>(face_idx,ghostFace[1],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.in, param.inNorm, idx, 
          param.stride, param.out[2], param.outNorm[2], face_idx, nFace*ghostFace[1]);
    } else {
      const int idx = indexFromFaceIndexExtendedStaggered<1,nFace,1>(face_idx,ghostFace[1],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.in, param.inNorm, idx, 
          param.stride, param.out[3], param.outNorm[3], face_idx, nFace*ghostFace[1]);
    }
  } else if (dim == 2) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexExtendedStaggered<2,nFace,0>(face_idx,ghostFace[2],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.in, param.inNorm, idx, 
          param.stride, param.out[4], param.outNorm[4], face_idx, nFace*ghostFace[2]);
    } else {
      const int idx = indexFromFaceIndexExtendedStaggered<2,nFace,1>(face_idx,ghostFace[2],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.in, param.inNorm, idx, 
          param.stride, param.out[5], param.outNorm[5], face_idx, nFace*ghostFace[2]);
    }
  } else {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexExtendedStaggered<3,nFace,0>(face_idx,ghostFace[3],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.in, param.inNorm, idx, 
          param.stride, param.out[6], param.outNorm[6], face_idx, nFace*ghostFace[3]);
    } else {
      const int idx = indexFromFaceIndexExtendedStaggered<3,nFace,1>(face_idx,ghostFace[3],param.parity,param.X,param.R);
      packFaceStaggeredCore(param.in, param.inNorm, idx, 
          param.stride, param.out[7], param.outNorm[7], face_idx, nFace*ghostFace[3]);
    }
  }

}


#undef SPINORTEXDOUBLE
#undef SPINORTEXSINGLE
#undef SPINORTEXHALF

#endif // GPU_STAGGERED_DIRAC


template <typename FloatN, typename Float>
class PackFaceStaggered : public PackFace<FloatN, Float> {

  private:
    const int* R; // boundary dimensions for extended field
    const bool unpack; 

    int inputPerSite() const { return 6; } // input is full spinor
    int outputPerSite() const { return 6; } // output is full spinor


  public:
    PackFaceStaggered(FloatN *faces, const cudaColorSpinorField *in, 
        const int nFace, const int dagger, const int parity, 
        const int dim, const int face_num, const int* R=NULL, const bool unpack=false)
      : PackFace<FloatN, Float>(faces, in, dagger, parity, nFace, dim, face_num), R(R), unpack(unpack) { }
    virtual ~PackFaceStaggered() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_STAGGERED_DIRAC

      PackParam<FloatN> param = this->prepareParam(this->dim, this->face_num);
      if(!R){
        if (PackFace<FloatN,Float>::nFace==1) {
          packFaceStaggeredKernel<FloatN, 1> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
        } else {
          packFaceStaggeredKernel<FloatN, 3> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
        }
      }else{ // R!=NULL => this is an extended field
        PackExtendedParam<FloatN> extendedParam(param);
        if(!unpack){
          for(int d=0; d<QUDA_MAX_DIM; ++d) extendedParam.R[d] = R[d];
          if(PackFace<FloatN,Float>::nFace==1){
            packFaceExtendedStaggeredKernel<FloatN,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(extendedParam);
          }else{
            packFaceExtendedStaggeredKernel<FloatN,3><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(extendedParam);
          }
        }else{
          if(PackFace<FloatN,Float>::nFace==1){
            unpackFaceExtendedStaggeredKernel<FloatN,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(extendedParam);
          }else{
            unpackFaceExtendedStaggeredKernel<FloatN,3><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(extendedParam);
          }
        }
      }
#else
      errorQuda("Staggered face packing kernel is not built");
#endif  
    }

    void apply_twisted(Float a, Float b, const cudaStream_t &stream) {}
    void apply_twisted_clover(Float a, const cudaStream_t &stream) {}

    long long flops() const { return 0; }
};


void packFaceStaggered(void *ghost_buf, cudaColorSpinorField &in, int nFace, 
    int dagger, int parity, const int dim, const int face_num, const cudaStream_t &stream) {

  switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
#if __COMPUTE_CAPABILITY__ >= 130
        PackFaceStaggered<double2, double> pack((double2*)ghost_buf, &in, nFace, dagger, parity, dim, face_num);
        pack.apply(stream);
#endif
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceStaggered<float2, float> pack((float2*)ghost_buf, &in, nFace, dagger, parity, dim, face_num);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceStaggered<short2, float> pack((short2*)ghost_buf, &in, nFace, dagger, parity, dim, face_num);
        pack.apply(stream);
      }
      break;
  }  
}

void packFaceExtendedStaggered(void *buffer, cudaColorSpinorField &field, const int nFace, const int R[],
    int dagger, int parity, const int dim, const int face_num, const cudaStream_t &stream, bool unpack=false)
{
  switch(field.Precision()){
    case QUDA_DOUBLE_PRECISION:
      {
#if __COMPUTE_CAPABILITY__ >= 130
        PackFaceStaggered<double2,double> pack(static_cast<double2*>(buffer), &field, nFace, dagger, parity, dim, face_num, R, unpack);
        pack.apply(stream);  
#endif
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceStaggered<float2,float> pack(static_cast<float2*>(buffer), &field, nFace, dagger, parity, dim, face_num, R, unpack);
        pack.apply(stream);  
      } 
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceStaggered<short2,float> pack(static_cast<short2*>(buffer), &field, nFace, dagger, parity, dim, face_num, R, unpack);
        pack.apply(stream);  
      }
      break;

  } // switch(field.Precision())
}

#ifdef GPU_DOMAIN_WALL_DIRAC
  template <int dagger, typename FloatN>
__global__ void packFaceDWKernel(PackParam<FloatN> param)
{
  const int nFace = 1; // 1 face for dwf

  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (face_idx >= param.threads) return;

  // determine which dimension we are packing
  const int dim = dimFromFaceIndex(face_idx, param);

  // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
  // FIXME these ghostFace constants do not incude the Ls dimension
  const int face_num = (face_idx >= nFace*Ls*ghostFace[dim]) ? 1 : 0; 
  face_idx -= face_num*nFace*Ls*ghostFace[dim];

  // compute where the output is located
  // compute an index into the local volume from the index into the face
  // read spinor, spin-project, and write half spinor to face
  if (dim == 0) {
    if (face_num == 0) {
      const int idx = indexFromDWFaceIndex<0,nFace,0>(face_idx,Ls*ghostFace[0],param.parity);
      packFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
          param.inNorm, idx, face_idx, Ls*ghostFace[0], param);
    } else {
      const int idx = indexFromDWFaceIndex<0,nFace,1>(face_idx,Ls*ghostFace[0],param.parity);
      packFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
          param.inNorm, idx, face_idx, Ls*ghostFace[0], param);
    }
  } else if (dim == 1) {
    if (face_num == 0) {
      const int idx = indexFromDWFaceIndex<1,nFace,0>(face_idx,Ls*ghostFace[1],param.parity);
      packFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
          param.inNorm, idx, face_idx, Ls*ghostFace[1], param);
    } else {
      const int idx = indexFromDWFaceIndex<1,nFace,1>(face_idx,Ls*ghostFace[1],param.parity);
      packFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
          param.inNorm, idx, face_idx, Ls*ghostFace[1], param);
    }
  } else if (dim == 2) {
    if (face_num == 0) {
      const int idx = indexFromDWFaceIndex<2,nFace,0>(face_idx,Ls*ghostFace[2],param.parity);
      packFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
          param.inNorm, idx, face_idx, Ls*ghostFace[2], param);
    } else {
      const int idx = indexFromDWFaceIndex<2,nFace,1>(face_idx,Ls*ghostFace[2],param.parity);
      packFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
          param.inNorm, idx, face_idx, Ls*ghostFace[2], param);
    }
  } else {
    if (face_num == 0) {
      const int idx = indexFromDWFaceIndex<3,nFace,0>(face_idx,Ls*ghostFace[3],param.parity);
      packFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
          param.inNorm, idx, face_idx, Ls*ghostFace[3], param);
    } else {
      const int idx = indexFromDWFaceIndex<3,nFace,1>(face_idx,Ls*ghostFace[3],param.parity);
      packFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
          param.inNorm, idx, face_idx, Ls*ghostFace[3], param);
    }
  }
}
#endif

template <typename FloatN, typename Float>
class PackFaceDW : public PackFace<FloatN, Float> {

  private:

    int inputPerSite() const { return 24; } // input is full spinor
    int outputPerSite() const { return 12; } // output is spin projected

  public:
    PackFaceDW(FloatN *faces, const cudaColorSpinorField *in, 
        const int dagger, const int parity)
      : PackFace<FloatN, Float>(faces, in, dagger, parity, 1) { }
    virtual ~PackFaceDW() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_DOMAIN_WALL_DIRAC
      PackParam<FloatN> param = this->prepareParam();
      if (this->dagger) {
        packFaceDWKernel<1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      } else {
        packFaceDWKernel<0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      }
#else
      errorQuda("DW face packing kernel is not built");
#endif  
    }

    void apply_twisted(Float a, Float b, const cudaStream_t &stream) {}
    void apply_twisted_clover(Float a, const cudaStream_t &stream) {}

    long long flops() const { return outputPerSite()*this->threads(); }
};

void packFaceDW(void *ghost_buf, cudaColorSpinorField &in, const int dagger,  
    const int parity, const cudaStream_t &stream) {

  switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceDW<double2, double> pack((double2*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceDW<float4, float> pack((float4*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceDW<short4, float> pack((short4*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
  }  
}

#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
  template <int dagger, typename FloatN>
__global__ void packFaceNdegTMKernel(PackParam<FloatN> param)
{
  const int nFace = 1; // 1 face for Wilson
  const int Nf = 2;

  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (face_idx >= param.threads) return;

  // determine which dimension we are packing
  const int dim = dimFromFaceIndex(face_idx, param);

  // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
  // FIXME these ghostFace constants do not include the Nf dimension
  const int face_num = (face_idx >= nFace*Nf*ghostFace[dim]) ? 1 : 0;
  face_idx -= face_num*nFace*Nf*ghostFace[dim];

  // compute where the output is located
  // compute an index into the local volume from the index into the face
  // read spinor, spin-project, and write half spinor to face
  if (dim == 0) {
    if (face_num == 0) {
      const int idx = indexFromNdegTMFaceIndex<0,nFace,0>(face_idx,Nf*ghostFace[0],param.parity);
      packFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
          param.inNorm, idx, face_idx, Nf*ghostFace[0], param);
    } else {
      const int idx = indexFromNdegTMFaceIndex<0,nFace,1>(face_idx,Nf*ghostFace[0],param.parity);
      packFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
          param.inNorm, idx, face_idx, Nf*ghostFace[0], param);
    }
  } else if (dim == 1) {
    if (face_num == 0) {
      const int idx = indexFromNdegTMFaceIndex<1,nFace,0>(face_idx,Nf*ghostFace[1],param.parity);
      packFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
          param.inNorm, idx, face_idx, Nf*ghostFace[1], param);
    } else {
      const int idx = indexFromNdegTMFaceIndex<1,nFace,1>(face_idx,Nf*ghostFace[1],param.parity);
      packFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
          param.inNorm, idx, face_idx, Nf*ghostFace[1], param);
    }
  } else if (dim == 2) {
    if (face_num == 0) {
      const int idx = indexFromNdegTMFaceIndex<2,nFace,0>(face_idx,Nf*ghostFace[2],param.parity);
      packFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
          param.inNorm, idx, face_idx, Nf*ghostFace[2], param);
    } else {
      const int idx = indexFromNdegTMFaceIndex<2,nFace,1>(face_idx,Nf*ghostFace[2],param.parity);
      packFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
          param.inNorm, idx, face_idx, Nf*ghostFace[2], param);
    }
  } else {
    if (face_num == 0) {
      const int idx = indexFromNdegTMFaceIndex<3,nFace,0>(face_idx,Nf*ghostFace[3],param.parity);
      packFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
          param.inNorm, idx, face_idx, Nf*ghostFace[3], param);
    } else {
      const int idx = indexFromNdegTMFaceIndex<3,nFace,1>(face_idx,Nf*ghostFace[3],param.parity);
      packFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
          param.inNorm, idx, face_idx, Nf*ghostFace[3], param);
    }
  }
}
#endif

template <typename FloatN, typename Float>
class PackFaceNdegTM : public PackFace<FloatN, Float> {

  private:

    int inputPerSite() const { return 24; } // input is full spinor
    int outputPerSite() const { return 12; } // output is spin projected

  public:
    PackFaceNdegTM(FloatN *faces, const cudaColorSpinorField *in, 
        const int dagger, const int parity)
      : PackFace<FloatN, Float>(faces, in, dagger, parity, 1) { }
    virtual ~PackFaceNdegTM() { }

    void apply(const cudaStream_t &stream) {    
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
      PackParam<FloatN> param = this->prepareParam();
      if (this->dagger) {
        packFaceNdegTMKernel<1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      } else {
        packFaceNdegTMKernel<0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      }
#else
      errorQuda("Non-degenerate twisted mass face packing kernel is not built");
#endif  
    }

    void apply_twisted(Float a, Float b, const cudaStream_t &stream) {}
    void apply_twisted_clover(Float a, const cudaStream_t &stream) {}

    long long flops() const { return outputPerSite()*this->threads(); }

};

void packFaceNdegTM(void *ghost_buf, cudaColorSpinorField &in, const int dagger, 
    const int parity, const cudaStream_t &stream) {

  switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceNdegTM<double2, double> pack((double2*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceNdegTM<float4, float> pack((float4*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceNdegTM<short4, float> pack((short4*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
  } 
}

void packFace(void *ghost_buf, cudaColorSpinorField &in, FullClover &clover, FullClover &clovInv,
    const int nFace, const int dagger, const int parity, 
     const int dim, const int face_num, const cudaStream_t &stream, 
    const double a)
{
  int nDimPack = 0;
  if(dim < 0){
    for (int d=0; d<4; d++) {
      if(!dslashParam.commDim[d]) continue;
      if (d != 3 || getKernelPackT() || a != 0.0) nDimPack++;
    }
  }else{
    if(dslashParam.commDim[dim]){
      if(dim!=3 || getKernelPackT() || a!=0.0) nDimPack++;
    }
  }
  if (!nDimPack) return; // if zero then we have nothing to pack 

  if (nFace != 1 && in.Nspin() != 1) 
    errorQuda("Unsupported number of faces %d", nFace);

  // Need to update this logic for other multi-src dslash packing
  packCloverTwistedFaceWilson(ghost_buf, in, clover, clovInv, dagger, parity, a, stream);
}

void packFace(void *ghost_buf, cudaColorSpinorField &in, const int nFace, 
    const int dagger, const int parity, 
    const int dim, const int face_num, 
    const cudaStream_t &stream, 
    const double a, const double b)
{
  int nDimPack = 0;
  if(dim < 0){
    for (int d=0; d<4; d++) {
      if(!dslashParam.commDim[d]) continue;
      if (d != 3 || getKernelPackT() || a != 0.0 || b!= 0.0) nDimPack++;
    }
  }else{
    if(dslashParam.commDim[dim]){
      if(dim!=3 || getKernelPackT() || a!=0.0 || b != 0.0) nDimPack++;
    }
  }
  if (!nDimPack) return; // if zero then we have nothing to pack 

  if (nFace != 1 && in.Nspin() != 1) 
    errorQuda("Unsupported number of faces %d", nFace);

  // Need to update this logic for other multi-src dslash packing
  if (in.Nspin() == 1) {
    packFaceStaggered(ghost_buf, in, nFace, dagger, parity, dim, face_num, stream);
  } else if (a!=0.0 || b!=0.0) {
    // Need to update this logic for other multi-src dslash packing
    if(in.TwistFlavor() == QUDA_TWIST_PLUS || in.TwistFlavor() == QUDA_TWIST_MINUS) {
      packTwistedFaceWilson(ghost_buf, in, dagger, parity, a, b, stream);
    } else {
      errorQuda("Cannot perform twisted packing for the spinor.");
    }
  } else if (in.Ndim() == 5) {
    if(in.TwistFlavor() == QUDA_TWIST_INVALID) {
      packFaceDW(ghost_buf, in, dagger, parity, stream);
    } else {
      packFaceNdegTM(ghost_buf, in, dagger, parity, stream);
    }
  } else {
    packFaceWilson(ghost_buf, in, dagger, parity, stream);
  }
}



void packFaceExtended(void* buffer, cudaColorSpinorField &field, const int nFace, const int R[],
    const int dagger, const int parity, const int dim, const int face_num, 
    const cudaStream_t &stream, const bool unpack)
{
  int nDimPack = 0;
  if(dim < 0){
    for(int d=0; d<4; d++){
      if(R[d]) nDimPack++;
    }
  }else{
    if(R[dim]) nDimPack++;
  }

  if(!nDimPack) return; // if zero then we have nothing to pack
  if(field.Nspin() == 1){
    packFaceExtendedStaggered(buffer, field, nFace, R, dagger, parity, dim, face_num, stream, unpack);
  }else{
    errorQuda("Extended quark field is not supported");
  }


}

#endif // MULTI_GPU

