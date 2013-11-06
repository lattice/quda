// The following indexing routines work for arbitrary (including odd) lattice dimensions.
// compute an index into the local volume from an index into the face (used by the face packing routines)

  template <int dim, int nLayers, int face_num>
static inline __device__ int indexFromFaceIndex(int face_idx, const int &face_volume, const int &parity)
{
  // dimensions of the face (FIXME: optimize using constant cache)

  int face_X = X1, face_Y = X2, face_Z = X3; // face_T = X4;
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

  int face_parity;
  switch (dim) {
    case 0:
      face_parity = (parity + face_num * (X1 - nLayers)) & 1;
      break;
    case 1:
      face_parity = (parity + face_num * (X2 - nLayers)) & 1;
      break;
    case 2:
      face_parity = (parity + face_num * (X3 - nLayers)) & 1;
      break;
    case 3:
      face_parity = (parity + face_num * (X4 - nLayers)) & 1;
      break;
  }

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
    const int &parity)
{
  // dimensions of the face (FIXME: optimize using constant cache)
  int dims[3];
  int V = 2*Vh;
  int face_X = X1, face_Y = X2, face_Z = X3; // face_T = X4;
  switch (dim) {
    case 0:
      face_X = nLayers;
      dims[0]=X2; dims[1]=X3; dims[2]=X4;
      break;
    case 1:
      face_Y = nLayers;
      dims[0]=X1;dims[1]=X3; dims[2]=X4;
      break;
    case 2:
      face_Z = nLayers;
      dims[0]=X1;dims[1]=X2; dims[2]=X4;
      break;
    case 3:
      // face_T = nLayers;
      dims[0]=X1;dims[1]=X2; dims[2]=X4;
      break;
  }
  int face_XYZ = face_X * face_Y * face_Z;
  int face_XY = face_X * face_Y;

  // intrinsic parity of the face depends on offset of first element

  int face_parity;
  switch (dim) {
    case 0:
      face_parity = (parity + face_num * (X1 - nLayers)) & 1;
      break;
    case 1:
      face_parity = (parity + face_num * (X2 - nLayers)) & 1;
      break;
    case 2:
      face_parity = (parity + face_num * (X3 - nLayers)) & 1;
      break;
    case 3:
      face_parity = (parity + face_num * (X4 - nLayers)) & 1;
      break;
  }


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
  int gap, aux;

  switch (dim) {
    case 0:
      gap = X1 - nLayers;
      aux = face_idx;
      idx += face_num*gap + aux*(X1-1);
      idx += idx/V*(1-V);    
      break;
    case 1:
      gap = X2 - nLayers;
      aux = face_idx / face_X;
      idx += face_num * gap * face_X + aux*(X2-1)*face_X;
      idx += idx/V*(X1-V);
      break;
    case 2:
      gap = X3 - nLayers;
      aux = face_idx / face_XY;    
      idx += face_num * gap * face_XY +aux*(X3-1)*face_XY;
      idx += idx/V*(X2X1-V);
      break;
    case 3:
      gap = X4 - nLayers;
      idx += face_num * gap * face_XYZ;
      break;
  }

  // return index into the checkerboard

  return idx >> 1;
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

  int threads; // total number of threads

  // offsets which determine thread mapping to dimension
  int threadDimMapLower[QUDA_MAX_DIM]; // lowest thread which maps to dim
  int threadDimMapUpper[QUDA_MAX_DIM]; // greatest thread + 1 which maps to dim

  int parity;
#ifdef USE_TEXTURE_OBJECTS
  cudaTextureObject_t inTex;
  cudaTextureObject_t inTexNorm;
#endif

  int dim;
  int face_num;


  int stride;
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
      const int idx = indexFromFaceIndex<0,nFace,0>(face_idx,ghostFace[0],param.parity);
      packFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
          param.inNorm,idx, face_idx, ghostFace[0], param);
    } else {
      const int idx = indexFromFaceIndex<0,nFace,1>(face_idx,ghostFace[0],param.parity);
      packFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
          param.inNorm,idx, face_idx, ghostFace[0], param);
    }
  } else if (dim == 1) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<1,nFace,0>(face_idx,ghostFace[1],param.parity);
      packFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
          param.inNorm,idx, face_idx, ghostFace[1], param);
    } else {
      const int idx = indexFromFaceIndex<1,nFace,1>(face_idx,ghostFace[1],param.parity);
      packFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
          param.inNorm,idx, face_idx, ghostFace[1], param);
    }
  } else if (dim == 2) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<2,nFace,0>(face_idx,ghostFace[2],param.parity);
      packFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
          param.inNorm,idx, face_idx, ghostFace[2], param);
    } else {
      const int idx = indexFromFaceIndex<2,nFace,1>(face_idx,ghostFace[2],param.parity);
      packFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
          param.inNorm,idx, face_idx, ghostFace[2], param);
    }
  } else {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<3,nFace,0>(face_idx,ghostFace[3],param.parity);
      packFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
          param.inNorm,idx, face_idx, ghostFace[3], param);
    } else {
      const int idx = indexFromFaceIndex<3,nFace,1>(face_idx,ghostFace[3],param.parity);
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
      const int idx = indexFromFaceIndex<0,nFace,0>(face_idx,ghostFace[0],param.parity);
      packTwistedFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[0], param);
    } else {
      const int idx = indexFromFaceIndex<0,nFace,1>(face_idx,ghostFace[0],param.parity);
      packTwistedFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[0], param);
    }
  } else if (dim == 1) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<1,nFace,0>(face_idx,ghostFace[1],param.parity);
      packTwistedFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[1], param);
    } else {
      const int idx = indexFromFaceIndex<1,nFace,1>(face_idx,ghostFace[1],param.parity);
      packTwistedFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[1], param);
    }
  } else if (dim == 2) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<2,nFace,0>(face_idx,ghostFace[2],param.parity);
      packTwistedFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[2], param);
    } else {
      const int idx = indexFromFaceIndex<2,nFace,1>(face_idx,ghostFace[2],param.parity);
      packTwistedFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[2], param);
    }
  } else {
    if (face_num == 0) {
      const int idx = indexFromFaceIndex<3,nFace,0>(face_idx,ghostFace[3],param.parity);
      packTwistedFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
          param.inNorm, a, b,idx, face_idx, ghostFace[3], param);
    } else {
      const int idx = indexFromFaceIndex<3,nFace,1>(face_idx,ghostFace[3],param.parity);
      packTwistedFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
          param.inNorm, a, b, idx, face_idx, ghostFace[3], param);
    }
  }

}

#endif // GPU_TWISTED_MASS_DIRAC



template <typename FloatN, typename Float>
class PackFace : public Tunable {

  protected:
    FloatN *faces;
    const cudaColorSpinorField *in;
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
      param.parity = parity;
      param.dim = dim;
      param.parity = parity;
#ifdef USE_TEXTURE_OBJECTS
      param.inTex = in->Tex();
      param.inTexNorm = in->TexNorm();
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
      : faces(faces), in(in), dagger(dagger), parity(parity), nFace(nFace), dim(dim), face_num(face_num) { }
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
__device__ void packFaceStaggeredCore(double2 *out, float *outNorm, const int out_idx, 
    const int out_stride, const double2 *in, const float *inNorm, 
    const int in_idx, const PackParam<double2> &param) {
  out[out_idx + 0*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 0*param.stride);
  out[out_idx + 1*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 1*param.stride);
  out[out_idx + 2*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 2*param.stride);
}	
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
  const int face_num = (face_idx >= nFace*ghostFace[dim]) ? 1 : 0;
  face_idx -= face_num*nFace*ghostFace[dim];

  // compute where the output is located
  // compute an index into the local volume from the index into the face
  // read spinor, spin-project, and write half spinor to face
  if (dim == 0) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexStaggered<0,nFace,0>(face_idx,ghostFace[0],param.parity);
      packFaceStaggeredCore(param.out[0], param.outNorm[0], face_idx, 
          nFace*ghostFace[0], param.in, param.inNorm, idx, param);
    } else {
      const int idx = indexFromFaceIndexStaggered<0,nFace,1>(face_idx,ghostFace[0],param.parity);
      packFaceStaggeredCore(param.out[1], param.outNorm[1], face_idx,
          nFace*ghostFace[0], param.in, param.inNorm, idx, param);
    }
  } else if (dim == 1) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexStaggered<1,nFace,0>(face_idx,ghostFace[1],param.parity);
      packFaceStaggeredCore(param.out[2], param.outNorm[2], face_idx, 
          nFace*ghostFace[1], param.in, param.inNorm, idx, param);
    } else {
      const int idx = indexFromFaceIndexStaggered<1,nFace,1>(face_idx,ghostFace[1],param.parity);
      packFaceStaggeredCore(param.out[3], param.outNorm[3], face_idx, 
          nFace*ghostFace[1], param.in, param.inNorm, idx, param);
    }
  } else if (dim == 2) {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexStaggered<2,nFace,0>(face_idx,ghostFace[2],param.parity);
      packFaceStaggeredCore(param.out[4], param.outNorm[4], face_idx,
          nFace*ghostFace[2], param.in, param.inNorm, idx, param);
    } else {
      const int idx = indexFromFaceIndexStaggered<2,nFace,1>(face_idx,ghostFace[2],param.parity);
      packFaceStaggeredCore(param.out[5], param.outNorm[5], face_idx,
          nFace*ghostFace[2], param.in, param.inNorm, idx, param);
    }
  } else {
    if (face_num == 0) {
      const int idx = indexFromFaceIndexStaggered<3,nFace,0>(face_idx,ghostFace[3],param.parity);
      packFaceStaggeredCore(param.out[6], param.outNorm[6], face_idx,
          nFace*ghostFace[3], param.in, param.inNorm,idx, param);
    } else {
      const int idx = indexFromFaceIndexStaggered<3,nFace,1>(face_idx,ghostFace[3],param.parity);
      packFaceStaggeredCore(param.out[7], param.outNorm[7], face_idx, 
          nFace*ghostFace[3], param.in, param.inNorm, idx, param);
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

    int inputPerSite() const { return 6; } // input is full spinor
    int outputPerSite() const { return 6; } // output is full spinor

  public:
    PackFaceStaggered(FloatN *faces, const cudaColorSpinorField *in, 
        const int nFace, const int dagger, const int parity, 
        const int dim, const int face_num)
      : PackFace<FloatN, Float>(faces, in, dagger, parity, nFace, dim, face_num) { }
    virtual ~PackFaceStaggered() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_STAGGERED_DIRAC
      PackParam<FloatN> param = this->prepareParam(this->dim, this->face_num);
      if (PackFace<FloatN,Float>::nFace==1) {
        packFaceStaggeredKernel<FloatN, 1> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      } else {
        packFaceStaggeredKernel<FloatN, 3> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      }
#else
      errorQuda("Staggered face packing kernel is not built");
#endif  
    }

    void apply_twisted(Float a, Float b, const cudaStream_t &stream) {}

    long long flops() const { return 0; }
};

void packFaceStaggered(void *ghost_buf, cudaColorSpinorField &in, int nFace, 
    int dagger, int parity, const int dim, const int face_num, const cudaStream_t &stream) {

  switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceStaggered<double2, double> pack((double2*)ghost_buf, &in, nFace, dagger, parity, dim, face_num);
        pack.apply(stream);
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

#endif // MULTI_GPU

