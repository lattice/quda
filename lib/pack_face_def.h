// The following indexing routines work for arbitrary (including odd) lattice dimensions.
// compute an index into the local volume from an index into the face (used by the face packing routines)

template <int dim, int nLayers>
static inline __device__ int indexFromFaceIndex(int face_idx, const int &face_volume,
						const int &face_num, const int &parity)
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
// G.Shi: the spinor order in ghost region is different between wilson and asqtad, thus different index
//	  computing routine.
template <int dim, int nLayers>
static inline __device__ int indexFromFaceIndexAsqtad(int face_idx, const int &face_volume,
						      const int &face_num, const int &parity)
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

template <int dim, int nLayers>
static inline __device__ int indexFromDWFaceIndex(int face_idx, const int &face_volume,
						const int &face_num, const int &parity)
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
template <int dim, int nLayers>
static inline __device__ int indexFromNdegTMFaceIndex(int face_idx, const int &face_volume,
						const int &face_num, const int &parity)
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
  
  FloatN *out;
  float *outNorm;

  FloatN *in;
  float *inNorm;

  int parity;
#ifdef USE_TEXTURE_OBJECTS
  cudaTextureObject_t inTex;
  cudaTextureObject_t inTexNorm;
#endif
};

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
template <int dim, int dagger>
static inline __device__ void packFaceWilsonCore(double2 *out, float *outNorm, const double2 *in, const float *inNorm,
						 const int &idx, const int &face_idx, const int &face_volume, 
						 const int &face_num, PackParam<double2> &param)
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
template <int dim, int dagger>
static inline __device__ void packFaceWilsonCore(float4 *out, float *outNorm, const float4 *in, const float *inNorm,
						 const int &idx, const int &face_idx, const int &face_volume, 
						 const int &face_num, const PackParam<float4> &param)
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
template <int dim, int dagger>
static inline __device__ void packFaceWilsonCore(short4 *out, float *outNorm, const short4 *in, const float *inNorm,
						 const int &idx, const int &face_idx, const int &face_volume, 
						 const int &face_num, const PackParam<short4> &param)
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


template <int dim, int dagger, typename FloatN>
__global__ void packFaceWilsonKernel(PackParam<FloatN> param)
{
  const int nFace = 1; // 1 face for Wilson
  const int Nint = 12; // output is spin projected
  size_t faceBytes = nFace*ghostFace[dim]*Nint*sizeof(param.out->x);
  if (sizeof(FloatN)==sizeof(short4)) faceBytes += nFace*ghostFace[dim]*sizeof(float);

  int face_volume = ghostFace[dim];
  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (face_idx >= 2*nFace*face_volume) return;

  // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
  const int face_num = (face_idx >= nFace*face_volume) ? 1 : 0;
  face_idx -= face_num*nFace*face_volume;

  // compute an index into the local volume from the index into the face
  const int idx = indexFromFaceIndex<dim, nFace>(face_idx, face_volume, face_num, param.parity);

  FloatN* out = (face_num) ? (FloatN*)((char*)param.out + faceBytes) : param.out;
  float* outNorm = (face_num) ? (float*)((char*)param.outNorm + faceBytes) : param.outNorm;

  // read spinor, spin-project, and write half spinor to face
  packFaceWilsonCore<dim, dagger>(out, outNorm, param.in, param.inNorm, idx, face_idx, 
				  face_volume, face_num, param);
}

#endif // GPU_WILSON_DIRAC || GPU_DOMAIN_WALL_DIRAC



template <typename FloatN>
class PackFace : public Tunable {

 protected:
  FloatN *faces;
  float *facesNorm;
  const cudaColorSpinorField *in;
  const int dim;
  const int dagger;
  const int parity;
  const int nFace;

  int sharedBytesPerThread() const { return 0; }
  int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  
  bool advanceGridDim(TuneParam &param) const { return false; } // Don't tune the grid dimensions.
  bool advanceBlockDim(TuneParam &param) const {
    bool advance = Tunable::advanceBlockDim(param);
    unsigned int threads = in->GhostFace()[dim]*nFace*2; // 2 for forwards and backwards faces
    if (advance) param.grid = dim3( (threads+param.block.x-1) / param.block.x, 1, 1);
    return advance;
  }

 public:
  PackFace(FloatN *faces, float *facesNorm, const cudaColorSpinorField *in, 
	   const int dim, const int dagger, const int parity, const int nFace)
    : faces(faces), facesNorm(facesNorm), in(in), dim(dim), dagger(dagger), parity(parity), nFace(nFace) { }
  virtual ~PackFace() { }
  
  virtual int tuningIter() const { return 100; }

  virtual TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << in->X()[0] << "x";
    vol << in->X()[1] << "x";
    vol << in->X()[2] << "x";
    vol << in->X()[3];    
    aux << "dim=" << dim << ",stride=" << in->Stride() << ",prec=" << sizeof(((FloatN*)0)->x);
    return TuneKey(vol.str(), typeid(*this).name(), aux.str());
  }  
  
  virtual void apply(const cudaStream_t &stream) = 0;

  virtual void initTuneParam(TuneParam &param) const
  {
    Tunable::initTuneParam(param);
    unsigned int threads = in->GhostFace()[dim]*nFace*2; // 2 for forwards and backwards faces
    param.grid = dim3( (threads+param.block.x-1) / param.block.x, 1, 1);
  }
  
  /** sets default values for when tuning is disabled */
  virtual void defaultTuneParam(TuneParam &param) const
  {
    Tunable::defaultTuneParam(param);
    unsigned int threads = in->GhostFace()[dim]*nFace*2; // 2 for forwards and backwards faces
    param.grid = dim3( (threads+param.block.x-1) / param.block.x, 1, 1);
  }

};

template <typename FloatN>
class PackFaceWilson : public PackFace<FloatN> {

 private:

 public:
  PackFaceWilson(FloatN *faces, float *facesNorm, const cudaColorSpinorField *in, 
		 const int dim, const int dagger, const int parity)
    : PackFace<FloatN>(faces, facesNorm, in, dim, dagger, parity, 1) { }
  virtual ~PackFaceWilson() { }
  
  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);

#ifdef GPU_WILSON_DIRAC
    PackParam<FloatN> param;
    param.out = this->faces;
    param.outNorm = this->facesNorm;
    param.in = (FloatN*)this->in->V();
    param.inNorm = (float*)this->in->Norm();
    param.parity = this->parity;
#ifdef USE_TEXTURE_OBJECTS
    param.inTex = this->in->Tex();
    param.inTexNorm = this->in->TexNorm();
#endif

    if (this->dagger) {
      switch (this->dim) {
      case 0: packFaceWilsonKernel<0,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      case 1: packFaceWilsonKernel<1,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      case 2: packFaceWilsonKernel<2,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      case 3: packFaceWilsonKernel<3,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      }
    } else {
      switch (this->dim) {
    case 0: packFaceWilsonKernel<0,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      case 1: packFaceWilsonKernel<1,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      case 2: packFaceWilsonKernel<2,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      case 3: packFaceWilsonKernel<3,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      }
    }
#else
    errorQuda("Wilson face packing kernel is not built");
#endif  
  }

  long long flops() const { return 12*this->in->GhostFace()[this->dim]; }
  long long bytes() const { 
    int Nint = 36; // input and output
    size_t faceBytes = 2*this->nFace*this->in->GhostFace()[this->dim]*Nint*sizeof(((FloatN*)0)->x);
    if (sizeof(((FloatN*)0)->x) == QUDA_HALF_PRECISION) 
      faceBytes += 2*this->nFace*this->in->GhostFace()[this->dim]*sizeof(float);
    return faceBytes;
  }

};

void packFaceWilson(void *ghost_buf, cudaColorSpinorField &in, const int dim, const int dagger, 
		    const int parity, const cudaStream_t &stream) {
  const int nFace = 1; // 1 face for Wilson

  // compute location of norm zone
  int Nint = in.Ncolor() * in.Nspin(); // assume spin projection
  float *ghostNorm = (float*)((char*)ghost_buf + Nint*nFace*in.GhostFace()[dim]*in.Precision());

  switch(in.Precision()) {
  case QUDA_DOUBLE_PRECISION:
    {
      PackFaceWilson<double2> pack((double2*)ghost_buf, ghostNorm, &in, dim, dagger, parity);
      pack.apply(stream);
    }
    break;
  case QUDA_SINGLE_PRECISION:
    {
      PackFaceWilson<float4> pack((float4*)ghost_buf, ghostNorm, &in, dim, dagger, parity);
      pack.apply(stream);
    }
    break;
  case QUDA_HALF_PRECISION:
    {
      PackFaceWilson<short4> pack((short4*)ghost_buf, ghostNorm, &in, dim, dagger, parity);
      pack.apply(stream);
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
__device__ void packSpinor(Float2 *out, float *outNorm, int out_idx, int out_stride, 
			   const Float2 *in, const float *inNorm, int in_idx, int in_stride) {
  out[out_idx + 0*out_stride] = in[in_idx + 0*in_stride];
  out[out_idx + 1*out_stride] = in[in_idx + 1*in_stride];
  out[out_idx + 2*out_stride] = in[in_idx + 2*in_stride];
}	
template<> __device__ void packSpinor(short2 *out, float *outNorm, int out_idx, int out_stride, 
				      const short2 *in, const float *inNorm, int in_idx, int in_stride) {
  out[out_idx + 0*out_stride] = in[in_idx + 0*in_stride];
  out[out_idx + 1*out_stride] = in[in_idx + 1*in_stride];
  out[out_idx + 2*out_stride] = in[in_idx + 2*in_stride];
  outNorm[out_idx] = inNorm[in_idx];
}
#else
__device__ void packSpinor(double2 *out, float *outNorm, int out_idx, int out_stride, 
			   const double2 *in, const float *inNorm, int in_idx, int in_stride, const PackParam<double2> &param) {
  out[out_idx + 0*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 0*in_stride);
  out[out_idx + 1*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 1*in_stride);
  out[out_idx + 2*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 2*in_stride);
}	
__device__ void packSpinor(float2 *out, float *outNorm, int out_idx, int out_stride, 
			   const float2 *in, const float *inNorm, int in_idx, int in_stride, const PackParam<float2> &param) {
  out[out_idx + 0*out_stride] = TEX1DFETCH(float2, SPINORTEXSINGLE, in_idx + 0*in_stride);
  out[out_idx + 1*out_stride] = TEX1DFETCH(float2, SPINORTEXSINGLE, in_idx + 1*in_stride);
  out[out_idx + 2*out_stride] = TEX1DFETCH(float2, SPINORTEXSINGLE, in_idx + 2*in_stride);	
}

// this is rather dumb: undoing the texture load because cudaNormalizedReadMode is used
// should really bind to an appropriate texture instead of reusing
static inline __device__ short2 float22short2(float c, float2 a) {
  return make_short2((short)(a.x*c*MAX_SHORT), (short)(a.y*c*MAX_SHORT));
}

__device__ void packSpinor(short2 *out, float *outNorm, int out_idx, int out_stride, 
			   const short2 *in, const float *inNorm, int in_idx, int in_stride, const PackParam<short2> &param) {
  out[out_idx + 0*out_stride] = float22short2(1.0f,TEX1DFETCH(float2,SPINORTEXHALF,in_idx+0*in_stride));
  out[out_idx + 1*out_stride] = float22short2(1.0f,TEX1DFETCH(float2,SPINORTEXHALF,in_idx+1*in_stride));
  out[out_idx + 2*out_stride] = float22short2(1.0f,TEX1DFETCH(float2,SPINORTEXHALF,in_idx+2*in_stride));
  outNorm[out_idx] = TEX1DFETCH(float, SPINORTEXHALFNORM, in_idx);
}
#endif

template <int dim, int ishalf, typename FloatN>
__global__ void packFaceAsqtadKernel(const PackParam<FloatN> param)
{
  const int nFace = 3; //3 faces for asqtad
  const int Nint = 6; // number of internal degrees of freedom
  size_t faceBytes = nFace*ghostFace[dim]*Nint*sizeof(param.out->x);
  if (ishalf) faceBytes += nFace*ghostFace[dim]*sizeof(float);

  int face_volume = ghostFace[dim];
  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (face_idx >= 2*nFace*face_volume) return;

  // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
  const int face_num = (face_idx >= nFace*face_volume) ? 1 : 0;
  face_idx -= face_num*nFace*face_volume;

  // compute an index into the local volume from the index into the face
  const int idx = indexFromFaceIndexAsqtad<dim, nFace>(face_idx, face_volume, face_num, param.parity);
  
  FloatN* out = (face_num) ? (FloatN*)((char*)param.out + faceBytes) : param.out;
  float* outNorm = (face_num) ? (float*)((char*)param.outNorm + faceBytes) : param.outNorm;

  packSpinor(out, outNorm, face_idx, nFace*face_volume, param.in, param.inNorm, idx, sp_stride, param);

}

#undef SPINORTEXDOUBLE
#undef SPINORTEXSINGLE
#undef SPINORTEXHALF

#endif // GPU_STAGGERED_DIRAC


template <typename FloatN>
class PackFaceAsqtad : public PackFace<FloatN> {

 private:

 public:
  PackFaceAsqtad(FloatN *faces, float *facesNorm, const cudaColorSpinorField *in, 
		 const int dim, const int dagger, const int parity)
    : PackFace<FloatN>(faces, facesNorm, in, dim, dagger, parity, 3) { }
  virtual ~PackFaceAsqtad() { }
  
  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
    
#ifdef GPU_STAGGERED_DIRAC
    PackParam<FloatN> param;
    param.out = this->faces;
    param.outNorm = this->facesNorm;
    param.in = (FloatN*)this->in->V();
    param.inNorm = (float*)this->in->Norm();
    param.parity = this->parity;
#ifdef USE_TEXTURE_OBJECTS
    param.inTex = this->in->Tex();
    param.inTexNorm = this->in->TexNorm();
#endif
    
    if(typeid(FloatN) != typeid(short2)){
      switch (this->dim) {
      case 0: packFaceAsqtadKernel<0,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      case 1: packFaceAsqtadKernel<1,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      case 2: packFaceAsqtadKernel<2,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      case 3: packFaceAsqtadKernel<3,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      }
    }else{
      switch(this->dim){
      case 0: packFaceAsqtadKernel<0,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      case 1: packFaceAsqtadKernel<1,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      case 2: packFaceAsqtadKernel<2,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      case 3: packFaceAsqtadKernel<3,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
      }
    }
#else
    errorQuda("Asqtad face packing kernel is not built");
#endif  
  }

  long long flops() const { return 0; }
  long long bytes() const { 
    int Nint = 12; // input and output
    size_t faceBytes = 2*this->nFace*this->in->GhostFace()[this->dim]*Nint*sizeof(((FloatN*)0)->x);
    if (sizeof(((FloatN*)0)->x) == QUDA_HALF_PRECISION) 
      faceBytes += 2*this->nFace*this->in->GhostFace()[this->dim]*sizeof(float);
    return faceBytes;
  }

};

void packFaceAsqtad(void *ghost_buf, cudaColorSpinorField &in, const int dim, const int dagger, 
		    const int parity, const cudaStream_t &stream) {
  const int nFace = 3; //3 faces for asqtad

  // compute location of norm zone
  int Nint = 6;
  float *ghostNorm = (float*)((char*)ghost_buf + Nint*nFace*in.GhostFace()[dim]*in.Precision());

  switch(in.Precision()) {
  case QUDA_DOUBLE_PRECISION:
    {
      PackFaceAsqtad<double2> pack((double2*)ghost_buf, ghostNorm, &in, dim, dagger, parity);
      pack.apply(stream);
    }
    break;
  case QUDA_SINGLE_PRECISION:
    {
      PackFaceAsqtad<float2> pack((float2*)ghost_buf, ghostNorm, &in, dim, dagger, parity);
      pack.apply(stream);
    }
    break;
  case QUDA_HALF_PRECISION:
    {
      PackFaceAsqtad<short2> pack((short2*)ghost_buf, ghostNorm, &in, dim, dagger, parity);
      pack.apply(stream);
    }
    break;
  }  

}

#ifdef GPU_DOMAIN_WALL_DIRAC
template <int dim, int dagger, typename FloatN>
__global__ void packFaceDWKernel(PackParam<FloatN> param)
{
  const int nFace = 1; // 1 face for Wilson
  const int Nint = 12; // output is spin projected
  size_t faceBytes = nFace*Ls*ghostFace[dim]*Nint*sizeof(param.out->x);
  if (sizeof(FloatN)==sizeof(short4)) faceBytes += nFace*Ls*ghostFace[dim]*sizeof(float);

  int face_volume = Ls*ghostFace[dim];
  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (face_idx >= 2*nFace*face_volume) return;

  // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
  const int face_num = (face_idx >= nFace*face_volume) ? 1 : 0;
  face_idx -= face_num*nFace*face_volume;

  // compute an index into the local volume from the index into the face
  int idx = indexFromDWFaceIndex<dim, 1>(face_idx, face_volume, face_num, param.parity);

  FloatN* out = (face_num) ? (FloatN*)((char*)param.out + faceBytes) : param.out;
  float* outNorm = (face_num) ? (float*)((char*)param.outNorm + faceBytes) : param.outNorm;
  
  // read spinor, spin-project, and write half spinor to face (the same kernel as for Wilson): 
  packFaceWilsonCore<dim, dagger>(out, outNorm, param.in, param.inNorm, idx, face_idx, 
				  face_volume, face_num, param);
}
#endif

template <typename FloatN>
class PackFaceDW : public PackFace<FloatN> {

 private:

 public:
  PackFaceDW(FloatN *faces, float *facesNorm, const cudaColorSpinorField *in, 
		 const int dim, const int dagger, const int parity)
    : PackFace<FloatN>(faces, facesNorm, in, dim, dagger, parity, 1) { }
  virtual ~PackFaceDW() { }
  
  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
    
#ifdef GPU_DOMAIN_WALL_DIRAC
    PackParam<FloatN> param;
    param.out = this->faces;
    param.outNorm = this->facesNorm;
    param.in = (FloatN*)this->in->V();
    param.inNorm = (float*)this->in->Norm();
    param.parity = this->parity;
#ifdef USE_TEXTURE_OBJECTS
    param.inTex = this->in->Tex();
    param.inTexNorm = this->in->TexNorm();
#endif
    
  if (this->dagger) {
    switch (this->dim) {
    case 0: packFaceDWKernel<0,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
    case 1: packFaceDWKernel<1,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;    
    case 2: packFaceDWKernel<2,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;	
    case 3: packFaceDWKernel<3,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;    }
  } else {
    switch (this->dim) {
    case 0: packFaceDWKernel<0,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
    case 1: packFaceDWKernel<1,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
    case 2: packFaceDWKernel<2,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
    case 3: packFaceDWKernel<3,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param); break;
    }
  }
#else
    errorQuda("DW face packing kernel is not built");
#endif  
  }

  long long flops() const { return 12*this->in->GhostFace()[this->dim]; }

  long long bytes() const { 
    int Nint = 36; // input and output
    size_t faceBytes = 2*this->nFace*this->in->GhostFace()[this->dim]*Nint*sizeof(((FloatN*)0)->x);
    if (sizeof(((FloatN*)0)->x) == QUDA_HALF_PRECISION) 
      faceBytes += 2*this->nFace*this->in->GhostFace()[this->dim]*sizeof(float);
    return faceBytes;
  }

};

void packFaceDW(void *ghost_buf, cudaColorSpinorField &in, const int dim, const int dagger,  
		const int parity, const cudaStream_t &stream) {
  const int nFace = 1; // 1 face for Wilson
  int Nint = in.Ncolor() * in.Nspin(); // assume spin projection
  float *ghostNorm = (float*)((char*)ghost_buf + Nint*nFace*in.GhostFace()[dim]*in.Precision()); // norm zone

  switch(in.Precision()) {
  case QUDA_DOUBLE_PRECISION:
    {
      PackFaceDW<double2> pack((double2*)ghost_buf, ghostNorm, &in, dim, dagger, parity);
      pack.apply(stream);
    }
    break;
  case QUDA_SINGLE_PRECISION:
    {
      PackFaceDW<float4> pack((float4*)ghost_buf, ghostNorm, &in, dim, dagger, parity);
      pack.apply(stream);
    }
    break;
  case QUDA_HALF_PRECISION:
    {
      PackFaceDW<short4> pack((short4*)ghost_buf, ghostNorm, &in, dim, dagger, parity);
      pack.apply(stream);
    }
    break;
  }  
}

void packFace(void *ghost_buf, cudaColorSpinorField &in, const int dim, const int dagger, 
	      const int parity, const cudaStream_t &stream)
{
  if (in.Nspin() == 1) {
    packFaceAsqtad(ghost_buf, in, dim, dagger, parity, stream);
  } else {  
    packFaceWilson(ghost_buf, in, dim, dagger, parity, stream);
  }
}


#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
template <int dim, int dagger, typename FloatN>
__global__ void packFaceNdegTMKernel(PackParam<FloatN> param)
{
  const int Nf = 2;
  const int nFace = 1; // 1 face for Wilson
  const int Nint = 12; // output is spin projected
  size_t faceBytes = nFace*Nf*ghostFace[dim]*Nint*sizeof(param.out->x);
  if (sizeof(FloatN)==sizeof(short4)) faceBytes += nFace*Nf*ghostFace[dim]*sizeof(float);

  int face_volume = Nf*ghostFace[dim];
  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (face_idx >= 2*nFace*face_volume) return;

  // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
  const int face_num = (face_idx >= nFace*face_volume) ? 1 : 0;
  face_idx -= face_num*nFace*face_volume;

  // compute an index into the local volume from the index into the face
  int idx = indexFromNdegTMFaceIndex<dim, 1>(face_idx, face_volume, face_num, param.parity);

  FloatN* out = (face_num) ? (FloatN*)((char*)param.out + faceBytes) : param.out;
  float* outNorm = (face_num) ? (float*)((char*)param.outNorm + faceBytes) : param.outNorm;
  
  // read spinor, spin-project, and write half spinor to face (the same kernel as for Wilson): 
  packFaceWilsonCore<dim, dagger>(out, outNorm, param.in, param.inNorm, idx, face_idx, face_volume, face_num, param);

}
#endif

template <typename FloatN>
class PackFaceNdegTM : public PackFace<FloatN> {

 private:

 public:
  PackFaceNdegTM(FloatN *faces, float *facesNorm, const cudaColorSpinorField *in, 
		 const int dim, const int dagger, const int parity)
    : PackFace<FloatN>(faces, facesNorm, in, dim, dagger, parity, 1) { }
  virtual ~PackFaceNdegTM() { }
  
  void apply(const cudaStream_t &stream) {
    
    //TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
    unsigned int threads = this->in->GhostFace()[this->dim]*this->nFace*2;//WARNING: this corresponds to a flavor doublet!
    dim3 blockDim(64, 1, 1); // TODO: make this a parameter for auto-tuning
    dim3 gridDim( (threads+blockDim.x-1) / blockDim.x, 1, 1);
    
#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
    PackParam<FloatN> param;
    param.out = this->faces;
    param.outNorm = this->facesNorm;
    param.in = (FloatN*)this->in->V();
    param.inNorm = (float*)this->in->Norm();
    param.parity = this->parity;
#ifdef USE_TEXTURE_OBJECTS
    param.inTex = this->in->Tex();
    param.inTexNorm = this->in->TexNorm();
#endif
    
  if (this->dagger) {
    switch (this->dim) {
    case 0:  packFaceNdegTMKernel<0,1><<<gridDim, blockDim, 0, stream>>>(param); break;
    case 1:  packFaceNdegTMKernel<1,1><<<gridDim, blockDim, 0, stream>>>(param); break;    
    case 2:  packFaceNdegTMKernel<2,1><<<gridDim, blockDim, 0, stream>>>(param); break;	
    case 3:  packFaceNdegTMKernel<3,1><<<gridDim, blockDim, 0, stream>>>(param); break;    }
  } else {
    switch (this->dim) {
    case 0:  packFaceNdegTMKernel<0,0><<<gridDim, blockDim, 0, stream>>>(param); break;
    case 1:  packFaceNdegTMKernel<1,0><<<gridDim, blockDim, 0, stream>>>(param); break;
    case 2:  packFaceNdegTMKernel<2,0><<<gridDim, blockDim, 0, stream>>>(param); break;
    case 3:  packFaceNdegTMKernel<3,0><<<gridDim, blockDim, 0, stream>>>(param); break;
    }
  }
#else
    errorQuda("Non-degenerate twisted mass face packing kernel is not built");
#endif  
  }

  long long flops() const { return 12*this->in->GhostFace()[this->dim]; }

  long long bytes() const { 
    int Nint = 36; // input and output
    size_t faceBytes = 2*this->nFace*this->in->GhostFace()[this->dim]*Nint*sizeof(((FloatN*)0)->x);
    if (sizeof(((FloatN*)0)->x) == QUDA_HALF_PRECISION) 
      faceBytes += 2*this->nFace*this->in->GhostFace()[this->dim]*sizeof(float);
    return faceBytes;
  }

};

void packFaceNdegTM(void *ghost_buf, cudaColorSpinorField &in, const int dim, const int dagger, 
		    const int parity, const cudaStream_t &stream) {
  const int nFace = 1; // 1 face for Wilson
  // compute location of norm zone
  int Nint = in.Ncolor() * in.Nspin(); // assume spin projection
  float *ghostNorm = (float*)((char*)ghost_buf + Nint*nFace*in.GhostFace()[dim]*in.Precision());

  switch(in.Precision()) {
  case QUDA_DOUBLE_PRECISION:
    {
      PackFaceNdegTM<double2> pack((double2*)ghost_buf, ghostNorm, &in, dim, dagger, parity);
      pack.apply(stream);
    }
    break;
  case QUDA_SINGLE_PRECISION:
    {
      PackFaceNdegTM<float4> pack((float4*)ghost_buf, ghostNorm, &in, dim, dagger, parity);
      pack.apply(stream);
    }
    break;
  case QUDA_HALF_PRECISION:
    {
      PackFaceNdegTM<short4> pack((short4*)ghost_buf, ghostNorm, &in, dim, dagger, parity);
      pack.apply(stream);
    }
    break;
  } 
}


#endif // MULTI_GPU
