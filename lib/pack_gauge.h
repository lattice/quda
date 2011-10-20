// Routines used to pack the gauge field matrices

#include <math.h>

#define SHORT_LENGTH 65536
#define SCALE_FLOAT ((SHORT_LENGTH-1) / 2.f)
#define SHIFT_FLOAT (-1.f / (SHORT_LENGTH-1))

template <typename Float>
inline short FloatToShort(const Float &a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

template <typename Float>
inline void ShortToFloat(Float &a, const short &b) {
  a = ((Float)b/SCALE_FLOAT-SHIFT_FLOAT);
}

/*template <int N, typename FloatN, typename Float>
inline void pack8(FloatN *res, Float *g, int dir, int V) {
  Float *r = res + N*dir*4*V;
  r[0] = atan2(g[1], g[0]);
  r[1] = atan2(g[13], g[12]);
}*/

template <typename Float>
inline void pack8(double2 *res, Float *g, int dir, int V) {
  double2 *r = res + dir*4*V;
  r[0].x = atan2(g[1], g[0]);
  r[0].y = atan2(g[13], g[12]);
  for (int j=1; j<4; j++) {
    r[j*V].x = g[2*j+0];
    r[j*V].y = g[2*j+1];
  }
}

template <typename Float>
inline void pack8(float4 *res, Float *g, int dir, int V) {
  float4 *r = res + dir*2*V;
  r[0].x = atan2(g[1], g[0]);
  r[0].y = atan2(g[13], g[12]);
  r[0].z = g[2];
  r[0].w = g[3];
  r[V].x = g[4];
  r[V].y = g[5];
  r[V].z = g[6];
  r[V].w = g[7];
}

template <typename Float>
inline void pack8(float2 *res, Float *g, int dir, int V) {
  float2 *r = res + dir*4*V;
  r[0].x = atan2(g[1], g[0]);
  r[0].y = atan2(g[13], g[12]);
  for (int j=1; j<4; j++) {
    r[j*V].x = g[2*j+0];
    r[j*V].y = g[2*j+1];
  }
}

template <typename Float>
inline void pack8(short4 *res, Float *g, int dir, int V) {
  short4 *r = res + dir*2*V;
  r[0].x = FloatToShort(atan2(g[1], g[0]) / M_PI);
  r[0].y = FloatToShort(atan2(g[13], g[12]) / M_PI);
  r[0].z = FloatToShort(g[2]);
  r[0].w = FloatToShort(g[3]);
  r[V].x = FloatToShort(g[4]);
  r[V].y = FloatToShort(g[5]);
  r[V].z = FloatToShort(g[6]);
  r[V].w = FloatToShort(g[7]);
}

template <typename Float>
inline void pack8(short2 *res, Float *g, int dir, int V) {
  short2 *r = res + dir*4*V;
  r[0].x = FloatToShort(atan2(g[1], g[0]) / M_PI);
  r[0].y = FloatToShort(atan2(g[13], g[12]) / M_PI);
  for (int j=1; j<4; j++) {
    r[j*V].x = FloatToShort(g[2*j+0]);
    r[j*V].y = FloatToShort(g[2*j+1]);
  }
}

template <typename Float>
inline void pack12(double2 *res, Float *g, int dir, int V) {
  double2 *r = res + dir*6*V;
  for (int j=0; j<6; j++) {
    r[j*V].x = g[j*2+0];
    r[j*V].y = g[j*2+1];
  }
}

template <typename Float>
inline void pack12(float4 *res, Float *g, int dir, int V) {
  float4 *r = res + dir*3*V;
  for (int j=0; j<3; j++) {
    r[j*V].x = g[j*4+0]; 
    r[j*V].y = g[j*4+1];
    r[j*V].z = g[j*4+2]; 
    r[j*V].w = g[j*4+3];
  }
}

template <typename Float>
inline void pack12(float2 *res, Float *g, int dir, int V) {
  float2 *r = res + dir*6*V;
  for (int j=0; j<6; j++) {
    r[j*V].x = g[j*2+0];
    r[j*V].y = g[j*2+1];
  }
}

template <typename Float>
inline void pack12(short4 *res, Float *g, int dir, int V) {
  short4 *r = res + dir*3*V;
  for (int j=0; j<3; j++) {
    r[j*V].x = FloatToShort(g[j*4+0]); 
    r[j*V].y = FloatToShort(g[j*4+1]);
    r[j*V].z = FloatToShort(g[j*4+2]);
    r[j*V].w = FloatToShort(g[j*4+3]);
  }
}

template <typename Float>
inline void pack12(short2 *res, Float *g, int dir, int V) {
  short2 *r = res + dir*6*V;
  for (int j=0; j<6; j++) {
    r[j*V].x = FloatToShort(g[j*2+0]);
    r[j*V].y = FloatToShort(g[j*2+1]);
  }
}

template <typename Float>
inline void pack18(double2 *res, Float *g, int dir, int V) {
  double2 *r = res + dir*9*V;
  for (int j=0; j<9; j++) {
    r[j*V].x = g[j*2+0]; 
    r[j*V].y = g[j*2+1]; 
  }
}

template <typename Float>
inline void pack18(float4 *res, Float *g, int dir, int V) {
  float4 *r = res + dir*5*V;
  for (int j=0; j<4; j++) {
    r[j*V].x = g[j*4+0]; 
    r[j*V].y = g[j*4+1]; 
    r[j*V].z = g[j*4+2]; 
    r[j*V].w = g[j*4+3]; 
  }
  r[4*V].x = g[16]; 
  r[4*V].y = g[17]; 
  r[4*V].z = 0.0;
  r[4*V].w = 0.0;
}

template <typename Float>
inline void pack18(float2 *res, Float *g, int dir, int V) {
  float2 *r = res + dir*9*V;
  for (int j=0; j<9; j++) {
    r[j*V].x = g[j*2+0]; 
    r[j*V].y = g[j*2+1]; 
  }
}

template <typename Float>
inline void pack18(short4 *res, Float *g, int dir, int V) {
  short4 *r = res + dir*5*V;
  for (int j=0; j<4; j++) {
    r[j*V].x = FloatToShort(g[j*4+0]); 
    r[j*V].y = FloatToShort(g[j*4+1]); 
    r[j*V].z = FloatToShort(g[j*4+2]); 
    r[j*V].w = FloatToShort(g[j*4+3]); 
  }
  r[4*V].x = FloatToShort(g[16]); 
  r[4*V].y = FloatToShort(g[17]); 
  r[4*V].z = (short)0;
  r[4*V].w = (short)0;
}

template <typename Float>
inline void pack18(short2 *res, Float *g, int dir, int V) 
{
  short2 *r = res + dir*9*V;
  for (int j=0; j<9; j++) {
    r[j*V].x = FloatToShort(g[j*2+0]); 
    r[j*V].y = FloatToShort(g[j*2+1]); 
  }
}

template<typename Float>
inline void fatlink_short_pack18(short2 *d_gauge, Float *h_gauge, int dir, int V) 
{
  short2 *dg = d_gauge + dir*9*V;
  for (int j=0; j<9; j++) {
    dg[j*V].x = FloatToShort((h_gauge[j*2+0]/fat_link_max_)); 
    dg[j*V].y = FloatToShort((h_gauge[j*2+1]/fat_link_max_)); 
  }
}



// a += b*c
template <typename Float>
inline void accumulateComplexProduct(Float *a, Float *b, Float *c, Float sign) {
  a[0] += sign*(b[0]*c[0] - b[1]*c[1]);
  a[1] += sign*(b[0]*c[1] + b[1]*c[0]);
}

// a = conj(b)*c
template <typename Float>
inline void complexDotProduct(Float *a, Float *b, Float *c) {
    a[0] = b[0]*c[0] + b[1]*c[1];
    a[1] = b[0]*c[1] - b[1]*c[0];
}

// a += conj(b) * conj(c)
template <typename Float>
inline void accumulateConjugateProduct(Float *a, Float *b, Float *c, int sign) {
  a[0] += sign * (b[0]*c[0] - b[1]*c[1]);
  a[1] -= sign * (b[0]*c[1] + b[1]*c[0]);
}

// a = conj(b)*conj(c)
template <typename Float>
inline void complexConjugateProduct(Float *a, Float *b, Float *c) {
    a[0] = b[0]*c[0] - b[1]*c[1];
    a[1] = -b[0]*c[1] - b[1]*c[0];
}


// Routines used to unpack the gauge field matrices
template <typename Float>
inline void reconstruct8(Float *mat, int dir, int idx) {
  // First reconstruct first row
  Float row_sum = 0.0;
  row_sum += mat[2]*mat[2];
  row_sum += mat[3]*mat[3];
  row_sum += mat[4]*mat[4];
  row_sum += mat[5]*mat[5];
  Float u0 = (dir < 3 ? anisotropy_ : (idx >= (X_[3]-1)*X_[0]*X_[1]*X_[2]/2 ? t_boundary_ : 1));
  Float diff = 1.f/(u0*u0) - row_sum;
  Float U00_mag = sqrt(diff >= 0 ? diff : 0.0);

  mat[14] = mat[0];
  mat[15] = mat[1];

  mat[0] = U00_mag * cos(mat[14]);
  mat[1] = U00_mag * sin(mat[14]);

  Float column_sum = 0.0;
  for (int i=0; i<2; i++) column_sum += mat[i]*mat[i];
  for (int i=6; i<8; i++) column_sum += mat[i]*mat[i];
  diff = 1.f/(u0*u0) - column_sum;
  Float U20_mag = sqrt(diff >= 0 ? diff : 0.0);

  mat[12] = U20_mag * cos(mat[15]);
  mat[13] = U20_mag * sin(mat[15]);

  // First column now restored

  // finally reconstruct last elements from SU(2) rotation
  Float r_inv2 = 1.0/(u0*row_sum);

  // U11
  Float A[2];
  complexDotProduct(A, mat+0, mat+6);
  complexConjugateProduct(mat+8, mat+12, mat+4);
  accumulateComplexProduct(mat+8, A, mat+2, u0);
  mat[8] *= -r_inv2;
  mat[9] *= -r_inv2;

  // U12
  complexConjugateProduct(mat+10, mat+12, mat+2);
  accumulateComplexProduct(mat+10, A, mat+4, -u0);
  mat[10] *= r_inv2;
  mat[11] *= r_inv2;

  // U21
  complexDotProduct(A, mat+0, mat+12);
  complexConjugateProduct(mat+14, mat+6, mat+4);
  accumulateComplexProduct(mat+14, A, mat+2, -u0);
  mat[14] *= r_inv2;
  mat[15] *= r_inv2;

  // U12
  complexConjugateProduct(mat+16, mat+6, mat+2);
  accumulateComplexProduct(mat+16, A, mat+4, u0);
  mat[16] *= -r_inv2;
  mat[17] *= -r_inv2;
}

template <typename Float>
inline void unpack8(Float *h_gauge, double2 *d_gauge, int dir, int V, int idx) {
  double2 *dg = d_gauge + dir*4*V;
  for (int j=0; j<4; j++) {
    h_gauge[2*j+0] = dg[j*V].x;
    h_gauge[2*j+1] = dg[j*V].y;
  }
  reconstruct8(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack8(Float *h_gauge, float4 *d_gauge, int dir, int V, int idx) {
  float4 *dg = d_gauge + dir*2*V;
  h_gauge[0] = dg[0].x;
  h_gauge[1] = dg[0].y;
  h_gauge[2] = dg[0].z;
  h_gauge[3] = dg[0].w;
  h_gauge[4] = dg[V].x;
  h_gauge[5] = dg[V].y;
  h_gauge[6] = dg[V].z;
  h_gauge[7] = dg[V].w;
  reconstruct8(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack8(Float *h_gauge, float2 *d_gauge, int dir, int V, int idx) {
  float2 *dg = d_gauge + dir*4*V;
  for (int j=0; j<4; j++) {
    h_gauge[2*j+0] = dg[j*V].x;
    h_gauge[2*j+1] = dg[j*V].y;
  }
  reconstruct8(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack8(Float *h_gauge, short4 *d_gauge, int dir, int V, int idx) {
  short4 *dg = d_gauge + dir*2*V;
  ShortToFloat(h_gauge[0], dg[0].x);
  ShortToFloat(h_gauge[1], dg[0].y);
  ShortToFloat(h_gauge[2], dg[0].z);
  ShortToFloat(h_gauge[3], dg[0].w);
  ShortToFloat(h_gauge[4], dg[V].x);
  ShortToFloat(h_gauge[5], dg[V].y);
  ShortToFloat(h_gauge[6], dg[V].z);
  ShortToFloat(h_gauge[7], dg[V].w);
  h_gauge[0] *= M_PI;
  h_gauge[1] *= M_PI;
  reconstruct8(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack8(Float *h_gauge, short2 *d_gauge, int dir, int V, int idx) {
  short2 *dg = d_gauge + dir*4*V;
  for (int j=0; j<4; j++) {
    ShortToFloat(h_gauge[2*j+0], dg[j*V].x);
    ShortToFloat(h_gauge[2*j+1], dg[j*V].y);
  }
  h_gauge[0] *= M_PI;
  h_gauge[1] *= M_PI;
  reconstruct8(h_gauge, dir, idx);
}

// do this using complex numbers (simplifies)
template <typename Float>
inline void reconstruct12(Float *mat, int dir, int idx) {
  Float *u = &mat[0*(3*2)];
  Float *v = &mat[1*(3*2)];
  Float *w = &mat[2*(3*2)];
  w[0] = 0.0; w[1] = 0.0; w[2] = 0.0; w[3] = 0.0; w[4] = 0.0; w[5] = 0.0;
  accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
  accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
  accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
  accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
  accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
  accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
  Float u0 = (dir < 3 ? anisotropy_ :
	      (idx >= (X_[3]-1)*X_[0]*X_[1]*X_[2]/2 ? t_boundary_ : 1));
  w[0]*=u0; w[1]*=u0; w[2]*=u0; w[3]*=u0; w[4]*=u0; w[5]*=u0;
}

template <typename Float>
inline void unpack12(Float *h_gauge, double2 *d_gauge, int dir, int V, int idx) {
  double2 *dg = d_gauge + dir*6*V;
  for (int j=0; j<6; j++) {
    h_gauge[j*2+0] = dg[j*V].x;
    h_gauge[j*2+1] = dg[j*V].y; 
  }
  reconstruct12(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack12(Float *h_gauge, float4 *d_gauge, int dir, int V, int idx) {
  float4 *dg = d_gauge + dir*3*V;
  for (int j=0; j<3; j++) {
    h_gauge[j*4+0] = dg[j*V].x;
    h_gauge[j*4+1] = dg[j*V].y; 
    h_gauge[j*4+2] = dg[j*V].z;
    h_gauge[j*4+3] = dg[j*V].w; 
  }
  reconstruct12(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack12(Float *h_gauge, float2 *d_gauge, int dir, int V, int idx) {
  float2 *dg = d_gauge + dir*6*V;
  for (int j=0; j<6; j++) {
    h_gauge[j*2+0] = dg[j*V].x;
    h_gauge[j*2+1] = dg[j*V].y; 
  }
  reconstruct12(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack12(Float *h_gauge, short4 *d_gauge, int dir, int V, int idx) {
  short4 *dg = d_gauge + dir*3*V;
  for (int j=0; j<3; j++) {
    ShortToFloat(h_gauge[j*4+0], dg[j*V].x);
    ShortToFloat(h_gauge[j*4+1], dg[j*V].y);
    ShortToFloat(h_gauge[j*4+2], dg[j*V].z);
    ShortToFloat(h_gauge[j*4+3], dg[j*V].w);
  }
  reconstruct12(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack12(Float *h_gauge, short2 *d_gauge, int dir, int V, int idx) {
  short2 *dg = d_gauge + dir*6*V;
  for (int j=0; j<6; j++) {
    ShortToFloat(h_gauge[j*2+0], dg[j*V].x);
    ShortToFloat(h_gauge[j*2+1], dg[j*V].y); 
  }
  reconstruct12(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack18(Float *h_gauge, double2 *d_gauge, int dir, int V) {
  double2 *dg = d_gauge + dir*9*V;
  for (int j=0; j<9; j++) {
    h_gauge[j*2+0] = dg[j*V].x; 
    h_gauge[j*2+1] = dg[j*V].y;
  }
}

template <typename Float>
inline void unpack18(Float *h_gauge, float4 *d_gauge, int dir, int V) {
  float4 *dg = d_gauge + dir*5*V;
  for (int j=0; j<4; j++) {
    h_gauge[j*4+0] = dg[j*V].x; 
    h_gauge[j*4+1] = dg[j*V].y;
    h_gauge[j*4+2] = dg[j*V].z; 
    h_gauge[j*4+3] = dg[j*V].w;
  }
  h_gauge[16] = dg[4*V].x; 
  h_gauge[17] = dg[4*V].y;
}

template <typename Float>
inline void unpack18(Float *h_gauge, float2 *d_gauge, int dir, int V) {
  float2 *dg = d_gauge + dir*9*V;
  for (int j=0; j<9; j++) {
    h_gauge[j*2+0] = dg[j*V].x; 
    h_gauge[j*2+1] = dg[j*V].y;
  }
}

template <typename Float>
inline void unpack18(Float *h_gauge, short4 *d_gauge, int dir, int V) {
  short4 *dg = d_gauge + dir*5*V;
  for (int j=0; j<4; j++) {
    ShortToFloat(h_gauge[j*4+0], dg[j*V].x);
    ShortToFloat(h_gauge[j*4+1], dg[j*V].y);
    ShortToFloat(h_gauge[j*4+2], dg[j*V].z);
    ShortToFloat(h_gauge[j*4+3], dg[j*V].w);
  }
  ShortToFloat(h_gauge[16],dg[4*V].x);
  ShortToFloat(h_gauge[17],dg[4*V].y);

}

template <typename Float>
inline void unpack18(Float *h_gauge, short2 *d_gauge, int dir, int V) {
  short2 *dg = d_gauge + dir*9*V;
  for (int j=0; j<9; j++) {
    ShortToFloat(h_gauge[j*2+0], dg[j*V].x); 
    ShortToFloat(h_gauge[j*2+1], dg[j*V].y);
  }
}



// Assume the gauge field is "QDP" ordered: directions outside of
// space-time, row-column ordering, even-odd space-time
// offset = 0 for body
// offset = Vh for face
// voxels = Vh for body
// voxels[i] = face volume[i]
template <typename Float, typename FloatN>
static void packQDPGaugeField(FloatN *d_gauge, Float **h_gauge, int oddBit, 
			      QudaReconstructType reconstruct, int Vh, int *voxels,
			      int pad, int offset, int nFace, QudaLinkType type) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      int nMat = nFace*voxels[dir];
      Float *g = h_gauge[dir] + oddBit*nMat*18;
      for (int i = 0; i < nMat; i++) pack12(d_gauge+offset+i, g+i*18, dir, Vh+pad);
    }
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      int nMat = nFace*voxels[dir];
      Float *g = h_gauge[dir] + oddBit*nMat*18;
      for (int i = 0; i < nMat; i++) pack8(d_gauge+offset+i, g+i*18, dir, Vh+pad);
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      int nMat = nFace*voxels[dir];
      Float *g = h_gauge[dir] + oddBit*nMat*18;
      if(type == QUDA_ASQTAD_FAT_LINKS && typeid(FloatN) == typeid(short2) ){
	  //we know it is half precison with fatlink at this stage
	for (int i = 0; i < nMat; i++) 
	  fatlink_short_pack18((short2*)(d_gauge+offset+i), g+i*18, dir, Vh+pad);
      }else{
	for (int i = 0; i < nMat; i++) pack18(d_gauge+offset+i, g+i*18, dir, Vh+pad);
      }
    }
  }
}

// Assume the gauge field is "QDP" ordered: directions outside of
// space-time, row-column ordering, even-odd space-time
template <typename Float, typename FloatN>
static void unpackQDPGaugeField(Float **h_gauge, FloatN *d_gauge, int oddBit, 
				QudaReconstructType reconstruct, int V, int pad) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) unpack12(g+i*18, d_gauge+i, dir, V+pad, i);
    }
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) unpack8(g+i*18, d_gauge+i, dir, V+pad, i);
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) unpack18(g+i*18, d_gauge+i, dir, V+pad);
    }
  }
}

// transpose and scale the matrix
template <typename Float, typename Float2>
static void transposeScale(Float *gT, Float *g, const Float2 &a) {
  for (int ic=0; ic<3; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
    gT[(ic*3+jc)*2+r] = a*g[(jc*3+ic)*2+r];
}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
template <typename Float, typename FloatN>
static void packCPSGaugeField(FloatN *d_gauge, Float *h_gauge, int oddBit, 
			      QudaReconstructType reconstruct, int V, int pad) {
  Float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	transposeScale(gT, g+4*i*18, 1.0 / anisotropy_);
	pack12(d_gauge+i, gT, dir, V+pad);
      }
    } 
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	transposeScale(gT, g+4*i*18, 1.0 / anisotropy_);
	pack8(d_gauge+i, gT, dir, V+pad);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	transposeScale(gT, g+4*i*18, 1.0 / anisotropy_);
	pack18(d_gauge+i, gT, dir, V+pad);
      }
    }
  }

}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
template <typename Float, typename FloatN>
static void unpackCPSGaugeField(Float *h_gauge, FloatN *d_gauge, int oddBit, 
				QudaReconstructType reconstruct, int V, int pad) {
  Float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack12(gT, d_gauge+i, dir, V+pad, i);
	transposeScale(hg+4*i*18, gT, anisotropy_);
      }
    } 
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack8(gT, d_gauge+i, dir, V+pad, i);
	transposeScale(hg+4*i*18, gT, anisotropy_);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack18(gT, d_gauge+i, dir, V+pad);
	transposeScale(hg+4*i*18, gT, anisotropy_);
      }
    }
  }

}


// Assume the gauge field is MILC ordered: directions inside of
// space-time row-column ordering even-odd space-time
template <typename Float, typename FloatN>
void packMILCGaugeField(FloatN *res, Float *gauge, int oddBit, 
			QudaReconstructType reconstruct, int Vh, int pad)
{
  int dir, i;
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (dir = 0; dir < 4; dir++) {
      Float *g = gauge + oddBit*Vh*gaugeSiteSize*4;
      for (i = 0; i < Vh; i++) {
	pack12(res+i, g+(4*i+dir)*gaugeSiteSize, dir, Vh);
      }
    }
  } else if (reconstruct == QUDA_RECONSTRUCT_8){
    for (dir = 0; dir < 4; dir++) {
      Float *g = gauge + oddBit*Vh*gaugeSiteSize*4;
      for (i = 0; i < Vh; i++) {
	pack8(res+i, g+(4*i+dir)*gaugeSiteSize, dir, Vh);
      }
    }
  }else{
    for (dir = 0; dir < 4; dir++) {
      Float *g = gauge + oddBit*Vh*gaugeSiteSize*4;
      for (i = 0; i < Vh; i++) {
	pack18(res+i, g+(4*i+dir)*gaugeSiteSize, dir, Vh);
      }
    }
  }
}

// Assume the gauge field is MILC ordered: directions inside of
// space-time row-column ordering even-odd space-time
template <typename Float, typename FloatN>
static void unpackMILCGaugeField(Float *h_gauge, FloatN *d_gauge, int oddBit, 
				 QudaReconstructType reconstruct, int V, int pad) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack12(hg+4*i*18, d_gauge+i, dir, V+pad, i);
      }
    } 
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack8(hg+4*i*18, d_gauge+i, dir, V+pad, i);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack18(hg+4*i*18, d_gauge+i, dir, V+pad);
      }
    }
  }

}

/*
  Momentum packing/unpacking routines: these are for length 10
  vectors, stored in Float2 format.
 */

template <typename Float, typename Float2>
inline void pack10(Float2 *res, Float *m, int dir, int Vh) 
{
  Float2 *r = res + dir*5*Vh;
  for (int j=0; j<5; j++) {
    r[j*Vh].x = (float)m[j*2+0]; 
    r[j*Vh].y = (float)m[j*2+1]; 
  }
}

template <typename Float, typename Float2>
void packMomField(Float2 *res, Float *mom, int oddBit, int Vh) 
{    
  for (int dir = 0; dir < 4; dir++) {
    Float *g = mom + (oddBit*Vh*4 + dir)*10;
    for (int i = 0; i < Vh; i++) {
      pack10(res+i, g + 4*i*10, dir, Vh);
    }
  }      
}



template <typename Float, typename Float2>
inline void unpack10(Float* m, Float2 *res, int dir, int Vh) 
{
  Float2 *r = res + dir*5*Vh;
  for (int j=0; j<5; j++) {
    m[j*2+0] = r[j*Vh].x;
    m[j*2+1] = r[j*Vh].y;
  }    
}

template <typename Float, typename Float2>
void unpackMomField(Float* mom, Float2 *res, int oddBit, int Vh) 
{
  int dir, i;
  Float *m = mom + oddBit*Vh*10*4;
  
  for (i = 0; i < Vh; i++) {
    for (dir = 0; dir < 4; dir++) {	
      Float* thismom = m + (4*i+dir)*10;
      unpack10(thismom, res+i, dir, Vh);
    }
  }
}

