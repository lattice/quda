#include <stdlib.h>
#include <stdio.h>

#include <quda.h>
#include <gauge_quda.h>

#include <xmmintrin.h>

#define SHORT_LENGTH 65536
#define SCALE_FLOAT (SHORT_LENGTH-1) / 2.f
#define SHIFT_FLOAT -1.f / (SHORT_LENGTH-1)

double Anisotropy;
QudaTboundary tBoundary;
int X[4];

template <typename Float>
inline short FloatToShort(const Float &a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

template <typename Float>
inline void ShortToFloat(Float &a, const short &b) {
  a = (((Float)b/SCALE_FLOAT-SHIFT_FLOAT)*SCALE_FLOAT);
}

// Routines used to pack the gauge field matrices

template <typename Float>
inline void pack8(double2 *d_gauge, Float *h_gauge, int dir, int V) {
  double2 *dg = d_gauge + dir*4*V;
  dg[0].x = atan2(h_gauge[1], h_gauge[0]);
  dg[0].y = atan2(h_gauge[13], h_gauge[12]);
  for (int j=1; j<4; j++) {
    dg[j*V].x = h_gauge[2*j+0];
    dg[j*V].y = h_gauge[2*j+1];
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
inline void pack8(short4 *res, Float *g, int dir, int V) {
  short4 *r = res + dir*2*V;
  r[0].x = FloatToShort(atan2(g[1], g[0])/ M_PI);
  r[0].y = FloatToShort(atan2(g[13], g[12])/ M_PI);
  r[0].z = FloatToShort(g[2]);
  r[0].w = FloatToShort(g[3]);
  r[V].x = FloatToShort(g[4]);
  r[V].y = FloatToShort(g[5]);
  r[V].z = FloatToShort(g[6]);
  r[V].w = FloatToShort(g[7]);
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
inline void pack18(double2 *d_gauge, Float *h_gauge, int dir, int V) {
  double2 *dg = d_gauge + dir*9*V;
  for (int j=0; j<9; j++) {
    dg[j*V].x = h_gauge[j*2+0]; 
    dg[j*V].y = h_gauge[j*2+1]; 
  }
}

template <typename Float>
inline void pack18(float4 *d_gauge, Float *h_gauge, int dir, int V) {
  float4 *dg = d_gauge + dir*5*V;
  for (int j=0; j<4; j++) {
    dg[j*V].x = h_gauge[j*4+0]; 
    dg[j*V].y = h_gauge[j*4+1]; 
    dg[j*V].z = h_gauge[j*4+2]; 
    dg[j*V].w = h_gauge[j*4+3]; 
  }
  dg[16*V].x = h_gauge[16]; 
  dg[17*V].y = h_gauge[17]; 
  dg[18*V].z = 0.0;
  dg[19*V].w = 0.0;
}

template <typename Float>
inline void pack18(short4 *d_gauge, Float *h_gauge, int dir, int V) {
  short4 *dg = d_gauge + dir*5*V;
  for (int j=0; j<4; j++) {
    dg[j*V].x = FloatToShort(h_gauge[j*4+0]); 
    dg[j*V].y = FloatToShort(h_gauge[j*4+1]); 
    dg[j*V].z = FloatToShort(h_gauge[j*4+2]); 
    dg[j*V].w = FloatToShort(h_gauge[j*4+3]); 
  }
  dg[16*V].x = FloatToShort(h_gauge[16]); 
  dg[17*V].y = FloatToShort(h_gauge[17]); 
  dg[18*V].z = (short)0;
  dg[19*V].w = (short)0;
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
void complexConjugateProduct(Float *a, Float *b, Float *c) {
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
  Float u0 = (dir < 3 ? Anisotropy : (idx >= (X[3]-1)*X[0]*X[1]*X[2]/2 ? tBoundary : 1));
  Float U00_mag = sqrt(1.f/(u0*u0) - row_sum);

  mat[14] = mat[0];
  mat[15] = mat[1];

  mat[0] = U00_mag * cos(mat[14]);
  mat[1] = U00_mag * sin(mat[14]);

  Float column_sum = 0.0;
  for (int i=0; i<2; i++) column_sum += mat[i]*mat[i];
  for (int i=6; i<8; i++) column_sum += mat[i]*mat[i];
  Float U20_mag = sqrt(1.f/(u0*u0) - column_sum);

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
  Float u0 = (dir < 3 ? Anisotropy :
	      (idx >= (X[3]-1)*X[0]*X[1]*X[2]/2 ? tBoundary : 1));
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
    h_gauge[j*2+0] = dg[j*V].x;
    h_gauge[j*2+1] = dg[j*V].y; 
    h_gauge[j*2+2] = dg[j*V].z;
    h_gauge[j*2+3] = dg[j*V].w; 
  }
  reconstruct12(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack12(Float *h_gauge, short4 *d_gauge, int dir, int V, int idx) {
  short4 *dg = d_gauge + dir*3*V;
  for (int j=0; j<3; j++) {
    ShortToFloat(h_gauge[j*2+0], dg[j*V].x);
    ShortToFloat(h_gauge[j*2+1], dg[j*V].y);
    ShortToFloat(h_gauge[j*2+2], dg[j*V].z);
    ShortToFloat(h_gauge[j*2+3], dg[j*V].w);
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

// Assume the gauge field is "QDP" ordered: directions outside of
// space-time, row-column ordering, even-odd space-time
template <typename Float, typename FloatN>
void packQDPGaugeField(FloatN *d_gauge, Float **h_gauge, int oddBit, ReconstructType reconstruct, int V) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) pack12(d_gauge+i, g+i*18, dir, V);
    }
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) pack8(d_gauge+i, g+i*18, dir, V);
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) pack18(d_gauge+i, g+i*18, dir, V);
    }
  }
}

// Assume the gauge field is "QDP" ordered: directions outside of
// space-time, row-column ordering, even-odd space-time
template <typename Float, typename FloatN>
void unpackQDPGaugeField(Float **h_gauge, FloatN *d_gauge, int oddBit, ReconstructType reconstruct, int V) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) unpack12(g+i*18, d_gauge+i, dir, V, i);
    }
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) unpack8(g+i*18, d_gauge+i, dir, V, i);
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) unpack18(g+i*18, d_gauge+i, dir, V);
    }
  }
}

// transpose and scale the matrix
template <typename Float, typename Float2>
void transposeScale(Float *gT, Float *g, const Float2 &a) {
  for (int ic=0; ic<3; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
    gT[(ic*3+jc)*2+r] = a*g[(jc*3+ic)*2+r];
}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
template <typename Float, typename FloatN>
void packCPSGaugeField(FloatN *d_gauge, Float *h_gauge, int oddBit, ReconstructType reconstruct, int V) {
  Float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	transposeScale(gT, g, 1.0 / Anisotropy);
	pack12(d_gauge+i, gT, dir, V);
      }
    } 
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	transposeScale(gT, g, 1.0 / Anisotropy);
	pack8(d_gauge+i, gT, dir, V);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	transposeScale(gT, g, 1.0 / Anisotropy);
	pack18(d_gauge+i, gT, dir, V);
      }
    }
  }

}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
template <typename Float, typename FloatN>
void unpackCPSGaugeField(Float *h_gauge, FloatN *d_gauge, int oddBit, ReconstructType reconstruct, int V) {
  Float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack12(gT, d_gauge+i, dir, V, i);
	transposeScale(hg, gT, Anisotropy);
      }
    } 
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack8(gT, d_gauge+i, dir, V, i);
	transposeScale(hg, gT, Anisotropy);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack18(gT, d_gauge+i, dir, V);
	transposeScale(hg, gT, Anisotropy);
      }
    }
  }

}

void allocateGaugeField(FullGauge *cudaGauge, ReconstructType reconstruct, Precision precision) {

  cudaGauge->reconstruct = reconstruct;
  cudaGauge->precision = precision;

  cudaGauge->Nc = 3;

  int floatSize;
  if (precision == QUDA_DOUBLE_PRECISION) floatSize = sizeof(double);
  else if (precision == QUDA_SINGLE_PRECISION) floatSize = sizeof(float);
  else floatSize = sizeof(float)/2;

  int elements = (reconstruct == QUDA_RECONSTRUCT_8) ? 8 : 12;
  cudaGauge->bytes = 4*cudaGauge->volume*elements*floatSize;

  if (!cudaGauge->even) {
    if (cudaMalloc((void **)&cudaGauge->even, cudaGauge->bytes) == cudaErrorMemoryAllocation) {
      printf("Error allocating even gauge field\n");
      exit(0);
    }
  }
   
  if (!cudaGauge->odd) {
    if (cudaMalloc((void **)&cudaGauge->odd, cudaGauge->bytes) == cudaErrorMemoryAllocation) {
      printf("Error allocating even odd gauge field\n");
      exit(0);
    }
  }

}

void freeGaugeField(FullGauge *cudaGauge) {
  if (cudaGauge->even) cudaFree(cudaGauge->even);
  if (cudaGauge->odd) cudaFree(cudaGauge->odd);
  cudaGauge->even = NULL;
  cudaGauge->odd = NULL;
}

template <typename Float, typename FloatN>
void loadGaugeField(FloatN *even, FloatN *odd, Float *cpuGauge, ReconstructType reconstruct, 
		    int bytes, int Vh) {

  // Use pinned memory
  FloatN *packedEven, *packedOdd;
    
#ifndef __DEVICE_EMULATION__
  cudaMallocHost((void**)&packedEven, bytes);
  cudaMallocHost((void**)&packedOdd, bytes);
#else
  packedEven = (FloatN*)malloc(bytes);
  packedOdd = (FloatN*)malloc(bytes);
#endif
    
  if (gauge_param->gauge_order == QUDA_QDP_GAUGE_ORDER) {
    packQDPGaugeField(packedEven, (Float**)cpuGauge, 0, reconstruct, Vh);
    packQDPGaugeField(packedOdd,  (Float**)cpuGauge, 1, reconstruct, Vh);
  } else if (gauge_param->gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
    packCPSGaugeField(packedEven, (Float*)cpuGauge, 0, reconstruct, Vh);
    packCPSGaugeField(packedOdd,  (Float*)cpuGauge, 1, reconstruct, Vh);
    
  } else {
    printf("Sorry, %d GaugeFieldOrder not supported\n", gauge_param->gauge_order);
    exit(-1);
  }
    
  cudaMemcpy(even, packedEven, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(odd,  packedOdd, bytes, cudaMemcpyHostToDevice);    
    
#ifndef __DEVICE_EMULATION__
  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
#else
  free(packedEven);
  free(packedOdd);
#endif

}

template <typename Float, typename FloatN>
void retrieveGaugeField(Float *cpuGauge, FloatN *even, FloatN *odd, ReconstructType reconstruct, 
			int bytes, int Vh) {

  // Use pinned memory
  FloatN *packedEven, *packedOdd;
    
#ifndef __DEVICE_EMULATION__
  cudaMallocHost((void**)&packedEven, bytes);
  cudaMallocHost((void**)&packedOdd, bytes);
#else
  packedEven = (FloatN*)malloc(bytes);
  packedOdd = (FloatN*)malloc(bytes);
#endif
    
  cudaMemcpy(packedEven, even, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(packedOdd, odd, bytes, cudaMemcpyHostToDevice);    
    
  if (gauge_param->gauge_order == QUDA_QDP_GAUGE_ORDER) {
    unpackQDPGaugeField((Float**)cpuGauge, packedEven, 0, reconstruct, Vh);
    unpackQDPGaugeField((Float**)cpuGauge, packedOdd, 1, reconstruct, Vh);
  } else if (gauge_param->gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
    unpackCPSGaugeField((Float*)cpuGauge, packedEven, 0, reconstruct, Vh);
    unpackCPSGaugeField((Float*)cpuGauge, packedOdd, 1, reconstruct, Vh);
  } else {
    printf("Sorry, %d GaugeFieldOrder not supported\n", gauge_param->gauge_order);
    exit(-1);
  }
    
#ifndef __DEVICE_EMULATION__
  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
#else
  free(packedEven);
  free(packedOdd);
#endif

}

void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, Precision precision, ReconstructType reconstruct, 
		      Tboundary t_boundary, int *XX, double anisotropy, int blockDim) {

  if (gauge_param->cpu_prec == QUDA_HALF_PRECISION) {
    printf("QUDA error: half precision not supported on cpu\n");
    exit(-1);
  }

  Anisotropy = anisotropy;
  tBoundary = t_boundary;

  cudaGauge->anisotropy = anisotropy;
  cudaGauge->volume = 1;
  for (int d=0; d<4; d++) {
    cudaGauge->X[d] = XX[d];
    cudaGauge->volume *= XX[d];
    X[d] = XX[d];
  }
  cudaGauge->X[0] /= 2; // actually store the even-odd sublattice dimensions
  cudaGauge->volume /= 2;

  cudaGauge->blockDim = blockDim;

  allocateGaugeField(cudaGauge, reconstruct, precision);

  if (precision == QUDA_DOUBLE_PRECISION) {

    if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION)
      loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (double*)cpuGauge, 
		     cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
    else if (gauge_param->cpu_prec == QUDA_SINGLE_PRECISION)
      loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (float*)cpuGauge, 
		     cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);

  } else if (precision == QUDA_SINGLE_PRECISION) {

    if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION)
      loadGaugeField((float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), (double*)cpuGauge, 
		     cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
    else if (gauge_param->cpu_prec == QUDA_SINGLE_PRECISION)
      loadGaugeField((float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), (float*)cpuGauge, 
		     cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
 
  } else if (precision == QUDA_HALF_PRECISION) {

    if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION)
      loadGaugeField((short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), (double*)cpuGauge, 
		     cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
    else if (gauge_param->cpu_prec == QUDA_SINGLE_PRECISION)
      loadGaugeField((short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), (float*)cpuGauge,
		     cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);

  }
}

void restoreGaugeField(void *cpuGauge, FullGauge *cudaGauge) {

  if (gauge_param->cpu_prec == QUDA_HALF_PRECISION) {
    printf("QUDA error: half precision not supported on cpu\n");
    exit(-1);
  }

  if (cudaGauge->precision == QUDA_DOUBLE_PRECISION) {

    if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION)
      retrieveGaugeField((double*)cpuGauge, (double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), 
			 cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
    else if (gauge_param->cpu_prec == QUDA_SINGLE_PRECISION)
      retrieveGaugeField((float*)cpuGauge, (double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), 
			 cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);

  } else if (cudaGauge->precision == QUDA_SINGLE_PRECISION) {

    if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION)
      retrieveGaugeField((double*)cpuGauge, (float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), 
			 cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
    else if (gauge_param->cpu_prec == QUDA_SINGLE_PRECISION)
      retrieveGaugeField((float*)cpuGauge, (float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), 
			 cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);

  } else if (cudaGauge->precision == QUDA_HALF_PRECISION) {

    if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION)
      retrieveGaugeField((double*)cpuGauge, (short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), 
			cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
    else if (gauge_param->cpu_prec == QUDA_SINGLE_PRECISION)
      retrieveGaugeField((float*)cpuGauge, (short4*)(cudaGauge->even), (short4*)(cudaGauge->odd),
			 cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);

  }
}
