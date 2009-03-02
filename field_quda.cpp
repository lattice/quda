
#include <stdlib.h>
#include <stdio.h>

#include <quda.h>
#include <field_quda.h>

#include <xmmintrin.h>

// GPU gauge field
FullGauge cudaGauge;
FullGauge cudaHGauge;

// Pinned memory for cpu-gpu memory copying
float4 *packedSpinor = 0;

// Half precision spinor field temporaries
ParityHSpinor hSpinor1, hSpinor2;

void allocateGaugeField(FullGauge *gauge, int packed_gauge_bytes) {
  if (!gauge->even) {
    if (cudaMalloc((void **)&gauge->even, packed_gauge_bytes) == cudaErrorMemoryAllocation) {
      printf("Error allocating even gauge field\n");
      exit(0);
    }
  }
   
  if (!gauge->odd) {
    if (cudaMalloc((void **)&gauge->odd, packed_gauge_bytes) == cudaErrorMemoryAllocation) {
      printf("Error allocating even odd gauge field\n");
      exit(0);
    }
  }
}


ParitySpinor allocateParitySpinor() {
  ParitySpinor ret;

  if (cudaMalloc((void**)&ret, SPINOR_BYTES) == cudaErrorMemoryAllocation) {
    printf("Error allocating spinor\n");
    exit(0);
  }   
    
  return ret;
}

FullSpinor allocateSpinorField() {
  FullSpinor ret;
  ret.even = allocateParitySpinor();
  ret.odd = allocateParitySpinor();
  return ret;
}

void freeParitySpinor(ParitySpinor spinor) {
  cudaFree(spinor);
}

void freeGaugeField() {
  if (cudaGauge.even) cudaFree(cudaGauge.even);
  if (cudaGauge.odd) cudaFree(cudaGauge.odd);
  cudaGauge.even = 0;
  cudaGauge.odd = 0;

  if (cudaHGauge.even) cudaFree(cudaHGauge.even);
  if (cudaHGauge.odd) cudaFree(cudaHGauge.odd);
  cudaHGauge.even = 0;
  cudaHGauge.odd = 0;
}

void freeSpinorField(FullSpinor spinor) {
  cudaFree(spinor.even);
  cudaFree(spinor.odd);
}

void freeSpinorBuffer() {
#ifndef __DEVICE_EMULATION__
  cudaFreeHost(packedSpinor);
#else
  free(packedSpinor);
#endif
  packedSpinor = 0;
}

void allocateSpinorHalf() {
  cudaMalloc((void**)&hSpinor1.spinorHalf, SPINOR_BYTES/2);
  cudaMalloc((void**)&hSpinor1.spinorNorm, SPINOR_BYTES/24);
  cudaMalloc((void**)&hSpinor2.spinorHalf, SPINOR_BYTES/2);
  cudaMalloc((void**)&hSpinor2.spinorNorm, SPINOR_BYTES/24);
}

void freeSpinorHalf() {
  cudaFree(hSpinor1.spinorHalf);
  cudaFree(hSpinor1.spinorNorm);
  cudaFree(hSpinor2.spinorHalf);
  cudaFree(hSpinor2.spinorNorm);
  hSpinor1.spinorHalf = 0;
  hSpinor1.spinorNorm = 0;
  hSpinor2.spinorHalf = 0;
  hSpinor2.spinorNorm = 0;
}


// Packing method taken from Bunk and Sommer
/*inline void pack8Double(float4 *res, double *g, double r0) {
  double f0 = 1.0 / sqrt(2.0*r0*(r0+g[0]));
  double r1_2 = 0.0;
  for (int i=2; i<6; i++) r1_2 += g[i]*g[i];
  double r1 = sqrt(r1_2);
  double f1 = 1.0 / sqrt(2*r1*(r1+g[13]));
  res[0].x = g[12]*f1;
  res[0].y = g[1]*f0;
  res[0].z = g[2]*f0;
  res[0].w = g[3]*f0;
  res[Nh].x = g[4]*f0;
  res[Nh].y = g[5]*f0;
  res[Nh].z = g[6]*f1;
  res[Nh].w = g[7]*f1;
  }*/

inline void pack8Double(float4 *res, double *g) {
  res[0].x = atan2(g[1], g[0]);
  res[0].y = atan2(g[13], g[12]);
  res[0].z = g[2];
  res[0].w = g[3];
  res[Nh].x = g[4];
  res[Nh].y = g[5];
  res[Nh].z = g[6];
  res[Nh].w = g[7];
}

inline void pack8Single(float4 *res, float *g) {
  res[0].x = atan2(g[1], g[0]);
  res[0].y = atan2(g[13], g[12]);
  res[0].z = g[2];
  res[0].w = g[3];
  res[Nh].x = g[4];
  res[Nh].y = g[5];
  res[Nh].z = g[6];
  res[Nh].w = g[7];
}


#define SHORT_LENGTH 65536
#define SCALE_FLOAT (SHORT_LENGTH-1) / 2.f
#define SHIFT_FLOAT -1.f / (SHORT_LENGTH-1)

inline short floatToShort(float a) {
  //return (short)(a*MAX_SHORT);
  short rtn = (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
  return rtn;
}

inline short doubleToShort(double a) {
  //return (short)(a*MAX_SHORT);
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

inline void packHalf8Double(short4 *res, double *g) {
  res[0].x = doubleToShort(atan2(g[1], g[0])/ M_PI);
  res[0].y = doubleToShort(atan2(g[13], g[12])/ M_PI);
  res[0].z = doubleToShort(g[2]);
  res[0].w = doubleToShort(g[3]);
  res[Nh].x = doubleToShort(g[4]);
  res[Nh].y = doubleToShort(g[5]);
  res[Nh].z = doubleToShort(g[6]);
  res[Nh].w = doubleToShort(g[7]);
}

inline void packHalf8Single(short4 *res, float *g) {
  res[0].x = floatToShort(atan2(g[1], g[0])/ M_PI);
  res[0].y = floatToShort(atan2(g[13], g[12])/ M_PI);
  
  /*float t1 = atan2(g[1], g[0])/ M_PI;
  float t2 = atan2(g[13], g[12])/ M_PI;
  if (fabs(t1)>0.99999) printf("%e %d\n", t1, res[0].x);
  if (fabs(t2)>0.99999) printf("%e %d\n", t2, res[0].y);*/
  res[0].z = floatToShort(g[2]);
  res[0].w = floatToShort(g[3]);
  res[Nh].x = floatToShort(g[4]);
  res[Nh].y = floatToShort(g[5]);
  res[Nh].z = floatToShort(g[6]);
  res[Nh].w = floatToShort(g[7]);
}

inline void pack12Double(float4 *res, double *g) {
  for (int j=0; j<3; j++) {
    res[j*Nh].x = (float)g[j*4+0]; 
    res[j*Nh].y = (float)g[j*4+1]; 
    res[j*Nh].z = (float)g[j*4+2]; 
    res[j*Nh].w = (float)g[j*4+3];
  }
}

// sse packing routine
inline void pack12Single(float4 *res, float *g) {
  __m128 a, b, c;
  a = _mm_loadu_ps((const float*)g);
  b = _mm_loadu_ps((const float*)(g+4));
  c = _mm_loadu_ps((const float*)(g+8));
  _mm_store_ps((float*)(res), a);
  _mm_store_ps((float*)(res+Nh), b);
  _mm_store_ps((float*)(res+2*Nh), c);
  /* for (int j=0; j<3; j++) {
    res[j*Nh].x = g[j*4+0]; 
    res[j*Nh].y = g[j*4+1]; 
    res[j*Nh].z = g[j*4+2]; 
    res[j*Nh].w = g[j*4+3];
    }*/
}

inline void packHalf12Double(short4 *res, double *g) {
  for (int j=0; j<3; j++) {
    res[j*Nh].x = doubleToShort(g[j*4+0]);
    res[j*Nh].y = doubleToShort(g[j*4+1]);
    res[j*Nh].z = doubleToShort(g[j*4+2]);
    res[j*Nh].w = doubleToShort(g[j*4+3]);
  }
}

inline void packHalf12Single(short4 *res, float *g) {
  for (int j=0; j<3; j++) {
    res[j*Nh].x = floatToShort(g[j*4+0]);
    res[j*Nh].y = floatToShort(g[j*4+1]);
    res[j*Nh].z = floatToShort(g[j*4+2]);
    res[j*Nh].w = floatToShort(g[j*4+3]);
  }
}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
void packQDPDoubleGaugeField(float4 *res, double **gauge, int oddBit, ReconstructType reconstruct) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      float4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	pack12Double(r+i, g+i*18);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      float4 *r = res + dir*2*Nh;
      for (int i = 0; i < Nh; i++) {
	pack8Double(r+i, g+i*18);
      }
    }
  }
}

void packHalfQDPDoubleGaugeField(short4 *res, double **gauge, int oddBit, ReconstructType reconstruct) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      short4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	packHalf12Double(r+i, g+i*18);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      short4 *r = res + dir*2*Nh;
      for (int i = 0; i < Nh; i++) {
	packHalf8Double(r+i, g+i*18);
      }
    }
  }
}

// Single precision version of the above
void packQDPSingleGaugeField(float4 *res, float **gauge, int oddBit, ReconstructType reconstruct) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {      
      float *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      float4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	pack12Single(r+i, g+i*18);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      float *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      float4 *r = res + dir*2*Nh;
      for (int i = 0; i < Nh; i++) {
	pack8Single(r+i, g+i*18);
      }
    }
  }
}

// Cuda half precision version of the above
void packHalfQDPSingleGaugeField(short4 *res, float **gauge, int oddBit, ReconstructType reconstruct) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      float *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      short4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	packHalf12Single(r+i, g+i*18);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      float *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      short4 *r = res + dir*2*Nh;
      for (int i = 0; i < Nh; i++) {
	packHalf8Single(r+i, g+i*18);
      }
    }
  }
}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
void packCPSDoubleGaugeField(float4 *res, double *gauge, int oddBit, ReconstructType reconstruct) {
  float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      float4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<2; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	      gT[(ic*3+jc)*2+r] = (float)g[4*i*18 + (jc*3+ic)*2+r];      
	pack12Single(r+i, gT);
      }
    } 
  } else {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      float4 *r = res + dir*2*Nh;
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<3; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	      gT[(ic*3+jc)*2+r] = (float)g[4*i*18 + (jc*3+ic)*2+r];      
	pack8Single(r+i, gT);
      }
    }
  }

}

void packHalfCPSDoubleGaugeField(short4 *res, double *gauge, int oddBit, ReconstructType reconstruct) {
  float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      short4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<2; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = (float)g[4*i*18 + (jc*3+ic)*2+r];      
	packHalf12Single(r+i, gT);
      }
    } 
  } else {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      short4 *r = res + dir*2*Nh;
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<3; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = (float)g[4*i*18 + (jc*3+ic)*2+r];      
	packHalf8Single(r+i, gT);
      }
    }
  }

}

// Single precision version of the above
void packCPSSingleGaugeField(float4 *res, float *gauge, int oddBit, ReconstructType reconstruct) {
  float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      float *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      float4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<2; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = g[4*i*18 + (jc*3+ic)*2+r];
	pack12Single(r+i, gT);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      float *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      float4 *r = res + dir*2*Nh;      
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<3; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = g[4*i*18 + (jc*3+ic)*2+r];
	pack8Single(r+i, gT);
      }
    }
  }

}

// Cuda half precision, single precision CPS field
void packHalfCPSSingleGaugeField(short4 *res, float *gauge, int oddBit, ReconstructType reconstruct) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      float *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      float gT[12];
      short4 *r = res + dir*3*Nh;      
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<2; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	      gT[(ic*3+jc)*2+r] = g[4*i*18 + (jc*3+ic)*2+r];
	packHalf12Single(r+i, gT);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      float *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      float gT[18];
      short4 *r = res + dir*2*Nh;      
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<3; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	      gT[(ic*3+jc)*2+r] = g[4*i*18 + (jc*3+ic)*2+r];
	packHalf8Single(r+i, gT);
      }
    }
  }

}

void loadGaugeField(void *gauge) {

  setCudaGaugeParam();

  if (gauge_param->X != L1 || gauge_param->Y != L2 || gauge_param->Z != L3 || gauge_param->T != L4) {
    printf("QUDA error: dimensions do not match: %d=%d, %d=%d, %d=%d, %d=%d\n", 
	   gauge_param->X, L1, gauge_param->Y, L2, gauge_param->Z, L3, gauge_param->T, L4);
    exit(-1);
  }

  if (gauge_param->cuda_prec == QUDA_DOUBLE_PRECISION) {
    printf("QUDA error: sorry, double precision not supported\n");
    exit(-1);
  }

  if (gauge_param->cpu_prec == QUDA_HALF_PRECISION) {
    printf("QUDA error: half precision not supported on cpu\n");
    exit(-1);
  }

  if (gauge_param->reconstruct == QUDA_RECONSTRUCT_NO) {
    printf("QUDA error: ReconstructType not yet supported\n");
    exit(-1);
  }

  int packed_gauge_bytes = (gauge_param->reconstruct == QUDA_RECONSTRUCT_8) ? 8 : 12;

  gauge_param->packed_size = packed_gauge_bytes;
  packed_gauge_bytes *= 4*Nh*sizeof(float);
  allocateGaugeField(&cudaGauge, packed_gauge_bytes);
  cudaGauge.reconstruct = gauge_param->reconstruct;
  cudaGauge.precision = QUDA_SINGLE_PRECISION;

  // 2 since even-odd
  gauge_param->gaugeGiB = (float)2*packed_gauge_bytes/ (1 << 30);

  // Use pinned memory
  float4 *packedEven, *packedOdd;
  
#ifndef __DEVICE_EMULATION__
  cudaMallocHost((void**)&packedEven, packed_gauge_bytes);
  cudaMallocHost((void**)&packedOdd, packed_gauge_bytes);
#else
  packedEven = (float4*)malloc(packed_gauge_bytes);
  packedOdd = (float4*)malloc(packed_gauge_bytes);
#endif
  
  if (gauge_param->gauge_order == QUDA_QDP_GAUGE_ORDER) {
    if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION) {
      packQDPDoubleGaugeField(packedEven, (double**)gauge, 0, gauge_param->reconstruct);
      packQDPDoubleGaugeField(packedOdd,  (double**)gauge, 1, gauge_param->reconstruct);
    } else {
      packQDPSingleGaugeField(packedEven, (float**)gauge, 0, gauge_param->reconstruct);
      packQDPSingleGaugeField(packedOdd,  (float**)gauge, 1, gauge_param->reconstruct);
    }
  } else if (gauge_param->gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
    
    if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION) {
      packCPSDoubleGaugeField(packedEven, (double*)gauge, 0, gauge_param->reconstruct);
      packCPSDoubleGaugeField(packedOdd,  (double*)gauge, 1, gauge_param->reconstruct);
    } else {
      packCPSSingleGaugeField(packedEven, (float*)gauge, 0, gauge_param->reconstruct);
      packCPSSingleGaugeField(packedOdd,  (float*)gauge, 1, gauge_param->reconstruct);
    }
  } else {
    printf("Sorry, %d GaugeFieldOrder not supported\n", gauge_param->gauge_order);
    exit(-1);
  }
  
  cudaMemcpy(cudaGauge.even, packedEven, packed_gauge_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaGauge.odd,  packedOdd,  packed_gauge_bytes, cudaMemcpyHostToDevice);    
  
#ifndef __DEVICE_EMULATION__
  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
#else
  free(packedEven);
  free(packedOdd);
#endif

  if (gauge_param->cuda_prec == QUDA_HALF_PRECISION) {
    packed_gauge_bytes /= 2;
    gauge_param->packed_size += packed_gauge_bytes/(4*Nh*sizeof(float));
    allocateGaugeField(&cudaHGauge, packed_gauge_bytes);
    cudaHGauge.reconstruct = gauge_param->reconstruct;
    cudaHGauge.precision = QUDA_HALF_PRECISION;
  }

  if (gauge_param->cuda_prec == QUDA_HALF_PRECISION) {
    // Use pinned memory
    short4 *packedEven, *packedOdd;
#ifndef __DEVICE_EMULATION__
    cudaMallocHost((void**)&packedEven, packed_gauge_bytes);
    cudaMallocHost((void**)&packedOdd, packed_gauge_bytes);
#else
    packedEven = (short4*)malloc(packed_gauge_bytes);
    packedOdd = (short4*)malloc(packed_gauge_bytes);
#endif
    
    if (gauge_param->gauge_order == QUDA_QDP_GAUGE_ORDER) {
      if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION) {
	packHalfQDPDoubleGaugeField(packedEven, (double**)gauge, 0, gauge_param->reconstruct);
	packHalfQDPDoubleGaugeField(packedOdd,  (double**)gauge, 1, gauge_param->reconstruct);
      } else {
	packHalfQDPSingleGaugeField(packedEven, (float**)gauge, 0, gauge_param->reconstruct);
	packHalfQDPSingleGaugeField(packedOdd,  (float**)gauge, 1, gauge_param->reconstruct);
      }
    } else if (gauge_param->gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
      if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION) {
	packHalfCPSDoubleGaugeField(packedEven, (double*)gauge, 0, gauge_param->reconstruct);
	packHalfCPSDoubleGaugeField(packedOdd,  (double*)gauge, 1, gauge_param->reconstruct);
      } else {
	packHalfCPSSingleGaugeField(packedEven, (float*)gauge, 0, gauge_param->reconstruct);
	packHalfCPSSingleGaugeField(packedOdd,  (float*)gauge, 1, gauge_param->reconstruct);
      }
    } else {
      printf("Sorry, %d GaugeFieldOrder not supported\n", gauge_param->gauge_order);
      exit(-1);
    }

    cudaMemcpy(cudaHGauge.even, packedEven, packed_gauge_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaHGauge.odd,  packedOdd,  packed_gauge_bytes, cudaMemcpyHostToDevice);    

#ifndef __DEVICE_EMULATION__
    cudaFreeHost(packedEven);
    cudaFreeHost(packedOdd);
#else
    free(packedEven);
    free(packedOdd);
#endif

  }

}


inline void packFloat4(float4* a, float *b) {
  __m128 SSEtmp;
  SSEtmp = _mm_loadu_ps((const float*)b);
  _mm_store_ps((float*)a, SSEtmp);
  //a->x = b[0]; a->y = b[1]; a->z = b[2]; a->w = b[3];
}

inline void unpackFloat4(float *a, float4 *b) {
  __m128 SSEtmp;
  SSEtmp = _mm_load_ps((const float*)b);
  _mm_storeu_ps((float*)a, SSEtmp);
  //a[0] = b->x; a[1] = b->y; a[2] = b->z; a[3] = b->w;
}

// Standard spinor packing, colour inside spin
void packDoubleParitySpinor(float4 *res, double *spinor) {
  double K = 1.0 / (2.0);
  float b[24];

  for (int i = 0; i < Nh; i++) {
    double *a = spinor+i*24;

    for (int c=0; c<3; c++) {
      for (int r=0; r<2; r++) {
	int cr = c*2+r;
	b[0*6+cr] = K*(a[1*6+cr]+a[3*6+cr]);
	b[1*6+cr] = -K*(a[0*6+cr]+a[2*6+cr]);
	b[2*6+cr] = K*(a[1*6+cr]-a[3*6+cr]);
	b[3*6+cr] = K*(a[2*6+cr]-a[0*6+cr]);
      }
    }

    for (int j = 0; j < 6; j++) packFloat4(res+j*Nh+i, b+j*4);
  }
}

// single precision version of the above
void packSingleParitySpinor(float4 *res, float *spinor) {
  float K = 1.0 / (2.0);
  float b[24];

  for (int i = 0; i < Nh; i++) {
    float *a = spinor+i*24;

    for (int c=0; c<3; c++) {
      for (int r=0; r<2; r++) {
	int cr = c*2+r;
	b[cr] = K*(a[6+cr]+a[18+cr]);
	b[6+cr] = -K*(a[cr]+a[12+cr]);
	b[12+cr] = K*(a[6+cr]-a[18+cr]);
	b[18+cr] = K*(a[12+cr]-a[cr]);
      }
    }

    for (int j = 0; j < 6; j++) packFloat4(res+j*Nh+i, b+j*4);
  }
}

// QDP spinor packing, spin inside colour
void packQDPDoubleParitySpinor(float4 *res, double *spinor) {
  double K = 1.0 / 2.0;
  
  float b[24];
  for (int i = 0; i < Nh; i++) {
    double *a = spinor+i*24;

    // reorder and change basis
    for (int c=0; c<3; c++) {
      for (int r=0; r<2; r++) {
	int cr = c*2+r;
	b[0*6+cr] = K*(a[(c*4+1)*2+r]+a[(c*4+3)*2+r]);
	b[1*6+cr] = -K*(a[(c*4+0)*2+r]+a[(c*4+2)*2+r]);
	b[2*6+cr] = K*(a[(c*4+1)*2+r]-a[(c*4+3)*2+r]);
	b[3*6+cr] = K*(a[(c*4+2)*2+r]-a[(c*4+0)*2+r]);
      }
    }

    for (int j = 0; j < 6; j++) packFloat4(res+j*Nh+i, b+j*4);
  }
}



// Single precision oversion of the above
void packQDPSingleParitySpinor(float4 *res, float *spinor) {
  float K = 1.0 / 2.0;
  
  float b[24];
  for (int i = 0; i < Nh; i++) {
    float *a = spinor+i*24;

    // reorder and change basis
    for (int c=0; c<3; c++) {
      for (int r=0; r<2; r++) {
	int cr = c*2+r;
	b[0*6+cr] = K*(a[(c*4+1)*2+r]+a[(c*4+3)*2+r]);
	b[1*6+cr] = -K*(a[(c*4+0)*2+r]+a[(c*4+2)*2+r]);
	b[2*6+cr] = K*(a[(c*4+1)*2+r]-a[(c*4+3)*2+r]);
	b[3*6+cr] = K*(a[(c*4+2)*2+r]-a[(c*4+0)*2+r]);
      }
    }

    for (int j = 0; j < 6; j++) packFloat4(res+j*Nh+i, b+j*4);
  }
}

void unpackDoubleParitySpinor(double *res, float4 *spinorPacked) {
  float K = 1.0;///sqrt(2.0);
  float b[24];

  for (int i = 0; i < Nh; i++) {
    double *a = res+i*24;

    for (int j = 0; j < 6; j++) unpackFloat4(b+j*4, spinorPacked+j*Nh+i);

    for (int c=0; c<3; c++) {
      for (int r=0; r<2; r++) {
	int cr = c*2+r;
	a[0*6+cr] = -K*(b[1*6+cr]+b[3*6+cr]);
	a[1*6+cr] = K*(b[0*6+cr]+b[2*6+cr]);
	a[2*6+cr] = -K*(b[1*6+cr]-b[3*6+cr]);
	a[3*6+cr] = -K*(b[2*6+cr]-b[0*6+cr]);
      }
    }

  }

}

void unpackSingleParitySpinor(float *res, float4 *spinorPacked) {
  float K = 1.0;///sqrt(2.0);
  float b[24];

  for (int i = 0; i < Nh; i++) {
    float *a = res+i*24;

    for (int j = 0; j < 6; j++) unpackFloat4(b+j*4, spinorPacked+j*Nh+i);

    for (int c=0; c<3; c++) {
      for (int r=0; r<2; r++) {
	int cr = c*2+r;
	a[0*6+cr] = -K*(b[1*6+cr]+b[3*6+cr]);
	a[1*6+cr] = K*(b[0*6+cr]+b[2*6+cr]);
	a[2*6+cr] = -K*(b[1*6+cr]-b[3*6+cr]);
	a[3*6+cr] = -K*(b[2*6+cr]-b[0*6+cr]);
      }
    }

  }

}

void unpackQDPDoubleParitySpinor(double *res, float4 *spinorPacked) {
  float K = 1.0;///sqrt(2.0);
  float b[24];

  for (int i = 0; i < Nh; i++) {
    double *a = res+i*24;

    for (int j = 0; j < 6; j++) unpackFloat4(b+j*4, spinorPacked+j*Nh+i);

    for (int c=0; c<3; c++) {
      for (int r=0; r<2; r++) {
	int cr = c*2+r;
	a[(c*4+0)*2+r] = -K*(b[1*6+cr]+b[3*6+cr]);
	a[(c*4+1)*2+r] = K*(b[0*6+cr]+b[2*6+cr]);
	a[(c*4+2)*2+r] = -K*(b[1*6+cr]-b[3*6+cr]);
	a[(c*4+3)*2+r] = -K*(b[2*6+cr]-b[0*6+cr]);
      }
    }
  }

}

void unpackQDPSingleParitySpinor(float *res, float4 *spinorPacked) {
  float K = 1.0;///sqrt(2.0);
  float b[24];

  for (int i = 0; i < Nh; i++) {
    float *a = res+i*24;

    for (int j = 0; j < 6; j++) unpackFloat4(b+j*4, spinorPacked+j*Nh+i);

    for (int c=0; c<3; c++) {
      for (int r=0; r<2; r++) {
	int cr = c*2+r;
	a[(c*4+0)*2+r] = -K*(b[1*6+cr]+b[3*6+cr]);
	a[(c*4+1)*2+r] = K*(b[0*6+cr]+b[2*6+cr]);
	a[(c*4+2)*2+r] = -K*(b[1*6+cr]-b[3*6+cr]);
	a[(c*4+3)*2+r] = -K*(b[2*6+cr]-b[0*6+cr]);
      }
    }

  }
}


void loadParitySpinor(ParitySpinor ret, void *spinor, Precision cpu_prec, 
		      Precision cuda_prec, DiracFieldOrder dirac_order) {
  if (cuda_prec == QUDA_DOUBLE_PRECISION) {
    printf("Sorry, only double precision not supported\n");
    exit(-1);
  }

  if (cuda_prec == QUDA_HALF_PRECISION) {
    if (!hSpinor1.spinorHalf && !hSpinor1.spinorNorm &&
	!hSpinor2.spinorHalf && !hSpinor2.spinorNorm ) {
      allocateSpinorHalf();
    } else if (!hSpinor1.spinorHalf || !hSpinor1.spinorNorm ||
	       !hSpinor2.spinorHalf || !hSpinor2.spinorNorm) {
      printf("allocateSpinorHalf error %lu %lu %lu %lu\n", 
	     (unsigned long)hSpinor1.spinorHalf, (unsigned long)hSpinor1.spinorNorm,
	     (unsigned long)hSpinor2.spinorHalf, (unsigned long)hSpinor2.spinorNorm);
      exit(-1);
    }
  }

#ifndef __DEVICE_EMULATION__
  if (!packedSpinor) cudaMallocHost((void**)&packedSpinor, SPINOR_BYTES);
#else
  if (!packedSpinor) packedSpinor = (float4*)malloc(SPINOR_BYTES);
#endif
  if (dirac_order == QUDA_DIRAC_ORDER || QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) packDoubleParitySpinor(packedSpinor, (double*)spinor);
    else packSingleParitySpinor(packedSpinor, (float*)spinor);
  } else if (dirac_order == QUDA_QDP_DIRAC_ORDER) {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) packQDPDoubleParitySpinor(packedSpinor, (double*)spinor);
    else packQDPSingleParitySpinor(packedSpinor, (float*)spinor);
  }
  cudaMemcpy(ret, packedSpinor, SPINOR_BYTES, cudaMemcpyHostToDevice);
}

void loadSpinorField(FullSpinor ret, void *spinor, Precision cpu_prec, 
		     Precision cuda_prec, DiracFieldOrder dirac_order) {
  void *spinor_odd;
  if (cpu_prec == QUDA_SINGLE_PRECISION) spinor_odd = (float*)spinor + Nh*spinorSiteSize;
  else spinor_odd = (double*)spinor + Nh*spinorSiteSize;

  if (dirac_order == QUDA_DIRAC_ORDER || dirac_order == QUDA_QDP_DIRAC_ORDER) {
    loadParitySpinor(ret.even, spinor, cpu_prec, cuda_prec, dirac_order);
    loadParitySpinor(ret.odd, spinor_odd, cpu_prec, cuda_prec, dirac_order);
  } else if (dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    // odd-even so reverse order
    loadParitySpinor(ret.even, spinor_odd, cpu_prec, cuda_prec, dirac_order);
    loadParitySpinor(ret.odd, spinor, cpu_prec, cuda_prec, dirac_order);
  } else {
    printf("DiracFieldOrder %d not supported\n", dirac_order);
    exit(-1);
  }
}

void retrieveParitySpinor(void *res, ParitySpinor spinor, Precision cpu_prec, Precision cuda_prec,
			  DiracFieldOrder dirac_order) {
  /*if (cuda_prec != QUDA_SINGLE_PRECISION) {
    printf("Sorry, only single precision supported\n");
    exit(-1);
    }*/

  if (!packedSpinor) cudaMallocHost((void**)&packedSpinor, SPINOR_BYTES);
  cudaMemcpy(packedSpinor, spinor, SPINOR_BYTES, cudaMemcpyDeviceToHost);
  if (dirac_order == QUDA_DIRAC_ORDER || QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) unpackDoubleParitySpinor((double*)res, packedSpinor);
    else unpackSingleParitySpinor((float*)res, packedSpinor);
  } else if (dirac_order == QUDA_QDP_DIRAC_ORDER) {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) unpackQDPDoubleParitySpinor((double*)res, packedSpinor);
    else unpackQDPSingleParitySpinor((float*)res, packedSpinor);
  }
}

void retrieveSpinorField(void *res, FullSpinor spinor, Precision cpu_prec, 
			 Precision cuda_prec, DiracFieldOrder dirac_order) {
  void *res_odd;
  if (cpu_prec == QUDA_SINGLE_PRECISION) res_odd = (float*)res + Nh*spinorSiteSize;
  else res_odd = (double*)res + Nh*spinorSiteSize;

  if (dirac_order == QUDA_DIRAC_ORDER || dirac_order == QUDA_QDP_DIRAC_ORDER) {
    retrieveParitySpinor(res, spinor.even, cpu_prec, cuda_prec, dirac_order);
    retrieveParitySpinor(res_odd, spinor.odd, cpu_prec, cuda_prec, dirac_order);
  } else if (dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    retrieveParitySpinor(res, spinor.odd, cpu_prec, cuda_prec, dirac_order);
    retrieveParitySpinor(res_odd, spinor.even, cpu_prec, cuda_prec, dirac_order);
  } else {
    printf("DiracFieldOrder %d not supported\n", dirac_order);
    exit(-1);
  }
  
}

void spinorHalfPack(float *c, short *s0, float *f0) {

  float *f = f0;
  short *s = s0;
  for (int i=0; i<24*Nh; i+=24) {
    c[i] = sqrt(f[0]*f[0] + f[1]*f[1]);
    for (int j=0; j<24; j+=2) {
      float k = sqrt(f[j]*f[j] + f[j+1]*f[j+1]);
      if (k > c[i]) c[i] = k;
    }

    for (int j=0; j<24; j++) s[j] = (short)(MAX_SHORT*f[j]/c[i]);
    f+=24;
    s+=24;
  }

}

void spinorHalfUnpack(float *f0, float *c, short *s0) {
  float *f = f0;
  short *s = s0;
  for (int i=0; i<24*Nh; i+=24) {
    for (int j=0; j<24; j++) f[j] = s[j] * (c[i] / MAX_SHORT);
    f+=24;
    s+=24;
  }

}
