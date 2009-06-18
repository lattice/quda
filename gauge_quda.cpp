#include <stdlib.h>
#include <stdio.h>

#include <quda.h>
#include <gauge_quda.h>

#include <xmmintrin.h>

#define __DEVICE_EMULATION__

#define SHORT_LENGTH 65536
#define SCALE_FLOAT (SHORT_LENGTH-1) / 2.f
#define SHIFT_FLOAT -1.f / (SHORT_LENGTH-1)

inline short floatToShort(float a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

inline short doubleToShort(double a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

inline void packD8D(double2 *res, double *g) {
  res[0].x = atan2(g[1], g[0]);
  res[0].y = atan2(g[13], g[12]);
  res[Nh].x = g[2];
  res[Nh].y = g[3];
  res[2*Nh].x = g[4];
  res[2*Nh].y = g[5];
  res[3*Nh].x = g[6];
  res[3*Nh].y = g[7];
}

inline void packS8D(float4 *res, double *g) {
  res[0].x = atan2(g[1], g[0]);
  res[0].y = atan2(g[13], g[12]);
  res[0].z = g[2];
  res[0].w = g[3];
  res[Nh].x = g[4];
  res[Nh].y = g[5];
  res[Nh].z = g[6];
  res[Nh].w = g[7];
}

inline void packS8S(float4 *res, float *g) {
  res[0].x = atan2(g[1], g[0]);
  res[0].y = atan2(g[13], g[12]);
  res[0].z = g[2];
  res[0].w = g[3];
  res[Nh].x = g[4];
  res[Nh].y = g[5];
  res[Nh].z = g[6];
  res[Nh].w = g[7];
}

inline void packH8D(short4 *res, double *g) {
  res[0].x = doubleToShort(atan2(g[1], g[0])/ M_PI);
  res[0].y = doubleToShort(atan2(g[13], g[12])/ M_PI);
  res[0].z = doubleToShort(g[2]);
  res[0].w = doubleToShort(g[3]);
  res[Nh].x = doubleToShort(g[4]);
  res[Nh].y = doubleToShort(g[5]);
  res[Nh].z = doubleToShort(g[6]);
  res[Nh].w = doubleToShort(g[7]);
}

inline void packH8S(short4 *res, float *g) {
  res[0].x = floatToShort(atan2(g[1], g[0])/ M_PI);
  res[0].y = floatToShort(atan2(g[13], g[12])/ M_PI);
  res[0].z = floatToShort(g[2]);
  res[0].w = floatToShort(g[3]);
  res[Nh].x = floatToShort(g[4]);
  res[Nh].y = floatToShort(g[5]);
  res[Nh].z = floatToShort(g[6]);
  res[Nh].w = floatToShort(g[7]);
}

inline void packD12D(double2 *res, double *g) {
  for (int j=0; j<6; j++) {
    res[j*Nh].x = g[j*2+0]; 
    res[j*Nh].y = g[j*2+1]; 
  }
}

inline void packS12D(float4 *res, double *g) {
  for (int j=0; j<3; j++) {
    res[j*Nh].x = (float)g[j*4+0]; 
    res[j*Nh].y = (float)g[j*4+1]; 
    res[j*Nh].z = (float)g[j*4+2]; 
    res[j*Nh].w = (float)g[j*4+3];
  }
}

// sse packing routine
inline void packS12S(float4 *res, float *g) {
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

inline void packH12D(short4 *res, double *g) {
  for (int j=0; j<3; j++) {
    res[j*Nh].x = doubleToShort(g[j*4+0]);
    res[j*Nh].y = doubleToShort(g[j*4+1]);
    res[j*Nh].z = doubleToShort(g[j*4+2]);
    res[j*Nh].w = doubleToShort(g[j*4+3]);
  }
}

inline void packH12S(short4 *res, float *g) {
  for (int j=0; j<3; j++) {
    res[j*Nh].x = floatToShort(g[j*4+0]);
    res[j*Nh].y = floatToShort(g[j*4+1]);
    res[j*Nh].z = floatToShort(g[j*4+2]);
    res[j*Nh].w = floatToShort(g[j*4+3]);
  }
}

// Assume the gauge field is "QDP" ordered directions inside of
// space-time column-row ordering even-odd space-time
void packQDPGaugeFieldDD(double2 *res, double **gauge, int oddBit, ReconstructType reconstruct) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      double2 *r = res + dir*6*Nh;
      for (int i = 0; i < Nh; i++) {
	packD12D(r+i, g+i*18);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      double2 *r = res + dir*4*Nh;
      for (int i = 0; i < Nh; i++) {
	packD8D(r+i, g+i*18);
      }
    }
  }
}

void packQDPGaugeFieldSD(float4 *res, double **gauge, int oddBit, ReconstructType reconstruct) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      float4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	packS12D(r+i, g+i*18);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      float4 *r = res + dir*2*Nh;
      for (int i = 0; i < Nh; i++) {
	packS8D(r+i, g+i*18);
      }
    }
  }
}

void packQDPGaugeFieldHD(short4 *res, double **gauge, int oddBit, ReconstructType reconstruct) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      short4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	packH12D(r+i, g+i*18);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      short4 *r = res + dir*2*Nh;
      for (int i = 0; i < Nh; i++) {
	packH8D(r+i, g+i*18);
      }
    }
  }
}

// Single precision version of the above
void packQDPGaugeFieldSS(float4 *res, float **gauge, int oddBit, ReconstructType reconstruct) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {      
      float *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      float4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	packS12S(r+i, g+i*18);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      float *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      float4 *r = res + dir*2*Nh;
      for (int i = 0; i < Nh; i++) {
	packS8S(r+i, g+i*18);
      }
    }
  }
}

// Cuda half precision version of the above
void packQDPGaugeFieldHS(short4 *res, float **gauge, int oddBit, ReconstructType reconstruct) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      float *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      short4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	packH12S(r+i, g+i*18);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      float *g = gauge[dir] + oddBit*Nh*gaugeSiteSize;
      short4 *r = res + dir*2*Nh;
      for (int i = 0; i < Nh; i++) {
	packH8S(r+i, g+i*18);
      }
    }
  }
}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
void packCPSGaugeFieldDD(double2 *res, double *gauge, int oddBit, ReconstructType reconstruct) {
  double gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      double2 *r = res + dir*6*Nh;
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<2; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = g[4*i*18 + (jc*3+ic)*2+r];      
	packD12D(r+i, gT);
      }
    } 
  } else {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      double2 *r = res + dir*4*Nh;
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<3; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = g[4*i*18 + (jc*3+ic)*2+r];      
	packD8D(r+i, gT);
      }
    }
  }

}

void packCPSGaugeFieldSD(float4 *res, double *gauge, int oddBit, ReconstructType reconstruct) {
  float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      float4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<2; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = (float)g[4*i*18 + (jc*3+ic)*2+r];      
	packS12S(r+i, gT);
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
	packS8S(r+i, gT);
      }
    }
  }

}

void packCPSGaugeFieldHD(short4 *res, double *gauge, int oddBit, ReconstructType reconstruct) {
  float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      double *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      short4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<2; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = (float)g[4*i*18 + (jc*3+ic)*2+r];      
	packH12S(r+i, gT);
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
	packH8S(r+i, gT);
      }
    }
  }

}

// Single precision version of the above
void packCPSGaugeFieldSS(float4 *res, float *gauge, int oddBit, ReconstructType reconstruct) {
  float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      float *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      float4 *r = res + dir*3*Nh;
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<2; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = g[4*i*18 + (jc*3+ic)*2+r];
	packS12S(r+i, gT);
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
	packS8S(r+i, gT);
      }
    }
  }

}

// Cuda half precision, single precision CPS field
void packCPSGaugeFieldHS(short4 *res, float *gauge, int oddBit, ReconstructType reconstruct) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      float *g = gauge + (oddBit*Nh*4+dir)*gaugeSiteSize;
      float gT[12];
      short4 *r = res + dir*3*Nh;      
      for (int i = 0; i < Nh; i++) {
	// Must reorder rows-columns
	for (int ic=0; ic<2; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	      gT[(ic*3+jc)*2+r] = g[4*i*18 + (jc*3+ic)*2+r];
	packH12S(r+i, gT);
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
	packH8S(r+i, gT);
      }
    }
  }

}

void allocateGaugeField(FullGauge *cudaGauge, ReconstructType reconstruct, Precision precision) {

  cudaGauge->reconstruct = reconstruct;
  cudaGauge->precision = precision;

  int floatSize;
  if (precision == QUDA_DOUBLE_PRECISION) floatSize = sizeof(double);
  else if (precision == QUDA_SINGLE_PRECISION) floatSize = sizeof(float);
  else floatSize = sizeof(float)/2;

  int elements = (reconstruct == QUDA_RECONSTRUCT_8) ? 8 : 12;
  cudaGauge->packedGaugeBytes = 4*Nh*elements*floatSize;

  if (!cudaGauge->even) {
    if (cudaMalloc((void **)&cudaGauge->even, cudaGauge->packedGaugeBytes) == cudaErrorMemoryAllocation) {
      printf("Error allocating even gauge field\n");
      exit(0);
    }
  }
   
  if (!cudaGauge->odd) {
    if (cudaMalloc((void **)&cudaGauge->odd, cudaGauge->packedGaugeBytes) == cudaErrorMemoryAllocation) {
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

void loadGaugeField(FullGauge *cudaGauge, void *cpuGauge) {

  if (cudaGauge->precision == QUDA_DOUBLE_PRECISION) {
    if (gauge_param->cpu_prec != QUDA_DOUBLE_PRECISION) {
      printf("Error: can only create a double GPU gauge field from a double CPU gauge field\n");
      exit(-1);
    }
    
    // Use pinned memory
    double2 *packedEven, *packedOdd;
    
#ifndef __DEVICE_EMULATION__
    cudaMallocHost((void**)&packedEven, cudaGauge->packedGaugeBytes);
    cudaMallocHost((void**)&packedOdd, cudaGauge->packedGaugeBytes);
#else
    packedEven = (double2*)malloc(cudaGauge->packedGaugeBytes);
    packedOdd = (double2*)malloc(cudaGauge->packedGaugeBytes);
#endif
    
    if (gauge_param->gauge_order == QUDA_QDP_GAUGE_ORDER) {
      packQDPGaugeFieldDD(packedEven, (double**)cpuGauge, 0, cudaGauge->reconstruct);
      packQDPGaugeFieldDD(packedOdd,  (double**)cpuGauge, 1, cudaGauge->reconstruct);
    } else if (gauge_param->gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
      packCPSGaugeFieldDD(packedEven, (double*)cpuGauge, 0, cudaGauge->reconstruct);
      packCPSGaugeFieldDD(packedOdd,  (double*)cpuGauge, 1, cudaGauge->reconstruct);
    } else {
      printf("Sorry, %d GaugeFieldOrder not supported\n", gauge_param->gauge_order);
      exit(-1);
    }
    
    cudaMemcpy(cudaGauge->even, packedEven, cudaGauge->packedGaugeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaGauge->odd,  packedOdd,  cudaGauge->packedGaugeBytes, cudaMemcpyHostToDevice);    
    
#ifndef __DEVICE_EMULATION__
    cudaFreeHost(packedEven);
    cudaFreeHost(packedOdd);
#else
    free(packedEven);
    free(packedOdd);
#endif
  } else if (cudaGauge->precision == QUDA_SINGLE_PRECISION) {
    // Use pinned memory
    float4 *packedEven, *packedOdd;
    
#ifndef __DEVICE_EMULATION__
    cudaMallocHost((void**)&packedEven, cudaGauge->packedGaugeBytes);
    cudaMallocHost((void**)&packedOdd, cudaGauge->packedGaugeBytes);
#else
    packedEven = (float4*)malloc(cudaGauge->packedGaugeBytes);
    packedOdd = (float4*)malloc(cudaGauge->packedGaugeBytes);
#endif
    
    if (gauge_param->gauge_order == QUDA_QDP_GAUGE_ORDER) {
      if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION) {
	packQDPGaugeFieldSD(packedEven, (double**)cpuGauge, 0, cudaGauge->reconstruct);
	packQDPGaugeFieldSD(packedOdd,  (double**)cpuGauge, 1, cudaGauge->reconstruct);
      } else {
	packQDPGaugeFieldSS(packedEven, (float**)cpuGauge, 0, cudaGauge->reconstruct);
	packQDPGaugeFieldSS(packedOdd,  (float**)cpuGauge, 1, cudaGauge->reconstruct);
      }
    } else if (gauge_param->gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
      if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION) {
	packCPSGaugeFieldSD(packedEven, (double*)cpuGauge, 0, cudaGauge->reconstruct);
	packCPSGaugeFieldSD(packedOdd,  (double*)cpuGauge, 1, cudaGauge->reconstruct);
      } else {
	packCPSGaugeFieldSS(packedEven, (float*)cpuGauge, 0, cudaGauge->reconstruct);
	packCPSGaugeFieldSS(packedOdd,  (float*)cpuGauge, 1, cudaGauge->reconstruct);
      }
    } else {
      printf("Sorry, %d GaugeFieldOrder not supported\n", gauge_param->gauge_order);
      exit(-1);
    }

    cudaMemcpy(cudaGauge->even, packedEven, cudaGauge->packedGaugeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaGauge->odd,  packedOdd,  cudaGauge->packedGaugeBytes, cudaMemcpyHostToDevice);    
    
#ifndef __DEVICE_EMULATION__
    cudaFreeHost(packedEven);
    cudaFreeHost(packedOdd);
#else
    free(packedEven);
    free(packedOdd);
#endif
  } else {
    // Use pinned memory
    short4 *packedEven, *packedOdd;
#ifndef __DEVICE_EMULATION__
    cudaMallocHost((void**)&packedEven, cudaGauge->packedGaugeBytes);
    cudaMallocHost((void**)&packedOdd, cudaGauge->packedGaugeBytes);
#else
    packedEven = (short4*)malloc(cudaGauge->packedGaugeBytes);
    packedOdd = (short4*)malloc(cudaGauge->packedGaugeBytes);
#endif
    
    if (gauge_param->gauge_order == QUDA_QDP_GAUGE_ORDER) {
      if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION) {
	packQDPGaugeFieldHD(packedEven, (double**)cpuGauge, 0, cudaGauge->reconstruct);
	packQDPGaugeFieldHD(packedOdd,  (double**)cpuGauge, 1, cudaGauge->reconstruct);
      } else {
	packQDPGaugeFieldHS(packedEven, (float**)cpuGauge, 0, cudaGauge->reconstruct);
	packQDPGaugeFieldHS(packedOdd,  (float**)cpuGauge, 1, cudaGauge->reconstruct);
      }
    } else if (gauge_param->gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
      if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION) {
	packCPSGaugeFieldHD(packedEven, (double*)cpuGauge, 0, cudaGauge->reconstruct);
	packCPSGaugeFieldHD(packedOdd,  (double*)cpuGauge, 1, cudaGauge->reconstruct);
      } else {
	packCPSGaugeFieldHS(packedEven, (float*)cpuGauge, 0, cudaGauge->reconstruct);
	packCPSGaugeFieldHS(packedOdd,  (float*)cpuGauge, 1, cudaGauge->reconstruct);
      }
    } else {
      printf("Sorry, %d GaugeFieldOrder not supported\n", gauge_param->gauge_order);
      exit(-1);
    }

    cudaMemcpy(cudaGauge->even, packedEven, cudaGauge->packedGaugeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaGauge->odd,  packedOdd,  cudaGauge->packedGaugeBytes, cudaMemcpyHostToDevice);    

#ifndef __DEVICE_EMULATION__
    cudaFreeHost(packedEven);
    cudaFreeHost(packedOdd);
#else
    free(packedEven);
    free(packedOdd);
#endif

  }

}

void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, ReconstructType reconstruct, Precision precision) {

  if (gauge_param->cpu_prec == QUDA_HALF_PRECISION) {
    printf("QUDA error: half precision not supported on cpu\n");
    exit(-1);
  }

  if (reconstruct == QUDA_RECONSTRUCT_NO) {
    printf("QUDA error: ReconstructType not yet supported\n");
    exit(-1);
  }

  allocateGaugeField(cudaGauge, reconstruct, precision);
  loadGaugeField(cudaGauge, cpuGauge);

}
