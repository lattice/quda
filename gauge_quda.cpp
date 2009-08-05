#include <stdlib.h>
#include <stdio.h>

#include <quda.h>
#include <gauge_quda.h>

#include <xmmintrin.h>

#define SHORT_LENGTH 65536
#define SCALE_FLOAT (SHORT_LENGTH-1) / 2.f
#define SHIFT_FLOAT -1.f / (SHORT_LENGTH-1)

double Anisotropy;

template <typename Float>
inline short FloatToShort(Float a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

template <typename Float>
inline void pack8(double2 *res, Float *g, int dir, int V) {
  double2 *r = res + dir*4*V;
  r[0].x = atan2(g[1], g[0]);
  r[0].y = atan2(g[13], g[12]);
  r[V].x = g[2];
  r[V].y = g[3];
  r[2*V].x = g[4];
  r[2*V].y = g[5];
  r[3*V].x = g[6];
  r[3*V].y = g[7];
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
    r[j*V].x = (float)g[j*4+0]; 
    r[j*V].y = (float)g[j*4+1]; 
    r[j*V].z = (float)g[j*4+2]; 
    r[j*V].w = (float)g[j*4+3];
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

// Assume the gauge field is "QDP" ordered directions inside of
// space-time column-row ordering even-odd space-time
template <typename Float, typename FloatN>
void packQDPGaugeField(FloatN *res, Float **gauge, int oddBit, ReconstructType reconstruct, int V) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) pack12(res+i, g+i*18, dir, V);
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) pack8(res+i, g+i*18, dir, V);
    }
  }
}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
template <typename Float, typename FloatN>
void packCPSGaugeField(FloatN *res, Float *gauge, int oddBit, ReconstructType reconstruct, int V) {
  Float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	// Must reorder rows-columns and divide by anisotropy
	for (int ic=0; ic<2; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = g[4*i*18 + (jc*3+ic)*2+r] / Anisotropy;
	pack12(res+i, gT, dir, V);
      }
    } 
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	// Must reorder rows-columns and divide by anisotropy
	for (int ic=0; ic<3; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = g[4*i*18 + (jc*3+ic)*2+r] / Anisotropy;
	pack8(res+i, gT, dir, V);
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

void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, ReconstructType reconstruct, 
		      Precision precision, int *X, double anisotropy) {

  if (gauge_param->cpu_prec == QUDA_HALF_PRECISION) {
    printf("QUDA error: half precision not supported on cpu\n");
    exit(-1);
  }

  if (reconstruct == QUDA_RECONSTRUCT_NO) {
    printf("QUDA error: ReconstructType not yet supported\n");
    exit(-1);
  }

  if (precision == QUDA_DOUBLE_PRECISION && gauge_param->cpu_prec != QUDA_DOUBLE_PRECISION) {
    printf("Error: can only create a double GPU gauge field from a double CPU gauge field\n");
    exit(-1);
  }

  Anisotropy = anisotropy;

  cudaGauge->anisotropy = anisotropy;
  cudaGauge->volume = 1;
  for (int d=0; d<4; d++) {
    cudaGauge->X[d] = X[d];
    cudaGauge->volume *= X[d];
  }
  cudaGauge->X[0] /= 2; // actually store the even-odd sublattice dimensions
  cudaGauge->volume /= 2;

  allocateGaugeField(cudaGauge, reconstruct, precision);

  if (precision == QUDA_DOUBLE_PRECISION) {
    loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (double*)cpuGauge, 
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
