#include <stdlib.h>
#include <stdio.h>

#include <quda.h>
#include <spinor_quda.h>

#include <xmmintrin.h>

// GPU clover matrix
FullClover cudaClover;

// Pinned memory for cpu-gpu memory copying
void *packedSpinor1 = 0;
void *packedSpinor2 = 0;

ParitySpinor allocateParitySpinor(int geometric_length, Precision precision) {
  ParitySpinor ret;

  ret.precision = precision;
  ret.length = geometric_length*spinorSiteSize;

  if (precision == QUDA_DOUBLE_PRECISION) {
    int spinor_bytes = ret.length*sizeof(double);
    if (cudaMalloc((void**)&ret.spinor, spinor_bytes) == cudaErrorMemoryAllocation) {
      printf("Error allocating spinor\n");
      exit(0);
    }
  } else if (precision == QUDA_SINGLE_PRECISION) {
    int spinor_bytes = ret.length*sizeof(float);
    if (cudaMalloc((void**)&ret.spinor, spinor_bytes) == cudaErrorMemoryAllocation) {
      printf("Error allocating spinor\n");
      exit(0);
    }
  } else if (precision == QUDA_HALF_PRECISION) {
    int spinor_bytes = ret.length*sizeof(float)/2;
    if (cudaMalloc((void**)&ret.spinor, spinor_bytes) == cudaErrorMemoryAllocation) {
      printf("Error allocating spinor\n");
      exit(0);
    }
    if (cudaMalloc((void**)&ret.spinorNorm, spinor_bytes/12) == cudaErrorMemoryAllocation) {
      printf("Error allocating spinorNorm\n");
      exit(0);
    }
  }

  return ret;
}


FullSpinor allocateSpinorField(int length, Precision precision) {
  FullSpinor ret;
  ret.even = allocateParitySpinor(length/2, precision);
  ret.odd = allocateParitySpinor(length/2, precision);
  return ret;
}

ParityClover allocateParityClover() {
  ParityClover ret;

  if (cudaMalloc((void**)&ret, CLOVER_BYTES) == cudaErrorMemoryAllocation) {
    printf("Error allocating clover term\n");
    exit(0);
  }   
  return ret;
}


FullClover allocateCloverField() {
  FullClover ret;
  ret.even = allocateParityClover();
  ret.odd = allocateParityClover();
  return ret;
}

void freeParitySpinor(ParitySpinor spinor) {

  cudaFree(spinor.spinor);
  if (spinor.precision == QUDA_HALF_PRECISION) cudaFree(spinor.spinorNorm);

  spinor.spinor = NULL;
  spinor.spinorNorm = NULL;
}

void freeParityClover(ParityClover clover) {
  cudaFree(clover);
}

void freeSpinorField(FullSpinor spinor) {
  freeParitySpinor(spinor.even);
  freeParitySpinor(spinor.odd);
}

void freeCloverField(FullClover clover) {
  cudaFree(clover.even);
  cudaFree(clover.odd);
}

void freeSpinorBuffer() {
#ifndef __DEVICE_EMULATION__
  cudaFreeHost(packedSpinor1);
#else
  free(packedSpinor1);
#endif
  packedSpinor1 = NULL;
}

inline void packDouble2(double2* a, double *b) {
  a->x = b[0]; a->y = b[1];
}

inline void unpackDouble2(double *a, double2 *b) {
  a[0] = b->x; a[1] = b->y;
}

inline void packFloat4(float4* a, float *b) {
  __m128 SSEtmp;
  SSEtmp = _mm_loadu_ps((const float*)b);
  _mm_storeu_ps((float*)a, SSEtmp);
  //a->x = b[0]; a->y = b[1]; a->z = b[2]; a->w = b[3];
}

inline void unpackFloat4(float *a, float4 *b) {
  __m128 SSEtmp;
  SSEtmp = _mm_load_ps((const float*)b);
  _mm_storeu_ps((float*)a, SSEtmp);
  //a[0] = b->x; a[1] = b->y; a[2] = b->z; a[3] = b->w;
}

template <typename Float>
inline void packSpinorVector(float4* a, Float *b) {
  Float K = 1.0 / 2.0;

  a[0*Nh].x = K*(b[1*6+0*2+0]+b[3*6+0*2+0]);
  a[0*Nh].y = K*(b[1*6+0*2+1]+b[3*6+0*2+1]);
  a[0*Nh].z = K*(b[1*6+1*2+0]+b[3*6+1*2+0]);
  a[0*Nh].w = K*(b[1*6+1*2+1]+b[3*6+1*2+1]);

  a[1*Nh].x = K*(b[1*6+2*2+0]+b[3*6+2*2+0]);
  a[1*Nh].y = K*(b[1*6+2*2+1]+b[3*6+2*2+1]);
  a[1*Nh].z = -K*(b[0*6+0*2+0]+b[2*6+0*2+0]);
  a[1*Nh].w = -K*(b[0*6+0*2+1]+b[2*6+0*2+1]);
  
  a[2*Nh].x = -K*(b[0*6+1*2+0]+b[2*6+1*2+0]);
  a[2*Nh].y = -K*(b[0*6+1*2+1]+b[2*6+1*2+1]);
  a[2*Nh].z = -K*(b[0*6+2*2+0]+b[2*6+2*2+0]);
  a[2*Nh].w = -K*(b[0*6+2*2+1]+b[2*6+2*2+1]);

  a[3*Nh].x = K*(b[1*6+0*2+0]-b[3*6+0*2+0]);
  a[3*Nh].y = K*(b[1*6+0*2+1]-b[3*6+0*2+1]);
  a[3*Nh].z = K*(b[1*6+1*2+0]-b[3*6+1*2+0]);
  a[3*Nh].w = K*(b[1*6+1*2+1]-b[3*6+1*2+1]);

  a[4*Nh].x = K*(b[1*6+2*2+0]-b[3*6+2*2+0]);
  a[4*Nh].y = K*(b[1*6+2*2+1]-b[3*6+2*2+1]);
  a[4*Nh].z = K*(b[2*6+0*2+0]-b[0*6+0*2+0]);
  a[4*Nh].w = K*(b[2*6+0*2+1]-b[0*6+0*2+1]);

  a[5*Nh].x = K*(b[2*6+1*2+0]-b[0*6+1*2+0]);
  a[5*Nh].y = K*(b[2*6+1*2+1]-b[0*6+1*2+1]);
  a[5*Nh].z = K*(b[2*6+2*2+0]-b[0*6+2*2+0]);
  a[5*Nh].w = K*(b[2*6+2*2+1]-b[0*6+2*2+1]);
    
}

template <typename Float>
inline void packQDPSpinorVector(float4* a, Float *b) {
  Float K = 1.0 / 2.0;

  a[0*Nh].x = K*(b[(0*4+1)*2+0]+b[(0*4+3)*2+0]);
  a[0*Nh].y = K*(b[(0*4+1)*2+1]+b[(0*4+3)*2+1]);
  a[0*Nh].z = K*(b[(1*4+1)*2+0]+b[(1*4+3)*2+0]);
  a[0*Nh].w = K*(b[(1*4+1)*2+1]+b[(1*4+3)*2+1]);

  a[1*Nh].x = K*(b[(2*4+1)*2+0]+b[(2*4+3)*2+0]);
  a[1*Nh].y = K*(b[(2*4+1)*2+1]+b[(2*4+3)*2+1]);
  a[1*Nh].z = -K*(b[(0*4+0)*2+0]+b[(0*4+2)*2+0]);
  a[1*Nh].w = -K*(b[(0*4+0)*2+1]+b[(0*4+2)*2+1]);

  a[2*Nh].x = -K*(b[(1*4+0)*2+0]+b[(1*4+2)*2+0]);
  a[2*Nh].y = -K*(b[(1*4+0)*2+1]+b[(1*4+2)*2+1]);
  a[2*Nh].z = -K*(b[(2*4+0)*2+0]+b[(2*4+2)*2+0]);
  a[2*Nh].w = -K*(b[(2*4+0)*2+1]+b[(2*4+2)*2+1]);

  a[3*Nh].x = K*(b[(0*4+1)*2+0]+b[(0*4+3)*2+0]);
  a[3*Nh].y = K*(b[(0*4+1)*2+1]+b[(0*4+3)*2+1]);
  a[3*Nh].z = K*(b[(1*4+1)*2+0]+b[(1*4+3)*2+0]);
  a[3*Nh].w = K*(b[(1*4+1)*2+1]+b[(1*4+3)*2+1]);

  a[4*Nh].x = K*(b[(2*4+1)*2+0]+b[(2*4+3)*2+0]);
  a[4*Nh].y = K*(b[(2*4+1)*2+1]+b[(2*4+3)*2+1]);
  a[4*Nh].z = K*(b[(0*4+2)*2+0]+b[(0*4+0)*2+0]);
  a[4*Nh].w = K*(b[(0*4+2)*2+1]+b[(0*4+0)*2+1]);

  a[5*Nh].x = K*(b[(1*4+2)*2+0]+b[(1*4+0)*2+0]);
  a[5*Nh].y = K*(b[(1*4+2)*2+1]+b[(1*4+0)*2+1]);
  a[5*Nh].z = K*(b[(2*4+2)*2+0]+b[(2*4+0)*2+0]);
  a[5*Nh].w = K*(b[(2*4+2)*2+1]+b[(2*4+0)*2+1]);
}

template <typename Float>
inline void packSpinorVector(double2* a, Float *b) {
  Float K = 1.0 / 2.0;

  for (int c=0; c<3; c++) {
    a[c*Nh].x = K*(b[1*6+c*2+0]+b[3*6+c*2+0]);
    a[c*Nh].y = K*(b[1*6+c*2+1]+b[3*6+c*2+1]);

    a[(3+c)*Nh].x = -K*(b[0*6+c*2+0]+b[2*6+c*2+0]);
    a[(3+c)*Nh].y = -K*(b[0*6+c*2+1]+b[2*6+c*2+1]);

    a[(6+c)*Nh].x = K*(b[1*6+c*2+0]-b[3*6+c*2+0]);
    a[(6+c)*Nh].y = K*(b[1*6+c*2+1]-b[3*6+c*2+1]);

    a[(9+c)*Nh].x = K*(b[2*6+c*2+0]-b[0*6+c*2+0]);
    a[(9+c)*Nh].y = K*(b[2*6+c*2+1]-b[0*6+c*2+1]);
  }

}

template <typename Float>
inline void packQDPSpinorVector(double2* a, Float *b) {
  Float K = 1.0 / 2.0;

  for (int c=0; c<3; c++) {
    a[c*Nh].x = K*(b[(c*4+1)*2+0]+b[(c*4+3)*2+0]);
    a[c*Nh].y = K*(b[(c*4+1)*2+1]+b[(c*4+3)*2+1]);

    a[(3+c)*Nh].x = -K*(b[(c*4+0)*2+0]+b[(c*4+2)*2+0]);
    a[(3+c)*Nh].y = -K*(b[(c*4+0)*2+1]+b[(c*4+2)*2+1]);

    a[(6+c)*Nh].x = K*(b[(c*4+1)*2+0]-b[(c*4+3)*2+0]);
    a[(6+c)*Nh].y = K*(b[(c*4+1)*2+1]-b[(c*4+3)*2+1]);

    a[(9+c)*Nh].x = K*(b[(c*4+2)*2+0]-b[(c*4+0)*2+0]);
    a[(9+c)*Nh].y = K*(b[(c*4+2)*2+1]-b[(c*4+0)*2+1]);
  }

}

template <typename Float>
inline void unpackSpinorVector(Float *a, float4 *b) {
  Float K = 1.0;

  a[0*6+0*2+0] = -K*(b[Nh].z+b[4*Nh].z);
  a[0*6+0*2+1] = -K*(b[Nh].w+b[4*Nh].w);
  a[0*6+1*2+0] = -K*(b[2*Nh].x+b[5*Nh].x);
  a[0*6+1*2+1] = -K*(b[2*Nh].y+b[5*Nh].y);
  a[0*6+2*2+0] = -K*(b[2*Nh].z+b[5*Nh].z);
  a[0*6+2*2+1] = -K*(b[2*Nh].w+b[5*Nh].w);
  
  a[1*6+0*2+0] = K*(b[0].x+b[3*Nh].x);
  a[1*6+0*2+1] = K*(b[0].y+b[3*Nh].y);
  a[1*6+1*2+0] = K*(b[0].z+b[3*Nh].z);
  a[1*6+1*2+1] = K*(b[0].w+b[3*Nh].w);  
  a[1*6+2*2+0] = K*(b[Nh].x+b[4*Nh].x);
  a[1*6+2*2+1] = K*(b[Nh].y+b[4*Nh].y);
  
  a[2*6+0*2+0] = -K*(b[Nh].z-b[4*Nh].z);
  a[2*6+0*2+1] = -K*(b[Nh].w-b[4*Nh].w);
  a[2*6+1*2+0] = -K*(b[2*Nh].x-b[5*Nh].x);
  a[2*6+1*2+1] = -K*(b[2*Nh].y-b[5*Nh].y);
  a[2*6+2*2+0] = -K*(b[2*Nh].z-b[5*Nh].z);
  a[2*6+2*2+1] = -K*(b[2*Nh].w-b[5*Nh].w);
  
  a[3*6+0*2+0] = -K*(b[3*Nh].x-b[0].x);
  a[3*6+0*2+1] = -K*(b[3*Nh].y-b[0].y);
  a[3*6+1*2+0] = -K*(b[3*Nh].z-b[0].z);
  a[3*6+1*2+1] = -K*(b[3*Nh].w-b[0].w);
  a[3*6+2*2+0] = -K*(b[4*Nh].x-b[Nh].x);
  a[3*6+2*2+1] = -K*(b[4*Nh].y-b[Nh].y);
}

template <typename Float>
inline void unpackQDPSpinorVector(Float *a, float4 *b) {
  Float K = 1.0;

  a[(0*4+0)*2+0] = -K*(b[Nh].z+b[4*Nh].z);
  a[(0*4+0)*2+1] = -K*(b[Nh].w+b[4*Nh].w);
  a[(1*4+0)*2+0] = -K*(b[2*Nh].x+b[5*Nh].x);
  a[(1*4+0)*2+1] = -K*(b[2*Nh].y+b[5*Nh].y);
  a[(2*4+0)*2+0] = -K*(b[2*Nh].z+b[5*Nh].z);
  a[(2*4+0)*2+1] = -K*(b[2*Nh].w+b[5*Nh].w);
  
  a[(0*4+1)*2+0] = K*(b[0].x+b[3*Nh].x);
  a[(0*4+1)*2+1] = K*(b[0].y+b[3*Nh].y);
  a[(1*4+1)*2+0] = K*(b[0].z+b[3*Nh].z);
  a[(1*4+1)*2+1] = K*(b[0].w+b[3*Nh].w);  
  a[(2*4+1)*2+0] = K*(b[Nh].x+b[4*Nh].x);
  a[(2*4+1)*2+1] = K*(b[Nh].y+b[4*Nh].y);
  
  a[(0*4+2)*2+0] = -K*(b[Nh].z-b[4*Nh].z);
  a[(0*4+2)*2+1] = -K*(b[Nh].w-b[4*Nh].w);
  a[(1*4+2)*2+0] = -K*(b[2*Nh].x-b[5*Nh].x);
  a[(1*4+2)*2+1] = -K*(b[2*Nh].y-b[5*Nh].y);
  a[(2*4+2)*2+0] = -K*(b[2*Nh].z-b[5*Nh].z);
  a[(2*4+2)*2+1] = -K*(b[2*Nh].w-b[5*Nh].w);
  
  a[(0*4+3)*2+0] = -K*(b[3*Nh].x-b[0].x);
  a[(0*4+3)*2+1] = -K*(b[3*Nh].y-b[0].y);
  a[(1*4+3)*2+0] = -K*(b[3*Nh].z-b[0].z);
  a[(1*4+3)*2+1] = -K*(b[3*Nh].w-b[0].w);
  a[(2*4+3)*2+0] = -K*(b[4*Nh].x-b[Nh].x);
  a[(2*4+3)*2+1] = -K*(b[4*Nh].y-b[Nh].y);
}

template <typename Float>
inline void unpackSpinorVector(Float *a, double2 *b) {
  Float K = 1.0;

  for (int c=0; c<3; c++) {
    a[0*6+c*2+0] = -K*(b[(3+c)*Nh].x+b[(9+c)*Nh].x);
    a[0*6+c*2+1] = -K*(b[(3+c)*Nh].y+b[(9+c)*Nh].y);

    a[1*6+c*2+0] = K*(b[c*Nh].x+b[(6+c)*Nh].x);
    a[1*6+c*2+1] = K*(b[c*Nh].y+b[(6+c)*Nh].y);

    a[2*6+c*2+0] = -K*(b[(3+c)*Nh].x-b[(9+c)*Nh].x);
    a[2*6+c*2+1] = -K*(b[(3+c)*Nh].y-b[(9+c)*Nh].y);
    
    a[3*6+c*2+0] = -K*(b[(6+c)*Nh].x-b[c*Nh].x);
    a[3*6+c*2+1] = -K*(b[(6+c)*Nh].y-b[c*Nh].y);
  }

}

template <typename Float>
inline void unpackQDPSpinorVector(Float *a, double2 *b) {
  Float K = 1.0;

  for (int c=0; c<3; c++) {
    a[(c*4+0)*2+0] = -K*(b[(3+c)*Nh].x+b[(9+c)*Nh].x);
    a[(c*4+0)*2+1] = -K*(b[(3+c)*Nh].y+b[(9+c)*Nh].y);

    a[(c*4+1)*2+0] = K*(b[c*Nh].x+b[(6+c)*Nh].x);
    a[(c*4+1)*2+1] = K*(b[c*Nh].y+b[(6+c)*Nh].y);

    a[(c*4+2)*2+0] = -K*(b[(3+c)*Nh].x-b[(9+c)*Nh].x);
    a[(c*4+2)*2+1] = -K*(b[(3+c)*Nh].y-b[(9+c)*Nh].y);
    
    a[(c*4+3)*2+0] = -K*(b[(6+c)*Nh].x-b[c*Nh].x);
    a[(c*4+3)*2+1] = -K*(b[(6+c)*Nh].y-b[c*Nh].y);
  }

}

// Standard spinor packing, colour inside spin
template <typename Float, typename FloatN>
void packParitySpinor(FloatN *res, Float *spinor) {
  for (int i = 0; i < Nh; i++) {
    packSpinorVector(res+i, spinor+24*i);
  }
}

template <typename Float, typename FloatN>
void packFullSpinor(FloatN *even, FloatN *odd, Float *spinor) {

  for (int i=0; i<Nh; i++) {

    int boundaryCrossings = i/L1h + i/(L2*L1h) + i/(L3*L2*L1h);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      packSpinorVector(even+i, spinor+24*k);
    }
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      packSpinorVector(odd+i, spinor+24*k);
    }
  }

}

template <typename Float, typename FloatN>
void unpackFullSpinor(Float *res, FloatN *even, FloatN *odd) {

  for (int i=0; i<Nh; i++) {

    int boundaryCrossings = i/L1h + i/(L2*L1h) + i/(L3*L2*L1h);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      unpackSpinorVector(res+24*k, even+i);
    }
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      unpackSpinorVector(res+24*k, odd+i);
    }
  }

}


template <typename Float, typename FloatN>
void unpackParitySpinor(Float *res, FloatN *spinorPacked) {

  for (int i = 0; i < Nh; i++) {
    unpackSpinorVector(res+i*24, spinorPacked+i);
  }

}

// QDP spinor packing, spin inside colour
template <typename Float, typename FloatN>
void packQDPParitySpinor(FloatN *res, Float *spinor) {
  for (int i = 0; i < Nh; i++) {
    packQDPSpinorVector(res+i, spinor+i*24);
  }
}

// QDP spinor packing, spin inside colour
template <typename Float, typename FloatN>
void unpackQDPParitySpinor(Float *res, FloatN *spinor) {
  for (int i = 0; i < Nh; i++) {
    unpackQDPSpinorVector(res+i*24, spinor+i);
  }
}

void loadParitySpinor(ParitySpinor ret, void *spinor, Precision cpu_prec, 
		      DiracFieldOrder dirac_order) {

  if (ret.precision == QUDA_DOUBLE_PRECISION && cpu_prec != QUDA_DOUBLE_PRECISION) {
    printf("Error, cannot have CUDA double precision without double CPU precision\n");
    exit(-1);
  }

  if (ret.precision != QUDA_HALF_PRECISION) {
    size_t spinor_bytes;
    if (ret.precision == QUDA_DOUBLE_PRECISION) spinor_bytes = Nh*spinorSiteSize*sizeof(double);
    else spinor_bytes = Nh*spinorSiteSize*sizeof(float);

#ifndef __DEVICE_EMULATION__
    if (!packedSpinor1) cudaMallocHost(&packedSpinor1, spinor_bytes);
#else
    if (!packedSpinor1) packedSpinor1 = malloc(spinor_bytes);
#endif
    
    if (dirac_order == QUDA_DIRAC_ORDER || QUDA_CPS_WILSON_DIRAC_ORDER) {
      if (ret.precision == QUDA_DOUBLE_PRECISION) {
	packParitySpinor((double2*)packedSpinor1, (double*)spinor);
      } else {
	if (cpu_prec == QUDA_DOUBLE_PRECISION) packParitySpinor((float4*)packedSpinor1, (double*)spinor);
	else packParitySpinor((float4*)packedSpinor1, (float*)spinor);
      }
    } else if (dirac_order == QUDA_QDP_DIRAC_ORDER) {
      if (ret.precision == QUDA_DOUBLE_PRECISION) {
	packQDPParitySpinor((double2*)packedSpinor1, (double*)spinor);
      } else {
	if (cpu_prec == QUDA_DOUBLE_PRECISION) packQDPParitySpinor((float4*)packedSpinor1, (double*)spinor);
	else packQDPParitySpinor((float4*)packedSpinor1, (float*)spinor);
      }
    }
    cudaMemcpy(ret.spinor, packedSpinor1, spinor_bytes, cudaMemcpyHostToDevice);
  } else {
    ParitySpinor tmp = allocateParitySpinor(ret.length/spinorSiteSize, QUDA_SINGLE_PRECISION);
    loadParitySpinor(tmp, spinor, cpu_prec, dirac_order);
    copyCuda(ret, tmp);
    freeParitySpinor(tmp);
  }

}

void loadFullSpinor(FullSpinor ret, void *spinor, Precision cpu_prec) {

  if (ret.even.precision != QUDA_HALF_PRECISION) {
    size_t spinor_bytes;
    if (ret.even.precision == QUDA_DOUBLE_PRECISION) spinor_bytes = Nh*spinorSiteSize*sizeof(double);
    else spinor_bytes = Nh*spinorSiteSize*sizeof(float);
    
#ifndef __DEVICE_EMULATION__
    if (!packedSpinor1) cudaMallocHost(&packedSpinor1, spinor_bytes);
    if (!packedSpinor2) cudaMallocHost(&packedSpinor2, spinor_bytes);
#else
    if (!packedSpinor1) packedSpinor1 = malloc(spinor_bytes);
    if (!packedSpinor2) packedSpinor2 = malloc(spinor_bytes);
#endif
    
    if (ret.even.precision == QUDA_DOUBLE_PRECISION) {
      packFullSpinor((double2*)packedSpinor1, (double2*)packedSpinor2, (double*)spinor);
    } else {
      if (cpu_prec == QUDA_DOUBLE_PRECISION) 
	packFullSpinor((float4*)packedSpinor1, (float4*)packedSpinor2, (double*)spinor);
      else 
	packFullSpinor((float4*)packedSpinor1, (float4*)packedSpinor2, (float*)spinor);
    }
    
    cudaMemcpy(ret.even.spinor, packedSpinor1, spinor_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ret.odd.spinor, packedSpinor2, spinor_bytes, cudaMemcpyHostToDevice);

#ifndef __DEVICE_EMULATION__
    cudaFreeHost(packedSpinor2);
#else
    free(packedSpinor2);
#endif
    packedSpinor2 = 0;
  } else {
    FullSpinor tmp = allocateSpinorField(2*ret.even.length/spinorSiteSize, QUDA_SINGLE_PRECISION);
    loadFullSpinor(tmp, spinor, cpu_prec);
    copyCuda(ret.even, tmp.even);
    copyCuda(ret.odd, tmp.odd);
    freeSpinorField(tmp);
  }

}

void loadSpinorField(FullSpinor ret, void *spinor, Precision cpu_prec, DiracFieldOrder dirac_order) {
  void *spinor_odd;
  if (cpu_prec == QUDA_SINGLE_PRECISION) spinor_odd = (float*)spinor + Nh*spinorSiteSize;
  else spinor_odd = (double*)spinor + Nh*spinorSiteSize;

  if (dirac_order == QUDA_LEX_DIRAC_ORDER) {
    loadFullSpinor(ret, spinor, cpu_prec);
  } else if (dirac_order == QUDA_DIRAC_ORDER || dirac_order == QUDA_QDP_DIRAC_ORDER) {
    loadParitySpinor(ret.even, spinor, cpu_prec, dirac_order);
    loadParitySpinor(ret.odd, spinor_odd, cpu_prec, dirac_order);
  } else if (dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    // odd-even so reverse order
    loadParitySpinor(ret.even, spinor_odd, cpu_prec, dirac_order);
    loadParitySpinor(ret.odd, spinor, cpu_prec, dirac_order);
  } else {
    printf("DiracFieldOrder %d not supported\n", dirac_order);
    exit(-1);
  }
}

void retrieveParitySpinor(void *res, ParitySpinor spinor, Precision cpu_prec, DiracFieldOrder dirac_order) {

  if (spinor.precision != QUDA_HALF_PRECISION) {
    size_t spinor_bytes;
    if (spinor.precision == QUDA_DOUBLE_PRECISION) spinor_bytes = Nh*spinorSiteSize*sizeof(double);
    else if (spinor.precision == QUDA_SINGLE_PRECISION) spinor_bytes = Nh*spinorSiteSize*sizeof(float);
    else spinor_bytes = Nh*spinorSiteSize*sizeof(float)/2;
    
    if (!packedSpinor1) cudaMallocHost((void**)&packedSpinor1, spinor_bytes);
    cudaMemcpy(packedSpinor1, spinor.spinor, spinor_bytes, cudaMemcpyDeviceToHost);
    if (dirac_order == QUDA_DIRAC_ORDER || QUDA_CPS_WILSON_DIRAC_ORDER) {
      if (spinor.precision == QUDA_DOUBLE_PRECISION) {
	unpackParitySpinor((double*)res, (double2*)packedSpinor1);
      } else {
	if (cpu_prec == QUDA_DOUBLE_PRECISION) unpackParitySpinor((double*)res, (float4*)packedSpinor1);
	else unpackParitySpinor((float*)res, (float4*)packedSpinor1);
      }
    } else if (dirac_order == QUDA_QDP_DIRAC_ORDER) {
      if (spinor.precision == QUDA_DOUBLE_PRECISION) {
	unpackQDPParitySpinor((double*)res, (double2*)packedSpinor1);
      } else {
	if (cpu_prec == QUDA_DOUBLE_PRECISION) unpackQDPParitySpinor((double*)res, (float4*)packedSpinor1);
	else unpackQDPParitySpinor((float*)res, (float4*)packedSpinor1);
      }
    }
  } else {
    ParitySpinor tmp = allocateParitySpinor(spinor.length/spinorSiteSize, QUDA_SINGLE_PRECISION);
    copyCuda(tmp, spinor);
    retrieveParitySpinor(res, tmp, cpu_prec, dirac_order);
    freeParitySpinor(tmp);
  }
}

void retrieveFullSpinor(void *res, FullSpinor spinor, Precision cpu_prec) {

  if (spinor.even.precision != QUDA_HALF_PRECISION) {
    size_t spinor_bytes;
    if (spinor.even.precision == QUDA_DOUBLE_PRECISION) spinor_bytes = Nh*spinorSiteSize*sizeof(double);
    else spinor_bytes = Nh*spinorSiteSize*sizeof(float);
    
    if (!packedSpinor1) cudaMallocHost((void**)&packedSpinor1, spinor_bytes);
    if (!packedSpinor2) cudaMallocHost((void**)&packedSpinor2, spinor_bytes);
    
    cudaMemcpy(packedSpinor1, spinor.even.spinor, spinor_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(packedSpinor2, spinor.odd.spinor, spinor_bytes, cudaMemcpyDeviceToHost);
    if (spinor.even.precision == QUDA_DOUBLE_PRECISION) {
      unpackFullSpinor((double*)res, (double2*)packedSpinor1, (double2*)packedSpinor2);
    } else {
      if (cpu_prec == QUDA_DOUBLE_PRECISION) 
	unpackFullSpinor((double*)res, (float4*)packedSpinor1, (float4*)packedSpinor2);
      else unpackFullSpinor((float*)res, (float4*)packedSpinor1, (float4*)packedSpinor2);
    }
    
#ifndef __DEVICE_EMULATION__
    cudaFreeHost(packedSpinor2);
#else
    free(packedSpinor2);
#endif
    packedSpinor2 = 0;
  } else {
    FullSpinor tmp = allocateSpinorField(2*spinor.even.length/spinorSiteSize, QUDA_SINGLE_PRECISION);
    copyCuda(tmp.even, spinor.even);
    copyCuda(tmp.odd, spinor.odd);
    retrieveFullSpinor(res, tmp, cpu_prec);
    freeSpinorField(tmp);
  }
}

void retrieveSpinorField(void *res, FullSpinor spinor, Precision cpu_prec, DiracFieldOrder dirac_order) {
  void *res_odd;
  if (cpu_prec == QUDA_SINGLE_PRECISION) res_odd = (float*)res + Nh*spinorSiteSize;
  else res_odd = (double*)res + Nh*spinorSiteSize;

  if (dirac_order == QUDA_LEX_DIRAC_ORDER) {
    retrieveFullSpinor(res, spinor, cpu_prec);
  } else if (dirac_order == QUDA_DIRAC_ORDER || dirac_order == QUDA_QDP_DIRAC_ORDER) {
    retrieveParitySpinor(res, spinor.even, cpu_prec, dirac_order);
    retrieveParitySpinor(res_odd, spinor.odd, cpu_prec, dirac_order);
  } else if (dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    retrieveParitySpinor(res, spinor.odd, cpu_prec, dirac_order);
    retrieveParitySpinor(res_odd, spinor.even, cpu_prec, dirac_order);
  } else {
    printf("DiracFieldOrder %d not supported\n", dirac_order);
    exit(-1);
  }
  
}

/*
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
*/
