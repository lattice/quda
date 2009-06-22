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

// Half precision spinor field temporaries
ParitySpinor hSpinor1, hSpinor2;

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

void allocateSpinorHalf() {
  Precision precision = QUDA_HALF_PRECISION;
  hSpinor1 = allocateParitySpinor(Nh, precision);
  hSpinor2 = allocateParitySpinor(Nh, precision);
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

void freeSpinorHalf() {
  freeParitySpinor(hSpinor1);
  freeParitySpinor(hSpinor2);
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

void packFullSpinorDD(double2 *even, double2 *odd, double *spinor) {
  double K = 1.0 / 2.0;
  double b[24];

  for (int i=0; i<Nh; i++) {

    int boundaryCrossings = i/L1h + i/(L2*L1h) + i/(L3*L2*L1h);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      double *a = spinor + k*24;
      for (int c=0; c<3; c++) {
	for (int r=0; r<2; r++) {
	  int cr = c*2+r;
	  b[0*6+cr] = K*(a[1*6+cr]+a[3*6+cr]);
	  b[1*6+cr] = -K*(a[0*6+cr]+a[2*6+cr]);
	  b[2*6+cr] = K*(a[1*6+cr]-a[3*6+cr]);
	  b[3*6+cr] = K*(a[2*6+cr]-a[0*6+cr]);
	}
      }
      
      for (int j = 0; j < 12; j++) packDouble2(even+j*Nh+i, b+j*2);
    }
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      double *a = spinor + k*24;
      for (int c=0; c<3; c++) {
	for (int r=0; r<2; r++) {
	  int cr = c*2+r;
	  b[0*6+cr] = K*(a[1*6+cr]+a[3*6+cr]);
	  b[1*6+cr] = -K*(a[0*6+cr]+a[2*6+cr]);
	  b[2*6+cr] = K*(a[1*6+cr]-a[3*6+cr]);
	  b[3*6+cr] = K*(a[2*6+cr]-a[0*6+cr]);
	}
      }
      
      for (int j=0; j<12; j++) packDouble2(odd+j*Nh+i, b+j*2);
    }
  }

}

void packFullSpinorSD(float4 *even, float4 *odd, double *spinor) {
  double K = 1.0 / 2.0;
  float b[24];

  for (int i=0; i<Nh; i++) {

    int boundaryCrossings = i/L1h + i/(L2*L1h) + i/(L3*L2*L1h);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      double *a = spinor + k*24;
      for (int c=0; c<3; c++) {
	for (int r=0; r<2; r++) {
	  int cr = c*2+r;
	  b[0*6+cr] = K*(a[1*6+cr]+a[3*6+cr]);
	  b[1*6+cr] = -K*(a[0*6+cr]+a[2*6+cr]);
	  b[2*6+cr] = K*(a[1*6+cr]-a[3*6+cr]);
	  b[3*6+cr] = K*(a[2*6+cr]-a[0*6+cr]);
	}
      }
      
      for (int j=0; j<6; j++) packFloat4(even+j*Nh+i, b+j*4);
    }
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      double *a = spinor + k*24;
      for (int c=0; c<3; c++) {
	for (int r=0; r<2; r++) {
	  int cr = c*2+r;
	  b[0*6+cr] = K*(a[1*6+cr]+a[3*6+cr]);
	  b[1*6+cr] = -K*(a[0*6+cr]+a[2*6+cr]);
	  b[2*6+cr] = K*(a[1*6+cr]-a[3*6+cr]);
	  b[3*6+cr] = K*(a[2*6+cr]-a[0*6+cr]);
	}
      }
      
      for (int j=0; j<6; j++) packFloat4(odd+j*Nh+i, b+j*4);
    }
  }

}

void packFullSpinorSS(float4 *even, float4 *odd, float *spinor) {
  float K = 1.0 / 2.0;
  float b[24];

  for (int i=0; i<Nh; i++) {

    int boundaryCrossings = i/L1h + i/(L2*L1h) + i/(L3*L2*L1h);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      float *a = spinor + k*24;
      for (int c=0; c<3; c++) {
	for (int r=0; r<2; r++) {
	  int cr = c*2+r;
	  b[0*6+cr] = K*(a[1*6+cr]+a[3*6+cr]);
	  b[1*6+cr] = -K*(a[0*6+cr]+a[2*6+cr]);
	  b[2*6+cr] = K*(a[1*6+cr]-a[3*6+cr]);
	  b[3*6+cr] = K*(a[2*6+cr]-a[0*6+cr]);
	}
      }
      
      for (int j=0; j<6; j++) packFloat4(even+j*Nh+i, b+j*4);
    }
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      float *a = spinor + k*24;
      for (int c=0; c<3; c++) {
	for (int r=0; r<2; r++) {
	  int cr = c*2+r;
	  b[0*6+cr] = K*(a[1*6+cr]+a[3*6+cr]);
	  b[1*6+cr] = -K*(a[0*6+cr]+a[2*6+cr]);
	  b[2*6+cr] = K*(a[1*6+cr]-a[3*6+cr]);
	  b[3*6+cr] = K*(a[2*6+cr]-a[0*6+cr]);
	}
      }
      
      for (int j=0; j<6; j++) packFloat4(odd+j*Nh+i, b+j*4);
    }
  }

}

// Standard spinor packing, colour inside spin
void packParitySpinorDD(double2 *res, double *spinor) {
  double K = 1.0 / (2.0);
  double b[24];

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

    for (int j=0; j<12; j++) packDouble2(res+j*Nh+i, b+j*2);
  }
}

void packParitySpinorSD(float4 *res, double *spinor) {
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

    for (int j=0; j<6; j++) packFloat4(res+j*Nh+i, b+j*4);
  }
}

// single precision version of the above
void packParitySpinorSS(float4 *res, float *spinor) {
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
void packQDPParitySpinorDD(double2 *res, double *spinor) {
  double K = 1.0 / 2.0;
  
  double b[24];
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

    for (int j = 0; j < 6; j++) packDouble2(res+j*Nh+i, b+j*2);
  }
}

// QDP spinor packing, spin inside colour
void packQDPParitySpinorSD(float4 *res, double *spinor) {
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

// Single precision version of the above
void packQDPParitySpinorSS(float4 *res, float *spinor) {
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

void unpackFullSpinorDD(double *res, double2 *even, double2 *odd) {
  double K = 1.0;
  double b[24];

  for (int i=0; i<Nh; i++) {

    int boundaryCrossings = i/L1h + i/(L2*L1h) + i/(L3*L2*L1h);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      double *a = res + k*24;

      for (int j = 0; j < 12; j++) unpackDouble2(b+j*2, even+j*Nh+i);

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
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      double *a = res + k*24;

      for (int j = 0; j < 12; j++) unpackDouble2(b+j*2, odd+j*Nh+i);

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

}

void unpackFullSpinorDS(double *res, float4 *even, float4 *odd) {
  double K = 1.0;
  float b[24];

  for (int i=0; i<Nh; i++) {

    int boundaryCrossings = i/L1h + i/(L2*L1h) + i/(L3*L2*L1h);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      double *a = res + k*24;

      for (int j = 0; j < 6; j++) unpackFloat4(b+j*4, even+j*Nh+i);

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
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      double *a = res + k*24;

      for (int j = 0; j < 6; j++) unpackFloat4(b+j*4, odd+j*Nh+i);

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

}

void unpackFullSpinorSS(float *res, float4 *even, float4 *odd) {
  float K = 1.0;
  float b[24];

  for (int i=0; i<Nh; i++) {

    int boundaryCrossings = i/L1h + i/(L2*L1h) + i/(L3*L2*L1h);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      float *a = res + k*24;

      for (int j = 0; j < 6; j++) unpackFloat4(b+j*4, even+j*Nh+i);

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
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      float *a = res + k*24;

      for (int j = 0; j < 6; j++) unpackFloat4(b+j*4, odd+j*Nh+i);

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

}

void unpackParitySpinorDD(double *res, double2 *spinorPacked) {
  double K = 1.0;///sqrt(2.0);
  double b[24];

  for (int i = 0; i < Nh; i++) {
    double *a = res+i*24;

    for (int j = 0; j < 12; j++) unpackDouble2(b+j*2, spinorPacked+j*Nh+i);

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

void unpackParitySpinorDS(double *res, float4 *spinorPacked) {
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

void unpackParitySpinorSS(float *res, float4 *spinorPacked) {
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

void unpackQDPParitySpinorDD(double *res, double2 *spinorPacked) {
  double K = 1.0;///sqrt(2.0);
  double b[24];

  for (int i = 0; i < Nh; i++) {
    double *a = res+i*24;

    for (int j = 0; j < 12; j++) unpackDouble2(b+j*2, spinorPacked+j*Nh+i);

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

void unpackQDPParitySpinorDS(double *res, float4 *spinorPacked) {
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

void unpackQDPParitySpinorSS(float *res, float4 *spinorPacked) {
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

  if (cuda_prec == QUDA_DOUBLE_PRECISION && cpu_prec != QUDA_DOUBLE_PRECISION) {
    printf("Error, cannot have CUDA double precision without double CPU precision\n");
    exit(-1);
  }

  if (cuda_prec == QUDA_HALF_PRECISION) {
    if (!hSpinor1.spinor && !hSpinor1.spinorNorm &&
	!hSpinor2.spinor && !hSpinor2.spinorNorm ) {
      allocateSpinorHalf();
    } else if (!hSpinor1.spinor || !hSpinor1.spinorNorm ||
	       !hSpinor2.spinor || !hSpinor2.spinorNorm) {
      printf("allocateSpinorHalf error %lu %lu %lu %lu\n", 
	     (unsigned long)hSpinor1.spinor, (unsigned long)hSpinor1.spinorNorm,
	     (unsigned long)hSpinor2.spinor, (unsigned long)hSpinor2.spinorNorm);
      exit(-1);
    }
  }

  size_t spinor_bytes;
  if (cuda_prec == QUDA_DOUBLE_PRECISION) spinor_bytes = Nh*spinorSiteSize*sizeof(double);
  else spinor_bytes = Nh*spinorSiteSize*sizeof(float);

#ifndef __DEVICE_EMULATION__
  //if (!packedSpinor1) cudaHostAlloc(&packedSpinor1, spinor_bytes, cudaHostAllocDefault);
  if (!packedSpinor1) cudaMallocHost(&packedSpinor1, spinor_bytes);
#else
  if (!packedSpinor1) packedSpinor1 = malloc(spinor_bytes);
#endif

  if (dirac_order == QUDA_DIRAC_ORDER || QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (cuda_prec == QUDA_DOUBLE_PRECISION) {
      packParitySpinorDD((double2*)packedSpinor1, (double*)spinor);
    } else {
      if (cpu_prec == QUDA_DOUBLE_PRECISION) packParitySpinorSD((float4*)packedSpinor1, (double*)spinor);
      else packParitySpinorSS((float4*)packedSpinor1, (float*)spinor);
    }
  } else if (dirac_order == QUDA_QDP_DIRAC_ORDER) {
    if (cuda_prec == QUDA_DOUBLE_PRECISION) {
      packQDPParitySpinorDD((double2*)packedSpinor1, (double*)spinor);
    } else {
      if (cpu_prec == QUDA_DOUBLE_PRECISION) packQDPParitySpinorSD((float4*)packedSpinor1, (double*)spinor);
      else packQDPParitySpinorSS((float4*)packedSpinor1, (float*)spinor);
    }
  }
  cudaMemcpy(ret.spinor, packedSpinor1, spinor_bytes, cudaMemcpyHostToDevice);
}

void loadFullSpinor(FullSpinor ret, void *spinor, Precision cpu_prec, Precision cuda_prec) {

  if (cuda_prec == QUDA_HALF_PRECISION) {
    printf("Sorry, half precision isn't supported\n");
    exit(-1);
  }

  size_t spinor_bytes;
  if (cuda_prec == QUDA_DOUBLE_PRECISION) spinor_bytes = Nh*spinorSiteSize*sizeof(double);
  else spinor_bytes = Nh*spinorSiteSize*sizeof(float);

#ifndef __DEVICE_EMULATION__
  if (!packedSpinor1) cudaMallocHost(&packedSpinor1, spinor_bytes);
  if (!packedSpinor2) cudaMallocHost(&packedSpinor2, spinor_bytes);
#else
  if (!packedSpinor1) packedSpinor1 = malloc(spinor_bytes);
  if (!packedSpinor2) packedSpinor2 = malloc(spinor_bytes);
#endif

  if (cuda_prec == QUDA_DOUBLE_PRECISION) {
    packFullSpinorDD((double2*)packedSpinor1, (double2*)packedSpinor2, (double*)spinor);
  } else {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) packFullSpinorSD((float4*)packedSpinor1, (float4*)packedSpinor2, (double*)spinor);
    else packFullSpinorSS((float4*)packedSpinor1, (float4*)packedSpinor2, (float*)spinor);
  }

  cudaMemcpy(ret.even.spinor, packedSpinor1, spinor_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(ret.odd.spinor, packedSpinor2, spinor_bytes, cudaMemcpyHostToDevice);

#ifndef __DEVICE_EMULATION__
  cudaFreeHost(packedSpinor2);
#else
  free(packedSpinor2);
#endif
  packedSpinor2 = 0;
}

void loadSpinorField(FullSpinor ret, void *spinor, Precision cpu_prec, 
		     Precision cuda_prec, DiracFieldOrder dirac_order) {
  void *spinor_odd;
  if (cpu_prec == QUDA_SINGLE_PRECISION) spinor_odd = (float*)spinor + Nh*spinorSiteSize;
  else spinor_odd = (double*)spinor + Nh*spinorSiteSize;

  if (dirac_order == QUDA_LEX_DIRAC_ORDER) {
    loadFullSpinor(ret, spinor, cpu_prec, cuda_prec);
  } else if (dirac_order == QUDA_DIRAC_ORDER || dirac_order == QUDA_QDP_DIRAC_ORDER) {
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

  size_t spinor_bytes;
  if (cuda_prec == QUDA_DOUBLE_PRECISION) spinor_bytes = Nh*spinorSiteSize*sizeof(double);
  else spinor_bytes = Nh*spinorSiteSize*sizeof(float);

  if (!packedSpinor1) cudaMallocHost((void**)&packedSpinor1, spinor_bytes);
  cudaMemcpy(packedSpinor1, spinor.spinor, spinor_bytes, cudaMemcpyDeviceToHost);
  if (dirac_order == QUDA_DIRAC_ORDER || QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (cuda_prec == QUDA_DOUBLE_PRECISION) {
      unpackParitySpinorDD((double*)res, (double2*)packedSpinor1);
    } else {
      if (cpu_prec == QUDA_DOUBLE_PRECISION) unpackParitySpinorDS((double*)res, (float4*)packedSpinor1);
      else unpackParitySpinorSS((float*)res, (float4*)packedSpinor1);
    }
  } else if (dirac_order == QUDA_QDP_DIRAC_ORDER) {
    if (cuda_prec == QUDA_DOUBLE_PRECISION) {
      unpackQDPParitySpinorDD((double*)res, (double2*)packedSpinor1);
    } else {
      if (cpu_prec == QUDA_DOUBLE_PRECISION) unpackQDPParitySpinorDS((double*)res, (float4*)packedSpinor1);
      else unpackQDPParitySpinorSS((float*)res, (float4*)packedSpinor1);
    }
  }
}

void retrieveFullSpinor(void *res, FullSpinor spinor, Precision cpu_prec, Precision cuda_prec) {

  size_t spinor_bytes;
  if (cuda_prec == QUDA_DOUBLE_PRECISION) spinor_bytes = Nh*spinorSiteSize*sizeof(double);
  else spinor_bytes = Nh*spinorSiteSize*sizeof(float);

  if (!packedSpinor1) cudaMallocHost((void**)&packedSpinor1, spinor_bytes);
  if (!packedSpinor2) cudaMallocHost((void**)&packedSpinor2, spinor_bytes);

  cudaMemcpy(packedSpinor1, spinor.even.spinor, spinor_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(packedSpinor2, spinor.odd.spinor, spinor_bytes, cudaMemcpyDeviceToHost);
  if (cuda_prec == QUDA_DOUBLE_PRECISION) {
    unpackFullSpinorDD((double*)res, (double2*)packedSpinor1, (double2*)packedSpinor2);
  } else {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) unpackFullSpinorDS((double*)res, (float4*)packedSpinor1, (float4*)packedSpinor2);
    else unpackFullSpinorSS((float*)res, (float4*)packedSpinor1, (float4*)packedSpinor2);
  }

#ifndef __DEVICE_EMULATION__
  cudaFreeHost(packedSpinor2);
#else
  free(packedSpinor2);
#endif
  packedSpinor2 = 0;
}

void retrieveSpinorField(void *res, FullSpinor spinor, Precision cpu_prec, 
			 Precision cuda_prec, DiracFieldOrder dirac_order) {
  void *res_odd;
  if (cpu_prec == QUDA_SINGLE_PRECISION) res_odd = (float*)res + Nh*spinorSiteSize;
  else res_odd = (double*)res + Nh*spinorSiteSize;

  if (dirac_order == QUDA_LEX_DIRAC_ORDER) {
    retrieveFullSpinor(res, spinor, cpu_prec, cuda_prec);
  } else if (dirac_order == QUDA_DIRAC_ORDER || dirac_order == QUDA_QDP_DIRAC_ORDER) {
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
