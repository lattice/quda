#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <quda_internal.h>
#include <clover_field.h>
#include <gauge_field.h>

CloverField::CloverField(const CloverFieldParam &param) :
  LatticeField(param), bytes(0), norm_bytes(0), nColor(3), nSpin(4)
{
  if (nDim != 4) errorQuda("Number of dimensions must be 4, not %d", nDim);

  real_length = 2*volumeCB*nColor*nColor*nSpin*nSpin/2;  // block-diagonal Hermitian (72 reals)
  length = 2*stride*nColor*nColor*nSpin*nSpin/2;

  bytes = length*precision;
  bytes = ALIGNMENT_ADJUST(bytes);
  if (precision == QUDA_HALF_PRECISION) {
    norm_bytes = sizeof(float)*2*stride*2; // 2 chirality
    norm_bytes = ALIGNMENT_ADJUST(norm_bytes);
  }

  total_bytes = bytes + norm_bytes;
}

CloverField::~CloverField() {

}

cudaCloverField::cudaCloverField(const void *h_clov, const void *h_clov_inv, 
				 const QudaPrecision cpu_prec, 
				 const QudaCloverFieldOrder cpu_order,
				 const CloverFieldParam &param)
  : CloverField(param), clover(0), norm(0), cloverInv(0), invNorm(0)
{
  

  if (h_clov) {
    if (cudaMalloc((void**)&clover, bytes) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating clover term");
    }   
    
    if (precision == QUDA_HALF_PRECISION) {
      if (cudaMalloc((void**)&norm, norm_bytes) == cudaErrorMemoryAllocation) {
	errorQuda("Error allocating clover norm");
      }
    }

    even = clover;
    odd = (char*)clover + bytes/2;
    
    evenNorm = norm;
    oddNorm = (char*)norm + norm_bytes/2;

    loadCPUField(clover, norm, h_clov, cpu_prec, cpu_order);
  } 

  if (h_clov_inv) {
    if (cudaMalloc((void**)&cloverInv, bytes) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating clover inverse term");
    }   
    
    if (precision == QUDA_HALF_PRECISION) {
      if (cudaMalloc((void**)&invNorm, norm_bytes) == cudaErrorMemoryAllocation) {
	errorQuda("Error allocating clover inverse norm");
      }
    }

    evenInv = cloverInv;
    oddInv = (char*)cloverInv + bytes/2;
    
    evenInvNorm = invNorm;
    oddInvNorm = (char*)invNorm + norm_bytes/2;

    total_bytes += bytes + norm_bytes;

    loadCPUField(cloverInv, invNorm, h_clov_inv, cpu_prec, cpu_order);

    // this is a hack to ensure that we can autotune the clover
    // operator when just using symmetric preconditioning
    if (!clover) {
      clover = cloverInv;
      even = evenInv;
      odd = oddInv;
    }
    if (!norm) {
      norm = invNorm;
      evenNorm = evenInvNorm;
      oddNorm = oddInvNorm;
    }
  } 

}

cudaCloverField::~cudaCloverField() {
  if (clover != cloverInv) {
    if ( clover ) cudaFree(clover);
    if ( norm ) cudaFree(norm);
  }

  if ( cloverInv ) cudaFree(cloverInv);
  if ( invNorm ) cudaFree(invNorm);

  checkCudaError();
}

template <typename Float>
static inline void packCloverMatrix(float4* a, Float *b, int Vh)
{
  const Float half = 0.5; // pre-include factor of 1/2 introduced by basis change

  for (int i=0; i<18; i++) {
    a[i*Vh].x = half * b[4*i+0];
    a[i*Vh].y = half * b[4*i+1];
    a[i*Vh].z = half * b[4*i+2];
    a[i*Vh].w = half * b[4*i+3];
  }
}

template <typename Float>
static inline void packCloverMatrix(double2* a, Float *b, int Vh)
{
  const Float half = 0.5; // pre-include factor of 1/2 introduced by basis change

  for (int i=0; i<36; i++) {
    a[i*Vh].x = half * b[2*i+0];
    a[i*Vh].y = half * b[2*i+1];
  }
}

template <typename Float, typename FloatN>
static void packParityClover(FloatN *res, Float *clover, int Vh, int pad)
{
  for (int i = 0; i < Vh; i++) {
    packCloverMatrix(res+i, clover+72*i, Vh+pad);
  }
}

template <typename Float, typename FloatN>
static void packFullClover(FloatN *even, FloatN *odd, Float *clover, int *X, int pad)
{
  int Vh = X[0]*X[1]*X[2]*X[3]/2;

  for (int i=0; i<Vh; i++) {

    int boundaryCrossings = i/X[0] + i/(X[1]*X[0]) + i/(X[2]*X[1]*X[0]);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      packCloverMatrix(even+i, clover+72*k, Vh+pad);
    }
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      packCloverMatrix(odd+i, clover+72*k, Vh+pad);
    }
  }
}

template<typename Float>
static inline void packCloverMatrixHalf(short4 *res, float *norm, Float *clover, int Vh)
{
  const Float half = 0.5; // pre-include factor of 1/2 introduced by basis change
  Float max, a, c;

  // treat the two chiral blocks separately
  for (int chi=0; chi<2; chi++) {
    max = fabs(clover[0]);
    for (int i=1; i<36; i++) {
      if ((a = fabs(clover[i])) > max) max = a;
    }
    c = MAX_SHORT/max;
    for (int i=0; i<9; i++) {
      res[i*Vh].x = (short) (c * clover[4*i+0]);
      res[i*Vh].y = (short) (c * clover[4*i+1]);
      res[i*Vh].z = (short) (c * clover[4*i+2]);
      res[i*Vh].w = (short) (c * clover[4*i+3]);
    }
    norm[chi*Vh] = half*max;
    res += 9*Vh;
    clover += 36;
  }
}

template <typename Float>
static void packParityCloverHalf(short4 *res, float *norm, Float *clover, int Vh, int pad)
{
  for (int i = 0; i < Vh; i++) {
    packCloverMatrixHalf(res+i, norm+i, clover+72*i, Vh+pad);
  }
}

template <typename Float>
static void packFullCloverHalf(short4 *even, float *evenNorm, short4 *odd, float *oddNorm,
			       Float *clover, int *X, int pad)
{
  int Vh = X[0]*X[1]*X[2]*X[3]/2;

  for (int i=0; i<Vh; i++) {

    int boundaryCrossings = i/X[0] + i/(X[1]*X[0]) + i/(X[2]*X[1]*X[0]);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      packCloverMatrixHalf(even+i, evenNorm+i, clover+72*k, Vh+pad);
    }
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      packCloverMatrixHalf(odd+i, oddNorm+i, clover+72*k, Vh+pad);
    }
  }
}

void cudaCloverField::loadCPUField(void *clover, void *norm, const void *h_clover, 
				   const QudaPrecision cpu_prec, const CloverFieldOrder cpu_order) {

  void *h_clover_odd = (char*)h_clover + cpu_prec*real_length/2;

  if (cpu_order == QUDA_LEX_PACKED_CLOVER_ORDER) {
    loadFullField(clover, norm, (char*)clover+bytes/2, (char*)norm+norm_bytes/2, h_clover, cpu_prec, cpu_order);
  } else if (cpu_order == QUDA_PACKED_CLOVER_ORDER) {
    loadParityField(clover, norm, h_clover, cpu_prec, cpu_order);
    loadParityField((char*)clover+bytes/2, (char*)norm+norm_bytes/2, h_clover_odd, cpu_prec, cpu_order);
  } else {
    errorQuda("Invalid clover_order");
  }
}

void cudaCloverField::loadParityField(void *clover, void *cloverNorm, const void *h_clover, 
				      const QudaPrecision cpu_prec, const CloverFieldOrder cpu_order)
{
  // use pinned memory                                                                                           
  void *packedClover, *packedCloverNorm;

  if (precision == QUDA_DOUBLE_PRECISION && cpu_prec != QUDA_DOUBLE_PRECISION) {
    errorQuda("Cannot have CUDA double precision without CPU double precision");
  }
  if (cpu_order != QUDA_PACKED_CLOVER_ORDER) 
    errorQuda("Invalid clover order %d", cpu_order);

  if (cudaMallocHost(&packedClover, bytes/2) == cudaErrorMemoryAllocation)
    errorQuda("Error allocating clover pinned memory");

  if (precision == QUDA_HALF_PRECISION) {
    if (cudaMallocHost(&packedCloverNorm, norm_bytes/2) == cudaErrorMemoryAllocation)
      {
	errorQuda("Error allocating clover pinned memory");
      } 
  }
    
  if (precision == QUDA_DOUBLE_PRECISION) {
    packParityClover((double2 *)packedClover, (double *)h_clover, volumeCB, pad);
  } else if (precision == QUDA_SINGLE_PRECISION) {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      packParityClover((float4 *)packedClover, (double *)h_clover, volumeCB, pad);
    } else {
      packParityClover((float4 *)packedClover, (float *)h_clover, volumeCB, pad);
    }
  } else {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      packParityCloverHalf((short4 *)packedClover, (float *)packedCloverNorm, 
			   (double *)h_clover, volumeCB, pad);
    } else {
      packParityCloverHalf((short4 *)packedClover, (float *)packedCloverNorm, 
			   (float *)h_clover, volumeCB, pad);
    }
  }
  
  cudaMemcpy(clover, packedClover, bytes/2, cudaMemcpyHostToDevice);
  if (precision == QUDA_HALF_PRECISION)
    cudaMemcpy(cloverNorm, packedCloverNorm, norm_bytes/2, cudaMemcpyHostToDevice);

  cudaFreeHost(packedClover);
  if (precision == QUDA_HALF_PRECISION) cudaFreeHost(packedCloverNorm);
}

void cudaCloverField::loadFullField(void *even, void *evenNorm, void *odd, void *oddNorm, 
				    const void *h_clover, const QudaPrecision cpu_prec, 
				    const CloverFieldOrder cpu_order)
{
  // use pinned memory                  
  void *packedEven, *packedEvenNorm, *packedOdd, *packedOddNorm;

  if (precision == QUDA_DOUBLE_PRECISION && cpu_prec != QUDA_DOUBLE_PRECISION) {
    errorQuda("Cannot have CUDA double precision without CPU double precision");
  }
  if (cpu_order != QUDA_LEX_PACKED_CLOVER_ORDER) {
    errorQuda("Invalid clover order");
  }

  if (cudaMallocHost(&packedEven, bytes/2) ==  cudaErrorMemoryAllocation){
    errorQuda("cudaMallocHost failed for packedEven\n");
  }
  if ( cudaMallocHost(&packedOdd, bytes/2) == cudaErrorMemoryAllocation){
    errorQuda("cudaMallocHost failed for packedOdd\n");
  }
  if (precision == QUDA_HALF_PRECISION) {
    if (cudaMallocHost(&packedEvenNorm, norm_bytes/2) == cudaErrorMemoryAllocation){
      errorQuda("cudaMallocHost failed for packedEvenNorm\n");
    }
    if (cudaMallocHost(&packedOddNorm, norm_bytes/2) == cudaErrorMemoryAllocation){
      errorQuda("cudaMallocHost failed for packedOddNorm\n");
    }
  }
    
  if (precision == QUDA_DOUBLE_PRECISION) {
    packFullClover((double2 *)packedEven, (double2 *)packedOdd, (double *)clover, x, pad);
  } else if (precision == QUDA_SINGLE_PRECISION) {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      packFullClover((float4 *)packedEven, (float4 *)packedOdd, (double *)clover, x, pad);
    } else {
      packFullClover((float4 *)packedEven, (float4 *)packedOdd, (float *)clover, x, pad);    
    }
  } else {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      packFullCloverHalf((short4 *)packedEven, (float *)packedEvenNorm, (short4 *)packedOdd,
			 (float *) packedOddNorm, (double *)clover, x, pad);
    } else {
      packFullCloverHalf((short4 *)packedEven, (float *)packedEvenNorm, (short4 *)packedOdd,
			 (float * )packedOddNorm, (float *)clover, x, pad);    
    }
  }

  cudaMemcpy(even, packedEven, bytes/2, cudaMemcpyHostToDevice);
  cudaMemcpy(odd, packedOdd, bytes/2, cudaMemcpyHostToDevice);
  if (precision == QUDA_HALF_PRECISION) {
    cudaMemcpy(evenNorm, packedEvenNorm, norm_bytes/2, cudaMemcpyHostToDevice);
    cudaMemcpy(oddNorm, packedOddNorm, norm_bytes/2, cudaMemcpyHostToDevice);
  }

  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
  if (precision == QUDA_HALF_PRECISION) {
    cudaFreeHost(packedEvenNorm);
    cudaFreeHost(packedOddNorm);
  }
}


/**
   Computes Fmunu given the gauge field U
 */
void cudaCloverField::compute(const cudaGaugeField &gauge) {

  if (gauge.Precision() != precision) 
    errorQuda("Gauge and clover precisions must match");

  computeCloverCuda(*this, gauge);

}
