#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <quda_internal.h>
#include <clover_field.h>
#include <gauge_field.h>

namespace quda {

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
      clover = device_malloc(bytes);
      if (precision == QUDA_HALF_PRECISION) {
	norm = device_malloc(norm_bytes);
      }

      even = clover;
      odd = (char*)clover + bytes/2;
    
      evenNorm = norm;
      oddNorm = (char*)norm + norm_bytes/2;

      loadCPUField(clover, norm, h_clov, cpu_prec, cpu_order);
    } 

    if (h_clov_inv) {
      cloverInv = device_malloc(bytes);
      if (precision == QUDA_HALF_PRECISION) {
	invNorm = device_malloc(bytes);
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

#ifdef USE_TEXTURE_OBJECTS
    createTexObject(evenTex, evenNormTex, even, evenNorm);
    createTexObject(oddTex, oddNormTex, odd, oddNorm);
    createTexObject(evenInvTex, evenInvNormTex, evenInv, evenInvNorm);
    createTexObject(oddInvTex, oddInvNormTex, oddInv, oddInvNorm);
#endif
    
  }

#ifdef USE_TEXTURE_OBJECTS
  void cudaCloverField::createTexObject(cudaTextureObject_t &tex, cudaTextureObject_t &texNorm,
					void *field, void *norm) {

    // create the texture for the field components

    cudaChannelFormatDesc desc;
    memset(&desc, 0, sizeof(cudaChannelFormatDesc));
    if (precision == QUDA_SINGLE_PRECISION) desc.f = cudaChannelFormatKindFloat;
    else desc.f = cudaChannelFormatKindSigned; // half is short, double is int2

    // always four components regardless of precision
    desc.x = (precision == QUDA_DOUBLE_PRECISION) ? 8*sizeof(int) : 8*precision;
    desc.y = (precision == QUDA_DOUBLE_PRECISION) ? 8*sizeof(int) : 8*precision;
    desc.z = (precision == QUDA_DOUBLE_PRECISION) ? 8*sizeof(int) : 8*precision;
    desc.w = (precision == QUDA_DOUBLE_PRECISION) ? 8*sizeof(int) : 8*precision;

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = field;
    resDesc.res.linear.desc = desc;
    resDesc.res.linear.sizeInBytes = bytes/2;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    if (precision == QUDA_HALF_PRECISION) texDesc.readMode = cudaReadModeNormalizedFloat;
    else texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    checkCudaError();

    // create the texture for the norm components
    if (precision == QUDA_HALF_PRECISION) {
      cudaChannelFormatDesc desc;
      memset(&desc, 0, sizeof(cudaChannelFormatDesc));
      desc.f = cudaChannelFormatKindFloat;
      desc.x = 8*QUDA_SINGLE_PRECISION; desc.y = 0; desc.z = 0; desc.w = 0;

      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeLinear;
      resDesc.res.linear.devPtr = norm;
      resDesc.res.linear.desc = desc;
      resDesc.res.linear.sizeInBytes = norm_bytes/2;

      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.readMode = cudaReadModeElementType;

      cudaCreateTextureObject(&texNorm, &resDesc, &texDesc, NULL);
      checkCudaError();
    }

  }

  void cudaCloverField::destroyTexObject() {
    cudaDestroyTextureObject(evenTex);
    cudaDestroyTextureObject(oddTex);
    cudaDestroyTextureObject(evenInvTex);
    cudaDestroyTextureObject(oddInvTex);
    if (precision == QUDA_HALF_PRECISION) {
      cudaDestroyTextureObject(evenNormTex);
      cudaDestroyTextureObject(oddNormTex);
      cudaDestroyTextureObject(evenInvNormTex);
      cudaDestroyTextureObject(oddInvNormTex);
    }
    checkCudaError();
  }
#endif

  cudaCloverField::~cudaCloverField()
  {
#ifdef USE_TEXTURE_OBJECTS
    destroyTexObject();
#endif

    if (clover != cloverInv) {
      if (clover) device_free(clover);
      if (norm) device_free(norm);
    }
    if (cloverInv) device_free(cloverInv);
    if (invNorm) device_free(invNorm);
    
    checkCudaError();
  }

  template <bool bqcd, typename Float>
  static inline void packCloverMatrix(float4* a, Float *b, int Vh)
  {
    const Float half = bqcd ? 1.0 : 0.5; // pre-include factor of 1/2 introduced by basis change
    
    for (int i=0; i<18; i++) {
      a[i*Vh].x = half * b[4*i+0];
      a[i*Vh].y = half * b[4*i+1];
      a[i*Vh].z = half * b[4*i+2];
      a[i*Vh].w = half * b[4*i+3];
    }
  }

  template <bool bqcd, typename Float>
  static inline void packCloverMatrix(double2* a, Float *b, int Vh)
  {
    const Float half = bqcd ? 1.0 : 0.5; // pre-include factor of 1/2 introduced by basis change

    for (int i=0; i<36; i++) {
      a[i*Vh].x = half * b[2*i+0];
      a[i*Vh].y = half * b[2*i+1];
    }
  }

  /**
     Function to reorder a BQCD clover matrix into the order that is
     expected by QUDA.  As well as reordering the clover matrix
     elements, we are also changing basis/
     
     @param quda The output clover matrix in QUDA order
     @param bqcd The input clover matrix in BQCD order
  */
  template <typename Float>
  static inline void reorderBQCD(Float *quda, Float *bqcd) {
    
    /*    int bq[36] = { 0,  1, 20, 21, 32, 33,                   // diagonal
		   2,  3,  4,  5,  6,  7,  8,  9, 10, 11,   // column 1
		   12, 13, 14, 15, 16, 17, 18, 19,          // column 2
		   22, 23, 24, 25, 26, 27,                  // column 3
		   28, 29, 30, 31,                          // column 4
		   34, 35}; */
    
    int bq[36] = { 21, 32, 33, 0,  1, 20,                   // diagonal
		   28, 29, 30, 31, 6, 7,  14, 15, 22, 23,   // column 1  6
		   34, 35, 8, 9, 16, 17, 24, 25,            // column 2  16
		   10, 11, 18, 19, 26, 27,                  // column 3  24
		    2,  3,  4,  5,                          // column 4  30
		   12, 13};
    
    // flip the sign of the imaginary components
    int sign[36];
    for (int i=0; i<6; i++) sign[i] = 1;
    for (int i=6; i<36; i+=2) {
      if ( (i >= 10 && i<= 15) || (i >= 18 && i <= 29) ) {
	sign[i] = -1; sign[i+1] = -1;	
	} else {
	sign[i] = 1; sign[i+1] = -1;
      }
    }
    
    // first chiral block
    for (int i=0; i<36; i++) quda[i] = sign[i] * bqcd[bq[i]];
    
    // second chiral block
    for (int i=0; i<36; i++) quda[i+36] = sign[i] * bqcd[bq[i]+36];
  }
  
  template <typename Float, typename FloatN>
  static void packParityClover(FloatN *res, Float *clover, int Vh, int pad, 
			       const QudaCloverFieldOrder cpu_order)
  {
    if (cpu_order == QUDA_PACKED_CLOVER_ORDER) {
      for (int i = 0; i < Vh; i++) {
	packCloverMatrix<false>(res+i, clover+72*i, Vh+pad);
      }
    } else { // must be doing BQCD order
      for (int i = 0; i < Vh; i++) {
	Float tmp[72];
	reorderBQCD(tmp, clover+72*i);
	packCloverMatrix<true>(res+i, tmp, Vh+pad);      
      }
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
	packCloverMatrix<false>(even+i, clover+72*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packCloverMatrix<false>(odd+i, clover+72*k, Vh+pad);
      }
    }
  }

  template<bool bqcd, typename Float>
  static inline void packCloverMatrixHalf(short4 *res, float *norm, Float *clover, int Vh)
  {
    const Float half = bqcd ? 1.0 : 0.5; // pre-include factor of 1/2 introduced by basis change
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
    static void packParityCloverHalf(short4 *res, float *norm, Float *clover, 
				     int Vh, int pad, const CloverFieldOrder cpu_order)
  {
    if (cpu_order == QUDA_PACKED_CLOVER_ORDER) {
      for (int i = 0; i < Vh; i++) {
	packCloverMatrixHalf<false>(res+i, norm+i, clover+72*i, Vh+pad);
      }
    } else { // must be doing BQCD order
      for (int i = 0; i < Vh; i++) {
	Float tmp[72];
	reorderBQCD(tmp, clover+72*i);
	packCloverMatrixHalf<true>(res+i, norm+i, tmp, Vh+pad);
      }
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
	packCloverMatrixHalf<false>(even+i, evenNorm+i, clover+72*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packCloverMatrixHalf<false>(odd+i, oddNorm+i, clover+72*k, Vh+pad);
      }
    }
  }

  void cudaCloverField::loadCPUField(void *clover, void *norm, const void *h_clover, 
				     const QudaPrecision cpu_prec, const CloverFieldOrder cpu_order) {

    void *h_clover_odd = (char*)h_clover + cpu_prec*real_length/2;
    
    if (cpu_order == QUDA_LEX_PACKED_CLOVER_ORDER) {
      loadFullField(clover, norm, (char*)clover+bytes/2, (char*)norm+norm_bytes/2, h_clover, cpu_prec, cpu_order);
    } else if (cpu_order == QUDA_PACKED_CLOVER_ORDER || cpu_order == QUDA_BQCD_CLOVER_ORDER) {
      loadParityField(clover, norm, h_clover, cpu_prec, cpu_order);
      loadParityField((char*)clover+bytes/2, (char*)norm+norm_bytes/2, h_clover_odd, cpu_prec, cpu_order);
    } else if(cpu_order == QUDA_INTERNAL_CLOVER_ORDER) { // No reordering necessary, just a plain copy
      cudaMemcpy(clover, h_clover, total_bytes, cudaMemcpyHostToDevice); 
    }else{
      errorQuda("Invalid clover_order");
    }
  }

  void cudaCloverField::loadParityField(void *clover, void *cloverNorm, const void *h_clover, 
					const QudaPrecision cpu_prec, const CloverFieldOrder cpu_order)
  {
    // use pinned memory                                                                                           
    void *packedClover, *packedCloverNorm=0;

    if (precision == QUDA_DOUBLE_PRECISION && cpu_prec != QUDA_DOUBLE_PRECISION) {
      errorQuda("Cannot have CUDA double precision without CPU double precision");
    }
    if (cpu_order != QUDA_PACKED_CLOVER_ORDER && cpu_order != QUDA_BQCD_CLOVER_ORDER) 
      errorQuda("Invalid clover order %d", cpu_order);

    resizeBuffer(bytes/2 + norm_bytes/2);
    packedClover = bufferPinned;
    if (precision == QUDA_HALF_PRECISION) packedCloverNorm = (char*)bufferPinned + bytes/2;

    if (precision == QUDA_DOUBLE_PRECISION) {
      packParityClover((double2 *)packedClover, (double *)h_clover, volumeCB, pad, cpu_order);
    } else if (precision == QUDA_SINGLE_PRECISION) {
      if (cpu_prec == QUDA_DOUBLE_PRECISION) {
	packParityClover((float4 *)packedClover, (double *)h_clover, volumeCB, pad, cpu_order);
      } else {
	packParityClover((float4 *)packedClover, (float *)h_clover, volumeCB, pad, cpu_order);
      }
    } else {
      if (cpu_prec == QUDA_DOUBLE_PRECISION) {
	packParityCloverHalf((short4 *)packedClover, (float *)packedCloverNorm, 
			     (double *)h_clover, volumeCB, pad, cpu_order);
      } else {
	packParityCloverHalf((short4 *)packedClover, (float *)packedCloverNorm, 
			     (float *)h_clover, volumeCB, pad, cpu_order);
      }
    }
  
    cudaMemcpy(clover, packedClover, bytes/2, cudaMemcpyHostToDevice);
    if (precision == QUDA_HALF_PRECISION)
      cudaMemcpy(cloverNorm, packedCloverNorm, norm_bytes/2, cudaMemcpyHostToDevice);
  }

  void cudaCloverField::loadFullField(void *even, void *evenNorm, void *odd, void *oddNorm, 
				      const void *h_clover, const QudaPrecision cpu_prec, 
				      const CloverFieldOrder cpu_order)
  {
    // use pinned memory                  
    void *packedEven, *packedOdd, *packedEvenNorm=0, *packedOddNorm=0;

    if (precision == QUDA_DOUBLE_PRECISION && cpu_prec != QUDA_DOUBLE_PRECISION) {
      errorQuda("Cannot have CUDA double precision without CPU double precision");
    }
    if (cpu_order != QUDA_LEX_PACKED_CLOVER_ORDER) {
      errorQuda("Invalid clover order");
    }

    resizeBuffer(bytes + norm_bytes);
    packedEven = bufferPinned;
    packedOdd = (char*)bufferPinned + bytes/2;

    if (precision == QUDA_HALF_PRECISION) {
      packedEvenNorm = (char*)bufferPinned + bytes;
      packedOddNorm = (char*)bufferPinned + bytes + norm_bytes/2;
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
  }


  /**
     Computes Fmunu given the gauge field U
  */
  void cudaCloverField::compute(const cudaGaugeField &gauge) {

    if (gauge.Precision() != precision) 
      errorQuda("Gauge and clover precisions must match");

    computeCloverCuda(*this, gauge);

  }

} // namespace quda
