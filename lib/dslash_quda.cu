
#include <stdlib.h>
#include <stdio.h>

#include <color_spinor_field.h>

#define BLOCK_DIM 64

// these control the Wilson-type actions
//#define DIRECT_ACCESS_LINK
//#define DIRECT_ACCESS_WILSON_SPINOR
//#define DIRECT_ACCESS_WILSON_ACCUM
//#define DIRECT_ACCESS_WILSON_PACK_SPINOR

//these are access control for staggered action
#if (__CUDA_ARCH__ >= 200)
//#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
#define DIRECT_ACCESS_SPINOR
#else
#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
//#define DIRECT_ACCESS_SPINOR
#endif

#include <quda_internal.h>
#include <dslash_quda.h>
#include <sys/time.h>

struct DslashParam {
  int tOffset; // offset into the T dimension (multi gpu only)
  int tMul;    // spatial volume distance between the T faces being updated (multi gpu only)
  int threads; // the desired number of active threads
  int parity;  // Even-Odd or Odd-Even
  int ghostDim[QUDA_MAX_DIM];
  int ghostOffset[QUDA_MAX_DIM];
};

DslashParam dslashParam;

// these are set in initDslashConst
int Vspatial;
#ifdef MULTI_GPU
static const int Nstream = 9;
#else
static const int Nstream = 1;
#endif
static cudaStream_t streams[Nstream];

FaceBuffer *face;
cudaColorSpinorField *inSpinor;

#include <dslash_textures.h>
#include <dslash_constants.h>

#define SHORT_LENGTH 65536
#define SCALE_FLOAT ((SHORT_LENGTH-1) * 0.5) // 32767.5
#define SHIFT_FLOAT (-1.f / (SHORT_LENGTH-1)) // 1.5259021897e-5

__device__ short float2short(float c, float a) {
  //return (short)(a*MAX_SHORT);
  short rtn = (short)((a+SHIFT_FLOAT)*SCALE_FLOAT*c);
  return rtn;
}

__device__ float short2float(short a) {
  return (float)a/SCALE_FLOAT - SHIFT_FLOAT;
}

__device__ short4 float42short4(float c, float4 a) {
  return make_short4(float2short(c, a.x), float2short(c, a.y), float2short(c, a.z), float2short(c, a.w));
}

__device__ float4 short42float4(short4 a) {
  return make_float4(short2float(a.x), short2float(a.y), short2float(a.z), short2float(a.w));
}

__device__ float2 short22float2(short2 a) {
  return make_float2(short2float(a.x), short2float(a.y));
}


#include <staggered_dslash_def.h> // staggered Dslash kernels
#include <wilson_dslash_def.h>    // Wilson Dslash kernels (including clover)
#include <dw_dslash_def.h>        // Domain Wall kernels
#include <tm_dslash_def.h>        // Twisted Mass kernels
#include <tm_core.h>              // solo twisted mass kernel
#include <clover_def.h>           // kernels for applying the clover term alone

#ifdef MULTI_GPU
#include <pack_face_def.h>        // kernels for packing the ghost zones
#endif

#ifndef SHARED_FLOATS_PER_THREAD
#define SHARED_FLOATS_PER_THREAD 0
#endif

#include <blas_quda.h>
#include <face_quda.h>


// dslashTuning = QUDA_TUNE_YES turns off error checking
static QudaTune dslashTuning = QUDA_TUNE_NO;

void setDslashTuning(QudaTune tune)
{
  dslashTuning = tune;
}

__global__ void dummyKernel() {
  // do nothing
}

void initCache() {

#if (__CUDA_ARCH__ >= 200)

  static int firsttime = 1;
  if (firsttime){	
    cudaFuncSetCacheConfig(dummyKernel, cudaFuncCachePreferL1);
    dummyKernel<<<1,1>>>();
    firsttime=0;
  }

#endif

}

void setFace(const FaceBuffer &Face) {
  face = (FaceBuffer*)&Face; // nasty
}

// Use an abstract class interface to drive the different CUDA dslash
// kernels.  All parameters are curried into the derived classes to
// allow a simple interface.
class DslashCuda {
public:
  DslashCuda() { ; }
  virtual ~DslashCuda() { ; }
  virtual void apply(const dim3 &blockDim, const int shared_bytes, const cudaStream_t &stream) = 0;
};

// Use this macro for all dslash types
#define DSLASH(FUNC, X, gridDim, blockDim, shared, stream, ...)		\
if (x==0) {								\
  if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
    if (!dagger) {							\
      FUNC ## 18Kernel <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ ); \
    } else {								\
      FUNC ## 18DaggerKernel <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ ); \
    }									\
  } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
    if (!dagger) {							\
      FUNC ## 12Kernel <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ ); \
    } else {								\
      FUNC ## 12DaggerKernel <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ ); \
    }									\
  } else {								\
    if (!dagger) {							\
      FUNC ## 8Kernel <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ ); \
    } else {								\
      FUNC ## 8DaggerKernel <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ ); \
    }									\
  }									\
 } else {								\
  if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
    if (!dagger) {							\
      FUNC ## 18 ## X ## Kernel <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ ); \
    } else {								\
      FUNC ## 18Dagger ## X ## Kernel <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ ); \
    }									\
  } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
    if (!dagger) {							\
      FUNC ## 12 ## X ## Kernel <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ ); \
    } else {								\
      FUNC ## 12Dagger ## X ## Kernel <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ ); \
    }									\
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
    if (!dagger) {							\
      FUNC ## 8 ## X ## Kernel <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ ); \
    } else {								\
      FUNC ## 8Dagger ## X ## Kernel <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ ); \
    }									\
  }									\
 }									
  

template <typename sFloat, typename gFloat>
class WilsonDslashCuda : public DslashCuda {

private:
  sFloat *out;
  float *outNorm;
  const sFloat *in, *x;
  const float *inNorm, *xNorm;
  const gFloat *gauge0, *gauge1;
  const QudaReconstructType reconstruct;
  const int dagger;
  const double a;

public:
  WilsonDslashCuda(sFloat *out, float *outNorm, const gFloat *gauge0, const gFloat *gauge1, 
		   const QudaReconstructType reconstruct, const sFloat *in, const float *inNorm,
		   const sFloat *x, const float *xNorm, const double a,
		   const int dagger, const size_t bytes, const size_t norm_bytes) :
    DslashCuda(), out(out), outNorm(outNorm), gauge0(gauge0), gauge1(gauge1), in(in), 
    inNorm(inNorm), reconstruct(reconstruct), dagger(dagger), x(x), xNorm(xNorm), a(a) { 
    bindSpinorTex(bytes, norm_bytes, in, inNorm, x, xNorm); 
  }
  virtual ~WilsonDslashCuda() { unbindSpinorTex(in, inNorm, x, xNorm); }

  void apply(const dim3 &blockDim, const int shared_bytes, const cudaStream_t &stream) {
    dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);
    DSLASH(dslash, Xpay, gridDim, blockDim, shared_bytes, stream, out, outNorm, 
	   gauge0, gauge1, in, inNorm, x, xNorm, a, dslashParam);
  }

};

template <typename sFloat, typename gFloat, typename cFloat>
class CloverDslashCuda : public DslashCuda {

private:
  sFloat *out;
  float *outNorm;
  const sFloat *in, *x;
  const float *inNorm, *xNorm;
  const gFloat *gauge0, *gauge1;
  const QudaReconstructType reconstruct;
  const cFloat *clover;
  const float *cloverNorm;
  const int dagger;
  const double a;

public:
  CloverDslashCuda(sFloat *out, float *outNorm, const gFloat *gauge0, const gFloat *gauge1, 
		   const QudaReconstructType reconstruct, const cFloat *clover, 
		   const float *cloverNorm, const sFloat *in, const float *inNorm,
		   const sFloat *x, const float *xNorm, const double a,
		   const int dagger, const size_t bytes, const size_t norm_bytes) :
    DslashCuda(), out(out), outNorm(outNorm), gauge0(gauge0), gauge1(gauge1), 
    clover(clover), cloverNorm(cloverNorm), in(in), inNorm(inNorm), 
    reconstruct(reconstruct), dagger(dagger), x(x), xNorm(xNorm), a(a) { 
    bindSpinorTex(bytes, norm_bytes, in, inNorm, x, xNorm); 
  }
  virtual ~CloverDslashCuda() { unbindSpinorTex(in, inNorm, x, xNorm); }

  void apply(const dim3 &blockDim, const int shared_bytes, const cudaStream_t &stream) {
    dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);
    DSLASH(cloverDslash, Xpay, gridDim, blockDim, shared_bytes, stream, out, outNorm, 
	   gauge0, gauge1, clover, cloverNorm, in, inNorm, x, xNorm, a, dslashParam);
  }

};

void setTwistParam(double &a, double &b, const double &kappa, const double &mu, 
		   const int dagger, const QudaTwistGamma5Type twist) {
  if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
    a = 2.0 * kappa * mu;
    b = 1.0;
  } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
    a = -2.0 * kappa * mu;
    b = 1.0 / (1.0 + a*a);
  } else {
    errorQuda("Twist type %d not defined\n", twist);
  }
  if (dagger) a *= -1.0;

}

template <typename sFloat, typename gFloat>
class TwistedDslashCuda : public DslashCuda {

private:
  sFloat *out;
  float *outNorm;
  const sFloat *in, *x;
  const float *inNorm, *xNorm;
  const gFloat *gauge0, *gauge1;
  const QudaReconstructType reconstruct;
  const int dagger;
  double a;
  double b;

public:
  TwistedDslashCuda(sFloat *out, float *outNorm, const gFloat *gauge0, const gFloat *gauge1, 
		    const QudaReconstructType reconstruct, const sFloat *in, const float *inNorm,
		    const sFloat *x, const float *xNorm, const double kappa, const double mu,
		    const double k, const int dagger, const size_t bytes, const size_t norm_bytes) :
    DslashCuda(), out(out), outNorm(outNorm), gauge0(gauge0), gauge1(gauge1), 
    in(in), inNorm(inNorm), reconstruct(reconstruct), dagger(dagger), x(x), xNorm(xNorm) { 
    bindSpinorTex(bytes, norm_bytes, in, inNorm, x, xNorm); 
    setTwistParam(a, b, kappa, mu, dagger, QUDA_TWIST_GAMMA5_INVERSE);
    if (x) b *= k;
  }
  virtual ~TwistedDslashCuda() { unbindSpinorTex(in, inNorm, x, xNorm); }

  void apply(const dim3 &blockDim, const int shared_bytes, const cudaStream_t &stream) {
    dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);
    DSLASH(twistedMassDslash, Xpay, gridDim, blockDim, shared_bytes, stream, out, outNorm, 
	   gauge0, gauge1, in, inNorm, a, b, x, xNorm, dslashParam);
  }

};

template <typename sFloat, typename gFloat>
class DomainWallDslashCuda : public DslashCuda {

private:
  sFloat *out;
  float *outNorm;
  const sFloat *in, *x;
  const float *inNorm, *xNorm;
  const gFloat *gauge0, *gauge1;
  const QudaReconstructType reconstruct;
  const int dagger;
  const double mferm;
  const double a;

public:
  DomainWallDslashCuda(sFloat *out, float *outNorm, const gFloat *gauge0, const gFloat *gauge1, 
		       const QudaReconstructType reconstruct, const sFloat *in, 
		       const float *inNorm, const sFloat *x, const float *xNorm, const double mferm, 
		       const double a, const int dagger, const size_t bytes, const size_t norm_bytes) :
    DslashCuda(), out(out), outNorm(outNorm), gauge0(gauge0), gauge1(gauge1), 
    in(in), inNorm(inNorm), mferm(mferm), reconstruct(reconstruct), dagger(dagger), x(x), xNorm(xNorm), a(a) { 
    bindSpinorTex(bytes, norm_bytes, in, inNorm, x, xNorm); 
  }
  virtual ~DomainWallDslashCuda() { unbindSpinorTex(in, inNorm, x, xNorm); }

  void apply(const dim3 &blockDim, const int shared_bytes, const cudaStream_t &stream) {
    dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);
    DSLASH(domainWallDslash, Xpay, gridDim, blockDim, shared_bytes, stream, out, outNorm, 
	   gauge0, gauge1, in, inNorm, mferm, x, xNorm, a, dslashParam);
  }

};

void dslashCuda(DslashCuda &dslash, const size_t regSize, const int parity, const int dagger, 
		const int volume, const dim3 block, const dim3 blockFace) {
  int shared_bytes = block.x*SHARED_FLOATS_PER_THREAD*regSize;

  dslashParam.parity = parity;

#ifndef MULTI_GPU
  dslashParam.tOffset = 0;
  dslashParam.tMul = 1;
  dslashParam.threads = volume;

  dslash.apply(block, shared_bytes, streams[0]);
#else

  // Gather from source spinor
  face->exchangeFacesStart(*inSpinor, 1-parity, dagger, streams);
  
#ifdef OVERLAP_COMMS // do body
  dslashParam.tOffset = 1;
  dslashParam.tMul = 1;
  dslashParam.threads = volume - 2*Vspatial;

  dslash.apply(block, shared_bytes, streams[Nstream-1]);
#endif // OVERLAP_COMMS

  // Finish gather and start comms
  face->exchangeFacesComms(3);

  // Wait for comms to finish, and scatter into the end zone
  face->exchangeFacesWait(*inSpinor, dagger,3);

  dslashParam.tOffset = 0;
#ifdef OVERLAP_COMMS // do faces
  dslashParam.tMul = volume/Vspatial - 1; // hacky way to get Nt
  dslashParam.threads = 2*Vspatial;
#else // do all
  dslashParam.tMul = 1;
  dslashParam.threads = volume;
#endif // OVERLAP_COMMS
  shared_bytes = blockFace.x*SHARED_FLOATS_PER_THREAD*regSize;
  dslash.apply(blockFace, shared_bytes, streams[Nstream-2]);

#endif // MULTI_GPU
}

// Wilson wrappers
void dslashCuda(cudaColorSpinorField *out, const FullGauge gauge, const cudaColorSpinorField *in,
		const int parity, const int dagger, const cudaColorSpinorField *x,
		const double &k, const dim3 &block, const dim3 &blockFace) {

  inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_WILSON_DIRAC
  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (in->precision != gauge.precision)
    errorQuda("Mixing gauge and spinor precision not supported");

  void *xv = (x ? x->v : 0);
  void *xn = (x ? x->norm : 0);

  DslashCuda *dslash = 0;
  size_t regSize = sizeof(float);
  if (in->precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    dslash = new WilsonDslashCuda<double2, double2>((double2*)out->v, (float*)out->norm, (double2*)gauge0, (double2*)gauge1, 
						    gauge.reconstruct, (double2*)in->v, (float*)in->norm, 
						    (double2*)xv, (float*)xn, k, dagger, in->bytes, in->norm_bytes);
    regSize = sizeof(double);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->precision == QUDA_SINGLE_PRECISION) {
    dslash = new WilsonDslashCuda<float4, float4>((float4*)out->v, (float*)out->norm, (float4*)gauge0, (float4*)gauge1,
						  gauge.reconstruct, (float4*)in->v, (float*)in->norm, 
						  (float4*)xv, (float*)xn, k, dagger, in->bytes, in->norm_bytes);
  } else if (in->precision == QUDA_HALF_PRECISION) {
    dslash = new WilsonDslashCuda<short4, short4>((short4*)out->v, (float*)out->norm, (short4*)gauge0, (short4*)gauge1,
						  gauge.reconstruct, (short4*)in->v, (float*)in->norm,
						  (short4*)xv, (float*)xn, k, dagger, in->bytes, in->norm_bytes);
  }
  dslashCuda(*dslash, regSize, parity, dagger, in->volume, block, blockFace);

  delete dslash;
  unbindGaugeTex(gauge);

  if (!dslashTuning) checkCudaError();
#else
  errorQuda("Wilson dslash has not been built");
#endif // GPU_WILSON_DIRAC

}

void cloverDslashCuda(cudaColorSpinorField *out, const FullGauge gauge, const FullClover cloverInv,
		      const cudaColorSpinorField *in, const int parity, const int dagger, 
		      const cudaColorSpinorField *x, const double &a,
		      const dim3 &block, const dim3 &blockFace) {

  inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_WILSON_DIRAC
  void *cloverP, *cloverNormP;
  QudaPrecision clover_prec = bindCloverTex(cloverInv, parity, &cloverP, &cloverNormP);

  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (in->precision != gauge.precision)
    errorQuda("Mixing gauge and spinor precision not supported");

  if (in->precision != clover_prec)
    errorQuda("Mixing clover and spinor precision not supported");

  void *xv = x ? x->v : 0;
  void *xn = x ? x->norm : 0;

  DslashCuda *dslash = 0;
  size_t regSize = sizeof(float);

  if (in->precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    dslash = new CloverDslashCuda<double2, double2, double2>((double2*)out->v, (float*)out->norm, (double2*)gauge0, 
							     (double2*)gauge1, gauge.reconstruct, (double2*)cloverP, 
							     (float*)cloverNormP, (double2*)in->v, (float*)in->norm,
							     (double2*)xv, (float*)xn, a, dagger, in->bytes, in->norm_bytes);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->precision == QUDA_SINGLE_PRECISION) {
    dslash = new CloverDslashCuda<float4, float4, float4>((float4*)out->v, (float*)out->norm, (float4*)gauge0, 
							  (float4*)gauge1, gauge.reconstruct, (float4*)cloverP, 
							  (float*)cloverNormP, (float4*)in->v, (float*)in->norm, 
							  (float4*)xv, (float*)xn, a, dagger, in->bytes, in->norm_bytes);
  } else if (in->precision == QUDA_HALF_PRECISION) {
    dslash = new CloverDslashCuda<short4, short4, short4>((short4*)out->v, (float*)out->norm, (short4*)gauge0, 
							  (short4*)gauge1, gauge.reconstruct, (short4*)cloverP, 
							  (float*)cloverNormP, (short4*)in->v, (float*)in->norm, 
							  (short4*)xv, (float*)xn, a, dagger, in->bytes, in->norm_bytes);
  }

  dslashCuda(*dslash, regSize, parity, dagger, in->volume, block, blockFace);

  delete dslash;
  unbindGaugeTex(gauge);
  unbindCloverTex(cloverInv);

  if (!dslashTuning) checkCudaError();
#else
  errorQuda("Clover dslash has not been built");
#endif

}


void twistedMassDslashCuda(cudaColorSpinorField *out, const FullGauge gauge, 
			   const cudaColorSpinorField *in, const int parity, const int dagger, 
			   const cudaColorSpinorField *x, const double &kappa, const double &mu, 
			   const double &a, const dim3 &block, const dim3 &blockFace) {

  inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_TWISTED_MASS_DIRAC
  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (in->precision != gauge.precision)
    errorQuda("Mixing gauge and spinor precision not supported");

  void *xv = x ? x->v : 0;
  void *xn = x ? x->norm : 0;

  DslashCuda *dslash = 0;
  size_t regSize = sizeof(float);

  if (in->precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    dslash = new TwistedDslashCuda<double2,double2>((double2*)out->v, (float*)out->norm, (double2*)gauge0, 
						    (double2*)gauge1, gauge.reconstruct, (double2*)in->v, 
						    (float*)in->norm, (double2*)xv, (float*)xn, 
						    kappa, mu, a, dagger, in->bytes, in->norm_bytes);
    regSize = sizeof(double);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->precision == QUDA_SINGLE_PRECISION) {
    dslash = new TwistedDslashCuda<float4,float4>((float4*)out->v, (float*)out->norm, (float4*)gauge0, (float4*)gauge1, 
						  gauge.reconstruct, (float4*)in->v, (float*)in->norm, 
						  (float4*)xv, (float*)xn, kappa, mu, a, dagger, in->bytes, in->norm_bytes);
  } else if (in->precision == QUDA_HALF_PRECISION) {
    dslash = new TwistedDslashCuda<short4,short4>((short4*)out->v, (float*)out->norm, (short4*)gauge0, (short4*)gauge1, 
						  gauge.reconstruct, (short4*)in->v, (float*)in->norm, 
						  (short4*)xv, (float*)xn, kappa, mu, a, dagger, in->bytes, in->norm_bytes);
    
  }

  dslashCuda(*dslash, regSize, parity, dagger, in->volume, block, blockFace);

  delete dslash;
  unbindGaugeTex(gauge);

  if (!dslashTuning) checkCudaError();
#else
  errorQuda("Twisted mass dslash has not been built");
#endif

}

void domainWallDslashCuda(cudaColorSpinorField *out, const FullGauge gauge, 
			  const cudaColorSpinorField *in, const int parity, const int dagger, 
			  const cudaColorSpinorField *x, const double &m_f, const double &k2,
			  const dim3 &block, const dim3 &blockFace) {

  inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef MULTI_GPU
  errorQuda("Multi-GPU domain wall not implemented\n");
#endif

  dslashParam.parity = parity;
  dslashParam.threads = in->volume;

#ifdef GPU_DOMAIN_WALL_DIRAC
  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (in->precision != gauge.precision)
    errorQuda("Mixing gauge and spinor precision not supported");

  void *xv = x ? x->v : 0;
  void *xn = x ? x->norm : 0;

  DslashCuda *dslash = 0;
  size_t regSize = sizeof(float);

  if (in->precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    dslash = new DomainWallDslashCuda<double2,double2>((double2*)out->v, (float*)out->norm, (double2*)gauge0, (double2*)gauge1, 
						       gauge.reconstruct, (double2*)in->v, (float*)in->norm, (double2*)xv, 
						       (float*)xn, m_f, k2, dagger, in->bytes, in->norm_bytes);
    regSize = sizeof(double);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->precision == QUDA_SINGLE_PRECISION) {
    dslash = new DomainWallDslashCuda<float4,float4>((float4*)out->v, (float*)out->norm, (float4*)gauge0, (float4*)gauge1, 
						     gauge.reconstruct, (float4*)in->v, (float*)in->norm, (float4*)xv, 
						     (float*)xn, m_f, k2, dagger, in->bytes, in->norm_bytes);
  } else if (in->precision == QUDA_HALF_PRECISION) {
    dslash = new DomainWallDslashCuda<short4,short4>((short4*)out->v, (float*)out->norm, (short4*)gauge0, (short4*)gauge1, 
						     gauge.reconstruct, (short4*)in->v, (float*)in->norm, (short4*)xv, 
						     (float*)xn, m_f, k2, dagger, in->bytes, in->norm_bytes);
  }

  dslashCuda(*dslash, regSize, parity, dagger, in->volume, block, blockFace);

  delete dslash;
  unbindGaugeTex(gauge);

  if (!dslashTuning) checkCudaError();
#else
  errorQuda("Domain wall dslash has not been built");
#endif

}

#define INTERIOR_KERNEL 0
#define EXTERIOR_KERNEL_X 1
#define EXTERIOR_KERNEL_Y 2
#define EXTERIOR_KERNEL_Z 3
#define EXTERIOR_KERNEL_T 4


void
initTLocation(int toffset, int tmul, int threads) 
{
  short2 tLocate = make_short2((short)toffset, (short)tmul);
  cudaMemcpyToSymbol("tLocate", &(tLocate), sizeof(short2));
  cudaMemcpyToSymbol("threads", &(threads), sizeof(threads));

}

template <typename spinorFloat, typename fatGaugeFloat, typename longGaugeFloat>
  void staggeredDslashCuda(spinorFloat *out, float *outNorm, const fatGaugeFloat *fatGauge0, const fatGaugeFloat *fatGauge1, 
			   const longGaugeFloat* longGauge0, const longGaugeFloat* longGauge1, 
			   const QudaReconstructType reconstruct, const spinorFloat *in, const float *inNorm,
			   const int parity, const int dagger, const spinorFloat *x, const float *xNorm, 
			   const double &a, const int volume, const int* Vsh, const int* dims,
			   const int length, const int ghost_length, dim3 blockDim) {
    
  dim3 interiorGridDim( (dslashParam.threads + blockDim.x -1)/blockDim.x, 1, 1);
  dim3 exteriorGridDim[4]  = {
    dim3((6*Vsh[0] + blockDim.x -1)/blockDim.x, 1, 1),
    dim3((6*Vsh[1] + blockDim.x -1)/blockDim.x, 1, 1),
    dim3((6*Vsh[2] + blockDim.x -1)/blockDim.x, 1, 1),
    dim3((6*Vsh[3] + blockDim.x -1)/blockDim.x, 1, 1)
  };
    
  int shared_bytes = blockDim.x*6*bindSpinorTex_mg(length, ghost_length, in, inNorm, x, xNorm); CUERR;

#ifdef DSLASH_PROFILE  
  cudaEvent_t interior_start, interior_stop;
  cudaEvent_t exterior_start[4], exterior_stop[4];
  struct timeval comm_start[4], comm_stop[4];
  struct timeval dslash_start, dslash_stop;
  cudaEventCreate(&interior_start);
  cudaEventCreate(&interior_stop);
  for(int i=0;i < 4;i++){
    cudaEventCreate(&exterior_start[i]);
    cudaEventCreate(&exterior_stop[i]);
  }
  cudaThreadSynchronize();
  gettimeofday(&dslash_start, NULL);  
#endif
  
  initTLocation(0, INTERIOR_KERNEL, volume);  CUERR;
  
#ifdef MULTI_GPU
  // Gather from source spinor
  face->exchangeFacesStart(*inSpinor, 1-parity, dagger, streams);
#endif

#ifdef DSLASH_PROFILE  
  cudaEventRecord(interior_start, streams[Nstream-1]);
#endif

  DSLASH(staggeredDslash, Axpy, interiorGridDim, blockDim, shared_bytes, streams[Nstream-1], out, outNorm, 
	 fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, x, xNorm, a, dslashParam); CUERR;
#ifdef DSLASH_PROFILE  
  cudaEventRecord(interior_stop, streams[Nstream-1]);
#endif

#ifdef MULTI_GPU


  int exterior_kernel_flag[4]={
    EXTERIOR_KERNEL_X, EXTERIOR_KERNEL_Y, EXTERIOR_KERNEL_Z, EXTERIOR_KERNEL_T
  };
  for(int i=0 ;i < 4;i++){
    if(!commDimPartitioned(i)){
      continue;
    }
#ifdef DSLASH_PROFILE  
    gettimeofday(&comm_start[i], NULL);
#endif
    // Finish gather and start comms
    face->exchangeFacesComms(i);
    // Wait for comms to finish, and scatter into the end zone
    face->exchangeFacesWait(*inSpinor, dagger,i);    
    
#ifdef DSLASH_PROFILE  
    gettimeofday(&comm_stop[i], NULL);
#endif

    initTLocation(dims[i]-6, exterior_kernel_flag[i] , 6*Vsh[i]);  

#ifdef DSLASH_PROFILE  
    cudaEventRecord(exterior_start[i], streams[Nstream-1]);
#endif
    DSLASH(staggeredDslash, Axpy, exteriorGridDim[i], blockDim, shared_bytes, streams[Nstream-1], out, outNorm, 
	   fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, x, xNorm, a, dslashParam); CUERR;
#ifdef DSLASH_PROFILE  
    cudaEventRecord(exterior_stop[i], streams[Nstream-1]);
#endif
  }

#ifdef DSLASH_PROFILE  
  cudaThreadSynchronize();
  gettimeofday(&dslash_stop, NULL);
  float interior_time, exterior_time[4], comm_time[4], dslash_time;
  cudaEventElapsedTime(&interior_time, interior_start, interior_stop);
  dslash_time = (dslash_stop.tv_sec - dslash_start.tv_sec)*1e+3
    + (dslash_stop.tv_usec - dslash_start.tv_usec)*1e-3;
  printfQuda("Interior kernel: %.2f ms, overall dslash time=%.2f ms\n", interior_time, dslash_time); 
  for(int i=0;i < 4;i++){
    if(commDimPartitioned(i)){
      cudaEventElapsedTime(&exterior_time[i], exterior_start[i], exterior_stop[i]);
#define TDIFF(a,b) ((a.tv_sec - b.tv_sec)*1e+3 + (a.tv_usec - b.tv_usec)*1e-3)
      comm_time[i] = TDIFF(comm_stop[i], comm_start[i]);
      printfQuda("dir=%d, comm=%.2f ms, exterior kernel=%.2f ms\n", i, comm_time[i], exterior_time[i]); 
    }
  }

  
  cudaEventDestroy(interior_start);
  cudaEventDestroy(interior_stop);
  for(int i=0;i < 4;i++){
    cudaEventDestroy(exterior_start[i]);
    cudaEventDestroy(exterior_stop[i]);
  }

#endif

#endif
}

void staggeredDslashCuda(cudaColorSpinorField *out, const FullGauge fatGauge, 
			 const FullGauge longGauge, const cudaColorSpinorField *in,
			 const int parity, const int dagger, const cudaColorSpinorField *x,
			 const double &k, const dim3 &block, const dim3 &blockFace)
{
  
  inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_STAGGERED_DIRAC

  dslashParam.parity = parity;
  dslashParam.threads = in->volume;
  for(int i=0;i < 4;i++){
    dslashParam.ghostDim[i] = commDimPartitioned(i);
    dslashParam.ghostOffset[i] = in->ghostOffset[i];
  }
  void *fatGauge0, *fatGauge1;
  void* longGauge0, *longGauge1;
  bindFatGaugeTex(fatGauge, parity, &fatGauge0, &fatGauge1);
  bindLongGaugeTex(longGauge, parity, &longGauge0, &longGauge1);
    
  if (in->precision != fatGauge.precision || in->precision != longGauge.precision){
    errorQuda("Mixing gauge and spinor precision not supported"
	      "(precision=%d, fatlinkGauge.precision=%d, longGauge.precision=%d",
	      in->precision, fatGauge.precision, longGauge.precision);
  }
    
  int Vsh[] = {
    in->x[1]*in->x[2]*in->x[3]/2,
    in->x[0]*in->x[2]*in->x[3],
    in->x[0]*in->x[1]*in->x[3],
    in->x[0]*in->x[1]*in->x[2]};

  void *xv = x ? x->v : 0;
  void *xn = x ? x->norm : 0;

  if (in->precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    staggeredDslashCuda((double2*)out->v, (float*)out->norm, (double2*)fatGauge0, (double2*)fatGauge1,
			(double2*)longGauge0, (double2*)longGauge1, longGauge.reconstruct, 
			(double2*)in->v, (float*)in->norm, parity, dagger, 
			(double2*)xv, (float*)x, k, in->volume, Vsh, 
			in->x, in->length, in->ghost_length, block);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->precision == QUDA_SINGLE_PRECISION) {
    staggeredDslashCuda((float2*)out->v, (float*)out->norm, (float2*)fatGauge0, (float2*)fatGauge1,
			(float4*)longGauge0, (float4*)longGauge1, longGauge.reconstruct, 
			(float2*)in->v, (float*)in->norm, parity, dagger, 
			(float2*)xv, (float*)xn, k, in->volume, Vsh, 
			in->x, in->length, in->ghost_length, block);
  } else if (in->precision == QUDA_HALF_PRECISION) {	
    staggeredDslashCuda((short2*)out->v, (float*)out->norm, (short2*)fatGauge0, (short2*)fatGauge1,
			(short4*)longGauge0, (short4*)longGauge1, longGauge.reconstruct, 
			(short2*)in->v, (float*)in->norm, parity, dagger, 
			(short2*)xv, (float*)xn, k, in->volume, Vsh, 
			in->x, in->length, in->ghost_length, block);
  }

  if (!dslashTuning) checkCudaError();
  
#else
  errorQuda("Staggered dslash has not been built");
#endif  
}


template <typename spinorFloat, typename cloverFloat>
void cloverCuda(spinorFloat *out, float *outNorm, const cloverFloat *clover,
		const float *cloverNorm, const spinorFloat *in, const float *inNorm, 
		const size_t bytes, const size_t norm_bytes, const dim3 blockDim)
{
  dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*bindSpinorTex(bytes, norm_bytes, in, inNorm);
  cloverKernel<<<gridDim, blockDim, shared_bytes>>> 
    (out, outNorm, clover, cloverNorm, in, inNorm, dslashParam);
  unbindSpinorTex(in, inNorm);
}

void cloverCuda(cudaColorSpinorField *out, const FullGauge gauge, const FullClover clover, 
		const cudaColorSpinorField *in, const int parity, const dim3 &blockDim) {

  dslashParam.parity = parity;
  dslashParam.tOffset = 0;
  dslashParam.tMul = 1;
  dslashParam.threads = in->volume;

#ifdef GPU_WILSON_DIRAC
  void *cloverP, *cloverNormP;
  QudaPrecision clover_prec = bindCloverTex(clover, parity, &cloverP, &cloverNormP);

  if (in->precision != clover_prec)
    errorQuda("Mixing clover and spinor precision not supported");

  if (in->precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    cloverCuda((double2*)out->v, (float*)out->norm, (double2*)cloverP, 
	       (float*)cloverNormP, (double2*)in->v, (float*)in->norm, 
	       in->bytes, in->norm_bytes, blockDim);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->precision == QUDA_SINGLE_PRECISION) {
    cloverCuda((float4*)out->v, (float*)out->norm, (float4*)cloverP, 
	       (float*)cloverNormP, (float4*)in->v, (float*)in->norm,
	       in->bytes, in->norm_bytes, blockDim);
  } else if (in->precision == QUDA_HALF_PRECISION) {
    cloverCuda((short4*)out->v, (float*)out->norm, (short4*)cloverP, 
	       (float*)cloverNormP, (short4*)in->v, (float*)in->norm, 
	       in->bytes, in->norm_bytes, blockDim);
  }
  unbindCloverTex(clover);

  if (!dslashTuning) checkCudaError();
#else
  errorQuda("Clover dslash has not been built");
#endif

}
// FIXME: twist kernel cannot be issued asynchronously because of texture unbinding
template <typename spinorFloat>
void twistGamma5Cuda(spinorFloat *out, float *outNorm, const spinorFloat *in, 
		     const float *inNorm, const int dagger, const double &kappa, 
		     const double &mu, const size_t bytes, const size_t norm_bytes, 
		     const QudaTwistGamma5Type twist, dim3 blockDim)
{
  dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);

  double a=0.0, b=0.0;
  setTwistParam(a, b, kappa, mu, dagger, twist);

  bindSpinorTex(bytes, norm_bytes, in, inNorm);
  twistGamma5Kernel<<<gridDim, blockDim, 0>>> (out, outNorm, a, b, dslashParam);
  unbindSpinorTex(in, inNorm);
}

void twistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		     const int dagger, const double &kappa, const double &mu,
		     const QudaTwistGamma5Type twist, const dim3 &block) {

  dslashParam.tOffset = 0;
  dslashParam.tMul = 1;
  dslashParam.threads = in->Volume();

#ifdef GPU_TWISTED_MASS_DIRAC
  if (in->precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    twistGamma5Cuda((double2*)out->v, (float*)out->norm, 
		    (double2*)in->v, (float*)in->norm, 
		    dagger, kappa, mu, in->bytes, 
		    in->norm_bytes, twist, block);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->precision == QUDA_SINGLE_PRECISION) {
    twistGamma5Cuda((float4*)out->v, (float*)out->norm,
		    (float4*)in->v, (float*)in->norm, 
		    dagger, kappa, mu, in->bytes, 
		    in->norm_bytes, twist, block);
  } else if (in->precision == QUDA_HALF_PRECISION) {
    twistGamma5Cuda((short4*)out->v, (float*)out->norm,
		    (short4*)in->v, (float*)in->norm, 
		    dagger, kappa, mu, in->bytes, 
		    in->norm_bytes, twist, block);
  }
  if (!dslashTuning) checkCudaError();
#else
  errorQuda("Twisted mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
}


#include "misc_helpers.cu"


#if defined(GPU_FATLINK)||defined(GPU_GAUGE_FORCE)|| defined(GPU_FERMION_FORCE)
#include <force_common.h>
#include "force_kernel_common.cu"
#endif

#ifdef GPU_FATLINK
#include "llfat_quda.cu"
#endif

#ifdef GPU_GAUGE_FORCE
#include "gauge_force_quda.cu"
#endif

#ifdef GPU_FERMION_FORCE
#include "fermion_force_quda.cu"
#endif
