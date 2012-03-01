#include <stdlib.h>
#include <stdio.h>

#include <color_spinor_field.h>
#include <clover_field.h>

#define BLOCK_DIM 64

// these control the Wilson-type actions
//#define DIRECT_ACCESS_LINK
//#define DIRECT_ACCESS_WILSON_SPINOR
//#define DIRECT_ACCESS_WILSON_ACCUM
//#define DIRECT_ACCESS_WILSON_INTER
//#define DIRECT_ACCESS_WILSON_PACK_SPINOR
//#define DIRECT_ACCESS_CLOVER

//these are access control for staggered action
#if (__COMPUTE_CAPABILITY__ >= 200)
//#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#else
#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
//#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#endif

#include <quda_internal.h>
#include <dslash_quda.h>
#include <sys/time.h>

enum KernelType {
  INTERIOR_KERNEL = 5,
  EXTERIOR_KERNEL_X = 0,
  EXTERIOR_KERNEL_Y = 1,
  EXTERIOR_KERNEL_Z = 2,
  EXTERIOR_KERNEL_T = 3
};

struct DslashParam {
  int threads; // the desired number of active threads
  int parity;  // Even-Odd or Odd-Even
  int commDim[QUDA_MAX_DIM]; // Whether to do comms or not
  int ghostDim[QUDA_MAX_DIM]; // Whether a ghost zone has been allocated for a given dimension
  int ghostOffset[QUDA_MAX_DIM];
  int ghostNormOffset[QUDA_MAX_DIM];
  KernelType kernel_type; //is it INTERIOR_KERNEL, EXTERIOR_KERNEL_X/Y/Z/T
};

// determines whether the temporal ghost zones are packed with a gather kernel,
// as opposed to multiple calls to cudaMemcpy()
bool kernelPackT = false;
bool dslash_launch = true;
bool getDslashLaunch() { return dslash_launch; }

DslashParam dslashParam;

// these are set in initDslashConst
int Vspatial;

static cudaEvent_t packEnd[Nstream];
static cudaEvent_t gatherStart[Nstream];
static cudaEvent_t gatherEnd[Nstream];
static cudaEvent_t scatterStart[Nstream];
static cudaEvent_t scatterEnd[Nstream];

static struct timeval dslashStart_h;
#ifdef MULTI_GPU
static struct timeval commsStart[Nstream];
static struct timeval commsEnd[Nstream];
#endif

// these events are only used for profiling
#ifdef DSLASH_PROFILING
#define DSLASH_TIME_PROFILE() dslashTimeProfile()

static cudaEvent_t dslashStart;
static cudaEvent_t dslashEnd;
static cudaEvent_t packStart[Nstream];
static cudaEvent_t kernelStart[Nstream];
static cudaEvent_t kernelEnd[Nstream];

// dimension 2 because we want absolute and relative
float packTime[Nstream][2];
float gatherTime[Nstream][2];
float commsTime[Nstream][2];
float scatterTime[Nstream][2];
float kernelTime[Nstream][2];
float dslashTime;
#define CUDA_EVENT_RECORD(a,b) cudaEventRecord(a,b)
#else
#define CUDA_EVENT_RECORD(a,b)
#define DSLASH_TIME_PROFILE()
#endif

FaceBuffer *face;
cudaColorSpinorField *inSpinor;

#include <dslash_textures.h>
#include <dslash_constants.h>

static inline __device__ float short2float(short a) {
  return (float)a/MAX_SHORT;
}

static inline __device__ short float2short(float c, float a) {
  return (short)(a*c*MAX_SHORT);
}

static inline __device__ short2 float22short2(float c, float2 a) {
  return make_short2((short)(a.x*c*MAX_SHORT), (short)(a.y*c*MAX_SHORT));
}

#if defined(DIRECT_ACCESS_LINK) || defined(DIRECT_ACCESS_WILSON_SPINOR) || \
  defined(DIRECT_ACCESS_WILSON_ACCUM) || defined(DIRECT_ACCESS_WILSON_PACK_SPINOR) || \
  defined(DIRECT_ACCESS_WILSON_INTER) || defined(DIRECT_ACCESS_WILSON_PACK_SPINOR)

static inline __device__ short4 float42short4(float c, float4 a) {
  return make_short4(float2short(c, a.x), float2short(c, a.y), float2short(c, a.z), float2short(c, a.w));
}

static inline __device__ float4 short42float4(short4 a) {
  return make_float4(short2float(a.x), short2float(a.y), short2float(a.z), short2float(a.w));
}

static inline __device__ float2 short22float2(short2 a) {
  return make_float2(short2float(a.x), short2float(a.y));
}
#endif // DIRECT_ACCESS inclusions

// dslashTuning = QUDA_TUNE_YES turns off error checking
static QudaTune dslashTuning = QUDA_TUNE_NO;

void setDslashTuning(QudaTune tune)
{
  dslashTuning = tune;
}

#include <pack_face_def.h>        // kernels for packing the ghost zones and general indexing
#include <staggered_dslash_def.h> // staggered Dslash kernels
#include <wilson_dslash_def.h>    // Wilson Dslash kernels (including clover)
#include <dw_dslash_def.h>        // Domain Wall kernels
#include <tm_dslash_def.h>        // Twisted Mass kernels
#include <tm_core.h>              // solo twisted mass kernel
#include <clover_def.h>           // kernels for applying the clover term alone

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#ifndef CLOVER_SHARED_FLOATS_PER_THREAD
#define CLOVER_SHARED_FLOATS_PER_THREAD 0
#endif

#include <blas_quda.h>
#include <face_quda.h>


__global__ void dummyKernel() {
  // do nothing
}

void initCache() {

#if (__COMPUTE_CAPABILITY__ >= 200)

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

#define MORE_GENERIC_DSLASH(FUNC, DAG, X, kernel_type, gridDim, blockDim, shared, stream, param,  ...)            \
  if (x==0) {                                                                                                     \
    if (reconstruct == QUDA_RECONSTRUCT_NO) {                                                                     \
      FUNC ## 18 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param);     \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {                                                              \
      FUNC ## 12 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param);     \
    } else {                                                                                                      \
      FUNC ## 8 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param);       \
    }                                                                                                             \
  } else {                                                                                                        \
    if (reconstruct == QUDA_RECONSTRUCT_NO) {                                                                     \
      FUNC ## 18 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {                                                              \
      FUNC ## 12 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {                                                               \
      FUNC ## 8 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }                                                                                                             \
  }

#ifndef MULTI_GPU

#define GENERIC_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...)                          \
  switch(param.kernel_type) {						                                      \
  case INTERIOR_KERNEL:							                                      \
    MORE_GENERIC_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								                                      \
  default:								                                      \
    errorQuda("KernelType %d not defined for single GPU", param.kernel_type);                                 \
  }

#else

#define GENERIC_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...)                            \
  switch(param.kernel_type) {						                                        \
  case INTERIOR_KERNEL:							                                        \
    MORE_GENERIC_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL,   gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								                                        \
  case EXTERIOR_KERNEL_X:							                                \
    MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_X, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								                                        \
  case EXTERIOR_KERNEL_Y:							                                \
    MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Y, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								                                        \
  case EXTERIOR_KERNEL_Z:							                                \
    MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Z, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								                                        \
  case EXTERIOR_KERNEL_T:							                                \
    MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_T, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								                                        \
  }

#endif

// macro used for dslash types with dagger kernel defined (Wilson, domain wall, etc.)
#define DSLASH(FUNC, gridDim, blockDim, shared, stream, param, ...)	\
  if (!dagger) {							\
    GENERIC_DSLASH(FUNC, , Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
  } else {								\
    GENERIC_DSLASH(FUNC, Dagger, Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
 }

// macro used for staggered dslash
#define STAGGERED_DSLASH(gridDim, blockDim, shared, stream, param, ...)	\
    GENERIC_DSLASH(staggeredDslash, , Axpy, gridDim, blockDim, shared, stream, param, __VA_ARGS__)


// Use an abstract class interface to drive the different CUDA dslash
// kernels.  All parameters are curried into the derived classes to
// allow a simple interface.
class DslashCuda {
public:
  DslashCuda() { ; }
  virtual ~DslashCuda() { ; }
  virtual void apply(const dim3 &blockDim, const int shared_bytes, const cudaStream_t &stream) = 0;
  virtual int SharedPerThread() = 0;
  virtual int Nface() { return 2; }
};


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
    bindSpinorTex(bytes, norm_bytes, in, inNorm, out, outNorm, x, xNorm); 
  }

  virtual ~WilsonDslashCuda() { unbindSpinorTex(in, inNorm, out, outNorm, x, xNorm); }

  void apply(const dim3 &blockDim, const int shared_bytes, const cudaStream_t &stream) {
    dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);
    DSLASH(dslash, gridDim, blockDim, shared_bytes, stream, dslashParam,
	   out, outNorm, gauge0, gauge1, in, inNorm, x, xNorm, a);
  }

  int SharedPerThread() { return DSLASH_SHARED_FLOATS_PER_THREAD; };
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
    bindSpinorTex(bytes, norm_bytes, in, inNorm, out, outNorm, x, xNorm); 
  }
  virtual ~CloverDslashCuda() { unbindSpinorTex(in, inNorm, out, outNorm, x, xNorm); }

  void apply(const dim3 &blockDim, const int shared_bytes, const cudaStream_t &stream) {
    dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);
    DSLASH(cloverDslash, gridDim, blockDim, shared_bytes, stream, dslashParam,
	   out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, x, xNorm, a);
  }

  int SharedPerThread() { return DSLASH_SHARED_FLOATS_PER_THREAD; };
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
    bindSpinorTex(bytes, norm_bytes, in, inNorm, out, outNorm, x, xNorm); 
    setTwistParam(a, b, kappa, mu, dagger, QUDA_TWIST_GAMMA5_INVERSE);
    if (x) b *= k;
  }
  virtual ~TwistedDslashCuda() { unbindSpinorTex(in, inNorm, out, outNorm, x, xNorm); }

  void apply(const dim3 &blockDim, const int shared_bytes, const cudaStream_t &stream) {
    dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);
    DSLASH(twistedMassDslash, gridDim, blockDim, shared_bytes, stream, dslashParam,
	   out, outNorm, gauge0, gauge1, in, inNorm, a, b, x, xNorm);
  }

  int SharedPerThread() { return DSLASH_SHARED_FLOATS_PER_THREAD; };
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
    bindSpinorTex(bytes, norm_bytes, in, inNorm, out, outNorm, x, xNorm); 
  }
  virtual ~DomainWallDslashCuda() { unbindSpinorTex(in, inNorm, out, outNorm, x, xNorm); }

  void apply(const dim3 &blockDim, const int shared_bytes, const cudaStream_t &stream) {
    dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);
    DSLASH(domainWallDslash, gridDim, blockDim, shared_bytes, stream, dslashParam,
    	   out, outNorm, gauge0, gauge1, in, inNorm, mferm, x, xNorm, a);
  }

  int SharedPerThread() { return 0; };
};

template <typename sFloat, typename fatGFloat, typename longGFloat>
class StaggeredDslashCuda : public DslashCuda {

private:
  sFloat *out;
  float *outNorm;
  const sFloat *in, *x;
  const float *inNorm, *xNorm;
  const fatGFloat *fat0, *fat1;
  const longGFloat *long0, *long1;
  const QudaReconstructType reconstruct;
  const int dagger;
  const double a;

public:
  StaggeredDslashCuda(sFloat *out, float *outNorm, const fatGFloat *fat0, const fatGFloat *fat1,
		      const longGFloat *long0, const longGFloat *long1,
		      const QudaReconstructType reconstruct, const sFloat *in, 
		      const float *inNorm, const sFloat *x, const float *xNorm, const double a,
		      const int dagger, const size_t bytes, const size_t norm_bytes) :
    DslashCuda(), out(out), outNorm(outNorm), fat0(fat0), fat1(fat1), long0(long0), long1(long1),
    in(in), inNorm(inNorm), reconstruct(reconstruct), dagger(dagger), x(x), xNorm(xNorm), a(a) { 
    bindSpinorTex(bytes, norm_bytes, in, inNorm, out, outNorm, x, xNorm); 
  }

  virtual ~StaggeredDslashCuda() { unbindSpinorTex(in, inNorm, out, outNorm, x, xNorm); }

  void apply(const dim3 &blockDim, const int shared_bytes, const cudaStream_t &stream) {
    dim3 gridDim((dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);
    STAGGERED_DSLASH(gridDim, blockDim, shared_bytes, stream, dslashParam,
		     out, outNorm, fat0, fat1, long0, long1, in, inNorm, x, xNorm, a);
  }

  int SharedPerThread() { return 6; }
  int Nface() { return 6; }
};

#ifdef DSLASH_PROFILING

#define TDIFF(a,b) 1e3*(b.tv_sec - a.tv_sec + 1e-6*(b.tv_usec - a.tv_usec))

void dslashTimeProfile() {

  cudaEventSynchronize(dslashEnd);
  float runTime;
  cudaEventElapsedTime(&runTime, dslashStart, dslashEnd);
  dslashTime += runTime;

  for (int i=4; i>=0; i--) {
    if (!dslashParam.commDim[i] && i<4) continue;

    // kernel timing
    cudaEventElapsedTime(&runTime, dslashStart, kernelStart[2*i]);
    kernelTime[2*i][0] += runTime; // start time
    cudaEventElapsedTime(&runTime, dslashStart, kernelEnd[2*i]);
    kernelTime[2*i][1] += runTime; // end time
  }
      
#ifdef MULTI_GPU
  for (int i=3; i>=0; i--) {
    if (!dslashParam.commDim[i]) continue;

    for (int dir = 0; dir < 2; dir ++) {
      // pack timing
      cudaEventElapsedTime(&runTime, dslashStart, packStart[2*i+dir]);
      packTime[2*i+dir][0] += runTime; // start time
      cudaEventElapsedTime(&runTime, dslashStart, packEnd[2*i+dir]);
      packTime[2*i+dir][1] += runTime; // end time
  
      // gather timing
      cudaEventElapsedTime(&runTime, dslashStart, gatherStart[2*i+dir]);
      gatherTime[2*i+dir][0] += runTime; // start time
      cudaEventElapsedTime(&runTime, dslashStart, gatherEnd[2*i+dir]);
      gatherTime[2*i+dir][1] += runTime; // end time
      
      // comms timing
      runTime = TDIFF(dslashStart_h, commsStart[2*i+dir]);
      commsTime[2*i+dir][0] += runTime; // start time
      runTime = TDIFF(dslashStart_h, commsEnd[2*i+dir]);
      commsTime[2*i+dir][1] += runTime; // end time

      // scatter timing
      cudaEventElapsedTime(&runTime, dslashStart, scatterStart[2*i+dir]);
      scatterTime[2*i+dir][0] += runTime; // start time
      cudaEventElapsedTime(&runTime, dslashStart, scatterEnd[2*i+dir]);
      scatterTime[2*i+dir][1] += runTime; // end time
    }
  }
#endif

}

void printDslashProfile() {
  
  printfQuda("Total Dslash time = %6.2f\n", dslashTime);

  char dimstr[8][8] = {"X-", "X+", "Y-", "Y+", "Z-", "Z+", "T-", "T+"};

  printfQuda("     %13s %13s %13s %13s %13s\n", "Pack", "Gather", "Comms", "Scatter", "Kernel");
  printfQuda("         %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s\n", 
	     "Start", "End", "Start", "End", "Start", "End", "Start", "End", "Start", "End");

  printfQuda("%8s %55s %6.2f %6.2f\n", "Interior", "", kernelTime[8][0], kernelTime[8][1]);
      
  for (int i=3; i>=0; i--) {
    if (!dslashParam.commDim[i]) continue;

    for (int dir = 0; dir < 2; dir ++) {
      printfQuda("%8s ", dimstr[2*i+dir]);
#ifdef MULTI_GPU
      printfQuda("%6.2f %6.2f ", packTime[2*i+dir][0], packTime[2*i+dir][1]);
      printfQuda("%6.2f %6.2f ", gatherTime[2*i+dir][0], gatherTime[2*i+dir][1]);
      printfQuda("%6.2f %6.2f ", commsTime[2*i+dir][0], commsTime[2*i+dir][1]);
      printfQuda("%6.2f %6.2f ", scatterTime[2*i+dir][0], scatterTime[2*i+dir][1]);
#endif

      if (dir==0) printfQuda("%6.2f %6.2f\n", kernelTime[2*i][0], kernelTime[2*i][1]);
      else printfQuda("\n");
    }
  }

}
#endif

bool checkLaunchParam(const int shared_bytes) {

  bool launch;

  // only launch if not over-allocating shared memory
  // hard code for the moment until we have more robust cache setting mechanism
  if (shared_bytes > 16384 /*deviceProp.sharedMemPerBlock*/) {
    launch = false;
  } else {
    launch = true;
  }

  return launch;
}

void dslashCuda(DslashCuda &dslash, const size_t regSize, const int parity, const int dagger, 
		const int volume, const int *faceVolumeCB, const TuneParam *tune) {

  dslashParam.parity = parity;
  dslashParam.kernel_type = INTERIOR_KERNEL;
  dslashParam.threads = volume;

  CUDA_EVENT_RECORD(dslashStart, 0);
  gettimeofday(&dslashStart_h, NULL);

#ifdef MULTI_GPU
  for(int i = 3; i >=0; i--){
    if (!dslashParam.commDim[i]) continue;

    // Record the start of the packing
    CUDA_EVENT_RECORD(packStart[2*i+0], streams[Nstream-1]);
    CUDA_EVENT_RECORD(packStart[2*i+1], streams[Nstream-1]);

    // Initialize pack from source spinor
    face->pack(*inSpinor, 1-parity, dagger, i, streams);
    
    // Record the end of the packing
    cudaEventRecord(packEnd[2*i+0], streams[Nstream-1]);
    cudaEventRecord(packEnd[2*i+1], streams[Nstream-1]);
  }

  for(int i = 3; i >=0; i--){
    if (!dslashParam.commDim[i]) continue;

    for (int dir=1; dir>=0; dir--) {
      cudaStreamWaitEvent(streams[2*i+dir], packEnd[2*i+dir], 0);

      // Record the start of the gathering
      CUDA_EVENT_RECORD(gatherStart[2*i+dir], streams[2*i+dir]);

      // Initialize host transfer from source spinor
      face->gather(*inSpinor, dagger, 2*i+dir);

      // Record the end of the gathering
      cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]);
    }
  }
#endif

  int shared_bytes = tune[0].block.x*dslash.SharedPerThread()*regSize;
  shared_bytes = tune[0].shared_bytes > shared_bytes ? tune[0].shared_bytes : shared_bytes;
  dslash_launch = true;
  if (checkLaunchParam(shared_bytes)) {
    CUDA_EVENT_RECORD(kernelStart[Nstream-1], streams[Nstream-1]);
    dslash.apply(tune[0].block, shared_bytes, streams[Nstream-1]);
    CUDA_EVENT_RECORD(kernelEnd[Nstream-1], streams[Nstream-1]);
  } else {
    dslash_launch = false;
  }

#ifdef MULTI_GPU

  int completeSum = 0;
  int gatherCompleted[Nstream];
  int commsCompleted[Nstream];
  for (int i=0; i<Nstream; i++) {
    gatherCompleted[i] = 0;
    commsCompleted[i] = 0;
  }

  for (int i=3; i>=0; i--) 
    if (dslashParam.commDim[i]) {
      gatherCompleted[2*i+2] = 1;
      commsCompleted[2*i+2] = 1;
      break;
    }
  
  int commDimTotal = 0;
  for (int i=0; i<4; i++) commDimTotal += dslashParam.commDim[i];
  commDimTotal *= 4; // 2 from pipe length, 2 from direction

  while (completeSum < commDimTotal) {
    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;
      
      for (int dir=1; dir>=0; dir--) {
	
	// Query if gather has completed
	if (!gatherCompleted[2*i+dir] && gatherCompleted[2*i+dir+1]) { 
	  if (cudaSuccess == cudaEventQuery(gatherEnd[2*i+dir])) {
	    gatherCompleted[2*i+dir] = 1;
	    completeSum++;
	    gettimeofday(&commsStart[2*i+dir], NULL);
	    face->commsStart(2*i+dir);
	  }
	}
	
	if (!commsCompleted[2*i+dir] && commsCompleted[2*i+dir+1] &&
	    gatherCompleted[2*i+dir]) {
	  if (face->commsQuery(2*i+dir)) { // Query if comms has finished
	    commsCompleted[2*i+dir] = 1;
	    completeSum++;
	    gettimeofday(&commsEnd[2*i+dir], NULL);
	    
	    // Record the end of the scattering
	    CUDA_EVENT_RECORD(scatterStart[2*i+dir], streams[2*i+dir]);
	    
	    // Scatter into the end zone
	    face->scatter(*inSpinor, dagger, 2*i+dir);
	    
	    // Record the end of the scattering
	    cudaEventRecord(scatterEnd[2*i+dir], streams[2*i+dir]);
	  }
	}

      }
    }
    
  }

  for (int i=3; i>=0; i--) {
    if (!dslashParam.commDim[i]) continue;

    dslashParam.kernel_type = static_cast<KernelType>(i);
    dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces

    // wait for scattering to finish and then launch dslash
    cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i], 0);
    cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i+1], 0);

    int shared_bytes = tune[i+1].block.x*dslash.SharedPerThread()*regSize;
    shared_bytes = tune[i+1].shared_bytes > shared_bytes ? tune[i+1].shared_bytes : shared_bytes;
    if (checkLaunchParam(shared_bytes)) {    
      CUDA_EVENT_RECORD(kernelStart[2*i], streams[Nstream-1]);
      dslash.apply(tune[i+1].block, shared_bytes, streams[Nstream-1]); // all faces use this stream
      CUDA_EVENT_RECORD(kernelEnd[2*i], streams[Nstream-1]);
    } else {
      dslash_launch = false;
    }
  }

  CUDA_EVENT_RECORD(dslashEnd, 0);
  if (!dslashTuning) DSLASH_TIME_PROFILE();

#endif // MULTI_GPU
}

// Wilson wrappers
void wilsonDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in,
		      const int parity, const int dagger, const cudaColorSpinorField *x,
		      const double &k, const TuneParam *tune, const int *commOverride) {

  inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_WILSON_DIRAC
  int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
  for(int i=0;i<4;i++){
    dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
    dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
    dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
    dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
  }

  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (in->Precision() != gauge.Precision())
    errorQuda("Mixing gauge %d and spinor %d precision not supported", 
	      gauge.Precision(), in->Precision());

  const void *xv = (x ? x->V() : 0);
  const void *xn = (x ? x->Norm() : 0);

  DslashCuda *dslash = 0;
  size_t regSize = sizeof(float);
  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    dslash = new WilsonDslashCuda<double2, double2>((double2*)out->V(), (float*)out->Norm(), 
						    (double2*)gauge0, (double2*)gauge1, 
						    gauge.Reconstruct(), (double2*)in->V(), 
						    (float*)in->Norm(), (double2*)xv, (float*)xn,
						    k, dagger, in->Bytes(), in->NormBytes());
    regSize = sizeof(double);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    dslash = new WilsonDslashCuda<float4, float4>((float4*)out->V(), (float*)out->Norm(), (float4*)gauge0, (float4*)gauge1,
						  gauge.Reconstruct(), (float4*)in->V(), (float*)in->Norm(), 
						  (float4*)xv, (float*)xn, k, dagger, in->Bytes(), in->NormBytes());
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    dslash = new WilsonDslashCuda<short4, short4>((short4*)out->V(), (float*)out->Norm(), (short4*)gauge0, (short4*)gauge1,
						  gauge.Reconstruct(), (short4*)in->V(), (float*)in->Norm(),
						  (short4*)xv, (float*)xn, k, dagger, in->Bytes(), in->NormBytes());
  }
  dslashCuda(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace(), tune);

  delete dslash;
  unbindGaugeTex(gauge);

  if (!dslashTuning) checkCudaError();
#else
  errorQuda("Wilson dslash has not been built");
#endif // GPU_WILSON_DIRAC

}

void cloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover cloverInv,
		      const cudaColorSpinorField *in, const int parity, const int dagger, 
		      const cudaColorSpinorField *x, const double &a,
		      const TuneParam *tune, const int *commOverride) {

  inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_CLOVER_DIRAC
  int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
  for(int i=0;i<4;i++){
    dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
    dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
    dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
    dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
  }

  void *cloverP, *cloverNormP;
  QudaPrecision clover_prec = bindCloverTex(cloverInv, parity, &cloverP, &cloverNormP);

  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (in->Precision() != gauge.Precision())
    errorQuda("Mixing gauge and spinor precision not supported");

  if (in->Precision() != clover_prec)
    errorQuda("Mixing clover and spinor precision not supported");

  const void *xv = x ? x->V() : 0;
  const void *xn = x ? x->Norm() : 0;

  DslashCuda *dslash = 0;
  size_t regSize = sizeof(float);

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    dslash = new CloverDslashCuda<double2, double2, double2>((double2*)out->V(), (float*)out->Norm(), (double2*)gauge0, 
							     (double2*)gauge1, gauge.Reconstruct(), (double2*)cloverP, 
							     (float*)cloverNormP, (double2*)in->V(), (float*)in->Norm(),
							     (double2*)xv, (float*)xn, a, dagger, in->Bytes(), in->NormBytes());
    regSize = sizeof(double);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    dslash = new CloverDslashCuda<float4, float4, float4>((float4*)out->V(), (float*)out->Norm(), (float4*)gauge0, 
							  (float4*)gauge1, gauge.Reconstruct(), (float4*)cloverP, 
							  (float*)cloverNormP, (float4*)in->V(), (float*)in->Norm(), 
							  (float4*)xv, (float*)xn, a, dagger, in->Bytes(), in->NormBytes());
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    dslash = new CloverDslashCuda<short4, short4, short4>((short4*)out->V(), (float*)out->Norm(), (short4*)gauge0, 
							  (short4*)gauge1, gauge.Reconstruct(), (short4*)cloverP, 
							  (float*)cloverNormP, (short4*)in->V(), (float*)in->Norm(), 
							  (short4*)xv, (float*)xn, a, dagger, in->Bytes(), in->NormBytes());
  }

  dslashCuda(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace(), tune);

  delete dslash;
  unbindGaugeTex(gauge);
  unbindCloverTex(cloverInv);

  if (!dslashTuning) checkCudaError();
#else
  errorQuda("Clover dslash has not been built");
#endif

}


void twistedMassDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			   const cudaColorSpinorField *in, const int parity, const int dagger, 
			   const cudaColorSpinorField *x, const double &kappa, const double &mu, 
			   const double &a, const TuneParam *tune, const int *commOverride) {

  inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_TWISTED_MASS_DIRAC
  int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
  for(int i=0;i<4;i++){
    dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
    dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
    dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
    dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
  }

  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (in->Precision() != gauge.Precision())
    errorQuda("Mixing gauge and spinor precision not supported");

  const void *xv = x ? x->V() : 0;
  const void *xn = x ? x->Norm() : 0;

  DslashCuda *dslash = 0;
  size_t regSize = sizeof(float);

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    dslash = new TwistedDslashCuda<double2,double2>((double2*)out->V(), (float*)out->Norm(), (double2*)gauge0, 
						    (double2*)gauge1, gauge.Reconstruct(), (double2*)in->V(), 
						    (float*)in->Norm(), (double2*)xv, (float*)xn, 
						    kappa, mu, a, dagger, in->Bytes(), in->NormBytes());
    regSize = sizeof(double);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    dslash = new TwistedDslashCuda<float4,float4>((float4*)out->V(), (float*)out->Norm(), (float4*)gauge0, (float4*)gauge1, 
						  gauge.Reconstruct(), (float4*)in->V(), (float*)in->Norm(), 
						  (float4*)xv, (float*)xn, kappa, mu, a, dagger, in->Bytes(), in->NormBytes());
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    dslash = new TwistedDslashCuda<short4,short4>((short4*)out->V(), (float*)out->Norm(), (short4*)gauge0, (short4*)gauge1, 
						  gauge.Reconstruct(), (short4*)in->V(), (float*)in->Norm(), 
						  (short4*)xv, (float*)xn, kappa, mu, a, dagger, in->Bytes(), in->NormBytes());
    
  }

  dslashCuda(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace(), tune);

  delete dslash;
  unbindGaugeTex(gauge);

  if (!dslashTuning) checkCudaError();
#else
  errorQuda("Twisted mass dslash has not been built");
#endif

}

void domainWallDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			  const cudaColorSpinorField *in, const int parity, const int dagger, 
			  const cudaColorSpinorField *x, const double &m_f, const double &k2,
			  const TuneParam *tune) {

  inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef MULTI_GPU
  errorQuda("Multi-GPU domain wall not implemented\n");
#endif

  dslashParam.parity = parity;
  dslashParam.threads = in->Volume();

#ifdef GPU_DOMAIN_WALL_DIRAC
  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (in->Precision() != gauge.Precision())
    errorQuda("Mixing gauge and spinor precision not supported");

  const void *xv = x ? x->V() : 0;
  const void *xn = x ? x->Norm() : 0;

  DslashCuda *dslash = 0;
  size_t regSize = sizeof(float);

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    dslash = new DomainWallDslashCuda<double2,double2>((double2*)out->V(), (float*)out->Norm(), (double2*)gauge0, (double2*)gauge1, 
						       gauge.Reconstruct(), (double2*)in->V(), (float*)in->Norm(), (double2*)xv, 
						       (float*)xn, m_f, k2, dagger, in->Bytes(), in->NormBytes());
    regSize = sizeof(double);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    dslash = new DomainWallDslashCuda<float4,float4>((float4*)out->V(), (float*)out->Norm(), (float4*)gauge0, (float4*)gauge1, 
						     gauge.Reconstruct(), (float4*)in->V(), (float*)in->Norm(), (float4*)xv, 
						     (float*)xn, m_f, k2, dagger, in->Bytes(), in->NormBytes());
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    dslash = new DomainWallDslashCuda<short4,short4>((short4*)out->V(), (float*)out->Norm(), (short4*)gauge0, (short4*)gauge1, 
						     gauge.Reconstruct(), (short4*)in->V(), (float*)in->Norm(), (short4*)xv, 
						     (float*)xn, m_f, k2, dagger, in->Bytes(), in->NormBytes());
  }

  dslashCuda(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace(), tune);

  delete dslash;
  unbindGaugeTex(gauge);

  if (!dslashTuning) checkCudaError();
#else
  errorQuda("Domain wall dslash has not been built");
#endif

}

void staggeredDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &fatGauge, 
			 const cudaGaugeField &longGauge, const cudaColorSpinorField *in,
			 const int parity, const int dagger, const cudaColorSpinorField *x,
			 const double &k, const TuneParam *tune, const int *commOverride)
{
  
  inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_STAGGERED_DIRAC
  int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code

  dslashParam.parity = parity;
  dslashParam.threads = in->Volume();
  for(int i=0;i<4;i++){
    dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
    dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
    dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
    dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
  }
  void *fatGauge0, *fatGauge1;
  void* longGauge0, *longGauge1;
  bindFatGaugeTex(fatGauge, parity, &fatGauge0, &fatGauge1);
  bindLongGaugeTex(longGauge, parity, &longGauge0, &longGauge1);
    
  if (in->Precision() != fatGauge.Precision() || in->Precision() != longGauge.Precision()){
    errorQuda("Mixing gauge and spinor precision not supported"
	      "(precision=%d, fatlinkGauge.precision=%d, longGauge.precision=%d",
	      in->Precision(), fatGauge.Precision(), longGauge.Precision());
  }
    
  const void *xv = x ? x->V() : 0;
  const void *xn = x ? x->Norm() : 0;

  DslashCuda *dslash = 0;
  size_t regSize = sizeof(float);

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    dslash = new StaggeredDslashCuda<double2, double2, double2>((double2*)out->V(), (float*)out->Norm(), 
								(double2*)fatGauge0, (double2*)fatGauge1,
								(double2*)longGauge0, (double2*)longGauge1, 
								longGauge.Reconstruct(), (double2*)in->V(), 
								(float*)in->Norm(), (double2*)xv, (float*)xn, 
								k, dagger, in->Bytes(), in->NormBytes());
    regSize = sizeof(double);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    dslash = new StaggeredDslashCuda<float2, float2, float4>((float2*)out->V(), (float*)out->Norm(), 
							     (float2*)fatGauge0, (float2*)fatGauge1,
							     (float4*)longGauge0, (float4*)longGauge1, 
							     longGauge.Reconstruct(), (float2*)in->V(),
							     (float*)in->Norm(), (float2*)xv, (float*)xn, 
							     k, dagger, in->Bytes(), in->NormBytes());
  } else if (in->Precision() == QUDA_HALF_PRECISION) {	
    dslash = new StaggeredDslashCuda<short2, short2, short4>((short2*)out->V(), (float*)out->Norm(), 
							     (short2*)fatGauge0, (short2*)fatGauge1,
							     (short4*)longGauge0, (short4*)longGauge1, 
							     longGauge.Reconstruct(), (short2*)in->V(), 
							     (float*)in->Norm(), (short2*)xv, (float*)xn, 
							     k, dagger,  in->Bytes(), in->NormBytes());
  }

  dslashCuda(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace(), tune);

  delete dslash;
  unbindGaugeTex(fatGauge);
  unbindGaugeTex(longGauge);

  if (!dslashTuning) checkCudaError();
  
#else
  errorQuda("Staggered dslash has not been built");
#endif  // GPU_STAGGERED_DIRAC
}


template <typename spinorFloat, typename cloverFloat>
void cloverCuda(spinorFloat *out, float *outNorm, const cloverFloat *clover,
		const float *cloverNorm, const spinorFloat *in, const float *inNorm, 
		const size_t bytes, const size_t norm_bytes, const TuneParam &tune)
{
  dim3 gridDim( (dslashParam.threads+tune.block.x-1) / tune.block.x, 1, 1);

  int shared_bytes = 
    tune.block.x*(CLOVER_SHARED_FLOATS_PER_THREAD*bindSpinorTex(bytes, norm_bytes, in, inNorm));
  shared_bytes = tune.shared_bytes > shared_bytes ? tune.shared_bytes : shared_bytes;
  if (!(dslash_launch = checkLaunchParam(shared_bytes))) return;

  cloverKernel<<<gridDim, tune.block, shared_bytes>>> 
    (out, outNorm, clover, cloverNorm, in, inNorm, dslashParam);
  unbindSpinorTex(in, inNorm);
}

void cloverCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover clover, 
		const cudaColorSpinorField *in, const int parity, const TuneParam &tune) {

  dslashParam.parity = parity;
  dslashParam.threads = in->Volume();

#ifdef GPU_CLOVER_DIRAC
  void *cloverP, *cloverNormP;
  QudaPrecision clover_prec = bindCloverTex(clover, parity, &cloverP, &cloverNormP);

  if (in->Precision() != clover_prec)
    errorQuda("Mixing clover and spinor precision not supported");

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    cloverCuda((double2*)out->V(), (float*)out->Norm(), (double2*)cloverP, 
	       (float*)cloverNormP, (double2*)in->V(), (float*)in->Norm(), 
	       in->Bytes(), in->NormBytes(), tune);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    cloverCuda((float4*)out->V(), (float*)out->Norm(), (float4*)cloverP, 
	       (float*)cloverNormP, (float4*)in->V(), (float*)in->Norm(),
	       in->Bytes(), in->NormBytes(), tune);
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    cloverCuda((short4*)out->V(), (float*)out->Norm(), (short4*)cloverP, 
	       (float*)cloverNormP, (short4*)in->V(), (float*)in->Norm(), 
	       in->Bytes(), in->NormBytes(), tune);
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
		     const QudaTwistGamma5Type twist, const TuneParam &tune)
{
  dim3 gridDim( (dslashParam.threads+tune.block.x-1) / tune.block.x, 1, 1);

  double a=0.0, b=0.0;
  setTwistParam(a, b, kappa, mu, dagger, twist);

  bindSpinorTex(bytes, norm_bytes, in, inNorm);
  twistGamma5Kernel<<<gridDim, tune.block, tune.shared_bytes>>> (out, outNorm, a, b, in, inNorm, dslashParam);
  unbindSpinorTex(in, inNorm);
}

void twistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		     const int dagger, const double &kappa, const double &mu,
		     const QudaTwistGamma5Type twist, const TuneParam &tune) {

  dslashParam.threads = in->Volume();

#ifdef GPU_TWISTED_MASS_DIRAC
  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    twistGamma5Cuda((double2*)out->V(), (float*)out->Norm(), 
		    (double2*)in->V(), (float*)in->Norm(), 
		    dagger, kappa, mu, in->Bytes(), 
		    in->NormBytes(), twist, tune);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    twistGamma5Cuda((float4*)out->V(), (float*)out->Norm(),
		    (float4*)in->V(), (float*)in->Norm(), 
		    dagger, kappa, mu, in->Bytes(), 
		    in->NormBytes(), twist, tune);
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    twistGamma5Cuda((short4*)out->V(), (float*)out->Norm(),
		    (short4*)in->V(), (float*)in->Norm(), 
		    dagger, kappa, mu, in->Bytes(), 
		    in->NormBytes(), twist, tune);
  }
  if (!dslashTuning) checkCudaError();
#else
  errorQuda("Twisted mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
}


#include "misc_helpers.cu"


#if defined(GPU_FATLINK) || defined(GPU_GAUGE_FORCE) || defined(GPU_FERMION_FORCE) || defined(GPU_HISQ_FORCE) || defined(GPU_UNITARIZE)
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

#ifdef GPU_UNITARIZE
#include "unitarize_links_quda.cu"
#endif

#ifdef GPU_HISQ_FORCE
#include "hisq_paths_force_quda.cu"
#include "unitarize_force_quda.cu"
#endif
