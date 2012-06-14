#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>

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
#include <inline_ptx.h>

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

static FaceBuffer *face;
static cudaColorSpinorField *inSpinor;

// For tuneLaunch() to uniquely identify a suitable set of launch parameters, we need copies of a few of
// the constants set by initDslashConstants().
static struct {
  int x[4];
  int Ls;
  unsigned long long VolumeCB() { return x[0]*x[1]*x[2]*x[3]/2; }
  // In the future, we may also want to add gauge_fixed, sp_stride, ga_stride, cl_stride, etc.
} dslashConstants;

// dslashTuning = QUDA_TUNE_YES enables autotuning when the dslash is
// first launched
static QudaTune dslashTuning = QUDA_TUNE_NO;
static QudaVerbosity verbosity = QUDA_SILENT;

void setDslashTuning(QudaTune tune, QudaVerbosity verbose)
{
  dslashTuning = tune;
  verbosity = verbose;
}

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

// Enable shared memory dslash for Fermi architecture
//#define SHARED_WILSON_DSLASH
//#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

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


void setFace(const FaceBuffer &Face) {
  face = (FaceBuffer*)&Face; // nasty
}


void createDslashEvents()
{
 #ifndef DSLASH_PROFILING
  // add cudaEventDisableTiming for lower sync overhead
  for (int i=0; i<Nstream; i++) {
    cudaEventCreate(&packEnd[i], cudaEventDisableTiming);
    cudaEventCreate(&gatherStart[i], cudaEventDisableTiming);
    cudaEventCreate(&gatherEnd[i], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&scatterStart[i], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&scatterEnd[i], cudaEventDisableTiming);
  }
#else
  cudaEventCreate(&dslashStart);
  cudaEventCreate(&dslashEnd);
  for (int i=0; i<Nstream; i++) {
    cudaEventCreate(&packStart[i]);
    cudaEventCreate(&packEnd[i]);

    cudaEventCreate(&gatherStart[i]);
    cudaEventCreate(&gatherEnd[i]);

    cudaEventCreate(&scatterStart[i]);
    cudaEventCreate(&scatterEnd[i]);

    cudaEventCreate(&kernelStart[i]);
    cudaEventCreate(&kernelEnd[i]);

    kernelTime[i][0] = 0.0;
    kernelTime[i][1] = 0.0;

    gatherTime[i][0] = 0.0;
    gatherTime[i][1] = 0.0;

    commsTime[i][0] = 0.0;
    commsTime[i][1] = 0.0;

    scatterTime[i][0] = 0.0;
    scatterTime[i][1] = 0.0;
  }
#endif

  checkCudaError();
}


void destroyDslashEvents()
{
  for (int i=0; i<Nstream; i++) {
    cudaEventDestroy(packEnd[i]);
    cudaEventDestroy(gatherStart[i]);
    cudaEventDestroy(gatherEnd[i]);
    cudaEventDestroy(scatterStart[i]);
    cudaEventDestroy(scatterEnd[i]);
  }

#ifdef DSLASH_PROFILING
  cudaEventDestroy(dslashStart);
  cudaEventDestroy(dslashEnd);

  for (int i=0; i<Nstream; i++) {
    cudaEventDestroy(packStart[i]);
    cudaEventDestroy(kernelStart[i]);
    cudaEventDestroy(kernelEnd[i]);
  }
#endif

  checkCudaError();
}


#define MORE_GENERIC_DSLASH(FUNC, DAG, X, kernel_type, gridDim, blockDim, shared, stream, param,  ...)            \
  if (x==0) {								                                          \
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				                                          \
      FUNC ## 18 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param);     \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {                                                              \
      FUNC ## 12 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param);     \
    } else {                                                                                                      \
      FUNC ## 8 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param);       \
    }									                                          \
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


#define MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, kernel_type, gridDim, blockDim, shared, stream, param,  ...)     \
  if (reconstruct == QUDA_RECONSTRUCT_NO) {				                                        \
    FUNC ## 18 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
  } else if (reconstruct == QUDA_RECONSTRUCT_12) {			                                        \
    FUNC ## 12 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {			                                        \
    FUNC ## 8 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
  }									

#ifndef MULTI_GPU

#define GENERIC_ASYM_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...)                          \
  switch(param.kernel_type) {						                                      \
  case INTERIOR_KERNEL:							                                      \
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								                                      \
  default:								                                      \
    errorQuda("KernelType %d not defined for single GPU", param.kernel_type);                                 \
  }

#else

#define GENERIC_ASYM_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...)                            \
  switch(param.kernel_type) {						                                        \
  case INTERIOR_KERNEL:							                                        \
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL,   gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								                                        \
  case EXTERIOR_KERNEL_X:							                                \
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_X, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								                                        \
  case EXTERIOR_KERNEL_Y:							                                \
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Y, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								                                        \
  case EXTERIOR_KERNEL_Z:							                                \
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Z, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								                                        \
  case EXTERIOR_KERNEL_T:							                                \
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_T, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								                                        \
  }

#endif

// macro used for dslash types with dagger kernel defined (Wilson, domain wall, etc.)
#define ASYM_DSLASH(FUNC, gridDim, blockDim, shared, stream, param, ...)	\
  if (!dagger) {							\
    GENERIC_ASYM_DSLASH(FUNC, , Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
  } else {								\
    GENERIC_ASYM_DSLASH(FUNC, Dagger, Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
 }


// Use an abstract class interface to drive the different CUDA dslash
// kernels. All parameters are curried into the derived classes to
// allow a simple interface.
class DslashCuda : public Tunable {
 protected:
  int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool advanceGridDim(TuneParam &param) const { return false; } // Don't tune the grid dimensions.
  bool advanceBlockDim(TuneParam &param) const {
    bool advance = Tunable::advanceBlockDim(param);
    if (advance) param.grid = dim3( (dslashParam.threads+param.block.x-1) / param.block.x, 1, 1);
    return advance;
  }

 public:
  DslashCuda() { }
  virtual ~DslashCuda() { }
  virtual TuneKey tuneKey() const;
  std::string paramString(const TuneParam &param) const // Don't bother printing the grid dim.
  {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  virtual int Nface() { return 2; }

  virtual void initTuneParam(TuneParam &param) const
  {
    Tunable::initTuneParam(param);
    param.grid = dim3( (dslashParam.threads+param.block.x-1) / param.block.x, 1, 1);
  }

  /** sets default values for when tuning is disabled */
  virtual void defaultTuneParam(TuneParam &param) const
  {
    Tunable::defaultTuneParam(param);
    param.grid = dim3( (dslashParam.threads+param.block.x-1) / param.block.x, 1, 1);
  }


};

TuneKey DslashCuda::tuneKey() const
{
  std::stringstream vol, aux;
  
  vol << dslashConstants.x[0] << "x";
  vol << dslashConstants.x[1] << "x";
  vol << dslashConstants.x[2] << "x";
  vol << dslashConstants.x[3];

  aux << "type=";
#ifdef MULTI_GPU
  char comm[5], ghost[5];
  switch (dslashParam.kernel_type) {
  case INTERIOR_KERNEL: aux << "interior"; break;
  case EXTERIOR_KERNEL_X: aux << "exterior_x"; break;
  case EXTERIOR_KERNEL_Y: aux << "exterior_y"; break;
  case EXTERIOR_KERNEL_Z: aux << "exterior_z"; break;
  case EXTERIOR_KERNEL_T: aux << "exterior_t"; break;
  }
  for (int i=0; i<4; i++) {
    comm[i] = (dslashParam.commDim[i] ? '1' : '0');
    ghost[i] = (dslashParam.ghostDim[i] ? '1' : '0');
  }
  comm[4] = '\0'; ghost[4] = '\0';
  aux << ",comm=" << comm;
  if (dslashParam.kernel_type == INTERIOR_KERNEL) {
    aux << ",ghost=" << ghost;
  }
#else
  aux << "single-GPU";
#endif // MULTI_GPU
  return TuneKey(vol.str(), typeid(*this).name(), aux.str());
}

/** This derived class is specifically for driving the Dslash kernels
    that use shared memory blocking.  This only applies on Fermi and
    upwards, and only for the interior kernels. */
#if (__COMPUTE_CAPABILITY__ >= 200 && defined(SHARED_WILSON_DSLASH)) 
class SharedDslashCuda : public DslashCuda {
 protected:
  int sharedBytesPerBlock(const TuneParam &param) const { return 0; } // FIXME: this isn't quite true, but works
  bool advanceSharedBytes(TuneParam &param) const { 
    if (dslashParam.kernel_type != INTERIOR_KERNEL) return DslashCuda::advanceSharedBytes(param);
    else return false;
  } // FIXME - shared memory tuning only supported on exterior kernels

  /** Helper function to set the shared memory size from the 3-d block size */
  int sharedBytes(const dim3 &block) const { 
    int warpSize = 32; // FIXME - query from device properties
    int block_xy = block.x*block.y;
    if (block_xy % warpSize != 0) block_xy = ((block_xy / warpSize) + 1)*warpSize;
    return block_xy*block.z*sharedBytesPerThread();
  }

  /** Helper function to set the 3-d grid size from the 3-d block size */
  dim3 createGrid(const dim3 &block) const {
    unsigned int gx = ((dslashConstants.x[0]/2)*dslashConstants.x[3] + block.x - 1) / block.x;
    unsigned int gy = (dslashConstants.x[1] + block.y - 1 ) / block.y;	
    unsigned int gz = (dslashConstants.x[2] + block.z - 1) / block.z;
    return dim3(gx, gy, gz);
  }

  /** Advance the 3-d block size. */
  bool advanceBlockDim(TuneParam &param) const {
    if (dslashParam.kernel_type != INTERIOR_KERNEL) return DslashCuda::advanceBlockDim(param);
    const unsigned int min_threads = 2;
    const unsigned int max_threads = 512; // FIXME: use deviceProp.maxThreadsDim[0];
    const unsigned int max_shared = 16384*3; // FIXME: use deviceProp.sharedMemPerBlock;
    
    // set the x-block dimension equal to the entire x dimension
    bool set = false;
    dim3 blockInit = param.block;
    blockInit.z++;
    for (unsigned bx=blockInit.x; bx<=dslashConstants.x[0]/2; bx++) {
      //unsigned int gx = (dslashConstants.x[0]*dslashConstants.x[3] + bx - 1) / bx;
      for (unsigned by=blockInit.y; by<=dslashConstants.x[1]; by++) {
	unsigned int gy = (dslashConstants.x[1] + by - 1 ) / by;	
	
	if (by > 1 && (by%2) != 0) continue; // can't handle odd blocks yet except by=1
	
	for (unsigned bz=blockInit.z; bz<=dslashConstants.x[2]; bz++) {
	  unsigned int gz = (dslashConstants.x[2] + bz - 1) / bz;
	  
	  if (bz > 1 && (bz%2) != 0) continue; // can't handle odd blocks yet except bz=1
	  if (bx*by*bz > max_threads) continue;
	  if (bx*by*bz < min_threads) continue;
	  // can't yet handle the last block properly in shared memory addressing
	  if (by*gy != dslashConstants.x[1]) continue;
	  if (bz*gz != dslashConstants.x[2]) continue;
	  if (sharedBytes(dim3(bx, by, bz)) > max_shared) continue;

	  param.block = dim3(bx, by, bz);	  
	  set = true; break;
	}
	if (set) break;
	blockInit.z = 1;
      }
      if (set) break;
      blockInit.y = 1;
    }

    if (param.block.x > dslashConstants.x[0]/2 && param.block.y > dslashConstants.x[1] &&
	param.block.z > dslashConstants.x[2] || !set) {
      //||sharedBytesPerThread()*param.block.x > max_shared) {
      param.block = dim3(dslashConstants.x[0]/2, 1, 1);
      return false;
    } else { 
      param.grid = createGrid(param.block);
      param.shared_bytes = sharedBytes(param.block);
      return true; 
    }
    
  }

 public:
  SharedDslashCuda() : DslashCuda() { ; }
  virtual ~SharedDslashCuda() { ; }
  std::string paramString(const TuneParam &param) const // override and print out grid as well
  {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
    ps << "grid=(" << param.grid.x << "," << param.grid.y << "," << param.grid.z << "), ";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }

  virtual void initTuneParam(TuneParam &param) const
  {
    if (dslashParam.kernel_type != INTERIOR_KERNEL) return DslashCuda::initTuneParam(param);

    param.block = dim3(dslashConstants.x[0]/2, 1, 1);
    param.grid = createGrid(param.block);
    param.shared_bytes = sharedBytes(param.block);
  }

  /** Sets default values for when tuning is disabled - this is guaranteed to work, but will be slow */
  virtual void defaultTuneParam(TuneParam &param) const
  {
    if (dslashParam.kernel_type != INTERIOR_KERNEL) DslashCuda::defaultTuneParam(param);
    else initTuneParam(param);
  }
};
#else /** For pre-Fermi architectures */
class SharedDslashCuda : public DslashCuda {
 public:
  SharedDslashCuda() : DslashCuda() { }
  virtual ~SharedDslashCuda() { }
};
#endif


template <typename sFloat, typename gFloat>
class WilsonDslashCuda : public SharedDslashCuda {

 private:
  const size_t bytes, norm_bytes;
  sFloat *out;
  float *outNorm;
  char *saveOut, *saveOutNorm;
  const sFloat *in, *x;
  const float *inNorm, *xNorm;
  const gFloat *gauge0, *gauge1;
  const QudaReconstructType reconstruct;
  const int dagger;
  const double a;

 protected:
  int sharedBytesPerThread() const
  {
#if (__COMPUTE_CAPABILITY__ >= 200) // Fermi uses shared memory for common input
    if (dslashParam.kernel_type == INTERIOR_KERNEL) { // Interior kernels use shared memory for common iunput
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
    } else { // Exterior kernels use no shared memory
      return 0;
    }
#else // Pre-Fermi uses shared memory only for pseudo-registers
    int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
    return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
#endif
  }

 public:
  WilsonDslashCuda(sFloat *out, float *outNorm, const gFloat *gauge0, const gFloat *gauge1, 
		   const QudaReconstructType reconstruct, const sFloat *in, const float *inNorm,
		   const sFloat *x, const float *xNorm, const double a,
		   const int dagger, const size_t bytes, const size_t norm_bytes)
    : SharedDslashCuda(), bytes(bytes), norm_bytes(norm_bytes), out(out), outNorm(outNorm), gauge0(gauge0), gauge1(gauge1), in(in), 
    inNorm(inNorm), reconstruct(reconstruct), dagger(dagger), x(x), xNorm(xNorm), a(a)
  { 
    bindSpinorTex(bytes, norm_bytes, in, inNorm, out, outNorm, x, xNorm); 
  }

  virtual ~WilsonDslashCuda() { unbindSpinorTex(in, inNorm, out, outNorm, x, xNorm); }

  TuneKey tuneKey() const
  {
    TuneKey key = DslashCuda::tuneKey();
    std::stringstream recon;
    recon << reconstruct;
    key.aux += ",reconstruct=" + recon.str();
    if (x) key.aux += ",Xpay";
    return key;
  }

  void apply(const cudaStream_t &stream)
  {
#ifdef SHARED_WILSON_DSLASH
    if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
      errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
    TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
    DSLASH(dslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	   out, outNorm, gauge0, gauge1, in, inNorm, x, xNorm, a);
  }

  void preTune()
  {
    if (dslashParam.kernel_type < 5) { // exterior kernel
      saveOut = new char[bytes];
      cudaMemcpy(saveOut, out, bytes, cudaMemcpyDeviceToHost);
      if (typeid(sFloat) == typeid(short4)) {
	saveOutNorm = new char[norm_bytes];
	cudaMemcpy(saveOutNorm, outNorm, norm_bytes, cudaMemcpyDeviceToHost);
      }
    }
  }

  void postTune()
  {
    if (dslashParam.kernel_type < 5) { // exterior kernel
      cudaMemcpy(out, saveOut, bytes, cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (typeid(sFloat) == typeid(short4)) {
	cudaMemcpy(outNorm, saveOutNorm, norm_bytes, cudaMemcpyHostToDevice);
	delete[] saveOutNorm;
      }
    }
  }

  long long flops() const { return (x ? 1368ll : 1320ll) * dslashConstants.VolumeCB(); } // FIXME for multi-GPU
};

template <typename sFloat, typename gFloat, typename cFloat>
class CloverDslashCuda : public SharedDslashCuda {

 private:
  const size_t bytes, norm_bytes;
  sFloat *out;
  float *outNorm;
  char *saveOut, *saveOutNorm;
  const sFloat *in, *x;
  const float *inNorm, *xNorm;
  const gFloat *gauge0, *gauge1;
  const QudaReconstructType reconstruct;
  const cFloat *clover;
  const float *cloverNorm;
  const int dagger;
  const double a;

 protected:
  int sharedBytesPerThread() const
  {
#if (__COMPUTE_CAPABILITY__ >= 200)
    if (dslashParam.kernel_type == INTERIOR_KERNEL) {
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
    } else {
      return 0;
    }
#else
    int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
    return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
#endif
  }

 public:
  CloverDslashCuda(sFloat *out, float *outNorm, const gFloat *gauge0, const gFloat *gauge1, 
		   const QudaReconstructType reconstruct, const cFloat *clover, 
		   const float *cloverNorm, const sFloat *in, const float *inNorm,
		   const sFloat *x, const float *xNorm, const double a,
		   const int dagger, const size_t bytes, const size_t norm_bytes)
    : SharedDslashCuda(), bytes(bytes), norm_bytes(norm_bytes), out(out), outNorm(outNorm), gauge0(gauge0), gauge1(gauge1), clover(clover),
    cloverNorm(cloverNorm), in(in), inNorm(inNorm), reconstruct(reconstruct), dagger(dagger), x(x), xNorm(xNorm), a(a)
  { 
    bindSpinorTex(bytes, norm_bytes, in, inNorm, out, outNorm, x, xNorm); 
  }
  virtual ~CloverDslashCuda() { unbindSpinorTex(in, inNorm, out, outNorm, x, xNorm); }

  TuneKey tuneKey() const
  {
    TuneKey key = DslashCuda::tuneKey();
    std::stringstream recon;
    recon << reconstruct;
    key.aux += ",reconstruct=" + recon.str();
    if (x) key.aux += ",Xpay";
    return key;
  }

  void apply(const cudaStream_t &stream)
  {
#ifdef SHARED_WILSON_DSLASH
    if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
      errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
    TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
    DSLASH(cloverDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	   out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, x, xNorm, a);
  }

  void preTune()
  {
    if (dslashParam.kernel_type < 5) { // exterior kernel
      saveOut = new char[bytes];
      cudaMemcpy(saveOut, out, bytes, cudaMemcpyDeviceToHost);
      if (typeid(sFloat) == typeid(short4)) {
	saveOutNorm = new char[norm_bytes];
	cudaMemcpy(saveOutNorm, outNorm, norm_bytes, cudaMemcpyDeviceToHost);
      }
    }
  }

  void postTune()
  {
    if (dslashParam.kernel_type < 5) { // exterior kernel
      cudaMemcpy(out, saveOut, bytes, cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (typeid(sFloat) == typeid(short4)) {
	cudaMemcpy(outNorm, saveOutNorm, norm_bytes, cudaMemcpyHostToDevice);
	delete[] saveOutNorm;
      }
    }
  }

  long long flops() const { return (x ? 1872ll : 1824ll) * dslashConstants.VolumeCB(); } // FIXME for multi-GPU
};

template <typename sFloat, typename gFloat, typename cFloat>
class AsymCloverDslashCuda : public SharedDslashCuda {

 private:
  const size_t bytes, norm_bytes;
  sFloat *out;
  float *outNorm;
  char *saveOut, *saveOutNorm;
  const sFloat *in, *x;
  const float *inNorm, *xNorm;
  const gFloat *gauge0, *gauge1;
  const QudaReconstructType reconstruct;
  const cFloat *clover;
  const float *cloverNorm;
  const int dagger;
  const double a;

 protected:
  int sharedBytesPerThread() const
  {
#if (__COMPUTE_CAPABILITY__ >= 200)
    if (dslashParam.kernel_type == INTERIOR_KERNEL) {
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
    } else {
      return 0;
    }
#else
    int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
    return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
#endif
  }

 public:
  AsymCloverDslashCuda(sFloat *out, float *outNorm, const gFloat *gauge0, const gFloat *gauge1, 
		   const QudaReconstructType reconstruct, const cFloat *clover, 
		   const float *cloverNorm, const sFloat *in, const float *inNorm,
		   const sFloat *x, const float *xNorm, const double a,
		   const int dagger, const size_t bytes, const size_t norm_bytes)
    : SharedDslashCuda(), bytes(bytes), norm_bytes(norm_bytes), out(out), outNorm(outNorm), gauge0(gauge0), gauge1(gauge1), clover(clover),
    cloverNorm(cloverNorm), in(in), inNorm(inNorm), reconstruct(reconstruct), dagger(dagger), x(x), xNorm(xNorm), a(a)
  { 
    bindSpinorTex(bytes, norm_bytes, in, inNorm, out, outNorm, x, xNorm);
    if (!x) errorQuda("Asymmetric clover dslash only defined for Xpay");
  }
  virtual ~AsymCloverDslashCuda() { unbindSpinorTex(in, inNorm, out, outNorm, x, xNorm); }

  TuneKey tuneKey() const
  {
    TuneKey key = DslashCuda::tuneKey();
    std::stringstream recon;
    recon << reconstruct;
    key.aux += ",reconstruct=" + recon.str() + ",Xpay";
    return key;
  }

  void apply(const cudaStream_t &stream)
  {
#ifdef SHARED_WILSON_DSLASH
    if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
      errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
    TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
    ASYM_DSLASH(asymCloverDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
		out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, x, xNorm, a);
  }

  void preTune()
  {
    if (dslashParam.kernel_type < 5) { // exterior kernel
      saveOut = new char[bytes];
      cudaMemcpy(saveOut, out, bytes, cudaMemcpyDeviceToHost);
      if (typeid(sFloat) == typeid(short4)) {
	saveOutNorm = new char[norm_bytes];
	cudaMemcpy(saveOutNorm, outNorm, norm_bytes, cudaMemcpyDeviceToHost);
      }
    }
  }

  void postTune()
  {
    if (dslashParam.kernel_type < 5) { // exterior kernel
      cudaMemcpy(out, saveOut, bytes, cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (typeid(sFloat) == typeid(short4)) {
	cudaMemcpy(outNorm, saveOutNorm, norm_bytes, cudaMemcpyHostToDevice);
	delete[] saveOutNorm;
      }
    }
  }

  long long flops() const { return 1872ll * dslashConstants.VolumeCB(); } // FIXME for multi-GPU
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
class TwistedDslashCuda : public SharedDslashCuda {

 private:
  const size_t bytes, norm_bytes;
  sFloat *out;
  float *outNorm;
  char *saveOut, *saveOutNorm;
  const sFloat *in, *x;
  const float *inNorm, *xNorm;
  const gFloat *gauge0, *gauge1;
  const QudaReconstructType reconstruct;
  const int dagger;
  double a;
  double b;

 protected:
  int sharedBytesPerThread() const
  {
#if (__COMPUTE_CAPABILITY__ >= 200)
    if (dslashParam.kernel_type == INTERIOR_KERNEL) {
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
    } else {
      return 0;
    }
#else
    int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
    return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
#endif
  }

 public:
  TwistedDslashCuda(sFloat *out, float *outNorm, const gFloat *gauge0, const gFloat *gauge1, 
		    const QudaReconstructType reconstruct, const sFloat *in, const float *inNorm,
		    const sFloat *x, const float *xNorm, const double kappa, const double mu,
		    const double k, const int dagger, const size_t bytes, const size_t norm_bytes)
    : SharedDslashCuda(), bytes(bytes), norm_bytes(norm_bytes), out(out), outNorm(outNorm), gauge0(gauge0), gauge1(gauge1), in(in),
    inNorm(inNorm), reconstruct(reconstruct), dagger(dagger), x(x), xNorm(xNorm)
  { 
    bindSpinorTex(bytes, norm_bytes, in, inNorm, out, outNorm, x, xNorm); 
    setTwistParam(a, b, kappa, mu, dagger, QUDA_TWIST_GAMMA5_INVERSE);
    if (x) b *= k;
  }
  virtual ~TwistedDslashCuda() { unbindSpinorTex(in, inNorm, out, outNorm, x, xNorm); }

  TuneKey tuneKey() const
  {
    TuneKey key = DslashCuda::tuneKey();
    std::stringstream recon;
    recon << reconstruct;
    key.aux += ",reconstruct=" + recon.str();
    if (x) key.aux += ",Xpay";
    return key;
  }

  void apply(const cudaStream_t &stream)
  {
#ifdef SHARED_WILSON_DSLASH
    if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
      errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
    TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
    DSLASH(twistedMassDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	   out, outNorm, gauge0, gauge1, in, inNorm, a, b, x, xNorm);
  }

  void preTune()
  {
    if (dslashParam.kernel_type < 5) { // exterior kernel
      saveOut = new char[bytes];
      cudaMemcpy(saveOut, out, bytes, cudaMemcpyDeviceToHost);
      if (typeid(sFloat) == typeid(short4)) {
	saveOutNorm = new char[norm_bytes];
	cudaMemcpy(saveOutNorm, outNorm, norm_bytes, cudaMemcpyDeviceToHost);
      }
    }
  }

  void postTune()
  {
    if (dslashParam.kernel_type < 5) { // exterior kernel
      cudaMemcpy(out, saveOut, bytes, cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (typeid(sFloat) == typeid(short4)) {
	cudaMemcpy(outNorm, saveOutNorm, norm_bytes, cudaMemcpyHostToDevice);
	delete[] saveOutNorm;
      }
    }
  }

  long long flops() const { return (x ? 1416ll : 1392ll) * dslashConstants.VolumeCB(); } // FIXME for multi-GPU
};

template <typename sFloat, typename gFloat>
class DomainWallDslashCuda : public DslashCuda {

 private:
  const size_t bytes, norm_bytes;
  sFloat *out;
  float *outNorm;
  char *saveOut, *saveOutNorm;
  const sFloat *in, *x;
  const float *inNorm, *xNorm;
  const gFloat *gauge0, *gauge1;
  const QudaReconstructType reconstruct;
  const int dagger;
  const double mferm;
  const double a;

 protected:
  int sharedBytesPerThread() const { return 0; }
  
 public:
  DomainWallDslashCuda(sFloat *out, float *outNorm, const gFloat *gauge0, const gFloat *gauge1, 
		       const QudaReconstructType reconstruct, const sFloat *in, 
		       const float *inNorm, const sFloat *x, const float *xNorm, const double mferm, 
		       const double a, const int dagger, const size_t bytes, const size_t norm_bytes)
    : DslashCuda(), bytes(bytes), norm_bytes(norm_bytes), out(out), outNorm(outNorm), gauge0(gauge0), gauge1(gauge1), 
    in(in), inNorm(inNorm), mferm(mferm), reconstruct(reconstruct), dagger(dagger), x(x), xNorm(xNorm), a(a)
  { 
    bindSpinorTex(bytes, norm_bytes, in, inNorm, out, outNorm, x, xNorm); 
  }
  virtual ~DomainWallDslashCuda() { unbindSpinorTex(in, inNorm, out, outNorm, x, xNorm); }

  TuneKey tuneKey() const
  {
    TuneKey key = DslashCuda::tuneKey();
    std::stringstream ls, recon;
    ls << dslashConstants.Ls;
    recon << reconstruct;
    key.volume += "x" + ls.str();
    key.aux += ",reconstruct=" + recon.str();
    if (x) key.aux += ",Xpay";
    return key;
  }

  void apply(const cudaStream_t &stream)
  {
    TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
    dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
    DSLASH(domainWallDslash, gridDim, tp.block, tp.shared_bytes, stream, dslashParam,
    	   out, outNorm, gauge0, gauge1, in, inNorm, mferm, x, xNorm, a);
  }

  void preTune()
  {
    if (dslashParam.kernel_type < 5) { // exterior kernel
      saveOut = new char[bytes];
      cudaMemcpy(saveOut, out, bytes, cudaMemcpyDeviceToHost);
      if (typeid(sFloat) == typeid(short4)) {
	saveOutNorm = new char[norm_bytes];
	cudaMemcpy(saveOutNorm, outNorm, norm_bytes, cudaMemcpyDeviceToHost);
      }
    }
  }

  void postTune()
  {
    if (dslashParam.kernel_type < 5) { // exterior kernel
      cudaMemcpy(out, saveOut, bytes, cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (typeid(sFloat) == typeid(short4)) {
	cudaMemcpy(outNorm, saveOutNorm, norm_bytes, cudaMemcpyHostToDevice);
	delete[] saveOutNorm;
      }
    }
  }

  long long flops() const { // FIXME for multi-GPU
    long long bulk = (dslashConstants.Ls-2)*(dslashConstants.VolumeCB()/dslashConstants.Ls);
    long long wall = 2*dslashConstants.VolumeCB()/dslashConstants.Ls;
    return (x ? 1368ll : 1320ll)*dslashConstants.VolumeCB() + 96ll*bulk + 120ll*wall;
  }
};

template <typename sFloat, typename fatGFloat, typename longGFloat>
class StaggeredDslashCuda : public DslashCuda {

private:
  const size_t bytes, norm_bytes;
  sFloat *out;
  float *outNorm;
  char *saveOut, *saveOutNorm;
  const sFloat *in, *x;
  const float *inNorm, *xNorm;
  const fatGFloat *fat0, *fat1;
  const longGFloat *long0, *long1;
  const QudaReconstructType reconstruct;
  const int dagger;
  const double a;

 protected:
  int sharedBytesPerThread() const
  {
    int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
    return 6 * reg_size;
  }

 public:
  StaggeredDslashCuda(sFloat *out, float *outNorm, const fatGFloat *fat0, const fatGFloat *fat1,
		      const longGFloat *long0, const longGFloat *long1,
		      const QudaReconstructType reconstruct, const sFloat *in, 
		      const float *inNorm, const sFloat *x, const float *xNorm, const double a,
		      const int dagger, const size_t bytes, const size_t norm_bytes)
    : DslashCuda(), bytes(bytes), norm_bytes(norm_bytes), out(out), outNorm(outNorm), fat0(fat0), fat1(fat1), long0(long0), long1(long1),
    in(in), inNorm(inNorm), reconstruct(reconstruct), dagger(dagger), x(x), xNorm(xNorm), a(a)
  { 
    bindSpinorTex(bytes, norm_bytes, in, inNorm, out, outNorm, x, xNorm); 
  }

  virtual ~StaggeredDslashCuda() { unbindSpinorTex(in, inNorm, out, outNorm, x, xNorm); }

  TuneKey tuneKey() const
  {
    TuneKey key = DslashCuda::tuneKey();
    std::stringstream recon;
    recon << reconstruct;
    key.aux += ",reconstruct=" + recon.str();
    if (x) key.aux += ",Axpy";
    return key;
  }

  void apply(const cudaStream_t &stream)
  {
    TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
    dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
    STAGGERED_DSLASH(gridDim, tp.block, tp.shared_bytes, stream, dslashParam,
		     out, outNorm, fat0, fat1, long0, long1, in, inNorm, x, xNorm, a);
  }

  void preTune()
  {
    if (dslashParam.kernel_type < 5) { // exterior kernel
      saveOut = new char[bytes];
      cudaMemcpy(saveOut, out, bytes, cudaMemcpyDeviceToHost);
      if (typeid(sFloat) == typeid(short2)) {
	saveOutNorm = new char[norm_bytes];
	cudaMemcpy(saveOutNorm, outNorm, norm_bytes, cudaMemcpyDeviceToHost);
      }
    }
  }

  void postTune()
  {
    if (dslashParam.kernel_type < 5) { // exterior kernel
      cudaMemcpy(out, saveOut, bytes, cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (typeid(sFloat) == typeid(short2)) {
	cudaMemcpy(outNorm, saveOutNorm, norm_bytes, cudaMemcpyHostToDevice);
	delete[] saveOutNorm;
      }
    }
  }

  int Nface() { return 6; }

  long long flops() const { return (x ? 1158ll : 1146ll) * dslashConstants.VolumeCB(); } // FIXME for multi-GPU
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

int gatherCompleted[Nstream];
int previousDir[Nstream];
int commsCompleted[Nstream];
int commDimTotal;

/**
 * Initialize the arrays used for the dynamic scheduling.
 */
void initDslashCommsPattern() {
  for (int i=0; i<Nstream-1; i++) {
    gatherCompleted[i] = 0;
    commsCompleted[i] = 0;
  }
  gatherCompleted[Nstream-1] = 1;
  commsCompleted[Nstream-1] = 1;

  //   We need to know which was the previous direction in which
  //   communication was issued, since we only query a given event /
  //   comms call after the previous the one has successfully
  //   completed.
  for (int i=3; i>=0; i--) {
    if (dslashParam.commDim[i]) {
      int prev = Nstream-1;
      for (int j=3; j>i; j--) if (dslashParam.commDim[j]) prev = 2*j;
      previousDir[2*i + 1] = prev;
      previousDir[2*i + 0] = 2*i + 1; // always valid
    }
  }

  // this tells us how many events / comms occurances there are in
  // total.  Used for exiting the while loop
  commDimTotal = 0;
  for (int i=3; i>=0; i--) commDimTotal += dslashParam.commDim[i];
  commDimTotal *= 4; // 2 from pipe length, 2 from direction
}

void dslashCuda(DslashCuda &dslash, const size_t regSize, const int parity, const int dagger, 
		const int volume, const int *faceVolumeCB) {

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

  CUDA_EVENT_RECORD(kernelStart[Nstream-1], streams[Nstream-1]);
  dslash.apply(streams[Nstream-1]);
  CUDA_EVENT_RECORD(kernelEnd[Nstream-1], streams[Nstream-1]);

#ifdef MULTI_GPU
  initDslashCommsPattern();

  int completeSum = 0;
  while (completeSum < commDimTotal) {
    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;
      
      for (int dir=1; dir>=0; dir--) {
	
	// Query if gather has completed
	if (!gatherCompleted[2*i+dir] && gatherCompleted[previousDir[2*i+dir]]) { 
	  if (cudaSuccess == cudaEventQuery(gatherEnd[2*i+dir])) {
	    gatherCompleted[2*i+dir] = 1;
	    completeSum++;
	    gettimeofday(&commsStart[2*i+dir], NULL);
	    face->commsStart(2*i+dir);
	  }
	}
	
	// Query if comms has finished
	if (!commsCompleted[2*i+dir] && commsCompleted[previousDir[2*i+dir]] &&
	    gatherCompleted[2*i+dir]) {
	  if (face->commsQuery(2*i+dir)) { 
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

    CUDA_EVENT_RECORD(kernelStart[2*i], streams[Nstream-1]);
    dslash.apply(streams[Nstream-1]); // all faces use this stream
    CUDA_EVENT_RECORD(kernelEnd[2*i], streams[Nstream-1]);
  }

  CUDA_EVENT_RECORD(dslashEnd, 0);
  DSLASH_TIME_PROFILE();

#endif // MULTI_GPU
}

// Wilson wrappers
void wilsonDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in, const int parity,
		      const int dagger, const cudaColorSpinorField *x, const double &k, const int *commOverride)
{
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
  dslashCuda(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace());

  delete dslash;
  unbindGaugeTex(gauge);

  checkCudaError();
#else
  errorQuda("Wilson dslash has not been built");
#endif // GPU_WILSON_DIRAC

}

void cloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover cloverInv,
		      const cudaColorSpinorField *in, const int parity, const int dagger, 
		      const cudaColorSpinorField *x, const double &a, const int *commOverride)
{
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

  dslashCuda(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace());

  delete dslash;
  unbindGaugeTex(gauge);
  unbindCloverTex(cloverInv);

  checkCudaError();
#else
  errorQuda("Clover dslash has not been built");
#endif

}


void asymCloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover cloverInv,
		      const cudaColorSpinorField *in, const int parity, const int dagger, 
		      const cudaColorSpinorField *x, const double &a, const int *commOverride)
{
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
    dslash = new AsymCloverDslashCuda<double2, double2, double2>((double2*)out->V(), (float*)out->Norm(), (double2*)gauge0, 
							     (double2*)gauge1, gauge.Reconstruct(), (double2*)cloverP, 
							     (float*)cloverNormP, (double2*)in->V(), (float*)in->Norm(),
							     (double2*)xv, (float*)xn, a, dagger, in->Bytes(), in->NormBytes());
    regSize = sizeof(double);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    dslash = new AsymCloverDslashCuda<float4, float4, float4>((float4*)out->V(), (float*)out->Norm(), (float4*)gauge0, 
							      (float4*)gauge1, gauge.Reconstruct(), (float4*)cloverP, 
							      (float*)cloverNormP, (float4*)in->V(), (float*)in->Norm(), 
							      (float4*)xv, (float*)xn, a, dagger, in->Bytes(), in->NormBytes());
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    dslash = new AsymCloverDslashCuda<short4, short4, short4>((short4*)out->V(), (float*)out->Norm(), (short4*)gauge0, 
							      (short4*)gauge1, gauge.Reconstruct(), (short4*)cloverP, 
							      (float*)cloverNormP, (short4*)in->V(), (float*)in->Norm(), 
							      (short4*)xv, (float*)xn, a, dagger, in->Bytes(), in->NormBytes());
  }

  dslashCuda(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace());

  delete dslash;
  unbindGaugeTex(gauge);
  unbindCloverTex(cloverInv);

  checkCudaError();
#else
  errorQuda("Clover dslash has not been built");
#endif

}


void twistedMassDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			   const cudaColorSpinorField *in, const int parity, const int dagger, 
			   const cudaColorSpinorField *x, const double &kappa, const double &mu, 
			   const double &a, const int *commOverride)
{
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

  dslashCuda(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace());

  delete dslash;
  unbindGaugeTex(gauge);

  checkCudaError();
#else
  errorQuda("Twisted mass dslash has not been built");
#endif

}

void domainWallDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			  const cudaColorSpinorField *in, const int parity, const int dagger, 
			  const cudaColorSpinorField *x, const double &m_f, const double &k2, const int *commOverride)
{
  inSpinor = (cudaColorSpinorField*)in; // EVIL

  dslashParam.parity = parity;
  dslashParam.threads = in->Volume();

#ifdef GPU_DOMAIN_WALL_DIRAC
//BEGIN NEW
  kernelPackT = true; 
  //currently splitting in space-time is impelemented:
  int dirs = 4;
  int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
  for(int i = 0;i < dirs; i++){
    dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
    dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
    dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
    dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
  }  
//END NEW

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

  dslashCuda(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace());

  delete dslash;
  unbindGaugeTex(gauge);

  checkCudaError();
#else
  errorQuda("Domain wall dslash has not been built");
#endif

}

void staggeredDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &fatGauge, 
			 const cudaGaugeField &longGauge, const cudaColorSpinorField *in,
			 const int parity, const int dagger, const cudaColorSpinorField *x,
			 const double &k, const int *commOverride)
{
  inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_STAGGERED_DIRAC

#ifdef MULTI_GPU
  for(int i=0;i < 4; i++){
    if(commDimPartitioned(i) && (fatGauge.X()[i] < 6)){
      errorQuda("ERROR: partitioned dimension with local size less than 6 is not supported in staggered dslash\n");
    }    
  }
#endif

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

  dslashCuda(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace());

  delete dslash;
  unbindGaugeTex(fatGauge);
  unbindGaugeTex(longGauge);

  checkCudaError();
  
#else
  errorQuda("Staggered dslash has not been built");
#endif  // GPU_STAGGERED_DIRAC
}


template <typename sFloat, typename cFloat>
class CloverCuda : public Tunable {
 private:
  const size_t bytes, norm_bytes;
  sFloat *out;
  float *outNorm;
  char *saveOut, *saveOutNorm;
  const cFloat *clover;
  const float *cloverNorm;
  const sFloat *in;
  const float *inNorm;

 protected:
  int sharedBytesPerThread() const
  {
    int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
    return CLOVER_SHARED_FLOATS_PER_THREAD * reg_size;
  }
  int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool advanceGridDim(TuneParam &param) const { return false; } // Don't tune the grid dimensions.

 public:
  CloverCuda(sFloat *out, float *outNorm, const cFloat *clover, const float *cloverNorm, const sFloat *in,
	     const float *inNorm, const size_t bytes, const size_t norm_bytes)
    : out(out), outNorm(outNorm), clover(clover), cloverNorm(cloverNorm), in(in), inNorm(inNorm),
      bytes(bytes), norm_bytes(norm_bytes)
  {
    bindSpinorTex(bytes, norm_bytes, in, inNorm);
  }
  virtual ~CloverCuda() { unbindSpinorTex(in, inNorm); }
  void apply(const cudaStream_t &stream)
  {
    TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
    dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
    cloverKernel<<<gridDim, tp.block, tp.shared_bytes, stream>>>(out, outNorm, clover, cloverNorm, in, inNorm, dslashParam);
  }
  virtual TuneKey tuneKey() const
  {
    std::stringstream vol, aux;
    vol << dslashConstants.x[0] << "x";
    vol << dslashConstants.x[1] << "x";
    vol << dslashConstants.x[2] << "x";
    vol << dslashConstants.x[3];
    return TuneKey(vol.str(), typeid(*this).name());
  }

  // Need to save the out field if it aliases the in field
  void preTune() {
    if (in == out) {
      saveOut = new char[bytes];
      cudaMemcpy(saveOut, out, bytes, cudaMemcpyDeviceToHost);
      if (typeid(sFloat) == typeid(short4)) {
	saveOutNorm = new char[norm_bytes];
	cudaMemcpy(saveOutNorm, outNorm, norm_bytes, cudaMemcpyDeviceToHost);
      }
    }
  }

  // Restore if the in and out fields alias
  void postTune() {
    if (in == out) {
      cudaMemcpy(out, saveOut, bytes, cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (typeid(sFloat) == typeid(short4)) {
	cudaMemcpy(outNorm, saveOutNorm, norm_bytes, cudaMemcpyHostToDevice);
	delete[] saveOutNorm;
      }
    }
  }

  std::string paramString(const TuneParam &param) const // Don't bother printing the grid dim.
  {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }

  long long flops() const { return 504ll * dslashConstants.VolumeCB(); }
};


void cloverCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover clover, 
		const cudaColorSpinorField *in, const int parity) {

  dslashParam.parity = parity;
  dslashParam.threads = in->Volume();

#ifdef GPU_CLOVER_DIRAC
  Tunable *clov = 0;
  void *cloverP, *cloverNormP;
  QudaPrecision clover_prec = bindCloverTex(clover, parity, &cloverP, &cloverNormP);

  if (in->Precision() != clover_prec)
    errorQuda("Mixing clover and spinor precision not supported");

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    clov = new CloverCuda<double2, double2>((double2*)out->V(), (float*)out->Norm(), (double2*)cloverP, 
					    (float*)cloverNormP, (double2*)in->V(), (float*)in->Norm(), 
					    in->Bytes(), in->NormBytes());
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    clov = new CloverCuda<float4, float4>((float4*)out->V(), (float*)out->Norm(), (float4*)cloverP, 
			  (float*)cloverNormP, (float4*)in->V(), (float*)in->Norm(),
			  in->Bytes(), in->NormBytes());
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    clov = new CloverCuda<short4, short4>((short4*)out->V(), (float*)out->Norm(), (short4*)cloverP, 
			  (float*)cloverNormP, (short4*)in->V(), (float*)in->Norm(), 
			  in->Bytes(), in->NormBytes());
  }
  clov->apply(0);

  unbindCloverTex(clover);
  checkCudaError();

  delete clov;
#else
  errorQuda("Clover dslash has not been built");
#endif
}

template <typename sFloat>
class TwistGamma5Cuda : public Tunable {

private:
  sFloat *out;
  float *outNorm;
  sFloat *in;
  float *inNorm;
  double a;
  double b;
  size_t bytes;
  size_t norm_bytes;

  int sharedBytesPerThread() const { return 0; }
  int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool advanceGridDim(TuneParam &param) const { return false; } // Don't tune the grid dimensions.

  char *saveOut, *saveOutNorm;

public:
  TwistGamma5Cuda(sFloat *out, float *outNorm, sFloat *in, float *inNorm,
		  double kappa, double mu, const int dagger, 
		  QudaTwistGamma5Type twist, size_t bytes, size_t norm_bytes) :
    out(out), outNorm(outNorm), in(in), inNorm(inNorm), 
    bytes(bytes), norm_bytes(norm_bytes){
    bindSpinorTex(bytes, norm_bytes, in, inNorm);
    setTwistParam(a, b, kappa, mu, dagger, twist);
  }
  virtual ~TwistGamma5Cuda() {
    unbindSpinorTex(in, inNorm);    
  }

  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << dslashConstants.x[0] << "x";
    vol << dslashConstants.x[1] << "x";
    vol << dslashConstants.x[2] << "x";
    vol << dslashConstants.x[3];    
    return TuneKey(vol.str(), typeid(*this).name(), aux.str());
  }  

  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
    dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
    twistGamma5Kernel<<<gridDim, tp.block, tp.shared_bytes, stream>>> 
      (out, outNorm, a, b, in, inNorm, dslashParam);
  }

  void preTune() {
    saveOut = new char[bytes];
    cudaMemcpy(saveOut, out, bytes, cudaMemcpyDeviceToHost);
    if (typeid(sFloat) == typeid(short4)) {
      saveOutNorm = new char[norm_bytes];
      cudaMemcpy(saveOutNorm, outNorm, norm_bytes, cudaMemcpyDeviceToHost);
    }
  }

  void postTune() {
    cudaMemcpy(out, saveOut, bytes, cudaMemcpyHostToDevice);
    delete[] saveOut;
    if (typeid(sFloat) == typeid(short4)) {
      cudaMemcpy(outNorm, saveOutNorm, norm_bytes, cudaMemcpyHostToDevice);
      delete[] saveOutNorm;
    }
  }

  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }

  long long flops() const { return 24ll * dslashConstants.VolumeCB(); }
};

void twistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		     const int dagger, const double &kappa, const double &mu,
		     const QudaTwistGamma5Type twist)
{
  dslashParam.threads = in->Volume();

#ifdef GPU_TWISTED_MASS_DIRAC
  Tunable *twistGamma5 = 0;

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    twistGamma5 = new TwistGamma5Cuda<double2>
      ((double2*)out->V(), (float*)out->Norm(), (double2*)in->V(), 
       (float*)in->Norm(), kappa, mu, dagger, twist, in->Bytes(), in->NormBytes());
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    twistGamma5 = new TwistGamma5Cuda<float4>
      ((float4*)out->V(), (float*)out->Norm(), (float4*)in->V(), 
       (float*)in->Norm(), kappa, mu, dagger, twist, in->Bytes(), in->NormBytes());
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    twistGamma5 = new TwistGamma5Cuda<short4>
      ((short4*)out->V(), (float*)out->Norm(), (short4*)in->V(), 
       (float*)in->Norm(), kappa, mu, dagger, twist, in->Bytes(), in->NormBytes());
  }

  twistGamma5->apply(streams[Nstream-1]);
  checkCudaError();

  delete twistGamma5;
#else
  errorQuda("Twisted mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
}


#include "misc_helpers.cu"


#if defined(GPU_FATLINK) || defined(GPU_GAUGE_FORCE) || defined(GPU_FERMION_FORCE) || defined(GPU_HISQ_FORCE) || defined(GPU_UNITARIZE)
#include <force_common.h>
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

