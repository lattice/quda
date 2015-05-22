#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>

#include <color_spinor_field.h>
#include <clover_field.h>	// Do we need this now?

// these control the Wilson-type actions
#ifdef GPU_WILSON_DIRAC
//#define DIRECT_ACCESS_WILSON_PACK_SPINOR
#endif // GPU_WILSON_DIRAC


#include <quda_internal.h>
#include <dslash_quda.h>
#include <sys/time.h>
#include <blas_quda.h>

#include <inline_ptx.h>

namespace quda {

  namespace pack {

#include <dslash_constants.h>
#include <dslash_textures.h>

  } // end namespace pack

  using namespace pack;

#ifdef MULTI_GPU
  static int commDim[QUDA_MAX_DIM]; // Whether to do comms or not
  void setPackComms(const int *comm_dim) {
    for (int i=0; i<QUDA_MAX_DIM; i++) commDim[i] = comm_dim[i];
  }
#else
  void setPackComms(const int *comm_dim) { ; }
#endif

#include <dslash_index.cuh>

  // routines for packing the ghost zones (multi-GPU only)


#ifdef MULTI_GPU

  template <typename FloatN>
  struct PackParam {

    FloatN *out[2*4];
    float *outNorm[2*4];
    
    FloatN *in;
    float *inNorm;
    
    int threads; // total number of threads
    
    // offsets which determine thread mapping to dimension
    int threadDimMapLower[4]; // lowest thread which maps to dim
    int threadDimMapUpper[4]; // greatest thread + 1 which maps to dim
    
    int parity;
#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t inTex;
    cudaTextureObject_t inTexNorm;
#endif

    int dim;
    int face_num;
    int X[QUDA_MAX_DIM]; // lattice dimensions
    int ghostFace[4];

    int sp_stride;
  };

  template<typename FloatN>
  std::ostream& operator<<(std::ostream& output, const PackParam<FloatN>& param) {
    output << "threads = " << param.threads << std::endl;
    output << "threadDimMapLower = {" << param.threadDimMapLower[0] << "," <<
      param.threadDimMapLower[1] << "," << param.threadDimMapLower[2] << "," << param.threadDimMapLower[3] << "}" << std::endl;
    output << "threadDimMapUpper = {" << param.threadDimMapUpper[0] << "," <<
      param.threadDimMapUpper[1] << "," << param.threadDimMapUpper[2] << "," << param.threadDimMapUpper[3] << "}" << std::endl;
    output << "parity = " << param.parity << std::endl;
    output << "dim = " << param.dim << std::endl;
    output << "face_num = " << param.face_num << std::endl;
    output << "X = {" << param.X[0] << ","<< param.X[1] << "," << param.X[2] << "," << param.X[3] << "}" << std::endl;
    output << "ghostFace = {" << param.ghostFace[0] << ","<< param.ghostFace[1] << "," 
	   << param.ghostFace[2] << "," << param.ghostFace[3] << "}" << std::endl;
    output << "sp_stride = " << param.sp_stride << std::endl;
    return output;
  }

  // Extend the PackParam class to PackExtendedParam
  template<typename Float>
    struct PackExtendedParam : public PackParam<Float>
    {
      PackExtendedParam(){}
    PackExtendedParam(const PackParam<Float>& base) : PackParam<Float>(base) {}
      int R[QUDA_MAX_DIM]; // boundary dimensions
    };

  /**
   * Determines which face a given thread is computing.  Also rescale
   * face_idx so that is relative to a given dimension.
   */
/*
  template <typename Param>
    __device__ inline int dimFromFaceIndex (int &face_idx, const Param &param) {
    if (face_idx < param.threadDimMapUpper[0]) {
      return 0;
    } else if (face_idx < param.threadDimMapUpper[1]) {
      face_idx -= param.threadDimMapLower[1];
      return 1;
    } else if (face_idx < param.threadDimMapUpper[2]) {
      face_idx -= param.threadDimMapLower[2];
      return 2;
    } else { // this is only called if we use T kernel packing 
      face_idx -= param.threadDimMapLower[3];
      return 3;
    }
  }
*/
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)

  // double precision
#if (defined DIRECT_ACCESS_WILSON_PACK_SPINOR) || (defined FERMI_NO_DBLE_TEX)
#define READ_SPINOR READ_SPINOR_DOUBLE
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_DOUBLE_TEX
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexDouble
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_DOUBLE2
#define SPINOR_DOUBLE
  template <int dim, int dagger, int face_num>
    static inline __device__ void packFaceWilsonCore(double2 *out, float *outNorm, const double2 *in, 
						     const float *inNorm, const int &idx, 
						     const int &face_idx, const int &face_volume, 
						     PackParam<double2> &param)
  {
#if (__COMPUTE_CAPABILITY__ >= 130)
    if (dagger) {
#include "wilson_pack_face_dagger_core.h"
    } else {
#include "wilson_pack_face_core.h"
    }
#endif // (__COMPUTE_CAPABILITY__ >= 130)
  }

  template <int dim, int dagger, int face_num>
    static inline __device__ void unpackFaceWilsonCore(double2 *out, float *outNorm, const double2 *in, 
						       const float *inNorm, const int &idx, 
						       const int &face_idx, const int &face_volume, 
						       PackParam<double2> &param)
  {
#if (__COMPUTE_CAPABILITY__ >= 130)
    if (dagger) {
#include "wilson_pack_face_dagger_core.h"
    } else {
#include "wilson_pack_face_core.h"
    }
#endif // (__COMPUTE_CAPABILITY__ >= 130)
  }

#undef READ_SPINOR
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR
#undef SPINOR_DOUBLE


  // single precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_SINGLE_TEX
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexSingle
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_FLOAT4
  template <int dim, int dagger, int face_num>
    static inline __device__ void packFaceWilsonCore(float4 *out, float *outNorm, const float4 *in, const float *inNorm,
						     const int &idx, const int &face_idx, 
						     const int &face_volume, 
						     const PackParam<float4> &param)
  {
    if (dagger) {
#include "wilson_pack_face_dagger_core.h"
    } else {
#include "wilson_pack_face_core.h"
    }
  }

  template <int dim, int dagger, int face_num>
    static inline __device__ void unpackFaceWilsonCore(float4 *out, float *outNorm, const float4 *in, const float *inNorm,
						       const int &idx, const int &face_idx, 
						       const int &face_volume, 
						       const PackParam<float4> &param)
  {
    if (dagger) {
#include "wilson_pack_face_dagger_core.h"
    } else {
#include "wilson_pack_face_core.h"
    }
  }
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR


  // half precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_HALF_TEX
#define READ_SPINOR_UP READ_SPINOR_HALF_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexHalf
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_SHORT4
  template <int dim, int dagger, int face_num>
    static inline __device__ void packFaceWilsonCore(short4 *out, float *outNorm, const short4 *in, const float *inNorm,
						     const int &idx, const int &face_idx, 
						     const int &face_volume, 
						     const PackParam<short4> &param)
  {
    if (dagger) {
#include "wilson_pack_face_dagger_core.h"
    } else {
#include "wilson_pack_face_core.h"
    }
  }

  template <int dim, int dagger, int face_num>
    static inline __device__ void unpackFaceWilsonCore(short4 *out, float *outNorm, const short4 *in, const float *inNorm,
						       const int &idx, const int &face_idx, 
						       const int &face_volume, 
						       const PackParam<short4> &param)
  {
    if (dagger) {
#include "wilson_pack_face_dagger_core.h"
    } else {
#include "wilson_pack_face_core.h"
    }
  }
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR

  template <int dagger, typename FloatN>
    __global__ void packFaceWilsonKernel(PackParam<FloatN> param)
  {
    const int nFace = 1; // 1 face for Wilson

    int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (face_idx >= param.threads) return;

    // determine which dimension we are packing
    const int dim = dimFromFaceIndex(face_idx, param);

    // compute where the output is located
    // compute an index into the local volume from the index into the face
    // read spinor, spin-project, and write half spinor to face
    if (dim == 0) {
      // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
      const int face_num = (face_idx >= nFace*param.ghostFace[0]) ? 1 : 0;
      face_idx -= face_num*nFace*param.ghostFace[0];
      if (face_num == 0) {
	const int idx = indexFromFaceIndex<0,nFace,0>(face_idx,param.ghostFace[0],param.parity,param.X);
	packFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
				       param.inNorm,idx, face_idx, param.ghostFace[0], param);
      } else {
	const int idx = indexFromFaceIndex<0,nFace,1>(face_idx,param.ghostFace[0],param.parity,param.X);
	packFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
				       param.inNorm,idx, face_idx, param.ghostFace[0], param);
      }
    } else if (dim == 1) {
      // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
      const int face_num = (face_idx >= nFace*param.ghostFace[1]) ? 1 : 0;
      face_idx -= face_num*nFace*param.ghostFace[1];
      if (face_num == 0) {
	const int idx = indexFromFaceIndex<1,nFace,0>(face_idx,param.ghostFace[1],param.parity,param.X);
	packFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
					param.inNorm,idx, face_idx, param.ghostFace[1], param);
      } else {
	const int idx = indexFromFaceIndex<1,nFace,1>(face_idx,param.ghostFace[1],param.parity,param.X);
	packFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
					param.inNorm,idx, face_idx, param.ghostFace[1], param);
      }
    } else if (dim == 2) {
      // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
      const int face_num = (face_idx >= nFace*param.ghostFace[2]) ? 1 : 0;
      face_idx -= face_num*nFace*param.ghostFace[2];
      if (face_num == 0) {
	const int idx = indexFromFaceIndex<2,nFace,0>(face_idx,param.ghostFace[2],param.parity,param.X);
	packFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
					param.inNorm,idx, face_idx, param.ghostFace[2], param);
      } else {
	const int idx = indexFromFaceIndex<2,nFace,1>(face_idx,param.ghostFace[2],param.parity,param.X);
	packFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
					param.inNorm,idx, face_idx, param.ghostFace[2], param);
      }
    } else {
      // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
      const int face_num = (face_idx >= nFace*param.ghostFace[3]) ? 1 : 0;
      face_idx -= face_num*nFace*param.ghostFace[3];
      if (face_num == 0) {
	const int idx = indexFromFaceIndex<3,nFace,0>(face_idx,param.ghostFace[3],param.parity,param.X);
	packFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
					param.inNorm,idx, face_idx, param.ghostFace[3], param);
      } else {
	const int idx = indexFromFaceIndex<3,nFace,1>(face_idx,param.ghostFace[3],param.parity,param.X);
	packFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
					param.inNorm,idx, face_idx, param.ghostFace[3], param);
      }
    }

  }


  template <int dagger, typename FloatN, int nFace>
    __global__ void packFaceExtendedWilsonKernel(PackParam<FloatN> param)
  {
    int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (face_idx >= param.threads) return;

    // determine which dimension we are packing
    const int dim = dimFromFaceIndex(face_idx, param);

    // compute where the output is located
    // compute an index into the local volume from the index into the face
    // read spinor, spin-project, and write half spinor to face
    if (dim == 0) {
      // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
      // if param.face_num==2 pack both the start and the end, otherwise pack the region of the lattice 
      // specified by param.face_num
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[0]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[0];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtended<0,nFace,0>(face_idx,param.ghostFace[0],param.parity,param.X,param.R);
	packFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
				       param.inNorm,idx, face_idx, param.ghostFace[0], param);
      } else {
	const int idx = indexFromFaceIndexExtended<0,nFace,1>(face_idx,param.ghostFace[0],param.parity,param.X,param.R);
	packFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
				       param.inNorm,idx, face_idx, param.ghostFace[0], param);
      }
    } else if (dim == 1) {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[1]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[1];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtended<1,nFace,0>(face_idx,param.ghostFace[1],param.parity,param.X,param.R);
	packFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
					param.inNorm,idx, face_idx, param.ghostFace[1], param);
      } else {
	const int idx = indexFromFaceIndexExtended<1,nFace,1>(face_idx,param.ghostFace[1],param.parity,param.X,param.R);
	packFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
					param.inNorm,idx, face_idx, param.ghostFace[1], param);
      }
    } else if (dim == 2) {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[2]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[2];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtended<2,nFace,0>(face_idx,param.ghostFace[2],param.parity,param.X,param.R);
	packFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
					param.inNorm,idx, face_idx, param.ghostFace[2], param);
      } else {
	const int idx = indexFromFaceIndexExtended<2,nFace,1>(face_idx,param.ghostFace[2],param.parity,param.X,param.R);
	packFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
					param.inNorm,idx, face_idx, param.ghostFace[2], param);
      }
    } else {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[3]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[3];

      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtended<3,nFace,0>(face_idx,param.ghostFace[3],param.parity,param.X,param.R);
	packFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
					param.inNorm,idx, face_idx, param.ghostFace[3], param);
      } else {
	const int idx = indexFromFaceIndexExtended<3,nFace,1>(face_idx,param.ghostFace[3],param.parity,param.X,param.R);
	packFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
					param.inNorm,idx, face_idx, param.ghostFace[3], param);
      }
    }

  }


  template <int dagger, typename FloatN, int nFace>
    __global__ void unpackFaceExtendedWilsonKernel(PackParam<FloatN> param)
  {
    int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (face_idx >= param.threads) return;

    // determine which dimension we are packing
    const int dim = dimFromFaceIndex(face_idx, param);

    // compute where the output is located
    // compute an index into the local volume from the index into the face
    // read spinor, spin-project, and write half spinor to face
    if (dim == 0) {
      // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
      // if param.face_num==2 pack both the start and the end, otherwise pack the region of the lattice 
      // specified by param.face_num
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[0]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[0];

      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtended<0,nFace,0>(face_idx,param.ghostFace[0],param.parity,param.X,param.R);
	unpackFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
					 param.inNorm,idx, face_idx, param.ghostFace[0], param);
      } else {
	const int idx = indexFromFaceIndexExtended<0,nFace,1>(face_idx,param.ghostFace[0],param.parity,param.X,param.R);
	unpackFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
					 param.inNorm,idx, face_idx, param.ghostFace[0], param);
      }
    } else if (dim == 1) {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[1]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[1];

      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtended<1,nFace,0>(face_idx,param.ghostFace[1],param.parity,param.X,param.R);
	unpackFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
					  param.inNorm,idx, face_idx, param.ghostFace[1], param);
      } else {
	const int idx = indexFromFaceIndexExtended<1,nFace,1>(face_idx,param.ghostFace[1],param.parity,param.X,param.R);
	unpackFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
					  param.inNorm,idx, face_idx, param.ghostFace[1], param);
      }
    } else if (dim == 2) {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[2]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[2];

      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtended<2,nFace,0>(face_idx,param.ghostFace[2],param.parity,param.X,param.R);
	unpackFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
					  param.inNorm,idx, face_idx, param.ghostFace[2], param);
      } else {
	const int idx = indexFromFaceIndexExtended<2,nFace,1>(face_idx,param.ghostFace[2],param.parity,param.X,param.R);
	unpackFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
					  param.inNorm,idx, face_idx, param.ghostFace[2], param);
      }
    } else {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[3]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[3];

      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtended<3,nFace,0>(face_idx,param.ghostFace[3],param.parity,param.X,param.R);
	unpackFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
					  param.inNorm,idx, face_idx, param.ghostFace[3], param);
      } else {
	const int idx = indexFromFaceIndexExtended<3,nFace,1>(face_idx,param.ghostFace[3],param.parity,param.X,param.R);
	unpackFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
					  param.inNorm,idx, face_idx, param.ghostFace[3], param);
      }
    }

  }

#endif // GPU_WILSON_DIRAC || GPU_DOMAIN_WALL_DIRAC


#if defined(GPU_WILSON_DIRAC) || defined(GPU_TWISTED_MASS_DIRAC)


#endif // GPU_WILSON_DIRAC || GPU_DOMAIN_WALL_DIRAC


#if defined(GPU_WILSON_DIRAC) || defined(GPU_TWISTED_MASS_DIRAC)

  // double precision

#endif // GPU_WILSON_DIRAC || GPU_DOMAIN_WALL_DIRAC


#if defined(GPU_WILSON_DIRAC) || defined(GPU_TWISTED_MASS_DIRAC)

  // double precision
#if (defined DIRECT_ACCESS_WILSON_PACK_SPINOR) || (defined FERMI_NO_DBLE_TEX)
#define READ_SPINOR READ_SPINOR_DOUBLE
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_DOUBLE_TEX
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexDouble
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_DOUBLE2
#define SPINOR_DOUBLE
  template <int dim, int dagger, int face_num>
    static inline __device__ void packTwistedFaceWilsonCore(double2 *out, float *outNorm, const double2 *in, 
							    const float *inNorm, double a, double b, const int &idx, 
							    const int &face_idx, const int &face_volume, 
							    PackParam<double2> &param)
  {
#if (__COMPUTE_CAPABILITY__ >= 130)
    if (dagger) {
#include "wilson_pack_twisted_face_dagger_core.h"
    } else {
#include "wilson_pack_twisted_face_core.h"
    }
#endif // (__COMPUTE_CAPABILITY__ >= 130)
  }
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR
#undef SPINOR_DOUBLE


  // single precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_SINGLE_TEX
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexSingle
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_FLOAT4
  template <int dim, int dagger, int face_num>
    static inline __device__ void packTwistedFaceWilsonCore(float4 *out, float *outNorm, const float4 *in, const float *inNorm, float a, float b,
							    const int &idx, const int &face_idx, 
							    const int &face_volume, 
							    const PackParam<float4> &param)
  {
    if (dagger) {
#include "wilson_pack_twisted_face_dagger_core.h"
    } else {
#include "wilson_pack_twisted_face_core.h"
    }
  }
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR


  // half precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_HALF_TEX
#define READ_SPINOR_UP READ_SPINOR_HALF_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexHalf
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_SHORT4
  template <int dim, int dagger, int face_num>
    static inline __device__ void packTwistedFaceWilsonCore(short4 *out, float *outNorm, const short4 *in, const float *inNorm, float a, float b,
							    const int &idx, const int &face_idx, 
							    const int &face_volume, 
							    const PackParam<short4> &param)
  {
    if (dagger) {
#include "wilson_pack_twisted_face_dagger_core.h"
    } else {
#include "wilson_pack_twisted_face_core.h"
    }
  }
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR

  template <int dagger, typename FloatN, typename Float>
    __global__ void packTwistedFaceWilsonKernel(Float a, Float b, PackParam<FloatN> param)
  {
    const int nFace = 1; // 1 face for Wilson

    int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (face_idx >= param.threads) return;

    // determine which dimension we are packing
    const int dim = dimFromFaceIndex(face_idx, param);

    // compute where the output is located
    // compute an index into the local volume from the index into the face
    // read spinor, spin-project, and write half spinor to face
    if (dim == 0) {
      // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
      const int face_num = (face_idx >= nFace*param.ghostFace[0]) ? 1 : 0;
      face_idx -= face_num*nFace*param.ghostFace[0];
      if (face_num == 0) {
	const int idx = indexFromFaceIndex<0,nFace,0>(face_idx,param.ghostFace[0],param.parity,param.X);
	packTwistedFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
					      param.inNorm, a, b, idx, face_idx, param.ghostFace[0], param);
      } else {
	const int idx = indexFromFaceIndex<0,nFace,1>(face_idx,param.ghostFace[0],param.parity,param.X);
	packTwistedFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
					      param.inNorm, a, b, idx, face_idx, param.ghostFace[0], param);
      }
    } else if (dim == 1) {
      const int face_num = (face_idx >= nFace*param.ghostFace[1]) ? 1 : 0;
      face_idx -= face_num*nFace*param.ghostFace[1];
      if (face_num == 0) {
	const int idx = indexFromFaceIndex<1,nFace,0>(face_idx,param.ghostFace[1],param.parity,param.X);
	packTwistedFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
					       param.inNorm, a, b, idx, face_idx, param.ghostFace[1], param);
      } else {
	const int idx = indexFromFaceIndex<1,nFace,1>(face_idx,param.ghostFace[1],param.parity,param.X);
	packTwistedFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
					       param.inNorm, a, b, idx, face_idx, param.ghostFace[1], param);
      }
    } else if (dim == 2) {
      const int face_num = (face_idx >= nFace*param.ghostFace[2]) ? 1 : 0;
      face_idx -= face_num*nFace*param.ghostFace[2];
      if (face_num == 0) {
	const int idx = indexFromFaceIndex<2,nFace,0>(face_idx,param.ghostFace[2],param.parity,param.X);
	packTwistedFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
					       param.inNorm, a, b, idx, face_idx, param.ghostFace[2], param);
      } else {
	const int idx = indexFromFaceIndex<2,nFace,1>(face_idx,param.ghostFace[2],param.parity,param.X);
	packTwistedFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
					       param.inNorm, a, b, idx, face_idx, param.ghostFace[2], param);
      }
    } else {
      const int face_num = (face_idx >= nFace*param.ghostFace[3]) ? 1 : 0;
      face_idx -= face_num*nFace*param.ghostFace[3];
      if (face_num == 0) {
	const int idx = indexFromFaceIndex<3,nFace,0>(face_idx,param.ghostFace[3],param.parity,param.X);
	packTwistedFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
					       param.inNorm, a, b,idx, face_idx, param.ghostFace[3], param);
      } else {
	const int idx = indexFromFaceIndex<3,nFace,1>(face_idx,param.ghostFace[3],param.parity,param.X);
	packTwistedFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
					       param.inNorm, a, b, idx, face_idx, param.ghostFace[3], param);
      }
    }

  }

#endif // GPU_TWISTED_MASS_DIRAC

  template <typename FloatN, typename Float>
    class PackFace : public Tunable {

  protected:
    FloatN *faces;
    const cudaColorSpinorField *in;
    const int dagger;
    const int parity;
    const int nFace;
    const int dim;
    const int face_num;

    // compute how many threads we need in total for the face packing
    unsigned int threads() const {
      unsigned int threads = 0;
      if(dim < 0){ // if dim is negative, pack all dimensions
        for (int i=0; i<4; i++) {
          if (!commDim[i]) continue;
          if ((i==3 && !(getKernelPackT() || getTwistPack()))) continue; 
          threads += 2*nFace*in->GhostFace()[i]; // 2 for forwards and backwards faces
        }
      }else{ // pack only in dim dimension
        if(commDim[dim] && dim!=3 || (getKernelPackT() || getTwistPack())){
          threads = nFace*in->GhostFace()[dim];
          if(face_num==2) threads *= 2; // sending data forwards and backwards
        }
      }
      return threads;
    }

    virtual int inputPerSite() const = 0;
    virtual int outputPerSite() const = 0;

    // prepare the param struct with kernel arguments
    PackParam<FloatN> prepareParam(int dim=-1, int face_num=2) {
      PackParam<FloatN> param;
      param.in = (FloatN*)in->V();
      param.inNorm = (float*)in->Norm();
      param.dim = dim;
      param.face_num = face_num;
      param.parity = parity;
      for(int d=0; d<QUDA_MAX_DIM; d++) param.X[d] = in->X()[d];
      param.X[0] *= 2;

#ifdef USE_TEXTURE_OBJECTS
      param.inTex = in->Tex();
      param.inTexNorm = in->TexNorm();
#endif

      param.threads = threads();
      param.sp_stride = in->Stride();

      int prev = -1; // previous dimension that was partitioned
      for (int i=0; i<4; i++) {
        param.threadDimMapLower[i] = 0;
        param.threadDimMapUpper[i] = 0;
        if (!commDim[i]) continue;
        param.threadDimMapLower[i] = (prev>=0 ? param.threadDimMapUpper[prev] : 0);
        param.threadDimMapUpper[i] = param.threadDimMapLower[i] + 2*nFace*in->GhostFace()[i];

        size_t faceBytes = nFace*outputPerSite()*in->GhostFace()[i]*sizeof(faces->x);

        if (typeid(FloatN) == typeid(short4) || typeid(FloatN) == typeid(short2)) {
          faceBytes += nFace*in->GhostFace()[i]*sizeof(float);
          param.out[2*i] = (FloatN*)((char*)faces + 
				     (outputPerSite()*sizeof(faces->x) + sizeof(float))*param.threadDimMapLower[i]);
          param.outNorm[2*i] = (float*)((char*)param.out[2*i] + 
					nFace*outputPerSite()*in->GhostFace()[i]*sizeof(faces->x));
        } else {
          param.out[2*i] = (FloatN*)((char*)faces+outputPerSite()*sizeof(faces->x)*param.threadDimMapLower[i]);
        }

        param.out[2*i+1] = (FloatN*)((char*)param.out[2*i] + faceBytes);
        param.outNorm[2*i+1] = (float*)((char*)param.outNorm[2*i] + faceBytes);

        prev=i;
      }

      param.ghostFace[0] = param.X[1]*param.X[2]*param.X[3]/2;
      param.ghostFace[1] = param.X[0]*param.X[2]*param.X[3]/2;
      param.ghostFace[2] = param.X[0]*param.X[1]*param.X[3]/2;
      param.ghostFace[3] = param.X[0]*param.X[1]*param.X[2]/2;

      return param;
    }

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return threads(); }

    void fillAux() {
      strcpy(aux, in->AuxString());
      char comm[5];
      comm[0] = (commDim[0] ? '1' : '0');
      comm[1] = (commDim[1] ? '1' : '0');
      comm[2] = (commDim[2] ? '1' : '0');
      comm[3] = (commDim[3] ? '1' : '0');
      comm[4] = '\0'; strcat(aux,",comm=");
      strcat(aux,comm);
      if (getKernelPackT() || getTwistPack()) { strcat(aux,",kernelPackT"); }
      switch (nFace) {
      case 1:
	strcat(aux,",nFace=1");
	break;
      case 3:
	strcat(aux,",nFace=3");
	break;
      default:
	errorQuda("Number of faces not supported");
      }
    }

  public:
    PackFace(FloatN *faces, const cudaColorSpinorField *in, 
	     const int dagger, const int parity, const int nFace, const int dim=-1, const int face_num=2)
      : faces(faces), in(in), dagger(dagger), 
	parity(parity), nFace(nFace), dim(dim), face_num(face_num) 
    { 
      fillAux(); 
      bindSpinorTex<FloatN>(in);
    }

    virtual ~PackFace() { 
      unbindSpinorTex<FloatN>(in);
    }

    virtual int tuningIter() const { return 3; }

    virtual TuneKey tuneKey() const {
      return TuneKey(in->VolString(), typeid(*this).name(), aux);
    }  

    virtual void apply(const cudaStream_t &stream) = 0;

    long long bytes() const { 
      size_t faceBytes = (inputPerSite() + outputPerSite())*this->threads()*sizeof(((FloatN*)0)->x);
      if (sizeof(((FloatN*)0)->x) == QUDA_HALF_PRECISION) 
        faceBytes += 2*this->threads()*sizeof(float); // 2 is from input and output
      return faceBytes;
    }
  };

  template <typename FloatN, typename Float>
    class PackFaceWilson : public PackFace<FloatN, Float> {

  private:

    int inputPerSite() const { return 24; } // input is full spinor
    int outputPerSite() const { return 12; } // output is spin projected

  public:
  PackFaceWilson(FloatN *faces, const cudaColorSpinorField *in, 
		 const int dagger, const int parity)
    : PackFace<FloatN, Float>(faces, in, dagger, parity, 1) { }
    virtual ~PackFaceWilson() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_WILSON_DIRAC
      PackParam<FloatN> param = this->prepareParam();
      if (this->dagger) {
        packFaceWilsonKernel<1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      } else {
        packFaceWilsonKernel<0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      }
#else
      errorQuda("Wilson face packing kernel is not built");
#endif  
    }

    long long flops() const { return outputPerSite()*this->threads(); }
  };

  void packFaceWilson(void *ghost_buf, cudaColorSpinorField &in, const int dagger, 
		      const int parity, const cudaStream_t &stream) {

    switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceWilson<double2, double> pack((double2*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceWilson<float4, float> pack((float4*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceWilson<short4, float> pack((short4*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
    }  
  }

  template <typename FloatN, typename Float>
    class PackFaceTwisted : public PackFace<FloatN, Float> {

  private:

    int inputPerSite() const { return 24; } // input is full spinor
    int outputPerSite() const { return 12; } // output is spin projected
    Float a;
    Float b;

  public:
  PackFaceTwisted(FloatN *faces, const cudaColorSpinorField *in, 
		  const int dagger, const int parity, Float a, Float b)
    : PackFace<FloatN, Float>(faces, in, dagger, parity, 1), a(a), b(b) { }
    virtual ~PackFaceTwisted() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_TWISTED_MASS_DIRAC
      PackParam<FloatN> param = this->prepareParam();
      if (this->dagger) {
        packTwistedFaceWilsonKernel<1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(a, b, param);
      } else {
        packTwistedFaceWilsonKernel<0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(a, b, param);
      }
#else
      errorQuda("Twisted face packing kernel is not built");
#endif  
    }

    long long flops() const { return outputPerSite()*this->threads(); }
  };

  //!
  void packTwistedFaceWilson(void *ghost_buf, cudaColorSpinorField &in, const int dagger, 
			     const int parity, const double a, const double b, const cudaStream_t &stream) {

    switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceTwisted<double2, double> pack((double2*)ghost_buf, &in, dagger, parity, a, b);
        pack.apply(stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceTwisted<float4, float> pack((float4*)ghost_buf, &in, dagger, parity, (float)a, (float)b);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceTwisted<short4, float> pack((short4*)ghost_buf, &in, dagger, parity, (float)a, (float)b);
        pack.apply(stream);
      }
      break;
    }  
  }

#ifdef GPU_STAGGERED_DIRAC

#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEXDOUBLE param.inTex
#define SPINORTEXSINGLE param.inTex
#define SPINORTEXHALF param.inTex
#define SPINORTEXHALFNORM param.inTexNorm
#else
#define SPINORTEXDOUBLE spinorTexDouble
#define SPINORTEXSINGLE spinorTexSingle2
#define SPINORTEXHALF spinorTexHalf2
#define SPINORTEXHALFNORM spinorTexHalf2Norm
#endif

  template <typename Float2>
    __device__ void packFaceStaggeredCore(Float2 *out, float *outNorm, const int out_idx, 
					  const int out_stride, const Float2 *in, const float *inNorm, 
					  const int in_idx, const int in_stride) {
    out[out_idx + 0*out_stride] = in[in_idx + 0*in_stride];
    out[out_idx + 1*out_stride] = in[in_idx + 1*in_stride];
    out[out_idx + 2*out_stride] = in[in_idx + 2*in_stride];
  }	
  template<> 
    __device__ void packFaceStaggeredCore(short2 *out, float *outNorm, const int out_idx, 
					  const int out_stride, const short2 *in, const float *inNorm, 
					  const int in_idx, const int in_stride) {
    out[out_idx + 0*out_stride] = in[in_idx + 0*in_stride];
    out[out_idx + 1*out_stride] = in[in_idx + 1*in_stride];
    out[out_idx + 2*out_stride] = in[in_idx + 2*in_stride];
    outNorm[out_idx] = inNorm[in_idx];
  }

#if (defined DIRECT_ACCESS_PACK) || (defined FERMI_NO_DBLE_TEX)
  template <typename Float2>
    __device__ void packFaceStaggeredCore(Float2 *out, float *outNorm, const int out_idx, 
					  const int out_stride, const Float2 *in, const float *inNorm, 
					  const int in_idx, const PackParam<double2> &param) {
    out[out_idx + 0*out_stride] = in[in_idx + 0*param.sp_stride];
    out[out_idx + 1*out_stride] = in[in_idx + 1*param.sp_stride];
    out[out_idx + 2*out_stride] = in[in_idx + 2*param.sp_stride];
  }	
  template<> 
    __device__ void packFaceStaggeredCore(short2 *out, float *outNorm, const int out_idx, 
					  const int out_stride, const short2 *in, const float *inNorm, 
					  const int in_idx, const PackParam<double2> &param) {
    out[out_idx + 0*out_stride] = in[in_idx + 0*param.sp_stride];
    out[out_idx + 1*out_stride] = in[in_idx + 1*param.sp_stride];
    out[out_idx + 2*out_stride] = in[in_idx + 2*param.sp_stride];
    outNorm[out_idx] = inNorm[in_idx];
  }


#else
#if __COMPUTE_CAPABILITY__ >= 130
  __device__ void packFaceStaggeredCore(double2 *out, float *outNorm, const int out_idx, 
					const int out_stride, const double2 *in, const float *inNorm, 
					const int in_idx, const PackParam<double2> &param) {
    out[out_idx + 0*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 0*param.sp_stride);
    out[out_idx + 1*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 1*param.sp_stride);
    out[out_idx + 2*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 2*param.sp_stride);
  }	
#endif
  __device__ void packFaceStaggeredCore(float2 *out, float *outNorm, const int out_idx, 
					const int out_stride, const float2 *in, 
					const float *inNorm, const int in_idx, 
					const PackParam<float2> &param) {
    out[out_idx + 0*out_stride] = TEX1DFETCH(float2, SPINORTEXSINGLE, in_idx + 0*param.sp_stride);
    out[out_idx + 1*out_stride] = TEX1DFETCH(float2, SPINORTEXSINGLE, in_idx + 1*param.sp_stride);
    out[out_idx + 2*out_stride] = TEX1DFETCH(float2, SPINORTEXSINGLE, in_idx + 2*param.sp_stride);	
  }

  // this is rather dumb: undoing the texture load because cudaNormalizedReadMode is used
  // should really bind to an appropriate texture instead of reusing
  static inline __device__ short2 float22short2(float c, float2 a) {
    return make_short2((short)(a.x*c*MAX_SHORT), (short)(a.y*c*MAX_SHORT));
  }

  __device__ void packFaceStaggeredCore(short2 *out, float *outNorm, const int out_idx, 
					const int out_stride, const short2 *in, 
					const float *inNorm, const int in_idx, 
					const PackParam<short2> &param) {
    out[out_idx + 0*out_stride] = float22short2(1.0f,TEX1DFETCH(float2,SPINORTEXHALF,in_idx+0*param.sp_stride));
    out[out_idx + 1*out_stride] = float22short2(1.0f,TEX1DFETCH(float2,SPINORTEXHALF,in_idx+1*param.sp_stride));
    out[out_idx + 2*out_stride] = float22short2(1.0f,TEX1DFETCH(float2,SPINORTEXHALF,in_idx+2*param.sp_stride));
    outNorm[out_idx] = TEX1DFETCH(float, SPINORTEXHALFNORM, in_idx);
  }
#endif


  template <typename FloatN, int nFace>
    __global__ void packFaceStaggeredKernel(PackParam<FloatN> param)
  {
    int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (face_idx >= param.threads) return;

    // determine which dimension we are packing
    const int dim = dimFromFaceIndex(face_idx, param);

    // compute where the output is located
    // compute an index into the local volume from the index into the face
    // read spinor and write to face
    if (dim == 0) {
      // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[0]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[0];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexStaggered<0,nFace,0>(face_idx,param.ghostFace[0],param.parity,param.X);
	packFaceStaggeredCore(param.out[0], param.outNorm[0], face_idx, 
			      nFace*param.ghostFace[0], param.in, param.inNorm, idx, param);
      } else {
	const int idx = indexFromFaceIndexStaggered<0,nFace,1>(face_idx,param.ghostFace[0],param.parity,param.X);
	packFaceStaggeredCore(param.out[1], param.outNorm[1], face_idx,
			      nFace*param.ghostFace[0], param.in, param.inNorm, idx, param);
      }
    } else if (dim == 1) {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[1]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[1];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexStaggered<1,nFace,0>(face_idx,param.ghostFace[1],param.parity,param.X);
	packFaceStaggeredCore(param.out[2], param.outNorm[2], face_idx, 
			      nFace*param.ghostFace[1], param.in, param.inNorm, idx, param);
      } else {
	const int idx = indexFromFaceIndexStaggered<1,nFace,1>(face_idx,param.ghostFace[1],param.parity,param.X);
	packFaceStaggeredCore(param.out[3], param.outNorm[3], face_idx, 
			      nFace*param.ghostFace[1], param.in, param.inNorm, idx, param);
      }
    } else if (dim == 2) {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[2]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[2];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexStaggered<2,nFace,0>(face_idx,param.ghostFace[2],param.parity,param.X);
	packFaceStaggeredCore(param.out[4], param.outNorm[4], face_idx,
			      nFace*param.ghostFace[2], param.in, param.inNorm, idx, param);
      } else {
	const int idx = indexFromFaceIndexStaggered<2,nFace,1>(face_idx,param.ghostFace[2],param.parity,param.X);
	packFaceStaggeredCore(param.out[5], param.outNorm[5], face_idx,
			      nFace*param.ghostFace[2], param.in, param.inNorm, idx, param);
      }
    } else {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[3]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[3];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexStaggered<3,nFace,0>(face_idx,param.ghostFace[3],param.parity,param.X);
	packFaceStaggeredCore(param.out[6], param.outNorm[6], face_idx,
			      nFace*param.ghostFace[3], param.in, param.inNorm,idx, param);
      } else {
	const int idx = indexFromFaceIndexStaggered<3,nFace,1>(face_idx,param.ghostFace[3],param.parity,param.X);
	packFaceStaggeredCore(param.out[7], param.outNorm[7], face_idx, 
			      nFace*param.ghostFace[3], param.in, param.inNorm, idx, param);
      }
    }

  }


  template <typename FloatN, int nFace>
    __global__ void packFaceExtendedStaggeredKernel(PackExtendedParam<FloatN> param)
  {
    int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (face_idx >= param.threads) return;

    // determine which dimension we are packing
    const int dim = dimFromFaceIndex(face_idx, param);

    // compute where the output is located
    // compute an index into the local volume from the index into the face
    // read spinor and write half spinor to face
    if (dim == 0) {
      // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
      // if param.face_num==2 pack both the start and the end, otherwise pack the region of the 
      // lattice specified by param.face_num
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[0]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[0];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtendedStaggered<0,nFace,0>(face_idx,param.ghostFace[0],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.out[0], param.outNorm[0], face_idx, 
			      nFace*param.ghostFace[0], param.in, param.inNorm, idx, param);
      } else {
	const int idx = indexFromFaceIndexExtendedStaggered<0,nFace,1>(face_idx,param.ghostFace[0],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.out[1], param.outNorm[1], face_idx,
			      nFace*param.ghostFace[0], param.in, param.inNorm, idx, param);
      }
    } else if (dim == 1) {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[1]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[1];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtendedStaggered<1,nFace,0>(face_idx,param.ghostFace[1],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.out[2], param.outNorm[2], face_idx, 
			      nFace*param.ghostFace[1], param.in, param.inNorm, idx, param);
      } else {
	const int idx = indexFromFaceIndexExtendedStaggered<1,nFace,1>(face_idx,param.ghostFace[1],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.out[3], param.outNorm[3], face_idx, 
			      nFace*param.ghostFace[1], param.in, param.inNorm, idx, param);
      }
    } else if (dim == 2) {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[2]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[2];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtendedStaggered<2,nFace,0>(face_idx,param.ghostFace[2],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.out[4], param.outNorm[4], face_idx,
			      nFace*param.ghostFace[2], param.in, param.inNorm, idx, param);
      } else {
	const int idx = indexFromFaceIndexExtendedStaggered<2,nFace,1>(face_idx,param.ghostFace[2],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.out[5], param.outNorm[5], face_idx,
			      nFace*param.ghostFace[2], param.in, param.inNorm, idx, param);
      }
    } else {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[3]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[3];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtendedStaggered<3,nFace,0>(face_idx,param.ghostFace[3],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.out[6], param.outNorm[6], face_idx,
			      nFace*param.ghostFace[3], param.in, param.inNorm,idx, param);
      } else {
	const int idx = indexFromFaceIndexExtendedStaggered<3,nFace,1>(face_idx,param.ghostFace[3],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.out[7], param.outNorm[7], face_idx, 
			      nFace*param.ghostFace[3], param.in, param.inNorm, idx, param);
      }
    }

  }


  template <typename FloatN, int nFace>
    __global__ void unpackFaceExtendedStaggeredKernel(PackExtendedParam<FloatN> param)
  {
    int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (face_idx >= param.threads) return;

    // determine which dimension we are packing
    const int dim = dimFromFaceIndex(face_idx, param);

    // compute where the output is located
    // compute an index into the local volume from the index into the face
    // read spinor, spin-project, and write half spinor to face
    if (dim == 0) {
      // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
      // if param.face_num==2 pack both the start and the end, otherwist pack the region of the 
      // lattice specified by param.face_num
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[0]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[0];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtendedStaggered<0,nFace,0>(face_idx,param.ghostFace[0],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.in, param.inNorm, idx, 
			      param.sp_stride, param.out[0], param.outNorm[0], face_idx, nFace*param.ghostFace[0]);
      } else {
	const int idx = indexFromFaceIndexExtendedStaggered<0,nFace,1>(face_idx,param.ghostFace[0],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.in, param.inNorm, idx, 
			      param.sp_stride, param.out[1], param.outNorm[1], face_idx, nFace*param.ghostFace[0]);
      }
    } else if (dim == 1) {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[1]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[1];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtendedStaggered<1,nFace,0>(face_idx,param.ghostFace[1],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.in, param.inNorm, idx, 
			      param.sp_stride, param.out[2], param.outNorm[2], face_idx, nFace*param.ghostFace[1]);
      } else {
	const int idx = indexFromFaceIndexExtendedStaggered<1,nFace,1>(face_idx,param.ghostFace[1],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.in, param.inNorm, idx, 
			      param.sp_stride, param.out[3], param.outNorm[3], face_idx, nFace*param.ghostFace[1]);
      }
    } else if (dim == 2) {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[2]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[2];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtendedStaggered<2,nFace,0>(face_idx,param.ghostFace[2],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.in, param.inNorm, idx, 
			      param.sp_stride, param.out[4], param.outNorm[4], face_idx, nFace*param.ghostFace[2]);
      } else {
	const int idx = indexFromFaceIndexExtendedStaggered<2,nFace,1>(face_idx,param.ghostFace[2],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.in, param.inNorm, idx, 
			      param.sp_stride, param.out[5], param.outNorm[5], face_idx, nFace*param.ghostFace[2]);
      }
    } else {
      const int face_num = (param.face_num==2) ? ((face_idx >= nFace*param.ghostFace[3]) ? 1 : 0) : param.face_num;
      if(param.face_num==2) face_idx -= face_num*nFace*param.ghostFace[3];
      if (face_num == 0) {
	const int idx = indexFromFaceIndexExtendedStaggered<3,nFace,0>(face_idx,param.ghostFace[3],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.in, param.inNorm, idx, 
			      param.sp_stride, param.out[6], param.outNorm[6], face_idx, nFace*param.ghostFace[3]);
      } else {
	const int idx = indexFromFaceIndexExtendedStaggered<3,nFace,1>(face_idx,param.ghostFace[3],param.parity,param.X,param.R);
	packFaceStaggeredCore(param.in, param.inNorm, idx, 
			      param.sp_stride, param.out[7], param.outNorm[7], face_idx, nFace*param.ghostFace[3]);
      }
    }

  }


#undef SPINORTEXDOUBLE
#undef SPINORTEXSINGLE
#undef SPINORTEXHALF

#endif // GPU_STAGGERED_DIRAC


  template <typename FloatN, typename Float>
    class PackFaceStaggered : public PackFace<FloatN, Float> {

  private:
    const int* R; // boundary dimensions for extended field
    const bool unpack; 

    int inputPerSite() const { return 6; } // input is full spinor
    int outputPerSite() const { return 6; } // output is full spinor


  public:
  PackFaceStaggered(FloatN *faces, const cudaColorSpinorField *in, 
		    const int nFace, const int dagger, const int parity, 
		    const int dim, const int face_num, const int* R=NULL, const bool unpack=false)
    : PackFace<FloatN, Float>(faces, in, dagger, parity, nFace, dim, face_num), R(R), unpack(unpack) { }
    virtual ~PackFaceStaggered() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_STAGGERED_DIRAC

      PackParam<FloatN> param = this->prepareParam(this->dim, this->face_num);
      if(!R){
        if (PackFace<FloatN,Float>::nFace==1) {
          packFaceStaggeredKernel<FloatN, 1> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
        } else {
          packFaceStaggeredKernel<FloatN, 3> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
        }
      }else{ // R!=NULL => this is an extended field
        PackExtendedParam<FloatN> extendedParam(param);
        if(!unpack){
          for(int d=0; d<QUDA_MAX_DIM; ++d) extendedParam.R[d] = R[d];
	  switch(PackFace<FloatN,Float>::nFace){
	  case 1:
	    packFaceExtendedStaggeredKernel<FloatN,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(extendedParam);
	    break;

	  case 2:
	    packFaceExtendedStaggeredKernel<FloatN,2><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(extendedParam);
	    break;
	
	  case 3:
	    packFaceExtendedStaggeredKernel<FloatN,3><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(extendedParam);
	    break;

	  case 4:
	    packFaceExtendedStaggeredKernel<FloatN,4><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(extendedParam);
	    break;

	  default: 
	    errorQuda("Unsupported boundary width");
	    break;
	  }
        }else{ // extended field unpack
	  switch(PackFace<FloatN,Float>::nFace){
	  case 1:
	    unpackFaceExtendedStaggeredKernel<FloatN,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(extendedParam);
	    break;

	  case 2:
	    unpackFaceExtendedStaggeredKernel<FloatN,2><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(extendedParam);
	    break;

	  case 3:
	    unpackFaceExtendedStaggeredKernel<FloatN,3><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(extendedParam);
	    break;

	  case 4:
	    unpackFaceExtendedStaggeredKernel<FloatN,4><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(extendedParam);
	    break;

	  default:
	    errorQuda("Unsupported boundary width");
	    break;
	  }
	}
      }
#else
      errorQuda("Staggered face packing kernel is not built");
#endif  
    }

    long long flops() const { return 0; }
  };


  void packFaceStaggered(void *ghost_buf, cudaColorSpinorField &in, int nFace, 
			 int dagger, int parity, const int dim, const int face_num, const cudaStream_t &stream) {

    switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
#if __COMPUTE_CAPABILITY__ >= 130
        PackFaceStaggered<double2, double> pack((double2*)ghost_buf, &in, nFace, dagger, parity, dim, face_num);
        pack.apply(stream);
#endif
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceStaggered<float2, float> pack((float2*)ghost_buf, &in, nFace, dagger, parity, dim, face_num);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceStaggered<short2, float> pack((short2*)ghost_buf, &in, nFace, dagger, parity, dim, face_num);
        pack.apply(stream);
      }
      break;
    }  
  }

  void packFaceExtendedStaggered(void *buffer, cudaColorSpinorField &field, const int nFace, const int R[],
				 int dagger, int parity, const int dim, const int face_num, const cudaStream_t &stream, bool unpack=false)
  {
    switch(field.Precision()){
    case QUDA_DOUBLE_PRECISION:
      {
#if __COMPUTE_CAPABILITY__ >= 130
        PackFaceStaggered<double2,double> pack(static_cast<double2*>(buffer), &field, nFace, dagger, parity, dim, face_num, R, unpack);
        pack.apply(stream);  
#endif
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceStaggered<float2,float> pack(static_cast<float2*>(buffer), &field, nFace, dagger, parity, dim, face_num, R, unpack);
        pack.apply(stream);  
      } 
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceStaggered<short2,float> pack(static_cast<short2*>(buffer), &field, nFace, dagger, parity, dim, face_num, R, unpack);
        pack.apply(stream);  
      }
      break;

    } // switch(field.Precision())
  }

#ifdef GPU_DOMAIN_WALL_DIRAC
  template <int dagger, typename FloatN>
    __global__ void packFaceDWKernel(PackParam<FloatN> param)
  {
    const int nFace = 1; // 1 face for dwf

    int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (face_idx >= param.threads) return;

    // determine which dimension we are packing
    const int dim = dimFromFaceIndex(face_idx, param);

    const int Ls = param.X[4];

    // compute where the output is located
    // compute an index into the local volume from the index into the face
    // read spinor, spin-project, and write half spinor to face
    if (dim == 0) {
      // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
      // FIXME these param.ghostFace constants do not incude the Ls dimension
      const int face_num = (face_idx >= nFace*Ls*param.ghostFace[0]) ? 1 : 0; 
      face_idx -= face_num*nFace*Ls*param.ghostFace[0];
      if (face_num == 0) {
	const int idx = indexFromDWFaceIndex<0,nFace,0>(face_idx,Ls*param.ghostFace[0],param.parity,param.X);
	packFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
				       param.inNorm, idx, face_idx, Ls*param.ghostFace[0], param);
      } else {
	const int idx = indexFromDWFaceIndex<0,nFace,1>(face_idx,Ls*param.ghostFace[0],param.parity,param.X);
	packFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
				       param.inNorm, idx, face_idx, Ls*param.ghostFace[0], param);
      }
    } else if (dim == 1) {
      const int face_num = (face_idx >= nFace*Ls*param.ghostFace[1]) ? 1 : 0; 
      face_idx -= face_num*nFace*Ls*param.ghostFace[1];
      if (face_num == 0) {
	const int idx = indexFromDWFaceIndex<1,nFace,0>(face_idx,Ls*param.ghostFace[1],param.parity,param.X);
	packFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
					param.inNorm, idx, face_idx, Ls*param.ghostFace[1], param);
      } else {
	const int idx = indexFromDWFaceIndex<1,nFace,1>(face_idx,Ls*param.ghostFace[1],param.parity,param.X);
	packFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
					param.inNorm, idx, face_idx, Ls*param.ghostFace[1], param);
      }
    } else if (dim == 2) {
      const int face_num = (face_idx >= nFace*Ls*param.ghostFace[2]) ? 1 : 0; 
      face_idx -= face_num*nFace*Ls*param.ghostFace[2];
      if (face_num == 0) {
	const int idx = indexFromDWFaceIndex<2,nFace,0>(face_idx,Ls*param.ghostFace[2],param.parity,param.X);
	packFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
					param.inNorm, idx, face_idx, Ls*param.ghostFace[2], param);
      } else {
	const int idx = indexFromDWFaceIndex<2,nFace,1>(face_idx,Ls*param.ghostFace[2],param.parity,param.X);
	packFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
					param.inNorm, idx, face_idx, Ls*param.ghostFace[2], param);
      }
    } else {
      const int face_num = (face_idx >= nFace*Ls*param.ghostFace[3]) ? 1 : 0; 
      face_idx -= face_num*nFace*Ls*param.ghostFace[3];
      if (face_num == 0) {
	const int idx = indexFromDWFaceIndex<3,nFace,0>(face_idx,Ls*param.ghostFace[3],param.parity,param.X);
	packFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
					param.inNorm, idx, face_idx, Ls*param.ghostFace[3], param);
      } else {
	const int idx = indexFromDWFaceIndex<3,nFace,1>(face_idx,Ls*param.ghostFace[3],param.parity,param.X);
	packFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
					param.inNorm, idx, face_idx, Ls*param.ghostFace[3], param);
      }
    }
  }


  template <int dagger, typename FloatN>
    __global__ void packFaceDW4DKernel(PackParam<FloatN> param)
  {
    const int nFace = 1; // 1 face for Wilson
  
    int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (face_idx >= param.threads) return;

    const int Ls = param.X[4];

    // determine which dimension we are packing
    const int dim = dimFromFaceIndex(face_idx, param);

    // compute where the output is located
    // compute an index into the local volume from the index into the face
    // read spinor, spin-project, and write half spinor to face
    if (dim == 0) {
      // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
      // FIXME these param.ghostFace constants do not incude the Ls dimension
      const int face_num = (face_idx >= nFace*Ls*param.ghostFace[0]) ? 1 : 0; 
      face_idx -= face_num*nFace*Ls*param.ghostFace[0];
        if (face_num == 0) {
	const int idx = indexFromDW4DFaceIndex<0,nFace,0>(face_idx,Ls*param.ghostFace[0],param.parity,param.X);
	packFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
				       param.inNorm, idx, face_idx, Ls*param.ghostFace[0], param);
      } else {
	const int idx = indexFromDW4DFaceIndex<0,nFace,1>(face_idx,Ls*param.ghostFace[0],param.parity,param.X);
	packFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
				       param.inNorm, idx, face_idx, Ls*param.ghostFace[0], param);
      }
    } else if (dim == 1) {
      const int face_num = (face_idx >= nFace*Ls*param.ghostFace[1]) ? 1 : 0; 
      face_idx -= face_num*nFace*Ls*param.ghostFace[1];
      if (face_num == 0) {
	const int idx = indexFromDW4DFaceIndex<1,nFace,0>(face_idx,Ls*param.ghostFace[1],param.parity,param.X);
	packFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
					param.inNorm, idx, face_idx, Ls*param.ghostFace[1], param);
      } else {
	const int idx = indexFromDW4DFaceIndex<1,nFace,1>(face_idx,Ls*param.ghostFace[1],param.parity,param.X);
	packFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
					param.inNorm, idx, face_idx, Ls*param.ghostFace[1], param);
      }
    } else if (dim == 2) {
      const int face_num = (face_idx >= nFace*Ls*param.ghostFace[2]) ? 1 : 0; 
      face_idx -= face_num*nFace*Ls*param.ghostFace[2];
      if (face_num == 0) {
	const int idx = indexFromDW4DFaceIndex<2,nFace,0>(face_idx,Ls*param.ghostFace[2],param.parity,param.X);
	packFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
					param.inNorm, idx, face_idx, Ls*param.ghostFace[2], param);
      } else {
	const int idx = indexFromDW4DFaceIndex<2,nFace,1>(face_idx,Ls*param.ghostFace[2],param.parity,param.X);
	packFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
					param.inNorm, idx, face_idx, Ls*param.ghostFace[2], param);
      }
    } else {
      const int face_num = (face_idx >= nFace*Ls*param.ghostFace[3]) ? 1 : 0; 
      face_idx -= face_num*nFace*Ls*param.ghostFace[3];
      if (face_num == 0) {
	const int idx = indexFromDW4DFaceIndex<3,nFace,0>(face_idx,Ls*param.ghostFace[3],param.parity,param.X);
	packFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
					param.inNorm, idx, face_idx, Ls*param.ghostFace[3], param);
      } else {
	const int idx = indexFromDW4DFaceIndex<3,nFace,1>(face_idx,Ls*param.ghostFace[3],param.parity,param.X);
	packFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
					param.inNorm, idx, face_idx, Ls*param.ghostFace[3], param);
      }
    }
  }

#endif

  template <typename FloatN, typename Float>
    class PackFaceDW : public PackFace<FloatN, Float> {

  private:

    int inputPerSite() const { return 24; } // input is full spinor
    int outputPerSite() const { return 12; } // output is spin projected

  public:
  PackFaceDW(FloatN *faces, const cudaColorSpinorField *in, 
	     const int dagger, const int parity)
    : PackFace<FloatN, Float>(faces, in, dagger, parity, 1) { }
    virtual ~PackFaceDW() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_DOMAIN_WALL_DIRAC
      PackParam<FloatN> param = this->prepareParam();
      if (this->dagger) {
        packFaceDWKernel<1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      } else {
        packFaceDWKernel<0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      }
#else
      errorQuda("DW face packing kernel is not built");
#endif  
    }

    long long flops() const { return outputPerSite()*this->threads(); }
  };

  template <typename FloatN, typename Float>
    class PackFaceDW4D : public PackFace<FloatN, Float> {

  private:

    int inputPerSite() const { return 24; } // input is full spinor
    int outputPerSite() const { return 12; } // output is spin projected

  public:
  PackFaceDW4D(FloatN *faces, const cudaColorSpinorField *in, 
	       const int dagger, const int parity)
    : PackFace<FloatN, Float>(faces, in, dagger, parity, 1) { }
    virtual ~PackFaceDW4D() { }
  
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    
#ifdef GPU_DOMAIN_WALL_DIRAC
      PackParam<FloatN> param = this->prepareParam();
      if (this->dagger) {
	packFaceDW4DKernel<1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      } else {
	packFaceDW4DKernel<0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      }
#else
      errorQuda("4D preconditioned DW face packing kernel is not built");
#endif  
    }

    long long flops() const { return outputPerSite()*this->threads(); }
  };

  void packFaceDW(void *ghost_buf, cudaColorSpinorField &in, const int dagger,  
		  const int parity, const cudaStream_t &stream) {


    if(in.DWFPCtype() == QUDA_4D_PC)
      {
	switch(in.Precision()) {
	case QUDA_DOUBLE_PRECISION:
	  {
	    PackFaceDW4D<double2, double> pack((double2*)ghost_buf, &in, dagger, parity);
	    pack.apply(stream);
	  }
	  break;
	case QUDA_SINGLE_PRECISION:
	  {
	    PackFaceDW4D<float4, float> pack((float4*)ghost_buf, &in, dagger, parity);
	    pack.apply(stream);
	  }
	  break;
	case QUDA_HALF_PRECISION:
	  {
	    PackFaceDW4D<short4, float> pack((short4*)ghost_buf, &in, dagger, parity);
	    pack.apply(stream);
	  }
	  break;
	}  
      }
    else
      {
	switch(in.Precision()) {
	case QUDA_DOUBLE_PRECISION:
	  {
	    PackFaceDW<double2, double> pack((double2*)ghost_buf, &in, dagger, parity);
	    pack.apply(stream);
	  }
	  break;
	case QUDA_SINGLE_PRECISION:
	  {
	    PackFaceDW<float4, float> pack((float4*)ghost_buf, &in, dagger, parity);
	    pack.apply(stream);
	  }
	  break;
	case QUDA_HALF_PRECISION:
	  {
	    PackFaceDW<short4, float> pack((short4*)ghost_buf, &in, dagger, parity);
	    pack.apply(stream);
	  }
	  break;
	}  
      }
  }

#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
  template <int dagger, typename FloatN>
    __global__ void packFaceNdegTMKernel(PackParam<FloatN> param)
  {
    const int nFace = 1; // 1 face for Wilson
    const int Nf = 2;

    int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (face_idx >= param.threads) return;

    // determine which dimension we are packing
    const int dim = dimFromFaceIndex(face_idx, param);

    // compute where the output is located
    // compute an index into the local volume from the index into the face
    // read spinor, spin-project, and write half spinor to face
    if (dim == 0) {
      // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
      // FIXME these param.ghostFace constants do not include the Nf dimension
      const int face_num = (face_idx >= nFace*Nf*param.ghostFace[0]) ? 1 : 0;
      face_idx -= face_num*nFace*Nf*param.ghostFace[0];
      if (face_num == 0) {
	const int idx = indexFromNdegTMFaceIndex<0,nFace,0>(face_idx,Nf*param.ghostFace[0],param.parity,param.X);
	packFaceWilsonCore<0,dagger,0>(param.out[0], param.outNorm[0], param.in, 
				       param.inNorm, idx, face_idx, Nf*param.ghostFace[0], param);
      } else {
	const int idx = indexFromNdegTMFaceIndex<0,nFace,1>(face_idx,Nf*param.ghostFace[0],param.parity,param.X);
	packFaceWilsonCore<0,dagger,1>(param.out[1], param.outNorm[1], param.in, 
				       param.inNorm, idx, face_idx, Nf*param.ghostFace[0], param);
      }
    } else if (dim == 1) {
      const int face_num = (face_idx >= nFace*Nf*param.ghostFace[1]) ? 1 : 0;
      face_idx -= face_num*nFace*Nf*param.ghostFace[1];
      if (face_num == 0) {
	const int idx = indexFromNdegTMFaceIndex<1,nFace,0>(face_idx,Nf*param.ghostFace[1],param.parity,param.X);
	packFaceWilsonCore<1, dagger,0>(param.out[2], param.outNorm[2], param.in, 
					param.inNorm, idx, face_idx, Nf*param.ghostFace[1], param);
      } else {
	const int idx = indexFromNdegTMFaceIndex<1,nFace,1>(face_idx,Nf*param.ghostFace[1],param.parity,param.X);
	packFaceWilsonCore<1, dagger,1>(param.out[3], param.outNorm[3], param.in, 
					param.inNorm, idx, face_idx, Nf*param.ghostFace[1], param);
      }
    } else if (dim == 2) {
      const int face_num = (face_idx >= nFace*Nf*param.ghostFace[2]) ? 1 : 0;
      face_idx -= face_num*nFace*Nf*param.ghostFace[2];
      if (face_num == 0) {
	const int idx = indexFromNdegTMFaceIndex<2,nFace,0>(face_idx,Nf*param.ghostFace[2],param.parity,param.X);
	packFaceWilsonCore<2, dagger,0>(param.out[4], param.outNorm[4], param.in, 
					param.inNorm, idx, face_idx, Nf*param.ghostFace[2], param);
      } else {
	const int idx = indexFromNdegTMFaceIndex<2,nFace,1>(face_idx,Nf*param.ghostFace[2],param.parity,param.X);
	packFaceWilsonCore<2, dagger,1>(param.out[5], param.outNorm[5], param.in, 
					param.inNorm, idx, face_idx, Nf*param.ghostFace[2], param);
      }
    } else {
      const int face_num = (face_idx >= nFace*Nf*param.ghostFace[3]) ? 1 : 0;
      face_idx -= face_num*nFace*Nf*param.ghostFace[3];
      if (face_num == 0) {
	const int idx = indexFromNdegTMFaceIndex<3,nFace,0>(face_idx,Nf*param.ghostFace[3],param.parity,param.X);
	packFaceWilsonCore<3, dagger,0>(param.out[6], param.outNorm[6], param.in, 
					param.inNorm, idx, face_idx, Nf*param.ghostFace[3], param);
      } else {
	const int idx = indexFromNdegTMFaceIndex<3,nFace,1>(face_idx,Nf*param.ghostFace[3],param.parity,param.X);
	packFaceWilsonCore<3, dagger,1>(param.out[7], param.outNorm[7], param.in, 
					param.inNorm, idx, face_idx, Nf*param.ghostFace[3], param);
      }
    }
  }
#endif

  template <typename FloatN, typename Float>
    class PackFaceNdegTM : public PackFace<FloatN, Float> {

  private:

    int inputPerSite() const { return 24; } // input is full spinor
    int outputPerSite() const { return 12; } // output is spin projected

  public:
  PackFaceNdegTM(FloatN *faces, const cudaColorSpinorField *in, 
		 const int dagger, const int parity)
    : PackFace<FloatN, Float>(faces, in, dagger, parity, 1) { }
    virtual ~PackFaceNdegTM() { }

    void apply(const cudaStream_t &stream) {    
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
      PackParam<FloatN> param = this->prepareParam();
      if (this->dagger) {
        packFaceNdegTMKernel<1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      } else {
        packFaceNdegTMKernel<0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(param);
      }
#else
      errorQuda("Non-degenerate twisted mass face packing kernel is not built");
#endif  
    }

    long long flops() const { return outputPerSite()*this->threads(); }
  };

  void packFaceNdegTM(void *ghost_buf, cudaColorSpinorField &in, const int dagger, 
		      const int parity, const cudaStream_t &stream) {

    switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceNdegTM<double2, double> pack((double2*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceNdegTM<float4, float> pack((float4*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceNdegTM<short4, float> pack((short4*)ghost_buf, &in, dagger, parity);
        pack.apply(stream);
      }
      break;
    } 
  }

  void packFace(void *ghost_buf, cudaColorSpinorField &in, const int nFace, 
		const int dagger, const int parity, 
		const int dim, const int face_num, 
		const cudaStream_t &stream, 
		const double a, const double b)
  {
    int nDimPack = 0;
    if(dim < 0){
      for (int d=0; d<4; d++) {
	if(!commDim[d]) continue;
	if (d != 3 || getKernelPackT() || a != 0.0 || b!= 0.0) nDimPack++;
      }
    }else{
      if(commDim[dim]){
	if(dim!=3 || getKernelPackT() || a!=0.0 || b != 0.0) nDimPack++;
      }
    }
    if (!nDimPack) return; // if zero then we have nothing to pack 

    if (nFace != 1 && in.Nspin() != 1) 
      errorQuda("Unsupported number of faces %d", nFace);

    // Need to update this logic for other multi-src dslash packing
    if (in.Nspin() == 1) {
      packFaceStaggered(ghost_buf, in, nFace, dagger, parity, dim, face_num, stream);
    } else if (a!=0.0 || b!=0.0) {
      // Need to update this logic for other multi-src dslash packing
      if(in.TwistFlavor() == QUDA_TWIST_PLUS || in.TwistFlavor() == QUDA_TWIST_MINUS) {
	packTwistedFaceWilson(ghost_buf, in, dagger, parity, a, b, stream);
      } else {
	errorQuda("Cannot perform twisted packing for the spinor.");
      }
    } else if (in.Ndim() == 5) {
      if(in.TwistFlavor() == QUDA_TWIST_INVALID) {
	packFaceDW(ghost_buf, in, dagger, parity, stream);
      } else {
	packFaceNdegTM(ghost_buf, in, dagger, parity, stream);
      }
    } else {
      packFaceWilson(ghost_buf, in, dagger, parity, stream);
    }
  }



  void packFaceExtended(void* buffer, cudaColorSpinorField &field, const int nFace, const int R[],
			const int dagger, const int parity, const int dim, const int face_num, 
			const cudaStream_t &stream, const bool unpack)
  {
    int nDimPack = 0;
    if(dim < 0){
      for(int d=0; d<4; d++){
	if(R[d]) nDimPack++;
      }
    }else{
      if(R[dim]) nDimPack++;
    }

    if(!nDimPack) return; // if zero then we have nothing to pack
    if(field.Nspin() == 1){
      packFaceExtendedStaggered(buffer, field, nFace, R, dagger, parity, dim, face_num, stream, unpack);
    }else{
      errorQuda("Extended quark field is not supported");
    }

  }

#endif // MULTI_GPU

}
