#ifdef USE_LEGACY_DSLASH

#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <typeinfo>

#include <color_spinor_field.h>

// these control the Wilson-type actions
#ifdef GPU_WILSON_DIRAC
//#define DIRECT_ACCESS_WILSON_PACK_SPINOR
#endif // GPU_WILSON_DIRAC


#include <quda_internal.h>
#include <dslash_quda.h>
#include <dslash.h>
#include <sys/time.h>
#include <blas_quda.h>

#include <inline_ptx.h>
#include <uint_to_char.h>

#include <index_helper.cuh>

namespace quda
{

#ifdef MULTI_GPU
  static int commDim[QUDA_MAX_DIM]; // Whether to do comms or not
  void setPackComms(const int *comm_dim) {
    for (int i=0; i<4; i++) commDim[i] = comm_dim[i];
    for (int i=4; i<QUDA_MAX_DIM; i++) commDim[i] = 0;
  }
#else
  void setPackComms(const int *comm_dim) { ; }
#endif

#define STRIPED
#ifdef STRIPED
#else
#define SWIZZLE
#endif

  namespace pack
  {

#include <dslash_constants.h>
#include <dslash_textures.h>

  } // end namespace pack

  using namespace pack;

#include <dslash_index.cuh>

  // routines for packing the ghost zones (multi-GPU only)


#ifdef MULTI_GPU

  template <typename FloatN>
  struct PackParam {

    FloatN *out[2*4];
    float *outNorm[2*4];

    FloatN *in;
    float *inNorm;

    int_fastdiv threads; // total number of threads

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

    DslashConstant dc;

    int sp_stride;

    int_fastdiv swizzle;
    int sites_per_block;
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
    output << "X = {" << param.dc.X[0] << ","<< param.dc.X[1] << "," << param.dc.X[2] << "," << param.dc.X[3] << "," << param.dc.X[4] << "}" << std::endl;
    output << "ghostFace = {" << param.dc.ghostFaceCB[0] << "," << param.dc.ghostFaceCB[1] << ","
           << param.dc.ghostFaceCB[2] << "," << param.dc.ghostFaceCB[3] << "}" << std::endl;
    output << "sp_stride = " << param.sp_stride << std::endl;

    output << "swizzle = " << param.swizzle << std::endl;
    output << "sites_per_block = " << param.sites_per_block << std::endl;
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
    if (dagger) {
#include "wilson_pack_face_dagger_core.h"
    } else {
#include "wilson_pack_face_core.h"
    }
  }

  template <int dim, int dagger, int face_num>
    static inline __device__ void unpackFaceWilsonCore(double2 *out, float *outNorm, const double2 *in,
						       const float *inNorm, const int &idx,
						       const int &face_idx, const int &face_volume,
						       PackParam<double2> &param)
  {
    if (dagger) {
#include "wilson_pack_face_dagger_core.h"
    } else {
#include "wilson_pack_face_core.h"
    }
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

  // quarter precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_QUARTER
#define READ_SPINOR_UP READ_SPINOR_QUARTER_UP
#define READ_SPINOR_DOWN READ_SPINOR_QUARTER_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_QUARTER_TEX
#define READ_SPINOR_UP READ_SPINOR_QUARTER_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_QUARTER_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexHalf
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_CHAR4
  template <int dim, int dagger, int face_num>
    static inline __device__ void packFaceWilsonCore(char4 *out, float *outNorm, const char4 *in, const float *inNorm,
                 const int &idx, const int &face_idx, 
                 const int &face_volume, 
                 const PackParam<char4> &param)
  {
    if (dagger) {
#include "wilson_pack_face_dagger_core.h"
    } else {
#include "wilson_pack_face_core.h"
    }
  }

  template <int dim, int dagger, int face_num>
    static inline __device__ void unpackFaceWilsonCore(char4 *out, float *outNorm, const char4 *in, const float *inNorm,
                   const int &idx, const int &face_idx, 
                   const int &face_volume, 
                   const PackParam<char4> &param)
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

#ifdef STRIPED
    const int sites_per_block = param.sites_per_block;
    int local_tid = threadIdx.x;
    int tid = sites_per_block * blockIdx.x + local_tid;
#else
    int tid = block_idx(param.swizzle) * blockDim.x + threadIdx.x;
    constexpr int sites_per_block = 1;
    constexpr int local_tid = 0;
#endif

    while ( local_tid < sites_per_block && tid < param.threads ) {

      // determine which dimension we are packing
      int face_idx;
      const int dim = dimFromFaceIndex(face_idx, tid, param);

      // compute where the output is located
      // compute an index into the local volume from the index into the face
      // read spinor, spin-project, and write half spinor to face
      if (dim == 0) {
        // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
        const int face_num = (face_idx >= nFace * param.dc.ghostFaceCB[0]) ? 1 : 0;
        face_idx -= face_num * nFace * param.dc.ghostFaceCB[0];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,0,nFace,0>(face_idx,param);
          packFaceWilsonCore<0, dagger, 0>(
              param.out[0], param.outNorm[0], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[0], param);
        } else {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,0,nFace,1>(face_idx,param);
          packFaceWilsonCore<0, dagger, 1>(
              param.out[1], param.outNorm[1], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[0], param);
        }
      } else if (dim == 1) {
        // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
        const int face_num = (face_idx >= nFace * param.dc.ghostFaceCB[1]) ? 1 : 0;
        face_idx -= face_num * nFace * param.dc.ghostFaceCB[1];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,1,nFace,0>(face_idx,param);
          packFaceWilsonCore<1, dagger, 0>(
              param.out[2], param.outNorm[2], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[1], param);
        } else {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,1,nFace,1>(face_idx,param);
          packFaceWilsonCore<1, dagger, 1>(
              param.out[3], param.outNorm[3], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[1], param);
        }
      } else if (dim == 2) {
        // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
        const int face_num = (face_idx >= nFace * param.dc.ghostFaceCB[2]) ? 1 : 0;
        face_idx -= face_num * nFace * param.dc.ghostFaceCB[2];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,2,nFace,0>(face_idx,param);
          packFaceWilsonCore<2, dagger, 0>(
              param.out[4], param.outNorm[4], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[2], param);
        } else {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,2,nFace,1>(face_idx,param);
          packFaceWilsonCore<2, dagger, 1>(
              param.out[5], param.outNorm[5], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[2], param);
        }
      } else {
        // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
        const int face_num = (face_idx >= nFace * param.dc.ghostFaceCB[3]) ? 1 : 0;
        face_idx -= face_num * nFace * param.dc.ghostFaceCB[3];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,3,nFace,0>(face_idx,param);
          packFaceWilsonCore<3, dagger, 0>(
              param.out[6], param.outNorm[6], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[3], param);
        } else {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,3,nFace,1>(face_idx,param);
          packFaceWilsonCore<3, dagger, 1>(
              param.out[7], param.outNorm[7], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[3], param);
        }
      }

#ifdef STRIPED
      local_tid += blockDim.x;
      tid += blockDim.x;
#else
      tid += blockDim.x*gridDim.x;
#endif
    } // while tid

  }


  template <int dagger, typename FloatN, int nFace>
    __global__ void packFaceExtendedWilsonKernel(PackParam<FloatN> param)
  {

#ifdef STRIPED
    const int sites_per_block = param.sites_per_block;
    int local_tid = threadIdx.x;
    int tid = sites_per_block * blockIdx.x + local_tid;
#else
    int tid = block_idx(param.swizzle) * blockDim.x + threadIdx.x;
    constexpr int sites_per_block = 1;
    constexpr int local_tid = 0;
#endif

    while ( local_tid < sites_per_block && tid < param.threads ) {

      // determine which dimension we are packing
      int face_idx;
      const int dim = dimFromFaceIndex(face_idx, tid, param);

      // compute where the output is located
      // compute an index into the local volume from the index into the face
      // read spinor, spin-project, and write half spinor to face
      if (dim == 0) {
        // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
        // if param.face_num==2 pack both the start and the end, otherwise pack the region of the lattice
        // specified by param.face_num
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[0]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[0];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtended<0,nFace,0>(face_idx,param);
          packFaceWilsonCore<0, dagger, 0>(
              param.out[0], param.outNorm[0], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[0], param);
        } else {
          const int idx = indexFromFaceIndexExtended<0,nFace,1>(face_idx,param);
          packFaceWilsonCore<0, dagger, 1>(
              param.out[1], param.outNorm[1], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[0], param);
        }
      } else if (dim == 1) {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[1]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[1];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtended<1,nFace,0>(face_idx,param);
          packFaceWilsonCore<1, dagger, 0>(
              param.out[2], param.outNorm[2], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[1], param);
        } else {
          const int idx = indexFromFaceIndexExtended<1,nFace,1>(face_idx,param);
          packFaceWilsonCore<1, dagger, 1>(
              param.out[3], param.outNorm[3], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[1], param);
        }
      } else if (dim == 2) {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[2]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[2];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtended<2,nFace,0>(face_idx,param);
          packFaceWilsonCore<2, dagger, 0>(
              param.out[4], param.outNorm[4], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[2], param);
        } else {
          const int idx = indexFromFaceIndexExtended<2,nFace,1>(face_idx,param);
          packFaceWilsonCore<2, dagger, 1>(
              param.out[5], param.outNorm[5], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[2], param);
        }
      } else {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[3]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[3];

        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtended<3,nFace,0>(face_idx,param);
          packFaceWilsonCore<3, dagger, 0>(
              param.out[6], param.outNorm[6], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[3], param);
        } else {
          const int idx = indexFromFaceIndexExtended<3,nFace,1>(face_idx,param);
          packFaceWilsonCore<3, dagger, 1>(
              param.out[7], param.outNorm[7], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[3], param);
        }
      }

#ifdef STRIPED
      local_tid += blockDim.x;
      tid += blockDim.x;
#else
      tid += blockDim.x*gridDim.x;
#endif
    } // while tid

  }


  template <int dagger, typename FloatN, int nFace>
    __global__ void unpackFaceExtendedWilsonKernel(PackParam<FloatN> param)
  {

#ifdef STRIPED
    const int sites_per_block = param.sites_per_block;
    int local_tid = threadIdx.x;
    int tid = sites_per_block * blockIdx.x + local_tid;
#else
    int tid = block_idx(param.swizzle) * blockDim.x + threadIdx.x;
    constexpr int sites_per_block = 1;
    constexpr int local_tid = 0;
#endif

    while ( local_tid < sites_per_block && tid < param.threads ) {

      // determine which dimension we are packing
      int face_idx;
      const int dim = dimFromFaceIndex(face_idx, tid, param);

      // compute where the output is located
      // compute an index into the local volume from the index into the face
      // read spinor, spin-project, and write half spinor to face
      if (dim == 0) {
        // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
        // if param.face_num==2 pack both the start and the end, otherwise pack the region of the lattice
        // specified by param.face_num
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[0]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[0];

        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtended<0,nFace,0>(face_idx,param);
          unpackFaceWilsonCore<0, dagger, 0>(
              param.out[0], param.outNorm[0], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[0], param);
        } else {
          const int idx = indexFromFaceIndexExtended<0,nFace,1>(face_idx,param);
          unpackFaceWilsonCore<0, dagger, 1>(
              param.out[1], param.outNorm[1], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[0], param);
        }
      } else if (dim == 1) {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[1]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[1];

        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtended<1,nFace,0>(face_idx,param);
          unpackFaceWilsonCore<1, dagger, 0>(
              param.out[2], param.outNorm[2], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[1], param);
        } else {
          const int idx = indexFromFaceIndexExtended<1,nFace,1>(face_idx,param);
          unpackFaceWilsonCore<1, dagger, 1>(
              param.out[3], param.outNorm[3], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[1], param);
        }
      } else if (dim == 2) {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[2]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[2];

        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtended<2,nFace,0>(face_idx,param);
          unpackFaceWilsonCore<2, dagger, 0>(
              param.out[4], param.outNorm[4], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[2], param);
        } else {
          const int idx = indexFromFaceIndexExtended<2,nFace,1>(face_idx,param);
          unpackFaceWilsonCore<2, dagger, 1>(
              param.out[5], param.outNorm[5], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[2], param);
        }
      } else {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[3]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[3];

        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtended<3,nFace,0>(face_idx,param);
          unpackFaceWilsonCore<3, dagger, 0>(
              param.out[6], param.outNorm[6], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[3], param);
        } else {
          const int idx = indexFromFaceIndexExtended<3,nFace,1>(face_idx,param);
          unpackFaceWilsonCore<3, dagger, 1>(
              param.out[7], param.outNorm[7], param.in, param.inNorm, idx, face_idx, param.dc.ghostFaceCB[3], param);
        }
      }

#ifdef STRIPED
      local_tid += blockDim.x;
      tid += blockDim.x;
#else
      tid += blockDim.x*gridDim.x;
#endif
    } // while tid

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

  // quarter precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_QUARTER
#define READ_SPINOR_UP READ_SPINOR_QUARTER_UP
#define READ_SPINOR_DOWN READ_SPINOR_QUARTER_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_QUARTER_TEX
#define READ_SPINOR_UP READ_SPINOR_QUARTER_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_QUARTER_DOWN_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexHalf
#endif
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_CHAR4
  template <int dim, int dagger, int face_num>
    static inline __device__ void packTwistedFaceWilsonCore(char4 *out, float *outNorm, const char4 *in, const float *inNorm, float a, float b,
                  const int &idx, const int &face_idx, 
                  const int &face_volume, 
                  const PackParam<char4> &param)
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

#ifdef STRIPED
    const int sites_per_block = param.sites_per_block;
    int local_tid = threadIdx.x;
    int tid = sites_per_block * blockIdx.x + local_tid;
#else
    int tid = block_idx(param.swizzle) * blockDim.x + threadIdx.x;
    constexpr int sites_per_block = 1;
    constexpr int local_tid = 0;
#endif

    while ( local_tid < sites_per_block && tid < param.threads ) {

      // determine which dimension we are packing
      int face_idx;
      const int dim = dimFromFaceIndex(face_idx, tid, param);

      // compute where the output is located
      // compute an index into the local volume from the index into the face
      // read spinor, spin-project, and write half spinor to face
      if (dim == 0) {
        // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
        const int face_num = (face_idx >= nFace * param.dc.ghostFaceCB[0]) ? 1 : 0;
        face_idx -= face_num * nFace * param.dc.ghostFaceCB[0];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,0,nFace,0>(face_idx,param);
          packTwistedFaceWilsonCore<0, dagger, 0>(param.out[0], param.outNorm[0], param.in, param.inNorm, a, b, idx,
              face_idx, param.dc.ghostFaceCB[0], param);
        } else {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,0,nFace,1>(face_idx,param);
          packTwistedFaceWilsonCore<0, dagger, 1>(param.out[1], param.outNorm[1], param.in, param.inNorm, a, b, idx,
              face_idx, param.dc.ghostFaceCB[0], param);
        }
      } else if (dim == 1) {
        const int face_num = (face_idx >= nFace * param.dc.ghostFaceCB[1]) ? 1 : 0;
        face_idx -= face_num * nFace * param.dc.ghostFaceCB[1];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,1,nFace,0>(face_idx,param);
          packTwistedFaceWilsonCore<1, dagger, 0>(param.out[2], param.outNorm[2], param.in, param.inNorm, a, b, idx,
              face_idx, param.dc.ghostFaceCB[1], param);
        } else {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,1,nFace,1>(face_idx,param);
          packTwistedFaceWilsonCore<1, dagger, 1>(param.out[3], param.outNorm[3], param.in, param.inNorm, a, b, idx,
              face_idx, param.dc.ghostFaceCB[1], param);
        }
      } else if (dim == 2) {
        const int face_num = (face_idx >= nFace * param.dc.ghostFaceCB[2]) ? 1 : 0;
        face_idx -= face_num * nFace * param.dc.ghostFaceCB[2];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,2,nFace,0>(face_idx,param);
          packTwistedFaceWilsonCore<2, dagger, 0>(param.out[4], param.outNorm[4], param.in, param.inNorm, a, b, idx,
              face_idx, param.dc.ghostFaceCB[2], param);
        } else {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,2,nFace,1>(face_idx,param);
          packTwistedFaceWilsonCore<2, dagger, 1>(param.out[5], param.outNorm[5], param.in, param.inNorm, a, b, idx,
              face_idx, param.dc.ghostFaceCB[2], param);
        }
      } else {
        const int face_num = (face_idx >= nFace * param.dc.ghostFaceCB[3]) ? 1 : 0;
        face_idx -= face_num * nFace * param.dc.ghostFaceCB[3];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,3,nFace,0>(face_idx,param);
          packTwistedFaceWilsonCore<3, dagger, 0>(param.out[6], param.outNorm[6], param.in, param.inNorm, a, b, idx,
              face_idx, param.dc.ghostFaceCB[3], param);
        } else {
          const int idx = indexFromFaceIndex<4,QUDA_4D_PC,3,nFace,1>(face_idx,param);
          packTwistedFaceWilsonCore<3, dagger, 1>(param.out[7], param.outNorm[7], param.in, param.inNorm, a, b, idx,
              face_idx, param.dc.ghostFaceCB[3], param);
        }
      }

#ifdef STRIPED
      local_tid += blockDim.x;
      tid += blockDim.x;
#else
      tid += blockDim.x*gridDim.x;
#endif
    } // while tid

  }

#endif // GPU_TWISTED_MASS_DIRAC

  template <typename FloatN, typename Float>
    class PackFace : public Tunable {

  protected:
    void *faces[2*QUDA_MAX_DIM];
    const cudaColorSpinorField *in;
    const int dagger;
    const int parity;
    const int nFace;
    const int dim;
    const int face_num;
    const MemoryLocation location;

    // compute how many threads we need in total for the face packing
    unsigned int threads() const {
      unsigned int threads = 0;
      if(dim < 0){ // if dim is negative, pack all dimensions
        for (int i=0; i<4; i++) {
          if (!commDim[i]) continue;
          if ( i==3 && !getKernelPackT() ) continue;
          threads += 2*nFace*in->GhostFace()[i]; // 2 for forwards and backwards faces
        }
      }else{ // pack only in dim dimension
        if( commDim[dim] && (dim!=3 || getKernelPackT() )){
          threads = nFace*in->GhostFace()[dim];
          if(face_num==2) threads *= 2; // sending data forwards and backwards
        }
      }
      return threads;
    }

    virtual int inputPerSite() const = 0;
    virtual int outputPerSite() const = 0;

    void prepareParam(PackParam<FloatN> &param, TuneParam &tp, int dim=-1, int face_num=2) {
      param.in = (FloatN*)in->V();
      param.inNorm = (float*)in->Norm();
      param.dim = dim;
      param.face_num = face_num;
      param.parity = parity;

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

	param.out[2*i+0] = static_cast<FloatN*>(faces[2*i+0]);
	param.out[2*i+1] = static_cast<FloatN*>(faces[2*i+1]);

	param.outNorm[2*i+0] = reinterpret_cast<float*>(static_cast<char*>(faces[2*i+0]) + nFace*outputPerSite()*in->GhostFace()[i]*in->Precision());
	param.outNorm[2*i+1] = reinterpret_cast<float*>(static_cast<char*>(faces[2*i+1]) + nFace*outputPerSite()*in->GhostFace()[i]*in->Precision());

        prev=i;
      }

      param.dc = in->getDslashConstant(); // get pre-computed constants

      param.swizzle = tp.aux.x;
      param.sites_per_block = (param.threads + tp.grid.x - 1) / tp.grid.x;
    }

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

#ifdef STRIPED
    bool tuneGridDim() const { return true; } // If striping, always tune grid dimension
    unsigned int maxGridSize() const {
      if (location & Host) {
	// if zero-copy policy then set a maximum number of blocks to be
	// the 3 * number of dimensions we are communicating
        int nDimComms = 0;
        for (int d=0; d<in->Ndim(); d++) nDimComms += commDim[d];
        return 3*nDimComms;
      } else {
        return Tunable::maxGridSize();
      }
    } // use no more than a quarter of the GPU
    unsigned int minGridSize() const {
      if (location & Host) {
	// if zero-copy policy then set a maximum number of blocks to be
	// the 1 * number of dimensions we are communicating
        int nDimComms = 0;
        for (int d=0; d<in->Ndim(); d++) nDimComms += commDim[d];
        return nDimComms;
      } else {
        return Tunable::minGridSize();
      }
    }
#else
    bool tuneGridDim() const { return location & Host; } // only tune grid dimension if doing zero-copy writing
    unsigned int maxGridSize() const
    {
      return tuneGridDim() ? deviceProp.multiProcessorCount / 4 : Tunable::maxGridSize();
    } // use no more than a quarter of the GPU
#endif

    bool tuneAuxDim() const { return true; } // Do tune the aux dimensions.
    unsigned int minThreads() const { return threads(); }

    void fillAux() {
      strcpy(aux,"policy_kernel,");
      strcat(aux, in->AuxString());
      char comm[5];
      comm[0] = (commDim[0] ? '1' : '0');
      comm[1] = (commDim[1] ? '1' : '0');
      comm[2] = (commDim[2] ? '1' : '0');
      comm[3] = (commDim[3] ? '1' : '0');
      comm[4] = '\0'; strcat(aux,",comm=");
      strcat(aux,comm);
      strcat(aux,comm_dim_topology_string());
      if (getKernelPackT()) { strcat(aux,",kernelPackT"); }
      switch (nFace) {
      case 1: strcat(aux,",nFace=1,"); break;
      case 3: strcat(aux,",nFace=3,"); break;
      default: errorQuda("Number of faces not supported");
      }

      // label the locations we are packing to
      // location lable is nonp2p-p2p
      switch ((int)location) {
      case Device|Remote: strcat(aux,"device-remote"); break;
      case   Host|Remote: strcat(aux,  "host-remote"); break;
      case        Device: strcat(aux,"device-device"); break;
      case          Host: strcat(aux, comm_peer2peer_enabled_global() ? "host-device" : "host-host"); break;
      default: errorQuda("Unknown pack target location %d\n", location);
      }

    }

  public:
    PackFace(void *faces_[], const cudaColorSpinorField *in, MemoryLocation location,
	     const int dagger, const int parity, const int nFace, const int dim=-1, const int face_num=2)
      : in(in), dagger(dagger),
	parity(parity), nFace(nFace), dim(dim), face_num(face_num), location(location)
    {
      memcpy(faces, faces_, 2*QUDA_MAX_DIM*sizeof(void*));
      fillAux();
#ifndef USE_TEXTURE_OBJECTS
      bindSpinorTex<FloatN>(in);
#endif
    }

    virtual ~PackFace() {
#ifndef USE_TEXTURE_OBJECTS
      unbindSpinorTex<FloatN>(in);
#endif
    }

    bool tuneSharedBytes() const { return location & Host ? false : Tunable::tuneSharedBytes(); }

    bool advanceAux(TuneParam &param) const
    {
#ifdef SWIZZLE
      if ( location & Remote ) {  // only swizzling if we're doing remote writing
        if (param.aux.x < (int)maxGridSize()) {
          param.aux.x++;
          return true;
        } else {
          param.aux.x = 1;
          return false;
        }
      } else {
        return false;
      }
#else
      return false;
#endif
    }

    void initTuneParam(TuneParam &param) const {
      Tunable::initTuneParam(param);
      param.aux.x = 1; // swizzle factor
      // if doing a zero-copy policy then ensure that each thread block
      // runs exclusively on a given SM - this is to ensure quality of
      // service for the packing kernel when running concurrently.
      if (location & Host) param.shared_bytes = deviceProp.sharedMemPerBlock / 2 + 1;
    }

    void defaultTuneParam(TuneParam &param) const {
      Tunable::defaultTuneParam(param);
      param.aux.x = 1; // swizzle factor
    }

    long long flops() const { return outputPerSite()*this->threads(); }

    virtual int tuningIter() const { return 3; }

    virtual TuneKey tuneKey() const { return TuneKey(in->VolString(), typeid(*this).name(), aux); }

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
    PackFaceWilson(void *faces[], const cudaColorSpinorField *in, MemoryLocation location,
		   const int dagger, const int parity)
      : PackFace<FloatN, Float>(faces, in, location, dagger, parity, 1) { }
    virtual ~PackFaceWilson() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_WILSON_DIRAC
      static PackParam<FloatN> param;
      this->prepareParam(param,tp);

      void *args[] = { &param };
      void (*func)(PackParam<FloatN>) = this->dagger ? &(packFaceWilsonKernel<1,FloatN>) : &(packFaceWilsonKernel<0,FloatN>);
      qudaLaunchKernel( (const void*)func, tp.grid, tp.block, args, tp.shared_bytes, stream);
#else
      errorQuda("Wilson face packing kernel is not built");
#endif
    }

  };

  void packFaceWilson(void *ghost_buf[], cudaColorSpinorField &in, MemoryLocation location,
		      const int dagger, const int parity, const cudaStream_t &stream) {

    switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceWilson<double2, double> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceWilson<float4, float> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceWilson<short4, float> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_QUARTER_PRECISION:
      {
        PackFaceWilson<char4, float> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      }
      break;
    default:
      errorQuda("Precision %d not supported", in.Precision());
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
    PackFaceTwisted(void *faces[], const cudaColorSpinorField *in, MemoryLocation location,
		    const int dagger, const int parity, Float a, Float b)
      : PackFace<FloatN, Float>(faces, in, location, dagger, parity, 1), a(a), b(b) { }
    virtual ~PackFaceTwisted() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_TWISTED_MASS_DIRAC
      static PackParam<FloatN> param;
      this->prepareParam(param,tp);
      void *args[] = { &a, &b, &param };
      void (*func)(Float,Float,PackParam<FloatN>) = this->dagger ? &(packTwistedFaceWilsonKernel<1,FloatN,Float>) : &(packTwistedFaceWilsonKernel<0,FloatN,Float>);
      cudaLaunchKernel( (const void*)func, tp.grid, tp.block, args, tp.shared_bytes, stream);
#else
      errorQuda("Twisted face packing kernel is not built");
#endif
    }

    long long flops() const { return outputPerSite()*this->threads(); }
  };

  //!
  void packTwistedFaceWilson(void *ghost_buf[], cudaColorSpinorField &in, MemoryLocation location, const int dagger,
			     const int parity, const double a, const double b, const cudaStream_t &stream) {

    switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceTwisted<double2, double> pack(ghost_buf, &in, location, dagger, parity, a, b);
        pack.apply(stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceTwisted<float4, float> pack(ghost_buf, &in, location, dagger, parity, (float)a, (float)b);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceTwisted<short4, float> pack(ghost_buf, &in, location, dagger, parity, (float)a, (float)b);
        pack.apply(stream);
      }
      break;
    case QUDA_QUARTER_PRECISION:
      {
        PackFaceTwisted<char4, float> pack(ghost_buf, &in, location, dagger, parity, (float)a, (float)b);
        pack.apply(stream);
      }
      break;
    default:
      errorQuda("Precision %d not supported", in.Precision());
    }
  }

#if (defined GPU_STAGGERED_DIRAC)

#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEXDOUBLE param.inTex
#define SPINORTEXSINGLE param.inTex
#define SPINORTEXHALF param.inTex
#define SPINORTEXHALFNORM param.inTexNorm
#define SPINORTEXQUARTER param.inTex
#define SPINORTEXQUARTERNORM param.inTexNorm
#else
#define SPINORTEXDOUBLE spinorTexDouble
#define SPINORTEXSINGLE spinorTexSingle2
#define SPINORTEXHALF spinorTexHalf2
#define SPINORTEXHALFNORM spinorTexHalf2Norm
#define SPINORTEXQUARTER spinorTexQuarter2
#define SPINORTEXQUARTERNORM spinorTexQuarter2Norm
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
  template<> 
    __device__ void packFaceStaggeredCore(char2 *out, float *outNorm, const int out_idx, 
            const int out_stride, const char2 *in, const float *inNorm, 
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
  template<> 
    __device__ void packFaceStaggeredCore(char2 *out, float *outNorm, const int out_idx, 
            const int out_stride, const char2 *in, const float *inNorm, 
            const int in_idx, const PackParam<double2> &param) {
    out[out_idx + 0*out_stride] = in[in_idx + 0*param.sp_stride];
    out[out_idx + 1*out_stride] = in[in_idx + 1*param.sp_stride];
    out[out_idx + 2*out_stride] = in[in_idx + 2*param.sp_stride];
    outNorm[out_idx] = inNorm[in_idx];
  }


#else
  __device__ void packFaceStaggeredCore(double2 *out, float *outNorm, const int out_idx,
					const int out_stride, const double2 *in, const float *inNorm,
					const int in_idx, const PackParam<double2> &param) {
    out[out_idx + 0*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 0*param.sp_stride);
    out[out_idx + 1*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 1*param.sp_stride);
    out[out_idx + 2*out_stride] = fetch_double2(SPINORTEXDOUBLE, in_idx + 2*param.sp_stride);
  }

  __device__ void packFaceStaggeredCore(float2 *out, float *outNorm, const int out_idx,
					const int out_stride, const float2 *in,
					const float *inNorm, const int in_idx,
					const PackParam<float2> &param) {
    out[out_idx + 0 * out_stride] = tex1Dfetch<float2>(SPINORTEXSINGLE, in_idx + 0 * param.sp_stride);
    out[out_idx + 1 * out_stride] = tex1Dfetch<float2>(SPINORTEXSINGLE, in_idx + 1 * param.sp_stride);
    out[out_idx + 2 * out_stride] = tex1Dfetch<float2>(SPINORTEXSINGLE, in_idx + 2 * param.sp_stride);
  }

  // this is rather dumb: undoing the texture load because cudaNormalizedReadMode is used
  // should really bind to an appropriate texture instead of reusing
  inline __device__ short2 float22short2(float c, float2 a) {
    return make_short2((short)(a.x*(c*fixedMaxValue<short>::value)), (short)(a.y*(c*fixedMaxValue<short>::value)));
  }
  
  inline __device__ char2 float22char2(float c, float2 a) {
    return make_char2((char)(a.x*(c*fixedMaxValue<char>::value)), (char)(a.y*(c*fixedMaxValue<char>::value)));
  }

  __device__ void packFaceStaggeredCore(short2 *out, float *outNorm, const int out_idx,
					const int out_stride, const short2 *in,
					const float *inNorm, const int in_idx,
					const PackParam<short2> &param) {
    out[out_idx + 0 * out_stride] = float22short2(1.0f, tex1Dfetch<float2>(SPINORTEXHALF, in_idx + 0 * param.sp_stride));
    out[out_idx + 1 * out_stride] = float22short2(1.0f, tex1Dfetch<float2>(SPINORTEXHALF, in_idx + 1 * param.sp_stride));
    out[out_idx + 2 * out_stride] = float22short2(1.0f, tex1Dfetch<float2>(SPINORTEXHALF, in_idx + 2 * param.sp_stride));
    outNorm[out_idx] = tex1Dfetch<float>(SPINORTEXHALFNORM, in_idx);
  }

  __device__ void packFaceStaggeredCore(char2 *out, float *outNorm, const int out_idx, 
          const int out_stride, const char2 *in, 
          const float *inNorm, const int in_idx, 
          const PackParam<char2> &param) {
    out[out_idx + 0 * out_stride]
        = float22char2(1.0f, tex1Dfetch<float2>(SPINORTEXQUARTER, in_idx + 0 * param.sp_stride));
    out[out_idx + 1 * out_stride]
        = float22char2(1.0f, tex1Dfetch<float2>(SPINORTEXQUARTER, in_idx + 1 * param.sp_stride));
    out[out_idx + 2 * out_stride]
        = float22char2(1.0f, tex1Dfetch<float2>(SPINORTEXQUARTER, in_idx + 2 * param.sp_stride));
    outNorm[out_idx] = tex1Dfetch<float>(SPINORTEXQUARTERNORM, in_idx);
  }
#endif


  template <typename FloatN, int nFace>
    __global__ void packFaceStaggeredKernel(PackParam<FloatN> param)
  {

#ifdef STRIPED
    const int sites_per_block = param.sites_per_block;
    int local_tid = threadIdx.x;
    int tid = sites_per_block * blockIdx.x + local_tid;
#else
    int tid = block_idx(param.swizzle) * blockDim.x + threadIdx.x;
    constexpr int sites_per_block = 1;
    constexpr int local_tid = 0;
#endif

    while ( local_tid < sites_per_block && tid < param.threads ) {

      const int Ls = param.dc.X[4];

      // determine which dimension we are packing
      int face_idx;
      const int dim = dimFromFaceIndex(face_idx, tid, param);

      // compute where the output is located
      // compute an index into the local volume from the index into the face
      // read spinor and write to face
      if (dim == 0) {
        // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= Ls * nFace * param.dc.ghostFaceCB[0]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * Ls * nFace * param.dc.ghostFaceCB[0];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexStaggered<0,nFace,0>(face_idx,param);
          packFaceStaggeredCore(param.out[0], param.outNorm[0], face_idx, Ls * nFace * param.dc.ghostFaceCB[0],
              param.in, param.inNorm, idx, param);
        } else {
          const int idx = indexFromFaceIndexStaggered<0,nFace,1>(face_idx,param);
          packFaceStaggeredCore(param.out[1], param.outNorm[1], face_idx, Ls * nFace * param.dc.ghostFaceCB[0],
              param.in, param.inNorm, idx, param);
        }
      } else if (dim == 1) {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= Ls * nFace * param.dc.ghostFaceCB[1]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * Ls * nFace * param.dc.ghostFaceCB[1];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexStaggered<1,nFace,0>(face_idx,param);
          packFaceStaggeredCore(param.out[2], param.outNorm[2], face_idx, Ls * nFace * param.dc.ghostFaceCB[1],
              param.in, param.inNorm, idx, param);
        } else {
          const int idx = indexFromFaceIndexStaggered<1,nFace,1>(face_idx,param);
          packFaceStaggeredCore(param.out[3], param.outNorm[3], face_idx, Ls * nFace * param.dc.ghostFaceCB[1],
              param.in, param.inNorm, idx, param);
        }
      } else if (dim == 2) {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= Ls * nFace * param.dc.ghostFaceCB[2]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * Ls * nFace * param.dc.ghostFaceCB[2];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexStaggered<2,nFace,0>(face_idx,param);
          packFaceStaggeredCore(param.out[4], param.outNorm[4], face_idx, Ls * nFace * param.dc.ghostFaceCB[2],
              param.in, param.inNorm, idx, param);
        } else {
          const int idx = indexFromFaceIndexStaggered<2,nFace,1>(face_idx,param);
          packFaceStaggeredCore(param.out[5], param.outNorm[5], face_idx, Ls * nFace * param.dc.ghostFaceCB[2],
              param.in, param.inNorm, idx, param);
        }
      } else {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= Ls * nFace * param.dc.ghostFaceCB[3]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * Ls * nFace * param.dc.ghostFaceCB[3];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexStaggered<3,nFace,0>(face_idx,param);
          packFaceStaggeredCore(param.out[6], param.outNorm[6], face_idx, Ls * nFace * param.dc.ghostFaceCB[3],
              param.in, param.inNorm, idx, param);
        } else {
          const int idx = indexFromFaceIndexStaggered<3,nFace,1>(face_idx,param);
          packFaceStaggeredCore(param.out[7], param.outNorm[7], face_idx, Ls * nFace * param.dc.ghostFaceCB[3],
              param.in, param.inNorm, idx, param);
        }
      }

#ifdef STRIPED
      local_tid += blockDim.x;
      tid += blockDim.x;
#else
      tid += blockDim.x*gridDim.x;
#endif
    } // while tid

  }


  template <typename FloatN, int nFace>
    __global__ void packFaceExtendedStaggeredKernel(PackExtendedParam<FloatN> param)
  {

#ifdef STRIPED
    const int sites_per_block = param.sites_per_block;
    int local_tid = threadIdx.x;
    int tid = sites_per_block * blockIdx.x + local_tid;
#else
    int tid = block_idx(param.swizzle) * blockDim.x + threadIdx.x;
    constexpr int sites_per_block = 1;
    constexpr int local_tid = 0;
#endif

    while ( local_tid < sites_per_block && tid < param.threads ) {

      // determine which dimension we are packing
      int face_idx;
      const int dim = dimFromFaceIndex(face_idx, tid, param);

      // compute where the output is located
      // compute an index into the local volume from the index into the face
      // read spinor and write half spinor to face
      if (dim == 0) {
        // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
        // if param.face_num==2 pack both the start and the end, otherwise pack the region of the
        // lattice specified by param.face_num
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[0]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[0];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtendedStaggered<0,nFace,0>(face_idx,param);
          packFaceStaggeredCore(param.out[0], param.outNorm[0], face_idx, nFace * param.dc.ghostFaceCB[0], param.in,
              param.inNorm, idx, param);
        } else {
          const int idx = indexFromFaceIndexExtendedStaggered<0,nFace,1>(face_idx,param);
          packFaceStaggeredCore(param.out[1], param.outNorm[1], face_idx, nFace * param.dc.ghostFaceCB[0], param.in,
              param.inNorm, idx, param);
        }
      } else if (dim == 1) {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[1]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[1];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtendedStaggered<1,nFace,0>(face_idx,param);
          packFaceStaggeredCore(param.out[2], param.outNorm[2], face_idx, nFace * param.dc.ghostFaceCB[1], param.in,
              param.inNorm, idx, param);
        } else {
          const int idx = indexFromFaceIndexExtendedStaggered<1,nFace,1>(face_idx,param);
          packFaceStaggeredCore(param.out[3], param.outNorm[3], face_idx, nFace * param.dc.ghostFaceCB[1], param.in,
              param.inNorm, idx, param);
        }
      } else if (dim == 2) {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[2]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[2];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtendedStaggered<2,nFace,0>(face_idx,param);
          packFaceStaggeredCore(param.out[4], param.outNorm[4], face_idx, nFace * param.dc.ghostFaceCB[2], param.in,
              param.inNorm, idx, param);
        } else {
          const int idx = indexFromFaceIndexExtendedStaggered<2,nFace,1>(face_idx,param);
          packFaceStaggeredCore(param.out[5], param.outNorm[5], face_idx, nFace * param.dc.ghostFaceCB[2], param.in,
              param.inNorm, idx, param);
        }
      } else {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[3]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[3];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtendedStaggered<3,nFace,0>(face_idx,param);
          packFaceStaggeredCore(param.out[6], param.outNorm[6], face_idx, nFace * param.dc.ghostFaceCB[3], param.in,
              param.inNorm, idx, param);
        } else {
          const int idx = indexFromFaceIndexExtendedStaggered<3,nFace,1>(face_idx,param);
          packFaceStaggeredCore(param.out[7], param.outNorm[7], face_idx, nFace * param.dc.ghostFaceCB[3], param.in,
              param.inNorm, idx, param);
        }
      }

#ifdef STRIPED
      local_tid += blockDim.x;
      tid += blockDim.x;
#else
      tid += blockDim.x*gridDim.x;
#endif
    } // while tid

  }


  template <typename FloatN, int nFace>
    __global__ void unpackFaceExtendedStaggeredKernel(PackExtendedParam<FloatN> param)
  {

#ifdef STRIPED
    const int sites_per_block = param.sites_per_block;
    int local_tid = threadIdx.x;
    int tid = sites_per_block * blockIdx.x + local_tid;
#else
    int tid = block_idx(param.swizzle) * blockDim.x + threadIdx.x;
    constexpr int sites_per_block = 1;
    constexpr int local_tid = 0;
#endif

    while ( local_tid < sites_per_block && tid < param.threads ) {

      // determine which dimension we are packing
      int face_idx;
      const int dim = dimFromFaceIndex(face_idx, tid, param);

      // compute where the output is located
      // compute an index into the local volume from the index into the face
      // read spinor, spin-project, and write half spinor to face
      if (dim == 0) {
        // face_num determines which end of the lattice we are packing: 0 = start, 1 = end
        // if param.face_num==2 pack both the start and the end, otherwist pack the region of the
        // lattice specified by param.face_num
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[0]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[0];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtendedStaggered<0,nFace,0>(face_idx,param);
          packFaceStaggeredCore(param.in, param.inNorm, idx, param.sp_stride, param.out[0], param.outNorm[0], face_idx,
              nFace * param.dc.ghostFaceCB[0]);
        } else {
          const int idx = indexFromFaceIndexExtendedStaggered<0,nFace,1>(face_idx,param);
          packFaceStaggeredCore(param.in, param.inNorm, idx, param.sp_stride, param.out[1], param.outNorm[1], face_idx,
              nFace * param.dc.ghostFaceCB[0]);
        }
      } else if (dim == 1) {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[1]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[1];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtendedStaggered<1,nFace,0>(face_idx,param);
          packFaceStaggeredCore(param.in, param.inNorm, idx, param.sp_stride, param.out[2], param.outNorm[2], face_idx,
              nFace * param.dc.ghostFaceCB[1]);
        } else {
          const int idx = indexFromFaceIndexExtendedStaggered<1,nFace,1>(face_idx,param);
          packFaceStaggeredCore(param.in, param.inNorm, idx, param.sp_stride, param.out[3], param.outNorm[3], face_idx,
              nFace * param.dc.ghostFaceCB[1]);
        }
      } else if (dim == 2) {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[2]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[2];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtendedStaggered<2,nFace,0>(face_idx,param);
          packFaceStaggeredCore(param.in, param.inNorm, idx, param.sp_stride, param.out[4], param.outNorm[4], face_idx,
              nFace * param.dc.ghostFaceCB[2]);
        } else {
          const int idx = indexFromFaceIndexExtendedStaggered<2,nFace,1>(face_idx,param);
          packFaceStaggeredCore(param.in, param.inNorm, idx, param.sp_stride, param.out[5], param.outNorm[5], face_idx,
              nFace * param.dc.ghostFaceCB[2]);
        }
      } else {
        const int face_num
            = (param.face_num == 2) ? ((face_idx >= nFace * param.dc.ghostFaceCB[3]) ? 1 : 0) : param.face_num;
        if (param.face_num == 2) face_idx -= face_num * nFace * param.dc.ghostFaceCB[3];
        if (face_num == 0) {
          const int idx = indexFromFaceIndexExtendedStaggered<3,nFace,0>(face_idx,param);
          packFaceStaggeredCore(param.in, param.inNorm, idx, param.sp_stride, param.out[6], param.outNorm[6], face_idx,
              nFace * param.dc.ghostFaceCB[3]);
        } else {
          const int idx = indexFromFaceIndexExtendedStaggered<3,nFace,1>(face_idx,param);
          packFaceStaggeredCore(param.in, param.inNorm, idx, param.sp_stride, param.out[7], param.outNorm[7], face_idx,
              nFace * param.dc.ghostFaceCB[3]);
        }
      }

#ifdef STRIPED
      local_tid += blockDim.x;
      tid += blockDim.x;
#else
      tid += blockDim.x*gridDim.x;
#endif
    } // while tid

  }


#undef SPINORTEXDOUBLE
#undef SPINORTEXSINGLE
#undef SPINORTEXHALF
#undef SPINORTEXQUARTER

#endif // GPU_STAGGERED_DIRAC


  template <typename FloatN, typename Float>
    class PackFaceStaggered : public PackFace<FloatN, Float> {

  private:
    const int* R; // boundary dimensions for extended field
    const bool unpack;

    int inputPerSite() const { return 6; } // input is full spinor
    int outputPerSite() const { return 6; } // output is full spinor


  public:
    PackFaceStaggered(void *faces[], const cudaColorSpinorField *in, MemoryLocation location,
		      const int nFace, const int dagger, const int parity,
		      const int dim, const int face_num, const int* R=NULL, const bool unpack=false)
      : PackFace<FloatN, Float>(faces, in, location, dagger, parity, nFace, dim, face_num), R(R), unpack(unpack) { }
    virtual ~PackFaceStaggered() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#if (defined GPU_STAGGERED_DIRAC)

      static PackParam<FloatN> param;
      this->prepareParam(param,tp,this->dim,this->face_num);
      if(!R){
        void *args[] = { &param };
        void (*func)(PackParam<FloatN>) = PackFace<FloatN,Float>::nFace==1 ? &(packFaceStaggeredKernel<FloatN,1>) : &(packFaceStaggeredKernel<FloatN,3>);
        cudaLaunchKernel( (const void*)func, tp.grid, tp.block, args, tp.shared_bytes, stream);
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


  void packFaceStaggered(void *ghost_buf[], cudaColorSpinorField &in, MemoryLocation location, int nFace,
			 int dagger, int parity, const int dim, const int face_num, const cudaStream_t &stream) {

    switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceStaggered<double2, double> pack(ghost_buf, &in, location, nFace, dagger, parity, dim, face_num);
        pack.apply(stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceStaggered<float2, float> pack(ghost_buf, &in, location, nFace, dagger, parity, dim, face_num);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceStaggered<short2, float> pack(ghost_buf, &in, location, nFace, dagger, parity, dim, face_num);
        pack.apply(stream);
      }
      break;
    case QUDA_QUARTER_PRECISION:
      {
        PackFaceStaggered<char2, float> pack(ghost_buf, &in, location, nFace, dagger, parity, dim, face_num);
        pack.apply(stream);
      }
      break;
    default:
      errorQuda("Precision %d not supported", in.Precision());
    }
  }

  void packFaceExtendedStaggered(void *buffer[], cudaColorSpinorField &field, MemoryLocation location,  const int nFace, const int R[],
				 int dagger, int parity, const int dim, const int face_num, const cudaStream_t &stream, bool unpack=false)
  {
    switch(field.Precision()){
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceStaggered<double2,double> pack(buffer, &field, location, nFace, dagger, parity, dim, face_num, R, unpack);
        pack.apply(stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceStaggered<float2,float> pack(buffer, &field, location, nFace, dagger, parity, dim, face_num, R, unpack);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceStaggered<short2,float> pack(buffer, &field, location, nFace, dagger, parity, dim, face_num, R, unpack);
        pack.apply(stream);
      }
      break;
    case QUDA_QUARTER_PRECISION:
      {
        PackFaceStaggered<char2,float> pack(buffer, &field, location, nFace, dagger, parity, dim, face_num, R, unpack);
        pack.apply(stream);  
      }
      break;
    default:
      errorQuda("Precision %d not supported", field.Precision());
    } // switch(field.Precision())
  }

#ifdef GPU_DOMAIN_WALL_DIRAC
  template <int dagger, typename FloatN>
    __global__ void packFaceDWKernel(PackParam<FloatN> param)
  {
    const int nFace = 1; // 1 face for dwf

#ifdef STRIPED
    const int sites_per_block = param.sites_per_block;
    int local_tid = threadIdx.x;
    int tid = sites_per_block * blockIdx.x + local_tid;
#else
    int tid = block_idx(param.swizzle) * blockDim.x + threadIdx.x;
    constexpr int sites_per_block = 1;
    constexpr int local_tid = 0;
#endif

    while ( local_tid < sites_per_block && tid < param.threads ) {

      // determine which dimension we are packing
      int face_idx;
      const int dim = dimFromFaceIndex(face_idx, tid, param);

      const int Ls = param.dc.X[4];

      // compute where the output is located
      // compute an index into the local volume from the index into the face
      // read spinor, spin-project, and write half spinor to face
      if (dim == 0) {
        // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
        // FIXME these param.dc.ghostFaceCB constants do not incude the Ls dimension
        const int face_num = (face_idx >= nFace * Ls * param.dc.ghostFaceCB[0]) ? 1 : 0;
        face_idx -= face_num * nFace * Ls * param.dc.ghostFaceCB[0];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<5,QUDA_5D_PC,0,nFace,0>(face_idx,param);
          packFaceWilsonCore<0, dagger, 0>(param.out[0], param.outNorm[0], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[0], param);
        } else {
          const int idx = indexFromFaceIndex<5,QUDA_5D_PC,0,nFace,1>(face_idx,param);
          packFaceWilsonCore<0, dagger, 1>(param.out[1], param.outNorm[1], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[0], param);
        }
      } else if (dim == 1) {
        const int face_num = (face_idx >= nFace * Ls * param.dc.ghostFaceCB[1]) ? 1 : 0;
        face_idx -= face_num * nFace * Ls * param.dc.ghostFaceCB[1];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<5,QUDA_5D_PC,1,nFace,0>(face_idx,param);
          packFaceWilsonCore<1, dagger, 0>(param.out[2], param.outNorm[2], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[1], param);
        } else {
          const int idx = indexFromFaceIndex<5,QUDA_5D_PC,1,nFace,1>(face_idx,param);
          packFaceWilsonCore<1, dagger, 1>(param.out[3], param.outNorm[3], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[1], param);
        }
      } else if (dim == 2) {
        const int face_num = (face_idx >= nFace * Ls * param.dc.ghostFaceCB[2]) ? 1 : 0;
        face_idx -= face_num * nFace * Ls * param.dc.ghostFaceCB[2];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<5,QUDA_5D_PC,2,nFace,0>(face_idx,param);
          packFaceWilsonCore<2, dagger, 0>(param.out[4], param.outNorm[4], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[2], param);
        } else {
          const int idx = indexFromFaceIndex<5,QUDA_5D_PC,2,nFace,1>(face_idx,param);
          packFaceWilsonCore<2, dagger, 1>(param.out[5], param.outNorm[5], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[2], param);
        }
      } else {
        const int face_num = (face_idx >= nFace * Ls * param.dc.ghostFaceCB[3]) ? 1 : 0;
        face_idx -= face_num * nFace * Ls * param.dc.ghostFaceCB[3];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<5,QUDA_5D_PC,3,nFace,0>(face_idx,param);
          packFaceWilsonCore<3, dagger, 0>(param.out[6], param.outNorm[6], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[3], param);
        } else {
          const int idx = indexFromFaceIndex<5,QUDA_5D_PC,3,nFace,1>(face_idx,param);
          packFaceWilsonCore<3, dagger, 1>(param.out[7], param.outNorm[7], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[3], param);
        }
      }

#ifdef STRIPED
      local_tid += blockDim.x;
      tid += blockDim.x;
#else
      tid += blockDim.x*gridDim.x;
#endif
    } // while tid

  }


  template <int dagger, typename FloatN>
    __global__ void packFaceDW4DKernel(PackParam<FloatN> param)
  {
    const int nFace = 1; // 1 face for Wilson

#ifdef STRIPED
    const int sites_per_block = param.sites_per_block;
    int local_tid = threadIdx.x;
    int tid = sites_per_block * blockIdx.x + local_tid;
#else
    int tid = block_idx(param.swizzle) * blockDim.x + threadIdx.x;
    constexpr int sites_per_block = 1;
    constexpr int local_tid = 0;
#endif

    while ( local_tid < sites_per_block && tid < param.threads ) {

      const int Ls = param.dc.X[4];

      // determine which dimension we are packing
      int face_idx;
      const int dim = dimFromFaceIndex(face_idx, tid, param);

      // compute where the output is located
      // compute an index into the local volume from the index into the face
      // read spinor, spin-project, and write half spinor to face
      if (dim == 0) {
        // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
        // FIXME these param.dc.ghostFaceCB constants do not incude the Ls dimension
        const int face_num = (face_idx >= nFace * Ls * param.dc.ghostFaceCB[0]) ? 1 : 0;
        face_idx -= face_num * nFace * Ls * param.dc.ghostFaceCB[0];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,0,nFace,0>(face_idx,param);
          packFaceWilsonCore<0, dagger, 0>(param.out[0], param.outNorm[0], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[0], param);
        } else {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,0,nFace,1>(face_idx,param);
          packFaceWilsonCore<0, dagger, 1>(param.out[1], param.outNorm[1], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[0], param);
        }
      } else if (dim == 1) {
        const int face_num = (face_idx >= nFace * Ls * param.dc.ghostFaceCB[1]) ? 1 : 0;
        face_idx -= face_num * nFace * Ls * param.dc.ghostFaceCB[1];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,1,nFace,0>(face_idx,param);
          packFaceWilsonCore<1, dagger, 0>(param.out[2], param.outNorm[2], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[1], param);
        } else {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,1,nFace,1>(face_idx,param);
          packFaceWilsonCore<1, dagger, 1>(param.out[3], param.outNorm[3], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[1], param);
        }
      } else if (dim == 2) {
        const int face_num = (face_idx >= nFace * Ls * param.dc.ghostFaceCB[2]) ? 1 : 0;
        face_idx -= face_num * nFace * Ls * param.dc.ghostFaceCB[2];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,2,nFace,0>(face_idx,param);
          packFaceWilsonCore<2, dagger, 0>(param.out[4], param.outNorm[4], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[2], param);
        } else {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,2,nFace,1>(face_idx,param);
          packFaceWilsonCore<2, dagger, 1>(param.out[5], param.outNorm[5], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[2], param);
        }
      } else {
        const int face_num = (face_idx >= nFace * Ls * param.dc.ghostFaceCB[3]) ? 1 : 0;
        face_idx -= face_num * nFace * Ls * param.dc.ghostFaceCB[3];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,3,nFace,0>(face_idx,param);
          packFaceWilsonCore<3, dagger, 0>(param.out[6], param.outNorm[6], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[3], param);
        } else {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,3,nFace,1>(face_idx,param);
          packFaceWilsonCore<3, dagger, 1>(param.out[7], param.outNorm[7], param.in, param.inNorm, idx, face_idx,
              Ls * param.dc.ghostFaceCB[3], param);
        }
      }

#ifdef STRIPED
      local_tid += blockDim.x;
      tid += blockDim.x;
#else
      tid += blockDim.x*gridDim.x;
#endif
    } // while tid

  }

#endif

  template <typename FloatN, typename Float>
    class PackFaceDW : public PackFace<FloatN, Float> {

  private:

    int inputPerSite() const { return 24; } // input is full spinor
    int outputPerSite() const { return 12; } // output is spin projected

  public:
    PackFaceDW(void *faces[], const cudaColorSpinorField *in, MemoryLocation location,
	       const int dagger, const int parity)
    : PackFace<FloatN, Float>(faces, in, location, dagger, parity, 1) { }
    virtual ~PackFaceDW() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_DOMAIN_WALL_DIRAC
      static PackParam<FloatN> param;
      this->prepareParam(param,tp);
      void *args[] = { &param };
      void (*func)(PackParam<FloatN>) = this->dagger ? &(packFaceDWKernel<1,FloatN>) : &(packFaceDWKernel<0,FloatN>);
      cudaLaunchKernel( (const void*)func, tp.grid, tp.block, args, tp.shared_bytes, stream);
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
    PackFaceDW4D(void *faces[], const cudaColorSpinorField *in, MemoryLocation location,
		 const int dagger, const int parity)
      : PackFace<FloatN, Float>(faces, in, location, dagger, parity, 1) { }
    virtual ~PackFaceDW4D() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_DOMAIN_WALL_DIRAC
      static PackParam<FloatN> param;
      this->prepareParam(param,tp);
      void *args[] = { &param };
      void (*func)(PackParam<FloatN>) = this->dagger ? &(packFaceDW4DKernel<1,FloatN>) : &(packFaceDW4DKernel<0,FloatN>);
      cudaLaunchKernel( (const void*)func, tp.grid, tp.block, args, tp.shared_bytes, stream);
#else
      errorQuda("4D preconditioned DW face packing kernel is not built");
#endif
    }

    long long flops() const { return outputPerSite()*this->threads(); }
  };

  void packFaceDW(void *ghost_buf[], cudaColorSpinorField &in, MemoryLocation location, const int dagger,
		  const int parity, const cudaStream_t &stream) {

    if (in.PCType() == QUDA_4D_PC) {
      switch (in.Precision()) {
      case QUDA_DOUBLE_PRECISION: {
        PackFaceDW4D<double2, double> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      } break;
      case QUDA_SINGLE_PRECISION: {
        PackFaceDW4D<float4, float> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      } break;
      case QUDA_HALF_PRECISION: {
        PackFaceDW4D<short4, float> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      } break;
      case QUDA_QUARTER_PRECISION: {
        PackFaceDW4D<char4, float> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      } break;
      default: errorQuda("Precision %d not supported", in.Precision());
      }
    } else {
      switch (in.Precision()) {
      case QUDA_DOUBLE_PRECISION: {
        PackFaceDW<double2, double> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      } break;
      case QUDA_SINGLE_PRECISION: {
        PackFaceDW<float4, float> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      } break;
      case QUDA_HALF_PRECISION: {
        PackFaceDW<short4, float> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      } break;
      case QUDA_QUARTER_PRECISION: {
        PackFaceDW<char4, float> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      } break;
      default: errorQuda("Precision %d not supported", in.Precision());
      }
    }
  }

#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
  template <int dagger, typename FloatN>
    __global__ void packFaceNdegTMKernel(PackParam<FloatN> param)
  {
    const int nFace = 1; // 1 face for Wilson
    const int Nf = 2;

#ifdef STRIPED
    const int sites_per_block = param.sites_per_block;
    int local_tid = threadIdx.x;
    int tid = sites_per_block * blockIdx.x + local_tid;
#else
    int tid = block_idx(param.swizzle) * blockDim.x + threadIdx.x;
    constexpr int sites_per_block = 1;
    constexpr int local_tid = 0;
#endif

    while ( local_tid < sites_per_block && tid < param.threads ) {

      // determine which dimension we are packing
      int face_idx;
      const int dim = dimFromFaceIndex(face_idx, tid, param);

      // compute where the output is located
      // compute an index into the local volume from the index into the face
      // read spinor, spin-project, and write half spinor to face
      if (dim == 0) {
        // face_num determines which end of the lattice we are packing:
        // 0 = beginning, 1 = end FIXME these param.dc.ghostFaceCB constants
        // do not include the Nf dimension
        const int face_num = (face_idx >= nFace * Nf * param.dc.ghostFaceCB[0]) ? 1 : 0;
        face_idx -= face_num * nFace * Nf * param.dc.ghostFaceCB[0];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,0,nFace,0>(face_idx,param);
          packFaceWilsonCore<0, dagger, 0>(param.out[0], param.outNorm[0], param.in, param.inNorm, idx, face_idx,
              Nf * param.dc.ghostFaceCB[0], param);
        } else {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,0,nFace,1>(face_idx,param);
          packFaceWilsonCore<0, dagger, 1>(param.out[1], param.outNorm[1], param.in, param.inNorm, idx, face_idx,
              Nf * param.dc.ghostFaceCB[0], param);
        }
      } else if (dim == 1) {
        const int face_num = (face_idx >= nFace * Nf * param.dc.ghostFaceCB[1]) ? 1 : 0;
        face_idx -= face_num * nFace * Nf * param.dc.ghostFaceCB[1];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,1,nFace,0>(face_idx,param);
          packFaceWilsonCore<1, dagger, 0>(param.out[2], param.outNorm[2], param.in, param.inNorm, idx, face_idx,
              Nf * param.dc.ghostFaceCB[1], param);
        } else {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,1,nFace,1>(face_idx,param);
          packFaceWilsonCore<1, dagger, 1>(param.out[3], param.outNorm[3], param.in, param.inNorm, idx, face_idx,
              Nf * param.dc.ghostFaceCB[1], param);
        }
      } else if (dim == 2) {
        const int face_num = (face_idx >= nFace * Nf * param.dc.ghostFaceCB[2]) ? 1 : 0;
        face_idx -= face_num * nFace * Nf * param.dc.ghostFaceCB[2];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,2,nFace,0>(face_idx,param);
          packFaceWilsonCore<2, dagger, 0>(param.out[4], param.outNorm[4], param.in, param.inNorm, idx, face_idx,
              Nf * param.dc.ghostFaceCB[2], param);
        } else {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,2,nFace,1>(face_idx,param);
          packFaceWilsonCore<2, dagger, 1>(param.out[5], param.outNorm[5], param.in, param.inNorm, idx, face_idx,
              Nf * param.dc.ghostFaceCB[2], param);
        }
      } else {
        const int face_num = (face_idx >= nFace * Nf * param.dc.ghostFaceCB[3]) ? 1 : 0;
        face_idx -= face_num * nFace * Nf * param.dc.ghostFaceCB[3];
        if (face_num == 0) {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,3,nFace,0>(face_idx,param);
          packFaceWilsonCore<3, dagger, 0>(param.out[6], param.outNorm[6], param.in, param.inNorm, idx, face_idx,
              Nf * param.dc.ghostFaceCB[3], param);
        } else {
          const int idx = indexFromFaceIndex<5,QUDA_4D_PC,3,nFace,1>(face_idx,param);
          packFaceWilsonCore<3, dagger, 1>(param.out[7], param.outNorm[7], param.in, param.inNorm, idx, face_idx,
              Nf * param.dc.ghostFaceCB[3], param);
        }
      }

#ifdef STRIPED
      local_tid += blockDim.x;
      tid += blockDim.x;
#else
      tid += blockDim.x*gridDim.x;
#endif
    } // while tid

  }

#endif

  template <typename FloatN, typename Float>
    class PackFaceNdegTM : public PackFace<FloatN, Float> {

  private:

    int inputPerSite() const { return 24; } // input is full spinor
    int outputPerSite() const { return 12; } // output is spin projected

  public:
    PackFaceNdegTM(void *faces[], const cudaColorSpinorField *in, MemoryLocation location,
		   const int dagger, const int parity)
      : PackFace<FloatN, Float>(faces, in, location, dagger, parity, 1) { }
    virtual ~PackFaceNdegTM() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
      static PackParam<FloatN> param;
      this->prepareParam(param,tp);
      void *args[] = { &param };
      void (*func)(PackParam<FloatN>) = this->dagger ? &(packFaceNdegTMKernel<1,FloatN>) : &(packFaceNdegTMKernel<0,FloatN>);
      cudaLaunchKernel( (const void*)func, tp.grid, tp.block, args, tp.shared_bytes, stream);
#else
      errorQuda("Non-degenerate twisted mass face packing kernel is not built");
#endif
    }

    long long flops() const { return outputPerSite()*this->threads(); }
  };

  void packFaceNdegTM(void *ghost_buf[], cudaColorSpinorField &in, MemoryLocation location, const int dagger,
		      const int parity, const cudaStream_t &stream) {

    switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceNdegTM<double2, double> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceNdegTM<float4, float> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceNdegTM<short4, float> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      }
      break;
    case QUDA_QUARTER_PRECISION:
      {
        PackFaceNdegTM<char4, float> pack(ghost_buf, &in, location, dagger, parity);
        pack.apply(stream);
      }
      break;
    default:
      errorQuda("Precision %d not supported", in.Precision());
    }
  }

  void packFace(void *ghost_buf[2*QUDA_MAX_DIM], cudaColorSpinorField &in,
		MemoryLocation location, const int nFace,
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
      packFaceStaggered(ghost_buf, in, location, nFace, dagger, parity, dim, face_num, stream);
    } else if (a!=0.0 || b!=0.0) {
      // Need to update this logic for other multi-src dslash packing
      if(in.TwistFlavor() == QUDA_TWIST_SINGLET) {
	packTwistedFaceWilson(ghost_buf, in, location, dagger, parity, a, b, stream);
      } else {
	errorQuda("Cannot perform twisted packing for the spinor.");
      }
    } else if (in.Ndim() == 5) {
      if(in.TwistFlavor() == QUDA_TWIST_INVALID) {
	packFaceDW(ghost_buf, in, location, dagger, parity, stream);
      } else {
	packFaceNdegTM(ghost_buf, in, location, dagger, parity, stream);
      }
    } else {
      packFaceWilson(ghost_buf, in, location, dagger, parity, stream);
    }
  }



  void packFaceExtended(void* buffer[2*QUDA_MAX_DIM], cudaColorSpinorField &field,
			MemoryLocation location, const int nFace, const int R[],
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
      packFaceExtendedStaggered(buffer, field, location, nFace, R, dagger, parity, dim, face_num, stream, unpack);
    }else{
      errorQuda("Extended quark field is not supported");
    }

  }

#endif // MULTI_GPU

} // namespace quda

#endif
