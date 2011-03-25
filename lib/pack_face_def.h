// compute an index into the local volume from an index into the face
template <int dir, int nLayers, int parity>
static inline __device__ int indexFromFaceIndex(int face_idx, const int &face_volume,
						const int &face_num)
{
  // dimensions of the face (FIXME: optimize using constant cache)

  int face_X = X1, face_Y = X2, face_Z = X3; // face_T = X4;
  switch (dir) {
  case 0:
    face_X = nLayers;
    break;
  case 1:
    face_Y = nLayers;
    break;
  case 2:
    face_Z = nLayers;
    break;
  case 3:
    // face_T = nLayers;
    break;
  }
  int face_XYZ = face_X * face_Y * face_Z;
  int face_XY = face_X * face_Y;

  // intrinsic parity of the face depends on offset of first element

  int face_parity;
  switch (dir) {
  case 0:
    face_parity = (parity + face_num * (X1 - nLayers)) & 1;
    break;
  case 1:
    face_parity = (parity + face_num * (X2 - nLayers)) & 1;
    break;
  case 2:
    face_parity = (parity + face_num * (X3 - nLayers)) & 1;
    break;
  case 3:
    face_parity = (parity + face_num * (X4 - nLayers)) & 1;
    break;
  }

  // reconstruct full face index from index into the checkerboard

  face_idx *= 2;

  if (face_XYZ & 1) {
    face_idx += face_parity;
  } else if (face_XY & 1) {
    int t = face_idx / face_XYZ;
    face_idx += (face_parity + t) & 1;
  } else if (face_X & 1) {
    int t = face_idx / face_XYZ;
    int z = (face_idx / face_XY) % face_Z;
    face_idx += (face_parity + t + z) & 1;
  } else { // FIXME: optimize to use only 3 divisions and no mods
    int t = face_idx / face_XYZ;
    int z = (face_idx / face_XY) % face_Z;
    int y = (face_idx / face_X) % face_Y;
    face_idx += (face_parity + t + z + y) & 1;
  }

  // compute index into the full local volume

  int idx = face_idx;
  int gap, aux;

  switch (dir) {
  case 0:
    gap = X1 - nLayers;
    aux = face_idx / face_X;
    idx += (aux + face_num) * gap;
    break;
  case 1:
    gap = X2 - nLayers;
    aux = face_idx / face_XY;
    idx += (aux + face_num) * gap * face_X;
    break;
  case 2:
    gap = X3 - nLayers;
    aux = face_idx / face_XYZ;
    idx += (aux + face_num) * gap * face_XY;
    break;
  case 3:
    gap = X4 - nLayers;
    idx += face_num * gap * face_XYZ;
    break;
  }

  // return index into the checkerboard

  return idx/2;
}


#ifdef GPU_WILSON_DIRAC
template <int dir, int dagger, int parity, typename FloatN>
__global__ void packFaceWilsonKernel(FloatN *out, float *outNorm, const FloatN *in, const float *inNorm)
{
  int face_volume = ghostFace[dir];

  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (face_idx >= 2*face_volume) return;

  // are we on face 0 or face 1?
  int face_num = (face_idx >= face_volume);

  face_idx -= face_num*face_volume;

  // set out to point to the beginning of face 1 if necessary
  if (typeid(FloatN) == typeid(double2)) {
    out += face_num * 6 * face_volume;
  } else {
    out += face_num * 3 * face_volume;
  }

  // compute an index into the local volume from the index into the face
  int idx = indexFromFaceIndex<dir, 1, parity>(face_idx, face_volume, face_num);

  if (typeid(FloatN) == typeid(double2)) {

#if (__CUDA_ARCH__ >= 130)
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_DOUBLE
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_DOUBLE_TEX
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN_TEX
#define SPINORTEX spinorTexDouble
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_DOUBLE2
#define SPINOR_DOUBLE
    if (dagger) {
#include "wilson_pack_face_dagger_core.h"
    } else {
#include "wilson_pack_face_core.h"
    }
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR
#undef SPINOR_DOUBLE
#endif // (__CUDA_ARCH__ >= 130)

  } else if (typeid(FloatN) == typeid(float4)) {

#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_SINGLE_TEX
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN_TEX
#define SPINORTEX spinorTexSingle
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_FLOAT4
    if (dagger) {
#include "wilson_pack_face_dagger_core.h"
    } else {
#include "wilson_pack_face_core.h"
    }
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR

  } else if (typeid(FloatN) == typeid(short4)) {

#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_HALF_TEX
#define READ_SPINOR_UP READ_SPINOR_HALF_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN_TEX
#define SPINORTEX spinorTexHalf
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_SHORT4
    if (dagger) {
#include "wilson_pack_face_dagger_core.h"
    } else {
#include "wilson_pack_face_core.h"
    }
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR

  }
}
#endif // GPU_WILSON_DIRAC


template <typename FloatN>
void packFaceWilson(FloatN *faces, float *facesNorm, const FloatN *in, const float *inNorm, const int dir,
		    const int dagger, const int parity, const dim3 &gridDim, const dim3 &blockDim, const cudaStream_t &stream)
{
#ifdef GPU_WILSON_DIRAC
  if (parity) {
    if (dagger) {
      switch (dir) {
      case 0: packFaceWilsonKernel<0,1,1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      case 1: packFaceWilsonKernel<1,1,1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      case 2: packFaceWilsonKernel<2,1,1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      case 3: packFaceWilsonKernel<3,1,1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      }
    } else {
      switch (dir) {
      case 0: packFaceWilsonKernel<0,0,1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      case 1: packFaceWilsonKernel<1,0,1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      case 2: packFaceWilsonKernel<2,0,1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      case 3: packFaceWilsonKernel<3,0,1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      }
    }
  } else {
    if (dagger) {
      switch (dir) {
      case 0: packFaceWilsonKernel<0,1,0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      case 1: packFaceWilsonKernel<1,1,0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      case 2: packFaceWilsonKernel<2,1,0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      case 3: packFaceWilsonKernel<3,1,0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      }
    } else {
      switch (dir) {
      case 0: packFaceWilsonKernel<0,0,0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      case 1: packFaceWilsonKernel<1,0,0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      case 2: packFaceWilsonKernel<2,0,0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      case 3: packFaceWilsonKernel<3,0,0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
      }
    }
  }
#else
  errorQuda("Wilson face packing kernel is not built");
#endif  
}

void packFaceWilson(cudaColorSpinorField &in, const int dir, const int dagger, const int parity,
		    const QudaPrecision precision, const cudaStream_t &stream) {

  dim3 blockDim(64, 1, 1); // make this a parameter for auto-tuning
  dim3 gridDim( (ghostFace[dir]+blockDim.x-1) / blockDim.x, 1, 1);

  char *ghost = (char*)in.v + (in.length + in.ghostOffset[dir]*in.nColor*in.nSpin*2)*precision;
  float *ghostNorm = in.norm + in.stride + in.ghostNormOffset[dir];

  switch(precision) {
  case QUDA_DOUBLE_PRECISION:
    packFaceWilson((double2*)ghost, ghostNorm, (double2*)in.v, (float*)in.norm, dir, dagger, parity, gridDim, blockDim, stream);
    break;
  case QUDA_SINGLE_PRECISION:
    packFaceWilson((float4*)ghost, ghostNorm, (float4*)in.v, (float*)in.norm, dir, dagger, parity, gridDim, blockDim, stream);
    break;
  case QUDA_HALF_PRECISION:
    packFaceWilson((short4*)ghost, ghostNorm, (short4*)in.v, (float*)in.norm, dir, dagger, parity, gridDim, blockDim, stream);
    break;
  }  

}


// TODO: add support for texture reads

#ifdef GPU_STAGGERED_DIRAC
template <int dir, typename Float2>
__global__ void packFaceAsqtadKernel(Float2 *out, float *outNorm, const Float2 *in, const float *inNorm)
{
  int face_volume = ghostFace[dir];

  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (face_idx >= 2*face_volume) return;

  // are we on face 0 or face 1?
  int face_num = (face_idx >= face_volume);

  face_idx -= face_num*face_volume;

  // set out to point to the beginning of face 1 if necessary
  out += face_num * 3 * face_volume;

  // compute an index into the local volume from the index into the face
  int idx = indexFromFaceIndex<dir, 3>(face_idx, face_volume, face_num);

  out[face_idx + 0*face_volume] = in[idx + 0*sp_stride];
  out[face_idx + 1*face_volume] = in[idx + 1*sp_stride];
  out[face_idx + 2*face_volume] = in[idx + 2*sp_stride];

  if (typeid(Float2) == typeid(short2)) outNorm[face_idx] = inNorm[idx];
}
#endif // GPU_STAGGERED_DIRAC


template <typename Float2>
void packFaceAsqtad(Float2 *faces, float *facesNorm, const Float2 *in, const float *inNorm, int dir,
		    const dim3 &gridDim, const dim3 &blockDim, const cudaStream_t &stream)
{
#ifdef GPU_STAGGERED_DIRAC
  switch (dir) {
  case 0: packFaceAsqtadKernel<0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
  case 1: packFaceAsqtadKernel<1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
  case 2: packFaceAsqtadKernel<2><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
  case 3: packFaceAsqtadKernel<3><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm); break;
  }
#else
  errorQuda("Asqtad face packing kernel is not built");
#endif  
}
