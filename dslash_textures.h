#if (__CUDA_ARCH__ == 130)
static __inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
{
  int4 v = tex1Dfetch(t,i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}
#endif

// Double precision gauge field
texture<int4, 1> gauge0TexDouble;
texture<int4, 1> gauge1TexDouble;

// Single precision gauge field
texture<float4, 1, cudaReadModeElementType> gauge0TexSingle;
texture<float4, 1, cudaReadModeElementType> gauge1TexSingle;

// Half precision gauge field
texture<short4, 1, cudaReadModeNormalizedFloat> gauge0TexHalf;
texture<short4, 1, cudaReadModeNormalizedFloat> gauge1TexHalf;

// Double precision input spinor field
texture<int4, 1> spinorTexDouble;

// Single precision input spinor field
texture<float4, 1, cudaReadModeElementType> spinorTexSingle;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> spinorTexHalf;
texture<float, 1, cudaReadModeElementType> spinorTexNorm;

// Double precision accumulate spinor field
texture<int4, 1> accumTexDouble;

// Single precision accumulate spinor field
texture<float4, 1, cudaReadModeElementType> accumTexSingle;

// Half precision accumulate spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> accumTexHalf;
texture<float, 1, cudaReadModeElementType> accumTexNorm;

// Double precision clover term
texture<int4, 1> cloverTexDouble;

// Single precision clover term
texture<float4, 1, cudaReadModeElementType> cloverTexSingle;

// Half precision clover term
texture<short4, 1, cudaReadModeNormalizedFloat> cloverTexHalf;
texture<float, 1, cudaReadModeElementType> cloverTexNorm;
