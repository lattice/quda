#include <typeinfo>

// Use this macro for texture fetching for supporting either texture objects of texture references
#ifdef USE_TEXTURE_OBJECTS
#define TEX1DFETCH(type, tex, idx) tex1Dfetch<type>((tex), idx)
#else
#define TEX1DFETCH(type, tex, idx) tex1Dfetch((tex), idx)
#endif

#if (__COMPUTE_CAPABILITY__ >= 130)
template <typename Tex>
static __inline__ __device__ double2 fetch_double2(Tex t, int i)
{
  int4 v = TEX1DFETCH(int4, t, i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}

static __inline__ __device__ double2 fetch_double2_old(texture<int4, 1> t, int i)
{
  int4 v = tex1Dfetch(t,i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}
#endif //__COMPUTE_CAPABILITY__ >= 130

#ifndef USE_TEXTURE_OBJECTS
// Double precision gauge field
texture<int4, 1> gauge0TexDouble2;
texture<int4, 1> gauge1TexDouble2;

// Single precision gauge field
texture<float4, 1, cudaReadModeElementType> gauge0TexSingle4;
texture<float4, 1, cudaReadModeElementType> gauge1TexSingle4;
texture<float2, 1, cudaReadModeElementType> gauge0TexSingle2;
texture<float2, 1, cudaReadModeElementType> gauge1TexSingle2;

// Half precision gauge field
texture<short4, 1, cudaReadModeNormalizedFloat> gauge0TexHalf4;
texture<short4, 1, cudaReadModeNormalizedFloat> gauge1TexHalf4;
texture<short2, 1, cudaReadModeNormalizedFloat> gauge0TexHalf2;
texture<short2, 1, cudaReadModeNormalizedFloat> gauge1TexHalf2;

texture<int4, 1> longGauge0TexDouble;
texture<int4, 1> longGauge1TexDouble;
texture<float4, 1, cudaReadModeElementType> longGauge0TexSingle;
texture<float4, 1, cudaReadModeElementType> longGauge1TexSingle;
texture<float2, 1, cudaReadModeElementType> longGauge0TexSingle_norecon;
texture<float2, 1, cudaReadModeElementType> longGauge1TexSingle_norecon;

texture<short4, 1, cudaReadModeNormalizedFloat> longGauge0TexHalf;
texture<short4, 1, cudaReadModeNormalizedFloat> longGauge1TexHalf;
texture<short2, 1, cudaReadModeNormalizedFloat> longGauge0TexHalf_norecon;
texture<short2, 1, cudaReadModeNormalizedFloat> longGauge1TexHalf_norecon;

// Double precision input spinor field
texture<int4, 1> spinorTexDouble;

// Single precision input spinor field
texture<float4, 1, cudaReadModeElementType> spinorTexSingle;
texture<float2, 1, cudaReadModeElementType> spinorTexSingle2;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> spinorTexHalf;
texture<short2, 1, cudaReadModeNormalizedFloat> spinorTexHalf2;
texture<float, 1, cudaReadModeElementType> spinorTexHalfNorm;
texture<float, 1, cudaReadModeElementType> spinorTexHalf2Norm;

// Double precision accumulate spinor field
texture<int4, 1> accumTexDouble;

// Single precision accumulate spinor field
texture<float4, 1, cudaReadModeElementType> accumTexSingle;
texture<float2, 1, cudaReadModeElementType> accumTexSingle2;

// Half precision accumulate spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> accumTexHalf;
texture<short2, 1, cudaReadModeNormalizedFloat> accumTexHalf2;
texture<float, 1, cudaReadModeElementType> accumTexHalfNorm;
texture<float, 1, cudaReadModeElementType> accumTexHalf2Norm;

// Double precision intermediate spinor field (used by exterior Dslash kernels)
texture<int4, 1> interTexDouble;

// Single precision intermediate spinor field
texture<float4, 1, cudaReadModeElementType> interTexSingle;
texture<float2, 1, cudaReadModeElementType> interTexSingle2;

// Half precision intermediate spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> interTexHalf;
texture<short2, 1, cudaReadModeNormalizedFloat> interTexHalf2;
texture<float, 1, cudaReadModeElementType> interTexHalfNorm;
texture<float, 1, cudaReadModeElementType> interTexHalf2Norm;
#endif // not defined USE_TEXTURE_OBJECTS

// FIXME update the below textures for texture objects

// fatGauge textures are still used by llfat so we need to define
texture<int4, 1> fatGauge0TexDouble;
texture<int4, 1> fatGauge1TexDouble;
texture<float2, 1, cudaReadModeElementType> fatGauge0TexSingle;
texture<float2, 1, cudaReadModeElementType> fatGauge1TexSingle;
texture<short2, 1, cudaReadModeNormalizedFloat> fatGauge0TexHalf;
texture<short2, 1, cudaReadModeNormalizedFloat> fatGauge1TexHalf;

//Double precision for site link
texture<int4, 1> siteLink0TexDouble;
texture<int4, 1> siteLink1TexDouble;

//Single precision for site link
texture<float2, 1, cudaReadModeElementType> siteLink0TexSingle;
texture<float2, 1, cudaReadModeElementType> siteLink1TexSingle;

texture<float4, 1, cudaReadModeElementType> siteLink0TexSingle_recon;
texture<float4, 1, cudaReadModeElementType> siteLink1TexSingle_recon;

texture<float2, 1, cudaReadModeElementType> siteLink0TexSingle_norecon;
texture<float2, 1, cudaReadModeElementType> siteLink1TexSingle_norecon;


texture<int4, 1> muLink0TexDouble;
texture<int4, 1> muLink1TexDouble;
// Single precision mulink field
texture<float2, 1, cudaReadModeElementType> muLink0TexSingle;
texture<float2, 1, cudaReadModeElementType> muLink1TexSingle;

void bindGaugeTex(const cudaGaugeField &gauge, const int oddBit, void **gauge0, void **gauge1)
{
  if(oddBit) {
    *gauge0 = gauge.odd;
    *gauge1 = gauge.even;
  } else {
    *gauge0 = gauge.even;
    *gauge1 = gauge.odd;
  }
  
#ifdef USE_TEXTURE_OBJECTS
  dslashParam.gauge0Tex = oddBit ? gauge.OddTex() : gauge.EvenTex();
  dslashParam.gauge1Tex = oddBit ? gauge.EvenTex() : gauge.OddTex();
#else
  if (gauge.reconstruct == QUDA_RECONSTRUCT_NO) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      cudaBindTexture(0, gauge0TexDouble2, *gauge0, gauge.bytes/2); 
      cudaBindTexture(0, gauge1TexDouble2, *gauge1, gauge.bytes/2);
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      cudaBindTexture(0, gauge0TexSingle2, *gauge0, gauge.bytes/2); 
      cudaBindTexture(0, gauge1TexSingle2, *gauge1, gauge.bytes/2);
    } else {
      cudaBindTexture(0, gauge0TexHalf2, *gauge0, gauge.bytes/2); 
      cudaBindTexture(0, gauge1TexHalf2, *gauge1, gauge.bytes/2);
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      cudaBindTexture(0, gauge0TexDouble2, *gauge0, gauge.bytes/2); 
      cudaBindTexture(0, gauge1TexDouble2, *gauge1, gauge.bytes/2);
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      cudaBindTexture(0, gauge0TexSingle4, *gauge0, gauge.bytes/2); 
      cudaBindTexture(0, gauge1TexSingle4, *gauge1, gauge.bytes/2);
    } else {
      cudaBindTexture(0, gauge0TexHalf4, *gauge0, gauge.bytes/2); 
      cudaBindTexture(0, gauge1TexHalf4, *gauge1, gauge.bytes/2);
    }
  }
#endif // USE_TEXTURE_OBJECTS

}

void unbindGaugeTex(const cudaGaugeField &gauge)
{
#if (!defined USE_TEXTURE_OBJECTS)
  if (gauge.reconstruct == QUDA_RECONSTRUCT_NO) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      cudaUnbindTexture(gauge0TexDouble2); 
      cudaUnbindTexture(gauge1TexDouble2);
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      cudaUnbindTexture(gauge0TexSingle2);
      cudaUnbindTexture(gauge1TexSingle2);
    } else {
      cudaUnbindTexture(gauge0TexHalf2); 
      cudaUnbindTexture(gauge1TexHalf2);
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      cudaUnbindTexture(gauge0TexDouble2); 
      cudaUnbindTexture(gauge1TexDouble2);
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      cudaUnbindTexture(gauge0TexSingle4); 
      cudaUnbindTexture(gauge1TexSingle4);
    } else {
      cudaUnbindTexture(gauge0TexHalf4); 
      cudaUnbindTexture(gauge1TexHalf4);
    }
  }
#endif
}

void bindFatGaugeTex(const cudaGaugeField &gauge, const int oddBit, void **gauge0, void **gauge1)
{
  if(oddBit) {
    *gauge0 = gauge.odd;
    *gauge1 = gauge.even;
  } else {
    *gauge0 = gauge.even;
    *gauge1 = gauge.odd;
  }
  
#ifdef USE_TEXTURE_OBJECTS
  dslashParam.gauge0Tex = oddBit ? gauge.OddTex() : gauge.EvenTex();
  dslashParam.gauge1Tex = oddBit ? gauge.EvenTex() : gauge.OddTex();
#else
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    cudaBindTexture(0, fatGauge0TexDouble, *gauge0, gauge.bytes/2); 
    cudaBindTexture(0, fatGauge1TexDouble, *gauge1, gauge.bytes/2);
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    cudaBindTexture(0, fatGauge0TexSingle, *gauge0, gauge.bytes/2); 
    cudaBindTexture(0, fatGauge1TexSingle, *gauge1, gauge.bytes/2);
  } else {
    cudaBindTexture(0, fatGauge0TexHalf, *gauge0, gauge.bytes/2); 
    cudaBindTexture(0, fatGauge1TexHalf, *gauge1, gauge.bytes/2);
  }
#endif // USE_TEXTURE_OBJECTS

}

void unbindFatGaugeTex(const cudaGaugeField &gauge)
{
#if (!defined USE_TEXTURE_OBJECTS)
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    cudaUnbindTexture(fatGauge0TexDouble);
    cudaUnbindTexture(fatGauge1TexDouble);
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    cudaUnbindTexture(fatGauge0TexSingle);
    cudaUnbindTexture(fatGauge1TexSingle);
  } else {
    cudaUnbindTexture(fatGauge0TexHalf);
    cudaUnbindTexture(fatGauge1TexHalf);
  }
#endif
}

void bindLongGaugeTex(const cudaGaugeField &gauge, const int oddBit, void **gauge0, void **gauge1)
{
  if(oddBit) {
    *gauge0 = gauge.odd;
    *gauge1 = gauge.even;
  } else {
    *gauge0 = gauge.even;
    *gauge1 = gauge.odd;
  }
  
#ifdef USE_TEXTURE_OBJECTS
  dslashParam.longGauge0Tex = oddBit ? gauge.OddTex() : gauge.EvenTex();
  dslashParam.longGauge1Tex = oddBit ? gauge.EvenTex() : gauge.OddTex();
#else
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    cudaBindTexture(0, longGauge0TexDouble, *gauge0, gauge.bytes/2); 
    cudaBindTexture(0, longGauge1TexDouble, *gauge1, gauge.bytes/2);
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_NO) { //18 reconstruct
      cudaBindTexture(0, longGauge0TexSingle_norecon, *gauge0, gauge.bytes/2); 
      cudaBindTexture(0, longGauge1TexSingle_norecon, *gauge1, gauge.bytes/2);	
    } else {
      cudaBindTexture(0, longGauge0TexSingle, *gauge0, gauge.bytes/2); 
      cudaBindTexture(0, longGauge1TexSingle, *gauge1, gauge.bytes/2);
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_NO) { //18 reconstruct
      cudaBindTexture(0, longGauge0TexHalf_norecon, *gauge0, gauge.bytes/2); 
      cudaBindTexture(0, longGauge1TexHalf_norecon, *gauge1, gauge.bytes/2);	
    } else {
      cudaBindTexture(0, longGauge0TexHalf, *gauge0, gauge.bytes/2); 
      cudaBindTexture(0, longGauge1TexHalf, *gauge1, gauge.bytes/2);
    }
  }
#endif // USE_TEXTURE_OBJECTS
}

void unbindLongGaugeTex(const cudaGaugeField &gauge)
{
#if (!defined USE_TEXTURE_OBJECTS)
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    cudaUnbindTexture(longGauge0TexDouble);
    cudaUnbindTexture(longGauge1TexDouble);
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_NO) { //18 reconstruct
      cudaUnbindTexture(longGauge0TexSingle_norecon);
      cudaUnbindTexture(longGauge1TexSingle_norecon);
    } else {
      cudaUnbindTexture(longGauge0TexSingle);
      cudaUnbindTexture(longGauge1TexSingle);
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_NO) { //18 reconstruct
      cudaUnbindTexture(longGauge0TexHalf_norecon);
      cudaUnbindTexture(longGauge1TexHalf_norecon);
    } else {
      cudaUnbindTexture(longGauge0TexHalf);
      cudaUnbindTexture(longGauge1TexHalf);
    }
  }
#endif
}
    

template <typename spinorFloat>
int bindSpinorTex(const cudaColorSpinorField *in, const cudaColorSpinorField *out=0, 
		  const cudaColorSpinorField *x=0) {
  int size = (sizeof(((spinorFloat*)0)->x) < sizeof(float)) ? sizeof(float) :
    sizeof(((spinorFloat*)0)->x);

#ifdef USE_TEXTURE_OBJECTS
  dslashParam.inTex = in->Tex();
  dslashParam.inTexNorm = in->TexNorm();
  if (out) dslashParam.outTex = out->Tex();
  if (out) dslashParam.outTexNorm = out->TexNorm();
  if (x) dslashParam.xTex = x->Tex();
  if (x) dslashParam.xTexNorm = x->TexNorm();
#else
  if (typeid(spinorFloat) == typeid(double2)) {
    cudaBindTexture(0, spinorTexDouble, in->V(), in->Bytes()); 
    if (out) cudaBindTexture(0, interTexDouble, out->V(), in->Bytes());
    if (x) cudaBindTexture(0, accumTexDouble, x->V(), in->Bytes());
  } else if (typeid(spinorFloat) == typeid(float4)) {
    cudaBindTexture(0, spinorTexSingle, in->V(), in->Bytes()); 
    if (out) cudaBindTexture(0, interTexSingle, out->V(), in->Bytes());
    if (x) cudaBindTexture(0, accumTexSingle, x->V(), in->Bytes()); 
  } else if  (typeid(spinorFloat) == typeid(float2)) {
    cudaBindTexture(0, spinorTexSingle2, in->V(), in->Bytes()); 
    if (out) cudaBindTexture(0, interTexSingle2, out->V(), in->Bytes()); 
    if (x) cudaBindTexture(0, accumTexSingle2, x->V(), in->Bytes()); 
  } else if (typeid(spinorFloat) == typeid(short4)) {
    cudaBindTexture(0, spinorTexHalf, in->V(), in->Bytes()); 
    cudaBindTexture(0, spinorTexHalfNorm, in->Norm(), in->NormBytes()); 
    if (out) cudaBindTexture(0, interTexHalf, out->V(), in->Bytes()); 
    if (out) cudaBindTexture(0, interTexHalfNorm, out->Norm(), in->NormBytes()); 
    if (x) cudaBindTexture(0, accumTexHalf, x->V(), in->Bytes()); 
    if (x) cudaBindTexture(0, accumTexHalfNorm, x->Norm(), in->NormBytes()); 
  } else if (typeid(spinorFloat) == typeid(short2)) {
    cudaBindTexture(0, spinorTexHalf2, in->V(), in->Bytes()); 
    cudaBindTexture(0, spinorTexHalf2Norm, in->Norm(), in->NormBytes()); 
    if (out) cudaBindTexture(0, interTexHalf2, out->V(), in->Bytes()); 
    if (out) cudaBindTexture(0, interTexHalf2Norm, out->Norm(), in->NormBytes()); 
    if (x) cudaBindTexture(0, accumTexHalf2, x->V(), in->Bytes()); 
    if (x) cudaBindTexture(0, accumTexHalf2Norm, x->Norm(), in->NormBytes()); 
  } else {
    errorQuda("Unsupported precision and short vector type");
  }
#endif // USE_TEXTURE_OBJECTS

  return size;
}

template <typename spinorFloat>
void unbindSpinorTex(const cudaColorSpinorField *in, const cudaColorSpinorField *out=0, 
		     const cudaColorSpinorField *x=0) {
#ifndef USE_TEXTURE_OBJECTS
  if (typeid(spinorFloat) == typeid(double2)) {
    cudaUnbindTexture(spinorTexDouble);
    if (out) cudaUnbindTexture(interTexDouble);
    if (x) cudaUnbindTexture(accumTexDouble);
  } else if (typeid(spinorFloat) == typeid(float4)) {
    cudaUnbindTexture(spinorTexSingle); 
    if (out) cudaUnbindTexture(interTexSingle); 
    if (x) cudaUnbindTexture(accumTexSingle); 
  } else if  (typeid(spinorFloat) == typeid(float2)) {
    cudaUnbindTexture(spinorTexSingle2); 
    if (out) cudaUnbindTexture(interTexSingle2); 
    if (x) cudaUnbindTexture(accumTexSingle2); 
  } else if (typeid(spinorFloat) == typeid(short4)) {
    cudaUnbindTexture(spinorTexHalf); 
    cudaUnbindTexture(spinorTexHalfNorm);
    if (out) cudaUnbindTexture(interTexHalf); 
    if (out) cudaUnbindTexture(interTexHalfNorm);
    if (x) cudaUnbindTexture(accumTexHalf); 
    if (x) cudaUnbindTexture(accumTexHalfNorm);
  } else if (typeid(spinorFloat) == typeid(short2)) {
    cudaUnbindTexture(spinorTexHalf2); 
    cudaUnbindTexture(spinorTexHalf2Norm);
    if (out) cudaUnbindTexture(interTexHalf2); 
    if (out) cudaUnbindTexture(interTexHalf2Norm);
    if (x) cudaUnbindTexture(accumTexHalf2); 
    if (x) cudaUnbindTexture(accumTexHalf2Norm);
  } else {
    errorQuda("Unsupported precision and short vector type");
  }
#endif // USE_TEXTURE_OBJECTS
}

// Double precision clover term
texture<int4, 1> cloverTexDouble;

// Single precision clover term
texture<float4, 1, cudaReadModeElementType> cloverTexSingle;

// Half precision clover term
texture<short4, 1, cudaReadModeNormalizedFloat> cloverTexHalf;
texture<float, 1, cudaReadModeElementType> cloverTexNorm;

QudaPrecision bindCloverTex(const FullClover clover, const int oddBit, 
				   void **cloverP, void **cloverNormP)
{

  if (oddBit) {
    *cloverP = clover.odd;
    *cloverNormP = clover.oddNorm;
  } else {
    *cloverP = clover.even;
    *cloverNormP = clover.evenNorm;
  }

#ifdef USE_TEXTURE_OBJECTS
  dslashParam.cloverTex = oddBit ? clover.OddTex() : clover.EvenTex();
  if (clover.precision == QUDA_HALF_PRECISION) dslashParam.cloverNormTex = oddBit ? clover.OddNormTex() : clover.EvenNormTex();
#else
  if (clover.precision == QUDA_DOUBLE_PRECISION) {
    cudaBindTexture(0, cloverTexDouble, *cloverP, clover.bytes); 
  } else if (clover.precision == QUDA_SINGLE_PRECISION) {
    cudaBindTexture(0, cloverTexSingle, *cloverP, clover.bytes); 
  } else {
    cudaBindTexture(0, cloverTexHalf, *cloverP, clover.bytes); 
    cudaBindTexture(0, cloverTexNorm, *cloverNormP, clover.norm_bytes);
  }
#endif // USE_TEXTURE_OBJECTS

  return clover.precision;
}

void unbindCloverTex(const FullClover clover)
{
#if (!defined USE_TEXTURE_OBJECTS)
  if (clover.precision == QUDA_DOUBLE_PRECISION) {
    cudaUnbindTexture(cloverTexDouble);
  } else if (clover.precision == QUDA_SINGLE_PRECISION) {
    cudaUnbindTexture(cloverTexSingle);
  } else {
    cudaUnbindTexture(cloverTexHalf);
    cudaUnbindTexture(cloverTexNorm);
  }
#endif // not defined USE_TEXTURE_OBJECTS
}

