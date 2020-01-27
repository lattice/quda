#include <typeinfo>

template<typename Tex>
static __inline__ __device__ double fetch_double(Tex t, int i)
{
  int2 v = tex1Dfetch<int2>(t, i);
  return __hiloint2double(v.y, v.x);
}

template <typename Tex>
static __inline__ __device__ double2 fetch_double2(Tex t, int i)
{
  int4 v = tex1Dfetch<int4>(t, i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}


template<typename T>
void bindGaugeTex(const cudaGaugeField &gauge, const int oddBit, T &dslashParam)
{
  if(oddBit) {
    dslashParam.gauge0 = const_cast<void*>(gauge.Odd_p());
    dslashParam.gauge1 = const_cast<void*>(gauge.Even_p());
  } else {
    dslashParam.gauge0 = const_cast<void*>(gauge.Even_p());
    dslashParam.gauge1 = const_cast<void*>(gauge.Odd_p());
  }

#ifdef USE_TEXTURE_OBJECTS
  dslashParam.gauge0Tex = oddBit ? gauge.OddTex() : gauge.EvenTex();
  dslashParam.gauge1Tex = oddBit ? gauge.EvenTex() : gauge.OddTex();
#endif
}

void unbindGaugeTex(const cudaGaugeField &gauge) {}

template <typename T>
void bindFatGaugeTex(const cudaGaugeField &gauge, const int oddBit, T &dslashParam)
{
  if(oddBit) {
    dslashParam.gauge0 = const_cast<void*>(gauge.Odd_p());
    dslashParam.gauge1 = const_cast<void*>(gauge.Even_p());
  } else {
    dslashParam.gauge0 = const_cast<void*>(gauge.Even_p());
    dslashParam.gauge1 = const_cast<void*>(gauge.Odd_p());
  }

#ifdef USE_TEXTURE_OBJECTS
  dslashParam.gauge0Tex = oddBit ? gauge.OddTex() : gauge.EvenTex();
  dslashParam.gauge1Tex = oddBit ? gauge.EvenTex() : gauge.OddTex();
#endif // USE_TEXTURE_OBJECTS
}

void unbindFatGaugeTex(const cudaGaugeField &gauge) {}

template <typename T>
void bindLongGaugeTex(const cudaGaugeField &gauge, const int oddBit, T &dslashParam)
{
  if (oddBit) {
    dslashParam.gauge0 = const_cast<void*>(gauge.Odd_p());
    dslashParam.gauge1 = const_cast<void*>(gauge.Even_p());
  } else {
    dslashParam.gauge0 = const_cast<void*>(gauge.Even_p());
    dslashParam.gauge1 = const_cast<void*>(gauge.Odd_p());
  }

  dslashParam.longPhase0 = static_cast<char*>(dslashParam.longGauge0) + gauge.PhaseOffset();
  dslashParam.longPhase1 = static_cast<char*>(dslashParam.longGauge1) + gauge.PhaseOffset();

#ifdef USE_TEXTURE_OBJECTS
  dslashParam.longGauge0Tex = oddBit ? gauge.OddTex() : gauge.EvenTex();
  dslashParam.longGauge1Tex = oddBit ? gauge.EvenTex() : gauge.OddTex();

  if(gauge.Reconstruct() == QUDA_RECONSTRUCT_13 || gauge.Reconstruct() == QUDA_RECONSTRUCT_9){
    dslashParam.longPhase0Tex = oddBit ? gauge.OddPhaseTex() : gauge.EvenPhaseTex();
    dslashParam.longPhase1Tex = oddBit ? gauge.EvenPhaseTex() : gauge.OddPhaseTex();
  }
#endif // USE_TEXTURE_OBJECTS
}

void unbindLongGaugeTex(const cudaGaugeField &gauge) {}

template <typename spinorFloat>
int bindSpinorTex(const cudaColorSpinorField *in, const cudaColorSpinorField *out=0,
		  const cudaColorSpinorField *x=0) {
  int size = (sizeof(((spinorFloat*)0)->x) < sizeof(float)) ? sizeof(float) :
    sizeof(((spinorFloat*)0)->x);

  return size;
}

template <typename spinorFloat>
void unbindSpinorTex(
    const cudaColorSpinorField *in, const cudaColorSpinorField *out = 0, const cudaColorSpinorField *x = 0)
{
}

template <typename T>
QudaPrecision bindCloverTex(const FullClover &clover, const int oddBit, T &dslashParam)
{
  if (oddBit) {
    dslashParam.clover = clover.odd;
    dslashParam.cloverNorm = (float*)clover.oddNorm;
  } else {
    dslashParam.clover = clover.even;
    dslashParam.cloverNorm = (float*)clover.evenNorm;
  }

#ifdef USE_TEXTURE_OBJECTS
  dslashParam.cloverTex = oddBit ? clover.OddTex() : clover.EvenTex();
  if (clover.precision == QUDA_HALF_PRECISION || clover.precision == QUDA_QUARTER_PRECISION) dslashParam.cloverNormTex = oddBit ? clover.OddNormTex() : clover.EvenNormTex();
#endif // USE_TEXTURE_OBJECTS

  return clover.precision;
}

void unbindCloverTex(const FullClover clover) {}

template <typename T>
QudaPrecision bindTwistedCloverTex(const FullClover clover, const FullClover cloverInv, const int oddBit, T &dslashParam)
{
  if (oddBit) {
    dslashParam.clover	 = clover.odd;
    dslashParam.cloverNorm = (float*)clover.oddNorm;
    if (!dynamic_clover_inverse()) {
      dslashParam.cloverInv = cloverInv.odd;
      dslashParam.cloverInvNorm = (float*)cloverInv.oddNorm;
    }
  } else {
    dslashParam.clover = clover.even;
    dslashParam.cloverNorm = (float*)clover.evenNorm;
    if (!dynamic_clover_inverse()) {
      dslashParam.clover = cloverInv.even;
      dslashParam.cloverInvNorm = (float*)cloverInv.evenNorm;
    }
  }

#ifdef USE_TEXTURE_OBJECTS
  dslashParam.cloverTex = oddBit ? clover.OddTex() : clover.EvenTex();
  if (clover.precision == QUDA_HALF_PRECISION || clover.precision == QUDA_QUARTER_PRECISION) dslashParam.cloverNormTex = oddBit ? clover.OddNormTex() : clover.EvenNormTex();
  if (!dynamic_clover_inverse()) {
    dslashParam.cloverInvTex = oddBit ? cloverInv.OddTex() : cloverInv.EvenTex();
    if (cloverInv.precision == QUDA_HALF_PRECISION || clover.precision == QUDA_QUARTER_PRECISION) dslashParam.cloverInvNormTex = oddBit ? cloverInv.OddNormTex() : cloverInv.EvenNormTex();
  }
#endif // USE_TEXTURE_OBJECTS

  return clover.precision;
}

void unbindTwistedCloverTex(const FullClover clover) {}

// define some function if we're not using textures (direct access)
#if defined(DIRECT_ACCESS_LINK) || defined(DIRECT_ACCESS_WILSON_SPINOR) || \
  defined(DIRECT_ACCESS_WILSON_ACCUM) || defined(DIRECT_ACCESS_WILSON_PACK_SPINOR) || \
  defined(DIRECT_ACCESS_WILSON_INTER) || defined(DIRECT_ACCESS_WILSON_PACK_SPINOR) || \
  defined(DIRECT_ACCESS_CLOVER) || defined(DIRECT_ACCESS_PACK) ||       \
  defined(DIRECT_ACCESS_LONG_LINK) || defined(DIRECT_ACCESS_FAT_LINK)

  // Half precision
  static inline __device__ float short2float(short a) {
    return (float)a/fixedMaxValue<short>::value;
  }

  static inline __device__ short float2short(float c, float a) {
    return (short)(a*c*fixedMaxValue<short>::value);
  }

  static inline __device__ short4 float42short4(float c, float4 a) {
    return make_short4(float2short(c, a.x), float2short(c, a.y), float2short(c, a.z), float2short(c, a.w));
  }

  static inline __device__ float4 short42float4(short4 a) {
    return make_float4(short2float(a.x), short2float(a.y), short2float(a.z), short2float(a.w));
  }

  static inline __device__ float2 short22float2(short2 a) {
    return make_float2(short2float(a.x), short2float(a.y));
  }

  // Quarter precision
  static inline __device__ float char2float(char a) {
    return (float)a/fixedMaxValue<char>::value;
  }

  static inline __device__ char float2char(float c, float a) {
    return (char)(a*c*fixedMaxValue<char>::value);
  }

  static inline __device__ char4 float42char4(float c, float4 a) {
    return make_char4(float2char(c, a.x), float2char(c, a.y), float2char(c, a.z), float2char(c, a.w));
  }

  static inline __device__ float4 char42float4(char4 a) {
    return make_float4(char2float(a.x), char2float(a.y), char2float(a.z), char2float(a.w));
  }

  static inline __device__ float2 char22float2(char2 a) {
    return make_float2(char2float(a.x), char2float(a.y));
  }
#endif // DIRECT_ACCESS inclusions
