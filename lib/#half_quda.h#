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
