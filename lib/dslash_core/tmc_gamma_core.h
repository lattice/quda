#ifndef _TMC_GAMMA_CORE_H
#define _TMC_GAMMA_CORE_H

//action of the operator b*(1 + i*a*gamma5)
//used also macros from io_spinor.h
/*
__device__ float4 operator*(const float &x, const float4 &y) 
{
  float4 res;

  res.x = x * y.x;
  res.y = x * y.y;  
  res.z = x * y.z;
  res.w = x * y.w;  

  return res;
}

__device__ double2 operator*(const double &x, const double2 &y) 
{
  double2 res;

  res.x = x * y.x;
  res.y = x * y.y;  

  return res;
}
*/

#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexDouble
#endif


// first chiral block of clover term
//Double precision
#define cd00_00_re C0.x
#define cd01_01_re C0.y
#define cd02_02_re C1.x
#define cd10_10_re C1.y
#define cd11_11_re C2.x
#define cd12_12_re C2.y
#define cd01_00_re C3.x
#define cd01_00_im C3.y
#define cd02_00_re C4.x
#define cd02_00_im C4.y
#define cd10_00_re C5.x
#define cd10_00_im C5.y
#define cd11_00_re C6.x
#define cd11_00_im C6.y
#define cd12_00_re C7.x
#define cd12_00_im C7.y
#define cd02_01_re C8.x
#define cd02_01_im C8.y
#define cd10_01_re C9.x
#define cd10_01_im C9.y
#define cd11_01_re C10.x
#define cd11_01_im C10.y
#define cd12_01_re C11.x
#define cd12_01_im C11.y
#define cd10_02_re C12.x
#define cd10_02_im C12.y
#define cd11_02_re C13.x
#define cd11_02_im C13.y
#define cd12_02_re C14.x
#define cd12_02_im C14.y
#define cd11_10_re C15.x
#define cd11_10_im C15.y
#define cd12_10_re C16.x
#define cd12_10_im C16.y
#define cd12_11_re C17.x
#define cd12_11_im C17.y

#define cd00_01_re (+cd01_00_re)
#define cd00_01_im (-cd01_00_im)
#define cd00_02_re (+cd02_00_re)
#define cd00_02_im (-cd02_00_im)
#define cd01_02_re (+cd02_01_re)
#define cd01_02_im (-cd02_01_im)
#define cd00_10_re (+cd10_00_re)
#define cd00_10_im (-cd10_00_im)
#define cd01_10_re (+cd10_01_re)
#define cd01_10_im (-cd10_01_im)
#define cd02_10_re (+cd10_02_re)
#define cd02_10_im (-cd10_02_im)
#define cd00_11_re (+cd11_00_re)
#define cd00_11_im (-cd11_00_im)
#define cd01_11_re (+cd11_01_re)
#define cd01_11_im (-cd11_01_im)
#define cd02_11_re (+cd11_02_re)
#define cd02_11_im (-cd11_02_im)
#define cd10_11_re (+cd11_10_re)
#define cd10_11_im (-cd11_10_im)
#define cd00_12_re (+cd12_00_re)
#define cd00_12_im (-cd12_00_im)
#define cd01_12_re (+cd12_01_re)
#define cd01_12_im (-cd12_01_im)
#define cd02_12_re (+cd12_02_re)
#define cd02_12_im (-cd12_02_im)
#define cd10_12_re (+cd12_10_re)
#define cd10_12_im (-cd12_10_im)
#define cd11_12_re (+cd12_11_re)
#define cd11_12_im (-cd12_11_im)

// second chiral block of clover term (reuses C0,...,C9)
#define cd20_20_re cd00_00_re
#define cd21_20_re cd01_00_re
#define cd21_20_im cd01_00_im
#define cd22_20_re cd02_00_re
#define cd22_20_im cd02_00_im
#define cd30_20_re cd10_00_re
#define cd30_20_im cd10_00_im
#define cd31_20_re cd11_00_re
#define cd31_20_im cd11_00_im
#define cd32_20_re cd12_00_re
#define cd32_20_im cd12_00_im
#define cd20_21_re cd00_01_re
#define cd20_21_im cd00_01_im
#define cd21_21_re cd01_01_re
#define cd22_21_re cd02_01_re
#define cd22_21_im cd02_01_im
#define cd30_21_re cd10_01_re
#define cd30_21_im cd10_01_im
#define cd31_21_re cd11_01_re
#define cd31_21_im cd11_01_im
#define cd32_21_re cd12_01_re
#define cd32_21_im cd12_01_im
#define cd20_22_re cd00_02_re
#define cd20_22_im cd00_02_im
#define cd21_22_re cd01_02_re
#define cd21_22_im cd01_02_im
#define cd22_22_re cd02_02_re
#define cd30_22_re cd10_02_re
#define cd30_22_im cd10_02_im
#define cd31_22_re cd11_02_re
#define cd31_22_im cd11_02_im
#define cd32_22_re cd12_02_re
#define cd32_22_im cd12_02_im
#define cd20_30_re cd00_10_re
#define cd20_30_im cd00_10_im
#define cd21_30_re cd01_10_re
#define cd21_30_im cd01_10_im
#define cd22_30_re cd02_10_re
#define cd22_30_im cd02_10_im
#define cd30_30_re cd10_10_re
#define cd31_30_re cd11_10_re
#define cd31_30_im cd11_10_im
#define cd32_30_re cd12_10_re
#define cd32_30_im cd12_10_im
#define cd20_31_re cd00_11_re
#define cd20_31_im cd00_11_im
#define cd21_31_re cd01_11_re
#define cd21_31_im cd01_11_im
#define cd22_31_re cd02_11_re
#define cd22_31_im cd02_11_im
#define cd30_31_re cd10_11_re
#define cd30_31_im cd10_11_im
#define cd31_31_re cd11_11_re
#define cd32_31_re cd12_11_re
#define cd32_31_im cd12_11_im
#define cd20_32_re cd00_12_re
#define cd20_32_im cd00_12_im
#define cd21_32_re cd01_12_re
#define cd21_32_im cd01_12_im
#define cd22_32_re cd02_12_re
#define cd22_32_im cd02_12_im
#define cd30_32_re cd10_12_re
#define cd30_32_im cd10_12_im
#define cd31_32_re cd11_12_re
#define cd31_32_im cd11_12_im
#define cd32_32_re cd12_12_re

//Single-half precision
#define c00_00_re C0.x
#define c01_01_re C0.y
#define c02_02_re C0.z
#define c10_10_re C0.w
#define c11_11_re C1.x
#define c12_12_re C1.y
#define c01_00_re C1.z
#define c01_00_im C1.w
#define c02_00_re C2.x
#define c02_00_im C2.y
#define c10_00_re C2.z
#define c10_00_im C2.w
#define c11_00_re C3.x
#define c11_00_im C3.y
#define c12_00_re C3.z
#define c12_00_im C3.w
#define c02_01_re C4.x
#define c02_01_im C4.y
#define c10_01_re C4.z
#define c10_01_im C4.w
#define c11_01_re C5.x
#define c11_01_im C5.y
#define c12_01_re C5.z
#define c12_01_im C5.w
#define c10_02_re C6.x
#define c10_02_im C6.y
#define c11_02_re C6.z
#define c11_02_im C6.w
#define c12_02_re C7.x
#define c12_02_im C7.y
#define c11_10_re C7.z
#define c11_10_im C7.w
#define c12_10_re C8.x
#define c12_10_im C8.y
#define c12_11_re C8.z
#define c12_11_im C8.w

#define c00_01_re (+c01_00_re)
#define c00_01_im (-c01_00_im)
#define c00_02_re (+c02_00_re)
#define c00_02_im (-c02_00_im)
#define c01_02_re (+c02_01_re)
#define c01_02_im (-c02_01_im)
#define c00_10_re (+c10_00_re)
#define c00_10_im (-c10_00_im)
#define c01_10_re (+c10_01_re)
#define c01_10_im (-c10_01_im)
#define c02_10_re (+c10_02_re)
#define c02_10_im (-c10_02_im)
#define c00_11_re (+c11_00_re)
#define c00_11_im (-c11_00_im)
#define c01_11_re (+c11_01_re)
#define c01_11_im (-c11_01_im)
#define c02_11_re (+c11_02_re)
#define c02_11_im (-c11_02_im)
#define c10_11_re (+c11_10_re)
#define c10_11_im (-c11_10_im)
#define c00_12_re (+c12_00_re)
#define c00_12_im (-c12_00_im)
#define c01_12_re (+c12_01_re)
#define c01_12_im (-c12_01_im)
#define c02_12_re (+c12_02_re)
#define c02_12_im (-c12_02_im)
#define c10_12_re (+c12_10_re)
#define c10_12_im (-c12_10_im)
#define c11_12_re (+c12_11_re)
#define c11_12_im (-c12_11_im)

// second chiral block of clover term (reuses C0,...,C9)
#define c20_20_re c00_00_re
#define c21_20_re c01_00_re
#define c21_20_im c01_00_im
#define c22_20_re c02_00_re
#define c22_20_im c02_00_im
#define c30_20_re c10_00_re
#define c30_20_im c10_00_im
#define c31_20_re c11_00_re
#define c31_20_im c11_00_im
#define c32_20_re c12_00_re
#define c32_20_im c12_00_im
#define c20_21_re c00_01_re
#define c20_21_im c00_01_im
#define c21_21_re c01_01_re
#define c22_21_re c02_01_re
#define c22_21_im c02_01_im
#define c30_21_re c10_01_re
#define c30_21_im c10_01_im
#define c31_21_re c11_01_re
#define c31_21_im c11_01_im
#define c32_21_re c12_01_re
#define c32_21_im c12_01_im
#define c20_22_re c00_02_re
#define c20_22_im c00_02_im
#define c21_22_re c01_02_re
#define c21_22_im c01_02_im
#define c22_22_re c02_02_re
#define c30_22_re c10_02_re
#define c30_22_im c10_02_im
#define c31_22_re c11_02_re
#define c31_22_im c11_02_im
#define c32_22_re c12_02_re
#define c32_22_im c12_02_im
#define c20_30_re c00_10_re
#define c20_30_im c00_10_im
#define c21_30_re c01_10_re
#define c21_30_im c01_10_im
#define c22_30_re c02_10_re
#define c22_30_im c02_10_im
#define c30_30_re c10_10_re
#define c31_30_re c11_10_re
#define c31_30_im c11_10_im
#define c32_30_re c12_10_re
#define c32_30_im c12_10_im
#define c20_31_re c00_11_re
#define c20_31_im c00_11_im
#define c21_31_re c01_11_re
#define c21_31_im c01_11_im
#define c22_31_re c02_11_re
#define c22_31_im c02_11_im
#define c30_31_re c10_11_re
#define c30_31_im c10_11_im
#define c31_31_re c11_11_re
#define c32_31_re c12_11_re
#define c32_31_im c12_11_im
#define c20_32_re c00_12_re
#define c20_32_im c00_12_im
#define c21_32_re c01_12_re
#define c21_32_im c01_12_im
#define c22_32_re c02_12_re
#define c22_32_im c02_12_im
#define c30_32_re c10_12_re
#define c30_32_im c10_12_im
#define c31_32_re c11_12_re
#define c31_32_im c11_12_im
#define c32_32_re c12_12_re

// first chiral block of inverted clover term (reuses C0,...,C9)
//Double-precision
#define cdinv00_00_re C0.x
#define cdinv01_01_re C0.y
#define cdinv02_02_re C1.x
#define cdinv10_10_re C1.y
#define cdinv11_11_re C2.x
#define cdinv12_12_re C2.y
#define cdinv01_00_re C3.x
#define cdinv01_00_im C3.y
#define cdinv02_00_re C4.x
#define cdinv02_00_im C4.y
#define cdinv10_00_re C5.x
#define cdinv10_00_im C5.y
#define cdinv11_00_re C6.x
#define cdinv11_00_im C6.y
#define cdinv12_00_re C7.x
#define cdinv12_00_im C7.y
#define cdinv02_01_re C8.x
#define cdinv02_01_im C8.y
#define cdinv10_01_re C9.x
#define cdinv10_01_im C9.y
#define cdinv11_01_re C10.x
#define cdinv11_01_im C10.y
#define cdinv12_01_re C11.x
#define cdinv12_01_im C11.y
#define cdinv10_02_re C12.x
#define cdinv10_02_im C12.y
#define cdinv11_02_re C13.x
#define cdinv11_02_im C13.y
#define cdinv12_02_re C14.x
#define cdinv12_02_im C14.y
#define cdinv11_10_re C15.x
#define cdinv11_10_im C15.y
#define cdinv12_10_re C16.x
#define cdinv12_10_im C16.y
#define cdinv12_11_re C17.x
#define cdinv12_11_im C17.y

#define cdinv00_01_re (+cdinv01_00_re)
#define cdinv00_01_im (-cdinv01_00_im)
#define cdinv00_02_re (+cdinv02_00_re)
#define cdinv00_02_im (-cdinv02_00_im)
#define cdinv01_02_re (+cdinv02_01_re)
#define cdinv01_02_im (-cdinv02_01_im)
#define cdinv00_10_re (+cdinv10_00_re)
#define cdinv00_10_im (-cdinv10_00_im)
#define cdinv01_10_re (+cdinv10_01_re)
#define cdinv01_10_im (-cdinv10_01_im)
#define cdinv02_10_re (+cdinv10_02_re)
#define cdinv02_10_im (-cdinv10_02_im)
#define cdinv00_11_re (+cdinv11_00_re)
#define cdinv00_11_im (-cdinv11_00_im)
#define cdinv01_11_re (+cdinv11_01_re)
#define cdinv01_11_im (-cdinv11_01_im)
#define cdinv02_11_re (+cdinv11_02_re)
#define cdinv02_11_im (-cdinv11_02_im)
#define cdinv10_11_re (+cdinv11_10_re)
#define cdinv10_11_im (-cdinv11_10_im)
#define cdinv00_12_re (+cdinv12_00_re)
#define cdinv00_12_im (-cdinv12_00_im)
#define cdinv01_12_re (+cdinv12_01_re)
#define cdinv01_12_im (-cdinv12_01_im)
#define cdinv02_12_re (+cdinv12_02_re)
#define cdinv02_12_im (-cdinv12_02_im)
#define cdinv10_12_re (+cdinv12_10_re)
#define cdinv10_12_im (-cdinv12_10_im)
#define cdinv11_12_re (+cdinv12_11_re)
#define cdinv11_12_im (-cdinv12_11_im)

// second chiral block of inverted clover term (reuses C0,...,C9)
#define cdinv20_20_re cdinv00_00_re
#define cdinv21_20_re cdinv01_00_re
#define cdinv21_20_im cdinv01_00_im
#define cdinv22_20_re cdinv02_00_re
#define cdinv22_20_im cdinv02_00_im
#define cdinv30_20_re cdinv10_00_re
#define cdinv30_20_im cdinv10_00_im
#define cdinv31_20_re cdinv11_00_re
#define cdinv31_20_im cdinv11_00_im
#define cdinv32_20_re cdinv12_00_re
#define cdinv32_20_im cdinv12_00_im
#define cdinv20_21_re cdinv00_01_re
#define cdinv20_21_im cdinv00_01_im
#define cdinv21_21_re cdinv01_01_re
#define cdinv22_21_re cdinv02_01_re
#define cdinv22_21_im cdinv02_01_im
#define cdinv30_21_re cdinv10_01_re
#define cdinv30_21_im cdinv10_01_im
#define cdinv31_21_re cdinv11_01_re
#define cdinv31_21_im cdinv11_01_im
#define cdinv32_21_re cdinv12_01_re
#define cdinv32_21_im cdinv12_01_im
#define cdinv20_22_re cdinv00_02_re
#define cdinv20_22_im cdinv00_02_im
#define cdinv21_22_re cdinv01_02_re
#define cdinv21_22_im cdinv01_02_im
#define cdinv22_22_re cdinv02_02_re
#define cdinv30_22_re cdinv10_02_re
#define cdinv30_22_im cdinv10_02_im
#define cdinv31_22_re cdinv11_02_re
#define cdinv31_22_im cdinv11_02_im
#define cdinv32_22_re cdinv12_02_re
#define cdinv32_22_im cdinv12_02_im
#define cdinv20_30_re cdinv00_10_re
#define cdinv20_30_im cdinv00_10_im
#define cdinv21_30_re cdinv01_10_re
#define cdinv21_30_im cdinv01_10_im
#define cdinv22_30_re cdinv02_10_re
#define cdinv22_30_im cdinv02_10_im
#define cdinv30_30_re cdinv10_10_re
#define cdinv31_30_re cdinv11_10_re
#define cdinv31_30_im cdinv11_10_im
#define cdinv32_30_re cdinv12_10_re
#define cdinv32_30_im cdinv12_10_im
#define cdinv20_31_re cdinv00_11_re
#define cdinv20_31_im cdinv00_11_im
#define cdinv21_31_re cdinv01_11_re
#define cdinv21_31_im cdinv01_11_im
#define cdinv22_31_re cdinv02_11_re
#define cdinv22_31_im cdinv02_11_im
#define cdinv30_31_re cdinv10_11_re
#define cdinv30_31_im cdinv10_11_im
#define cdinv31_31_re cdinv11_11_re
#define cdinv32_31_re cdinv12_11_re
#define cdinv32_31_im cdinv12_11_im
#define cdinv20_32_re cdinv00_12_re
#define cdinv20_32_im cdinv00_12_im
#define cdinv21_32_re cdinv01_12_re
#define cdinv21_32_im cdinv01_12_im
#define cdinv22_32_re cdinv02_12_re
#define cdinv22_32_im cdinv02_12_im
#define cdinv30_32_re cdinv10_12_re
#define cdinv30_32_im cdinv10_12_im
#define cdinv31_32_re cdinv11_12_re
#define cdinv31_32_im cdinv11_12_im
#define cdinv32_32_re cdinv12_12_re

//Single-half precision
#define cinv00_00_re C0.x
#define cinv01_01_re C0.y
#define cinv02_02_re C0.z
#define cinv10_10_re C0.w
#define cinv11_11_re C1.x
#define cinv12_12_re C1.y
#define cinv01_00_re C1.z
#define cinv01_00_im C1.w
#define cinv02_00_re C2.x
#define cinv02_00_im C2.y
#define cinv10_00_re C2.z
#define cinv10_00_im C2.w
#define cinv11_00_re C3.x
#define cinv11_00_im C3.y
#define cinv12_00_re C3.z
#define cinv12_00_im C3.w
#define cinv02_01_re C4.x
#define cinv02_01_im C4.y
#define cinv10_01_re C4.z
#define cinv10_01_im C4.w
#define cinv11_01_re C5.x
#define cinv11_01_im C5.y
#define cinv12_01_re C5.z
#define cinv12_01_im C5.w
#define cinv10_02_re C6.x
#define cinv10_02_im C6.y
#define cinv11_02_re C6.z
#define cinv11_02_im C6.w
#define cinv12_02_re C7.x
#define cinv12_02_im C7.y
#define cinv11_10_re C7.z
#define cinv11_10_im C7.w
#define cinv12_10_re C8.x
#define cinv12_10_im C8.y
#define cinv12_11_re C8.z
#define cinv12_11_im C8.w

#define cinv00_01_re (+cinv01_00_re)
#define cinv00_01_im (-cinv01_00_im)
#define cinv00_02_re (+cinv02_00_re)
#define cinv00_02_im (-cinv02_00_im)
#define cinv01_02_re (+cinv02_01_re)
#define cinv01_02_im (-cinv02_01_im)
#define cinv00_10_re (+cinv10_00_re)
#define cinv00_10_im (-cinv10_00_im)
#define cinv01_10_re (+cinv10_01_re)
#define cinv01_10_im (-cinv10_01_im)
#define cinv02_10_re (+cinv10_02_re)
#define cinv02_10_im (-cinv10_02_im)
#define cinv00_11_re (+cinv11_00_re)
#define cinv00_11_im (-cinv11_00_im)
#define cinv01_11_re (+cinv11_01_re)
#define cinv01_11_im (-cinv11_01_im)
#define cinv02_11_re (+cinv11_02_re)
#define cinv02_11_im (-cinv11_02_im)
#define cinv10_11_re (+cinv11_10_re)
#define cinv10_11_im (-cinv11_10_im)
#define cinv00_12_re (+cinv12_00_re)
#define cinv00_12_im (-cinv12_00_im)
#define cinv01_12_re (+cinv12_01_re)
#define cinv01_12_im (-cinv12_01_im)
#define cinv02_12_re (+cinv12_02_re)
#define cinv02_12_im (-cinv12_02_im)
#define cinv10_12_re (+cinv12_10_re)
#define cinv10_12_im (-cinv12_10_im)
#define cinv11_12_re (+cinv12_11_re)
#define cinv11_12_im (-cinv12_11_im)

// second chiral block of inverted clover term (reuses C0,...,C9)
#define cinv20_20_re cinv00_00_re
#define cinv21_20_re cinv01_00_re
#define cinv21_20_im cinv01_00_im
#define cinv22_20_re cinv02_00_re
#define cinv22_20_im cinv02_00_im
#define cinv30_20_re cinv10_00_re
#define cinv30_20_im cinv10_00_im
#define cinv31_20_re cinv11_00_re
#define cinv31_20_im cinv11_00_im
#define cinv32_20_re cinv12_00_re
#define cinv32_20_im cinv12_00_im
#define cinv20_21_re cinv00_01_re
#define cinv20_21_im cinv00_01_im
#define cinv21_21_re cinv01_01_re
#define cinv22_21_re cinv02_01_re
#define cinv22_21_im cinv02_01_im
#define cinv30_21_re cinv10_01_re
#define cinv30_21_im cinv10_01_im
#define cinv31_21_re cinv11_01_re
#define cinv31_21_im cinv11_01_im
#define cinv32_21_re cinv12_01_re
#define cinv32_21_im cinv12_01_im
#define cinv20_22_re cinv00_02_re
#define cinv20_22_im cinv00_02_im
#define cinv21_22_re cinv01_02_re
#define cinv21_22_im cinv01_02_im
#define cinv22_22_re cinv02_02_re
#define cinv30_22_re cinv10_02_re
#define cinv30_22_im cinv10_02_im
#define cinv31_22_re cinv11_02_re
#define cinv31_22_im cinv11_02_im
#define cinv32_22_re cinv12_02_re
#define cinv32_22_im cinv12_02_im
#define cinv20_30_re cinv00_10_re
#define cinv20_30_im cinv00_10_im
#define cinv21_30_re cinv01_10_re
#define cinv21_30_im cinv01_10_im
#define cinv22_30_re cinv02_10_re
#define cinv22_30_im cinv02_10_im
#define cinv30_30_re cinv10_10_re
#define cinv31_30_re cinv11_10_re
#define cinv31_30_im cinv11_10_im
#define cinv32_30_re cinv12_10_re
#define cinv32_30_im cinv12_10_im
#define cinv20_31_re cinv00_11_re
#define cinv20_31_im cinv00_11_im
#define cinv21_31_re cinv01_11_re
#define cinv21_31_im cinv01_11_im
#define cinv22_31_re cinv02_11_re
#define cinv22_31_im cinv02_11_im
#define cinv30_31_re cinv10_11_re
#define cinv30_31_im cinv10_11_im
#define cinv31_31_re cinv11_11_re
#define cinv32_31_re cinv12_11_re
#define cinv32_31_im cinv12_11_im
#define cinv20_32_re cinv00_12_re
#define cinv20_32_im cinv00_12_im
#define cinv21_32_re cinv01_12_re
#define cinv21_32_im cinv01_12_im
#define cinv22_32_re cinv02_12_re
#define cinv22_32_im cinv02_12_im
#define cinv30_32_re cinv10_12_re
#define cinv30_32_im cinv10_12_im
#define cinv31_32_re cinv11_12_re
#define cinv31_32_im cinv11_12_im
#define cinv32_32_re cinv12_12_re

#if (__COMPUTE_CAPABILITY__ >= 130)

#define S00_re	I0.x
#define S00_im	I0.y
#define S01_re	I1.x
#define S01_im	I1.y
#define S02_re	I2.x
#define S02_im	I2.y
#define S10_re	I3.x
#define S10_im	I3.y
#define S11_re	I4.x
#define S11_im	I4.y
#define S12_re	I5.x
#define S12_im	I5.y
#define S20_re	I6.x
#define S20_im	I6.y
#define S21_re	I7.x
#define S21_im	I7.y
#define S22_re	I8.x
#define S22_im	I8.y
#define S30_re	I9.x
#define S30_im	I9.y
#define S31_re	I10.x
#define S31_im	I10.y
#define S32_re	I11.x
#define S32_im	I11.y
#define spinorFloat double

#if (defined DIRECT_ACCESS_CLOVER) || (defined FERMI_NO_DBLE_TEX)
	#define TMCLOVERTEX clover
	#define TM_INV_CLOVERTEX cloverInv
	#define READ_CLOVER READ_CLOVER_DOUBLE_STR
	#define ASSN_CLOVER ASSN_CLOVER_DOUBLE_STR
#else
	#ifdef USE_TEXTURE_OBJECTS
		#define TMCLOVERTEX (param.cloverTex)
		#define TM_INV_CLOVERTEX (param.cloverInvTex)
	#else
		#define TMCLOVERTEX cloverTexDouble
		#define TM_INV_CLOVERTEX cloverInvTexDouble
	#endif
	#define READ_CLOVER READ_CLOVER_DOUBLE_TEX
	#define ASSN_CLOVER ASSN_CLOVER_DOUBLE_TEX
#endif

#define CLOVER_DOUBLE

__global__ void twistCloverGamma5Kernel(double2 *spinor, float *null, double a, const double2 *in, const float *null2, DslashParam param,
					 const double2 *clover, const float *cNorm, const double2 *cloverInv, const float *cNrm2)
{
#ifdef GPU_TWISTED_CLOVER_DIRAC

   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

#ifndef FERMI_NO_DBLE_TEX
   double2 I0  = fetch_double2(SPINORTEX, sid + 0 * sp_stride);   
   double2 I1  = fetch_double2(SPINORTEX, sid + 1 * sp_stride);   
   double2 I2  = fetch_double2(SPINORTEX, sid + 2 * sp_stride);   
   double2 I3  = fetch_double2(SPINORTEX, sid + 3 * sp_stride);   
   double2 I4  = fetch_double2(SPINORTEX, sid + 4 * sp_stride);   
   double2 I5  = fetch_double2(SPINORTEX, sid + 5 * sp_stride);   
   double2 I6  = fetch_double2(SPINORTEX, sid + 6 * sp_stride);   
   double2 I7  = fetch_double2(SPINORTEX, sid + 7 * sp_stride);   
   double2 I8  = fetch_double2(SPINORTEX, sid + 8 * sp_stride);   
   double2 I9  = fetch_double2(SPINORTEX, sid + 9 * sp_stride);   
   double2 I10 = fetch_double2(SPINORTEX, sid + 10 * sp_stride); 
   double2 I11 = fetch_double2(SPINORTEX, sid + 11 * sp_stride);
#else
   double2 I0  = in[sid + 0 * sp_stride];   
   double2 I1  = in[sid + 1 * sp_stride];   
   double2 I2  = in[sid + 2 * sp_stride];   
   double2 I3  = in[sid + 3 * sp_stride];   
   double2 I4  = in[sid + 4 * sp_stride];   
   double2 I5  = in[sid + 5 * sp_stride];   
   double2 I6  = in[sid + 6 * sp_stride];   
   double2 I7  = in[sid + 7 * sp_stride];   
   double2 I8  = in[sid + 8 * sp_stride];   
   double2 I9  = in[sid + 9 * sp_stride];   
   double2 I10 = in[sid + 10 * sp_stride]; 
   double2 I11 = in[sid + 11 * sp_stride];
#endif

   double2 C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17;

   //apply (Clover + i*a*gamma_5) to the input spinor
   APPLY_CLOVER_TWIST(cd, a, S);
      
   spinor[sid + 0  * sp_stride] = I0;   
   spinor[sid + 1  * sp_stride] = I1;   
   spinor[sid + 2  * sp_stride] = I2;   
   spinor[sid + 3  * sp_stride] = I3;   
   spinor[sid + 4  * sp_stride] = I4;   
   spinor[sid + 5  * sp_stride] = I5;   
   spinor[sid + 6  * sp_stride] = I6;   
   spinor[sid + 7  * sp_stride] = I7;   
   spinor[sid + 8  * sp_stride] = I8;   
   spinor[sid + 9  * sp_stride] = I9;   
   spinor[sid + 10 * sp_stride] = I10;   
   spinor[sid + 11 * sp_stride] = I11;

#endif
}

__global__ void twistCloverGamma5InvKernel(double2 *spinor, float *null, double a, const double2 *in, const float *null2, DslashParam param,
					   const double2 *clover, const float *cNorm, const double2 *cloverInv, const float *cNrm2)
{
#ifdef GPU_TWISTED_CLOVER_DIRAC

   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

#ifndef FERMI_NO_DBLE_TEX
   double2 I0  = fetch_double2(SPINORTEX, sid + 0 * sp_stride);   
   double2 I1  = fetch_double2(SPINORTEX, sid + 1 * sp_stride);   
   double2 I2  = fetch_double2(SPINORTEX, sid + 2 * sp_stride);   
   double2 I3  = fetch_double2(SPINORTEX, sid + 3 * sp_stride);   
   double2 I4  = fetch_double2(SPINORTEX, sid + 4 * sp_stride);   
   double2 I5  = fetch_double2(SPINORTEX, sid + 5 * sp_stride);   
   double2 I6  = fetch_double2(SPINORTEX, sid + 6 * sp_stride);   
   double2 I7  = fetch_double2(SPINORTEX, sid + 7 * sp_stride);   
   double2 I8  = fetch_double2(SPINORTEX, sid + 8 * sp_stride);   
   double2 I9  = fetch_double2(SPINORTEX, sid + 9 * sp_stride);   
   double2 I10 = fetch_double2(SPINORTEX, sid + 10 * sp_stride); 
   double2 I11 = fetch_double2(SPINORTEX, sid + 11 * sp_stride);
#else
   double2 I0  = in[sid + 0 * sp_stride];   
   double2 I1  = in[sid + 1 * sp_stride];   
   double2 I2  = in[sid + 2 * sp_stride];   
   double2 I3  = in[sid + 3 * sp_stride];   
   double2 I4  = in[sid + 4 * sp_stride];   
   double2 I5  = in[sid + 5 * sp_stride];   
   double2 I6  = in[sid + 6 * sp_stride];   
   double2 I7  = in[sid + 7 * sp_stride];   
   double2 I8  = in[sid + 8 * sp_stride];   
   double2 I9  = in[sid + 9 * sp_stride];   
   double2 I10 = in[sid + 10 * sp_stride]; 
   double2 I11 = in[sid + 11 * sp_stride];
#endif

   double2 C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17;

   //apply (Clover + i*a*gamma_5)/(Clover^2 + a^2) to the input spinor
   APPLY_CLOVER_TWIST_INV(cd, cdinv, a, S);
      
   spinor[sid + 0  * sp_stride] = I0;   
   spinor[sid + 1  * sp_stride] = I1;   
   spinor[sid + 2  * sp_stride] = I2;   
   spinor[sid + 3  * sp_stride] = I3;   
   spinor[sid + 4  * sp_stride] = I4;   
   spinor[sid + 5  * sp_stride] = I5;   
   spinor[sid + 6  * sp_stride] = I6;   
   spinor[sid + 7  * sp_stride] = I7;   
   spinor[sid + 8  * sp_stride] = I8;   
   spinor[sid + 9  * sp_stride] = I9;   
   spinor[sid + 10 * sp_stride] = I10;   
   spinor[sid + 11 * sp_stride] = I11;

#endif
}

#undef TMCLOVERTEX
#undef TM_INV_CLOVERTEX
#undef READ_CLOVER
#undef ASSN_CLOVER
#undef CLOVER_DOUBLE

#undef S00_re
#undef S00_im
#undef S01_re
#undef S01_im
#undef S02_re
#undef S02_im
#undef S10_re
#undef S10_im
#undef S11_re
#undef S11_im
#undef S12_re
#undef S12_im
#undef S20_re
#undef S20_im
#undef S21_re
#undef S21_im
#undef S22_re
#undef S22_im
#undef S30_re
#undef S30_im
#undef S31_re
#undef S31_im
#undef S32_re
#undef S32_im
#undef spinorFloat

#endif // (__COMPUTE_CAPABILITY__ >= 130)

#undef SPINORTEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexSingle
#endif

#define S00_re I0.x
#define S00_im I0.y
#define S01_re I0.z
#define S01_im I0.w
#define S02_re I1.x
#define S02_im I1.y
#define S10_re I1.z
#define S10_im I1.w
#define S11_re I2.x
#define S11_im I2.y
#define S12_re I2.z
#define S12_im I2.w
#define S20_re I3.x
#define S20_im I3.y
#define S21_re I3.z
#define S21_im I3.w
#define S22_re I4.x
#define S22_im I4.y
#define S30_re I4.z
#define S30_im I4.w
#define S31_re I5.x
#define S31_im I5.y
#define S32_re I5.z
#define S32_im I5.w

#define spinorFloat float

#ifdef DIRECT_ACCESS_CLOVER
	#define TMCLOVERTEX clover
	#define TM_INV_CLOVERTEX cloverInv
	#define READ_CLOVER READ_CLOVER_SINGLE
	#define ASSN_CLOVER ASSN_CLOVER_SINGLE
#else
	#ifdef USE_TEXTURE_OBJECTS
		#define TMCLOVERTEX (param.cloverTex)
		#define TM_INV_CLOVERTEX (param.cloverInvTex)
	#else
		#define TMCLOVERTEX cloverTexSingle
		#define TM_INV_CLOVERTEX cloverInvTexSingle
	#endif
	#define READ_CLOVER READ_CLOVER_SINGLE_TEX
	#define ASSN_CLOVER ASSN_CLOVER_SINGLE_TEX
#endif

__global__ void twistCloverGamma5Kernel(float4 *spinor, float *null, float a, const float4 *in, const float *null2, DslashParam param,
					const float4 *clover, const float *cNorm, const float4 *cloverInv, const float *cNrm2)
{
#ifdef GPU_TWISTED_CLOVER_DIRAC
   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

   float4 I0 = TEX1DFETCH(float4, SPINORTEX, sid + 0 * sp_stride);   
   float4 I1 = TEX1DFETCH(float4, SPINORTEX, sid + 1 * sp_stride);   
   float4 I2 = TEX1DFETCH(float4, SPINORTEX, sid + 2 * sp_stride);   
   float4 I3 = TEX1DFETCH(float4, SPINORTEX, sid + 3 * sp_stride);   
   float4 I4 = TEX1DFETCH(float4, SPINORTEX, sid + 4 * sp_stride);   
   float4 I5 = TEX1DFETCH(float4, SPINORTEX, sid + 5 * sp_stride);

   float4 C0, C1, C2, C3, C4, C5, C6, C7, C8;

   //apply (Clover + i*a*gamma_5) to the input spinor
   APPLY_CLOVER_TWIST(c, a, S);
   
   spinor[sid + 0  * sp_stride] = I0;   
   spinor[sid + 1  * sp_stride] = I1;   
   spinor[sid + 2  * sp_stride] = I2;   
   spinor[sid + 3  * sp_stride] = I3;   
   spinor[sid + 4  * sp_stride] = I4;   
   spinor[sid + 5  * sp_stride] = I5;   

#endif 
}

__global__ void twistCloverGamma5InvKernel(float4 *spinor, float *null, float a, const float4 *in, const float *null2, DslashParam param,
					   const float4 *clover, const float *cNorm, const float4 *cloverInv, const float *cNrm2)
{
#ifdef GPU_TWISTED_CLOVER_DIRAC
   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

   float4 I0 = TEX1DFETCH(float4, SPINORTEX, sid + 0 * sp_stride);   
   float4 I1 = TEX1DFETCH(float4, SPINORTEX, sid + 1 * sp_stride);   
   float4 I2 = TEX1DFETCH(float4, SPINORTEX, sid + 2 * sp_stride);   
   float4 I3 = TEX1DFETCH(float4, SPINORTEX, sid + 3 * sp_stride);   
   float4 I4 = TEX1DFETCH(float4, SPINORTEX, sid + 4 * sp_stride);   
   float4 I5 = TEX1DFETCH(float4, SPINORTEX, sid + 5 * sp_stride);

   float4 C0, C1, C2, C3, C4, C5, C6, C7, C8;

   //apply (Clover + i*a*gamma_5)/(Clover^2 + a^2) to the input spinor
   APPLY_CLOVER_TWIST_INV(c, cinv, a, S);
   
   spinor[sid + 0  * sp_stride] = I0;   
   spinor[sid + 1  * sp_stride] = I1;   
   spinor[sid + 2  * sp_stride] = I2;   
   spinor[sid + 3  * sp_stride] = I3;   
   spinor[sid + 4  * sp_stride] = I4;   
   spinor[sid + 5  * sp_stride] = I5;   

#endif 
}

#undef TMCLOVERTEX
#undef TM_INV_CLOVERTEX
#undef READ_CLOVER
#undef ASSN_CLOVER


#undef SPINORTEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#define SPINORTEXNORM param.inTexNorm
#else
#define SPINORTEX spinorTexHalf
#define SPINORTEXNORM spinorTexHalfNorm
#endif

#ifdef DIRECT_ACCESS_CLOVER
	#define CLOVERTEX clover
	#define READ_CLOVER READ_CLOVER_HALF
	#define ASSN_CLOVER ASSN_CLOVER_HALF
#else
	#ifdef USE_TEXTURE_OBJECTS
		#define TMCLOVERTEX (param.cloverTex)
		#define TMCLOVERTEXNORM (param.cloverNormTex)
		#define TM_INV_CLOVERTEX (param.cloverInvTex)
		#define TM_INV_CLOVERTEXNORM (param.cloverInvNormTex)
	#else
		#define TMCLOVERTEX cloverTexHalf
		#define TMCLOVERTEXNORM cloverTexNorm
		#define TM_INV_CLOVERTEX cloverInvTexHalf
		#define TM_INV_CLOVERTEXNORM cloverInvTexNorm
	#endif
	#define READ_CLOVER READ_CLOVER_HALF_TEX
	#define ASSN_CLOVER ASSN_CLOVER_HALF_TEX
#endif

__global__ void twistCloverGamma5Kernel(short4* spinor, float *spinorNorm, float a, const short4 *in, const float *inNorm, DslashParam param,
					const short4 *clover, const float *cNorm, const short4 *cloverInv, const float *cNrm2)
{
#ifdef GPU_TWISTED_CLOVER_DIRAC
   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

   float4 I0 = TEX1DFETCH(float4, SPINORTEX, sid + 0 * sp_stride);   
   float4 I1 = TEX1DFETCH(float4, SPINORTEX, sid + 1 * sp_stride);   
   float4 I2 = TEX1DFETCH(float4, SPINORTEX, sid + 2 * sp_stride);   
   float4 I3 = TEX1DFETCH(float4, SPINORTEX, sid + 3 * sp_stride);   
   float4 I4 = TEX1DFETCH(float4, SPINORTEX, sid + 4 * sp_stride);   
   float4 I5 = TEX1DFETCH(float4, SPINORTEX, sid + 5 * sp_stride);
   
   float KC = TEX1DFETCH(float, SPINORTEXNORM, sid);
   
   I0 = KC * I0;
   I1 = KC * I1;
   I2 = KC * I2;
   I3 = KC * I3;
   I4 = KC * I4;
   I5 = KC * I5;    
   
   float4 C0, C1, C2, C3, C4, C5, C6, C7, C8;
   float K;

   //apply (Clover + i*a*gamma_5) to the input spinor
   APPLY_CLOVER_TWIST(c, a, S);
   
   float k0  = fmaxf(fabsf(I0.x), fabsf(I0.y));			
   float k1  = fmaxf(fabsf(I0.z), fabsf(I0.w));			
   float k2  = fmaxf(fabsf(I1.x), fabsf(I1.y));			
   float k3  = fmaxf(fabsf(I1.z), fabsf(I1.w));			
   float k4  = fmaxf(fabsf(I2.x), fabsf(I2.y));			
   float k5  = fmaxf(fabsf(I2.z), fabsf(I2.w));			
   float k6  = fmaxf(fabsf(I3.x), fabsf(I3.y));			
   float k7  = fmaxf(fabsf(I3.z), fabsf(I3.w));			
   float k8  = fmaxf(fabsf(I4.x), fabsf(I4.y));			
   float k9  = fmaxf(fabsf(I4.z), fabsf(I4.w));			
   float k10 = fmaxf(fabsf(I5.x), fabsf(I5.y));			
   float k11 = fmaxf(fabsf(I5.z), fabsf(I5.w));			
   k0 = fmaxf(k0, k1);							
   k1 = fmaxf(k2, k3);							
   k2 = fmaxf(k4, k5);							
   k3 = fmaxf(k6, k7);							
   k4 = fmaxf(k8, k9);							
   k5 = fmaxf(k10, k11);							
   k0 = fmaxf(k0, k1);							
   k1 = fmaxf(k2, k3);							
   k2 = fmaxf(k4, k5);							
   k0 = fmaxf(k0, k1);							
   k0 = fmaxf(k0, k2);							
   spinorNorm[sid] = k0;								
   float scale = __fdividef(MAX_SHORT, k0);
   
   I0 = scale * I0; 	
   I1 = scale * I1;
   I2 = scale * I2;
   I3 = scale * I3;
   I4 = scale * I4;
   I5 = scale * I5;
   
   spinor[sid+0*(sp_stride)] = make_short4((short)I0.x, (short)I0.y, (short)I0.z, (short)I0.w); 
   spinor[sid+1*(sp_stride)] = make_short4((short)I1.x, (short)I1.y, (short)I1.z, (short)I1.w); 
   spinor[sid+2*(sp_stride)] = make_short4((short)I2.x, (short)I2.y, (short)I2.z, (short)I2.w); 
   spinor[sid+3*(sp_stride)] = make_short4((short)I3.x, (short)I3.y, (short)I3.z, (short)I3.w); 
   spinor[sid+4*(sp_stride)] = make_short4((short)I4.x, (short)I4.y, (short)I4.z, (short)I4.w); 
   spinor[sid+5*(sp_stride)] = make_short4((short)I5.x, (short)I5.y, (short)I5.z, (short)I5.w);

#endif 
}

__global__ void twistCloverGamma5InvKernel(short4* spinor, float *spinorNorm, float a, const short4 *in, const float *inNorm, DslashParam param,
					   const short4 *clover, const float *cNorm, const short4 *cloverInv, const float *cNrm2)
{
#ifdef GPU_TWISTED_CLOVER_DIRAC
   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

   float4 I0 = TEX1DFETCH(float4, SPINORTEX, sid + 0 * sp_stride);   
   float4 I1 = TEX1DFETCH(float4, SPINORTEX, sid + 1 * sp_stride);   
   float4 I2 = TEX1DFETCH(float4, SPINORTEX, sid + 2 * sp_stride);   
   float4 I3 = TEX1DFETCH(float4, SPINORTEX, sid + 3 * sp_stride);   
   float4 I4 = TEX1DFETCH(float4, SPINORTEX, sid + 4 * sp_stride);   
   float4 I5 = TEX1DFETCH(float4, SPINORTEX, sid + 5 * sp_stride);
   
   float KC = TEX1DFETCH(float, SPINORTEXNORM, sid);
   
   I0 = KC * I0;
   I1 = KC * I1;
   I2 = KC * I2;
   I3 = KC * I3;
   I4 = KC * I4;
   I5 = KC * I5;    
   
   float4 C0, C1, C2, C3, C4, C5, C6, C7, C8;
   float K;

   //apply (Clover + i*a*gamma_5)/(Clover^2 + a^2) to the input spinor
   APPLY_CLOVER_TWIST_INV(c, cinv, a, S);
   
   float k0  = fmaxf(fabsf(I0.x), fabsf(I0.y));			
   float k1  = fmaxf(fabsf(I0.z), fabsf(I0.w));			
   float k2  = fmaxf(fabsf(I1.x), fabsf(I1.y));			
   float k3  = fmaxf(fabsf(I1.z), fabsf(I1.w));			
   float k4  = fmaxf(fabsf(I2.x), fabsf(I2.y));			
   float k5  = fmaxf(fabsf(I2.z), fabsf(I2.w));			
   float k6  = fmaxf(fabsf(I3.x), fabsf(I3.y));			
   float k7  = fmaxf(fabsf(I3.z), fabsf(I3.w));			
   float k8  = fmaxf(fabsf(I4.x), fabsf(I4.y));			
   float k9  = fmaxf(fabsf(I4.z), fabsf(I4.w));			
   float k10 = fmaxf(fabsf(I5.x), fabsf(I5.y));			
   float k11 = fmaxf(fabsf(I5.z), fabsf(I5.w));			
   k0 = fmaxf(k0, k1);							
   k1 = fmaxf(k2, k3);							
   k2 = fmaxf(k4, k5);							
   k3 = fmaxf(k6, k7);							
   k4 = fmaxf(k8, k9);							
   k5 = fmaxf(k10, k11);							
   k0 = fmaxf(k0, k1);							
   k1 = fmaxf(k2, k3);							
   k2 = fmaxf(k4, k5);							
   k0 = fmaxf(k0, k1);							
   k0 = fmaxf(k0, k2);							
   spinorNorm[sid] = k0;								
   float scale = __fdividef(MAX_SHORT, k0);
   
   I0 = scale * I0; 	
   I1 = scale * I1;
   I2 = scale * I2;
   I3 = scale * I3;
   I4 = scale * I4;
   I5 = scale * I5;
   
   spinor[sid+0*(sp_stride)] = make_short4((short)I0.x, (short)I0.y, (short)I0.z, (short)I0.w); 
   spinor[sid+1*(sp_stride)] = make_short4((short)I1.x, (short)I1.y, (short)I1.z, (short)I1.w); 
   spinor[sid+2*(sp_stride)] = make_short4((short)I2.x, (short)I2.y, (short)I2.z, (short)I2.w); 
   spinor[sid+3*(sp_stride)] = make_short4((short)I3.x, (short)I3.y, (short)I3.z, (short)I3.w); 
   spinor[sid+4*(sp_stride)] = make_short4((short)I4.x, (short)I4.y, (short)I4.z, (short)I4.w); 
   spinor[sid+5*(sp_stride)] = make_short4((short)I5.x, (short)I5.y, (short)I5.z, (short)I5.w);

#endif 
}

#undef CLOVERTEX 
#undef READ_CLOVER
#undef TMCLOVERTEX
#undef TMCLOVERTEXNORM
#undef TM_INV_CLOVERTEX
#undef TM_INV_CLOVERTEXNORM


#undef SPINORTEX
#undef SPINORTEXNORM

#undef S00_re
#undef S00_im
#undef S01_re
#undef S01_im
#undef S02_re
#undef S02_im
#undef S10_re
#undef S10_im
#undef S11_re
#undef S11_im
#undef S12_re
#undef S12_im
#undef S20_re
#undef S20_im
#undef S21_re
#undef S21_im
#undef S22_re
#undef S22_im
#undef S30_re
#undef S30_im
#undef S31_re
#undef S31_im
#undef S32_re
#undef S32_im
#undef spinorFloat

#endif //_TM_GAMMA_CORE_H


