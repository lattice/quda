#ifdef MULTI_GPU

// *** CUDA DSLASH DAGGER ***

#define DSLASH_SHARED_FLOATS_PER_THREAD 0


#if (CUDA_VERSION >= 4010)
#define VOLATILE
#else
#define VOLATILE volatile
#endif
// input spinor
#ifdef SPINOR_DOUBLE
#define spinorFloat double
#define i00_re I0.x
#define i00_im I0.y
#define i01_re I1.x
#define i01_im I1.y
#define i02_re I2.x
#define i02_im I2.y
#define i10_re I3.x
#define i10_im I3.y
#define i11_re I4.x
#define i11_im I4.y
#define i12_re I5.x
#define i12_im I5.y
#define i20_re I6.x
#define i20_im I6.y
#define i21_re I7.x
#define i21_im I7.y
#define i22_re I8.x
#define i22_im I8.y
#define i30_re I9.x
#define i30_im I9.y
#define i31_re I10.x
#define i31_im I10.y
#define i32_re I11.x
#define i32_im I11.y
#else
#define spinorFloat float
#define i00_re I0.x
#define i00_im I0.y
#define i01_re I0.z
#define i01_im I0.w
#define i02_re I1.x
#define i02_im I1.y
#define i10_re I1.z
#define i10_im I1.w
#define i11_re I2.x
#define i11_im I2.y
#define i12_re I2.z
#define i12_im I2.w
#define i20_re I3.x
#define i20_im I3.y
#define i21_re I3.z
#define i21_im I3.w
#define i22_re I4.x
#define i22_im I4.y
#define i30_re I4.z
#define i30_im I4.w
#define i31_re I5.x
#define i31_im I5.y
#define i32_re I5.z
#define i32_im I5.w
#endif // SPINOR_DOUBLE

// output spinor
VOLATILE spinorFloat o00_re;
VOLATILE spinorFloat o00_im;
VOLATILE spinorFloat o01_re;
VOLATILE spinorFloat o01_im;
VOLATILE spinorFloat o02_re;
VOLATILE spinorFloat o02_im;
VOLATILE spinorFloat o10_re;
VOLATILE spinorFloat o10_im;
VOLATILE spinorFloat o11_re;
VOLATILE spinorFloat o11_im;
VOLATILE spinorFloat o12_re;
VOLATILE spinorFloat o12_im;
VOLATILE spinorFloat o20_re;
VOLATILE spinorFloat o20_im;
VOLATILE spinorFloat o21_re;
VOLATILE spinorFloat o21_im;
VOLATILE spinorFloat o22_re;
VOLATILE spinorFloat o22_im;
VOLATILE spinorFloat o30_re;
VOLATILE spinorFloat o30_im;
VOLATILE spinorFloat o31_re;
VOLATILE spinorFloat o31_im;
VOLATILE spinorFloat o32_re;
VOLATILE spinorFloat o32_im;

#ifdef SPINOR_DOUBLE
#if (__COMPUTE_CAPABILITY__ >= 200)
#define SHARED_STRIDE 16 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE 8 // to avoid bank conflicts on G80 and GT200
#endif
#else
#if (__COMPUTE_CAPABILITY__ >= 200)
#define SHARED_STRIDE 32 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE 16 // to avoid bank conflicts on G80 and GT200
#endif
#endif
#include "io_spinor.h"

int sid = ((blockIdx.y*blockDim.y + threadIdx.y)*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
if (sid >= param.threads*param.Ls) return;

int dim;
int face_num;
int face_idx;
int Y[4] = {X1,X2,X3,X4};
int faceVolume[4];
faceVolume[0] = (X2*X3*X4)>>1;
faceVolume[1] = (X1*X3*X4)>>1;
faceVolume[2] = (X1*X2*X4)>>1;
faceVolume[3] = (X1*X2*X3)>>1;




int boundaryCrossing;

int X, xs;
{

#ifdef DSLASH_XPAY
 READ_ACCUM(ACCUMTEX, param.sp_stride)
 VOLATILE spinorFloat coeff;

#ifdef MDWF_mode
 coeff = (spinorFloat)(0.5/(mdwf_b5[xs]*(m5+4.0) + 1.0));
 coeff *= -coeff;
#else
 coeff = a;
#endif

#ifdef YPAX
#ifdef SPINOR_DOUBLE
 o00_re = o00_re + coeff*accum0.x;
 o00_im = o00_im + coeff*accum0.y;
 o01_re = o01_re + coeff*accum1.x;
 o01_im = o01_im + coeff*accum1.y;
 o02_re = o02_re + coeff*accum2.x;
 o02_im = o02_im + coeff*accum2.y;
 o10_re = o10_re + coeff*accum3.x;
 o10_im = o10_im + coeff*accum3.y;
 o11_re = o11_re + coeff*accum4.x;
 o11_im = o11_im + coeff*accum4.y;
 o12_re = o12_re + coeff*accum5.x;
 o12_im = o12_im + coeff*accum5.y;
 o20_re = o20_re + coeff*accum6.x;
 o20_im = o20_im + coeff*accum6.y;
 o21_re = o21_re + coeff*accum7.x;
 o21_im = o21_im + coeff*accum7.y;
 o22_re = o22_re + coeff*accum8.x;
 o22_im = o22_im + coeff*accum8.y;
 o30_re = o30_re + coeff*accum9.x;
 o30_im = o30_im + coeff*accum9.y;
 o31_re = o31_re + coeff*accum10.x;
 o31_im = o31_im + coeff*accum10.y;
 o32_re = o32_re + coeff*accum11.x;
 o32_im = o32_im + coeff*accum11.y;
#else
 o00_re = o00_re + coeff*accum0.x;
 o00_im = o00_im + coeff*accum0.y;
 o01_re = o01_re + coeff*accum0.z;
 o01_im = o01_im + coeff*accum0.w;
 o02_re = o02_re + coeff*accum1.x;
 o02_im = o02_im + coeff*accum1.y;
 o10_re = o10_re + coeff*accum1.z;
 o10_im = o10_im + coeff*accum1.w;
 o11_re = o11_re + coeff*accum2.x;
 o11_im = o11_im + coeff*accum2.y;
 o12_re = o12_re + coeff*accum2.z;
 o12_im = o12_im + coeff*accum2.w;
 o20_re = o20_re + coeff*accum3.x;
 o20_im = o20_im + coeff*accum3.y;
 o21_re = o21_re + coeff*accum3.z;
 o21_im = o21_im + coeff*accum3.w;
 o22_re = o22_re + coeff*accum4.x;
 o22_im = o22_im + coeff*accum4.y;
 o30_re = o30_re + coeff*accum4.z;
 o30_im = o30_im + coeff*accum4.w;
 o31_re = o31_re + coeff*accum5.x;
 o31_im = o31_im + coeff*accum5.y;
 o32_re = o32_re + coeff*accum5.z;
 o32_im = o32_im + coeff*accum5.w;
#endif // SPINOR_DOUBLE
#else
#ifdef SPINOR_DOUBLE
 o00_re = coeff*o00_re + accum0.x;
 o00_im = coeff*o00_im + accum0.y;
 o01_re = coeff*o01_re + accum1.x;
 o01_im = coeff*o01_im + accum1.y;
 o02_re = coeff*o02_re + accum2.x;
 o02_im = coeff*o02_im + accum2.y;
 o10_re = coeff*o10_re + accum3.x;
 o10_im = coeff*o10_im + accum3.y;
 o11_re = coeff*o11_re + accum4.x;
 o11_im = coeff*o11_im + accum4.y;
 o12_re = coeff*o12_re + accum5.x;
 o12_im = coeff*o12_im + accum5.y;
 o20_re = coeff*o20_re + accum6.x;
 o20_im = coeff*o20_im + accum6.y;
 o21_re = coeff*o21_re + accum7.x;
 o21_im = coeff*o21_im + accum7.y;
 o22_re = coeff*o22_re + accum8.x;
 o22_im = coeff*o22_im + accum8.y;
 o30_re = coeff*o30_re + accum9.x;
 o30_im = coeff*o30_im + accum9.y;
 o31_re = coeff*o31_re + accum10.x;
 o31_im = coeff*o31_im + accum10.y;
 o32_re = coeff*o32_re + accum11.x;
 o32_im = coeff*o32_im + accum11.y;
#else
 o00_re = coeff*o00_re + accum0.x;
 o00_im = coeff*o00_im + accum0.y;
 o01_re = coeff*o01_re + accum0.z;
 o01_im = coeff*o01_im + accum0.w;
 o02_re = coeff*o02_re + accum1.x;
 o02_im = coeff*o02_im + accum1.y;
 o10_re = coeff*o10_re + accum1.z;
 o10_im = coeff*o10_im + accum1.w;
 o11_re = coeff*o11_re + accum2.x;
 o11_im = coeff*o11_im + accum2.y;
 o12_re = coeff*o12_re + accum2.z;
 o12_im = coeff*o12_im + accum2.w;
 o20_re = coeff*o20_re + accum3.x;
 o20_im = coeff*o20_im + accum3.y;
 o21_re = coeff*o21_re + accum3.z;
 o21_im = coeff*o21_im + accum3.w;
 o22_re = coeff*o22_re + accum4.x;
 o22_im = coeff*o22_im + accum4.y;
 o30_re = coeff*o30_re + accum4.z;
 o30_im = coeff*o30_im + accum4.w;
 o31_re = coeff*o31_re + accum5.x;
 o31_im = coeff*o31_im + accum5.y;
 o32_re = coeff*o32_re + accum5.z;
 o32_im = coeff*o32_im + accum5.w;
#endif // SPINOR_DOUBLE
#endif // YPAX
#endif // DSLASH_XPAY
}

// write spinor field back to device memory
WRITE_SPINOR(param.sp_stride);

// undefine to prevent warning when precision is changed
#undef spinorFloat
#undef SHARED_STRIDE

#undef i00_re
#undef i00_im
#undef i01_re
#undef i01_im
#undef i02_re
#undef i02_im
#undef i10_re
#undef i10_im
#undef i11_re
#undef i11_im
#undef i12_re
#undef i12_im
#undef i20_re
#undef i20_im
#undef i21_re
#undef i21_im
#undef i22_re
#undef i22_im
#undef i30_re
#undef i30_im
#undef i31_re
#undef i31_im
#undef i32_re
#undef i32_im



#undef VOLATILE
#endif // MULTI_GPU
