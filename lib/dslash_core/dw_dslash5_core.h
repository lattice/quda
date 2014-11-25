// *** CUDA DSLASH ***

#define DSLASH_SHARED_FLOATS_PER_THREAD 0

// NB! Don't trust any MULTI_GPU code

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
#define m5 m5_d
#define mdwf_b5 mdwf_b5_d
#define mdwf_c5 mdwf_c5_d
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
#define m5 m5_f
#define mdwf_b5 mdwf_b5_f
#define mdwf_c5 mdwf_c5_f
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

int boundaryCrossing;

int X, xs;

// Inline by hand for the moment and assume even dimensions
//coordsFromIndex(X, x1, x2, x3, x4, sid, param.parity);

boundaryCrossing = sid/X1h + sid/(X2*X1h) + sid/(X3*X2*X1h);

X = 2*sid + (boundaryCrossing + param.parity) % 2;
xs = X/(X1*X2*X3*X4);

 o00_re = 0; o00_im = 0;
 o01_re = 0; o01_im = 0;
 o02_re = 0; o02_im = 0;
 o10_re = 0; o10_im = 0;
 o11_re = 0; o11_im = 0;
 o12_re = 0; o12_im = 0;
 o20_re = 0; o20_im = 0;
 o21_re = 0; o21_im = 0;
 o22_re = 0; o22_im = 0;
 o30_re = 0; o30_im = 0;
 o31_re = 0; o31_im = 0;
 o32_re = 0; o32_im = 0;


// 5th dimension -- NB: not partitionable!
{
// 2 P_L = 2 P_- = ( ( +1, -1 ), ( -1, +1 ) )
  {
     int sp_idx = ( xs == param.Ls-1 ? X-(param.Ls-1)*2*Vh : X+2*Vh ) / 2;

// read spinor from device memory
     READ_SPINOR( SPINORTEX, param.sp_stride, sp_idx, sp_idx );

     if ( xs != param.Ls-1 )
     {
   o00_re += +i00_re-i20_re;   o00_im += +i00_im-i20_im;
   o01_re += +i01_re-i21_re;   o01_im += +i01_im-i21_im;
   o02_re += +i02_re-i22_re;   o02_im += +i02_im-i22_im;

   o10_re += +i10_re-i30_re;   o10_im += +i10_im-i30_im;
   o11_re += +i11_re-i31_re;   o11_im += +i11_im-i31_im;
   o12_re += +i12_re-i32_re;   o12_im += +i12_im-i32_im;

   o20_re += -i00_re+i20_re;   o20_im += -i00_im+i20_im;
   o21_re += -i01_re+i21_re;   o21_im += -i01_im+i21_im;
   o22_re += -i02_re+i22_re;   o22_im += -i02_im+i22_im;

   o30_re += -i10_re+i30_re;   o30_im += -i10_im+i30_im;
   o31_re += -i11_re+i31_re;   o31_im += -i11_im+i31_im;
   o32_re += -i12_re+i32_re;   o32_im += -i12_im+i32_im;
    }
    else
    {
   o00_re += -mferm*(+i00_re-i20_re);   o00_im += -mferm*(+i00_im-i20_im);
   o01_re += -mferm*(+i01_re-i21_re);   o01_im += -mferm*(+i01_im-i21_im);
   o02_re += -mferm*(+i02_re-i22_re);   o02_im += -mferm*(+i02_im-i22_im);

   o10_re += -mferm*(+i10_re-i30_re);   o10_im += -mferm*(+i10_im-i30_im);
   o11_re += -mferm*(+i11_re-i31_re);   o11_im += -mferm*(+i11_im-i31_im);
   o12_re += -mferm*(+i12_re-i32_re);   o12_im += -mferm*(+i12_im-i32_im);

   o20_re += -mferm*(-i00_re+i20_re);   o20_im += -mferm*(-i00_im+i20_im);
   o21_re += -mferm*(-i01_re+i21_re);   o21_im += -mferm*(-i01_im+i21_im);
   o22_re += -mferm*(-i02_re+i22_re);   o22_im += -mferm*(-i02_im+i22_im);

   o30_re += -mferm*(-i10_re+i30_re);   o30_im += -mferm*(-i10_im+i30_im);
   o31_re += -mferm*(-i11_re+i31_re);   o31_im += -mferm*(-i11_im+i31_im);
   o32_re += -mferm*(-i12_re+i32_re);   o32_im += -mferm*(-i12_im+i32_im);
    } // end if ( xs != param.Ls-1 )
  } // end P_L

 // 2 P_R = 2 P_+ = ( ( +1, +1 ), ( +1, +1 ) )
  {
    int sp_idx = ( xs == 0 ? X+(param.Ls-1)*2*Vh : X-2*Vh ) / 2;

// read spinor from device memory
    READ_SPINOR( SPINORTEX, param.sp_stride, sp_idx, sp_idx );

    if ( xs != 0 )
    {
   o00_re += +i00_re+i20_re;   o00_im += +i00_im+i20_im;
   o01_re += +i01_re+i21_re;   o01_im += +i01_im+i21_im;
   o02_re += +i02_re+i22_re;   o02_im += +i02_im+i22_im;

   o10_re += +i10_re+i30_re;   o10_im += +i10_im+i30_im;
   o11_re += +i11_re+i31_re;   o11_im += +i11_im+i31_im;
   o12_re += +i12_re+i32_re;   o12_im += +i12_im+i32_im;

   o20_re += +i00_re+i20_re;   o20_im += +i00_im+i20_im;
   o21_re += +i01_re+i21_re;   o21_im += +i01_im+i21_im;
   o22_re += +i02_re+i22_re;   o22_im += +i02_im+i22_im;

   o30_re += +i10_re+i30_re;   o30_im += +i10_im+i30_im;
   o31_re += +i11_re+i31_re;   o31_im += +i11_im+i31_im;
   o32_re += +i12_re+i32_re;   o32_im += +i12_im+i32_im;
    }
    else
    {
   o00_re += -mferm*(+i00_re+i20_re);   o00_im += -mferm*(+i00_im+i20_im);
   o01_re += -mferm*(+i01_re+i21_re);   o01_im += -mferm*(+i01_im+i21_im);
   o02_re += -mferm*(+i02_re+i22_re);   o02_im += -mferm*(+i02_im+i22_im);

   o10_re += -mferm*(+i10_re+i30_re);   o10_im += -mferm*(+i10_im+i30_im);
   o11_re += -mferm*(+i11_re+i31_re);   o11_im += -mferm*(+i11_im+i31_im);
   o12_re += -mferm*(+i12_re+i32_re);   o12_im += -mferm*(+i12_im+i32_im);

   o20_re += -mferm*(+i00_re+i20_re);   o20_im += -mferm*(+i00_im+i20_im);
   o21_re += -mferm*(+i01_re+i21_re);   o21_im += -mferm*(+i01_im+i21_im);
   o22_re += -mferm*(+i02_re+i22_re);   o22_im += -mferm*(+i02_im+i22_im);

   o30_re += -mferm*(+i10_re+i30_re);   o30_im += -mferm*(+i10_im+i30_im);
   o31_re += -mferm*(+i11_re+i31_re);   o31_im += -mferm*(+i11_im+i31_im);
   o32_re += -mferm*(+i12_re+i32_re);   o32_im += -mferm*(+i12_im+i32_im);
    } // end if ( xs != 0 )
  } // end P_R

  // MDWF Dslash_5 operator is given as follow
  // Dslash4pre = [c_5(s)(P_+\delta_{s,s`+1} - mP_+\delta_{s,0}\delta_{s`,L_s-1}
  //         + P_-\delta_{s,s`-1}-mP_-\delta_{s,L_s-1}\delta_{s`,0})
  //         + b_5(s)\delta_{s,s`}]\delta_{x,x`}
  // For Dslash4pre
  // C_5 \equiv c_5(s)*0.5
  // B_5 \equiv b_5(s)
  // For Dslash5
  // C_5 \equiv 0.5*{c_5(s)(4+M_5)-1}/{b_5(s)(4+M_5)+1}
  // B_5 \equiv 1.0
#ifdef MDWF_mode   // Check whether MDWF option is enabled
#if (MDWF_mode==1)
  VOLATILE spinorFloat C_5;
  VOLATILE spinorFloat B_5;
  C_5 = (spinorFloat)mdwf_c5[xs]*0.5;
  B_5 = (spinorFloat)mdwf_b5[xs];

  READ_SPINOR( SPINORTEX, param.sp_stride, X/2, X/2 );
  o00_re = C_5*o00_re + B_5*i00_re;
  o00_im = C_5*o00_im + B_5*i00_im;
  o01_re = C_5*o01_re + B_5*i01_re;
  o01_im = C_5*o01_im + B_5*i01_im;
  o02_re = C_5*o02_re + B_5*i02_re;
  o02_im = C_5*o02_im + B_5*i02_im;
  o10_re = C_5*o10_re + B_5*i10_re;
  o10_im = C_5*o10_im + B_5*i10_im;
  o11_re = C_5*o11_re + B_5*i11_re;
  o11_im = C_5*o11_im + B_5*i11_im;
  o12_re = C_5*o12_re + B_5*i12_re;
  o12_im = C_5*o12_im + B_5*i12_im;
  o20_re = C_5*o20_re + B_5*i20_re;
  o20_im = C_5*o20_im + B_5*i20_im;
  o21_re = C_5*o21_re + B_5*i21_re;
  o21_im = C_5*o21_im + B_5*i21_im;
  o22_re = C_5*o22_re + B_5*i22_re;
  o22_im = C_5*o22_im + B_5*i22_im;
  o30_re = C_5*o30_re + B_5*i30_re;
  o30_im = C_5*o30_im + B_5*i30_im;
  o31_re = C_5*o31_re + B_5*i31_re;
  o31_im = C_5*o31_im + B_5*i31_im;
  o32_re = C_5*o32_re + B_5*i32_re;
  o32_im = C_5*o32_im + B_5*i32_im;
#elif (MDWF_mode==2)
  VOLATILE spinorFloat C_5;
  C_5 = (spinorFloat)(0.5*(mdwf_c5[xs]*(m5+4.0) - 1.0)/(mdwf_b5[xs]*(m5+4.0) + 1.0));

  READ_SPINOR( SPINORTEX, param.sp_stride, X/2, X/2 );
  o00_re = C_5*o00_re + i00_re;
  o00_im = C_5*o00_im + i00_im;
  o01_re = C_5*o01_re + i01_re;
  o01_im = C_5*o01_im + i01_im;
  o02_re = C_5*o02_re + i02_re;
  o02_im = C_5*o02_im + i02_im;
  o10_re = C_5*o10_re + i10_re;
  o10_im = C_5*o10_im + i10_im;
  o11_re = C_5*o11_re + i11_re;
  o11_im = C_5*o11_im + i11_im;
  o12_re = C_5*o12_re + i12_re;
  o12_im = C_5*o12_im + i12_im;
  o20_re = C_5*o20_re + i20_re;
  o20_im = C_5*o20_im + i20_im;
  o21_re = C_5*o21_re + i21_re;
  o21_im = C_5*o21_im + i21_im;
  o22_re = C_5*o22_re + i22_re;
  o22_im = C_5*o22_im + i22_im;
  o30_re = C_5*o30_re + i30_re;
  o30_im = C_5*o30_im + i30_im;
  o31_re = C_5*o31_re + i31_re;
  o31_im = C_5*o31_im + i31_im;
  o32_re = C_5*o32_re + i32_re;
  o32_im = C_5*o32_im + i32_im;
#endif  // select MDWF mode
#endif  // check MDWF on/off
} // end 5th dimension

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
#undef m5
#undef mdwf_b5
#undef mdwf_c5
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
