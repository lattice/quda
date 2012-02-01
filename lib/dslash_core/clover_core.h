// *** CUDA CLOVER ***

#define CLOVER_SHARED_FLOATS_PER_THREAD 0


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

// first chiral block of inverted clover term
#ifdef CLOVER_DOUBLE
#define c00_00_re C0.x
#define c01_01_re C0.y
#define c02_02_re C1.x
#define c10_10_re C1.y
#define c11_11_re C2.x
#define c12_12_re C2.y
#define c01_00_re C3.x
#define c01_00_im C3.y
#define c02_00_re C4.x
#define c02_00_im C4.y
#define c10_00_re C5.x
#define c10_00_im C5.y
#define c11_00_re C6.x
#define c11_00_im C6.y
#define c12_00_re C7.x
#define c12_00_im C7.y
#define c02_01_re C8.x
#define c02_01_im C8.y
#define c10_01_re C9.x
#define c10_01_im C9.y
#define c11_01_re C10.x
#define c11_01_im C10.y
#define c12_01_re C11.x
#define c12_01_im C11.y
#define c10_02_re C12.x
#define c10_02_im C12.y
#define c11_02_re C13.x
#define c11_02_im C13.y
#define c12_02_re C14.x
#define c12_02_im C14.y
#define c11_10_re C15.x
#define c11_10_im C15.y
#define c12_10_re C16.x
#define c12_10_im C16.y
#define c12_11_re C17.x
#define c12_11_im C17.y
#else
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
#endif // CLOVER_DOUBLE

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

// second chiral block of inverted clover term (reuses C0,...,C9)
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
#define SHARED_STRIDE  8 // to avoid bank conflicts on G80 and GT200
#endif
#else
#if (__COMPUTE_CAPABILITY__ >= 200)
#define SHARED_STRIDE 32 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE 16 // to avoid bank conflicts on G80 and GT200
#endif
#endif

#include "read_clover.h"
#include "io_spinor.h"

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= param.threads) return;

// read spinor from device memory
READ_SPINOR(SPINORTEX, sp_stride, sid, sid);

{
  
  // change to chiral basis
  {
    spinorFloat a00_re = -i10_re - i30_re;
    spinorFloat a00_im = -i10_im - i30_im;
    spinorFloat a10_re =  i00_re + i20_re;
    spinorFloat a10_im =  i00_im + i20_im;
    spinorFloat a20_re = -i10_re + i30_re;
    spinorFloat a20_im = -i10_im + i30_im;
    spinorFloat a30_re =  i00_re - i20_re;
    spinorFloat a30_im =  i00_im - i20_im;
    
    o00_re = a00_re;  o00_im = a00_im;
    o10_re = a10_re;  o10_im = a10_im;
    o20_re = a20_re;  o20_im = a20_im;
    o30_re = a30_re;  o30_im = a30_im;
  }
  
  {
    spinorFloat a01_re = -i11_re - i31_re;
    spinorFloat a01_im = -i11_im - i31_im;
    spinorFloat a11_re =  i01_re + i21_re;
    spinorFloat a11_im =  i01_im + i21_im;
    spinorFloat a21_re = -i11_re + i31_re;
    spinorFloat a21_im = -i11_im + i31_im;
    spinorFloat a31_re =  i01_re - i21_re;
    spinorFloat a31_im =  i01_im - i21_im;
    
    o01_re = a01_re;  o01_im = a01_im;
    o11_re = a11_re;  o11_im = a11_im;
    o21_re = a21_re;  o21_im = a21_im;
    o31_re = a31_re;  o31_im = a31_im;
  }
  
  {
    spinorFloat a02_re = -i12_re - i32_re;
    spinorFloat a02_im = -i12_im - i32_im;
    spinorFloat a12_re =  i02_re + i22_re;
    spinorFloat a12_im =  i02_im + i22_im;
    spinorFloat a22_re = -i12_re + i32_re;
    spinorFloat a22_im = -i12_im + i32_im;
    spinorFloat a32_re =  i02_re - i22_re;
    spinorFloat a32_im =  i02_im - i22_im;
    
    o02_re = a02_re;  o02_im = a02_im;
    o12_re = a12_re;  o12_im = a12_im;
    o22_re = a22_re;  o22_im = a22_im;
    o32_re = a32_re;  o32_im = a32_im;
  }
  
  // apply first chiral block
  {
    READ_CLOVER(CLOVERTEX, 0)
    
    spinorFloat a00_re = 0; spinorFloat a00_im = 0;
    spinorFloat a01_re = 0; spinorFloat a01_im = 0;
    spinorFloat a02_re = 0; spinorFloat a02_im = 0;
    spinorFloat a10_re = 0; spinorFloat a10_im = 0;
    spinorFloat a11_re = 0; spinorFloat a11_im = 0;
    spinorFloat a12_re = 0; spinorFloat a12_im = 0;
    
    a00_re += c00_00_re * o00_re;
    a00_im += c00_00_re * o00_im;
    a00_re += c00_01_re * o01_re;
    a00_re -= c00_01_im * o01_im;
    a00_im += c00_01_re * o01_im;
    a00_im += c00_01_im * o01_re;
    a00_re += c00_02_re * o02_re;
    a00_re -= c00_02_im * o02_im;
    a00_im += c00_02_re * o02_im;
    a00_im += c00_02_im * o02_re;
    a00_re += c00_10_re * o10_re;
    a00_re -= c00_10_im * o10_im;
    a00_im += c00_10_re * o10_im;
    a00_im += c00_10_im * o10_re;
    a00_re += c00_11_re * o11_re;
    a00_re -= c00_11_im * o11_im;
    a00_im += c00_11_re * o11_im;
    a00_im += c00_11_im * o11_re;
    a00_re += c00_12_re * o12_re;
    a00_re -= c00_12_im * o12_im;
    a00_im += c00_12_re * o12_im;
    a00_im += c00_12_im * o12_re;
    
    a01_re += c01_00_re * o00_re;
    a01_re -= c01_00_im * o00_im;
    a01_im += c01_00_re * o00_im;
    a01_im += c01_00_im * o00_re;
    a01_re += c01_01_re * o01_re;
    a01_im += c01_01_re * o01_im;
    a01_re += c01_02_re * o02_re;
    a01_re -= c01_02_im * o02_im;
    a01_im += c01_02_re * o02_im;
    a01_im += c01_02_im * o02_re;
    a01_re += c01_10_re * o10_re;
    a01_re -= c01_10_im * o10_im;
    a01_im += c01_10_re * o10_im;
    a01_im += c01_10_im * o10_re;
    a01_re += c01_11_re * o11_re;
    a01_re -= c01_11_im * o11_im;
    a01_im += c01_11_re * o11_im;
    a01_im += c01_11_im * o11_re;
    a01_re += c01_12_re * o12_re;
    a01_re -= c01_12_im * o12_im;
    a01_im += c01_12_re * o12_im;
    a01_im += c01_12_im * o12_re;
    
    a02_re += c02_00_re * o00_re;
    a02_re -= c02_00_im * o00_im;
    a02_im += c02_00_re * o00_im;
    a02_im += c02_00_im * o00_re;
    a02_re += c02_01_re * o01_re;
    a02_re -= c02_01_im * o01_im;
    a02_im += c02_01_re * o01_im;
    a02_im += c02_01_im * o01_re;
    a02_re += c02_02_re * o02_re;
    a02_im += c02_02_re * o02_im;
    a02_re += c02_10_re * o10_re;
    a02_re -= c02_10_im * o10_im;
    a02_im += c02_10_re * o10_im;
    a02_im += c02_10_im * o10_re;
    a02_re += c02_11_re * o11_re;
    a02_re -= c02_11_im * o11_im;
    a02_im += c02_11_re * o11_im;
    a02_im += c02_11_im * o11_re;
    a02_re += c02_12_re * o12_re;
    a02_re -= c02_12_im * o12_im;
    a02_im += c02_12_re * o12_im;
    a02_im += c02_12_im * o12_re;
    
    a10_re += c10_00_re * o00_re;
    a10_re -= c10_00_im * o00_im;
    a10_im += c10_00_re * o00_im;
    a10_im += c10_00_im * o00_re;
    a10_re += c10_01_re * o01_re;
    a10_re -= c10_01_im * o01_im;
    a10_im += c10_01_re * o01_im;
    a10_im += c10_01_im * o01_re;
    a10_re += c10_02_re * o02_re;
    a10_re -= c10_02_im * o02_im;
    a10_im += c10_02_re * o02_im;
    a10_im += c10_02_im * o02_re;
    a10_re += c10_10_re * o10_re;
    a10_im += c10_10_re * o10_im;
    a10_re += c10_11_re * o11_re;
    a10_re -= c10_11_im * o11_im;
    a10_im += c10_11_re * o11_im;
    a10_im += c10_11_im * o11_re;
    a10_re += c10_12_re * o12_re;
    a10_re -= c10_12_im * o12_im;
    a10_im += c10_12_re * o12_im;
    a10_im += c10_12_im * o12_re;
    
    a11_re += c11_00_re * o00_re;
    a11_re -= c11_00_im * o00_im;
    a11_im += c11_00_re * o00_im;
    a11_im += c11_00_im * o00_re;
    a11_re += c11_01_re * o01_re;
    a11_re -= c11_01_im * o01_im;
    a11_im += c11_01_re * o01_im;
    a11_im += c11_01_im * o01_re;
    a11_re += c11_02_re * o02_re;
    a11_re -= c11_02_im * o02_im;
    a11_im += c11_02_re * o02_im;
    a11_im += c11_02_im * o02_re;
    a11_re += c11_10_re * o10_re;
    a11_re -= c11_10_im * o10_im;
    a11_im += c11_10_re * o10_im;
    a11_im += c11_10_im * o10_re;
    a11_re += c11_11_re * o11_re;
    a11_im += c11_11_re * o11_im;
    a11_re += c11_12_re * o12_re;
    a11_re -= c11_12_im * o12_im;
    a11_im += c11_12_re * o12_im;
    a11_im += c11_12_im * o12_re;
    
    a12_re += c12_00_re * o00_re;
    a12_re -= c12_00_im * o00_im;
    a12_im += c12_00_re * o00_im;
    a12_im += c12_00_im * o00_re;
    a12_re += c12_01_re * o01_re;
    a12_re -= c12_01_im * o01_im;
    a12_im += c12_01_re * o01_im;
    a12_im += c12_01_im * o01_re;
    a12_re += c12_02_re * o02_re;
    a12_re -= c12_02_im * o02_im;
    a12_im += c12_02_re * o02_im;
    a12_im += c12_02_im * o02_re;
    a12_re += c12_10_re * o10_re;
    a12_re -= c12_10_im * o10_im;
    a12_im += c12_10_re * o10_im;
    a12_im += c12_10_im * o10_re;
    a12_re += c12_11_re * o11_re;
    a12_re -= c12_11_im * o11_im;
    a12_im += c12_11_re * o11_im;
    a12_im += c12_11_im * o11_re;
    a12_re += c12_12_re * o12_re;
    a12_im += c12_12_re * o12_im;
    
    o00_re = a00_re;  o00_im = a00_im;
    o01_re = a01_re;  o01_im = a01_im;
    o02_re = a02_re;  o02_im = a02_im;
    o10_re = a10_re;  o10_im = a10_im;
    o11_re = a11_re;  o11_im = a11_im;
    o12_re = a12_re;  o12_im = a12_im;
    
  }
  
  // apply second chiral block
  {
    READ_CLOVER(CLOVERTEX, 1)
    
    spinorFloat a20_re = 0; spinorFloat a20_im = 0;
    spinorFloat a21_re = 0; spinorFloat a21_im = 0;
    spinorFloat a22_re = 0; spinorFloat a22_im = 0;
    spinorFloat a30_re = 0; spinorFloat a30_im = 0;
    spinorFloat a31_re = 0; spinorFloat a31_im = 0;
    spinorFloat a32_re = 0; spinorFloat a32_im = 0;
    
    a20_re += c20_20_re * o20_re;
    a20_im += c20_20_re * o20_im;
    a20_re += c20_21_re * o21_re;
    a20_re -= c20_21_im * o21_im;
    a20_im += c20_21_re * o21_im;
    a20_im += c20_21_im * o21_re;
    a20_re += c20_22_re * o22_re;
    a20_re -= c20_22_im * o22_im;
    a20_im += c20_22_re * o22_im;
    a20_im += c20_22_im * o22_re;
    a20_re += c20_30_re * o30_re;
    a20_re -= c20_30_im * o30_im;
    a20_im += c20_30_re * o30_im;
    a20_im += c20_30_im * o30_re;
    a20_re += c20_31_re * o31_re;
    a20_re -= c20_31_im * o31_im;
    a20_im += c20_31_re * o31_im;
    a20_im += c20_31_im * o31_re;
    a20_re += c20_32_re * o32_re;
    a20_re -= c20_32_im * o32_im;
    a20_im += c20_32_re * o32_im;
    a20_im += c20_32_im * o32_re;
    
    a21_re += c21_20_re * o20_re;
    a21_re -= c21_20_im * o20_im;
    a21_im += c21_20_re * o20_im;
    a21_im += c21_20_im * o20_re;
    a21_re += c21_21_re * o21_re;
    a21_im += c21_21_re * o21_im;
    a21_re += c21_22_re * o22_re;
    a21_re -= c21_22_im * o22_im;
    a21_im += c21_22_re * o22_im;
    a21_im += c21_22_im * o22_re;
    a21_re += c21_30_re * o30_re;
    a21_re -= c21_30_im * o30_im;
    a21_im += c21_30_re * o30_im;
    a21_im += c21_30_im * o30_re;
    a21_re += c21_31_re * o31_re;
    a21_re -= c21_31_im * o31_im;
    a21_im += c21_31_re * o31_im;
    a21_im += c21_31_im * o31_re;
    a21_re += c21_32_re * o32_re;
    a21_re -= c21_32_im * o32_im;
    a21_im += c21_32_re * o32_im;
    a21_im += c21_32_im * o32_re;
    
    a22_re += c22_20_re * o20_re;
    a22_re -= c22_20_im * o20_im;
    a22_im += c22_20_re * o20_im;
    a22_im += c22_20_im * o20_re;
    a22_re += c22_21_re * o21_re;
    a22_re -= c22_21_im * o21_im;
    a22_im += c22_21_re * o21_im;
    a22_im += c22_21_im * o21_re;
    a22_re += c22_22_re * o22_re;
    a22_im += c22_22_re * o22_im;
    a22_re += c22_30_re * o30_re;
    a22_re -= c22_30_im * o30_im;
    a22_im += c22_30_re * o30_im;
    a22_im += c22_30_im * o30_re;
    a22_re += c22_31_re * o31_re;
    a22_re -= c22_31_im * o31_im;
    a22_im += c22_31_re * o31_im;
    a22_im += c22_31_im * o31_re;
    a22_re += c22_32_re * o32_re;
    a22_re -= c22_32_im * o32_im;
    a22_im += c22_32_re * o32_im;
    a22_im += c22_32_im * o32_re;
    
    a30_re += c30_20_re * o20_re;
    a30_re -= c30_20_im * o20_im;
    a30_im += c30_20_re * o20_im;
    a30_im += c30_20_im * o20_re;
    a30_re += c30_21_re * o21_re;
    a30_re -= c30_21_im * o21_im;
    a30_im += c30_21_re * o21_im;
    a30_im += c30_21_im * o21_re;
    a30_re += c30_22_re * o22_re;
    a30_re -= c30_22_im * o22_im;
    a30_im += c30_22_re * o22_im;
    a30_im += c30_22_im * o22_re;
    a30_re += c30_30_re * o30_re;
    a30_im += c30_30_re * o30_im;
    a30_re += c30_31_re * o31_re;
    a30_re -= c30_31_im * o31_im;
    a30_im += c30_31_re * o31_im;
    a30_im += c30_31_im * o31_re;
    a30_re += c30_32_re * o32_re;
    a30_re -= c30_32_im * o32_im;
    a30_im += c30_32_re * o32_im;
    a30_im += c30_32_im * o32_re;
    
    a31_re += c31_20_re * o20_re;
    a31_re -= c31_20_im * o20_im;
    a31_im += c31_20_re * o20_im;
    a31_im += c31_20_im * o20_re;
    a31_re += c31_21_re * o21_re;
    a31_re -= c31_21_im * o21_im;
    a31_im += c31_21_re * o21_im;
    a31_im += c31_21_im * o21_re;
    a31_re += c31_22_re * o22_re;
    a31_re -= c31_22_im * o22_im;
    a31_im += c31_22_re * o22_im;
    a31_im += c31_22_im * o22_re;
    a31_re += c31_30_re * o30_re;
    a31_re -= c31_30_im * o30_im;
    a31_im += c31_30_re * o30_im;
    a31_im += c31_30_im * o30_re;
    a31_re += c31_31_re * o31_re;
    a31_im += c31_31_re * o31_im;
    a31_re += c31_32_re * o32_re;
    a31_re -= c31_32_im * o32_im;
    a31_im += c31_32_re * o32_im;
    a31_im += c31_32_im * o32_re;
    
    a32_re += c32_20_re * o20_re;
    a32_re -= c32_20_im * o20_im;
    a32_im += c32_20_re * o20_im;
    a32_im += c32_20_im * o20_re;
    a32_re += c32_21_re * o21_re;
    a32_re -= c32_21_im * o21_im;
    a32_im += c32_21_re * o21_im;
    a32_im += c32_21_im * o21_re;
    a32_re += c32_22_re * o22_re;
    a32_re -= c32_22_im * o22_im;
    a32_im += c32_22_re * o22_im;
    a32_im += c32_22_im * o22_re;
    a32_re += c32_30_re * o30_re;
    a32_re -= c32_30_im * o30_im;
    a32_im += c32_30_re * o30_im;
    a32_im += c32_30_im * o30_re;
    a32_re += c32_31_re * o31_re;
    a32_re -= c32_31_im * o31_im;
    a32_im += c32_31_re * o31_im;
    a32_im += c32_31_im * o31_re;
    a32_re += c32_32_re * o32_re;
    a32_im += c32_32_re * o32_im;
    
    o20_re = a20_re;  o20_im = a20_im;
    o21_re = a21_re;  o21_im = a21_im;
    o22_re = a22_re;  o22_im = a22_im;
    o30_re = a30_re;  o30_im = a30_im;
    o31_re = a31_re;  o31_im = a31_im;
    o32_re = a32_re;  o32_im = a32_im;
    
  }
  
  // change back from chiral basis
  // (note: required factor of 1/2 is included in clover term normalization)
  {
    spinorFloat a00_re =  o10_re + o30_re;
    spinorFloat a00_im =  o10_im + o30_im;
    spinorFloat a10_re = -o00_re - o20_re;
    spinorFloat a10_im = -o00_im - o20_im;
    spinorFloat a20_re =  o10_re - o30_re;
    spinorFloat a20_im =  o10_im - o30_im;
    spinorFloat a30_re = -o00_re + o20_re;
    spinorFloat a30_im = -o00_im + o20_im;
    
    o00_re = a00_re;  o00_im = a00_im;
    o10_re = a10_re;  o10_im = a10_im;
    o20_re = a20_re;  o20_im = a20_im;
    o30_re = a30_re;  o30_im = a30_im;
  }
  
  {
    spinorFloat a01_re =  o11_re + o31_re;
    spinorFloat a01_im =  o11_im + o31_im;
    spinorFloat a11_re = -o01_re - o21_re;
    spinorFloat a11_im = -o01_im - o21_im;
    spinorFloat a21_re =  o11_re - o31_re;
    spinorFloat a21_im =  o11_im - o31_im;
    spinorFloat a31_re = -o01_re + o21_re;
    spinorFloat a31_im = -o01_im + o21_im;
    
    o01_re = a01_re;  o01_im = a01_im;
    o11_re = a11_re;  o11_im = a11_im;
    o21_re = a21_re;  o21_im = a21_im;
    o31_re = a31_re;  o31_im = a31_im;
  }
  
  {
    spinorFloat a02_re =  o12_re + o32_re;
    spinorFloat a02_im =  o12_im + o32_im;
    spinorFloat a12_re = -o02_re - o22_re;
    spinorFloat a12_im = -o02_im - o22_im;
    spinorFloat a22_re =  o12_re - o32_re;
    spinorFloat a22_im =  o12_im - o32_im;
    spinorFloat a32_re = -o02_re + o22_re;
    spinorFloat a32_im = -o02_im + o22_im;
    
    o02_re = a02_re;  o02_im = a02_im;
    o12_re = a12_re;  o12_im = a12_im;
    o22_re = a22_re;  o22_im = a22_im;
    o32_re = a32_re;  o32_im = a32_im;
  }
  
#ifdef DSLASH_XPAY
  
  READ_ACCUM(ACCUMTEX, sp_stride)
  
#ifdef SPINOR_DOUBLE
  o00_re = a*o00_re + accum0.x;
  o00_im = a*o00_im + accum0.y;
  o01_re = a*o01_re + accum1.x;
  o01_im = a*o01_im + accum1.y;
  o02_re = a*o02_re + accum2.x;
  o02_im = a*o02_im + accum2.y;
  o10_re = a*o10_re + accum3.x;
  o10_im = a*o10_im + accum3.y;
  o11_re = a*o11_re + accum4.x;
  o11_im = a*o11_im + accum4.y;
  o12_re = a*o12_re + accum5.x;
  o12_im = a*o12_im + accum5.y;
  o20_re = a*o20_re + accum6.x;
  o20_im = a*o20_im + accum6.y;
  o21_re = a*o21_re + accum7.x;
  o21_im = a*o21_im + accum7.y;
  o22_re = a*o22_re + accum8.x;
  o22_im = a*o22_im + accum8.y;
  o30_re = a*o30_re + accum9.x;
  o30_im = a*o30_im + accum9.y;
  o31_re = a*o31_re + accum10.x;
  o31_im = a*o31_im + accum10.y;
  o32_re = a*o32_re + accum11.x;
  o32_im = a*o32_im + accum11.y;
#else
  o00_re = a*o00_re + accum0.x;
  o00_im = a*o00_im + accum0.y;
  o01_re = a*o01_re + accum0.z;
  o01_im = a*o01_im + accum0.w;
  o02_re = a*o02_re + accum1.x;
  o02_im = a*o02_im + accum1.y;
  o10_re = a*o10_re + accum1.z;
  o10_im = a*o10_im + accum1.w;
  o11_re = a*o11_re + accum2.x;
  o11_im = a*o11_im + accum2.y;
  o12_re = a*o12_re + accum2.z;
  o12_im = a*o12_im + accum2.w;
  o20_re = a*o20_re + accum3.x;
  o20_im = a*o20_im + accum3.y;
  o21_re = a*o21_re + accum3.z;
  o21_im = a*o21_im + accum3.w;
  o22_re = a*o22_re + accum4.x;
  o22_im = a*o22_im + accum4.y;
  o30_re = a*o30_re + accum4.z;
  o30_im = a*o30_im + accum4.w;
  o31_re = a*o31_re + accum5.x;
  o31_im = a*o31_im + accum5.y;
  o32_re = a*o32_re + accum5.z;
  o32_im = a*o32_im + accum5.w;
#endif // SPINOR_DOUBLE
  
#endif // DSLASH_XPAY
}

// write spinor field back to device memory
WRITE_SPINOR(sp_stride);

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

#undef c00_00_re
#undef c01_01_re
#undef c02_02_re
#undef c10_10_re
#undef c11_11_re
#undef c12_12_re
#undef c01_00_re
#undef c01_00_im
#undef c02_00_re
#undef c02_00_im
#undef c10_00_re
#undef c10_00_im
#undef c11_00_re
#undef c11_00_im
#undef c12_00_re
#undef c12_00_im
#undef c02_01_re
#undef c02_01_im
#undef c10_01_re
#undef c10_01_im
#undef c11_01_re
#undef c11_01_im
#undef c12_01_re
#undef c12_01_im
#undef c10_02_re
#undef c10_02_im
#undef c11_02_re
#undef c11_02_im
#undef c12_02_re
#undef c12_02_im
#undef c11_10_re
#undef c11_10_im
#undef c12_10_re
#undef c12_10_im
#undef c12_11_re
#undef c12_11_im


#undef VOLATILE
