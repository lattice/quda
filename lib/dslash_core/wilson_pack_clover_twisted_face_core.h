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

// first chiral block of clover term
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
#ifdef CLOVER_DOUBLE
#define cinv00_00_re C0.x
#define cinv01_01_re C0.y
#define cinv02_02_re C1.x
#define cinv10_10_re C1.y
#define cinv11_11_re C2.x
#define cinv12_12_re C2.y
#define cinv01_00_re C3.x
#define cinv01_00_im C3.y
#define cinv02_00_re C4.x
#define cinv02_00_im C4.y
#define cinv10_00_re C5.x
#define cinv10_00_im C5.y
#define cinv11_00_re C6.x
#define cinv11_00_im C6.y
#define cinv12_00_re C7.x
#define cinv12_00_im C7.y
#define cinv02_01_re C8.x
#define cinv02_01_im C8.y
#define cinv10_01_re C9.x
#define cinv10_01_im C9.y
#define cinv11_01_re C10.x
#define cinv11_01_im C10.y
#define cinv12_01_re C11.x
#define cinv12_01_im C11.y
#define cinv10_02_re C12.x
#define cinv10_02_im C12.y
#define cinv11_02_re C13.x
#define cinv11_02_im C13.y
#define cinv12_02_re C14.x
#define cinv12_02_im C14.y
#define cinv11_10_re C15.x
#define cinv11_10_im C15.y
#define cinv12_10_re C16.x
#define cinv12_10_im C16.y
#define cinv12_11_re C17.x
#define cinv12_11_im C17.y
#else
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
#endif // CLOVER_DOUBLE

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

#include "io_spinor.h"
#include "read_clover.h"
#include "tmc_core.h"

if (face_num) {
  
  switch(dim) {
  case 0:
    {
      // read spinor from device memory
      READ_SPINOR(SPINORTEX, sp_stride, idx, idx);
      APPLY_CLOVER_TWIST_INV(c, cinv, a, i);
      
      spinorFloat a0_re, a0_im;
      spinorFloat a1_re, a1_im;
      spinorFloat a2_re, a2_im;
      spinorFloat b0_re, b0_im;
      spinorFloat b1_re, b1_im;
      spinorFloat b2_re, b2_im;
      
      // project spinor into half spinors
      a0_re = +i00_re-i30_im;
      a0_im = +i00_im+i30_re;
      a1_re = +i01_re-i31_im;
      a1_im = +i01_im+i31_re;
      a2_re = +i02_re-i32_im;
      a2_im = +i02_im+i32_re;
      b0_re = +i10_re-i20_im;
      b0_im = +i10_im+i20_re;
      b1_re = +i11_re-i21_im;
      b1_im = +i11_im+i21_re;
      b2_re = +i12_re-i22_im;
      b2_im = +i12_im+i22_re;
      
      // write half spinor back to device memory
      WRITE_HALF_SPINOR(face_volume, face_idx);
    }
    break;
  case 1:
    {
      // read spinor from device memory
      READ_SPINOR(SPINORTEX, sp_stride, idx, idx);
      APPLY_CLOVER_TWIST_INV(c, cinv, a, i);
      
      spinorFloat a0_re, a0_im;
      spinorFloat a1_re, a1_im;
      spinorFloat a2_re, a2_im;
      spinorFloat b0_re, b0_im;
      spinorFloat b1_re, b1_im;
      spinorFloat b2_re, b2_im;
      
      // project spinor into half spinors
      a0_re = +i00_re+i30_re;
      a0_im = +i00_im+i30_im;
      a1_re = +i01_re+i31_re;
      a1_im = +i01_im+i31_im;
      a2_re = +i02_re+i32_re;
      a2_im = +i02_im+i32_im;
      b0_re = +i10_re-i20_re;
      b0_im = +i10_im-i20_im;
      b1_re = +i11_re-i21_re;
      b1_im = +i11_im-i21_im;
      b2_re = +i12_re-i22_re;
      b2_im = +i12_im-i22_im;
      
      // write half spinor back to device memory
      WRITE_HALF_SPINOR(face_volume, face_idx);
    }
    break;
  case 2:
    {
      // read spinor from device memory
      READ_SPINOR(SPINORTEX, sp_stride, idx, idx);
      APPLY_CLOVER_TWIST_INV(c, cinv, a, i);
      
      spinorFloat a0_re, a0_im;
      spinorFloat a1_re, a1_im;
      spinorFloat a2_re, a2_im;
      spinorFloat b0_re, b0_im;
      spinorFloat b1_re, b1_im;
      spinorFloat b2_re, b2_im;
      
      // project spinor into half spinors
      a0_re = +i00_re-i20_im;
      a0_im = +i00_im+i20_re;
      a1_re = +i01_re-i21_im;
      a1_im = +i01_im+i21_re;
      a2_re = +i02_re-i22_im;
      a2_im = +i02_im+i22_re;
      b0_re = +i10_re+i30_im;
      b0_im = +i10_im-i30_re;
      b1_re = +i11_re+i31_im;
      b1_im = +i11_im-i31_re;
      b2_re = +i12_re+i32_im;
      b2_im = +i12_im-i32_re;
      
      // write half spinor back to device memory
      WRITE_HALF_SPINOR(face_volume, face_idx);
    }
    break;
  case 3:
    {
      // read spinor from device memory
      READ_SPINOR(SPINORTEX, sp_stride, idx, idx);
      APPLY_CLOVER_TWIST_INV(c, cinv, a, i);
      
      spinorFloat a0_re, a0_im;
      spinorFloat a1_re, a1_im;
      spinorFloat a2_re, a2_im;
      spinorFloat b0_re, b0_im;
      spinorFloat b1_re, b1_im;
      spinorFloat b2_re, b2_im;
      
      // project spinor into half spinors
      a0_re = +2*i00_re;
      a0_im = +2*i00_im;
      a1_re = +2*i01_re;
      a1_im = +2*i01_im;
      a2_re = +2*i02_re;
      a2_im = +2*i02_im;
      b0_re = +2*i10_re;
      b0_im = +2*i10_im;
      b1_re = +2*i11_re;
      b1_im = +2*i11_im;
      b2_re = +2*i12_re;
      b2_im = +2*i12_im;
      
      // write half spinor back to device memory
      WRITE_HALF_SPINOR(face_volume, face_idx);
    }
    break;
  }
  
} else {
  
  switch(dim) {
  case 0:
    {
      // read spinor from device memory
      READ_SPINOR(SPINORTEX, sp_stride, idx, idx);
      APPLY_CLOVER_TWIST_INV(c, cinv, a, i);
      
      spinorFloat a0_re, a0_im;
      spinorFloat a1_re, a1_im;
      spinorFloat a2_re, a2_im;
      spinorFloat b0_re, b0_im;
      spinorFloat b1_re, b1_im;
      spinorFloat b2_re, b2_im;
      
      // project spinor into half spinors
      a0_re = +i00_re+i30_im;
      a0_im = +i00_im-i30_re;
      a1_re = +i01_re+i31_im;
      a1_im = +i01_im-i31_re;
      a2_re = +i02_re+i32_im;
      a2_im = +i02_im-i32_re;
      b0_re = +i10_re+i20_im;
      b0_im = +i10_im-i20_re;
      b1_re = +i11_re+i21_im;
      b1_im = +i11_im-i21_re;
      b2_re = +i12_re+i22_im;
      b2_im = +i12_im-i22_re;
      
      // write half spinor back to device memory
      WRITE_HALF_SPINOR(face_volume, face_idx);
    }
    break;
  case 1:
    {
      // read spinor from device memory
      READ_SPINOR(SPINORTEX, sp_stride, idx, idx);
      APPLY_CLOVER_TWIST_INV(c, cinv, a, i);
      
      spinorFloat a0_re, a0_im;
      spinorFloat a1_re, a1_im;
      spinorFloat a2_re, a2_im;
      spinorFloat b0_re, b0_im;
      spinorFloat b1_re, b1_im;
      spinorFloat b2_re, b2_im;
      
      // project spinor into half spinors
      a0_re = +i00_re-i30_re;
      a0_im = +i00_im-i30_im;
      a1_re = +i01_re-i31_re;
      a1_im = +i01_im-i31_im;
      a2_re = +i02_re-i32_re;
      a2_im = +i02_im-i32_im;
      b0_re = +i10_re+i20_re;
      b0_im = +i10_im+i20_im;
      b1_re = +i11_re+i21_re;
      b1_im = +i11_im+i21_im;
      b2_re = +i12_re+i22_re;
      b2_im = +i12_im+i22_im;
      
      // write half spinor back to device memory
      WRITE_HALF_SPINOR(face_volume, face_idx);
    }
    break;
  case 2:
    {
      // read spinor from device memory
      READ_SPINOR(SPINORTEX, sp_stride, idx, idx);
      APPLY_CLOVER_TWIST_INV(c, cinv, a, i);
      
      spinorFloat a0_re, a0_im;
      spinorFloat a1_re, a1_im;
      spinorFloat a2_re, a2_im;
      spinorFloat b0_re, b0_im;
      spinorFloat b1_re, b1_im;
      spinorFloat b2_re, b2_im;
      
      // project spinor into half spinors
      a0_re = +i00_re+i20_im;
      a0_im = +i00_im-i20_re;
      a1_re = +i01_re+i21_im;
      a1_im = +i01_im-i21_re;
      a2_re = +i02_re+i22_im;
      a2_im = +i02_im-i22_re;
      b0_re = +i10_re-i30_im;
      b0_im = +i10_im+i30_re;
      b1_re = +i11_re-i31_im;
      b1_im = +i11_im+i31_re;
      b2_re = +i12_re-i32_im;
      b2_im = +i12_im+i32_re;
      
      // write half spinor back to device memory
      WRITE_HALF_SPINOR(face_volume, face_idx);
    }
    break;
  case 3:
    {
      // read spinor from device memory
      READ_SPINOR(SPINORTEX, sp_stride, idx, idx);
      APPLY_CLOVER_TWIST_INV(c, cinv, a, i);
      
      spinorFloat a0_re, a0_im;
      spinorFloat a1_re, a1_im;
      spinorFloat a2_re, a2_im;
      spinorFloat b0_re, b0_im;
      spinorFloat b1_re, b1_im;
      spinorFloat b2_re, b2_im;
      
      // project spinor into half spinors
      a0_re = +2*i20_re;
      a0_im = +2*i20_im;
      a1_re = +2*i21_re;
      a1_im = +2*i21_im;
      a2_re = +2*i22_re;
      a2_im = +2*i22_im;
      b0_re = +2*i30_re;
      b0_im = +2*i30_im;
      b1_re = +2*i31_re;
      b1_im = +2*i31_im;
      b2_re = +2*i32_re;
      b2_im = +2*i32_im;
      
      // write half spinor back to device memory
      WRITE_HALF_SPINOR(face_volume, face_idx);
    }
    break;
  }
  
}

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

#undef c20_20_re
#undef c21_21_re
#undef c22_22_re
#undef c30_30_re
#undef c31_31_re
#undef c32_32_re
#undef c21_20_re
#undef c21_20_im
#undef c22_20_re
#undef c22_20_im
#undef c30_20_re
#undef c30_20_im
#undef c31_20_re
#undef c31_20_im
#undef c32_20_re
#undef c32_20_im
#undef c22_21_re
#undef c22_21_im
#undef c30_21_re
#undef c30_21_im
#undef c31_21_re
#undef c31_21_im
#undef c32_21_re
#undef c32_21_im
#undef c30_22_re
#undef c30_22_im
#undef c31_22_re
#undef c31_22_im
#undef c32_22_re
#undef c32_22_im
#undef c31_30_re
#undef c31_30_im
#undef c32_30_re
#undef c32_30_im
#undef c32_31_re
#undef c32_31_im

#undef cinv00_00_re
#undef cinv01_01_re
#undef cinv02_02_re
#undef cinv10_10_re
#undef cinv11_11_re
#undef cinv12_12_re
#undef cinv01_00_re
#undef cinv01_00_im
#undef cinv02_00_re
#undef cinv02_00_im
#undef cinv10_00_re
#undef cinv10_00_im
#undef cinv11_00_re
#undef cinv11_00_im
#undef cinv12_00_re
#undef cinv12_00_im
#undef cinv02_01_re
#undef cinv02_01_im
#undef cinv10_01_re
#undef cinv10_01_im
#undef cinv11_01_re
#undef cinv11_01_im
#undef cinv12_01_re
#undef cinv12_01_im
#undef cinv10_02_re
#undef cinv10_02_im
#undef cinv11_02_re
#undef cinv11_02_im
#undef cinv12_02_re
#undef cinv12_02_im
#undef cinv11_10_re
#undef cinv11_10_im
#undef cinv12_10_re
#undef cinv12_10_im
#undef cinv12_11_re
#undef cinv12_11_im

#undef cinv20_20_re
#undef cinv21_21_re
#undef cinv22_22_re
#undef cinv30_30_re
#undef cinv31_31_re
#undef cinv32_32_re
#undef cinv21_20_re
#undef cinv21_20_im
#undef cinv22_20_re
#undef cinv22_20_im
#undef cinv30_20_re
#undef cinv30_20_im
#undef cinv31_20_re
#undef cinv31_20_im
#undef cinv32_20_re
#undef cinv32_20_im
#undef cinv22_21_re
#undef cinv22_21_im
#undef cinv30_21_re
#undef cinv30_21_im
#undef cinv31_21_re
#undef cinv31_21_im
#undef cinv32_21_re
#undef cinv32_21_im
#undef cinv30_22_re
#undef cinv30_22_im
#undef cinv31_22_re
#undef cinv31_22_im
#undef cinv32_22_re
#undef cinv32_22_im
#undef cinv31_30_re
#undef cinv31_30_im
#undef cinv32_30_re
#undef cinv32_30_im
#undef cinv32_31_re
#undef cinv32_31_im
