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

#include "io_spinor.h"

if (face_num) {
  
  switch(dim) {
  case 0:
    {
      // read spinor from device memory
      READ_SPINOR(SPINORTEX, sp_stride, idx, idx);
      APPLY_TWIST_INV(-a, b, i);
      
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
      APPLY_TWIST_INV(-a, b, i);
      
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
      APPLY_TWIST_INV(-a, b, i);
      
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
      APPLY_TWIST_INV(-a, b, i);
      
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
  
} else {
  
  switch(dim) {
  case 0:
    {
      // read spinor from device memory
      READ_SPINOR(SPINORTEX, sp_stride, idx, idx);
      APPLY_TWIST_INV(-a, b, i);
      
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
      APPLY_TWIST_INV(-a, b, i);
      
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
      APPLY_TWIST_INV(-a, b, i);
      
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
      APPLY_TWIST_INV(-a, b, i);
      
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

