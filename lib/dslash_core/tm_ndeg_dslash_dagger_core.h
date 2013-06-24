// *** CUDA NDEG TWISTED MASS DSLASH DAGGER ***

// Arguments (double) mu, (double)eta and (double)delta 
#define SHARED_TMNDEG_FLOATS_PER_THREAD 0
#define FLAVORS 2


#if ((CUDA_VERSION >= 4010) && (__COMPUTE_CAPABILITY__ >= 200)) // NVVM compiler
#define VOLATILE
#else // Open64 compiler
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

// gauge link
#ifdef GAUGE_FLOAT2
#define g00_re G0.x
#define g00_im G0.y
#define g01_re G1.x
#define g01_im G1.y
#define g02_re G2.x
#define g02_im G2.y
#define g10_re G3.x
#define g10_im G3.y
#define g11_re G4.x
#define g11_im G4.y
#define g12_re G5.x
#define g12_im G5.y
#define g20_re G6.x
#define g20_im G6.y
#define g21_re G7.x
#define g21_im G7.y
#define g22_re G8.x
#define g22_im G8.y

#else
#define g00_re G0.x
#define g00_im G0.y
#define g01_re G0.z
#define g01_im G0.w
#define g02_re G1.x
#define g02_im G1.y
#define g10_re G1.z
#define g10_im G1.w
#define g11_re G2.x
#define g11_im G2.y
#define g12_re G2.z
#define g12_im G2.w
#define g20_re G3.x
#define g20_im G3.y
#define g21_re G3.z
#define g21_im G3.w
#define g22_re G4.x
#define g22_im G4.y

#endif // GAUGE_DOUBLE

// conjugated gauge link
#define gT00_re (+g00_re)
#define gT00_im (-g00_im)
#define gT01_re (+g10_re)
#define gT01_im (-g10_im)
#define gT02_re (+g20_re)
#define gT02_im (-g20_im)
#define gT10_re (+g01_re)
#define gT10_im (-g01_im)
#define gT11_re (+g11_re)
#define gT11_im (-g11_im)
#define gT12_re (+g21_re)
#define gT12_im (-g21_im)
#define gT20_re (+g02_re)
#define gT20_im (-g02_im)
#define gT21_re (+g12_re)
#define gT21_im (-g12_im)
#define gT22_re (+g22_re)
#define gT22_im (-g22_im)

// output spinor for flavor 1
VOLATILE spinorFloat o1_00_re;
VOLATILE spinorFloat o1_00_im;
VOLATILE spinorFloat o1_01_re;
VOLATILE spinorFloat o1_01_im;
VOLATILE spinorFloat o1_02_re;
VOLATILE spinorFloat o1_02_im;
VOLATILE spinorFloat o1_10_re;
VOLATILE spinorFloat o1_10_im;
VOLATILE spinorFloat o1_11_re;
VOLATILE spinorFloat o1_11_im;
VOLATILE spinorFloat o1_12_re;
VOLATILE spinorFloat o1_12_im;
VOLATILE spinorFloat o1_20_re;
VOLATILE spinorFloat o1_20_im;
VOLATILE spinorFloat o1_21_re;
VOLATILE spinorFloat o1_21_im;
VOLATILE spinorFloat o1_22_re;
VOLATILE spinorFloat o1_22_im;
VOLATILE spinorFloat o1_30_re;
VOLATILE spinorFloat o1_30_im;
VOLATILE spinorFloat o1_31_re;
VOLATILE spinorFloat o1_31_im;
VOLATILE spinorFloat o1_32_re;
VOLATILE spinorFloat o1_32_im;
// output spinor for flavor 2
VOLATILE spinorFloat o2_00_re;
VOLATILE spinorFloat o2_00_im;
VOLATILE spinorFloat o2_01_re;
VOLATILE spinorFloat o2_01_im;
VOLATILE spinorFloat o2_02_re;
VOLATILE spinorFloat o2_02_im;
VOLATILE spinorFloat o2_10_re;
VOLATILE spinorFloat o2_10_im;
VOLATILE spinorFloat o2_11_re;
VOLATILE spinorFloat o2_11_im;
VOLATILE spinorFloat o2_12_re;
VOLATILE spinorFloat o2_12_im;
VOLATILE spinorFloat o2_20_re;
VOLATILE spinorFloat o2_20_im;
VOLATILE spinorFloat o2_21_re;
VOLATILE spinorFloat o2_21_im;
VOLATILE spinorFloat o2_22_re;
VOLATILE spinorFloat o2_22_im;
VOLATILE spinorFloat o2_30_re;
VOLATILE spinorFloat o2_30_im;
VOLATILE spinorFloat o2_31_re;
VOLATILE spinorFloat o2_31_im;
VOLATILE spinorFloat o2_32_re;
VOLATILE spinorFloat o2_32_im;

#include "read_gauge.h"
#include "io_spinor.h"

int x1, x2, x3, x4;
int X;

#if (defined MULTI_GPU) && (DD_PREC==2) // half precision
int sp_norm_idx;
#endif // MULTI_GPU half precision

int sid;

#ifdef MULTI_GPU
int face_idx;
if (kernel_type == INTERIOR_KERNEL) {
#endif

  sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= param.threads) return;

  // Inline by hand for the moment and assume even dimensions
  coordsFromIndex<EVEN_X>(X, x1, x2, x3, x4, sid, param.parity);

  o1_00_re = 0;  o1_00_im = 0;
  o1_01_re = 0;  o1_01_im = 0;
  o1_02_re = 0;  o1_02_im = 0;
  o1_10_re = 0;  o1_10_im = 0;
  o1_11_re = 0;  o1_11_im = 0;
  o1_12_re = 0;  o1_12_im = 0;
  o1_20_re = 0;  o1_20_im = 0;
  o1_21_re = 0;  o1_21_im = 0;
  o1_22_re = 0;  o1_22_im = 0;
  o1_30_re = 0;  o1_30_im = 0;
  o1_31_re = 0;  o1_31_im = 0;
  o1_32_re = 0;  o1_32_im = 0;
  
  o2_00_re = 0;  o2_00_im = 0;
  o2_01_re = 0;  o2_01_im = 0;
  o2_02_re = 0;  o2_02_im = 0;
  o2_10_re = 0;  o2_10_im = 0;
  o2_11_re = 0;  o2_11_im = 0;
  o2_12_re = 0;  o2_12_im = 0;
  o2_20_re = 0;  o2_20_im = 0;
  o2_21_re = 0;  o2_21_im = 0;
  o2_22_re = 0;  o2_22_im = 0;
  o2_30_re = 0;  o2_30_im = 0;
  o2_31_re = 0;  o2_31_im = 0;
  o2_32_re = 0;  o2_32_im = 0;

#ifdef MULTI_GPU
} else { // exterior kernel

  sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= param.threads) return;

  const int dim = static_cast<int>(kernel_type);
  const int face_volume = (param.threads >> 1);           // volume of one face (per flavor)
  const int face_num = (sid >= face_volume);              // is this thread updating face 0 or 1
  face_idx = sid - face_num*face_volume;        // index into the respective face

  // ghostOffset is scaled to include body (includes stride) and number of FloatN arrays (SPINOR_HOP)
  // face_idx not sid since faces are spin projected and share the same volume index (modulo UP/DOWN reading)
  //sp_idx = face_idx + param.ghostOffset[dim];

#if (DD_PREC==2) // half precision
  sp_norm_idx = sid + param.ghostNormOffset[static_cast<int>(kernel_type)] + face_num*ghostFace[static_cast<int>(kernel_type)];
#endif

  coordsFromFaceIndex<1>(X, sid, x1, x2, x3, x4, face_idx, face_volume, dim, face_num, param.parity);


  {
     READ_INTERMEDIATE_SPINOR(INTERTEX, sp_stride, sid, sid);
     o1_00_re = i00_re;  o1_00_im = i00_im;
     o1_01_re = i01_re;  o1_01_im = i01_im;
     o1_02_re = i02_re;  o1_02_im = i02_im;
     o1_10_re = i10_re;  o1_10_im = i10_im;
     o1_11_re = i11_re;  o1_11_im = i11_im;
     o1_12_re = i12_re;  o1_12_im = i12_im;
     o1_20_re = i20_re;  o1_20_im = i20_im;
     o1_21_re = i21_re;  o1_21_im = i21_im;
     o1_22_re = i22_re;  o1_22_im = i22_im;
     o1_30_re = i30_re;  o1_30_im = i30_im;
     o1_31_re = i31_re;  o1_31_im = i31_im;
     o1_32_re = i32_re;  o1_32_im = i32_im;
     

  }
  {
     READ_INTERMEDIATE_SPINOR(INTERTEX, sp_stride, sid+fl_stride, sid+fl_stride);
     o2_00_re = i00_re;  o2_00_im = i00_im;
     o2_01_re = i01_re;  o2_01_im = i01_im;
     o2_02_re = i02_re;  o2_02_im = i02_im;
     o2_10_re = i10_re;  o2_10_im = i10_im;
     o2_11_re = i11_re;  o2_11_im = i11_im;
     o2_12_re = i12_re;  o2_12_im = i12_im;
     o2_20_re = i20_re;  o2_20_im = i20_im;
     o2_21_re = i21_re;  o2_21_im = i21_im;
     o2_22_re = i22_re;  o2_22_im = i22_im;
     o2_30_re = i30_re;  o2_30_im = i30_im;
     o2_31_re = i31_re;  o2_31_im = i31_im;
     o2_32_re = i32_re;  o2_32_im = i32_im;
     

  }
}
#endif // MULTI_GPU


#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[0] || x1<X1m1)) ||
     (kernel_type == EXTERIOR_KERNEL_X && x1==X1m1) )
#endif
{
  // Projector P0+
  // 1 0 0 i 
  // 0 1 i 0 
  // 0 -i 1 0 
  // -i 0 0 1 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x1==X1m1 ? X-X1m1 : X+1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x1==X1m1 ? X-X1m1 : X+1) >> 1;
#endif
  
  const int ga_idx = sid;
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE0TEX, 0, ga_idx, ga_stride);
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(0);
  
  {
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read flavor 1 from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
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
  
#ifdef MULTI_GPU
  } else {
  
  const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
  
    // read half spinor for the first flavor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += g00_re * a0_re;
  A0_re -= g00_im * a0_im;
  A0_re += g01_re * a1_re;
  A0_re -= g01_im * a1_im;
  A0_re += g02_re * a2_re;
  A0_re -= g02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += g00_re * a0_im;
  A0_im += g00_im * a0_re;
  A0_im += g01_re * a1_im;
  A0_im += g01_im * a1_re;
  A0_im += g02_re * a2_im;
  A0_im += g02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += g00_re * b0_re;
  B0_re -= g00_im * b0_im;
  B0_re += g01_re * b1_re;
  B0_re -= g01_im * b1_im;
  B0_re += g02_re * b2_re;
  B0_re -= g02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += g00_re * b0_im;
  B0_im += g00_im * b0_re;
  B0_im += g01_re * b1_im;
  B0_im += g01_im * b1_re;
  B0_im += g02_re * b2_im;
  B0_im += g02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += g10_re * a0_re;
  A1_re -= g10_im * a0_im;
  A1_re += g11_re * a1_re;
  A1_re -= g11_im * a1_im;
  A1_re += g12_re * a2_re;
  A1_re -= g12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += g10_re * a0_im;
  A1_im += g10_im * a0_re;
  A1_im += g11_re * a1_im;
  A1_im += g11_im * a1_re;
  A1_im += g12_re * a2_im;
  A1_im += g12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += g10_re * b0_re;
  B1_re -= g10_im * b0_im;
  B1_re += g11_re * b1_re;
  B1_re -= g11_im * b1_im;
  B1_re += g12_re * b2_re;
  B1_re -= g12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += g10_re * b0_im;
  B1_im += g10_im * b0_re;
  B1_im += g11_re * b1_im;
  B1_im += g11_im * b1_re;
  B1_im += g12_re * b2_im;
  B1_im += g12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += g20_re * a0_re;
  A2_re -= g20_im * a0_im;
  A2_re += g21_re * a1_re;
  A2_re -= g21_im * a1_im;
  A2_re += g22_re * a2_re;
  A2_re -= g22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += g20_re * a0_im;
  A2_im += g20_im * a0_re;
  A2_im += g21_re * a1_im;
  A2_im += g21_im * a1_re;
  A2_im += g22_re * a2_im;
  A2_im += g22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += g20_re * b0_re;
  B2_re -= g20_im * b0_im;
  B2_re += g21_re * b1_re;
  B2_re -= g21_im * b1_im;
  B2_re += g22_re * b2_re;
  B2_re -= g22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += g20_re * b0_im;
  B2_im += g20_im * b0_re;
  B2_im += g21_re * b1_im;
  B2_im += g21_im * b1_re;
  B2_im += g22_re * b2_im;
  B2_im += g22_im * b2_re;
  
  o1_00_re += A0_re;
  o1_00_im += A0_im;
  o1_10_re += B0_re;
  o1_10_im += B0_im;
  o1_20_re += B0_im;
  o1_20_im -= B0_re;
  o1_30_re += A0_im;
  o1_30_im -= A0_re;
  
  o1_01_re += A1_re;
  o1_01_im += A1_im;
  o1_11_re += B1_re;
  o1_11_im += B1_im;
  o1_21_re += B1_im;
  o1_21_im -= B1_re;
  o1_31_re += A1_im;
  o1_31_im -= A1_re;
  
  o1_02_re += A2_re;
  o1_02_im += A2_im;
  o1_12_re += B2_re;
  o1_12_im += B2_im;
  o1_22_re += B2_im;
  o1_22_im -= B2_re;
  o1_32_re += A2_im;
  o1_32_im -= A2_re;
  
  }
  {
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read flavor 2 from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
    
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
  
#ifdef MULTI_GPU
  } else {
  
  const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
  
    // read half spinor for the second flavor from device memory
    const int fl_idx = sp_idx + ghostFace[static_cast<int>(kernel_type)];
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, fl_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx+ghostFace[static_cast<int>(kernel_type)]);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += g00_re * a0_re;
  A0_re -= g00_im * a0_im;
  A0_re += g01_re * a1_re;
  A0_re -= g01_im * a1_im;
  A0_re += g02_re * a2_re;
  A0_re -= g02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += g00_re * a0_im;
  A0_im += g00_im * a0_re;
  A0_im += g01_re * a1_im;
  A0_im += g01_im * a1_re;
  A0_im += g02_re * a2_im;
  A0_im += g02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += g00_re * b0_re;
  B0_re -= g00_im * b0_im;
  B0_re += g01_re * b1_re;
  B0_re -= g01_im * b1_im;
  B0_re += g02_re * b2_re;
  B0_re -= g02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += g00_re * b0_im;
  B0_im += g00_im * b0_re;
  B0_im += g01_re * b1_im;
  B0_im += g01_im * b1_re;
  B0_im += g02_re * b2_im;
  B0_im += g02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += g10_re * a0_re;
  A1_re -= g10_im * a0_im;
  A1_re += g11_re * a1_re;
  A1_re -= g11_im * a1_im;
  A1_re += g12_re * a2_re;
  A1_re -= g12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += g10_re * a0_im;
  A1_im += g10_im * a0_re;
  A1_im += g11_re * a1_im;
  A1_im += g11_im * a1_re;
  A1_im += g12_re * a2_im;
  A1_im += g12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += g10_re * b0_re;
  B1_re -= g10_im * b0_im;
  B1_re += g11_re * b1_re;
  B1_re -= g11_im * b1_im;
  B1_re += g12_re * b2_re;
  B1_re -= g12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += g10_re * b0_im;
  B1_im += g10_im * b0_re;
  B1_im += g11_re * b1_im;
  B1_im += g11_im * b1_re;
  B1_im += g12_re * b2_im;
  B1_im += g12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += g20_re * a0_re;
  A2_re -= g20_im * a0_im;
  A2_re += g21_re * a1_re;
  A2_re -= g21_im * a1_im;
  A2_re += g22_re * a2_re;
  A2_re -= g22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += g20_re * a0_im;
  A2_im += g20_im * a0_re;
  A2_im += g21_re * a1_im;
  A2_im += g21_im * a1_re;
  A2_im += g22_re * a2_im;
  A2_im += g22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += g20_re * b0_re;
  B2_re -= g20_im * b0_im;
  B2_re += g21_re * b1_re;
  B2_re -= g21_im * b1_im;
  B2_re += g22_re * b2_re;
  B2_re -= g22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += g20_re * b0_im;
  B2_im += g20_im * b0_re;
  B2_im += g21_re * b1_im;
  B2_im += g21_im * b1_re;
  B2_im += g22_re * b2_im;
  B2_im += g22_im * b2_re;
  
  o2_00_re += A0_re;
  o2_00_im += A0_im;
  o2_10_re += B0_re;
  o2_10_im += B0_im;
  o2_20_re += B0_im;
  o2_20_im -= B0_re;
  o2_30_re += A0_im;
  o2_30_im -= A0_re;
  
  o2_01_re += A1_re;
  o2_01_im += A1_im;
  o2_11_re += B1_re;
  o2_11_im += B1_im;
  o2_21_re += B1_im;
  o2_21_im -= B1_re;
  o2_31_re += A1_im;
  o2_31_im -= A1_re;
  
  o2_02_re += A2_re;
  o2_02_im += A2_im;
  o2_12_re += B2_re;
  o2_12_im += B2_im;
  o2_22_re += B2_im;
  o2_22_im -= B2_re;
  o2_32_re += A2_im;
  o2_32_im -= A2_re;
  
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[0] || x1>0)) ||
     (kernel_type == EXTERIOR_KERNEL_X && x1==0) )
#endif
{
  // Projector P0-
  // 1 0 0 -i 
  // 0 1 -i 0 
  // 0 i 1 0 
  // i 0 0 1 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x1==0 ? X+X1m1 : X-1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x1==0 ? X+X1m1 : X-1) >> 1;
#endif
  
#ifdef MULTI_GPU
  const int ga_idx = ((kernel_type == INTERIOR_KERNEL) ? sp_idx : Vh+face_idx);
#else
  const int ga_idx = sp_idx;
#endif
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE1TEX, 1, ga_idx, ga_stride);
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(1);
  
  {
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read flavor 1 from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
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
  
#ifdef MULTI_GPU
  } else {
  
  const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
  
    // read half spinor for the first flavor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += gT00_re * a0_re;
  A0_re -= gT00_im * a0_im;
  A0_re += gT01_re * a1_re;
  A0_re -= gT01_im * a1_im;
  A0_re += gT02_re * a2_re;
  A0_re -= gT02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += gT00_re * a0_im;
  A0_im += gT00_im * a0_re;
  A0_im += gT01_re * a1_im;
  A0_im += gT01_im * a1_re;
  A0_im += gT02_re * a2_im;
  A0_im += gT02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += gT00_re * b0_re;
  B0_re -= gT00_im * b0_im;
  B0_re += gT01_re * b1_re;
  B0_re -= gT01_im * b1_im;
  B0_re += gT02_re * b2_re;
  B0_re -= gT02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += gT00_re * b0_im;
  B0_im += gT00_im * b0_re;
  B0_im += gT01_re * b1_im;
  B0_im += gT01_im * b1_re;
  B0_im += gT02_re * b2_im;
  B0_im += gT02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += gT10_re * a0_re;
  A1_re -= gT10_im * a0_im;
  A1_re += gT11_re * a1_re;
  A1_re -= gT11_im * a1_im;
  A1_re += gT12_re * a2_re;
  A1_re -= gT12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += gT10_re * a0_im;
  A1_im += gT10_im * a0_re;
  A1_im += gT11_re * a1_im;
  A1_im += gT11_im * a1_re;
  A1_im += gT12_re * a2_im;
  A1_im += gT12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += gT10_re * b0_re;
  B1_re -= gT10_im * b0_im;
  B1_re += gT11_re * b1_re;
  B1_re -= gT11_im * b1_im;
  B1_re += gT12_re * b2_re;
  B1_re -= gT12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += gT10_re * b0_im;
  B1_im += gT10_im * b0_re;
  B1_im += gT11_re * b1_im;
  B1_im += gT11_im * b1_re;
  B1_im += gT12_re * b2_im;
  B1_im += gT12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += gT20_re * a0_re;
  A2_re -= gT20_im * a0_im;
  A2_re += gT21_re * a1_re;
  A2_re -= gT21_im * a1_im;
  A2_re += gT22_re * a2_re;
  A2_re -= gT22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += gT20_re * a0_im;
  A2_im += gT20_im * a0_re;
  A2_im += gT21_re * a1_im;
  A2_im += gT21_im * a1_re;
  A2_im += gT22_re * a2_im;
  A2_im += gT22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += gT20_re * b0_re;
  B2_re -= gT20_im * b0_im;
  B2_re += gT21_re * b1_re;
  B2_re -= gT21_im * b1_im;
  B2_re += gT22_re * b2_re;
  B2_re -= gT22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += gT20_re * b0_im;
  B2_im += gT20_im * b0_re;
  B2_im += gT21_re * b1_im;
  B2_im += gT21_im * b1_re;
  B2_im += gT22_re * b2_im;
  B2_im += gT22_im * b2_re;
  
  o1_00_re += A0_re;
  o1_00_im += A0_im;
  o1_10_re += B0_re;
  o1_10_im += B0_im;
  o1_20_re -= B0_im;
  o1_20_im += B0_re;
  o1_30_re -= A0_im;
  o1_30_im += A0_re;
  
  o1_01_re += A1_re;
  o1_01_im += A1_im;
  o1_11_re += B1_re;
  o1_11_im += B1_im;
  o1_21_re -= B1_im;
  o1_21_im += B1_re;
  o1_31_re -= A1_im;
  o1_31_im += A1_re;
  
  o1_02_re += A2_re;
  o1_02_im += A2_im;
  o1_12_re += B2_re;
  o1_12_im += B2_im;
  o1_22_re -= B2_im;
  o1_22_im += B2_re;
  o1_32_re -= A2_im;
  o1_32_im += A2_re;
  
  }
  {
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read flavor 2 from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
    
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
  
#ifdef MULTI_GPU
  } else {
  
  const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
  
    // read half spinor for the second flavor from device memory
    const int fl_idx = sp_idx + ghostFace[static_cast<int>(kernel_type)];
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, fl_idx, sp_norm_idx+ghostFace[static_cast<int>(kernel_type)]);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += gT00_re * a0_re;
  A0_re -= gT00_im * a0_im;
  A0_re += gT01_re * a1_re;
  A0_re -= gT01_im * a1_im;
  A0_re += gT02_re * a2_re;
  A0_re -= gT02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += gT00_re * a0_im;
  A0_im += gT00_im * a0_re;
  A0_im += gT01_re * a1_im;
  A0_im += gT01_im * a1_re;
  A0_im += gT02_re * a2_im;
  A0_im += gT02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += gT00_re * b0_re;
  B0_re -= gT00_im * b0_im;
  B0_re += gT01_re * b1_re;
  B0_re -= gT01_im * b1_im;
  B0_re += gT02_re * b2_re;
  B0_re -= gT02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += gT00_re * b0_im;
  B0_im += gT00_im * b0_re;
  B0_im += gT01_re * b1_im;
  B0_im += gT01_im * b1_re;
  B0_im += gT02_re * b2_im;
  B0_im += gT02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += gT10_re * a0_re;
  A1_re -= gT10_im * a0_im;
  A1_re += gT11_re * a1_re;
  A1_re -= gT11_im * a1_im;
  A1_re += gT12_re * a2_re;
  A1_re -= gT12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += gT10_re * a0_im;
  A1_im += gT10_im * a0_re;
  A1_im += gT11_re * a1_im;
  A1_im += gT11_im * a1_re;
  A1_im += gT12_re * a2_im;
  A1_im += gT12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += gT10_re * b0_re;
  B1_re -= gT10_im * b0_im;
  B1_re += gT11_re * b1_re;
  B1_re -= gT11_im * b1_im;
  B1_re += gT12_re * b2_re;
  B1_re -= gT12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += gT10_re * b0_im;
  B1_im += gT10_im * b0_re;
  B1_im += gT11_re * b1_im;
  B1_im += gT11_im * b1_re;
  B1_im += gT12_re * b2_im;
  B1_im += gT12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += gT20_re * a0_re;
  A2_re -= gT20_im * a0_im;
  A2_re += gT21_re * a1_re;
  A2_re -= gT21_im * a1_im;
  A2_re += gT22_re * a2_re;
  A2_re -= gT22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += gT20_re * a0_im;
  A2_im += gT20_im * a0_re;
  A2_im += gT21_re * a1_im;
  A2_im += gT21_im * a1_re;
  A2_im += gT22_re * a2_im;
  A2_im += gT22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += gT20_re * b0_re;
  B2_re -= gT20_im * b0_im;
  B2_re += gT21_re * b1_re;
  B2_re -= gT21_im * b1_im;
  B2_re += gT22_re * b2_re;
  B2_re -= gT22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += gT20_re * b0_im;
  B2_im += gT20_im * b0_re;
  B2_im += gT21_re * b1_im;
  B2_im += gT21_im * b1_re;
  B2_im += gT22_re * b2_im;
  B2_im += gT22_im * b2_re;
  
  o2_00_re += A0_re;
  o2_00_im += A0_im;
  o2_10_re += B0_re;
  o2_10_im += B0_im;
  o2_20_re -= B0_im;
  o2_20_im += B0_re;
  o2_30_re -= A0_im;
  o2_30_im += A0_re;
  
  o2_01_re += A1_re;
  o2_01_im += A1_im;
  o2_11_re += B1_re;
  o2_11_im += B1_im;
  o2_21_re -= B1_im;
  o2_21_im += B1_re;
  o2_31_re -= A1_im;
  o2_31_im += A1_re;
  
  o2_02_re += A2_re;
  o2_02_im += A2_im;
  o2_12_re += B2_re;
  o2_12_im += B2_im;
  o2_22_re -= B2_im;
  o2_22_im += B2_re;
  o2_32_re -= A2_im;
  o2_32_im += A2_re;
  
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[1] || x2<X2m1)) ||
     (kernel_type == EXTERIOR_KERNEL_Y && x2==X2m1) )
#endif
{
  // Projector P1+
  // 1 0 0 1 
  // 0 1 -1 0 
  // 0 -1 1 0 
  // 1 0 0 1 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x2==X2m1 ? X-X2X1mX1 : X+X1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x2==X2m1 ? X-X2X1mX1 : X+X1) >> 1;
#endif
  
  const int ga_idx = sid;
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE0TEX, 2, ga_idx, ga_stride);
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(2);
  
  {
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read flavor 1 from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
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
  
#ifdef MULTI_GPU
  } else {
  
  const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
  
    // read half spinor for the first flavor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += g00_re * a0_re;
  A0_re -= g00_im * a0_im;
  A0_re += g01_re * a1_re;
  A0_re -= g01_im * a1_im;
  A0_re += g02_re * a2_re;
  A0_re -= g02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += g00_re * a0_im;
  A0_im += g00_im * a0_re;
  A0_im += g01_re * a1_im;
  A0_im += g01_im * a1_re;
  A0_im += g02_re * a2_im;
  A0_im += g02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += g00_re * b0_re;
  B0_re -= g00_im * b0_im;
  B0_re += g01_re * b1_re;
  B0_re -= g01_im * b1_im;
  B0_re += g02_re * b2_re;
  B0_re -= g02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += g00_re * b0_im;
  B0_im += g00_im * b0_re;
  B0_im += g01_re * b1_im;
  B0_im += g01_im * b1_re;
  B0_im += g02_re * b2_im;
  B0_im += g02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += g10_re * a0_re;
  A1_re -= g10_im * a0_im;
  A1_re += g11_re * a1_re;
  A1_re -= g11_im * a1_im;
  A1_re += g12_re * a2_re;
  A1_re -= g12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += g10_re * a0_im;
  A1_im += g10_im * a0_re;
  A1_im += g11_re * a1_im;
  A1_im += g11_im * a1_re;
  A1_im += g12_re * a2_im;
  A1_im += g12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += g10_re * b0_re;
  B1_re -= g10_im * b0_im;
  B1_re += g11_re * b1_re;
  B1_re -= g11_im * b1_im;
  B1_re += g12_re * b2_re;
  B1_re -= g12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += g10_re * b0_im;
  B1_im += g10_im * b0_re;
  B1_im += g11_re * b1_im;
  B1_im += g11_im * b1_re;
  B1_im += g12_re * b2_im;
  B1_im += g12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += g20_re * a0_re;
  A2_re -= g20_im * a0_im;
  A2_re += g21_re * a1_re;
  A2_re -= g21_im * a1_im;
  A2_re += g22_re * a2_re;
  A2_re -= g22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += g20_re * a0_im;
  A2_im += g20_im * a0_re;
  A2_im += g21_re * a1_im;
  A2_im += g21_im * a1_re;
  A2_im += g22_re * a2_im;
  A2_im += g22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += g20_re * b0_re;
  B2_re -= g20_im * b0_im;
  B2_re += g21_re * b1_re;
  B2_re -= g21_im * b1_im;
  B2_re += g22_re * b2_re;
  B2_re -= g22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += g20_re * b0_im;
  B2_im += g20_im * b0_re;
  B2_im += g21_re * b1_im;
  B2_im += g21_im * b1_re;
  B2_im += g22_re * b2_im;
  B2_im += g22_im * b2_re;
  
  o1_00_re += A0_re;
  o1_00_im += A0_im;
  o1_10_re += B0_re;
  o1_10_im += B0_im;
  o1_20_re -= B0_re;
  o1_20_im -= B0_im;
  o1_30_re += A0_re;
  o1_30_im += A0_im;
  
  o1_01_re += A1_re;
  o1_01_im += A1_im;
  o1_11_re += B1_re;
  o1_11_im += B1_im;
  o1_21_re -= B1_re;
  o1_21_im -= B1_im;
  o1_31_re += A1_re;
  o1_31_im += A1_im;
  
  o1_02_re += A2_re;
  o1_02_im += A2_im;
  o1_12_re += B2_re;
  o1_12_im += B2_im;
  o1_22_re -= B2_re;
  o1_22_im -= B2_im;
  o1_32_re += A2_re;
  o1_32_im += A2_im;
  
  }
  {
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read flavor 2 from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
    
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
  
#ifdef MULTI_GPU
  } else {
  
  const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
  
    // read half spinor for the second flavor from device memory
    const int fl_idx = sp_idx + ghostFace[static_cast<int>(kernel_type)];
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, fl_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx+ghostFace[static_cast<int>(kernel_type)]);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += g00_re * a0_re;
  A0_re -= g00_im * a0_im;
  A0_re += g01_re * a1_re;
  A0_re -= g01_im * a1_im;
  A0_re += g02_re * a2_re;
  A0_re -= g02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += g00_re * a0_im;
  A0_im += g00_im * a0_re;
  A0_im += g01_re * a1_im;
  A0_im += g01_im * a1_re;
  A0_im += g02_re * a2_im;
  A0_im += g02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += g00_re * b0_re;
  B0_re -= g00_im * b0_im;
  B0_re += g01_re * b1_re;
  B0_re -= g01_im * b1_im;
  B0_re += g02_re * b2_re;
  B0_re -= g02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += g00_re * b0_im;
  B0_im += g00_im * b0_re;
  B0_im += g01_re * b1_im;
  B0_im += g01_im * b1_re;
  B0_im += g02_re * b2_im;
  B0_im += g02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += g10_re * a0_re;
  A1_re -= g10_im * a0_im;
  A1_re += g11_re * a1_re;
  A1_re -= g11_im * a1_im;
  A1_re += g12_re * a2_re;
  A1_re -= g12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += g10_re * a0_im;
  A1_im += g10_im * a0_re;
  A1_im += g11_re * a1_im;
  A1_im += g11_im * a1_re;
  A1_im += g12_re * a2_im;
  A1_im += g12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += g10_re * b0_re;
  B1_re -= g10_im * b0_im;
  B1_re += g11_re * b1_re;
  B1_re -= g11_im * b1_im;
  B1_re += g12_re * b2_re;
  B1_re -= g12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += g10_re * b0_im;
  B1_im += g10_im * b0_re;
  B1_im += g11_re * b1_im;
  B1_im += g11_im * b1_re;
  B1_im += g12_re * b2_im;
  B1_im += g12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += g20_re * a0_re;
  A2_re -= g20_im * a0_im;
  A2_re += g21_re * a1_re;
  A2_re -= g21_im * a1_im;
  A2_re += g22_re * a2_re;
  A2_re -= g22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += g20_re * a0_im;
  A2_im += g20_im * a0_re;
  A2_im += g21_re * a1_im;
  A2_im += g21_im * a1_re;
  A2_im += g22_re * a2_im;
  A2_im += g22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += g20_re * b0_re;
  B2_re -= g20_im * b0_im;
  B2_re += g21_re * b1_re;
  B2_re -= g21_im * b1_im;
  B2_re += g22_re * b2_re;
  B2_re -= g22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += g20_re * b0_im;
  B2_im += g20_im * b0_re;
  B2_im += g21_re * b1_im;
  B2_im += g21_im * b1_re;
  B2_im += g22_re * b2_im;
  B2_im += g22_im * b2_re;
  
  o2_00_re += A0_re;
  o2_00_im += A0_im;
  o2_10_re += B0_re;
  o2_10_im += B0_im;
  o2_20_re -= B0_re;
  o2_20_im -= B0_im;
  o2_30_re += A0_re;
  o2_30_im += A0_im;
  
  o2_01_re += A1_re;
  o2_01_im += A1_im;
  o2_11_re += B1_re;
  o2_11_im += B1_im;
  o2_21_re -= B1_re;
  o2_21_im -= B1_im;
  o2_31_re += A1_re;
  o2_31_im += A1_im;
  
  o2_02_re += A2_re;
  o2_02_im += A2_im;
  o2_12_re += B2_re;
  o2_12_im += B2_im;
  o2_22_re -= B2_re;
  o2_22_im -= B2_im;
  o2_32_re += A2_re;
  o2_32_im += A2_im;
  
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[1] || x2>0)) ||
     (kernel_type == EXTERIOR_KERNEL_Y && x2==0) )
#endif
{
  // Projector P1-
  // 1 0 0 -1 
  // 0 1 1 0 
  // 0 1 1 0 
  // -1 0 0 1 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x2==0 ? X+X2X1mX1 : X-X1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x2==0 ? X+X2X1mX1 : X-X1) >> 1;
#endif
  
#ifdef MULTI_GPU
  const int ga_idx = ((kernel_type == INTERIOR_KERNEL) ? sp_idx : Vh+face_idx);
#else
  const int ga_idx = sp_idx;
#endif
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE1TEX, 3, ga_idx, ga_stride);
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(3);
  
  {
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read flavor 1 from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
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
  
#ifdef MULTI_GPU
  } else {
  
  const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
  
    // read half spinor for the first flavor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += gT00_re * a0_re;
  A0_re -= gT00_im * a0_im;
  A0_re += gT01_re * a1_re;
  A0_re -= gT01_im * a1_im;
  A0_re += gT02_re * a2_re;
  A0_re -= gT02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += gT00_re * a0_im;
  A0_im += gT00_im * a0_re;
  A0_im += gT01_re * a1_im;
  A0_im += gT01_im * a1_re;
  A0_im += gT02_re * a2_im;
  A0_im += gT02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += gT00_re * b0_re;
  B0_re -= gT00_im * b0_im;
  B0_re += gT01_re * b1_re;
  B0_re -= gT01_im * b1_im;
  B0_re += gT02_re * b2_re;
  B0_re -= gT02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += gT00_re * b0_im;
  B0_im += gT00_im * b0_re;
  B0_im += gT01_re * b1_im;
  B0_im += gT01_im * b1_re;
  B0_im += gT02_re * b2_im;
  B0_im += gT02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += gT10_re * a0_re;
  A1_re -= gT10_im * a0_im;
  A1_re += gT11_re * a1_re;
  A1_re -= gT11_im * a1_im;
  A1_re += gT12_re * a2_re;
  A1_re -= gT12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += gT10_re * a0_im;
  A1_im += gT10_im * a0_re;
  A1_im += gT11_re * a1_im;
  A1_im += gT11_im * a1_re;
  A1_im += gT12_re * a2_im;
  A1_im += gT12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += gT10_re * b0_re;
  B1_re -= gT10_im * b0_im;
  B1_re += gT11_re * b1_re;
  B1_re -= gT11_im * b1_im;
  B1_re += gT12_re * b2_re;
  B1_re -= gT12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += gT10_re * b0_im;
  B1_im += gT10_im * b0_re;
  B1_im += gT11_re * b1_im;
  B1_im += gT11_im * b1_re;
  B1_im += gT12_re * b2_im;
  B1_im += gT12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += gT20_re * a0_re;
  A2_re -= gT20_im * a0_im;
  A2_re += gT21_re * a1_re;
  A2_re -= gT21_im * a1_im;
  A2_re += gT22_re * a2_re;
  A2_re -= gT22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += gT20_re * a0_im;
  A2_im += gT20_im * a0_re;
  A2_im += gT21_re * a1_im;
  A2_im += gT21_im * a1_re;
  A2_im += gT22_re * a2_im;
  A2_im += gT22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += gT20_re * b0_re;
  B2_re -= gT20_im * b0_im;
  B2_re += gT21_re * b1_re;
  B2_re -= gT21_im * b1_im;
  B2_re += gT22_re * b2_re;
  B2_re -= gT22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += gT20_re * b0_im;
  B2_im += gT20_im * b0_re;
  B2_im += gT21_re * b1_im;
  B2_im += gT21_im * b1_re;
  B2_im += gT22_re * b2_im;
  B2_im += gT22_im * b2_re;
  
  o1_00_re += A0_re;
  o1_00_im += A0_im;
  o1_10_re += B0_re;
  o1_10_im += B0_im;
  o1_20_re += B0_re;
  o1_20_im += B0_im;
  o1_30_re -= A0_re;
  o1_30_im -= A0_im;
  
  o1_01_re += A1_re;
  o1_01_im += A1_im;
  o1_11_re += B1_re;
  o1_11_im += B1_im;
  o1_21_re += B1_re;
  o1_21_im += B1_im;
  o1_31_re -= A1_re;
  o1_31_im -= A1_im;
  
  o1_02_re += A2_re;
  o1_02_im += A2_im;
  o1_12_re += B2_re;
  o1_12_im += B2_im;
  o1_22_re += B2_re;
  o1_22_im += B2_im;
  o1_32_re -= A2_re;
  o1_32_im -= A2_im;
  
  }
  {
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read flavor 2 from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
    
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
  
#ifdef MULTI_GPU
  } else {
  
  const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
  
    // read half spinor for the second flavor from device memory
    const int fl_idx = sp_idx + ghostFace[static_cast<int>(kernel_type)];
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, fl_idx, sp_norm_idx+ghostFace[static_cast<int>(kernel_type)]);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += gT00_re * a0_re;
  A0_re -= gT00_im * a0_im;
  A0_re += gT01_re * a1_re;
  A0_re -= gT01_im * a1_im;
  A0_re += gT02_re * a2_re;
  A0_re -= gT02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += gT00_re * a0_im;
  A0_im += gT00_im * a0_re;
  A0_im += gT01_re * a1_im;
  A0_im += gT01_im * a1_re;
  A0_im += gT02_re * a2_im;
  A0_im += gT02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += gT00_re * b0_re;
  B0_re -= gT00_im * b0_im;
  B0_re += gT01_re * b1_re;
  B0_re -= gT01_im * b1_im;
  B0_re += gT02_re * b2_re;
  B0_re -= gT02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += gT00_re * b0_im;
  B0_im += gT00_im * b0_re;
  B0_im += gT01_re * b1_im;
  B0_im += gT01_im * b1_re;
  B0_im += gT02_re * b2_im;
  B0_im += gT02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += gT10_re * a0_re;
  A1_re -= gT10_im * a0_im;
  A1_re += gT11_re * a1_re;
  A1_re -= gT11_im * a1_im;
  A1_re += gT12_re * a2_re;
  A1_re -= gT12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += gT10_re * a0_im;
  A1_im += gT10_im * a0_re;
  A1_im += gT11_re * a1_im;
  A1_im += gT11_im * a1_re;
  A1_im += gT12_re * a2_im;
  A1_im += gT12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += gT10_re * b0_re;
  B1_re -= gT10_im * b0_im;
  B1_re += gT11_re * b1_re;
  B1_re -= gT11_im * b1_im;
  B1_re += gT12_re * b2_re;
  B1_re -= gT12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += gT10_re * b0_im;
  B1_im += gT10_im * b0_re;
  B1_im += gT11_re * b1_im;
  B1_im += gT11_im * b1_re;
  B1_im += gT12_re * b2_im;
  B1_im += gT12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += gT20_re * a0_re;
  A2_re -= gT20_im * a0_im;
  A2_re += gT21_re * a1_re;
  A2_re -= gT21_im * a1_im;
  A2_re += gT22_re * a2_re;
  A2_re -= gT22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += gT20_re * a0_im;
  A2_im += gT20_im * a0_re;
  A2_im += gT21_re * a1_im;
  A2_im += gT21_im * a1_re;
  A2_im += gT22_re * a2_im;
  A2_im += gT22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += gT20_re * b0_re;
  B2_re -= gT20_im * b0_im;
  B2_re += gT21_re * b1_re;
  B2_re -= gT21_im * b1_im;
  B2_re += gT22_re * b2_re;
  B2_re -= gT22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += gT20_re * b0_im;
  B2_im += gT20_im * b0_re;
  B2_im += gT21_re * b1_im;
  B2_im += gT21_im * b1_re;
  B2_im += gT22_re * b2_im;
  B2_im += gT22_im * b2_re;
  
  o2_00_re += A0_re;
  o2_00_im += A0_im;
  o2_10_re += B0_re;
  o2_10_im += B0_im;
  o2_20_re += B0_re;
  o2_20_im += B0_im;
  o2_30_re -= A0_re;
  o2_30_im -= A0_im;
  
  o2_01_re += A1_re;
  o2_01_im += A1_im;
  o2_11_re += B1_re;
  o2_11_im += B1_im;
  o2_21_re += B1_re;
  o2_21_im += B1_im;
  o2_31_re -= A1_re;
  o2_31_im -= A1_im;
  
  o2_02_re += A2_re;
  o2_02_im += A2_im;
  o2_12_re += B2_re;
  o2_12_im += B2_im;
  o2_22_re += B2_re;
  o2_22_im += B2_im;
  o2_32_re -= A2_re;
  o2_32_im -= A2_im;
  
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[2] || x3<X3m1)) ||
     (kernel_type == EXTERIOR_KERNEL_Z && x3==X3m1) )
#endif
{
  // Projector P2+
  // 1 0 i 0 
  // 0 1 0 -i 
  // -i 0 1 0 
  // 0 i 0 1 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x3==X3m1 ? X-X3X2X1mX2X1 : X+X2X1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x3==X3m1 ? X-X3X2X1mX2X1 : X+X2X1) >> 1;
#endif
  
  const int ga_idx = sid;
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE0TEX, 4, ga_idx, ga_stride);
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(4);
  
  {
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read flavor 1 from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
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
  
#ifdef MULTI_GPU
  } else {
  
  const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
  
    // read half spinor for the first flavor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += g00_re * a0_re;
  A0_re -= g00_im * a0_im;
  A0_re += g01_re * a1_re;
  A0_re -= g01_im * a1_im;
  A0_re += g02_re * a2_re;
  A0_re -= g02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += g00_re * a0_im;
  A0_im += g00_im * a0_re;
  A0_im += g01_re * a1_im;
  A0_im += g01_im * a1_re;
  A0_im += g02_re * a2_im;
  A0_im += g02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += g00_re * b0_re;
  B0_re -= g00_im * b0_im;
  B0_re += g01_re * b1_re;
  B0_re -= g01_im * b1_im;
  B0_re += g02_re * b2_re;
  B0_re -= g02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += g00_re * b0_im;
  B0_im += g00_im * b0_re;
  B0_im += g01_re * b1_im;
  B0_im += g01_im * b1_re;
  B0_im += g02_re * b2_im;
  B0_im += g02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += g10_re * a0_re;
  A1_re -= g10_im * a0_im;
  A1_re += g11_re * a1_re;
  A1_re -= g11_im * a1_im;
  A1_re += g12_re * a2_re;
  A1_re -= g12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += g10_re * a0_im;
  A1_im += g10_im * a0_re;
  A1_im += g11_re * a1_im;
  A1_im += g11_im * a1_re;
  A1_im += g12_re * a2_im;
  A1_im += g12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += g10_re * b0_re;
  B1_re -= g10_im * b0_im;
  B1_re += g11_re * b1_re;
  B1_re -= g11_im * b1_im;
  B1_re += g12_re * b2_re;
  B1_re -= g12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += g10_re * b0_im;
  B1_im += g10_im * b0_re;
  B1_im += g11_re * b1_im;
  B1_im += g11_im * b1_re;
  B1_im += g12_re * b2_im;
  B1_im += g12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += g20_re * a0_re;
  A2_re -= g20_im * a0_im;
  A2_re += g21_re * a1_re;
  A2_re -= g21_im * a1_im;
  A2_re += g22_re * a2_re;
  A2_re -= g22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += g20_re * a0_im;
  A2_im += g20_im * a0_re;
  A2_im += g21_re * a1_im;
  A2_im += g21_im * a1_re;
  A2_im += g22_re * a2_im;
  A2_im += g22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += g20_re * b0_re;
  B2_re -= g20_im * b0_im;
  B2_re += g21_re * b1_re;
  B2_re -= g21_im * b1_im;
  B2_re += g22_re * b2_re;
  B2_re -= g22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += g20_re * b0_im;
  B2_im += g20_im * b0_re;
  B2_im += g21_re * b1_im;
  B2_im += g21_im * b1_re;
  B2_im += g22_re * b2_im;
  B2_im += g22_im * b2_re;
  
  o1_00_re += A0_re;
  o1_00_im += A0_im;
  o1_10_re += B0_re;
  o1_10_im += B0_im;
  o1_20_re += A0_im;
  o1_20_im -= A0_re;
  o1_30_re -= B0_im;
  o1_30_im += B0_re;
  
  o1_01_re += A1_re;
  o1_01_im += A1_im;
  o1_11_re += B1_re;
  o1_11_im += B1_im;
  o1_21_re += A1_im;
  o1_21_im -= A1_re;
  o1_31_re -= B1_im;
  o1_31_im += B1_re;
  
  o1_02_re += A2_re;
  o1_02_im += A2_im;
  o1_12_re += B2_re;
  o1_12_im += B2_im;
  o1_22_re += A2_im;
  o1_22_im -= A2_re;
  o1_32_re -= B2_im;
  o1_32_im += B2_re;
  
  }
  {
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read flavor 2 from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
    
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
  
#ifdef MULTI_GPU
  } else {
  
  const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
  
    // read half spinor for the second flavor from device memory
    const int fl_idx = sp_idx + ghostFace[static_cast<int>(kernel_type)];
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, fl_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx+ghostFace[static_cast<int>(kernel_type)]);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += g00_re * a0_re;
  A0_re -= g00_im * a0_im;
  A0_re += g01_re * a1_re;
  A0_re -= g01_im * a1_im;
  A0_re += g02_re * a2_re;
  A0_re -= g02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += g00_re * a0_im;
  A0_im += g00_im * a0_re;
  A0_im += g01_re * a1_im;
  A0_im += g01_im * a1_re;
  A0_im += g02_re * a2_im;
  A0_im += g02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += g00_re * b0_re;
  B0_re -= g00_im * b0_im;
  B0_re += g01_re * b1_re;
  B0_re -= g01_im * b1_im;
  B0_re += g02_re * b2_re;
  B0_re -= g02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += g00_re * b0_im;
  B0_im += g00_im * b0_re;
  B0_im += g01_re * b1_im;
  B0_im += g01_im * b1_re;
  B0_im += g02_re * b2_im;
  B0_im += g02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += g10_re * a0_re;
  A1_re -= g10_im * a0_im;
  A1_re += g11_re * a1_re;
  A1_re -= g11_im * a1_im;
  A1_re += g12_re * a2_re;
  A1_re -= g12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += g10_re * a0_im;
  A1_im += g10_im * a0_re;
  A1_im += g11_re * a1_im;
  A1_im += g11_im * a1_re;
  A1_im += g12_re * a2_im;
  A1_im += g12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += g10_re * b0_re;
  B1_re -= g10_im * b0_im;
  B1_re += g11_re * b1_re;
  B1_re -= g11_im * b1_im;
  B1_re += g12_re * b2_re;
  B1_re -= g12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += g10_re * b0_im;
  B1_im += g10_im * b0_re;
  B1_im += g11_re * b1_im;
  B1_im += g11_im * b1_re;
  B1_im += g12_re * b2_im;
  B1_im += g12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += g20_re * a0_re;
  A2_re -= g20_im * a0_im;
  A2_re += g21_re * a1_re;
  A2_re -= g21_im * a1_im;
  A2_re += g22_re * a2_re;
  A2_re -= g22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += g20_re * a0_im;
  A2_im += g20_im * a0_re;
  A2_im += g21_re * a1_im;
  A2_im += g21_im * a1_re;
  A2_im += g22_re * a2_im;
  A2_im += g22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += g20_re * b0_re;
  B2_re -= g20_im * b0_im;
  B2_re += g21_re * b1_re;
  B2_re -= g21_im * b1_im;
  B2_re += g22_re * b2_re;
  B2_re -= g22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += g20_re * b0_im;
  B2_im += g20_im * b0_re;
  B2_im += g21_re * b1_im;
  B2_im += g21_im * b1_re;
  B2_im += g22_re * b2_im;
  B2_im += g22_im * b2_re;
  
  o2_00_re += A0_re;
  o2_00_im += A0_im;
  o2_10_re += B0_re;
  o2_10_im += B0_im;
  o2_20_re += A0_im;
  o2_20_im -= A0_re;
  o2_30_re -= B0_im;
  o2_30_im += B0_re;
  
  o2_01_re += A1_re;
  o2_01_im += A1_im;
  o2_11_re += B1_re;
  o2_11_im += B1_im;
  o2_21_re += A1_im;
  o2_21_im -= A1_re;
  o2_31_re -= B1_im;
  o2_31_im += B1_re;
  
  o2_02_re += A2_re;
  o2_02_im += A2_im;
  o2_12_re += B2_re;
  o2_12_im += B2_im;
  o2_22_re += A2_im;
  o2_22_im -= A2_re;
  o2_32_re -= B2_im;
  o2_32_im += B2_re;
  
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[2] || x3>0)) ||
     (kernel_type == EXTERIOR_KERNEL_Z && x3==0) )
#endif
{
  // Projector P2-
  // 1 0 -i 0 
  // 0 1 0 i 
  // i 0 1 0 
  // 0 -i 0 1 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x3==0 ? X+X3X2X1mX2X1 : X-X2X1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x3==0 ? X+X3X2X1mX2X1 : X-X2X1) >> 1;
#endif
  
#ifdef MULTI_GPU
  const int ga_idx = ((kernel_type == INTERIOR_KERNEL) ? sp_idx : Vh+face_idx);
#else
  const int ga_idx = sp_idx;
#endif
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE1TEX, 5, ga_idx, ga_stride);
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(5);
  
  {
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read flavor 1 from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
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
  
#ifdef MULTI_GPU
  } else {
  
  const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
  
    // read half spinor for the first flavor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += gT00_re * a0_re;
  A0_re -= gT00_im * a0_im;
  A0_re += gT01_re * a1_re;
  A0_re -= gT01_im * a1_im;
  A0_re += gT02_re * a2_re;
  A0_re -= gT02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += gT00_re * a0_im;
  A0_im += gT00_im * a0_re;
  A0_im += gT01_re * a1_im;
  A0_im += gT01_im * a1_re;
  A0_im += gT02_re * a2_im;
  A0_im += gT02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += gT00_re * b0_re;
  B0_re -= gT00_im * b0_im;
  B0_re += gT01_re * b1_re;
  B0_re -= gT01_im * b1_im;
  B0_re += gT02_re * b2_re;
  B0_re -= gT02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += gT00_re * b0_im;
  B0_im += gT00_im * b0_re;
  B0_im += gT01_re * b1_im;
  B0_im += gT01_im * b1_re;
  B0_im += gT02_re * b2_im;
  B0_im += gT02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += gT10_re * a0_re;
  A1_re -= gT10_im * a0_im;
  A1_re += gT11_re * a1_re;
  A1_re -= gT11_im * a1_im;
  A1_re += gT12_re * a2_re;
  A1_re -= gT12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += gT10_re * a0_im;
  A1_im += gT10_im * a0_re;
  A1_im += gT11_re * a1_im;
  A1_im += gT11_im * a1_re;
  A1_im += gT12_re * a2_im;
  A1_im += gT12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += gT10_re * b0_re;
  B1_re -= gT10_im * b0_im;
  B1_re += gT11_re * b1_re;
  B1_re -= gT11_im * b1_im;
  B1_re += gT12_re * b2_re;
  B1_re -= gT12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += gT10_re * b0_im;
  B1_im += gT10_im * b0_re;
  B1_im += gT11_re * b1_im;
  B1_im += gT11_im * b1_re;
  B1_im += gT12_re * b2_im;
  B1_im += gT12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += gT20_re * a0_re;
  A2_re -= gT20_im * a0_im;
  A2_re += gT21_re * a1_re;
  A2_re -= gT21_im * a1_im;
  A2_re += gT22_re * a2_re;
  A2_re -= gT22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += gT20_re * a0_im;
  A2_im += gT20_im * a0_re;
  A2_im += gT21_re * a1_im;
  A2_im += gT21_im * a1_re;
  A2_im += gT22_re * a2_im;
  A2_im += gT22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += gT20_re * b0_re;
  B2_re -= gT20_im * b0_im;
  B2_re += gT21_re * b1_re;
  B2_re -= gT21_im * b1_im;
  B2_re += gT22_re * b2_re;
  B2_re -= gT22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += gT20_re * b0_im;
  B2_im += gT20_im * b0_re;
  B2_im += gT21_re * b1_im;
  B2_im += gT21_im * b1_re;
  B2_im += gT22_re * b2_im;
  B2_im += gT22_im * b2_re;
  
  o1_00_re += A0_re;
  o1_00_im += A0_im;
  o1_10_re += B0_re;
  o1_10_im += B0_im;
  o1_20_re -= A0_im;
  o1_20_im += A0_re;
  o1_30_re += B0_im;
  o1_30_im -= B0_re;
  
  o1_01_re += A1_re;
  o1_01_im += A1_im;
  o1_11_re += B1_re;
  o1_11_im += B1_im;
  o1_21_re -= A1_im;
  o1_21_im += A1_re;
  o1_31_re += B1_im;
  o1_31_im -= B1_re;
  
  o1_02_re += A2_re;
  o1_02_im += A2_im;
  o1_12_re += B2_re;
  o1_12_im += B2_im;
  o1_22_re -= A2_im;
  o1_22_im += A2_re;
  o1_32_re += B2_im;
  o1_32_im -= B2_re;
  
  }
  {
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read flavor 2 from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
    
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
  
#ifdef MULTI_GPU
  } else {
  
  const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
  
    // read half spinor for the second flavor from device memory
    const int fl_idx = sp_idx + ghostFace[static_cast<int>(kernel_type)];
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, fl_idx, sp_norm_idx+ghostFace[static_cast<int>(kernel_type)]);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += gT00_re * a0_re;
  A0_re -= gT00_im * a0_im;
  A0_re += gT01_re * a1_re;
  A0_re -= gT01_im * a1_im;
  A0_re += gT02_re * a2_re;
  A0_re -= gT02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += gT00_re * a0_im;
  A0_im += gT00_im * a0_re;
  A0_im += gT01_re * a1_im;
  A0_im += gT01_im * a1_re;
  A0_im += gT02_re * a2_im;
  A0_im += gT02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += gT00_re * b0_re;
  B0_re -= gT00_im * b0_im;
  B0_re += gT01_re * b1_re;
  B0_re -= gT01_im * b1_im;
  B0_re += gT02_re * b2_re;
  B0_re -= gT02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += gT00_re * b0_im;
  B0_im += gT00_im * b0_re;
  B0_im += gT01_re * b1_im;
  B0_im += gT01_im * b1_re;
  B0_im += gT02_re * b2_im;
  B0_im += gT02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += gT10_re * a0_re;
  A1_re -= gT10_im * a0_im;
  A1_re += gT11_re * a1_re;
  A1_re -= gT11_im * a1_im;
  A1_re += gT12_re * a2_re;
  A1_re -= gT12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += gT10_re * a0_im;
  A1_im += gT10_im * a0_re;
  A1_im += gT11_re * a1_im;
  A1_im += gT11_im * a1_re;
  A1_im += gT12_re * a2_im;
  A1_im += gT12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += gT10_re * b0_re;
  B1_re -= gT10_im * b0_im;
  B1_re += gT11_re * b1_re;
  B1_re -= gT11_im * b1_im;
  B1_re += gT12_re * b2_re;
  B1_re -= gT12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += gT10_re * b0_im;
  B1_im += gT10_im * b0_re;
  B1_im += gT11_re * b1_im;
  B1_im += gT11_im * b1_re;
  B1_im += gT12_re * b2_im;
  B1_im += gT12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += gT20_re * a0_re;
  A2_re -= gT20_im * a0_im;
  A2_re += gT21_re * a1_re;
  A2_re -= gT21_im * a1_im;
  A2_re += gT22_re * a2_re;
  A2_re -= gT22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += gT20_re * a0_im;
  A2_im += gT20_im * a0_re;
  A2_im += gT21_re * a1_im;
  A2_im += gT21_im * a1_re;
  A2_im += gT22_re * a2_im;
  A2_im += gT22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += gT20_re * b0_re;
  B2_re -= gT20_im * b0_im;
  B2_re += gT21_re * b1_re;
  B2_re -= gT21_im * b1_im;
  B2_re += gT22_re * b2_re;
  B2_re -= gT22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += gT20_re * b0_im;
  B2_im += gT20_im * b0_re;
  B2_im += gT21_re * b1_im;
  B2_im += gT21_im * b1_re;
  B2_im += gT22_re * b2_im;
  B2_im += gT22_im * b2_re;
  
  o2_00_re += A0_re;
  o2_00_im += A0_im;
  o2_10_re += B0_re;
  o2_10_im += B0_im;
  o2_20_re -= A0_im;
  o2_20_im += A0_re;
  o2_30_re += B0_im;
  o2_30_im -= B0_re;
  
  o2_01_re += A1_re;
  o2_01_im += A1_im;
  o2_11_re += B1_re;
  o2_11_im += B1_im;
  o2_21_re -= A1_im;
  o2_21_im += A1_re;
  o2_31_re += B1_im;
  o2_31_im -= B1_re;
  
  o2_02_re += A2_re;
  o2_02_im += A2_im;
  o2_12_re += B2_re;
  o2_12_im += B2_im;
  o2_22_re -= A2_im;
  o2_22_im += A2_re;
  o2_32_re += B2_im;
  o2_32_im -= B2_re;
  
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[3] || x4<X4m1)) ||
     (kernel_type == EXTERIOR_KERNEL_T && x4==X4m1) )
#endif
{
  // Projector P3+
  // 2 0 0 0 
  // 0 2 0 0 
  // 0 0 0 0 
  // 0 0 0 0 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x4==X4m1 ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x4==X4m1 ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1;
#endif
  
  const int ga_idx = sid;
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
  if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h)
  {
    {
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read flavor 1 from device memory
      READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
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
    
#ifdef MULTI_GPU
    } else {
    
    const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
    
      // read half spinor for the first flavor from device memory
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
      
      a0_re = 2*i00_re;  a0_im = 2*i00_im;
      a1_re = 2*i01_re;  a1_im = 2*i01_im;
      a2_re = 2*i02_re;  a2_im = 2*i02_im;
      b0_re = 2*i10_re;  b0_im = 2*i10_im;
      b1_re = 2*i11_re;  b1_im = 2*i11_im;
      b2_re = 2*i12_re;  b2_im = 2*i12_im;
      
    }
#endif // MULTI_GPU
    
    // identity gauge matrix
    spinorFloat A0_re = a0_re; spinorFloat A0_im = a0_im;
    spinorFloat B0_re = b0_re; spinorFloat B0_im = b0_im;
    spinorFloat A1_re = a1_re; spinorFloat A1_im = a1_im;
    spinorFloat B1_re = b1_re; spinorFloat B1_im = b1_im;
    spinorFloat A2_re = a2_re; spinorFloat A2_im = a2_im;
    spinorFloat B2_re = b2_re; spinorFloat B2_im = b2_im;
    
    o1_00_re += A0_re;
    o1_00_im += A0_im;
    o1_10_re += B0_re;
    o1_10_im += B0_im;
    
    o1_01_re += A1_re;
    o1_01_im += A1_im;
    o1_11_re += B1_re;
    o1_11_im += B1_im;
    
    o1_02_re += A2_re;
    o1_02_im += A2_im;
    o1_12_re += B2_re;
    o1_12_im += B2_im;
    
    }
    {
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read flavor 2 from device memory
      READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
      
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
    
#ifdef MULTI_GPU
    } else {
    
    const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
    
      // read half spinor for the second flavor from device memory
      const int fl_idx = sp_idx + ghostFace[static_cast<int>(kernel_type)];
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, fl_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx+ghostFace[static_cast<int>(kernel_type)]);
      
      a0_re = 2*i00_re;  a0_im = 2*i00_im;
      a1_re = 2*i01_re;  a1_im = 2*i01_im;
      a2_re = 2*i02_re;  a2_im = 2*i02_im;
      b0_re = 2*i10_re;  b0_im = 2*i10_im;
      b1_re = 2*i11_re;  b1_im = 2*i11_im;
      b2_re = 2*i12_re;  b2_im = 2*i12_im;
      
    }
#endif // MULTI_GPU
    
    // identity gauge matrix
    spinorFloat A0_re = a0_re; spinorFloat A0_im = a0_im;
    spinorFloat B0_re = b0_re; spinorFloat B0_im = b0_im;
    spinorFloat A1_re = a1_re; spinorFloat A1_im = a1_im;
    spinorFloat B1_re = b1_re; spinorFloat B1_im = b1_im;
    spinorFloat A2_re = a2_re; spinorFloat A2_im = a2_im;
    spinorFloat B2_re = b2_re; spinorFloat B2_im = b2_im;
    
    o2_00_re += A0_re;
    o2_00_im += A0_im;
    o2_10_re += B0_re;
    o2_10_im += B0_im;
    
    o2_01_re += A1_re;
    o2_01_im += A1_im;
    o2_11_re += B1_re;
    o2_11_im += B1_im;
    
    o2_02_re += A2_re;
    o2_02_im += A2_im;
    o2_12_re += B2_re;
    o2_12_im += B2_im;
    
    }
  } else {
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE0TEX, 6, ga_idx, ga_stride);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(6);
    
    {
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read flavor 1 from device memory
      READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
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
    
#ifdef MULTI_GPU
    } else {
    
    const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
    
      // read half spinor for the first flavor from device memory
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
      
      a0_re = 2*i00_re;  a0_im = 2*i00_im;
      a1_re = 2*i01_re;  a1_im = 2*i01_im;
      a2_re = 2*i02_re;  a2_im = 2*i02_im;
      b0_re = 2*i10_re;  b0_im = 2*i10_im;
      b1_re = 2*i11_re;  b1_im = 2*i11_im;
      b2_re = 2*i12_re;  b2_im = 2*i12_im;
      
    }
#endif // MULTI_GPU
    
    // multiply row 0
    spinorFloat A0_re = 0;
    A0_re += g00_re * a0_re;
    A0_re -= g00_im * a0_im;
    A0_re += g01_re * a1_re;
    A0_re -= g01_im * a1_im;
    A0_re += g02_re * a2_re;
    A0_re -= g02_im * a2_im;
    spinorFloat A0_im = 0;
    A0_im += g00_re * a0_im;
    A0_im += g00_im * a0_re;
    A0_im += g01_re * a1_im;
    A0_im += g01_im * a1_re;
    A0_im += g02_re * a2_im;
    A0_im += g02_im * a2_re;
    spinorFloat B0_re = 0;
    B0_re += g00_re * b0_re;
    B0_re -= g00_im * b0_im;
    B0_re += g01_re * b1_re;
    B0_re -= g01_im * b1_im;
    B0_re += g02_re * b2_re;
    B0_re -= g02_im * b2_im;
    spinorFloat B0_im = 0;
    B0_im += g00_re * b0_im;
    B0_im += g00_im * b0_re;
    B0_im += g01_re * b1_im;
    B0_im += g01_im * b1_re;
    B0_im += g02_re * b2_im;
    B0_im += g02_im * b2_re;
    
    // multiply row 1
    spinorFloat A1_re = 0;
    A1_re += g10_re * a0_re;
    A1_re -= g10_im * a0_im;
    A1_re += g11_re * a1_re;
    A1_re -= g11_im * a1_im;
    A1_re += g12_re * a2_re;
    A1_re -= g12_im * a2_im;
    spinorFloat A1_im = 0;
    A1_im += g10_re * a0_im;
    A1_im += g10_im * a0_re;
    A1_im += g11_re * a1_im;
    A1_im += g11_im * a1_re;
    A1_im += g12_re * a2_im;
    A1_im += g12_im * a2_re;
    spinorFloat B1_re = 0;
    B1_re += g10_re * b0_re;
    B1_re -= g10_im * b0_im;
    B1_re += g11_re * b1_re;
    B1_re -= g11_im * b1_im;
    B1_re += g12_re * b2_re;
    B1_re -= g12_im * b2_im;
    spinorFloat B1_im = 0;
    B1_im += g10_re * b0_im;
    B1_im += g10_im * b0_re;
    B1_im += g11_re * b1_im;
    B1_im += g11_im * b1_re;
    B1_im += g12_re * b2_im;
    B1_im += g12_im * b2_re;
    
    // multiply row 2
    spinorFloat A2_re = 0;
    A2_re += g20_re * a0_re;
    A2_re -= g20_im * a0_im;
    A2_re += g21_re * a1_re;
    A2_re -= g21_im * a1_im;
    A2_re += g22_re * a2_re;
    A2_re -= g22_im * a2_im;
    spinorFloat A2_im = 0;
    A2_im += g20_re * a0_im;
    A2_im += g20_im * a0_re;
    A2_im += g21_re * a1_im;
    A2_im += g21_im * a1_re;
    A2_im += g22_re * a2_im;
    A2_im += g22_im * a2_re;
    spinorFloat B2_re = 0;
    B2_re += g20_re * b0_re;
    B2_re -= g20_im * b0_im;
    B2_re += g21_re * b1_re;
    B2_re -= g21_im * b1_im;
    B2_re += g22_re * b2_re;
    B2_re -= g22_im * b2_im;
    spinorFloat B2_im = 0;
    B2_im += g20_re * b0_im;
    B2_im += g20_im * b0_re;
    B2_im += g21_re * b1_im;
    B2_im += g21_im * b1_re;
    B2_im += g22_re * b2_im;
    B2_im += g22_im * b2_re;
    
    o1_00_re += A0_re;
    o1_00_im += A0_im;
    o1_10_re += B0_re;
    o1_10_im += B0_im;
    
    o1_01_re += A1_re;
    o1_01_im += A1_im;
    o1_11_re += B1_re;
    o1_11_im += B1_im;
    
    o1_02_re += A2_re;
    o1_02_im += A2_im;
    o1_12_re += B2_re;
    o1_12_im += B2_im;
    
    }
    {
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read flavor 2 from device memory
      READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
      
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
    
#ifdef MULTI_GPU
    } else {
    
    const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
    
      // read half spinor for the second flavor from device memory
      const int fl_idx = sp_idx + ghostFace[static_cast<int>(kernel_type)];
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, fl_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx+ghostFace[static_cast<int>(kernel_type)]);
      
      a0_re = 2*i00_re;  a0_im = 2*i00_im;
      a1_re = 2*i01_re;  a1_im = 2*i01_im;
      a2_re = 2*i02_re;  a2_im = 2*i02_im;
      b0_re = 2*i10_re;  b0_im = 2*i10_im;
      b1_re = 2*i11_re;  b1_im = 2*i11_im;
      b2_re = 2*i12_re;  b2_im = 2*i12_im;
      
    }
#endif // MULTI_GPU
    
    // multiply row 0
    spinorFloat A0_re = 0;
    A0_re += g00_re * a0_re;
    A0_re -= g00_im * a0_im;
    A0_re += g01_re * a1_re;
    A0_re -= g01_im * a1_im;
    A0_re += g02_re * a2_re;
    A0_re -= g02_im * a2_im;
    spinorFloat A0_im = 0;
    A0_im += g00_re * a0_im;
    A0_im += g00_im * a0_re;
    A0_im += g01_re * a1_im;
    A0_im += g01_im * a1_re;
    A0_im += g02_re * a2_im;
    A0_im += g02_im * a2_re;
    spinorFloat B0_re = 0;
    B0_re += g00_re * b0_re;
    B0_re -= g00_im * b0_im;
    B0_re += g01_re * b1_re;
    B0_re -= g01_im * b1_im;
    B0_re += g02_re * b2_re;
    B0_re -= g02_im * b2_im;
    spinorFloat B0_im = 0;
    B0_im += g00_re * b0_im;
    B0_im += g00_im * b0_re;
    B0_im += g01_re * b1_im;
    B0_im += g01_im * b1_re;
    B0_im += g02_re * b2_im;
    B0_im += g02_im * b2_re;
    
    // multiply row 1
    spinorFloat A1_re = 0;
    A1_re += g10_re * a0_re;
    A1_re -= g10_im * a0_im;
    A1_re += g11_re * a1_re;
    A1_re -= g11_im * a1_im;
    A1_re += g12_re * a2_re;
    A1_re -= g12_im * a2_im;
    spinorFloat A1_im = 0;
    A1_im += g10_re * a0_im;
    A1_im += g10_im * a0_re;
    A1_im += g11_re * a1_im;
    A1_im += g11_im * a1_re;
    A1_im += g12_re * a2_im;
    A1_im += g12_im * a2_re;
    spinorFloat B1_re = 0;
    B1_re += g10_re * b0_re;
    B1_re -= g10_im * b0_im;
    B1_re += g11_re * b1_re;
    B1_re -= g11_im * b1_im;
    B1_re += g12_re * b2_re;
    B1_re -= g12_im * b2_im;
    spinorFloat B1_im = 0;
    B1_im += g10_re * b0_im;
    B1_im += g10_im * b0_re;
    B1_im += g11_re * b1_im;
    B1_im += g11_im * b1_re;
    B1_im += g12_re * b2_im;
    B1_im += g12_im * b2_re;
    
    // multiply row 2
    spinorFloat A2_re = 0;
    A2_re += g20_re * a0_re;
    A2_re -= g20_im * a0_im;
    A2_re += g21_re * a1_re;
    A2_re -= g21_im * a1_im;
    A2_re += g22_re * a2_re;
    A2_re -= g22_im * a2_im;
    spinorFloat A2_im = 0;
    A2_im += g20_re * a0_im;
    A2_im += g20_im * a0_re;
    A2_im += g21_re * a1_im;
    A2_im += g21_im * a1_re;
    A2_im += g22_re * a2_im;
    A2_im += g22_im * a2_re;
    spinorFloat B2_re = 0;
    B2_re += g20_re * b0_re;
    B2_re -= g20_im * b0_im;
    B2_re += g21_re * b1_re;
    B2_re -= g21_im * b1_im;
    B2_re += g22_re * b2_re;
    B2_re -= g22_im * b2_im;
    spinorFloat B2_im = 0;
    B2_im += g20_re * b0_im;
    B2_im += g20_im * b0_re;
    B2_im += g21_re * b1_im;
    B2_im += g21_im * b1_re;
    B2_im += g22_re * b2_im;
    B2_im += g22_im * b2_re;
    
    o2_00_re += A0_re;
    o2_00_im += A0_im;
    o2_10_re += B0_re;
    o2_10_im += B0_im;
    
    o2_01_re += A1_re;
    o2_01_im += A1_im;
    o2_11_re += B1_re;
    o2_11_im += B1_im;
    
    o2_02_re += A2_re;
    o2_02_im += A2_im;
    o2_12_re += B2_re;
    o2_12_im += B2_im;
    
    }
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[3] || x4>0)) ||
     (kernel_type == EXTERIOR_KERNEL_T && x4==0) )
#endif
{
  // Projector P3-
  // 0 0 0 0 
  // 0 0 0 0 
  // 0 0 2 0 
  // 0 0 0 2 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x4==0 ? X+X4X3X2X1mX3X2X1 : X-X3X2X1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x4==0 ? X+X4X3X2X1mX3X2X1 : X-X3X2X1) >> 1;
#endif
  
#ifdef MULTI_GPU
  const int ga_idx = ((kernel_type == INTERIOR_KERNEL) ? sp_idx : Vh+face_idx);
#else
  const int ga_idx = sp_idx;
#endif
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
  if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h)
  {
    {
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read flavor 1 from device memory
      READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
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
    
#ifdef MULTI_GPU
    } else {
    
    const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
    
      // read half spinor for the first flavor from device memory
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
      
      a0_re = 2*i00_re;  a0_im = 2*i00_im;
      a1_re = 2*i01_re;  a1_im = 2*i01_im;
      a2_re = 2*i02_re;  a2_im = 2*i02_im;
      b0_re = 2*i10_re;  b0_im = 2*i10_im;
      b1_re = 2*i11_re;  b1_im = 2*i11_im;
      b2_re = 2*i12_re;  b2_im = 2*i12_im;
      
    }
#endif // MULTI_GPU
    
    // identity gauge matrix
    spinorFloat A0_re = a0_re; spinorFloat A0_im = a0_im;
    spinorFloat B0_re = b0_re; spinorFloat B0_im = b0_im;
    spinorFloat A1_re = a1_re; spinorFloat A1_im = a1_im;
    spinorFloat B1_re = b1_re; spinorFloat B1_im = b1_im;
    spinorFloat A2_re = a2_re; spinorFloat A2_im = a2_im;
    spinorFloat B2_re = b2_re; spinorFloat B2_im = b2_im;
    
    o1_20_re += A0_re;
    o1_20_im += A0_im;
    o1_30_re += B0_re;
    o1_30_im += B0_im;
    
    o1_21_re += A1_re;
    o1_21_im += A1_im;
    o1_31_re += B1_re;
    o1_31_im += B1_im;
    
    o1_22_re += A2_re;
    o1_22_im += A2_im;
    o1_32_re += B2_re;
    o1_32_im += B2_im;
    
    }
    {
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read flavor 2 from device memory
      READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
      
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
    
#ifdef MULTI_GPU
    } else {
    
    const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
    
      // read half spinor for the second flavor from device memory
      const int fl_idx = sp_idx + ghostFace[static_cast<int>(kernel_type)];
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, fl_idx, sp_norm_idx+ghostFace[static_cast<int>(kernel_type)]);
      
      a0_re = 2*i00_re;  a0_im = 2*i00_im;
      a1_re = 2*i01_re;  a1_im = 2*i01_im;
      a2_re = 2*i02_re;  a2_im = 2*i02_im;
      b0_re = 2*i10_re;  b0_im = 2*i10_im;
      b1_re = 2*i11_re;  b1_im = 2*i11_im;
      b2_re = 2*i12_re;  b2_im = 2*i12_im;
      
    }
#endif // MULTI_GPU
    
    // identity gauge matrix
    spinorFloat A0_re = a0_re; spinorFloat A0_im = a0_im;
    spinorFloat B0_re = b0_re; spinorFloat B0_im = b0_im;
    spinorFloat A1_re = a1_re; spinorFloat A1_im = a1_im;
    spinorFloat B1_re = b1_re; spinorFloat B1_im = b1_im;
    spinorFloat A2_re = a2_re; spinorFloat A2_im = a2_im;
    spinorFloat B2_re = b2_re; spinorFloat B2_im = b2_im;
    
    o2_20_re += A0_re;
    o2_20_im += A0_im;
    o2_30_re += B0_re;
    o2_30_im += B0_im;
    
    o2_21_re += A1_re;
    o2_21_im += A1_im;
    o2_31_re += B1_re;
    o2_31_im += B1_im;
    
    o2_22_re += A2_re;
    o2_22_im += A2_im;
    o2_32_re += B2_re;
    o2_32_im += B2_im;
    
    }
  } else {
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE1TEX, 7, ga_idx, ga_stride);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(7);
    
    {
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read flavor 1 from device memory
      READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
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
    
#ifdef MULTI_GPU
    } else {
    
    const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
    
      // read half spinor for the first flavor from device memory
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
      
      a0_re = 2*i00_re;  a0_im = 2*i00_im;
      a1_re = 2*i01_re;  a1_im = 2*i01_im;
      a2_re = 2*i02_re;  a2_im = 2*i02_im;
      b0_re = 2*i10_re;  b0_im = 2*i10_im;
      b1_re = 2*i11_re;  b1_im = 2*i11_im;
      b2_re = 2*i12_re;  b2_im = 2*i12_im;
      
    }
#endif // MULTI_GPU
    
    // multiply row 0
    spinorFloat A0_re = 0;
    A0_re += gT00_re * a0_re;
    A0_re -= gT00_im * a0_im;
    A0_re += gT01_re * a1_re;
    A0_re -= gT01_im * a1_im;
    A0_re += gT02_re * a2_re;
    A0_re -= gT02_im * a2_im;
    spinorFloat A0_im = 0;
    A0_im += gT00_re * a0_im;
    A0_im += gT00_im * a0_re;
    A0_im += gT01_re * a1_im;
    A0_im += gT01_im * a1_re;
    A0_im += gT02_re * a2_im;
    A0_im += gT02_im * a2_re;
    spinorFloat B0_re = 0;
    B0_re += gT00_re * b0_re;
    B0_re -= gT00_im * b0_im;
    B0_re += gT01_re * b1_re;
    B0_re -= gT01_im * b1_im;
    B0_re += gT02_re * b2_re;
    B0_re -= gT02_im * b2_im;
    spinorFloat B0_im = 0;
    B0_im += gT00_re * b0_im;
    B0_im += gT00_im * b0_re;
    B0_im += gT01_re * b1_im;
    B0_im += gT01_im * b1_re;
    B0_im += gT02_re * b2_im;
    B0_im += gT02_im * b2_re;
    
    // multiply row 1
    spinorFloat A1_re = 0;
    A1_re += gT10_re * a0_re;
    A1_re -= gT10_im * a0_im;
    A1_re += gT11_re * a1_re;
    A1_re -= gT11_im * a1_im;
    A1_re += gT12_re * a2_re;
    A1_re -= gT12_im * a2_im;
    spinorFloat A1_im = 0;
    A1_im += gT10_re * a0_im;
    A1_im += gT10_im * a0_re;
    A1_im += gT11_re * a1_im;
    A1_im += gT11_im * a1_re;
    A1_im += gT12_re * a2_im;
    A1_im += gT12_im * a2_re;
    spinorFloat B1_re = 0;
    B1_re += gT10_re * b0_re;
    B1_re -= gT10_im * b0_im;
    B1_re += gT11_re * b1_re;
    B1_re -= gT11_im * b1_im;
    B1_re += gT12_re * b2_re;
    B1_re -= gT12_im * b2_im;
    spinorFloat B1_im = 0;
    B1_im += gT10_re * b0_im;
    B1_im += gT10_im * b0_re;
    B1_im += gT11_re * b1_im;
    B1_im += gT11_im * b1_re;
    B1_im += gT12_re * b2_im;
    B1_im += gT12_im * b2_re;
    
    // multiply row 2
    spinorFloat A2_re = 0;
    A2_re += gT20_re * a0_re;
    A2_re -= gT20_im * a0_im;
    A2_re += gT21_re * a1_re;
    A2_re -= gT21_im * a1_im;
    A2_re += gT22_re * a2_re;
    A2_re -= gT22_im * a2_im;
    spinorFloat A2_im = 0;
    A2_im += gT20_re * a0_im;
    A2_im += gT20_im * a0_re;
    A2_im += gT21_re * a1_im;
    A2_im += gT21_im * a1_re;
    A2_im += gT22_re * a2_im;
    A2_im += gT22_im * a2_re;
    spinorFloat B2_re = 0;
    B2_re += gT20_re * b0_re;
    B2_re -= gT20_im * b0_im;
    B2_re += gT21_re * b1_re;
    B2_re -= gT21_im * b1_im;
    B2_re += gT22_re * b2_re;
    B2_re -= gT22_im * b2_im;
    spinorFloat B2_im = 0;
    B2_im += gT20_re * b0_im;
    B2_im += gT20_im * b0_re;
    B2_im += gT21_re * b1_im;
    B2_im += gT21_im * b1_re;
    B2_im += gT22_re * b2_im;
    B2_im += gT22_im * b2_re;
    
    o1_20_re += A0_re;
    o1_20_im += A0_im;
    o1_30_re += B0_re;
    o1_30_im += B0_im;
    
    o1_21_re += A1_re;
    o1_21_im += A1_im;
    o1_31_re += B1_re;
    o1_31_im += B1_im;
    
    o1_22_re += A2_re;
    o1_22_im += A2_im;
    o1_32_re += B2_re;
    o1_32_im += B2_im;
    
    }
    {
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read flavor 2 from device memory
      READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
      
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
    
#ifdef MULTI_GPU
    } else {
    
    const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];
    
      // read half spinor for the second flavor from device memory
      const int fl_idx = sp_idx + ghostFace[static_cast<int>(kernel_type)];
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, fl_idx, sp_norm_idx+ghostFace[static_cast<int>(kernel_type)]);
      
      a0_re = 2*i00_re;  a0_im = 2*i00_im;
      a1_re = 2*i01_re;  a1_im = 2*i01_im;
      a2_re = 2*i02_re;  a2_im = 2*i02_im;
      b0_re = 2*i10_re;  b0_im = 2*i10_im;
      b1_re = 2*i11_re;  b1_im = 2*i11_im;
      b2_re = 2*i12_re;  b2_im = 2*i12_im;
      
    }
#endif // MULTI_GPU
    
    // multiply row 0
    spinorFloat A0_re = 0;
    A0_re += gT00_re * a0_re;
    A0_re -= gT00_im * a0_im;
    A0_re += gT01_re * a1_re;
    A0_re -= gT01_im * a1_im;
    A0_re += gT02_re * a2_re;
    A0_re -= gT02_im * a2_im;
    spinorFloat A0_im = 0;
    A0_im += gT00_re * a0_im;
    A0_im += gT00_im * a0_re;
    A0_im += gT01_re * a1_im;
    A0_im += gT01_im * a1_re;
    A0_im += gT02_re * a2_im;
    A0_im += gT02_im * a2_re;
    spinorFloat B0_re = 0;
    B0_re += gT00_re * b0_re;
    B0_re -= gT00_im * b0_im;
    B0_re += gT01_re * b1_re;
    B0_re -= gT01_im * b1_im;
    B0_re += gT02_re * b2_re;
    B0_re -= gT02_im * b2_im;
    spinorFloat B0_im = 0;
    B0_im += gT00_re * b0_im;
    B0_im += gT00_im * b0_re;
    B0_im += gT01_re * b1_im;
    B0_im += gT01_im * b1_re;
    B0_im += gT02_re * b2_im;
    B0_im += gT02_im * b2_re;
    
    // multiply row 1
    spinorFloat A1_re = 0;
    A1_re += gT10_re * a0_re;
    A1_re -= gT10_im * a0_im;
    A1_re += gT11_re * a1_re;
    A1_re -= gT11_im * a1_im;
    A1_re += gT12_re * a2_re;
    A1_re -= gT12_im * a2_im;
    spinorFloat A1_im = 0;
    A1_im += gT10_re * a0_im;
    A1_im += gT10_im * a0_re;
    A1_im += gT11_re * a1_im;
    A1_im += gT11_im * a1_re;
    A1_im += gT12_re * a2_im;
    A1_im += gT12_im * a2_re;
    spinorFloat B1_re = 0;
    B1_re += gT10_re * b0_re;
    B1_re -= gT10_im * b0_im;
    B1_re += gT11_re * b1_re;
    B1_re -= gT11_im * b1_im;
    B1_re += gT12_re * b2_re;
    B1_re -= gT12_im * b2_im;
    spinorFloat B1_im = 0;
    B1_im += gT10_re * b0_im;
    B1_im += gT10_im * b0_re;
    B1_im += gT11_re * b1_im;
    B1_im += gT11_im * b1_re;
    B1_im += gT12_re * b2_im;
    B1_im += gT12_im * b2_re;
    
    // multiply row 2
    spinorFloat A2_re = 0;
    A2_re += gT20_re * a0_re;
    A2_re -= gT20_im * a0_im;
    A2_re += gT21_re * a1_re;
    A2_re -= gT21_im * a1_im;
    A2_re += gT22_re * a2_re;
    A2_re -= gT22_im * a2_im;
    spinorFloat A2_im = 0;
    A2_im += gT20_re * a0_im;
    A2_im += gT20_im * a0_re;
    A2_im += gT21_re * a1_im;
    A2_im += gT21_im * a1_re;
    A2_im += gT22_re * a2_im;
    A2_im += gT22_im * a2_re;
    spinorFloat B2_re = 0;
    B2_re += gT20_re * b0_re;
    B2_re -= gT20_im * b0_im;
    B2_re += gT21_re * b1_re;
    B2_re -= gT21_im * b1_im;
    B2_re += gT22_re * b2_re;
    B2_re -= gT22_im * b2_im;
    spinorFloat B2_im = 0;
    B2_im += gT20_re * b0_im;
    B2_im += gT20_im * b0_re;
    B2_im += gT21_re * b1_im;
    B2_im += gT21_im * b1_re;
    B2_im += gT22_re * b2_im;
    B2_im += gT22_im * b2_re;
    
    o2_20_re += A0_re;
    o2_20_im += A0_im;
    o2_30_re += B0_re;
    o2_30_im += B0_im;
    
    o2_21_re += A1_re;
    o2_21_im += A1_im;
    o2_31_re += B1_re;
    o2_31_im += B1_im;
    
    o2_22_re += A2_re;
    o2_22_im += A2_im;
    o2_32_re += B2_re;
    o2_32_im += B2_im;
    
    }
  }
}

#ifdef MULTI_GPU

int incomplete = 0; // Have all 8 contributions been computed for this site?

switch(kernel_type) { // intentional fall-through
case INTERIOR_KERNEL:
  incomplete = incomplete || (param.commDim[3] && (x4==0 || x4==X4m1));
case EXTERIOR_KERNEL_T:
  incomplete = incomplete || (param.commDim[2] && (x3==0 || x3==X3m1));
case EXTERIOR_KERNEL_Z:
  incomplete = incomplete || (param.commDim[1] && (x2==0 || x2==X2m1));
case EXTERIOR_KERNEL_Y:
  incomplete = incomplete || (param.commDim[0] && (x1==0 || x1==X1m1));
}


if (!incomplete)
#endif // MULTI_GPU
// apply twisted mass rotation
{
  
#ifdef DSLASH_TWIST
  {
    //Perform twist rotation first:
    //(1 + i*a*gamma_5 * tau_3 + b * tau_1)
    volatile spinorFloat x1_re, x1_im, y1_re, y1_im;
    volatile spinorFloat x2_re, x2_im, y2_re, y2_im;
    
    x1_re = 0.0, x1_im = 0.0;
    y1_re = 0.0, y1_im = 0.0;
    x2_re = 0.0, x2_im = 0.0;
    y2_re = 0.0, y2_im = 0.0;
    
    
    // using o1 regs:
    x1_re = o1_00_re - a *o1_20_im;
    x1_im = o1_00_im + a *o1_20_re;
    x2_re = b * o1_00_re;
    x2_im = b * o1_00_im;
    
    y1_re = o1_20_re - a *o1_00_im;
    y1_im = o1_20_im + a *o1_00_re;
    y2_re = b * o1_20_re;
    y2_im = b * o1_20_im;
    
    
    // using o2 regs:
    x2_re += o2_00_re + a *o2_20_im;
    x2_im += o2_00_im - a *o2_20_re;
    x1_re += b * o2_00_re;
    x1_im += b * o2_00_im;
    
    y2_re += o2_20_re + a *o2_00_im;
    y2_im += o2_20_im - a *o2_00_re;
    y1_re += b * o2_20_re;
    y1_im += b * o2_20_im;
    
    
    o1_00_re = x1_re;  o1_00_im = x1_im;
    o1_20_re = y1_re;  o1_20_im = y1_im;
    
    o2_00_re = x2_re;  o2_00_im = x2_im;
    o2_20_re = y2_re;  o2_20_im = y2_im;
    
    // using o1 regs:
    x1_re = o1_10_re - a *o1_30_im;
    x1_im = o1_10_im + a *o1_30_re;
    x2_re = b * o1_10_re;
    x2_im = b * o1_10_im;
    
    y1_re = o1_30_re - a *o1_10_im;
    y1_im = o1_30_im + a *o1_10_re;
    y2_re = b * o1_30_re;
    y2_im = b * o1_30_im;
    
    
    // using o2 regs:
    x2_re += o2_10_re + a *o2_30_im;
    x2_im += o2_10_im - a *o2_30_re;
    x1_re += b * o2_10_re;
    x1_im += b * o2_10_im;
    
    y2_re += o2_30_re + a *o2_10_im;
    y2_im += o2_30_im - a *o2_10_re;
    y1_re += b * o2_30_re;
    y1_im += b * o2_30_im;
    
    
    o1_10_re = x1_re;  o1_10_im = x1_im;
    o1_30_re = y1_re;  o1_30_im = y1_im;
    
    o2_10_re = x2_re;  o2_10_im = x2_im;
    o2_30_re = y2_re;  o2_30_im = y2_im;
    
    // using o1 regs:
    x1_re = o1_01_re - a *o1_21_im;
    x1_im = o1_01_im + a *o1_21_re;
    x2_re = b * o1_01_re;
    x2_im = b * o1_01_im;
    
    y1_re = o1_21_re - a *o1_01_im;
    y1_im = o1_21_im + a *o1_01_re;
    y2_re = b * o1_21_re;
    y2_im = b * o1_21_im;
    
    
    // using o2 regs:
    x2_re += o2_01_re + a *o2_21_im;
    x2_im += o2_01_im - a *o2_21_re;
    x1_re += b * o2_01_re;
    x1_im += b * o2_01_im;
    
    y2_re += o2_21_re + a *o2_01_im;
    y2_im += o2_21_im - a *o2_01_re;
    y1_re += b * o2_21_re;
    y1_im += b * o2_21_im;
    
    
    o1_01_re = x1_re;  o1_01_im = x1_im;
    o1_21_re = y1_re;  o1_21_im = y1_im;
    
    o2_01_re = x2_re;  o2_01_im = x2_im;
    o2_21_re = y2_re;  o2_21_im = y2_im;
    
    // using o1 regs:
    x1_re = o1_11_re - a *o1_31_im;
    x1_im = o1_11_im + a *o1_31_re;
    x2_re = b * o1_11_re;
    x2_im = b * o1_11_im;
    
    y1_re = o1_31_re - a *o1_11_im;
    y1_im = o1_31_im + a *o1_11_re;
    y2_re = b * o1_31_re;
    y2_im = b * o1_31_im;
    
    
    // using o2 regs:
    x2_re += o2_11_re + a *o2_31_im;
    x2_im += o2_11_im - a *o2_31_re;
    x1_re += b * o2_11_re;
    x1_im += b * o2_11_im;
    
    y2_re += o2_31_re + a *o2_11_im;
    y2_im += o2_31_im - a *o2_11_re;
    y1_re += b * o2_31_re;
    y1_im += b * o2_31_im;
    
    
    o1_11_re = x1_re;  o1_11_im = x1_im;
    o1_31_re = y1_re;  o1_31_im = y1_im;
    
    o2_11_re = x2_re;  o2_11_im = x2_im;
    o2_31_re = y2_re;  o2_31_im = y2_im;
    
    // using o1 regs:
    x1_re = o1_02_re - a *o1_22_im;
    x1_im = o1_02_im + a *o1_22_re;
    x2_re = b * o1_02_re;
    x2_im = b * o1_02_im;
    
    y1_re = o1_22_re - a *o1_02_im;
    y1_im = o1_22_im + a *o1_02_re;
    y2_re = b * o1_22_re;
    y2_im = b * o1_22_im;
    
    
    // using o2 regs:
    x2_re += o2_02_re + a *o2_22_im;
    x2_im += o2_02_im - a *o2_22_re;
    x1_re += b * o2_02_re;
    x1_im += b * o2_02_im;
    
    y2_re += o2_22_re + a *o2_02_im;
    y2_im += o2_22_im - a *o2_02_re;
    y1_re += b * o2_22_re;
    y1_im += b * o2_22_im;
    
    
    o1_02_re = x1_re;  o1_02_im = x1_im;
    o1_22_re = y1_re;  o1_22_im = y1_im;
    
    o2_02_re = x2_re;  o2_02_im = x2_im;
    o2_22_re = y2_re;  o2_22_im = y2_im;
    
    // using o1 regs:
    x1_re = o1_12_re - a *o1_32_im;
    x1_im = o1_12_im + a *o1_32_re;
    x2_re = b * o1_12_re;
    x2_im = b * o1_12_im;
    
    y1_re = o1_32_re - a *o1_12_im;
    y1_im = o1_32_im + a *o1_12_re;
    y2_re = b * o1_32_re;
    y2_im = b * o1_32_im;
    
    
    // using o2 regs:
    x2_re += o2_12_re + a *o2_32_im;
    x2_im += o2_12_im - a *o2_32_re;
    x1_re += b * o2_12_re;
    x1_im += b * o2_12_im;
    
    y2_re += o2_32_re + a *o2_12_im;
    y2_im += o2_32_im - a *o2_12_re;
    y1_re += b * o2_32_re;
    y1_im += b * o2_32_im;
    
    
    o1_12_re = x1_re;  o1_12_im = x1_im;
    o1_32_re = y1_re;  o1_32_im = y1_im;
    
    o2_12_re = x2_re;  o2_12_im = x2_im;
    o2_32_re = y2_re;  o2_32_im = y2_im;
    
  }
#endif
  
#ifndef DSLASH_XPAY
  o1_00_re *= c;
  o1_00_im *= c;
  o1_01_re *= c;
  o1_01_im *= c;
  o1_02_re *= c;
  o1_02_im *= c;
  o1_10_re *= c;
  o1_10_im *= c;
  o1_11_re *= c;
  o1_11_im *= c;
  o1_12_re *= c;
  o1_12_im *= c;
  o1_20_re *= c;
  o1_20_im *= c;
  o1_21_re *= c;
  o1_21_im *= c;
  o1_22_re *= c;
  o1_22_im *= c;
  o1_30_re *= c;
  o1_30_im *= c;
  o1_31_re *= c;
  o1_31_im *= c;
  o1_32_re *= c;
  o1_32_im *= c;
  
  o2_00_re *= c;
  o2_00_im *= c;
  o2_01_re *= c;
  o2_01_im *= c;
  o2_02_re *= c;
  o2_02_im *= c;
  o2_10_re *= c;
  o2_10_im *= c;
  o2_11_re *= c;
  o2_11_im *= c;
  o2_12_re *= c;
  o2_12_im *= c;
  o2_20_re *= c;
  o2_20_im *= c;
  o2_21_re *= c;
  o2_21_im *= c;
  o2_22_re *= c;
  o2_22_im *= c;
  o2_30_re *= c;
  o2_30_im *= c;
  o2_31_re *= c;
  o2_31_im *= c;
  o2_32_re *= c;
  o2_32_im *= c;
#else
#ifdef DSLASH_TWIST
  // accum spinor
#ifdef SPINOR_DOUBLE
  
#define acc_00_re accum0.x
#define acc_00_im accum0.y
#define acc_01_re accum1.x
#define acc_01_im accum1.y
#define acc_02_re accum2.x
#define acc_02_im accum2.y
#define acc_10_re accum3.x
#define acc_10_im accum3.y
#define acc_11_re accum4.x
#define acc_11_im accum4.y
#define acc_12_re accum5.x
#define acc_12_im accum5.y
#define acc_20_re accum6.x
#define acc_20_im accum6.y
#define acc_21_re accum7.x
#define acc_21_im accum7.y
#define acc_22_re accum8.x
#define acc_22_im accum8.y
#define acc_30_re accum9.x
#define acc_30_im accum9.y
#define acc_31_re accum10.x
#define acc_31_im accum10.y
#define acc_32_re accum11.x
#define acc_32_im accum11.y
  
#else
#define acc_00_re accum0.x
#define acc_00_im accum0.y
#define acc_01_re accum0.z
#define acc_01_im accum0.w
#define acc_02_re accum1.x
#define acc_02_im accum1.y
#define acc_10_re accum1.z
#define acc_10_im accum1.w
#define acc_11_re accum2.x
#define acc_11_im accum2.y
#define acc_12_re accum2.z
#define acc_12_im accum2.w
#define acc_20_re accum3.x
#define acc_20_im accum3.y
#define acc_21_re accum3.z
#define acc_21_im accum3.w
#define acc_22_re accum4.x
#define acc_22_im accum4.y
#define acc_30_re accum4.z
#define acc_30_im accum4.w
#define acc_31_re accum5.x
#define acc_31_im accum5.y
#define acc_32_re accum5.z
#define acc_32_im accum5.w
  
#endif // SPINOR_DOUBLE
  
  {
    READ_ACCUM(ACCUMTEX, sp_stride)
  
    o1_00_re = c*o1_00_re + acc_00_re;
    o1_00_im = c*o1_00_im + acc_00_im;
    o1_01_re = c*o1_01_re + acc_01_re;
    o1_01_im = c*o1_01_im + acc_01_im;
    o1_02_re = c*o1_02_re + acc_02_re;
    o1_02_im = c*o1_02_im + acc_02_im;
    o1_10_re = c*o1_10_re + acc_10_re;
    o1_10_im = c*o1_10_im + acc_10_im;
    o1_11_re = c*o1_11_re + acc_11_re;
    o1_11_im = c*o1_11_im + acc_11_im;
    o1_12_re = c*o1_12_re + acc_12_re;
    o1_12_im = c*o1_12_im + acc_12_im;
    o1_20_re = c*o1_20_re + acc_20_re;
    o1_20_im = c*o1_20_im + acc_20_im;
    o1_21_re = c*o1_21_re + acc_21_re;
    o1_21_im = c*o1_21_im + acc_21_im;
    o1_22_re = c*o1_22_re + acc_22_re;
    o1_22_im = c*o1_22_im + acc_22_im;
    o1_30_re = c*o1_30_re + acc_30_re;
    o1_30_im = c*o1_30_im + acc_30_im;
    o1_31_re = c*o1_31_re + acc_31_re;
    o1_31_im = c*o1_31_im + acc_31_im;
    o1_32_re = c*o1_32_re + acc_32_re;
    o1_32_im = c*o1_32_im + acc_32_im;
  
    ASSN_ACCUM(ACCUMTEX, sp_stride, fl_stride)
  
    o2_00_re = c*o2_00_re + acc_00_re;
    o2_00_im = c*o2_00_im + acc_00_im;
    o2_01_re = c*o2_01_re + acc_01_re;
    o2_01_im = c*o2_01_im + acc_01_im;
    o2_02_re = c*o2_02_re + acc_02_re;
    o2_02_im = c*o2_02_im + acc_02_im;
    o2_10_re = c*o2_10_re + acc_10_re;
    o2_10_im = c*o2_10_im + acc_10_im;
    o2_11_re = c*o2_11_re + acc_11_re;
    o2_11_im = c*o2_11_im + acc_11_im;
    o2_12_re = c*o2_12_re + acc_12_re;
    o2_12_im = c*o2_12_im + acc_12_im;
    o2_20_re = c*o2_20_re + acc_20_re;
    o2_20_im = c*o2_20_im + acc_20_im;
    o2_21_re = c*o2_21_re + acc_21_re;
    o2_21_im = c*o2_21_im + acc_21_im;
    o2_22_re = c*o2_22_re + acc_22_re;
    o2_22_im = c*o2_22_im + acc_22_im;
    o2_30_re = c*o2_30_re + acc_30_re;
    o2_30_im = c*o2_30_im + acc_30_im;
    o2_31_re = c*o2_31_re + acc_31_re;
    o2_31_im = c*o2_31_im + acc_31_im;
    o2_32_re = c*o2_32_re + acc_32_re;
    o2_32_im = c*o2_32_im + acc_32_im;
  }
  
#undef acc_00_re
#undef acc_00_im
#undef acc_01_re
#undef acc_01_im
#undef acc_02_re
#undef acc_02_im
#undef acc_10_re
#undef acc_10_im
#undef acc_11_re
#undef acc_11_im
#undef acc_12_re
#undef acc_12_im
#undef acc_20_re
#undef acc_20_im
#undef acc_21_re
#undef acc_21_im
#undef acc_22_re
#undef acc_22_im
#undef acc_30_re
#undef acc_30_im
#undef acc_31_re
#undef acc_31_im
#undef acc_32_re
#undef acc_32_im
  
#else
  // accum spinor
#ifdef SPINOR_DOUBLE
  
#define acc1_00_re flv1_accum0.x
#define acc1_00_im flv1_accum0.y
#define acc1_01_re flv1_accum1.x
#define acc1_01_im flv1_accum1.y
#define acc1_02_re flv1_accum2.x
#define acc1_02_im flv1_accum2.y
#define acc1_10_re flv1_accum3.x
#define acc1_10_im flv1_accum3.y
#define acc1_11_re flv1_accum4.x
#define acc1_11_im flv1_accum4.y
#define acc1_12_re flv1_accum5.x
#define acc1_12_im flv1_accum5.y
#define acc1_20_re flv1_accum6.x
#define acc1_20_im flv1_accum6.y
#define acc1_21_re flv1_accum7.x
#define acc1_21_im flv1_accum7.y
#define acc1_22_re flv1_accum8.x
#define acc1_22_im flv1_accum8.y
#define acc1_30_re flv1_accum9.x
#define acc1_30_im flv1_accum9.y
#define acc1_31_re flv1_accum10.x
#define acc1_31_im flv1_accum10.y
#define acc1_32_re flv1_accum11.x
#define acc1_32_im flv1_accum11.y
  
#define acc2_00_re flv2_accum0.x
#define acc2_00_im flv2_accum0.y
#define acc2_01_re flv2_accum1.x
#define acc2_01_im flv2_accum1.y
#define acc2_02_re flv2_accum2.x
#define acc2_02_im flv2_accum2.y
#define acc2_10_re flv2_accum3.x
#define acc2_10_im flv2_accum3.y
#define acc2_11_re flv2_accum4.x
#define acc2_11_im flv2_accum4.y
#define acc2_12_re flv2_accum5.x
#define acc2_12_im flv2_accum5.y
#define acc2_20_re flv2_accum6.x
#define acc2_20_im flv2_accum6.y
#define acc2_21_re flv2_accum7.x
#define acc2_21_im flv2_accum7.y
#define acc2_22_re flv2_accum8.x
#define acc2_22_im flv2_accum8.y
#define acc2_30_re flv2_accum9.x
#define acc2_30_im flv2_accum9.y
#define acc2_31_re flv2_accum10.x
#define acc2_31_im flv2_accum10.y
#define acc2_32_re flv2_accum11.x
#define acc2_32_im flv2_accum11.y
  
#else
  
#define acc1_00_re flv1_accum0.x
#define acc1_00_im flv1_accum0.y
#define acc1_01_re flv1_accum0.z
#define acc1_01_im flv1_accum0.w
#define acc1_02_re flv1_accum1.x
#define acc1_02_im flv1_accum1.y
#define acc1_10_re flv1_accum1.z
#define acc1_10_im flv1_accum1.w
#define acc1_11_re flv1_accum2.x
#define acc1_11_im flv1_accum2.y
#define acc1_12_re flv1_accum2.z
#define acc1_12_im flv1_accum2.w
#define acc1_20_re flv1_accum3.x
#define acc1_20_im flv1_accum3.y
#define acc1_21_re flv1_accum3.z
#define acc1_21_im flv1_accum3.w
#define acc1_22_re flv1_accum4.x
#define acc1_22_im flv1_accum4.y
#define acc1_30_re flv1_accum4.z
#define acc1_30_im flv1_accum4.w
#define acc1_31_re flv1_accum5.x
#define acc1_31_im flv1_accum5.y
#define acc1_32_re flv1_accum5.z
#define acc1_32_im flv1_accum5.w
  
#define acc2_00_re flv2_accum0.x
#define acc2_00_im flv2_accum0.y
#define acc2_01_re flv2_accum0.z
#define acc2_01_im flv2_accum0.w
#define acc2_02_re flv2_accum1.x
#define acc2_02_im flv2_accum1.y
#define acc2_10_re flv2_accum1.z
#define acc2_10_im flv2_accum1.w
#define acc2_11_re flv2_accum2.x
#define acc2_11_im flv2_accum2.y
#define acc2_12_re flv2_accum2.z
#define acc2_12_im flv2_accum2.w
#define acc2_20_re flv2_accum3.x
#define acc2_20_im flv2_accum3.y
#define acc2_21_re flv2_accum3.z
#define acc2_21_im flv2_accum3.w
#define acc2_22_re flv2_accum4.x
#define acc2_22_im flv2_accum4.y
#define acc2_30_re flv2_accum4.z
#define acc2_30_im flv2_accum4.w
#define acc2_31_re flv2_accum5.x
#define acc2_31_im flv2_accum5.y
#define acc2_32_re flv2_accum5.z
#define acc2_32_im flv2_accum5.w
  
#endif // SPINOR_DOUBLE
  
  {
    READ_ACCUM_FLAVOR(ACCUMTEX, sp_stride, fl_stride)
  
    //Perform twist rotation:
  //(1 + i*a*gamma_5 * tau_3 + b * tau_1)
    volatile spinorFloat x1_re, x1_im, y1_re, y1_im;
    volatile spinorFloat x2_re, x2_im, y2_re, y2_im;
  
    x1_re = 0.0, x1_im = 0.0;
    y1_re = 0.0, y1_im = 0.0;
    x2_re = 0.0, x2_im = 0.0;
    y2_re = 0.0, y2_im = 0.0;
  
  
    // using acc1 regs:
    x1_re = acc1_00_re - a *acc1_20_im;
    x1_im = acc1_00_im + a *acc1_20_re;
    x2_re = b * acc1_00_re;
    x2_im = b * acc1_00_im;
  
    y1_re = acc1_20_re - a *acc1_00_im;
    y1_im = acc1_20_im + a *acc1_00_re;
    y2_re = b * acc1_20_re;
    y2_im = b * acc1_20_im;
  
  
    // using acc2 regs:
    x2_re += acc2_00_re + a *acc2_20_im;
    x2_im += acc2_00_im - a *acc2_20_re;
    x1_re += b * acc2_00_re;
    x1_im += b * acc2_00_im;
  
    y2_re += acc2_20_re + a *acc2_00_im;
    y2_im += acc2_20_im - a *acc2_00_re;
    y1_re += b * acc2_20_re;
    y1_im += b * acc2_20_im;
  
  
  acc1_00_re = x1_re;  acc1_00_im = x1_im;
  acc1_20_re = y1_re;  acc1_20_im = y1_im;
  
  acc2_00_re = x2_re;  acc2_00_im = x2_im;
  acc2_20_re = y2_re;  acc2_20_im = y2_im;
  
    // using acc1 regs:
    x1_re = acc1_10_re - a *acc1_30_im;
    x1_im = acc1_10_im + a *acc1_30_re;
    x2_re = b * acc1_10_re;
    x2_im = b * acc1_10_im;
  
    y1_re = acc1_30_re - a *acc1_10_im;
    y1_im = acc1_30_im + a *acc1_10_re;
    y2_re = b * acc1_30_re;
    y2_im = b * acc1_30_im;
  
  
    // using acc2 regs:
    x2_re += acc2_10_re + a *acc2_30_im;
    x2_im += acc2_10_im - a *acc2_30_re;
    x1_re += b * acc2_10_re;
    x1_im += b * acc2_10_im;
  
    y2_re += acc2_30_re + a *acc2_10_im;
    y2_im += acc2_30_im - a *acc2_10_re;
    y1_re += b * acc2_30_re;
    y1_im += b * acc2_30_im;
  
  
  acc1_10_re = x1_re;  acc1_10_im = x1_im;
  acc1_30_re = y1_re;  acc1_30_im = y1_im;
  
  acc2_10_re = x2_re;  acc2_10_im = x2_im;
  acc2_30_re = y2_re;  acc2_30_im = y2_im;
  
    // using acc1 regs:
    x1_re = acc1_01_re - a *acc1_21_im;
    x1_im = acc1_01_im + a *acc1_21_re;
    x2_re = b * acc1_01_re;
    x2_im = b * acc1_01_im;
  
    y1_re = acc1_21_re - a *acc1_01_im;
    y1_im = acc1_21_im + a *acc1_01_re;
    y2_re = b * acc1_21_re;
    y2_im = b * acc1_21_im;
  
  
    // using acc2 regs:
    x2_re += acc2_01_re + a *acc2_21_im;
    x2_im += acc2_01_im - a *acc2_21_re;
    x1_re += b * acc2_01_re;
    x1_im += b * acc2_01_im;
  
    y2_re += acc2_21_re + a *acc2_01_im;
    y2_im += acc2_21_im - a *acc2_01_re;
    y1_re += b * acc2_21_re;
    y1_im += b * acc2_21_im;
  
  
  acc1_01_re = x1_re;  acc1_01_im = x1_im;
  acc1_21_re = y1_re;  acc1_21_im = y1_im;
  
  acc2_01_re = x2_re;  acc2_01_im = x2_im;
  acc2_21_re = y2_re;  acc2_21_im = y2_im;
  
    // using acc1 regs:
    x1_re = acc1_11_re - a *acc1_31_im;
    x1_im = acc1_11_im + a *acc1_31_re;
    x2_re = b * acc1_11_re;
    x2_im = b * acc1_11_im;
  
    y1_re = acc1_31_re - a *acc1_11_im;
    y1_im = acc1_31_im + a *acc1_11_re;
    y2_re = b * acc1_31_re;
    y2_im = b * acc1_31_im;
  
  
    // using acc2 regs:
    x2_re += acc2_11_re + a *acc2_31_im;
    x2_im += acc2_11_im - a *acc2_31_re;
    x1_re += b * acc2_11_re;
    x1_im += b * acc2_11_im;
  
    y2_re += acc2_31_re + a *acc2_11_im;
    y2_im += acc2_31_im - a *acc2_11_re;
    y1_re += b * acc2_31_re;
    y1_im += b * acc2_31_im;
  
  
  acc1_11_re = x1_re;  acc1_11_im = x1_im;
  acc1_31_re = y1_re;  acc1_31_im = y1_im;
  
  acc2_11_re = x2_re;  acc2_11_im = x2_im;
  acc2_31_re = y2_re;  acc2_31_im = y2_im;
  
    // using acc1 regs:
    x1_re = acc1_02_re - a *acc1_22_im;
    x1_im = acc1_02_im + a *acc1_22_re;
    x2_re = b * acc1_02_re;
    x2_im = b * acc1_02_im;
  
    y1_re = acc1_22_re - a *acc1_02_im;
    y1_im = acc1_22_im + a *acc1_02_re;
    y2_re = b * acc1_22_re;
    y2_im = b * acc1_22_im;
  
  
    // using acc2 regs:
    x2_re += acc2_02_re + a *acc2_22_im;
    x2_im += acc2_02_im - a *acc2_22_re;
    x1_re += b * acc2_02_re;
    x1_im += b * acc2_02_im;
  
    y2_re += acc2_22_re + a *acc2_02_im;
    y2_im += acc2_22_im - a *acc2_02_re;
    y1_re += b * acc2_22_re;
    y1_im += b * acc2_22_im;
  
  
  acc1_02_re = x1_re;  acc1_02_im = x1_im;
  acc1_22_re = y1_re;  acc1_22_im = y1_im;
  
  acc2_02_re = x2_re;  acc2_02_im = x2_im;
  acc2_22_re = y2_re;  acc2_22_im = y2_im;
  
    // using acc1 regs:
    x1_re = acc1_12_re - a *acc1_32_im;
    x1_im = acc1_12_im + a *acc1_32_re;
    x2_re = b * acc1_12_re;
    x2_im = b * acc1_12_im;
  
    y1_re = acc1_32_re - a *acc1_12_im;
    y1_im = acc1_32_im + a *acc1_12_re;
    y2_re = b * acc1_32_re;
    y2_im = b * acc1_32_im;
  
  
    // using acc2 regs:
    x2_re += acc2_12_re + a *acc2_32_im;
    x2_im += acc2_12_im - a *acc2_32_re;
    x1_re += b * acc2_12_re;
    x1_im += b * acc2_12_im;
  
    y2_re += acc2_32_re + a *acc2_12_im;
    y2_im += acc2_32_im - a *acc2_12_re;
    y1_re += b * acc2_32_re;
    y1_im += b * acc2_32_im;
  
  
  acc1_12_re = x1_re;  acc1_12_im = x1_im;
  acc1_32_re = y1_re;  acc1_32_im = y1_im;
  
  acc2_12_re = x2_re;  acc2_12_im = x2_im;
  acc2_32_re = y2_re;  acc2_32_im = y2_im;
  
    o1_00_re = k*o1_00_re + acc1_00_re;
    o1_00_im = k*o1_00_im + acc1_00_im;
    o1_01_re = k*o1_01_re + acc1_01_re;
    o1_01_im = k*o1_01_im + acc1_01_im;
    o1_02_re = k*o1_02_re + acc1_02_re;
    o1_02_im = k*o1_02_im + acc1_02_im;
    o1_10_re = k*o1_10_re + acc1_10_re;
    o1_10_im = k*o1_10_im + acc1_10_im;
    o1_11_re = k*o1_11_re + acc1_11_re;
    o1_11_im = k*o1_11_im + acc1_11_im;
    o1_12_re = k*o1_12_re + acc1_12_re;
    o1_12_im = k*o1_12_im + acc1_12_im;
    o1_20_re = k*o1_20_re + acc1_20_re;
    o1_20_im = k*o1_20_im + acc1_20_im;
    o1_21_re = k*o1_21_re + acc1_21_re;
    o1_21_im = k*o1_21_im + acc1_21_im;
    o1_22_re = k*o1_22_re + acc1_22_re;
    o1_22_im = k*o1_22_im + acc1_22_im;
    o1_30_re = k*o1_30_re + acc1_30_re;
    o1_30_im = k*o1_30_im + acc1_30_im;
    o1_31_re = k*o1_31_re + acc1_31_re;
    o1_31_im = k*o1_31_im + acc1_31_im;
    o1_32_re = k*o1_32_re + acc1_32_re;
    o1_32_im = k*o1_32_im + acc1_32_im;
  
    o2_00_re = k*o2_00_re + acc2_00_re;
    o2_00_im = k*o2_00_im + acc2_00_im;
    o2_01_re = k*o2_01_re + acc2_01_re;
    o2_01_im = k*o2_01_im + acc2_01_im;
    o2_02_re = k*o2_02_re + acc2_02_re;
    o2_02_im = k*o2_02_im + acc2_02_im;
    o2_10_re = k*o2_10_re + acc2_10_re;
    o2_10_im = k*o2_10_im + acc2_10_im;
    o2_11_re = k*o2_11_re + acc2_11_re;
    o2_11_im = k*o2_11_im + acc2_11_im;
    o2_12_re = k*o2_12_re + acc2_12_re;
    o2_12_im = k*o2_12_im + acc2_12_im;
    o2_20_re = k*o2_20_re + acc2_20_re;
    o2_20_im = k*o2_20_im + acc2_20_im;
    o2_21_re = k*o2_21_re + acc2_21_re;
    o2_21_im = k*o2_21_im + acc2_21_im;
    o2_22_re = k*o2_22_re + acc2_22_re;
    o2_22_im = k*o2_22_im + acc2_22_im;
    o2_30_re = k*o2_30_re + acc2_30_re;
    o2_30_im = k*o2_30_im + acc2_30_im;
    o2_31_re = k*o2_31_re + acc2_31_re;
    o2_31_im = k*o2_31_im + acc2_31_im;
    o2_32_re = k*o2_32_re + acc2_32_re;
    o2_32_im = k*o2_32_im + acc2_32_im;
  }
  
#undef acc1_00_re
#undef acc1_00_im
#undef acc1_01_re
#undef acc1_01_im
#undef acc1_02_re
#undef acc1_02_im
#undef acc1_10_re
#undef acc1_10_im
#undef acc1_11_re
#undef acc1_11_im
#undef acc1_12_re
#undef acc1_12_im
#undef acc1_20_re
#undef acc1_20_im
#undef acc1_21_re
#undef acc1_21_im
#undef acc1_22_re
#undef acc1_22_im
#undef acc1_30_re
#undef acc1_30_im
#undef acc1_31_re
#undef acc1_31_im
#undef acc1_32_re
#undef acc1_32_im
  
#undef acc2_00_re
#undef acc2_00_im
#undef acc2_01_re
#undef acc2_01_im
#undef acc2_02_re
#undef acc2_02_im
#undef acc2_10_re
#undef acc2_10_im
#undef acc2_11_re
#undef acc2_11_im
#undef acc2_12_re
#undef acc2_12_im
#undef acc2_20_re
#undef acc2_20_im
#undef acc2_21_re
#undef acc2_21_im
#undef acc2_22_re
#undef acc2_22_im
#undef acc2_30_re
#undef acc2_30_im
#undef acc2_31_re
#undef acc2_31_im
#undef acc2_32_re
#undef acc2_32_im
  
#endif//DSLASH_TWIST
  
#endif // DSLASH_XPAY
}

// write spinor field back to device memory
WRITE_FLAVOR_SPINOR();

// undefine to prevent warning when precision is changed
#undef spinorFloat
#undef g00_re
#undef g00_im
#undef g01_re
#undef g01_im
#undef g02_re
#undef g02_im
#undef g10_re
#undef g10_im
#undef g11_re
#undef g11_im
#undef g12_re
#undef g12_im
#undef g20_re
#undef g20_im
#undef g21_re
#undef g21_im
#undef g22_re
#undef g22_im

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
