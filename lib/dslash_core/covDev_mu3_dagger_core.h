// *** CUDA DSLASH DAGGER ***

#define DSLASH_SHARED_FLOATS_PER_THREAD 0


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

#include "read_gauge.h"
#include "io_spinor.h"

int x1, x2, x3, x4;
int X;

#if (defined MULTI_GPU) && (DD_PREC==2) // half precision
int sp_norm_idx;
#endif // MULTI_GPU half precision

int sid;

sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= param.threads) return;

#ifdef MULTI_GPU
int face_idx;
if (kernel_type == INTERIOR_KERNEL) {
#endif

  // Inline by hand for the moment and assume even dimensions
  //coordsFromIndex(X, x1, x2, x3, x4, sid, param.parity);

  X = 2*sid;
  int aux1 = X / X1;
  x1 = X - aux1 * X1;
  int aux2 = aux1 / X2;
  x2 = aux1 - aux2 * X2;
  x4 = aux2 / X3;
  x3 = aux2 - x4 * X3;
  aux1 = (param.parity + x4 + x3 + x2) & 1;
  x1 += aux1;
  X += aux1;

  o00_re = 0;  o00_im = 0;
  o01_re = 0;  o01_im = 0;
  o02_re = 0;  o02_im = 0;
  o10_re = 0;  o10_im = 0;
  o11_re = 0;  o11_im = 0;
  o12_re = 0;  o12_im = 0;
  o20_re = 0;  o20_im = 0;
  o21_re = 0;  o21_im = 0;
  o22_re = 0;  o22_im = 0;
  o30_re = 0;  o30_im = 0;
  o31_re = 0;  o31_im = 0;
  o32_re = 0;  o32_im = 0;

#ifdef MULTI_GPU
} else { // exterior kernel

  const int dim = static_cast<int>(kernel_type);
  const int face_volume = param.threads;           // volume of one face
  const int face_num = 0;

  face_idx = sid;               // index into the respective face

  // ghostOffset is scaled to include body (includes stride) and number of FloatN arrays (SPINOR_HOP)
  // face_idx not sid since faces are spin projected and share the same volume index (modulo UP/DOWN reading)
  //sp_idx = face_idx + param.ghostOffset[dim];

#if (DD_PREC==2) // half precision
  sp_norm_idx = sid + param.ghostNormOffset[static_cast<int>(kernel_type)];
#endif

  coordsFromFaceIndex<1>(X, sid, x1, x2, x3, x4, face_idx, face_volume, dim, face_num, param.parity);

  o00_re = 0.;  o00_im = 0.;
  o01_re = 0.;  o01_im = 0.;
  o02_re = 0.;  o02_im = 0.;
  o10_re = 0.;  o10_im = 0.;
  o11_re = 0.;  o11_im = 0.;
  o12_re = 0.;  o12_im = 0.;
  o20_re = 0.;  o20_im = 0.;
  o21_re = 0.;  o21_im = 0.;
  o22_re = 0.;  o22_im = 0.;
  o30_re = 0.;  o30_im = 0.;
  o31_re = 0.;  o31_im = 0.;
  o32_re = 0.;  o32_im = 0.;
}
#endif // MULTI_GPU


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
  
  {
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE1TEX, 7, ga_idx, ga_stride);
    
    
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read spinor from device memory
      READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
      // Do nothing useful with the spinors
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(7);
    
    // multiply row 0
    o00_re += gT00_re * i00_re;
    o00_re -= gT00_im * i00_im;
    o00_re += gT01_re * i01_re;
    o00_re -= gT01_im * i01_im;
    o00_re += gT02_re * i02_re;
    o00_re -= gT02_im * i02_im;
    o00_im += gT00_re * i00_im;
    o00_im += gT00_im * i00_re;
    o00_im += gT01_re * i01_im;
    o00_im += gT01_im * i01_re;
    o00_im += gT02_re * i02_im;
    o00_im += gT02_im * i02_re;
    o10_re += gT00_re * i10_re;
    o10_re -= gT00_im * i10_im;
    o10_re += gT01_re * i11_re;
    o10_re -= gT01_im * i11_im;
    o10_re += gT02_re * i12_re;
    o10_re -= gT02_im * i12_im;
    o10_im += gT00_re * i10_im;
    o10_im += gT00_im * i10_re;
    o10_im += gT01_re * i11_im;
    o10_im += gT01_im * i11_re;
    o10_im += gT02_re * i12_im;
    o10_im += gT02_im * i12_re;
    o20_re += gT00_re * i20_re;
    o20_re -= gT00_im * i20_im;
    o20_re += gT01_re * i21_re;
    o20_re -= gT01_im * i21_im;
    o20_re += gT02_re * i22_re;
    o20_re -= gT02_im * i22_im;
    o20_im += gT00_re * i20_im;
    o20_im += gT00_im * i20_re;
    o20_im += gT01_re * i21_im;
    o20_im += gT01_im * i21_re;
    o20_im += gT02_re * i22_im;
    o20_im += gT02_im * i22_re;
    o30_re += gT00_re * i30_re;
    o30_re -= gT00_im * i30_im;
    o30_re += gT01_re * i31_re;
    o30_re -= gT01_im * i31_im;
    o30_re += gT02_re * i32_re;
    o30_re -= gT02_im * i32_im;
    o30_im += gT00_re * i30_im;
    o30_im += gT00_im * i30_re;
    o30_im += gT01_re * i31_im;
    o30_im += gT01_im * i31_re;
    o30_im += gT02_re * i32_im;
    o30_im += gT02_im * i32_re;
    
    // multiply row 1
    o01_re += gT10_re * i00_re;
    o01_re -= gT10_im * i00_im;
    o01_re += gT11_re * i01_re;
    o01_re -= gT11_im * i01_im;
    o01_re += gT12_re * i02_re;
    o01_re -= gT12_im * i02_im;
    o01_im += gT10_re * i00_im;
    o01_im += gT10_im * i00_re;
    o01_im += gT11_re * i01_im;
    o01_im += gT11_im * i01_re;
    o01_im += gT12_re * i02_im;
    o01_im += gT12_im * i02_re;
    o11_re += gT10_re * i10_re;
    o11_re -= gT10_im * i10_im;
    o11_re += gT11_re * i11_re;
    o11_re -= gT11_im * i11_im;
    o11_re += gT12_re * i12_re;
    o11_re -= gT12_im * i12_im;
    o11_im += gT10_re * i10_im;
    o11_im += gT10_im * i10_re;
    o11_im += gT11_re * i11_im;
    o11_im += gT11_im * i11_re;
    o11_im += gT12_re * i12_im;
    o11_im += gT12_im * i12_re;
    o21_re += gT10_re * i20_re;
    o21_re -= gT10_im * i20_im;
    o21_re += gT11_re * i21_re;
    o21_re -= gT11_im * i21_im;
    o21_re += gT12_re * i22_re;
    o21_re -= gT12_im * i22_im;
    o21_im += gT10_re * i20_im;
    o21_im += gT10_im * i20_re;
    o21_im += gT11_re * i21_im;
    o21_im += gT11_im * i21_re;
    o21_im += gT12_re * i22_im;
    o21_im += gT12_im * i22_re;
    o31_re += gT10_re * i30_re;
    o31_re -= gT10_im * i30_im;
    o31_re += gT11_re * i31_re;
    o31_re -= gT11_im * i31_im;
    o31_re += gT12_re * i32_re;
    o31_re -= gT12_im * i32_im;
    o31_im += gT10_re * i30_im;
    o31_im += gT10_im * i30_re;
    o31_im += gT11_re * i31_im;
    o31_im += gT11_im * i31_re;
    o31_im += gT12_re * i32_im;
    o31_im += gT12_im * i32_re;
    
    // multiply row 2
    o02_re += gT20_re * i00_re;
    o02_re -= gT20_im * i00_im;
    o02_re += gT21_re * i01_re;
    o02_re -= gT21_im * i01_im;
    o02_re += gT22_re * i02_re;
    o02_re -= gT22_im * i02_im;
    o02_im += gT20_re * i00_im;
    o02_im += gT20_im * i00_re;
    o02_im += gT21_re * i01_im;
    o02_im += gT21_im * i01_re;
    o02_im += gT22_re * i02_im;
    o02_im += gT22_im * i02_re;
    o12_re += gT20_re * i10_re;
    o12_re -= gT20_im * i10_im;
    o12_re += gT21_re * i11_re;
    o12_re -= gT21_im * i11_im;
    o12_re += gT22_re * i12_re;
    o12_re -= gT22_im * i12_im;
    o12_im += gT20_re * i10_im;
    o12_im += gT20_im * i10_re;
    o12_im += gT21_re * i11_im;
    o12_im += gT21_im * i11_re;
    o12_im += gT22_re * i12_im;
    o12_im += gT22_im * i12_re;
    o22_re += gT20_re * i20_re;
    o22_re -= gT20_im * i20_im;
    o22_re += gT21_re * i21_re;
    o22_re -= gT21_im * i21_im;
    o22_re += gT22_re * i22_re;
    o22_re -= gT22_im * i22_im;
    o22_im += gT20_re * i20_im;
    o22_im += gT20_im * i20_re;
    o22_im += gT21_re * i21_im;
    o22_im += gT21_im * i21_re;
    o22_im += gT22_re * i22_im;
    o22_im += gT22_im * i22_re;
    o32_re += gT20_re * i30_re;
    o32_re -= gT20_im * i30_im;
    o32_re += gT21_re * i31_re;
    o32_re -= gT21_im * i31_im;
    o32_re += gT22_re * i32_re;
    o32_re -= gT22_im * i32_im;
    o32_im += gT20_re * i30_im;
    o32_im += gT20_im * i30_re;
    o32_im += gT21_re * i31_im;
    o32_im += gT21_im * i31_re;
    o32_im += gT22_re * i32_im;
    o32_im += gT22_im * i32_re;
    
    
#ifdef MULTI_GPU
    //JARLLLL 
    } else {
    
      const int sp_stride_pad = ghostFace[static_cast<int>(kernel_type)];
      
      // read full spinor from device memory
      READ_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
      
      // reconstruct gauge matrix
      RECONSTRUCT_GAUGE_MATRIX(7);
      
      // multiply row 0
      o00_re += gT00_re * i00_re;
      o00_re -= gT00_im * i00_im;
      o00_re += gT01_re * i01_re;
      o00_re -= gT01_im * i01_im;
      o00_re += gT02_re * i02_re;
      o00_re -= gT02_im * i02_im;
      o00_im += gT00_re * i00_im;
      o00_im += gT00_im * i00_re;
      o00_im += gT01_re * i01_im;
      o00_im += gT01_im * i01_re;
      o00_im += gT02_re * i02_im;
      o00_im += gT02_im * i02_re;
      o10_re += gT00_re * i10_re;
      o10_re -= gT00_im * i10_im;
      o10_re += gT01_re * i11_re;
      o10_re -= gT01_im * i11_im;
      o10_re += gT02_re * i12_re;
      o10_re -= gT02_im * i12_im;
      o10_im += gT00_re * i10_im;
      o10_im += gT00_im * i10_re;
      o10_im += gT01_re * i11_im;
      o10_im += gT01_im * i11_re;
      o10_im += gT02_re * i12_im;
      o10_im += gT02_im * i12_re;
      o20_re += gT00_re * i20_re;
      o20_re -= gT00_im * i20_im;
      o20_re += gT01_re * i21_re;
      o20_re -= gT01_im * i21_im;
      o20_re += gT02_re * i22_re;
      o20_re -= gT02_im * i22_im;
      o20_im += gT00_re * i20_im;
      o20_im += gT00_im * i20_re;
      o20_im += gT01_re * i21_im;
      o20_im += gT01_im * i21_re;
      o20_im += gT02_re * i22_im;
      o20_im += gT02_im * i22_re;
      o30_re += gT00_re * i30_re;
      o30_re -= gT00_im * i30_im;
      o30_re += gT01_re * i31_re;
      o30_re -= gT01_im * i31_im;
      o30_re += gT02_re * i32_re;
      o30_re -= gT02_im * i32_im;
      o30_im += gT00_re * i30_im;
      o30_im += gT00_im * i30_re;
      o30_im += gT01_re * i31_im;
      o30_im += gT01_im * i31_re;
      o30_im += gT02_re * i32_im;
      o30_im += gT02_im * i32_re;
      
      // multiply row 1
      o01_re += gT10_re * i00_re;
      o01_re -= gT10_im * i00_im;
      o01_re += gT11_re * i01_re;
      o01_re -= gT11_im * i01_im;
      o01_re += gT12_re * i02_re;
      o01_re -= gT12_im * i02_im;
      o01_im += gT10_re * i00_im;
      o01_im += gT10_im * i00_re;
      o01_im += gT11_re * i01_im;
      o01_im += gT11_im * i01_re;
      o01_im += gT12_re * i02_im;
      o01_im += gT12_im * i02_re;
      o11_re += gT10_re * i10_re;
      o11_re -= gT10_im * i10_im;
      o11_re += gT11_re * i11_re;
      o11_re -= gT11_im * i11_im;
      o11_re += gT12_re * i12_re;
      o11_re -= gT12_im * i12_im;
      o11_im += gT10_re * i10_im;
      o11_im += gT10_im * i10_re;
      o11_im += gT11_re * i11_im;
      o11_im += gT11_im * i11_re;
      o11_im += gT12_re * i12_im;
      o11_im += gT12_im * i12_re;
      o21_re += gT10_re * i20_re;
      o21_re -= gT10_im * i20_im;
      o21_re += gT11_re * i21_re;
      o21_re -= gT11_im * i21_im;
      o21_re += gT12_re * i22_re;
      o21_re -= gT12_im * i22_im;
      o21_im += gT10_re * i20_im;
      o21_im += gT10_im * i20_re;
      o21_im += gT11_re * i21_im;
      o21_im += gT11_im * i21_re;
      o21_im += gT12_re * i22_im;
      o21_im += gT12_im * i22_re;
      o31_re += gT10_re * i30_re;
      o31_re -= gT10_im * i30_im;
      o31_re += gT11_re * i31_re;
      o31_re -= gT11_im * i31_im;
      o31_re += gT12_re * i32_re;
      o31_re -= gT12_im * i32_im;
      o31_im += gT10_re * i30_im;
      o31_im += gT10_im * i30_re;
      o31_im += gT11_re * i31_im;
      o31_im += gT11_im * i31_re;
      o31_im += gT12_re * i32_im;
      o31_im += gT12_im * i32_re;
      
      // multiply row 2
      o02_re += gT20_re * i00_re;
      o02_re -= gT20_im * i00_im;
      o02_re += gT21_re * i01_re;
      o02_re -= gT21_im * i01_im;
      o02_re += gT22_re * i02_re;
      o02_re -= gT22_im * i02_im;
      o02_im += gT20_re * i00_im;
      o02_im += gT20_im * i00_re;
      o02_im += gT21_re * i01_im;
      o02_im += gT21_im * i01_re;
      o02_im += gT22_re * i02_im;
      o02_im += gT22_im * i02_re;
      o12_re += gT20_re * i10_re;
      o12_re -= gT20_im * i10_im;
      o12_re += gT21_re * i11_re;
      o12_re -= gT21_im * i11_im;
      o12_re += gT22_re * i12_re;
      o12_re -= gT22_im * i12_im;
      o12_im += gT20_re * i10_im;
      o12_im += gT20_im * i10_re;
      o12_im += gT21_re * i11_im;
      o12_im += gT21_im * i11_re;
      o12_im += gT22_re * i12_im;
      o12_im += gT22_im * i12_re;
      o22_re += gT20_re * i20_re;
      o22_re -= gT20_im * i20_im;
      o22_re += gT21_re * i21_re;
      o22_re -= gT21_im * i21_im;
      o22_re += gT22_re * i22_re;
      o22_re -= gT22_im * i22_im;
      o22_im += gT20_re * i20_im;
      o22_im += gT20_im * i20_re;
      o22_im += gT21_re * i21_im;
      o22_im += gT21_im * i21_re;
      o22_im += gT22_re * i22_im;
      o22_im += gT22_im * i22_re;
      o32_re += gT20_re * i30_re;
      o32_re -= gT20_im * i30_im;
      o32_re += gT21_re * i31_re;
      o32_re -= gT21_im * i31_im;
      o32_re += gT22_re * i32_re;
      o32_re -= gT22_im * i32_im;
      o32_im += gT20_re * i30_im;
      o32_im += gT20_im * i30_re;
      o32_im += gT21_re * i31_im;
      o32_im += gT21_im * i31_re;
      o32_im += gT22_re * i32_im;
      o32_im += gT22_im * i32_re;
      
    }
#endif // MULTI_GPU
    
  }
}



// write spinor field back to device memory
WRITE_SPINOR(sp_stride);

// undefine to prevent warning when precision is changed
#undef spinorFloat
#undef SHARED_STRIDE

#undef g00_re
#undef g00_im
#undef gT00_re
#undef gT00_im
#undef g01_re
#undef g01_im
#undef gT01_re
#undef gT01_im
#undef g02_re
#undef g02_im
#undef gT02_re
#undef gT02_im
#undef g10_re
#undef g10_im
#undef gT10_re
#undef gT10_im
#undef g11_re
#undef g11_im
#undef gT11_re
#undef gT11_im
#undef g12_re
#undef g12_im
#undef gT12_re
#undef gT12_im
#undef g20_re
#undef g20_im
#undef gT20_re
#undef gT20_im
#undef g21_re
#undef g21_im
#undef gT21_re
#undef gT21_im
#undef g22_re
#undef g22_im
#undef gT22_re
#undef gT22_im

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


