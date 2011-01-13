//J  dslash_dagger_dwf_core.h
//J  Ver. 09.10.a

//  goto HERE to continue checking


//J  Q. Where do the diagonal components
//J       += (m0-5) psi(x,s)
//J   get performed?  Not in this hopping
//J   file.  Here, m0 is the dwf barrier
//J   height, related to Andrew P.'s documentation mdwf.pdf
//J   by m0= -M5.
//J  A. They get carried out using the xpay
//J   operations in dslash_dwf_cuda.cu.
//J   These are defined in the dslash_dwf_post.h that is
//J   included at the end of this file.
//

//J  Carry out the 4d operations with this include.
// It does not undefine things.  That comes
// at the end of this file, through another include.
//#include "dslash_dagger_core_ante.h"

// *** CUDA DSLASH DAGGER ***

//#define SHARED_FLOATS_PER_THREAD 0 // FIXME
#define SHARED_BYTES_DOUBLE (BLOCK_DIM*SHARED_FLOATS_PER_THREAD*sizeof(double))

#define SHARED_BYTES_SINGLE (BLOCK_DIM*SHARED_FLOATS_PER_THREAD*sizeof(float))

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
#endif

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
// temporaries
#define A_re G9.x
#define A_im G9.y

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
// temporaries
#define A_re G4.z
#define A_im G4.w

#endif

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
volatile spinorFloat o00_re;
volatile spinorFloat o00_im;
volatile spinorFloat o01_re;
volatile spinorFloat o01_im;
volatile spinorFloat o02_re;
volatile spinorFloat o02_im;
volatile spinorFloat o10_re;
volatile spinorFloat o10_im;
volatile spinorFloat o11_re;
volatile spinorFloat o11_im;
volatile spinorFloat o12_re;
volatile spinorFloat o12_im;
volatile spinorFloat o20_re;
volatile spinorFloat o20_im;
volatile spinorFloat o21_re;
volatile spinorFloat o21_im;
volatile spinorFloat o22_re;
volatile spinorFloat o22_im;
volatile spinorFloat o30_re;
volatile spinorFloat o30_im;
volatile spinorFloat o31_re;
volatile spinorFloat o31_im;
volatile spinorFloat o32_re;
volatile spinorFloat o32_im;



#include "read_gauge.h"
//#include "read_clover.h"
#include "io_spinor.h"

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= param.threads) return;
int boundaryCrossings = sid/X1h + sid/(X2*X1h) + sid/(X3*X2*X1h) + sid/(X4*X3*X2*X1h);
int boundaryCrossings4d = sid/X1h + sid/(X2*X1h) + sid/(X3*X2*X1h);
int X = 2*sid + (boundaryCrossings + param.parity) % 2;
int xs = X/(X4*X3*X2*X1);
int x4 = (X/(X3*X2*X1)) % X4;
int x3 = (X/(X2*X1)) % X3;
int x2 = (X/X1) % X2;
int x1 = X % X1;

o00_re = o00_im = 0;
o01_re = o01_im = 0;
o02_re = o02_im = 0;
o10_re = o10_im = 0;
o11_re = o11_im = 0;
o12_re = o12_im = 0;
o20_re = o20_im = 0;
o21_re = o21_im = 0;
o22_re = o22_im = 0;
o30_re = o30_im = 0;
o31_re = o31_im = 0;
o32_re = o32_im = 0;

{
  // Projector P0+
  // 1 0 0 i 
  // 0 1 i 0 
  // 0 -i 1 0 
  // -i 0 0 1 
    
  int sp_idx = ((x1==X1-1) ? X-(X1-1) : X+1) / 2;
  int ga_idx = sid % Vh;
  if ( !( (boundaryCrossings-boundaryCrossings4d) % 2) ) {    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 0);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(0);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re-i30_im;
    spinorFloat a0_im = +i00_im+i30_re;
    spinorFloat a1_re = +i01_re-i31_im;
    spinorFloat a1_im = +i01_im+i31_re;
    spinorFloat a2_re = +i02_re-i32_im;
    spinorFloat a2_im = +i02_im+i32_re;
    
    spinorFloat b0_re = +i10_re-i20_im;
    spinorFloat b0_im = +i10_im+i20_re;
    spinorFloat b1_re = +i11_re-i21_im;
    spinorFloat b1_im = +i11_im+i21_re;
    spinorFloat b2_re = +i12_re-i22_im;
    spinorFloat b2_im = +i12_im+i22_re;
    
    // multiply row 0
    spinorFloat A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
    spinorFloat A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
    spinorFloat B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
    spinorFloat B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
    spinorFloat A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
    spinorFloat B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
    spinorFloat B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
    spinorFloat A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
    spinorFloat B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
    spinorFloat B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re += B0_im;
    o20_im -= B0_re;
    o30_re += A0_im;
    o30_im -= A0_re;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re += B1_im;
    o21_im -= B1_re;
    o31_re += A1_im;
    o31_im -= A1_re;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re += B2_im;
    o22_im -= B2_re;
    o32_re += A2_im;
    o32_im -= A2_re;
  
  } else {
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 0);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(0);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re-i30_im;
    spinorFloat a0_im = +i00_im+i30_re;
    spinorFloat a1_re = +i01_re-i31_im;
    spinorFloat a1_im = +i01_im+i31_re;
    spinorFloat a2_re = +i02_re-i32_im;
    spinorFloat a2_im = +i02_im+i32_re;
    
    spinorFloat b0_re = +i10_re-i20_im;
    spinorFloat b0_im = +i10_im+i20_re;
    spinorFloat b1_re = +i11_re-i21_im;
    spinorFloat b1_im = +i11_im+i21_re;
    spinorFloat b2_re = +i12_re-i22_im;
    spinorFloat b2_im = +i12_im+i22_re;
    
    // multiply row 0
    spinorFloat A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
    spinorFloat A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
    spinorFloat B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
    spinorFloat B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
    spinorFloat A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
    spinorFloat B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
    spinorFloat B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
    spinorFloat A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
    spinorFloat B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
    spinorFloat B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re += B0_im;
    o20_im -= B0_re;
    o30_re += A0_im;
    o30_im -= A0_re;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re += B1_im;
    o21_im -= B1_re;
    o31_re += A1_im;
    o31_im -= A1_re;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re += B2_im;
    o22_im -= B2_re;
    o32_re += A2_im;
    o32_im -= A2_re;
  }
}

{
  // Projector P0-
  // 1 0 0 -i 
  // 0 1 -i 0 
  // 0 i 1 0 
  // i 0 0 1 
    
  int sp_idx = ((x1==0)    ? X+(X1-1) : X-1) / 2;
  int ga_idx = sp_idx % Vh;

  if ( !( (boundaryCrossings-boundaryCrossings4d) % 2) ) {    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 1);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(1);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re+i30_im;
    spinorFloat a0_im = +i00_im-i30_re;
    spinorFloat a1_re = +i01_re+i31_im;
    spinorFloat a1_im = +i01_im-i31_re;
    spinorFloat a2_re = +i02_re+i32_im;
    spinorFloat a2_im = +i02_im-i32_re;
    
    spinorFloat b0_re = +i10_re+i20_im;
    spinorFloat b0_im = +i10_im-i20_re;
    spinorFloat b1_re = +i11_re+i21_im;
    spinorFloat b1_im = +i11_im-i21_re;
    spinorFloat b2_re = +i12_re+i22_im;
    spinorFloat b2_im = +i12_im-i22_re;
    
    // multiply row 0
    spinorFloat A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
    spinorFloat A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
    spinorFloat B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
    spinorFloat B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
    spinorFloat A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
    spinorFloat B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
    spinorFloat B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
    spinorFloat A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
    spinorFloat B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
    spinorFloat B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re -= B0_im;
    o20_im += B0_re;
    o30_re -= A0_im;
    o30_im += A0_re;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re -= B1_im;
    o21_im += B1_re;
    o31_re -= A1_im;
    o31_im += A1_re;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re -= B2_im;
    o22_im += B2_re;
    o32_re -= A2_im;
    o32_im += A2_re;
  } else {
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 1);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(1);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re+i30_im;
    spinorFloat a0_im = +i00_im-i30_re;
    spinorFloat a1_re = +i01_re+i31_im;
    spinorFloat a1_im = +i01_im-i31_re;
    spinorFloat a2_re = +i02_re+i32_im;
    spinorFloat a2_im = +i02_im-i32_re;
    
    spinorFloat b0_re = +i10_re+i20_im;
    spinorFloat b0_im = +i10_im-i20_re;
    spinorFloat b1_re = +i11_re+i21_im;
    spinorFloat b1_im = +i11_im-i21_re;
    spinorFloat b2_re = +i12_re+i22_im;
    spinorFloat b2_im = +i12_im-i22_re;
    
    // multiply row 0
    spinorFloat A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
    spinorFloat A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
    spinorFloat B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
    spinorFloat B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
    spinorFloat A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
    spinorFloat B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
    spinorFloat B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
    spinorFloat A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
    spinorFloat B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
    spinorFloat B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re -= B0_im;
    o20_im += B0_re;
    o30_re -= A0_im;
    o30_im += A0_re;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re -= B1_im;
    o21_im += B1_re;
    o31_re -= A1_im;
    o31_im += A1_re;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re -= B2_im;
    o22_im += B2_re;
    o32_re -= A2_im;
    o32_im += A2_re;
  }
}

{
  // Projector P1+
  // 1 0 0 1 
  // 0 1 -1 0 
  // 0 -1 1 0 
  // 1 0 0 1 
    
  int sp_idx = ((x2==X2-1) ? X-(X2-1)*X1 : X+X1) / 2;
  int ga_idx = sid % Vh;
  
  if ( !( (boundaryCrossings-boundaryCrossings4d) % 2) ) {
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 2);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(2);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re+i30_re;
    spinorFloat a0_im = +i00_im+i30_im;
    spinorFloat a1_re = +i01_re+i31_re;
    spinorFloat a1_im = +i01_im+i31_im;
    spinorFloat a2_re = +i02_re+i32_re;
    spinorFloat a2_im = +i02_im+i32_im;
    
    spinorFloat b0_re = +i10_re-i20_re;
    spinorFloat b0_im = +i10_im-i20_im;
    spinorFloat b1_re = +i11_re-i21_re;
    spinorFloat b1_im = +i11_im-i21_im;
    spinorFloat b2_re = +i12_re-i22_re;
    spinorFloat b2_im = +i12_im-i22_im;
    
    // multiply row 0
    spinorFloat A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
    spinorFloat A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
    spinorFloat B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
    spinorFloat B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
    spinorFloat A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
    spinorFloat B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
    spinorFloat B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
    spinorFloat A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
    spinorFloat B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
    spinorFloat B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re -= B0_re;
    o20_im -= B0_im;
    o30_re += A0_re;
    o30_im += A0_im;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re -= B1_re;
    o21_im -= B1_im;
    o31_re += A1_re;
    o31_im += A1_im;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re -= B2_re;
    o22_im -= B2_im;
    o32_re += A2_re;
    o32_im += A2_im;
  
  } else {
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 2);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(2);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re+i30_re;
    spinorFloat a0_im = +i00_im+i30_im;
    spinorFloat a1_re = +i01_re+i31_re;
    spinorFloat a1_im = +i01_im+i31_im;
    spinorFloat a2_re = +i02_re+i32_re;
    spinorFloat a2_im = +i02_im+i32_im;
    
    spinorFloat b0_re = +i10_re-i20_re;
    spinorFloat b0_im = +i10_im-i20_im;
    spinorFloat b1_re = +i11_re-i21_re;
    spinorFloat b1_im = +i11_im-i21_im;
    spinorFloat b2_re = +i12_re-i22_re;
    spinorFloat b2_im = +i12_im-i22_im;
    
    // multiply row 0
    spinorFloat A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
    spinorFloat A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
    spinorFloat B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
    spinorFloat B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
    spinorFloat A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
    spinorFloat B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
    spinorFloat B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
    spinorFloat A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
    spinorFloat B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
    spinorFloat B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re -= B0_re;
    o20_im -= B0_im;
    o30_re += A0_re;
    o30_im += A0_im;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re -= B1_re;
    o21_im -= B1_im;
    o31_re += A1_re;
    o31_im += A1_im;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re -= B2_re;
    o22_im -= B2_im;
    o32_re += A2_re;
    o32_im += A2_im;

  }
}

{
    // Projector P1-
    // 1 0 0 -1 
    // 0 1 1 0 
    // 0 1 1 0 
    // -1 0 0 1 
    
    int sp_idx = ((x2==0)    ? X+(X2-1)*X1 : X-X1) / 2;
    int ga_idx = sp_idx % Vh;
  
  if ( !( (boundaryCrossings-boundaryCrossings4d) % 2) ) {
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 3);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(3);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re-i30_re;
    spinorFloat a0_im = +i00_im-i30_im;
    spinorFloat a1_re = +i01_re-i31_re;
    spinorFloat a1_im = +i01_im-i31_im;
    spinorFloat a2_re = +i02_re-i32_re;
    spinorFloat a2_im = +i02_im-i32_im;
    
    spinorFloat b0_re = +i10_re+i20_re;
    spinorFloat b0_im = +i10_im+i20_im;
    spinorFloat b1_re = +i11_re+i21_re;
    spinorFloat b1_im = +i11_im+i21_im;
    spinorFloat b2_re = +i12_re+i22_re;
    spinorFloat b2_im = +i12_im+i22_im;
    
    // multiply row 0
    spinorFloat A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
    spinorFloat A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
    spinorFloat B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
    spinorFloat B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
    spinorFloat A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
    spinorFloat B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
    spinorFloat B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
    spinorFloat A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
    spinorFloat B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
    spinorFloat B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re += B0_re;
    o20_im += B0_im;
    o30_re -= A0_re;
    o30_im -= A0_im;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re += B1_re;
    o21_im += B1_im;
    o31_re -= A1_re;
    o31_im -= A1_im;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re += B2_re;
    o22_im += B2_im;
    o32_re -= A2_re;
    o32_im -= A2_im;
  } else {
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 3);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(3);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re-i30_re;
    spinorFloat a0_im = +i00_im-i30_im;
    spinorFloat a1_re = +i01_re-i31_re;
    spinorFloat a1_im = +i01_im-i31_im;
    spinorFloat a2_re = +i02_re-i32_re;
    spinorFloat a2_im = +i02_im-i32_im;
    
    spinorFloat b0_re = +i10_re+i20_re;
    spinorFloat b0_im = +i10_im+i20_im;
    spinorFloat b1_re = +i11_re+i21_re;
    spinorFloat b1_im = +i11_im+i21_im;
    spinorFloat b2_re = +i12_re+i22_re;
    spinorFloat b2_im = +i12_im+i22_im;
    
    // multiply row 0
    spinorFloat A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
    spinorFloat A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
    spinorFloat B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
    spinorFloat B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
    spinorFloat A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
    spinorFloat B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
    spinorFloat B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
    spinorFloat A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
    spinorFloat B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
    spinorFloat B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re += B0_re;
    o20_im += B0_im;
    o30_re -= A0_re;
    o30_im -= A0_im;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re += B1_re;
    o21_im += B1_im;
    o31_re -= A1_re;
    o31_im -= A1_im;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re += B2_re;
    o22_im += B2_im;
    o32_re -= A2_re;
    o32_im -= A2_im;
  }
}

{
    // Projector P2+
    // 1 0 i 0 
    // 0 1 0 -i 
    // -i 0 1 0 
    // 0 i 0 1 
    
    int sp_idx = ((x3==X3-1) ? X-(X3-1)*X2*X1 : X+X2*X1) / 2;
    int ga_idx = sid % Vh;
  
  if ( !( (boundaryCrossings-boundaryCrossings4d) % 2) ) {
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 4);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(4);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re-i20_im;
    spinorFloat a0_im = +i00_im+i20_re;
    spinorFloat a1_re = +i01_re-i21_im;
    spinorFloat a1_im = +i01_im+i21_re;
    spinorFloat a2_re = +i02_re-i22_im;
    spinorFloat a2_im = +i02_im+i22_re;
    
    spinorFloat b0_re = +i10_re+i30_im;
    spinorFloat b0_im = +i10_im-i30_re;
    spinorFloat b1_re = +i11_re+i31_im;
    spinorFloat b1_im = +i11_im-i31_re;
    spinorFloat b2_re = +i12_re+i32_im;
    spinorFloat b2_im = +i12_im-i32_re;
    
    // multiply row 0
    spinorFloat A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
    spinorFloat A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
    spinorFloat B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
    spinorFloat B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
    spinorFloat A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
    spinorFloat B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
    spinorFloat B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
    spinorFloat A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
    spinorFloat B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
    spinorFloat B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re += A0_im;
    o20_im -= A0_re;
    o30_re -= B0_im;
    o30_im += B0_re;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re += A1_im;
    o21_im -= A1_re;
    o31_re -= B1_im;
    o31_im += B1_re;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re += A2_im;
    o22_im -= A2_re;
    o32_re -= B2_im;
    o32_im += B2_re;
  
  } else {
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 4);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(4);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re-i20_im;
    spinorFloat a0_im = +i00_im+i20_re;
    spinorFloat a1_re = +i01_re-i21_im;
    spinorFloat a1_im = +i01_im+i21_re;
    spinorFloat a2_re = +i02_re-i22_im;
    spinorFloat a2_im = +i02_im+i22_re;
    
    spinorFloat b0_re = +i10_re+i30_im;
    spinorFloat b0_im = +i10_im-i30_re;
    spinorFloat b1_re = +i11_re+i31_im;
    spinorFloat b1_im = +i11_im-i31_re;
    spinorFloat b2_re = +i12_re+i32_im;
    spinorFloat b2_im = +i12_im-i32_re;
    
    // multiply row 0
    spinorFloat A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
    spinorFloat A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
    spinorFloat B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
    spinorFloat B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
    spinorFloat A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
    spinorFloat B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
    spinorFloat B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
    spinorFloat A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
    spinorFloat B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
    spinorFloat B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re += A0_im;
    o20_im -= A0_re;
    o30_re -= B0_im;
    o30_im += B0_re;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re += A1_im;
    o21_im -= A1_re;
    o31_re -= B1_im;
    o31_im += B1_re;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re += A2_im;
    o22_im -= A2_re;
    o32_re -= B2_im;
    o32_im += B2_re;
  }
}

{
    // Projector P2-
    // 1 0 -i 0 
    // 0 1 0 i 
    // i 0 1 0 
    // 0 -i 0 1 
    
    int sp_idx = ((x3==0)    ? X+(X3-1)*X2*X1 : X-X2*X1) / 2;
    int ga_idx = sp_idx % Vh;
    
  if ( !( (boundaryCrossings-boundaryCrossings4d) % 2) ) {
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 5);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(5);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re+i20_im;
    spinorFloat a0_im = +i00_im-i20_re;
    spinorFloat a1_re = +i01_re+i21_im;
    spinorFloat a1_im = +i01_im-i21_re;
    spinorFloat a2_re = +i02_re+i22_im;
    spinorFloat a2_im = +i02_im-i22_re;
    
    spinorFloat b0_re = +i10_re-i30_im;
    spinorFloat b0_im = +i10_im+i30_re;
    spinorFloat b1_re = +i11_re-i31_im;
    spinorFloat b1_im = +i11_im+i31_re;
    spinorFloat b2_re = +i12_re-i32_im;
    spinorFloat b2_im = +i12_im+i32_re;
    
    // multiply row 0
    spinorFloat A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
    spinorFloat A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
    spinorFloat B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
    spinorFloat B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
    spinorFloat A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
    spinorFloat B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
    spinorFloat B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
    spinorFloat A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
    spinorFloat B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
    spinorFloat B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re -= A0_im;
    o20_im += A0_re;
    o30_re += B0_im;
    o30_im -= B0_re;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re -= A1_im;
    o21_im += A1_re;
    o31_re += B1_im;
    o31_im -= B1_re;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re -= A2_im;
    o22_im += A2_re;
    o32_re += B2_im;
    o32_im -= B2_re;
  
  } else {
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 5);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(5);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re+i20_im;
    spinorFloat a0_im = +i00_im-i20_re;
    spinorFloat a1_re = +i01_re+i21_im;
    spinorFloat a1_im = +i01_im-i21_re;
    spinorFloat a2_re = +i02_re+i22_im;
    spinorFloat a2_im = +i02_im-i22_re;
    
    spinorFloat b0_re = +i10_re-i30_im;
    spinorFloat b0_im = +i10_im+i30_re;
    spinorFloat b1_re = +i11_re-i31_im;
    spinorFloat b1_im = +i11_im+i31_re;
    spinorFloat b2_re = +i12_re-i32_im;
    spinorFloat b2_im = +i12_im+i32_re;
    
    // multiply row 0
    spinorFloat A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
    spinorFloat A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
    spinorFloat B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
    spinorFloat B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
    spinorFloat A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
    spinorFloat B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
    spinorFloat B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
    spinorFloat A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
    spinorFloat B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
    spinorFloat B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re -= A0_im;
    o20_im += A0_re;
    o30_re += B0_im;
    o30_im -= B0_re;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re -= A1_im;
    o21_im += A1_re;
    o31_re += B1_im;
    o31_im -= B1_re;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re -= A2_im;
    o22_im += A2_re;
    o32_re += B2_im;
    o32_im -= B2_re;

  }
}

{
    // Projector P3+
    // 2 0 0 0 
    // 0 2 0 0 
    // 0 0 0 0 
    // 0 0 0 0 
    
    int sp_idx = ((x4==X4-1) ? X-(X4-1)*X3*X2*X1 : X+X3*X2*X1) / 2;
    int ga_idx = sid % Vh;
    
    if (gauge_fixed && ga_idx < (X4-1)*X1h*X2*X3) {
        // read spinor from device memory
        READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);
        
        // project spinor into half spinors
        spinorFloat a0_re = +2*i00_re;
        spinorFloat a0_im = +2*i00_im;
        spinorFloat a1_re = +2*i01_re;
        spinorFloat a1_im = +2*i01_im;
        spinorFloat a2_re = +2*i02_re;
        spinorFloat a2_im = +2*i02_im;
        
        spinorFloat b0_re = +2*i10_re;
        spinorFloat b0_im = +2*i10_im;
        spinorFloat b1_re = +2*i11_re;
        spinorFloat b1_im = +2*i11_im;
        spinorFloat b2_re = +2*i12_re;
        spinorFloat b2_im = +2*i12_im;
        
        // identity gauge matrix
        spinorFloat A0_re = a0_re; spinorFloat A0_im = a0_im;
        spinorFloat B0_re = b0_re; spinorFloat B0_im = b0_im;
        spinorFloat A1_re = a1_re; spinorFloat A1_im = a1_im;
        spinorFloat B1_re = b1_re; spinorFloat B1_im = b1_im;
        spinorFloat A2_re = a2_re; spinorFloat A2_im = a2_im;
        spinorFloat B2_re = b2_re; spinorFloat B2_im = b2_im;
        
        o00_re += A0_re;
        o00_im += A0_im;
        o10_re += B0_re;
        o10_im += B0_im;
        
        o01_re += A1_re;
        o01_im += A1_im;
        o11_re += B1_re;
        o11_im += B1_im;
        
        o02_re += A2_re;
        o02_im += A2_im;
        o12_re += B2_re;
        o12_im += B2_im;
        
    } else {
     
     if ( !( (boundaryCrossings-boundaryCrossings4d) % 2) ) {
       
       // read gauge matrix from device memory
        READ_GAUGE_MATRIX(GAUGE0TEX, 6);
        
        // read spinor from device memory
        READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);
        
        // reconstruct gauge matrix
        RECONSTRUCT_GAUGE_MATRIX(6);
        
        // project spinor into half spinors
        spinorFloat a0_re = +2*i00_re;
        spinorFloat a0_im = +2*i00_im;
        spinorFloat a1_re = +2*i01_re;
        spinorFloat a1_im = +2*i01_im;
        spinorFloat a2_re = +2*i02_re;
        spinorFloat a2_im = +2*i02_im;
        
        spinorFloat b0_re = +2*i10_re;
        spinorFloat b0_im = +2*i10_im;
        spinorFloat b1_re = +2*i11_re;
        spinorFloat b1_im = +2*i11_im;
        spinorFloat b2_re = +2*i12_re;
        spinorFloat b2_im = +2*i12_im;
        
        // multiply row 0
        spinorFloat A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
        spinorFloat A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
        spinorFloat B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
        spinorFloat B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
        
        // multiply row 1
        spinorFloat A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
        spinorFloat A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
        spinorFloat B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
        spinorFloat B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
        
        // multiply row 2
        spinorFloat A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
        spinorFloat A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
        spinorFloat B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
        spinorFloat B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
        
        o00_re += A0_re;
        o00_im += A0_im;
        o10_re += B0_re;
        o10_im += B0_im;
        
        o01_re += A1_re;
        o01_im += A1_im;
        o11_re += B1_re;
        o11_im += B1_im;
        
        o02_re += A2_re;
        o02_im += A2_im;
        o12_re += B2_re;
        o12_im += B2_im;
      } else {
        
        // read gauge matrix from device memory
        READ_GAUGE_MATRIX(GAUGE1TEX, 6);
        
        // read spinor from device memory
        READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);
        
        // reconstruct gauge matrix
        RECONSTRUCT_GAUGE_MATRIX(6);
        
        // project spinor into half spinors
        spinorFloat a0_re = +2*i00_re;
        spinorFloat a0_im = +2*i00_im;
        spinorFloat a1_re = +2*i01_re;
        spinorFloat a1_im = +2*i01_im;
        spinorFloat a2_re = +2*i02_re;
        spinorFloat a2_im = +2*i02_im;
        
        spinorFloat b0_re = +2*i10_re;
        spinorFloat b0_im = +2*i10_im;
        spinorFloat b1_re = +2*i11_re;
        spinorFloat b1_im = +2*i11_im;
        spinorFloat b2_re = +2*i12_re;
        spinorFloat b2_im = +2*i12_im;
        
        // multiply row 0
        spinorFloat A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
        spinorFloat A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
        spinorFloat B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
        spinorFloat B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
        
        // multiply row 1
        spinorFloat A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
        spinorFloat A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
        spinorFloat B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
        spinorFloat B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
        
        // multiply row 2
        spinorFloat A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
        spinorFloat A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
        spinorFloat B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
        spinorFloat B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
        
        o00_re += A0_re;
        o00_im += A0_im;
        o10_re += B0_re;
        o10_im += B0_im;
        
        o01_re += A1_re;
        o01_im += A1_im;
        o11_re += B1_re;
        o11_im += B1_im;
        
        o02_re += A2_re;
        o02_im += A2_im;
        o12_re += B2_re;
        o12_im += B2_im;

      }
   }
}

{
    // Projector P3-
    // 0 0 0 0 
    // 0 0 0 0 
    // 0 0 2 0 
    // 0 0 0 2 
    
    int sp_idx = ((x4==0)    ? X+(X4-1)*X3*X2*X1 : X-X3*X2*X1) / 2;
    int ga_idx = sp_idx % Vh;
    
    if (gauge_fixed && ga_idx < (X4-1)*X1h*X2*X3) {
        // read spinor from device memory
        READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);
        
        // project spinor into half spinors
        spinorFloat a0_re = +2*i20_re;
        spinorFloat a0_im = +2*i20_im;
        spinorFloat a1_re = +2*i21_re;
        spinorFloat a1_im = +2*i21_im;
        spinorFloat a2_re = +2*i22_re;
        spinorFloat a2_im = +2*i22_im;
        
        spinorFloat b0_re = +2*i30_re;
        spinorFloat b0_im = +2*i30_im;
        spinorFloat b1_re = +2*i31_re;
        spinorFloat b1_im = +2*i31_im;
        spinorFloat b2_re = +2*i32_re;
        spinorFloat b2_im = +2*i32_im;
        
        // identity gauge matrix
        spinorFloat A0_re = a0_re; spinorFloat A0_im = a0_im;
        spinorFloat B0_re = b0_re; spinorFloat B0_im = b0_im;
        spinorFloat A1_re = a1_re; spinorFloat A1_im = a1_im;
        spinorFloat B1_re = b1_re; spinorFloat B1_im = b1_im;
        spinorFloat A2_re = a2_re; spinorFloat A2_im = a2_im;
        spinorFloat B2_re = b2_re; spinorFloat B2_im = b2_im;
        
        o20_re += A0_re;
        o20_im += A0_im;
        o30_re += B0_re;
        o30_im += B0_im;
        
        o21_re += A1_re;
        o21_im += A1_im;
        o31_re += B1_re;
        o31_im += B1_im;
        
        o22_re += A2_re;
        o22_im += A2_im;
        o32_re += B2_re;
        o32_im += B2_im;
        
    } else {
        
      if ( !( (boundaryCrossings-boundaryCrossings4d) % 2) ) {
        
        // read gauge matrix from device memory
        READ_GAUGE_MATRIX(GAUGE1TEX, 7);
        
        // read spinor from device memory
        READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);
        
        // reconstruct gauge matrix
        RECONSTRUCT_GAUGE_MATRIX(7);
        
        // project spinor into half spinors
        spinorFloat a0_re = +2*i20_re;
        spinorFloat a0_im = +2*i20_im;
        spinorFloat a1_re = +2*i21_re;
        spinorFloat a1_im = +2*i21_im;
        spinorFloat a2_re = +2*i22_re;
        spinorFloat a2_im = +2*i22_im;
        
        spinorFloat b0_re = +2*i30_re;
        spinorFloat b0_im = +2*i30_im;
        spinorFloat b1_re = +2*i31_re;
        spinorFloat b1_im = +2*i31_im;
        spinorFloat b2_re = +2*i32_re;
        spinorFloat b2_im = +2*i32_im;
        
        // multiply row 0
        spinorFloat A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
        spinorFloat A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
        spinorFloat B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
        spinorFloat B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
        
        // multiply row 1
        spinorFloat A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
        spinorFloat A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
        spinorFloat B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
        spinorFloat B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
        
        // multiply row 2
        spinorFloat A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
        spinorFloat A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
        spinorFloat B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
        spinorFloat B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
        
        o20_re += A0_re;
        o20_im += A0_im;
        o30_re += B0_re;
        o30_im += B0_im;
        
        o21_re += A1_re;
        o21_im += A1_im;
        o31_re += B1_re;
        o31_im += B1_im;
        
        o22_re += A2_re;
        o22_im += A2_im;
        o32_re += B2_re;
        o32_im += B2_im;
      
      } else {
        
        // read gauge matrix from device memory
        READ_GAUGE_MATRIX(GAUGE0TEX, 7);
        
        // read spinor from device memory
        READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);
        
        // reconstruct gauge matrix
        RECONSTRUCT_GAUGE_MATRIX(7);
        
        // project spinor into half spinors
        spinorFloat a0_re = +2*i20_re;
        spinorFloat a0_im = +2*i20_im;
        spinorFloat a1_re = +2*i21_re;
        spinorFloat a1_im = +2*i21_im;
        spinorFloat a2_re = +2*i22_re;
        spinorFloat a2_im = +2*i22_im;
        
        spinorFloat b0_re = +2*i30_re;
        spinorFloat b0_im = +2*i30_im;
        spinorFloat b1_re = +2*i31_re;
        spinorFloat b1_im = +2*i31_im;
        spinorFloat b2_re = +2*i32_re;
        spinorFloat b2_im = +2*i32_im;
        
        // multiply row 0
        spinorFloat A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
        spinorFloat A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
        spinorFloat B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
        spinorFloat B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
        
        // multiply row 1
        spinorFloat A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
        spinorFloat A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
        spinorFloat B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
        spinorFloat B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
        
        // multiply row 2
        spinorFloat A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
        spinorFloat A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
        spinorFloat B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
        spinorFloat B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
        
        o20_re += A0_re;
        o20_im += A0_im;
        o30_re += B0_re;
        o30_im += B0_im;
        
        o21_re += A1_re;
        o21_im += A1_im;
        o31_re += B1_re;
        o31_im += B1_im;
        
        o22_re += A2_re;
        o22_im += A2_im;
        o32_re += B2_re;
        o32_im += B2_im;
      
      }
   }
}




//J  ----------------------------------
//J  --- DWF code for 5th dimension ---
//J  ----------------------------------
//
//J  Begin scope.
{ 
   //J  TODO  Insert/check handler for s-direction here.

   //J  Decided to not change to chiral basis.  Then:
   // 2 P_+ = 2 P_R =  1  1
   //                  1  1 
   // --- Begin right-handed spinor projection. ---
   {
      //J  We are right-handed, so for the dslash_dagger we hop backwards.  If we are at 
      //J  boundary in s-direction, special
      //J  things will need to be done.  xs is defined in dslash_dagger_core_ante.h.
      //J  See near Line 328.  N is the 4d volume; cf. quda.h. 
      //J  Cf. hand-written notes 8/6/09 for check of logic.
      //J  The logic sets xs to the s-coordinate of the output
      //J  spinor, which is accumulated by this thread.
      //J  I.e., it uses the thread index to determine xs.
      int sp_idx = ((xs==0) ? X+(Ls-1)*2*Vh : X-2*Vh) / 2;
      // --- Read spinor from device memory. ---
      //J  Q.  How does it know which direction to hop in?  
      //J  A.  It uses sp_idx as the origin and picks up 0*Vh_5d ... 5*Vh_5d
      //J      offsets in the READ_SPINOR_UP that is below.
      //J      This has to do with the "concurrency" optimization.
      //J  Q.  Where does Vh_5d get set and does it know about the dwf
      //J      modification?  Does it care?
      //J
      //
      READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
      if (xs != 0) {
         //J  OK, now the input spinor should be at:
         //J     0 < s <= Ls-1
         //
         //J  Project spinor into half spinors, i.e., this is the term
         //J     " + 2 P_R psi(s-1) "
                  
         //J  ------------------------------------
         //J  --- Dirac index 0, Colors 0,1,2. ---
         //J  ------------------------------------
         //J  dagger takes P_R instead of P_L
         o00_re += i00_re+i20_re;  //ok
         o00_im += i00_im+i20_im;  //ok
         o01_re += i01_re+i21_re;  //ok  
         o01_im += i01_im+i21_im;  //ok
         o02_re += i02_re+i22_re;  //ok
         o02_im += i02_im+i22_im;  //ok
         
         //J  -------------------------------------
         //J  --- Dirac index 1, Colors 0,1,2.  ---
         //J  -------------------------------------
         o10_re += i10_re+i30_re;  //ok
         o10_im += i10_im+i30_im;  //ok
         o11_re += i11_re+i31_re;  //ok
         o11_im += i11_im+i31_im;  //ok
         o12_re += i12_re+i32_re;  //ok
         o12_im += i12_im+i32_im;  //ok
         
         //J  ------------------------------------
         //J  --- Dirac index 2, Colors 0,1,2. ---
         //J  ------------------------------------
         o20_re += i00_re+i20_re;  //ok
         o20_im += i00_im+i20_im;  //ok
         o21_re += i01_re+i21_re;  //ok
         o21_im += i01_im+i21_im;  //ok
         o22_re += i02_re+i22_re;  //ok
         o22_im += i02_im+i22_im;  //ok
         
         //J  -------------------------------------
         //J  --- Dirac index 3, Colors 0,1,2.  ---
         //J  -------------------------------------
         // color 0 (second index)
         o30_re += i10_re+i30_re;  //ok
         o30_im += i10_im+i30_im;  //ok
         // color 1 (second index)
         o31_re += i11_re+i31_re;  //ok
         o31_im += i11_im+i31_im;  //ok
         // color 2 (second index)
         o32_re += i12_re+i32_re;  //ok
         o32_im += i12_im+i32_im;  //ok

      } // End (x,0) < (x,s) <= (x,Ls-1).
      else {
         //J  LH boundary s=0, backwards hop to Ls-1.
         //J  Term to add:  -mferm*P_R*psi(x,Ls-1)
         //J  With any luck, sp_idx is linear equiv. to "(x,Ls-1)"
         //J  Above, we set:
         //J     sp_idx= (X+(Ls-1)*X4*X3*X2*X1)/2    (*).
         //J  efs:  do some case examples where xs=0 comes out of
         //J  dslash_ante_core.h procedure, and check that sp_idx is
         //J  really coming out correct (and in permissable range)
         //J  in the operation (*).
         //J  We need mferm to get passed.  A modification
         //J  was made to DD_PARAM2 in the C preprocessing file
         //J  dslash_dwf_def.h, adding
         //J  an extra argument to the kernel declarations.
         //
         //J  --- Dirac index 0, Colors 0,1,2.  ---
         // color 0 (second index)
         o00_re += -mferm*(i00_re+i20_re); //ok
         o00_im += -mferm*(i00_im+i20_im); //ok
         // color 1
         o01_re += -mferm*(i01_re+i21_re); //ok
         o01_im += -mferm*(i01_im+i21_im); //ok
         // color 2
         o02_re += -mferm*(i02_re+i22_re); //ok
         o02_im += -mferm*(i02_im+i22_im); //ok

         //J  --- Dirac index 1, Colors 0,1,2.  ---
         // color 0
         o10_re += -mferm*(i10_re+i30_re); //ok
         o10_im += -mferm*(i10_im+i30_im); //ok
         // color 1
         o11_re += -mferm*(i11_re+i31_re); //ok
         o11_im += -mferm*(i11_im+i31_im); //ok
         // color 2
         o12_re += -mferm*(i12_re+i32_re); //ok
         o12_im += -mferm*(i12_im+i32_im); //ok
         
         //J  --- Dirac index 2, Colors 0,1,2.  ---
         // color 0 (second index)
         o20_re += -mferm*(i00_re+i20_re); //ok
         o20_im += -mferm*(i00_im+i20_im); //ok
         // color 1
         o21_re += -mferm*(i01_re+i21_re); //ok
         o21_im += -mferm*(i01_im+i21_im); //ok
         // color 2
         o22_re += -mferm*(i02_re+i22_re); //ok
         o22_im += -mferm*(i02_im+i22_im); //ok

         //J  --- Dirac index 3, Colors 0,1,2.  ---
         // color 0
         o30_re += -mferm*(i10_re+i30_re); //ok
         o30_im += -mferm*(i10_im+i30_im); //ok
         // color 1
         o31_re += -mferm*(i11_re+i31_re); //ok
         o31_im += -mferm*(i11_im+i31_im); //ok
         // color 2
         o32_re += -mferm*(i12_re+i32_re); //ok
         o32_im += -mferm*(i12_im+i32_im); //ok
                  
      }  // End (x,s)=(x,0)
   }  
   // --- End of right-handed spinor projection. ---

   // In the GPU Dirac matrix basis:
   // 2 P_- = 2 P_L =  1 -1
   //                 -1  1 
   //J  Begin scope for 2 P_L projection of forward-hopped spinor.
   {
      //J  For P_L spinor, dslash_dagger, we hop forwards.
      
      //J  This bit mimics what is done for x4==X4-1 in dslash_core_ante.h.
      //J  
      //J  Checked logic w/ case examples.
      //J  Cf. hand-written notes 8/6/09 for check of logic.
      int sp_idx = ((xs==(Ls-1)) ? X-(Ls-1)*2*Vh : X+2*Vh) / 2;
         
      //J  Read spinor from device memory.
      //
      READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);

      // 
      //
      if ( xs < (Ls-1) ) {
         //J  Case of not at RH boundary.   Then we just do += P_L psi(s+1).
         
         //J  ------------------------------------
         //J  --- Dirac index 0, Colors 0,1,2. ---
         //J  ------------------------------------
         // color 0 (second index)
         o00_re += i00_re-i20_re;  //ok
         o00_im += i00_im-i20_im;  //ok
         // color 1 (second index)
         o01_re += i01_re-i21_re;  //ok
         o01_im += i01_im-i21_im;  //ok
         // color 2 (second index)
         o02_re += i02_re-i22_re;  //ok
         o02_im += i02_im-i22_im;  //ok
         
         //J  -------------------------------------
         //J  --- Dirac index 1, Colors 0,1,2.  ---
         //J  -------------------------------------
         // color 0 (second index)
         o10_re += i10_re-i30_re;  //ok
         o10_im += i10_im-i30_im;  //ok
         // color 1 (second index)
         o11_re += i11_re-i31_re;  //ok
         o11_im += i11_im-i31_im;  //ok
         // color 2 (second index)
         o12_re += i12_re-i32_re;  //ok
         o12_im += i12_im-i32_im;  //ok
         
         //J  ------------------------------------
         //J  --- Dirac index 2, Colors 0,1,2. ---
         //J  ------------------------------------
         // color 0 (second index)
         o20_re += -i00_re+i20_re;  //ok
         o20_im += -i00_im+i20_im;  //ok
         // color 1 (second index)
         o21_re += -i01_re+i21_re;  //ok
         o21_im += -i01_im+i21_im;  //ok
         // color 2 (second index)
         o22_re += -i02_re+i22_re;  //ok
         o22_im += -i02_im+i22_im;  //ok
         
         //J  -------------------------------------
         //J  --- Dirac index 3, Colors 0,1,2.  ---
         //J  -------------------------------------
         // color 0 (second index)
         o30_re += -i10_re+i30_re;  //ok
         o30_im += -i10_im+i30_im;  //ok
         // color 1 (second index)
         o31_re += -i11_re+i31_re;  //ok
         o31_im += -i11_im+i31_im;  //ok
         // color 2 (second index)
         o32_re += -i12_re+i32_re;  //ok
         o32_im += -i12_im+i32_im;  //ok

      } // End (x,0) <= (x,s) < (x,Ls-1).
      else {
         //J  RH boundary s=Ls-1, forwards hop to s=0.
         //J  Term to add:  -mferm*P_L*psi(x,0)
         
         //J  --- Dirac index 0, Colors 0,1,2.  ---
         // color 0 (second index)
         o00_re += -mferm*(i00_re-i20_re);  //ok
         o00_im += -mferm*(i00_im-i20_im);  //ok
         // color 1
         o01_re += -mferm*(i01_re-i21_re);  //ok
         o01_im += -mferm*(i01_im-i21_im);  //ok
         // color 2
         o02_re += -mferm*(i02_re-i22_re);  //ok
         o02_im += -mferm*(i02_im-i22_im);  //ok

         //J  --- Dirac index 1, Colors 0,1,2.  ---
         // color 0
         o10_re += -mferm*(i10_re-i30_re);  //ok
         o10_im += -mferm*(i10_im-i30_im);  //ok
         // color 1
         o11_re += -mferm*(i11_re-i31_re);  //ok
         o11_im += -mferm*(i11_im-i31_im);  //ok
         // color 2
         o12_re += -mferm*(i12_re-i32_re);  //ok
         o12_im += -mferm*(i12_im-i32_im);  //ok
         
         //J  --- Dirac index 2, Colors 0,1,2.  ---
         // color 0 (second index)
         o20_re += -mferm*(-i00_re+i20_re);  //ok
         o20_im += -mferm*(-i00_im+i20_im);  //ok
         // color 1
         o21_re += -mferm*(-i01_re+i21_re);  //ok
         o21_im += -mferm*(-i01_im+i21_im);  //ok
         // color 2
         o22_re += -mferm*(-i02_re+i22_re);  //ok
         o22_im += -mferm*(-i02_im+i22_im);  //ok

         //J  --- Dirac index 3, Colors 0,1,2.  ---
         // color 0
         o30_re += -mferm*(-i10_re+i30_re);  //ok
         o30_im += -mferm*(-i10_im+i30_im);  //ok
         // color 1
         o31_re += -mferm*(-i11_re+i31_re);  //ok
         o31_im += -mferm*(-i11_im+i31_im);  //ok
         // color 2
         o32_re += -mferm*(-i12_re+i32_re);  //ok
         o32_im += -mferm*(-i12_im+i32_im);  //ok
         //
      }  // End (x,s)=(x,Ls-1)
   }
   // -----  end dwf s-direction ----
   
}  // end s-direction block


// Perform the DSLASH_XPAY operations.
// Undefine all the macros.  TODO  Make sure that this
// is working right for the diagonal terms of DWF.
//#include "dslash_dagger_core_post.h"


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
#endif // DD_SPREC
#endif // DSLASH_XPAY


    // write spinor field back to device memory
    WRITE_SPINOR(sp_stride);

// undefine to prevent warning when precision is changed
#undef spinorFloat
#undef A_re
#undef A_im

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

