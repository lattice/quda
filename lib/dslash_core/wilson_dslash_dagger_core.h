// *** CUDA DSLASH DAGGER ***

#define SHARED_FLOATS_PER_THREAD 8

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
#ifdef GAUGE_DOUBLE
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
#define o00_re s[0*SHARED_STRIDE]
#define o00_im s[1*SHARED_STRIDE]
#define o01_re s[2*SHARED_STRIDE]
#define o01_im s[3*SHARED_STRIDE]
#define o02_re s[4*SHARED_STRIDE]
#define o02_im s[5*SHARED_STRIDE]
#define o10_re s[6*SHARED_STRIDE]
#define o10_im s[7*SHARED_STRIDE]
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
#include "read_clover.h"
#include "io_spinor.h"

int sid = blockIdx.x*blockDim.x + threadIdx.x;
int z1 = FAST_INT_DIVIDE(sid, X1h);
int x1h = sid - z1*X1h;
int z2 = FAST_INT_DIVIDE(z1, X2);
int x2 = z1 - z2*X2;
int x4 = FAST_INT_DIVIDE(z2, X3);
int x3 = z2 - x4*X3;
int x1odd = (x2 + x3 + x4 + oddBit) & 1;
int x1 = 2*x1h + x1odd;
int X = 2*sid + x1odd;

#ifdef SPINOR_DOUBLE
#define SHARED_STRIDE 8  // to avoid bank conflicts
extern __shared__ spinorFloat sd_data[];
volatile spinorFloat *s = sd_data + SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)
                                  + (threadIdx.x % SHARED_STRIDE);
#else
#define SHARED_STRIDE 16 // to avoid bank conflicts
extern __shared__ spinorFloat ss_data[];
volatile spinorFloat *s = ss_data + SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)
                                  + (threadIdx.x % SHARED_STRIDE);
#endif

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
    
    int sp_idx = ((x1==X1m1) ? X-X1m1 : X+1) >> 1;
    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 0);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX);
    
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

{
    // Projector P0-
    // 1 0 0 -i 
    // 0 1 -i 0 
    // 0 i 1 0 
    // i 0 0 1 
    
    int sp_idx = ((x1==0)    ? X+X1m1 : X-1) >> 1;
    int ga_idx = sp_idx;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 1);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX);
    
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

{
    // Projector P1+
    // 1 0 0 1 
    // 0 1 -1 0 
    // 0 -1 1 0 
    // 1 0 0 1 
    
    int sp_idx = ((x2==X2m1) ? X-X2X1mX1 : X+X1) >> 1;
    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 2);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX);
    
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

{
    // Projector P1-
    // 1 0 0 -1 
    // 0 1 1 0 
    // 0 1 1 0 
    // -1 0 0 1 
    
    int sp_idx = ((x2==0)    ? X+X2X1mX1 : X-X1) >> 1;
    int ga_idx = sp_idx;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 3);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX);
    
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

{
    // Projector P2+
    // 1 0 i 0 
    // 0 1 0 -i 
    // -i 0 1 0 
    // 0 i 0 1 
    
    int sp_idx = ((x3==X3m1) ? X-X3X2X1mX2X1 : X+X2X1) >> 1;
    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 4);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX);
    
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

{
    // Projector P2-
    // 1 0 -i 0 
    // 0 1 0 i 
    // i 0 1 0 
    // 0 -i 0 1 
    
    int sp_idx = ((x3==0)    ? X+X3X2X1mX2X1 : X-X2X1) >> 1;
    int ga_idx = sp_idx;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 5);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX);
    
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

{
    // Projector P3+
    // 2 0 0 0 
    // 0 2 0 0 
    // 0 0 0 0 
    // 0 0 0 0 
    
    int sp_idx = ((x4==X4m1) ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1;
    int ga_idx = sid;
    
    if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h) {
        // read spinor from device memory
        READ_SPINOR_UP(SPINORTEX);
        
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
        
    }
    else {
        // read gauge matrix from device memory
        READ_GAUGE_MATRIX(GAUGE0TEX, 6);
        
        // read spinor from device memory
        READ_SPINOR_UP(SPINORTEX);
        
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

{
    // Projector P3-
    // 0 0 0 0 
    // 0 0 0 0 
    // 0 0 2 0 
    // 0 0 0 2 
    
    int sp_idx = ((x4==0)    ? X+X4X3X2X1mX3X2X1 : X-X3X2X1) >> 1;
    int ga_idx = sp_idx;
    
    if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h) {
        // read spinor from device memory
        READ_SPINOR_DOWN(SPINORTEX);
        
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
        
    }
    else {
        // read gauge matrix from device memory
        READ_GAUGE_MATRIX(GAUGE1TEX, 7);
        
        // read spinor from device memory
        READ_SPINOR_DOWN(SPINORTEX);
        
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

#ifdef DSLASH_CLOVER

// change to chiral basis
{
    spinorFloat a00_re = -o10_re - o30_re;
    spinorFloat a00_im = -o10_im - o30_im;
    spinorFloat a10_re =  o00_re + o20_re;
    spinorFloat a10_im =  o00_im + o20_im;
    spinorFloat a20_re = -o10_re + o30_re;
    spinorFloat a20_im = -o10_im + o30_im;
    spinorFloat a30_re =  o00_re - o20_re;
    spinorFloat a30_im =  o00_im - o20_im;
    
    o00_re = a00_re;  o00_im = a00_im;
    o10_re = a10_re;  o10_im = a10_im;
    o20_re = a20_re;  o20_im = a20_im;
    o30_re = a30_re;  o30_im = a30_im;
}
{
    spinorFloat a01_re = -o11_re - o31_re;
    spinorFloat a01_im = -o11_im - o31_im;
    spinorFloat a11_re =  o01_re + o21_re;
    spinorFloat a11_im =  o01_im + o21_im;
    spinorFloat a21_re = -o11_re + o31_re;
    spinorFloat a21_im = -o11_im + o31_im;
    spinorFloat a31_re =  o01_re - o21_re;
    spinorFloat a31_im =  o01_im - o21_im;
    
    o01_re = a01_re;  o01_im = a01_im;
    o11_re = a11_re;  o11_im = a11_im;
    o21_re = a21_re;  o21_im = a21_im;
    o31_re = a31_re;  o31_im = a31_im;
}
{
    spinorFloat a02_re = -o12_re - o32_re;
    spinorFloat a02_im = -o12_im - o32_im;
    spinorFloat a12_re =  o02_re + o22_re;
    spinorFloat a12_im =  o02_im + o22_im;
    spinorFloat a22_re = -o12_re + o32_re;
    spinorFloat a22_im = -o12_im + o32_im;
    spinorFloat a32_re =  o02_re - o22_re;
    spinorFloat a32_im =  o02_im - o22_im;
    
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
    a00_re += c00_01_re * o01_re - c00_01_im * o01_im;
    a00_im += c00_01_re * o01_im + c00_01_im * o01_re;
    a00_re += c00_02_re * o02_re - c00_02_im * o02_im;
    a00_im += c00_02_re * o02_im + c00_02_im * o02_re;
    a00_re += c00_10_re * o10_re - c00_10_im * o10_im;
    a00_im += c00_10_re * o10_im + c00_10_im * o10_re;
    a00_re += c00_11_re * o11_re - c00_11_im * o11_im;
    a00_im += c00_11_re * o11_im + c00_11_im * o11_re;
    a00_re += c00_12_re * o12_re - c00_12_im * o12_im;
    a00_im += c00_12_re * o12_im + c00_12_im * o12_re;
    
    a01_re += c01_00_re * o00_re - c01_00_im * o00_im;
    a01_im += c01_00_re * o00_im + c01_00_im * o00_re;
    a01_re += c01_01_re * o01_re;
    a01_im += c01_01_re * o01_im;
    a01_re += c01_02_re * o02_re - c01_02_im * o02_im;
    a01_im += c01_02_re * o02_im + c01_02_im * o02_re;
    a01_re += c01_10_re * o10_re - c01_10_im * o10_im;
    a01_im += c01_10_re * o10_im + c01_10_im * o10_re;
    a01_re += c01_11_re * o11_re - c01_11_im * o11_im;
    a01_im += c01_11_re * o11_im + c01_11_im * o11_re;
    a01_re += c01_12_re * o12_re - c01_12_im * o12_im;
    a01_im += c01_12_re * o12_im + c01_12_im * o12_re;
    
    a02_re += c02_00_re * o00_re - c02_00_im * o00_im;
    a02_im += c02_00_re * o00_im + c02_00_im * o00_re;
    a02_re += c02_01_re * o01_re - c02_01_im * o01_im;
    a02_im += c02_01_re * o01_im + c02_01_im * o01_re;
    a02_re += c02_02_re * o02_re;
    a02_im += c02_02_re * o02_im;
    a02_re += c02_10_re * o10_re - c02_10_im * o10_im;
    a02_im += c02_10_re * o10_im + c02_10_im * o10_re;
    a02_re += c02_11_re * o11_re - c02_11_im * o11_im;
    a02_im += c02_11_re * o11_im + c02_11_im * o11_re;
    a02_re += c02_12_re * o12_re - c02_12_im * o12_im;
    a02_im += c02_12_re * o12_im + c02_12_im * o12_re;
    
    a10_re += c10_00_re * o00_re - c10_00_im * o00_im;
    a10_im += c10_00_re * o00_im + c10_00_im * o00_re;
    a10_re += c10_01_re * o01_re - c10_01_im * o01_im;
    a10_im += c10_01_re * o01_im + c10_01_im * o01_re;
    a10_re += c10_02_re * o02_re - c10_02_im * o02_im;
    a10_im += c10_02_re * o02_im + c10_02_im * o02_re;
    a10_re += c10_10_re * o10_re;
    a10_im += c10_10_re * o10_im;
    a10_re += c10_11_re * o11_re - c10_11_im * o11_im;
    a10_im += c10_11_re * o11_im + c10_11_im * o11_re;
    a10_re += c10_12_re * o12_re - c10_12_im * o12_im;
    a10_im += c10_12_re * o12_im + c10_12_im * o12_re;
    
    a11_re += c11_00_re * o00_re - c11_00_im * o00_im;
    a11_im += c11_00_re * o00_im + c11_00_im * o00_re;
    a11_re += c11_01_re * o01_re - c11_01_im * o01_im;
    a11_im += c11_01_re * o01_im + c11_01_im * o01_re;
    a11_re += c11_02_re * o02_re - c11_02_im * o02_im;
    a11_im += c11_02_re * o02_im + c11_02_im * o02_re;
    a11_re += c11_10_re * o10_re - c11_10_im * o10_im;
    a11_im += c11_10_re * o10_im + c11_10_im * o10_re;
    a11_re += c11_11_re * o11_re;
    a11_im += c11_11_re * o11_im;
    a11_re += c11_12_re * o12_re - c11_12_im * o12_im;
    a11_im += c11_12_re * o12_im + c11_12_im * o12_re;
    
    a12_re += c12_00_re * o00_re - c12_00_im * o00_im;
    a12_im += c12_00_re * o00_im + c12_00_im * o00_re;
    a12_re += c12_01_re * o01_re - c12_01_im * o01_im;
    a12_im += c12_01_re * o01_im + c12_01_im * o01_re;
    a12_re += c12_02_re * o02_re - c12_02_im * o02_im;
    a12_im += c12_02_re * o02_im + c12_02_im * o02_re;
    a12_re += c12_10_re * o10_re - c12_10_im * o10_im;
    a12_im += c12_10_re * o10_im + c12_10_im * o10_re;
    a12_re += c12_11_re * o11_re - c12_11_im * o11_im;
    a12_im += c12_11_re * o11_im + c12_11_im * o11_re;
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
    a20_re += c20_21_re * o21_re - c20_21_im * o21_im;
    a20_im += c20_21_re * o21_im + c20_21_im * o21_re;
    a20_re += c20_22_re * o22_re - c20_22_im * o22_im;
    a20_im += c20_22_re * o22_im + c20_22_im * o22_re;
    a20_re += c20_30_re * o30_re - c20_30_im * o30_im;
    a20_im += c20_30_re * o30_im + c20_30_im * o30_re;
    a20_re += c20_31_re * o31_re - c20_31_im * o31_im;
    a20_im += c20_31_re * o31_im + c20_31_im * o31_re;
    a20_re += c20_32_re * o32_re - c20_32_im * o32_im;
    a20_im += c20_32_re * o32_im + c20_32_im * o32_re;
    
    a21_re += c21_20_re * o20_re - c21_20_im * o20_im;
    a21_im += c21_20_re * o20_im + c21_20_im * o20_re;
    a21_re += c21_21_re * o21_re;
    a21_im += c21_21_re * o21_im;
    a21_re += c21_22_re * o22_re - c21_22_im * o22_im;
    a21_im += c21_22_re * o22_im + c21_22_im * o22_re;
    a21_re += c21_30_re * o30_re - c21_30_im * o30_im;
    a21_im += c21_30_re * o30_im + c21_30_im * o30_re;
    a21_re += c21_31_re * o31_re - c21_31_im * o31_im;
    a21_im += c21_31_re * o31_im + c21_31_im * o31_re;
    a21_re += c21_32_re * o32_re - c21_32_im * o32_im;
    a21_im += c21_32_re * o32_im + c21_32_im * o32_re;
    
    a22_re += c22_20_re * o20_re - c22_20_im * o20_im;
    a22_im += c22_20_re * o20_im + c22_20_im * o20_re;
    a22_re += c22_21_re * o21_re - c22_21_im * o21_im;
    a22_im += c22_21_re * o21_im + c22_21_im * o21_re;
    a22_re += c22_22_re * o22_re;
    a22_im += c22_22_re * o22_im;
    a22_re += c22_30_re * o30_re - c22_30_im * o30_im;
    a22_im += c22_30_re * o30_im + c22_30_im * o30_re;
    a22_re += c22_31_re * o31_re - c22_31_im * o31_im;
    a22_im += c22_31_re * o31_im + c22_31_im * o31_re;
    a22_re += c22_32_re * o32_re - c22_32_im * o32_im;
    a22_im += c22_32_re * o32_im + c22_32_im * o32_re;
    
    a30_re += c30_20_re * o20_re - c30_20_im * o20_im;
    a30_im += c30_20_re * o20_im + c30_20_im * o20_re;
    a30_re += c30_21_re * o21_re - c30_21_im * o21_im;
    a30_im += c30_21_re * o21_im + c30_21_im * o21_re;
    a30_re += c30_22_re * o22_re - c30_22_im * o22_im;
    a30_im += c30_22_re * o22_im + c30_22_im * o22_re;
    a30_re += c30_30_re * o30_re;
    a30_im += c30_30_re * o30_im;
    a30_re += c30_31_re * o31_re - c30_31_im * o31_im;
    a30_im += c30_31_re * o31_im + c30_31_im * o31_re;
    a30_re += c30_32_re * o32_re - c30_32_im * o32_im;
    a30_im += c30_32_re * o32_im + c30_32_im * o32_re;
    
    a31_re += c31_20_re * o20_re - c31_20_im * o20_im;
    a31_im += c31_20_re * o20_im + c31_20_im * o20_re;
    a31_re += c31_21_re * o21_re - c31_21_im * o21_im;
    a31_im += c31_21_re * o21_im + c31_21_im * o21_re;
    a31_re += c31_22_re * o22_re - c31_22_im * o22_im;
    a31_im += c31_22_re * o22_im + c31_22_im * o22_re;
    a31_re += c31_30_re * o30_re - c31_30_im * o30_im;
    a31_im += c31_30_re * o30_im + c31_30_im * o30_re;
    a31_re += c31_31_re * o31_re;
    a31_im += c31_31_re * o31_im;
    a31_re += c31_32_re * o32_re - c31_32_im * o32_im;
    a31_im += c31_32_re * o32_im + c31_32_im * o32_re;
    
    a32_re += c32_20_re * o20_re - c32_20_im * o20_im;
    a32_im += c32_20_re * o20_im + c32_20_im * o20_re;
    a32_re += c32_21_re * o21_re - c32_21_im * o21_im;
    a32_im += c32_21_re * o21_im + c32_21_im * o21_re;
    a32_re += c32_22_re * o22_re - c32_22_im * o22_im;
    a32_im += c32_22_re * o22_im + c32_22_im * o22_re;
    a32_re += c32_30_re * o30_re - c32_30_im * o30_im;
    a32_im += c32_30_re * o30_im + c32_30_im * o30_re;
    a32_re += c32_31_re * o31_re - c32_31_im * o31_im;
    a32_im += c32_31_re * o31_im + c32_31_im * o31_re;
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
#endif // DSLASH_CLOVER


#ifdef DSLASH_XPAY
    READ_ACCUM(ACCUMTEX)
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


    // write spinor field back to device memory
    WRITE_SPINOR();

// undefine to prevent warning when precision is changed
#undef spinorFloat
#undef SHARED_STRIDE

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
#undef o00_re
#undef o00_im
#undef o01_re
#undef o01_im
#undef o02_re
#undef o02_im
#undef o10_re
#undef o10_im

