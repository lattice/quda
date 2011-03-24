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
if (sid >= param.threads) return;
int z1 = FAST_INT_DIVIDE(sid, X1h);
int x1h = sid - z1*X1h;
int z2 = FAST_INT_DIVIDE(z1, X2);
int x2 = z1 - z2*X2;
int x4 = FAST_INT_DIVIDE(z2, X3);
int x3 = z2 - x4*X3;

#ifdef MULTI_GPU
// now calculate the new x4 given the T offset and space between T slices
int x4_new = (x4 + param.tOffset) * param.tMul;
sid += Vs*(x4_new - x4); // new spatial index
x4 = x4_new;
#endif

int x1odd = (x2 + x3 + x4 + param.parity) & 1;
int x1 = 2*x1h + x1odd;
int X = 2*sid + x1odd;

#ifdef SPINOR_DOUBLE
#if (__CUDA_ARCH__ >= 200)
#define SHARED_STRIDE 16 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE  8 // to avoid bank conflicts on G80 and GT200
#endif
extern __shared__ spinorFloat sd_data[];
volatile spinorFloat *s = sd_data + SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)
                                  + (threadIdx.x % SHARED_STRIDE);
#else
#if (__CUDA_ARCH__ >= 200)
#define SHARED_STRIDE 32 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE 16 // to avoid bank conflicts on G80 and GT200
#endif
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
    READ_GAUGE_MATRIX(G, GAUGE0TEX, 0, ga_idx, ga_stride);
    
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
    READ_GAUGE_MATRIX(G, GAUGE1TEX, 1, ga_idx, ga_stride);
    
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
    READ_GAUGE_MATRIX(G, GAUGE0TEX, 2, ga_idx, ga_stride);
    
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
    READ_GAUGE_MATRIX(G, GAUGE1TEX, 3, ga_idx, ga_stride);
    
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
    READ_GAUGE_MATRIX(G, GAUGE0TEX, 4, ga_idx, ga_stride);
    
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
    READ_GAUGE_MATRIX(G, GAUGE1TEX, 5, ga_idx, ga_stride);
    
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
    
    #ifndef MULTI_GPU
        int sp_idx = ((x4==X4m1) ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1;
    #define sp_stride_t sp_stride
    #define sp_norm_idx sp_idx
    #else
        int sp_idx;
        int sp_stride_t;
    #if (DD_PREC==2)
        int sp_norm_idx;
    #else
    #define sp_norm_idx sp_idx
    #endif
        if (x4 == X4m1) { // front face (lower spin components)
          sp_stride_t = Vs;
          sp_idx = sid - (Vh - Vs) + SPINOR_HOP*sp_stride; // starts at Npad*Vs (precalculate more)
    #if (DD_PREC==2)
          sp_norm_idx = sid - (Vh - Vs) + sp_stride;
    #endif
        } else {
          sp_stride_t = sp_stride;
          sp_idx = (X+X3X2X1) >> 1;
    #if (DD_PREC==2)
          sp_norm_idx = sp_idx;
    #endif
        }
    #endif // MULTI_GPU
    
    int ga_idx = sid;
    
    if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h) {
        // read spinor from device memory
        READ_SPINOR_UP(SPINORTEX, sp_stride_t, sp_idx, sp_norm_idx);
        
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
        // read gauge matrix from device memory
        READ_GAUGE_MATRIX(G, GAUGE0TEX, 6, ga_idx, ga_stride);
        
        // read spinor from device memory
        READ_SPINOR_UP(SPINORTEX, sp_stride_t, sp_idx, sp_norm_idx);
        
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
    
    #ifndef MULTI_GPU
        int sp_idx = ((x4==0)    ? X+X4X3X2X1mX3X2X1 : X-X3X2X1) >> 1;
    #define sp_stride_t sp_stride
    #define sp_norm_idx sp_idx
        int ga_idx = sp_idx;
    #else
        int sp_idx;
        int sp_stride_t;
    #if (DD_PREC==2)
        int sp_norm_idx;
    #else
    #define sp_norm_idx sp_idx
    #endif
        if (x4 == 0) { // back face
          sp_stride_t = Vs;
          sp_idx = sid + SPINOR_HOP*sp_stride;
    #if (DD_PREC==2)
          sp_norm_idx = sid + sp_stride + Vs; // need extra Vs addition since we require the lower norm buffer
    #endif
        } else {
          sp_stride_t = sp_stride;
          sp_idx = (X - X3X2X1) >> 1;
    #if (DD_PREC==2)
          sp_norm_idx = sp_idx;
    #endif
        }
        // back links in pad, which is offset by Vh+sid from buffer start
        int ga_idx = (x4==0) ? sid+Vh : sp_idx;
    #endif // MULTI_GPU
    
    if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h) {
        // read spinor from device memory
        READ_SPINOR_DOWN(SPINORTEX, sp_stride_t, sp_idx, sp_norm_idx);
        
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
        // read gauge matrix from device memory
        READ_GAUGE_MATRIX(G, GAUGE1TEX, 7, ga_idx, ga_stride);
        
        // read spinor from device memory
        READ_SPINOR_DOWN(SPINORTEX, sp_stride_t, sp_idx, sp_norm_idx);
        
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

{
    // apply twisted mass rotation
    volatile spinorFloat tmp00_re = +o00_re-o20_im*a;
    volatile spinorFloat tmp00_im = +o00_im+o20_re*a;
    volatile spinorFloat tmp01_re = +o01_re-o21_im*a;
    volatile spinorFloat tmp01_im = +o01_im+o21_re*a;
    volatile spinorFloat tmp02_re = +o02_re-o22_im*a;
    volatile spinorFloat tmp02_im = +o02_im+o22_re*a;
    
    volatile spinorFloat tmp10_re = +o10_re-o30_im*a;
    volatile spinorFloat tmp10_im = +o10_im+o30_re*a;
    volatile spinorFloat tmp11_re = +o11_re-o31_im*a;
    volatile spinorFloat tmp11_im = +o11_im+o31_re*a;
    volatile spinorFloat tmp12_re = +o12_re-o32_im*a;
    volatile spinorFloat tmp12_im = +o12_im+o32_re*a;
    
    volatile spinorFloat tmp20_re = -o00_im*a+o20_re;
    volatile spinorFloat tmp20_im = +o00_re*a+o20_im;
    volatile spinorFloat tmp21_re = -o01_im*a+o21_re;
    volatile spinorFloat tmp21_im = +o01_re*a+o21_im;
    volatile spinorFloat tmp22_re = -o02_im*a+o22_re;
    volatile spinorFloat tmp22_im = +o02_re*a+o22_im;
    
    volatile spinorFloat tmp30_re = -o10_im*a+o30_re;
    volatile spinorFloat tmp30_im = +o10_re*a+o30_im;
    volatile spinorFloat tmp31_re = -o11_im*a+o31_re;
    volatile spinorFloat tmp31_im = +o11_re*a+o31_im;
    volatile spinorFloat tmp32_re = -o12_im*a+o32_re;
    volatile spinorFloat tmp32_im = +o12_re*a+o32_im;
    
    
    #ifndef DSLASH_XPAY
    //scale by b = 1/(1 + a*a) 
    o00_re = b*tmp00_re;
    o00_im = b*tmp00_im;
    o01_re = b*tmp01_re;
    o01_im = b*tmp01_im;
    o02_re = b*tmp02_re;
    o02_im = b*tmp02_im;
    o10_re = b*tmp10_re;
    o10_im = b*tmp10_im;
    o11_re = b*tmp11_re;
    o11_im = b*tmp11_im;
    o12_re = b*tmp12_re;
    o12_im = b*tmp12_im;
    o20_re = b*tmp20_re;
    o20_im = b*tmp20_im;
    o21_re = b*tmp21_re;
    o21_im = b*tmp21_im;
    o22_re = b*tmp22_re;
    o22_im = b*tmp22_im;
    o30_re = b*tmp30_re;
    o30_im = b*tmp30_im;
    o31_re = b*tmp31_re;
    o31_im = b*tmp31_im;
    o32_re = b*tmp32_re;
    o32_im = b*tmp32_im;
    #else
    o00_re = tmp00_re;
    o00_im = tmp00_im;
    o01_re = tmp01_re;
    o01_im = tmp01_im;
    o02_re = tmp02_re;
    o02_im = tmp02_im;
    o10_re = tmp10_re;
    o10_im = tmp10_im;
    o11_re = tmp11_re;
    o11_im = tmp11_im;
    o12_re = tmp12_re;
    o12_im = tmp12_im;
    o20_re = tmp20_re;
    o20_im = tmp20_im;
    o21_re = tmp21_re;
    o21_im = tmp21_im;
    o22_re = tmp22_re;
    o22_im = tmp22_im;
    o30_re = tmp30_re;
    o30_im = tmp30_im;
    o31_re = tmp31_re;
    o31_im = tmp31_im;
    o32_re = tmp32_re;
    o32_im = tmp32_im;
    #endif // DSLASH_XPAY
    
}

#ifdef DSLASH_XPAY
    READ_ACCUM(ACCUMTEX, sp_stride)
#ifdef SPINOR_DOUBLE
    o00_re = b*o00_re + accum0.x;
    o00_im = b*o00_im + accum0.y;
    o01_re = b*o01_re + accum1.x;
    o01_im = b*o01_im + accum1.y;
    o02_re = b*o02_re + accum2.x;
    o02_im = b*o02_im + accum2.y;
    o10_re = b*o10_re + accum3.x;
    o10_im = b*o10_im + accum3.y;
    o11_re = b*o11_re + accum4.x;
    o11_im = b*o11_im + accum4.y;
    o12_re = b*o12_re + accum5.x;
    o12_im = b*o12_im + accum5.y;
    o20_re = b*o20_re + accum6.x;
    o20_im = b*o20_im + accum6.y;
    o21_re = b*o21_re + accum7.x;
    o21_im = b*o21_im + accum7.y;
    o22_re = b*o22_re + accum8.x;
    o22_im = b*o22_im + accum8.y;
    o30_re = b*o30_re + accum9.x;
    o30_im = b*o30_im + accum9.y;
    o31_re = b*o31_re + accum10.x;
    o31_im = b*o31_im + accum10.y;
    o32_re = b*o32_re + accum11.x;
    o32_im = b*o32_im + accum11.y;
#else
    o00_re = b*o00_re + accum0.x;
    o00_im = b*o00_im + accum0.y;
    o01_re = b*o01_re + accum0.z;
    o01_im = b*o01_im + accum0.w;
    o02_re = b*o02_re + accum1.x;
    o02_im = b*o02_im + accum1.y;
    o10_re = b*o10_re + accum1.z;
    o10_im = b*o10_im + accum1.w;
    o11_re = b*o11_re + accum2.x;
    o11_im = b*o11_im + accum2.y;
    o12_re = b*o12_re + accum2.z;
    o12_im = b*o12_im + accum2.w;
    o20_re = b*o20_re + accum3.x;
    o20_im = b*o20_im + accum3.y;
    o21_re = b*o21_re + accum3.z;
    o21_im = b*o21_im + accum3.w;
    o22_re = b*o22_re + accum4.x;
    o22_im = b*o22_im + accum4.y;
    o30_re = b*o30_re + accum4.z;
    o30_im = b*o30_im + accum4.w;
    o31_re = b*o31_re + accum5.x;
    o31_im = b*o31_im + accum5.y;
    o32_re = b*o32_re + accum5.z;
    o32_im = b*o32_im + accum5.w;
#endif // SPINOR_DOUBLE
#endif // DSLASH_XPAY


    // write spinor field back to device memory
    WRITE_SPINOR(sp_stride);

// undefine to prevent warning when precision is changed
#undef spinorFloat
#undef SHARED_STRIDE

#undef A_re
#undef A_im

#ifdef sp_norm_idx
#undef sp_norm_idx
#endif
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

#undef o00_re
#undef o00_im
#undef o01_re
#undef o01_im
#undef o02_re
#undef o02_im
#undef o10_re
#undef o10_im

