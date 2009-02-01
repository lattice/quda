// *** CUDA DSLASH DAGGER ***

#define SHARED_FLOATS_PER_THREAD 19
#define SHARED_BYTES (BLOCK_DIM*SHARED_FLOATS_PER_THREAD*sizeof(float))

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

#define A_re G4.z
#define A_im G4.w

#define o00_re s[0]
#define o00_im s[1]
#define o01_re s[2]
#define o01_im s[3]
#define o02_re s[4]
#define o02_im s[5]
#define o10_re s[6]
#define o10_im s[7]
#define o11_re s[8]
#define o11_im s[9]
#define o12_re s[10]
#define o12_im s[11]
#define o20_re s[12]
#define o20_im s[13]
#define o21_re s[14]
#define o21_im s[15]
#define o22_re s[16]
#define o22_im s[17]
#define o30_re s[18]
volatile float o30_im;
volatile float o31_re;
volatile float o31_im;
volatile float o32_re;
volatile float o32_im;



#include "read_gauge.h"
#include "io_spinor.h"

int sid = BLOCK_DIM*blockIdx.x + threadIdx.x;
int boundaryCrossings = sid/L1h + sid/(L2*L1h) + sid/(L3*L2*L1h);
int X = 2*sid + (boundaryCrossings + oddBit) % 2;
int x4 = X/(L3*L2*L1);
int x3 = (X/(L2*L1)) % L3;
int x2 = (X/L1) % L2;
int x1 = X % L1;

extern __shared__ float s_data[];
volatile float *s = s_data+SHARED_FLOATS_PER_THREAD*threadIdx.x;

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
    
    int sp_idx = ((x1==L1-1) ? X-(L1-1) : X+1) / 2;
    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 0);
    
    // read spinor from device memory
    READ_SPINOR(spinorTex);
    
    // project spinor into half spinors
    float a0_re = +i00_re-i30_im;
    float a0_im = +i00_im+i30_re;
    float a1_re = +i01_re-i31_im;
    float a1_im = +i01_im+i31_re;
    float a2_re = +i02_re-i32_im;
    float a2_im = +i02_im+i32_re;
    
    float b0_re = +i10_re-i20_im;
    float b0_im = +i10_im+i20_re;
    float b1_re = +i11_re-i21_im;
    float b1_im = +i11_im+i21_re;
    float b2_re = +i12_re-i22_im;
    float b2_im = +i12_im+i22_re;
    
    // multiply row 0
    float A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
    float A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
    float B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
    float B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
    
    // multiply row 1
    float A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
    float A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
    float B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
    float B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
    
    // multiply row 2
    float A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
    float A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
    float B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
    float B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
    
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
    
    int sp_idx = ((x1==0)    ? X+(L1-1) : X-1) / 2;
    int ga_idx = sp_idx;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 1);
    
    // read spinor from device memory
    READ_SPINOR(spinorTex);
    
    // project spinor into half spinors
    float a0_re = +i00_re+i30_im;
    float a0_im = +i00_im-i30_re;
    float a1_re = +i01_re+i31_im;
    float a1_im = +i01_im-i31_re;
    float a2_re = +i02_re+i32_im;
    float a2_im = +i02_im-i32_re;
    
    float b0_re = +i10_re+i20_im;
    float b0_im = +i10_im-i20_re;
    float b1_re = +i11_re+i21_im;
    float b1_im = +i11_im-i21_re;
    float b2_re = +i12_re+i22_im;
    float b2_im = +i12_im-i22_re;
    
    // multiply row 0
    float A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
    float A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
    float B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
    float B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
    
    // multiply row 1
    float A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
    float A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
    float B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
    float B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
    
    // multiply row 2
    float A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
    float A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
    float B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
    float B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
    
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
    
    int sp_idx = ((x2==L2-1) ? X-(L2-1)*L1 : X+L1) / 2;
    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 2);
    
    // read spinor from device memory
    READ_SPINOR(spinorTex);
    
    // project spinor into half spinors
    float a0_re = +i00_re+i30_re;
    float a0_im = +i00_im+i30_im;
    float a1_re = +i01_re+i31_re;
    float a1_im = +i01_im+i31_im;
    float a2_re = +i02_re+i32_re;
    float a2_im = +i02_im+i32_im;
    
    float b0_re = +i10_re-i20_re;
    float b0_im = +i10_im-i20_im;
    float b1_re = +i11_re-i21_re;
    float b1_im = +i11_im-i21_im;
    float b2_re = +i12_re-i22_re;
    float b2_im = +i12_im-i22_im;
    
    // multiply row 0
    float A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
    float A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
    float B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
    float B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
    
    // multiply row 1
    float A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
    float A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
    float B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
    float B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
    
    // multiply row 2
    float A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
    float A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
    float B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
    float B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
    
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
    
    int sp_idx = ((x2==0)    ? X+(L2-1)*L1 : X-L1) / 2;
    int ga_idx = sp_idx;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 3);
    
    // read spinor from device memory
    READ_SPINOR(spinorTex);
    
    // project spinor into half spinors
    float a0_re = +i00_re-i30_re;
    float a0_im = +i00_im-i30_im;
    float a1_re = +i01_re-i31_re;
    float a1_im = +i01_im-i31_im;
    float a2_re = +i02_re-i32_re;
    float a2_im = +i02_im-i32_im;
    
    float b0_re = +i10_re+i20_re;
    float b0_im = +i10_im+i20_im;
    float b1_re = +i11_re+i21_re;
    float b1_im = +i11_im+i21_im;
    float b2_re = +i12_re+i22_re;
    float b2_im = +i12_im+i22_im;
    
    // multiply row 0
    float A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
    float A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
    float B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
    float B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
    
    // multiply row 1
    float A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
    float A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
    float B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
    float B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
    
    // multiply row 2
    float A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
    float A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
    float B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
    float B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
    
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
    
    int sp_idx = ((x3==L3-1) ? X-(L3-1)*L2*L1 : X+L2*L1) / 2;
    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 4);
    
    // read spinor from device memory
    READ_SPINOR(spinorTex);
    
    // project spinor into half spinors
    float a0_re = +i00_re-i20_im;
    float a0_im = +i00_im+i20_re;
    float a1_re = +i01_re-i21_im;
    float a1_im = +i01_im+i21_re;
    float a2_re = +i02_re-i22_im;
    float a2_im = +i02_im+i22_re;
    
    float b0_re = +i10_re+i30_im;
    float b0_im = +i10_im-i30_re;
    float b1_re = +i11_re+i31_im;
    float b1_im = +i11_im-i31_re;
    float b2_re = +i12_re+i32_im;
    float b2_im = +i12_im-i32_re;
    
    // multiply row 0
    float A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
    float A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
    float B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
    float B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
    
    // multiply row 1
    float A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
    float A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
    float B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
    float B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
    
    // multiply row 2
    float A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
    float A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
    float B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
    float B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
    
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
    
    int sp_idx = ((x3==0)    ? X+(L3-1)*L2*L1 : X-L2*L1) / 2;
    int ga_idx = sp_idx;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 5);
    
    // read spinor from device memory
    READ_SPINOR(spinorTex);
    
    // project spinor into half spinors
    float a0_re = +i00_re+i20_im;
    float a0_im = +i00_im-i20_re;
    float a1_re = +i01_re+i21_im;
    float a1_im = +i01_im-i21_re;
    float a2_re = +i02_re+i22_im;
    float a2_im = +i02_im-i22_re;
    
    float b0_re = +i10_re-i30_im;
    float b0_im = +i10_im+i30_re;
    float b1_re = +i11_re-i31_im;
    float b1_im = +i11_im+i31_re;
    float b2_re = +i12_re-i32_im;
    float b2_im = +i12_im+i32_re;
    
    // multiply row 0
    float A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
    float A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
    float B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
    float B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
    
    // multiply row 1
    float A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
    float A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
    float B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
    float B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
    
    // multiply row 2
    float A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
    float A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
    float B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
    float B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
    
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
    
    int sp_idx = ((x4==L4-1) ? X-(L4-1)*L3*L2*L1 : X+L3*L2*L1) / 2;
    int ga_idx = sid;
    
    if (gauge_fixed && ga_idx < (L4-1)*L1h*L2*L3) {
        // read spinor from device memory
        READ_SPINOR_UP(spinorTex);
        
        // project spinor into half spinors
        float a0_re = +2*i00_re;
        float a0_im = +2*i00_im;
        float a1_re = +2*i01_re;
        float a1_im = +2*i01_im;
        float a2_re = +2*i02_re;
        float a2_im = +2*i02_im;
        
        float b0_re = +2*i10_re;
        float b0_im = +2*i10_im;
        float b1_re = +2*i11_re;
        float b1_im = +2*i11_im;
        float b2_re = +2*i12_re;
        float b2_im = +2*i12_im;
        
        // identity gauge matrix
        float A0_re = a0_re; float A0_im = a0_im;
        float B0_re = b0_re; float B0_im = b0_im;
        float A1_re = a1_re; float A1_im = a1_im;
        float B1_re = b1_re; float B1_im = b1_im;
        float A2_re = a2_re; float A2_im = a2_im;
        float B2_re = b2_re; float B2_im = b2_im;
        
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
        READ_SPINOR_UP(spinorTex);
        
        // project spinor into half spinors
        float a0_re = +2*i00_re;
        float a0_im = +2*i00_im;
        float a1_re = +2*i01_re;
        float a1_im = +2*i01_im;
        float a2_re = +2*i02_re;
        float a2_im = +2*i02_im;
        
        float b0_re = +2*i10_re;
        float b0_im = +2*i10_im;
        float b1_re = +2*i11_re;
        float b1_im = +2*i11_im;
        float b2_re = +2*i12_re;
        float b2_im = +2*i12_im;
        
        // multiply row 0
        float A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
        float A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
        float B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
        float B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
        
        // multiply row 1
        float A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
        float A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
        float B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
        float B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
        
        // multiply row 2
        float A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
        float A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
        float B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
        float B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
        
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
    
    int sp_idx = ((x4==0)    ? X+(L4-1)*L3*L2*L1 : X-L3*L2*L1) / 2;
    int ga_idx = sp_idx;
    
    if (gauge_fixed && ga_idx < (L4-1)*L1h*L2*L3) {
        // read spinor from device memory
        READ_SPINOR_DOWN(spinorTex);
        
        // project spinor into half spinors
        float a0_re = +2*i20_re;
        float a0_im = +2*i20_im;
        float a1_re = +2*i21_re;
        float a1_im = +2*i21_im;
        float a2_re = +2*i22_re;
        float a2_im = +2*i22_im;
        
        float b0_re = +2*i30_re;
        float b0_im = +2*i30_im;
        float b1_re = +2*i31_re;
        float b1_im = +2*i31_im;
        float b2_re = +2*i32_re;
        float b2_im = +2*i32_im;
        
        // identity gauge matrix
        float A0_re = a0_re; float A0_im = a0_im;
        float B0_re = b0_re; float B0_im = b0_im;
        float A1_re = a1_re; float A1_im = a1_im;
        float B1_re = b1_re; float B1_im = b1_im;
        float A2_re = a2_re; float A2_im = a2_im;
        float B2_re = b2_re; float B2_im = b2_im;
        
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
        READ_SPINOR_DOWN(spinorTex);
        
        // project spinor into half spinors
        float a0_re = +2*i20_re;
        float a0_im = +2*i20_im;
        float a1_re = +2*i21_re;
        float a1_im = +2*i21_im;
        float a2_re = +2*i22_re;
        float a2_im = +2*i22_im;
        
        float b0_re = +2*i30_re;
        float b0_im = +2*i30_im;
        float b1_re = +2*i31_re;
        float b1_im = +2*i31_im;
        float b2_re = +2*i32_re;
        float b2_im = +2*i32_im;
        
        // multiply row 0
        float A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
        float A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
        float B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
        float B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
        
        // multiply row 1
        float A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
        float A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
        float B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
        float B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
        
        // multiply row 2
        float A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
        float A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
        float B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
        float B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
        
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


#ifdef DSLASH_XPAY
    float4 accum0 = tex1Dfetch(accumTex, sid + 0*Nh);
    float4 accum1 = tex1Dfetch(accumTex, sid + 1*Nh);
    float4 accum2 = tex1Dfetch(accumTex, sid + 2*Nh);
    float4 accum3 = tex1Dfetch(accumTex, sid + 3*Nh);
    float4 accum4 = tex1Dfetch(accumTex, sid + 4*Nh);
    float4 accum5 = tex1Dfetch(accumTex, sid + 5*Nh);
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
#endif


    // write spinor field back to device memory
    WRITE_SPINOR();

#undef o00_re
#undef o00_im
#undef o01_re
#undef o01_im
#undef o02_re
#undef o02_im
#undef o10_re
#undef o10_im
#undef o11_re
#undef o11_im
#undef o12_re
#undef o12_im
#undef o20_re
#undef o20_im
#undef o21_re
#undef o21_im
#undef o22_re
#undef o22_im
#undef o30_re
