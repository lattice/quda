// *** CUDA DSLASH ***

#define SHARED_FLOATS_PER_THREAD 0
// input spinor
#if (DD_PREC==0)
#define spinorFloat double
#define i00_re I0.x
#define i00_im I0.y
#define i01_re I1.x
#define i01_im I1.y
#define i02_re I2.x
#define i02_im I2.y

#define t00_re T0.x
#define t00_im T0.y
#define t01_re T1.x
#define t01_im T1.y
#define t02_re T2.x
#define t02_im T2.y

#else

#define spinorFloat float
#define i00_re I0.x
#define i00_im I0.y
#define i01_re I1.x
#define i01_im I1.y
#define i02_re I2.x
#define i02_im I2.y

#define t00_re T0.x
#define t00_im T0.y
#define t01_re T1.x
#define t01_im T1.y
#define t02_re T2.x
#define t02_im T2.y

#endif

// gauge link
#if (DD_PREC==0)

#define fat00_re FAT0.x
#define fat00_im FAT0.y
#define fat01_re FAT1.x
#define fat01_im FAT1.y
#define fat02_re FAT2.x
#define fat02_im FAT2.y
#define fat10_re FAT3.x
#define fat10_im FAT3.y
#define fat11_re FAT4.x
#define fat11_im FAT4.y
#define fat12_re FAT5.x
#define fat12_im FAT5.y
#define fat20_re FAT6.x
#define fat20_im FAT6.y
#define fat21_re FAT7.x
#define fat21_im FAT7.y
#define fat22_re FAT8.x
#define fat22_im FAT8.y

#define long00_re LONG0.x
#define long00_im LONG0.y
#define long01_re LONG1.x
#define long01_im LONG1.y
#define long02_re LONG2.x
#define long02_im LONG2.y
#define long10_re LONG3.x
#define long10_im LONG3.y
#define long11_re LONG4.x
#define long11_im LONG4.y
#define long12_re LONG5.x
#define long12_im LONG5.y
#define long20_re LONG6.x
#define long20_im LONG6.y
#define long21_re LONG7.x
#define long21_im LONG7.y
#define long22_re LONG8.x
#define long22_im LONG8.y

#define A_re LONG9.x
#define A_im LONG9.y

#else


#define fat00_re FAT0.x
#define fat00_im FAT0.y
#define fat01_re FAT1.x
#define fat01_im FAT1.y
#define fat02_re FAT2.x
#define fat02_im FAT2.y
#define fat10_re FAT3.x
#define fat10_im FAT3.y
#define fat11_re FAT4.x
#define fat11_im FAT4.y
#define fat12_re FAT5.x
#define fat12_im FAT5.y
#define fat20_re FAT6.x
#define fat20_im FAT6.y
#define fat21_re FAT7.x
#define fat21_im FAT7.y
#define fat22_re FAT8.x
#define fat22_im FAT8.y

#if (DD_RECON == 2) //18 (no) reconstruct
#define long00_re LONG0.x
#define long00_im LONG0.y
#define long01_re LONG1.x
#define long01_im LONG1.y
#define long02_re LONG2.x
#define long02_im LONG2.y
#define long10_re LONG3.x
#define long10_im LONG3.y
#define long11_re LONG4.x
#define long11_im LONG4.y
#define long12_re LONG5.x
#define long12_im LONG5.y
#define long20_re LONG6.x
#define long20_im LONG6.y
#define long21_re LONG7.x
#define long21_im LONG7.y
#define long22_re LONG8.x
#define long22_im LONG8.y
#else
#define long00_re LONG0.x
#define long00_im LONG0.y
#define long01_re LONG0.z
#define long01_im LONG0.w
#define long02_re LONG1.x
#define long02_im LONG1.y
#define long10_re LONG1.z
#define long10_im LONG1.w
#define long11_re LONG2.x
#define long11_im LONG2.y
#define long12_re LONG2.z
#define long12_im LONG2.w
#define long20_re LONG3.x
#define long20_im LONG3.y
#define long21_re LONG3.z
#define long21_im LONG3.w
#define long22_re LONG4.x
#define long22_im LONG4.y
#define A_re LONG4.z
#define A_im LONG4.w
#endif

#endif

// conjugated gauge link
#define fatT00_re (+fat00_re)
#define fatT00_im (-fat00_im)
#define fatT01_re (+fat10_re)
#define fatT01_im (-fat10_im)
#define fatT02_re (+fat20_re)
#define fatT02_im (-fat20_im)
#define fatT10_re (+fat01_re)
#define fatT10_im (-fat01_im)
#define fatT11_re (+fat11_re)
#define fatT11_im (-fat11_im)
#define fatT12_re (+fat21_re)
#define fatT12_im (-fat21_im)
#define fatT20_re (+fat02_re)
#define fatT20_im (-fat02_im)
#define fatT21_re (+fat12_re)
#define fatT21_im (-fat12_im)
#define fatT22_re (+fat22_re)
#define fatT22_im (-fat22_im)

#define longT00_re (+long00_re)
#define longT00_im (-long00_im)
#define longT01_re (+long10_re)
#define longT01_im (-long10_im)
#define longT02_re (+long20_re)
#define longT02_im (-long20_im)
#define longT10_re (+long01_re)
#define longT10_im (-long01_im)
#define longT11_re (+long11_re)
#define longT11_im (-long11_im)
#define longT12_re (+long21_re)
#define longT12_im (-long21_im)
#define longT20_re (+long02_re)
#define longT20_im (-long02_im)
#define longT21_re (+long12_re)
#define longT21_im (-long12_im)
#define longT22_re (+long22_re)
#define longT22_im (-long22_im)

// output spinor
volatile spinorFloat o00_re;
volatile spinorFloat o00_im;
volatile spinorFloat o01_re;
volatile spinorFloat o01_im;
volatile spinorFloat o02_re;
volatile spinorFloat o02_im;


#include "read_gauge.h"
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

#if (DD_RECON_F != 18)
int sign;
#endif

o00_re = o00_im = 0.f;
o01_re = o01_im = 0.f;
o02_re = o02_im = 0.f;

#define MAT_MUL_V(VOUT, M, V)						\
    spinorFloat VOUT##0_re = (M##00_re * V##00_re - M##00_im * V##00_im) + (M##01_re * V##01_re - M##01_im * V##01_im) + (M##02_re * V##02_re - M##02_im * V##02_im); \
    spinorFloat VOUT##0_im = (M##00_re * V##00_im + M##00_im * V##00_re) + (M##01_re * V##01_im + M##01_im * V##01_re) + (M##02_re * V##02_im + M##02_im * V##02_re); \
    spinorFloat VOUT##1_re = (M##10_re * V##00_re - M##10_im * V##00_im) + (M##11_re * V##01_re - M##11_im * V##01_im) + (M##12_re * V##02_re - M##12_im * V##02_im); \
    spinorFloat VOUT##1_im = (M##10_re * V##00_im + M##10_im * V##00_re) + (M##11_re * V##01_im + M##11_im * V##01_re) + (M##12_re * V##02_im + M##12_im * V##02_re); \
    spinorFloat VOUT##2_re = (M##20_re * V##00_re - M##20_im * V##00_im) + (M##21_re * V##01_re - M##21_im * V##01_im) + (M##22_re * V##02_re - M##22_im * V##02_im); \
    spinorFloat VOUT##2_im = (M##20_re * V##00_im + M##20_im * V##00_re) + (M##21_re * V##01_im + M##21_im * V##01_re) + (M##22_re * V##02_im + M##22_im * V##02_re);

#define ADJ_MAT_MUL_V(VOUT, M, V)						\
    spinorFloat VOUT##0_re = (M##00_re * V##00_re + M##00_im * V##00_im) + (M##10_re * V##01_re + M##10_im * V##01_im) + (M##20_re * V##02_re + M##20_im * V##02_im); \
    spinorFloat VOUT##0_im = (M##00_re * V##00_im - M##00_im * V##00_re) + (M##10_re * V##01_im - M##10_im * V##01_re) + (M##20_re * V##02_im - M##20_im * V##02_re); \
    spinorFloat VOUT##1_re = (M##01_re * V##00_re + M##01_im * V##00_im) + (M##11_re * V##01_re + M##11_im * V##01_im) + (M##21_re * V##02_re + M##21_im * V##02_im); \
    spinorFloat VOUT##1_im = (M##01_re * V##00_im - M##01_im * V##00_re) + (M##11_re * V##01_im - M##11_im * V##01_re) + (M##21_re * V##02_im - M##21_im * V##02_re); \
    spinorFloat VOUT##2_re = (M##02_re * V##00_re + M##02_im * V##00_im) + (M##12_re * V##01_re + M##12_im * V##01_im) + (M##22_re * V##02_re + M##22_im * V##02_im); \
    spinorFloat VOUT##2_im = (M##02_re * V##00_im - M##02_im * V##00_re) + (M##12_re * V##01_im - M##12_im * V##01_re) + (M##22_re * V##02_im - M##22_im * V##02_re);



{
    //direction: +X
#if (DD_RECON_F != 18)
    if(x4%2 ==1){
	sign = -1;
    }else{
	sign =1;
    }
#endif
    int sp_idx_1st_nbr = ((x1==X1m1) ? X-X1m1 : X+1) >> 1;
    int sp_idx_3rd_nbr = ((x1 > (X1 -4)) ? X -X1 +3 : X+3) >> 1;

    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_FAT_MATRIX(FATLINK0TEX, 0, ga_idx);
    READ_LONG_MATRIX(LONGLINK0TEX, 0, ga_idx);
    
    // read spinor from device memory
    READ_1ST_NBR_SPINOR(SPINORTEX, sp_idx_1st_nbr, sp_stride);
    READ_3RD_NBR_SPINOR(SPINORTEX, sp_idx_3rd_nbr, sp_stride);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(0, long, ga_idx, sign);
      
    MAT_MUL_V(A, fat, i);
    MAT_MUL_V(B, long, t);    
    
    o00_re += A0_re;
    o00_im += A0_im;
    o01_re += A1_re;
    o01_im += A1_im;
    o02_re += A2_re;
    o02_im += A2_im;
    
    o00_re += B0_re;
    o00_im += B0_im;
    o01_re += B1_re;
    o01_im += B1_im;
    o02_re += B2_re;
    o02_im += B2_im;  
}


{
    // direction: -X
    
    int sp_idx_1st_nbr = ((x1==0) ? X+X1m1 : X-1) >> 1;
    int sp_idx_3rd_nbr = ((x1<3) ? X + X1-3: X -3)>>1; 

    // read gauge matrix from device memory
    READ_FAT_MATRIX(FATLINK1TEX, 1, sp_idx_1st_nbr);
    READ_LONG_MATRIX(LONGLINK1TEX, 1, sp_idx_3rd_nbr); 
    
    // read spinor from device memory
    READ_1ST_NBR_SPINOR(SPINORTEX, sp_idx_1st_nbr, sp_stride);
    READ_3RD_NBR_SPINOR(SPINORTEX, sp_idx_3rd_nbr, sp_stride);

    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(1, long, sp_idx_3rd_nbr, sign);
   
    ADJ_MAT_MUL_V(A, fat, i);
    ADJ_MAT_MUL_V(B, long, t);    

    o00_re -= A0_re;
    o00_im -= A0_im;
    o01_re -= A1_re;
    o01_im -= A1_im;
    o02_re -= A2_re;
    o02_im -= A2_im;
    
        
    o00_re -= B0_re;
    o00_im -= B0_im;
    o01_re -= B1_re;
    o01_im -= B1_im;
    o02_re -= B2_re;
    o02_im -= B2_im;  
        
}

{
    //direction: +Y
#if (DD_RECON_F != 18)
    if((x4+x1)%2 ==1){
	sign = -1;
    }else{
	sign =1;
    }
#endif
   
    int ga_idx = sid;

    int sp_idx_1st_nbr = ((x2==X2m1) ? X-X2X1mX1 : X+X1) >> 1;
    int sp_idx_3rd_nbr = ((x2 >= (X2 - 3) ) ? X + (-X2 + 3)*X1 : X+3*X1) >> 1;
    
    // read gauge matrix from device memory
    READ_FAT_MATRIX(FATLINK0TEX, 2, ga_idx);
    READ_LONG_MATRIX(LONGLINK0TEX, 2, ga_idx);

    // read spinor from device memory
    READ_1ST_NBR_SPINOR(SPINORTEX, sp_idx_1st_nbr, sp_stride);
    READ_3RD_NBR_SPINOR(SPINORTEX, sp_idx_3rd_nbr, sp_stride);

 
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(2, long, ga_idx, sign);

    MAT_MUL_V(A, fat, i);
    MAT_MUL_V(B, long, t);    
    
    o00_re += A0_re;
    o00_im += A0_im;
    o01_re += A1_re;
    o01_im += A1_im;
    o02_re += A2_re;
    o02_im += A2_im;
    
    o00_re += B0_re;
    o00_im += B0_im;
    o01_re += B1_re;
    o01_im += B1_im;
    o02_re += B2_re;
    o02_im += B2_im;  

}


{
    //direction: -Y
    
    int sp_idx_1st_nbr = ((x2==0)    ? X+X2X1mX1 : X-X1) >> 1;
    int sp_idx_3rd_nbr = ((x2 < 3) ? X + ( X2 - 3)*X1: X -3*X1 )>> 1; 
    
    // read gauge matrix from device memory
    READ_FAT_MATRIX(FATLINK1TEX, 3, sp_idx_1st_nbr);
    READ_LONG_MATRIX(LONGLINK1TEX, 3, sp_idx_3rd_nbr); 
    
    // read spinor from device memory
    READ_1ST_NBR_SPINOR(SPINORTEX, sp_idx_1st_nbr, sp_stride);
    READ_3RD_NBR_SPINOR(SPINORTEX, sp_idx_3rd_nbr, sp_stride);   

    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(3, long, sp_idx_3rd_nbr,sign);

    ADJ_MAT_MUL_V(A, fat, i);
    ADJ_MAT_MUL_V(B, long, t);    
    
    o00_re -= A0_re;
    o00_im -= A0_im;
    o01_re -= A1_re;
    o01_im -= A1_im;
    o02_re -= A2_re;
    o02_im -= A2_im;
    
    o00_re -= B0_re;
    o00_im -= B0_im;
    o01_re -= B1_re;
    o01_im -= B1_im;
    o02_re -= B2_re;
    o02_im -= B2_im;  
    
}



{
    //direction: +Z

#if (DD_RECON_F != 18)
    if((x4+x1+x2)%2 ==1){
	sign = -1;
    }else{
	sign =1;
    }
#endif
    
    int ga_idx = sid;
    
    int sp_idx_1st_nbr = ((x3==X3m1) ? X-X3X2X1mX2X1 : X+X2X1) >> 1;
    int sp_idx_3rd_nbr = ((x3>= (X2 - 3))? X + (-X3 + 3)*X2*X1: X + 3*X2*X1)>> 1;
    
    // read gauge matrix from device memory
    READ_FAT_MATRIX(FATLINK0TEX, 4, ga_idx);
    READ_LONG_MATRIX(LONGLINK0TEX, 4, ga_idx);
    
    // read spinor from device memory
    READ_1ST_NBR_SPINOR(SPINORTEX, sp_idx_1st_nbr, sp_stride);
    READ_3RD_NBR_SPINOR(SPINORTEX, sp_idx_3rd_nbr, sp_stride);   


    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(4, long, ga_idx, sign);

    MAT_MUL_V(A, fat, i);
    MAT_MUL_V(B, long, t);    
    
    o00_re += A0_re;
    o00_im += A0_im;
    o01_re += A1_re;
    o01_im += A1_im;
    o02_re += A2_re;
    o02_im += A2_im;
    
    o00_re += B0_re;
    o00_im += B0_im;
    o01_re += B1_re;
    o01_im += B1_im;
    o02_re += B2_re;
    o02_im += B2_im;      
 
}


{
    //direction: -Z
    
    int sp_idx_1st_nbr = ((x3==0)    ? X+X3X2X1mX2X1 : X-X2X1) >> 1;
    int sp_idx_3rd_nbr = ((x3 <3) ? X + (X3 -3)*X2X1: X - 3*X2X1)>>1;

    // read gauge matrix from device memory
    READ_FAT_MATRIX(FATLINK1TEX, 5, sp_idx_1st_nbr);
    READ_LONG_MATRIX(LONGLINK1TEX, 5, sp_idx_3rd_nbr); 
    
    // read spinor from device memory
    READ_1ST_NBR_SPINOR(SPINORTEX, sp_idx_1st_nbr, sp_stride);
    READ_3RD_NBR_SPINOR(SPINORTEX, sp_idx_3rd_nbr, sp_stride);      
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(5, long, sp_idx_3rd_nbr,sign);
   
    ADJ_MAT_MUL_V(A, fat, i);
    ADJ_MAT_MUL_V(B, long, t);    
    
    o00_re -= A0_re;
    o00_im -= A0_im;
    o01_re -= A1_re;
    o01_im -= A1_im;
    o02_re -= A2_re;
    o02_im -= A2_im;
    
    o00_re -= B0_re;
    o00_im -= B0_im;
    o01_re -= B1_re;
    o01_im -= B1_im;
    o02_re -= B2_re;
    o02_im -= B2_im;    
    
}



{
    //direction: +T
#if (DD_RECON_F != 18)
    if (x4>= (X4-3)){
	sign = -1;
    }else{
	sign =1;
    }
#endif

    int ga_idx = sid;
    
    int sp_idx_1st_nbr = ((x4==X4m1) ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1;
    int sp_idx_3rd_nbr = ((x4>=(X4-3))? X + ( - X4 +3)*X3X2X1 : X + 3*X3X2X1)>> 1; 
    
    // read gauge matrix from device memory
    READ_FAT_MATRIX(FATLINK0TEX, 6, ga_idx);
    READ_LONG_MATRIX(LONGLINK0TEX, 6, ga_idx);
    
    READ_1ST_NBR_SPINOR(SPINORTEX, sp_idx_1st_nbr, sp_stride);
    READ_3RD_NBR_SPINOR(SPINORTEX, sp_idx_3rd_nbr, sp_stride); 

    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(6, long, ga_idx, sign);
    
    MAT_MUL_V(A, fat, i);
    MAT_MUL_V(B, long, t);    

  
    o00_re += A0_re;
    o00_im += A0_im;
    o01_re += A1_re;
    o01_im += A1_im;
    o02_re += A2_re;
    o02_im += A2_im;
    
    o00_re += B0_re;
    o00_im += B0_im;
    o01_re += B1_re;
    o01_im += B1_im;
    o02_re += B2_re;
    o02_im += B2_im;      



}

{
    //direction: -T
#if (DD_RECON_F != 18)
    if ( ((x4-3+X4)%X4)>= (X4-3) ){
	sign = -1;
    }else{
	sign =1;
    }
#endif

    int sp_idx_1st_nbr = ((x4==0)    ? X+X4X3X2X1mX3X2X1 : X-X3X2X1) >> 1;
    int sp_idx_3rd_nbr = ((x4<3) ? X + (X4 -3)*X3X2X1: X - 3*X3X2X1) >> 1;
    
    READ_FAT_MATRIX(FATLINK1TEX, 7, sp_idx_1st_nbr);
    READ_LONG_MATRIX(LONGLINK1TEX, 7, sp_idx_3rd_nbr); 
    READ_1ST_NBR_SPINOR(SPINORTEX, sp_idx_1st_nbr, sp_stride);
    READ_3RD_NBR_SPINOR(SPINORTEX, sp_idx_3rd_nbr, sp_stride);       
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(7, long, sp_idx_3rd_nbr, sign);
    
    ADJ_MAT_MUL_V(A, fat, i);
    ADJ_MAT_MUL_V(B, long, t);    

    o00_re -= A0_re;
    o00_im -= A0_im;
    o01_re -= A1_re;
    o01_im -= A1_im;
    o02_re -= A2_re;
    o02_im -= A2_im;
    
    
    o00_re -= B0_re;
    o00_im -= B0_im;
    o01_re -= B1_re;
    o01_im -= B1_im;
    o02_re -= B2_re;
    o02_im -= B2_im;    

    
}

#if (DD_DAG == 1)
{
    o00_re = - o00_re;
    o00_im = - o00_im;
    o01_re = - o01_re;
    o01_im = - o01_im;
    o02_re = - o02_re;
    o02_im = - o02_im;
}

#endif




#ifdef DSLASH_XPAY
READ_ACCUM(ACCUMTEX)
o00_re = a*o00_re + accum0.x;
o00_im = a*o00_im + accum0.y;
o01_re = a*o01_re + accum1.x;
o01_im = a*o01_im + accum1.y;
o02_re = a*o02_re + accum2.x;
o02_im = a*o02_im + accum2.y;
#endif // DSLASH_XPAY

#ifdef DSLASH_AXPY
READ_ACCUM(ACCUMTEX)
o00_re = -o00_re + a*accum0.x;
o00_im = -o00_im + a*accum0.y;
o01_re = -o01_re + a*accum1.x;
o01_im = -o01_im + a*accum1.y;
o02_re = -o02_re + a*accum2.x;
o02_im = -o02_im + a*accum2.y;
#endif // DSLASH_AXPY
 
// write spinor field back to device memory
WRITE_SPINOR();


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

#undef fat_re
#undef fat_im

#undef fat00_re
#undef fat00_im
#undef fat01_re
#undef fat01_im
#undef fat02_re
#undef fat02_im
#undef fat10_re
#undef fat10_im
#undef fat11_re
#undef fat11_im
#undef fat12_re
#undef fat12_im
#undef fat20_re
#undef fat20_im
#undef fat21_re
#undef fat21_im
#undef fat22_re
#undef fat22_im

#undef long00_re
#undef long00_im
#undef long01_re
#undef long01_im
#undef long02_re
#undef long02_im
#undef long10_re
#undef long10_im
#undef long11_re
#undef long11_im
#undef long12_re
#undef long12_im
#undef long20_re
#undef long20_im
#undef long21_re
#undef long21_im
#undef long22_re
#undef long22_im

#undef long_re
#undef long_im

#undef i00_re
#undef i00_im
#undef i01_re
#undef i01_im
#undef i02_re
#undef i02_im

#undef t00_re
#undef t00_im
#undef t01_re
#undef t01_im
#undef t02_re
#undef t02_im

#undef SHARED_FLOATS_PER_THREAD
