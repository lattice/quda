// *** CUDA DSLASH ***
#undef SHARED_FLOATS_PER_THREAD 
#define SHARED_FLOATS_PER_THREAD 6

//#define kernel_type param.kernel_type

// input spinor
#if (DD_PREC==0)
#define spinorFloat double
double2 I0, I1, I2;
double2 T0, T1, T2;
#else
#define spinorFloat float
float2 I0, I1, I2;
float2 T0, T1, T2;
#endif

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
#endif

#endif

// conjugated gauge link
/*
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
 */

#if ((CUDA_VERSION >= 4010) && (__COMPUTE_CAPABILITY__ >= 200)) // NVVM compiler
#define VOLATILE
#else // Open64 compiler
#define VOLATILE volatile
#endif

// output spinor
#if (DD_PREC == 0)
#if (__COMPUTE_CAPABILITY__ >= 200)
#define SHARED_STRIDE 16 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE  8 // to avoid bank conflicts on G80 and GT200
#endif
extern __shared__ spinorFloat sd_data[];
VOLATILE spinorFloat *s = sd_data + SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)
  + (threadIdx.x % SHARED_STRIDE);
#else
#if (__COMPUTE_CAPABILITY__ >= 200)
#define SHARED_STRIDE 32 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE 16 // to avoid bank conflicts on G80 and GT200
#endif
  extern __shared__ spinorFloat ss_data[];
VOLATILE spinorFloat *s = ss_data + SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)
  + (threadIdx.x % SHARED_STRIDE);
#endif

  // output spinor
#define o0_re s[0*SHARED_STRIDE]
#define o0_im s[1*SHARED_STRIDE]
#define o1_re s[2*SHARED_STRIDE]
#define o1_im s[3*SHARED_STRIDE]
#define o2_re s[4*SHARED_STRIDE]
#define o2_im s[5*SHARED_STRIDE]


#include "read_gauge.h"
#include "io_spinor.h"

#define MAT_MUL_V(VOUT, M, V)                   \
    spinorFloat VOUT##0_re = M##00_re * V##00_re; \
  VOUT##0_re -= M##00_im * V##00_im;            \
  VOUT##0_re += M##01_re * V##01_re;            \
  VOUT##0_re -= M##01_im * V##01_im;            \
  VOUT##0_re += M##02_re * V##02_re;            \
  VOUT##0_re -= M##02_im * V##02_im;            \
  spinorFloat VOUT##0_im = M##00_re * V##00_im; \
  VOUT##0_im += M##00_im * V##00_re;            \
  VOUT##0_im += M##01_re * V##01_im;            \
  VOUT##0_im += M##01_im * V##01_re;            \
  VOUT##0_im += M##02_re * V##02_im;            \
  VOUT##0_im += M##02_im * V##02_re;            \
  spinorFloat VOUT##1_re = M##10_re * V##00_re; \
  VOUT##1_re -= M##10_im * V##00_im;            \
  VOUT##1_re += M##11_re * V##01_re;            \
  VOUT##1_re -= M##11_im * V##01_im;            \
  VOUT##1_re += M##12_re * V##02_re;            \
  VOUT##1_re -= M##12_im * V##02_im;            \
  spinorFloat VOUT##1_im = M##10_re * V##00_im; \
  VOUT##1_im += M##10_im * V##00_re;            \
  VOUT##1_im += M##11_re * V##01_im;            \
  VOUT##1_im += M##11_im * V##01_re;            \
  VOUT##1_im += M##12_re * V##02_im;            \
  VOUT##1_im += M##12_im * V##02_re;            \
  spinorFloat VOUT##2_re = M##20_re * V##00_re; \
  VOUT##2_re -= M##20_im * V##00_im;            \
  VOUT##2_re += M##21_re * V##01_re;            \
  VOUT##2_re -= M##21_im * V##01_im;            \
  VOUT##2_re += M##22_re * V##02_re;            \
  VOUT##2_re -= M##22_im * V##02_im;            \
  spinorFloat VOUT##2_im = M##20_re * V##00_im; \
  VOUT##2_im += M##20_im * V##00_re;            \
  VOUT##2_im += M##21_re * V##01_im;            \
  VOUT##2_im += M##21_im * V##01_re;            \
  VOUT##2_im += M##22_re * V##02_im;            \
  VOUT##2_im += M##22_im * V##02_re;

#define ADJ_MAT_MUL_V(VOUT, M, V)               \
    spinorFloat VOUT##0_re = M##00_re * V##00_re; \
  VOUT##0_re += M##00_im * V##00_im;            \
  VOUT##0_re += M##10_re * V##01_re;            \
  VOUT##0_re += M##10_im * V##01_im;            \
  VOUT##0_re += M##20_re * V##02_re;            \
  VOUT##0_re += M##20_im * V##02_im;            \
  spinorFloat VOUT##0_im = M##00_re * V##00_im; \
  VOUT##0_im -= M##00_im * V##00_re;            \
  VOUT##0_im += M##10_re * V##01_im;            \
  VOUT##0_im -= M##10_im * V##01_re;            \
  VOUT##0_im += M##20_re * V##02_im;            \
  VOUT##0_im -= M##20_im * V##02_re;            \
  spinorFloat VOUT##1_re = M##01_re * V##00_re; \
  VOUT##1_re += M##01_im * V##00_im;            \
  VOUT##1_re += M##11_re * V##01_re;            \
  VOUT##1_re += M##11_im * V##01_im;            \
  VOUT##1_re += M##21_re * V##02_re;            \
  VOUT##1_re += M##21_im * V##02_im;            \
  spinorFloat VOUT##1_im = M##01_re * V##00_im; \
  VOUT##1_im -= M##01_im * V##00_re;            \
  VOUT##1_im += M##11_re * V##01_im;            \
  VOUT##1_im -= M##11_im * V##01_re;            \
  VOUT##1_im += M##21_re * V##02_im;            \
  VOUT##1_im -= M##21_im * V##02_re;            \
  spinorFloat VOUT##2_re = M##02_re * V##00_re; \
  VOUT##2_re += M##02_im * V##00_im;            \
  VOUT##2_re += M##12_re * V##01_re;            \
  VOUT##2_re += M##12_im * V##01_im;            \
  VOUT##2_re += M##22_re * V##02_re;            \
  VOUT##2_re += M##22_im * V##02_im;            \
  spinorFloat VOUT##2_im = M##02_re * V##00_im; \
  VOUT##2_im -= M##02_im * V##00_re;            \
  VOUT##2_im += M##12_re * V##01_im;            \
  VOUT##2_im -= M##12_im * V##01_re;            \
  VOUT##2_im += M##22_re * V##02_im;            \
  VOUT##2_im -= M##22_im * V##02_re;


  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid >= param.threads) return;

  int za,zb; 
  int x1h, x2h;
  int x1,x2,x3,x4;
  int x1_new, x2_new, x3_new, x4_new;
  int af;
  int x1odd,x2odd;
  int X;

  /* template for kparam.X4 kparam.X3 kparam.X2 kparam.X1h
     za = sid / X_one_h;
     x_one_h = sid - za*X_one_h;
     zb = za / X_two_;
     x_two_ = za - zb*X_two_;
     x_four_ = zb / X_three_;
     x_three_ = zb - x_four_*X_three_;
     af = (x_four_ >= 3)?(X_four_-6):0;
     x_four__new = x_four_ + af;
     x_four_=x_four__new;
     x_one_odd = (x_two_ + x_three_ + x_four_ + param.parity) & 1;
     x_one_ = 2*x_one_h + x_one_odd;
     X = x4*kparam.X3X2X1+x3*kparam.X2X1+x2*kparam.X1+x1;
     sid = X>>1;

   */

  //if(kernel_type != INTERIOR_KERNEL) return; // hack added by J.F.

  if(kernel_type == INTERIOR_KERNEL){

    //data order: kparam.X4 kparam.X3 kparam.X2 kparam.X1h
    za = sid / kparam.X1h;
    x1h = sid - za*kparam.X1h;
    zb = za / kparam.X2;
    x2 = za - zb*kparam.X2;
    x4 = zb / kparam.X3;
    x3 = zb - x4*kparam.X3;
    x1odd = (x2 + x3 + x4 + param.parity) & 1;
    x1 = 2*x1h + x1odd;
    X = 2*sid + x1odd;

  }else if (kernel_type == EXTERIOR_KERNEL_X){
    //data order: kparam.X1 kparam.X4 kparam.X3 kparam.X2h
    za = sid / kparam.X2h;
    x2h = sid - za*kparam.X2h;
    zb = za / kparam.X3;
    x3 = za - zb*kparam.X3;
    x1 = zb / kparam.X4;
    x4 = zb - x1*kparam.X4;
    af = (x1 >= Nface)?(kparam.X1-2*Nface):0;
    x1_new = x1 + af;
    x1=x1_new;
    x2odd = (x3 + x4 + x1 + param.parity) & 1;
    x2 = 2*x2h + x2odd;
    X = x4*kparam.X3X2X1+x3*kparam.X2X1+x2*kparam.X1+x1;
    sid = X>>1;
  }else if (kernel_type == EXTERIOR_KERNEL_Y){
    //data order: kparam.X2 kparam.X4 kparam.X3 kparam.X1h
    za = sid / kparam.X1h;
    x1h = sid - za*kparam.X1h;
    zb = za / kparam.X3;
    x3 = za - zb*kparam.X3;
    x2 = zb / kparam.X4;
    x4 = zb - x2*kparam.X4;
    af = (x2 >= Nface)?(kparam.X2-2*Nface):0;
    x2_new = x2 + af;
    x2=x2_new;
    x1odd = (x3 + x4 + x2 + param.parity) & 1;
    x1 = 2*x1h + x1odd;
    X = x4*kparam.X3X2X1+x3*kparam.X2X1+x2*kparam.X1+x1;
    sid = X>>1;

  }else if (kernel_type == EXTERIOR_KERNEL_Z){
    //data order: kparam.X3 kparam.X4 kparam.X2 kparam.X1h
    za = sid / kparam.X1h;
    x1h = sid - za*kparam.X1h;
    zb = za / kparam.X2;
    x2 = za - zb*kparam.X2;
    x3 = zb / kparam.X4;
    x4 = zb - x3*kparam.X4;
    af = (x3 >= Nface)?(kparam.X3-2*Nface):0;
    x3_new = x3 + af;
    x3=x3_new;
    x1odd = (x2 + x4 + x3 + param.parity) & 1;
    x1 = 2*x1h + x1odd;
    X = x4*kparam.X3X2X1+x3*kparam.X2X1+x2*kparam.X1+x1;
    sid = X>>1;
  }else if (kernel_type == EXTERIOR_KERNEL_T){
    //data order: kparam.X4 kparam.X3 kparam.X2 kparam.X1h
    za = sid / kparam.X1h;
    x1h = sid - za*kparam.X1h;
    zb = za / kparam.X2;
    x2 = za - zb*kparam.X2;
    x4 = zb / kparam.X3;
    x3 = zb - x4*kparam.X3;
    af = (x4 >= Nface)?(kparam.X4-2*Nface):0;
    x4_new = x4 + af;
    sid +=kparam.Vsh*(x4_new -x4);
    x4=x4_new;
    x1odd = (x2 + x3 + x4 + param.parity) & 1;
    x1 = 2*x1h + x1odd;
    X = 2*sid + x1odd;
  }

o0_re = o0_im = 0.f;
o1_re = o1_im = 0.f;
o2_re = o2_im = 0.f;

const int fatlinkStride = kparam.fatlinkStride;
const int longlinkStride = kparam.longlinkStride;
#if (DD_PREC == 2)
const float fatlinkMax = kparam.fatlinkMax;
#endif

{  //direction: +X
#if (DD_RECON < 2)
  int sign = (x4%2 == 1) ? -1 : 1;
#endif
  int ga_idx = sid;
#ifdef MULTI_GPU
  if ( (kernel_type == INTERIOR_KERNEL && ( (!param.ghostDim[0]) || x1 < kparam.X1m1) )|| (kernel_type == EXTERIOR_KERNEL_X && x1 >= kparam.X1m1))
#endif
  {
    int sp_idx_1st_nbr = ((x1==kparam.X1m1) ? X-kparam.X1m1 : X+1) >> 1;
    READ_FAT_MATRIX(FAT, FATLINK0TEX, 0, ga_idx, fatlinkStride);
    int nbr_idx1 = sp_idx_1st_nbr;
    int stride1 = kparam.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx = nbr_idx1;
#endif	    
#ifdef MULTI_GPU
    if ( (kernel_type == EXTERIOR_KERNEL_X)){
      int space_con = (x4*kparam.X3X2+x3*kparam.X2+x2)/2;	
      if (x1 >= kparam.X1m1){
        nbr_idx1 = param.ghostOffset[0] + 3*Nface*kparam.ghostFace[0] +(x1-kparam.X1m1)*kparam.ghostFace[0]+ space_con;
        stride1 = Nface*kparam.ghostFace[0];
#if (DD_PREC == 2) //half precision
        norm_idx = param.ghostNormOffset[0] + Nface*kparam.ghostFace[0] + (x1-kparam.X1m1)*kparam.ghostFace[0]+ space_con;
#endif		    
      }
    }
#endif
    READ_KS_NBR_SPINOR(I, SPINORTEX, nbr_idx1, stride1);
    MAT_MUL_V(A, fat, i);    

    if(kernel_type == EXTERIOR_KERNEL_X){
      if(x1 >= kparam.X1m1){
        printf("+X x = (%d, %d, %d, %d), I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }
    }else if(kparam.X1 == 12){
      if(x1 == kparam.X1-3){
        printf("+X x = (%d, %d, %d, %d), neighbor I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }     
    }

    o0_re += A0_re;
    o0_im += A0_im;
    o1_re += A1_re;
    o1_im += A1_im;
    o2_re += A2_re;
    o2_im += A2_im;
  }

  if(hasNaik){
#ifdef MULTI_GPU
    if ( (kernel_type == INTERIOR_KERNEL && ( (!param.ghostDim[0]) || x1 < kparam.X1m3) ) || (kernel_type == EXTERIOR_KERNEL_X && x1 >= kparam.X1m3)  )
#endif
    {
      int sp_idx_3rd_nbr = ((x1 >= kparam.X1m3) ? X -kparam.X1m3 : X+3) >> 1;
      READ_LONG_MATRIX(LONG, LONGLINK0TEX, 0, ga_idx, longlinkStride);        
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = kparam.sp_stride;    
#if (DD_PREC == 2) //half precision
      int norm_idx = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
      if( (kernel_type == EXTERIOR_KERNEL_X)){
        int space_con = (x4*kparam.X3X2+x3*kparam.X2+x2)/2;		
        if(x1  >= kparam.X1m3){
          nbr_idx3 = param.ghostOffset[0] + 3*Nface*kparam.ghostFace[0] +(x1-kparam.X1m3)*kparam.ghostFace[0]+ space_con;
          stride3 = Nface*kparam.ghostFace[0];
#if (DD_PREC == 2) //half precision
          norm_idx = param.ghostNormOffset[0] + Nface*kparam.ghostFace[0] + (x1-kparam.X1m3)*kparam.ghostFace[0]+ space_con;
#endif	
        }
      }
#endif
      READ_KS_NBR_SPINOR(T, SPINORTEX, nbr_idx3, stride3);   
      RECONSTRUCT_GAUGE_MATRIX(0, long, ga_idx, sign);

      MAT_MUL_V(B, long, t);        
      o0_re += B0_re;
      o0_im += B0_im;
      o1_re += B1_re;
      o1_im += B1_im;
      o2_re += B2_re;
      o2_im += B2_im;  
    }
  } // hasNaik    
} // +X

{  // direction: -X
#if (DD_RECON < 2)
  int sign = (x4%2 == 1) ? -1 : 1;
#endif
  int dir =1;
#ifdef MULTI_GPU
  int space_con = (x4*kparam.X3X2 + x3*kparam.X2+ x2) >>1;
  if ( (kernel_type == INTERIOR_KERNEL && ( (!param.ghostDim[0]) || x1 >= 1)) || (kernel_type == EXTERIOR_KERNEL_X && x1 < 1))
#endif
  {
    int sp_idx_1st_nbr = ((x1==0) ? X+kparam.X1m1 : X-1) >> 1;
    int fat_idx = sp_idx_1st_nbr;
#ifdef MULTI_GPU
    if(x1 < 1){
      fat_idx = kparam.Vh + space_con;
    }
#endif
    READ_FAT_MATRIX(FAT, FATLINK1TEX, dir, fat_idx, fatlinkStride);
    int nbr_idx1 = sp_idx_1st_nbr;
    int stride1 = kparam.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
    if(kernel_type == EXTERIOR_KERNEL_X){
      if (x1 == 0){
        nbr_idx1 = param.ghostOffset[0] + (Nface-1)*kparam.ghostFace[0]+ space_con;
        stride1 = Nface*kparam.ghostFace[0];
#if (DD_PREC == 2) //half precision
        norm_idx = param.ghostNormOffset[0]  + (Nface-1)*kparam.ghostFace[0]+ space_con;
#endif	
      }        
    }
#endif
    READ_KS_NBR_SPINOR(I, SPINORTEX, nbr_idx1, stride1);
    ADJ_MAT_MUL_V(A, fat, i); 
      
    if(kernel_type == EXTERIOR_KERNEL_X){
      if(x1 == 0){
        printf("-X x = (%d, %d, %d, %d), I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }
    }else if(kparam.X1 == 12){ // Need to think more about this
      if(x1 == 2){
        printf("-X x = (%d, %d, %d, %d), I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }
    }


    o0_re -= A0_re;
    o0_im -= A0_im;
    o1_re -= A1_re;
    o1_im -= A1_im;
    o2_re -= A2_re;
    o2_im -= A2_im;
  }

  if(hasNaik){
#ifdef MULTI_GPU    
    if ( (kernel_type == INTERIOR_KERNEL && ( (!param.ghostDim[0]) || x1 >= 3)) || (kernel_type == EXTERIOR_KERNEL_X && x1 < 3))
#endif
    {
      int sp_idx_3rd_nbr = ((x1<3) ? X + kparam.X1m3: X -3)>>1; 
      int long_idx = sp_idx_3rd_nbr;
#ifdef MULTI_GPU
      if (x1 < 3){
        long_idx = kparam.Vh + x1*kparam.X4X3X2h + space_con;
      }    
#endif
      READ_LONG_MATRIX(LONG, LONGLINK1TEX, dir, long_idx, longlinkStride); 		
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = kparam.sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx = nbr_idx3;
#endif	     
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_X){
        if (x1 < 3){
          nbr_idx3 = param.ghostOffset[0] + (Nface-3 + x1)*kparam.ghostFace[0]+ space_con;
          stride3 = Nface*kparam.ghostFace[0];
#if (DD_PREC == 2) //half precision
          norm_idx = param.ghostNormOffset[0]  + (Nface-3 + x1)*kparam.ghostFace[0]+ space_con;
#endif
        }
      }
#endif
      READ_KS_NBR_SPINOR(T, SPINORTEX, nbr_idx3, stride3);  
      RECONSTRUCT_GAUGE_MATRIX(1, long, sp_idx_3rd_nbr, sign);
      ADJ_MAT_MUL_V(B, long, t);    
      o0_re -= B0_re;
      o0_im -= B0_im;
      o1_re -= B1_re;
      o1_im -= B1_im;
      o2_re -= B2_re;
      o2_im -= B2_im;  
    }
  } // hasNaik
} // -X



{ //direction: +Y
#if (DD_RECON < 2)
  int sign = ((x4+x1)%2 == 1) ? -1 : 1;
#endif
  int ga_idx = sid;
#ifdef MULTI_GPU
  int space_con = (x4*kparam.X3X1+x3*kparam.X1+x1)/2;
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[1]) || x2 < kparam.X2m1))|| (kernel_type == EXTERIOR_KERNEL_Y && x2 >= kparam.X2m1))
#endif
  {
    int sp_idx_1st_nbr = ((x2==kparam.X2m1) ? X-kparam.X2X1mX1 : X+kparam.X1) >> 1;
    READ_FAT_MATRIX(FAT, FATLINK0TEX, 2, ga_idx, fatlinkStride);
    int nbr_idx1 = sp_idx_1st_nbr;
    int stride1 = kparam.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
    if(kernel_type == EXTERIOR_KERNEL_Y){	    
      if(x2 >= kparam.X2m1){
        nbr_idx1 = param.ghostOffset[1] + 3*Nface*kparam.ghostFace[1] +(x2-kparam.X2m1)*kparam.ghostFace[1]+ space_con;
        stride1 = Nface*kparam.ghostFace[1];
#if (DD_PREC == 2) //half precision
        norm_idx = param.ghostNormOffset[1] + Nface*kparam.ghostFace[1] + (x2-kparam.X2m1)*kparam.ghostFace[1]+ space_con;
#endif		    
      }      
    }
#endif
    READ_KS_NBR_SPINOR(I, SPINORTEX, nbr_idx1, stride1);
    MAT_MUL_V(A, fat, i);

    if(kernel_type == EXTERIOR_KERNEL_Y){
      if(x2 >= kparam.X2m1){
        printf("+Y x = (%d, %d, %d, %d), I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }
    }else if(kparam.X2 == 12){
      if(x2 == kparam.X2-3){
        printf("+Y x = (%d, %d, %d, %d), neighbor I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }     
    }

    o0_re += A0_re;
    o0_im += A0_im;
    o1_re += A1_re;
    o1_im += A1_im;
    o2_re += A2_re;
    o2_im += A2_im;
  }

  if(hasNaik){ 
#ifdef MULTI_GPU
    if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[1]) || x2 < kparam.X2m3))|| (kernel_type == EXTERIOR_KERNEL_Y && x2 >= kparam.X2m3))    
#endif
    {
      int sp_idx_3rd_nbr = ((x2 >= kparam.X2m3 ) ? X-kparam.X2m3*kparam.X1 : X+3*kparam.X1) >> 1;    
      READ_LONG_MATRIX(LONG, LONGLINK0TEX, 2, ga_idx, longlinkStride);
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = kparam.sp_stride;        
#if (DD_PREC == 2) //half precision
      int norm_idx = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_Y){
        if (x2>= kparam.X2m3){
          nbr_idx3 = param.ghostOffset[1] + 3*Nface*kparam.ghostFace[1] +(x2-kparam.X2m3)*kparam.ghostFace[1]+ space_con;
          stride3 = Nface*kparam.ghostFace[1];
#if (DD_PREC == 2) //half precision
          norm_idx = param.ghostNormOffset[1] + Nface*kparam.ghostFace[1] + (x2-kparam.X2m3)*kparam.ghostFace[1]+ space_con;
#endif		    
        }
      }
#endif    
      READ_KS_NBR_SPINOR(T, SPINORTEX, nbr_idx3, stride3);
      RECONSTRUCT_GAUGE_MATRIX(2, long, ga_idx, sign);
      MAT_MUL_V(B, long, t);            
      o0_re += B0_re;
      o0_im += B0_im;
      o1_re += B1_re;
      o1_im += B1_im;
      o2_re += B2_re;
      o2_im += B2_im;  
    }
  } // hasNaik
} // +Y




{  //direction: -Y
#if (DD_RECON < 2)
  int sign = ((x4+x1)%2 == 1) ? -1 : 1;
#endif

  int dir=3;
#ifdef MULTI_GPU
  int space_con = (x4*kparam.X3X1 + x3*kparam.X1+ x1) >>1;    
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[1]) || x2 >= 1)) || (kernel_type == EXTERIOR_KERNEL_Y && x2 < 1))
#endif
  {
    int sp_idx_1st_nbr = ((x2==0) ? X+kparam.X2X1mX1 : X-kparam.X1) >> 1;
    int fat_idx=sp_idx_1st_nbr;
#ifdef MULTI_GPU
    if (x2 == 0){
      fat_idx = kparam.Vh + space_con;
    }    
#endif
    READ_FAT_MATRIX(FAT, FATLINK1TEX, dir, fat_idx, fatlinkStride);
    int nbr_idx1 = sp_idx_1st_nbr;
    int stride1 = kparam.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
    if (kernel_type == EXTERIOR_KERNEL_Y){
      if (x2 == 0){
        nbr_idx1 = param.ghostOffset[1] + (Nface-1)*kparam.ghostFace[1]+ space_con;
        stride1 = Nface*kparam.ghostFace[1];
#if (DD_PREC == 2) //half precision
        norm_idx = param.ghostNormOffset[1]  + (Nface-1)*kparam.ghostFace[1]+ space_con;
#endif	

      }              
    }
#endif
    READ_KS_NBR_SPINOR(I, SPINORTEX, nbr_idx1, stride1);
    ADJ_MAT_MUL_V(A, fat, i);

    if(kernel_type == EXTERIOR_KERNEL_Y){
      if(x2 == 0){
        printf("-Y x = (%d, %d, %d, %d), I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }
    }else if(kparam.X2 == 12){
      if(x2 == 2){
        printf("-Y x = (%d, %d, %d, %d), I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }
    }

    o0_re -= A0_re;
    o0_im -= A0_im;
    o1_re -= A1_re;
    o1_im -= A1_im;
    o2_re -= A2_re;
    o2_im -= A2_im;
  }

  if(hasNaik){
#ifdef MULTI_GPU
    if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[1]) || x2 >= 3)) || (kernel_type == EXTERIOR_KERNEL_Y && x2 < 3))
#endif
    {
      int sp_idx_3rd_nbr = ((x2 < 3) ? X + kparam.X2m3*kparam.X1: X-3*kparam.X1 )>> 1; 
      int long_idx = sp_idx_3rd_nbr;
#ifdef MULTI_GPU
      if (x2 < 3){
        long_idx = kparam.Vh+ x2*kparam.X4X3X1h + space_con;
      }    
#endif
      READ_LONG_MATRIX(LONG, LONGLINK1TEX, dir, long_idx, longlinkStride); 
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = kparam.sp_stride;    
#if (DD_PREC == 2) //half precision
      int norm_idx = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_Y){
        if (x2 < 3){
          nbr_idx3 = param.ghostOffset[1] + (Nface-3 + x2)*kparam.ghostFace[1]+ space_con;
          stride3 = Nface*kparam.ghostFace[1];
#if (DD_PREC == 2) //half precision
          norm_idx = param.ghostNormOffset[1]  + (Nface-3 + x2)*kparam.ghostFace[1]+ space_con;
#endif
        }
      }
#endif
      READ_KS_NBR_SPINOR(T, SPINORTEX, nbr_idx3, stride3);
      RECONSTRUCT_GAUGE_MATRIX(3, long, sp_idx_3rd_nbr,sign);

      ADJ_MAT_MUL_V(B, long, t);    
      o0_re -= B0_re;
      o0_im -= B0_im;
      o1_re -= B1_re;
      o1_im -= B1_im;
      o2_re -= B2_re;
      o2_im -= B2_im;  
    }
  } // hasNaik    
} // -Y



{  //direction: +Z
#if (DD_RECON < 2)
  int sign = ((x4+x1+x2)%2 == 1) ? -1 : 1;
#endif
  int ga_idx = sid;

#ifdef MULTI_GPU
  int space_con = (x4*kparam.X2X1+x2*kparam.X1+x1)/2;
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[2]) || x3 < kparam.X3m1))|| (kernel_type == EXTERIOR_KERNEL_Z && x3 >= kparam.X3m1))
#endif
  {
    int sp_idx_1st_nbr = ((x3==kparam.X3m1) ? X-kparam.X3X2X1mX2X1 : X+kparam.X2X1) >> 1;
    READ_FAT_MATRIX(FAT, FATLINK0TEX, 4, ga_idx, fatlinkStride);
    int nbr_idx1 = sp_idx_1st_nbr;
    int stride1 = kparam.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
    if(kernel_type == EXTERIOR_KERNEL_Z){	
      if (x3 >= kparam.X3m1){
        nbr_idx1 = param.ghostOffset[2] + 3*Nface*kparam.ghostFace[2] +(x3-kparam.X3m1)*kparam.ghostFace[2]+ space_con;
        stride1 = Nface*kparam.ghostFace[2];	    
#if (DD_PREC == 2) //half precision
        norm_idx = param.ghostNormOffset[2] + Nface*kparam.ghostFace[2] + (x3-kparam.X3m1)*kparam.ghostFace[2]+ space_con;
#endif		
      }      
    }
#endif

    READ_KS_NBR_SPINOR(I, SPINORTEX, nbr_idx1, stride1);
    MAT_MUL_V(A, fat, i);	 

    if(kernel_type == EXTERIOR_KERNEL_Z){
      if(x3 >= kparam.X3m1){
        printf("+Z x = (%d, %d, %d, %d), I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }
    }else if(kparam.X3 == 12){
      if(x3 == kparam.X3-3){
        printf("+Z x = (%d, %d, %d, %d), neighbor I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }     
    }

    o0_re += A0_re;
    o0_im += A0_im;
    o1_re += A1_re;
    o1_im += A1_im;
    o2_re += A2_re;
    o2_im += A2_im;
  }


  if(hasNaik){ 
#ifdef MULTI_GPU
    if ((kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[2]) || x3 < kparam.X3m3))|| (kernel_type == EXTERIOR_KERNEL_Z && x3 >= kparam.X3m3))
#endif
    {
      int sp_idx_3rd_nbr = ((x3>= kparam.X3m3)? X -kparam.X3m3*kparam.X2X1: X + 3*kparam.X2X1)>> 1;    
      READ_LONG_MATRIX(LONG, LONGLINK0TEX, 4, ga_idx, longlinkStride);
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = kparam.sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_Z){
        if (x3 >= kparam.X3m3){
          nbr_idx3 = param.ghostOffset[2] + 3*Nface*kparam.ghostFace[2] +(x3-kparam.X3m3)*kparam.ghostFace[2]+ space_con;
          stride3 = Nface*kparam.ghostFace[2];
#if (DD_PREC == 2) //half precision
          norm_idx = param.ghostNormOffset[2] + Nface*kparam.ghostFace[2] + (x3-kparam.X3m3)*kparam.ghostFace[2]+ space_con;
#endif
        }
      }
#endif
      READ_KS_NBR_SPINOR(T, SPINORTEX, nbr_idx3, stride3);
      RECONSTRUCT_GAUGE_MATRIX(4, long, ga_idx, sign);    
      MAT_MUL_V(B, long, t);        
      o0_re += B0_re;
      o0_im += B0_im;
      o1_re += B1_re;
      o1_im += B1_im;
      o2_re += B2_re;
      o2_im += B2_im;      
    }
  } // hasNaik
} // Z 



{    //direction: -Z
#if (DD_RECON < 2)
  int sign = ((x4+x1+x2)%2 == 1) ? -1 : 1;
#endif
  int dir = 5;
#ifdef MULTI_GPU
  int space_con = (x4*kparam.X2X1 + x2*kparam.X1+ x1) >>1;    
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[2]) || x3 >= 1)) || (kernel_type == EXTERIOR_KERNEL_Z && x3 < 1))
#endif
  {
    int sp_idx_1st_nbr = ((x3==0) ? X+kparam.X3X2X1mX2X1 : X-kparam.X2X1) >> 1;
    int fat_idx = sp_idx_1st_nbr;
#ifdef MULTI_GPU
    if ((x3 -1) < 0){
      fat_idx = kparam.Vh + space_con;
    }    
#endif
    READ_FAT_MATRIX(FAT, FATLINK1TEX, dir, fat_idx, fatlinkStride);
    int nbr_idx1 = sp_idx_1st_nbr;
    int stride1 = kparam.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
    if (kernel_type == EXTERIOR_KERNEL_Z){
      if (x3 == 0){
        nbr_idx1 = param.ghostOffset[2] + (Nface-1)*kparam.ghostFace[2]+ space_con;
        stride1 = Nface*kparam.ghostFace[2];
#if (DD_PREC == 2) //half precision
        norm_idx = param.ghostNormOffset[2]  + (Nface-1)*kparam.ghostFace[2]+ space_con;
#endif			    
      }        
    }
#endif
    READ_KS_NBR_SPINOR(I, SPINORTEX, nbr_idx1, stride1);
    ADJ_MAT_MUL_V(A, fat, i);

    if(kernel_type == EXTERIOR_KERNEL_Z){
      if(x3 == 0){
        printf("-Z x = (%d, %d, %d, %d), parity = %d,  I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, param.parity, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }
    }else if(kparam.X3 == 12){
      if(x3 == 2){
        printf("-Z x = (%d, %d, %d, %d), parity = %d,  I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, param.parity, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }
    }

    o0_re -= A0_re;
    o0_im -= A0_im;
    o1_re -= A1_re;
    o1_im -= A1_im;
    o2_re -= A2_re;
    o2_im -= A2_im;
  }

  if(hasNaik){ 
#ifdef MULTI_GPU
    if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[2]) || x3 >= 3)) || (kernel_type == EXTERIOR_KERNEL_Z && x3 < 3))
#endif
    {
      int sp_idx_3rd_nbr = ((x3 <3) ? X + kparam.X3m3*kparam.X2X1: X - 3*kparam.X2X1)>>1;
      int long_idx = sp_idx_3rd_nbr;
#ifdef MULTI_GPU
      if ((x3-3) < 0){
        long_idx = kparam.Vh + x3*kparam.X4X2X1h + space_con;
      }    
#endif
      READ_LONG_MATRIX(LONG, LONGLINK1TEX, dir, long_idx, longlinkStride);         
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = kparam.sp_stride;    
#if (DD_PREC == 2) //half precision
      int norm_idx = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_Z){
        if(x3 - 3 < 0){
          nbr_idx3 = param.ghostOffset[2] + (Nface-3 + x3)*kparam.ghostFace[2]+ space_con;
          stride3 = Nface*kparam.ghostFace[2];
#if (DD_PREC == 2) //half precision
          norm_idx = param.ghostNormOffset[2]  + (Nface-3 + x3)*kparam.ghostFace[2]+ space_con;
#endif			    
        }
      }
#endif
      READ_KS_NBR_SPINOR(T, SPINORTEX, nbr_idx3, stride3);
      RECONSTRUCT_GAUGE_MATRIX(5, long, sp_idx_3rd_nbr,sign);
      ADJ_MAT_MUL_V(B, long, t);    

      o0_re -= B0_re;
      o0_im -= B0_im;
      o1_re -= B1_re;
      o1_im -= B1_im;
      o2_re -= B2_re;
      o2_im -= B2_im;    
    }
  } // hasNaik
} // -Z direction


{  //direction: +T
#if (DD_RECON < 2)
  int sign = (x4 >= (kparam.X4-3)) ? -1 : 1;
#endif
  int ga_idx = sid;
#ifdef MULTI_GPU
  int space_con = (x3*kparam.X2X1+x2*kparam.X1+x1)/2;
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[3]) || x4 < kparam.X4m1))|| (kernel_type == EXTERIOR_KERNEL_T && x4 >= kparam.X4m1))
#endif
  {    
    int sp_idx_1st_nbr = ((x4==kparam.X4m1) ? X-kparam.X4X3X2X1mX3X2X1 : X+kparam.X3X2X1) >> 1;
    READ_FAT_MATRIX(FAT, FATLINK0TEX, 6, ga_idx, fatlinkStride);
    int nbr_idx1 = sp_idx_1st_nbr;
    int stride1 = kparam.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx = nbr_idx1;
#endif
#ifdef MULTI_GPU
    if (kernel_type == EXTERIOR_KERNEL_T){      
      if (x4 >= kparam.X4m1){
        nbr_idx1 = param.ghostOffset[3] + 3*Nface*kparam.ghostFace[3] +(x4-kparam.X4m1)*kparam.ghostFace[3]+ space_con;
        stride1 = Nface*kparam.ghostFace[3];
#if (DD_PREC == 2) //half precision
        norm_idx = param.ghostNormOffset[3] + Nface*kparam.ghostFace[3] + (x4-kparam.X4m1)*kparam.ghostFace[3]+ space_con;
#endif
      }
    }
#endif
    READ_KS_NBR_SPINOR(I, SPINORTEX, nbr_idx1, stride1);    
    MAT_MUL_V(A, fat, i);

    if(kernel_type == EXTERIOR_KERNEL_T){
      if(x4 >= kparam.X4m1){
        printf("+T x = (%d, %d, %d, %d), I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }
    }else if(kparam.X4 == 12){
      if(x4 == kparam.X4-3){
        printf("+T x = (%d, %d, %d, %d), neighbor I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }     
    }

    o0_re += A0_re;
    o0_im += A0_im;
    o1_re += A1_re;
    o1_im += A1_im;
    o2_re += A2_re;
    o2_im += A2_im;
  }

  if(hasNaik){    
#ifdef MULTI_GPU
    if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[3]) || x4 < kparam.X4m3))|| (kernel_type == EXTERIOR_KERNEL_T && x4 >= kparam.X4m3))
#endif
    {
      int sp_idx_3rd_nbr = ((x4>=kparam.X4m3)? X -kparam.X4m3*kparam.X3X2X1 : X + 3*kparam.X3X2X1)>> 1;     
      READ_LONG_MATRIX(LONG, LONGLINK0TEX, 6, ga_idx, longlinkStride);    
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = kparam.sp_stride;    
#if (DD_PREC == 2) //half precision
      int norm_idx = nbr_idx3;
#endif
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_T){
        if (x4  >= kparam.X4m3){
          nbr_idx3 = param.ghostOffset[3] + 3*Nface*kparam.ghostFace[3] +(x4-kparam.X4m3)*kparam.ghostFace[3]+ space_con;
          stride3 = Nface*kparam.ghostFace[3];
#if (DD_PREC == 2) //half precision
          norm_idx = param.ghostNormOffset[3] + Nface*kparam.ghostFace[3] + (x4-kparam.X4m3)*kparam.ghostFace[3]+ space_con;
#endif
        }
      }
#endif
      READ_KS_NBR_SPINOR(T, SPINORTEX, nbr_idx3, stride3); 
      RECONSTRUCT_GAUGE_MATRIX(6, long, ga_idx, sign);
      MAT_MUL_V(B, long, t);    
      o0_re += B0_re;
      o0_im += B0_im;
      o1_re += B1_re;
      o1_im += B1_im;
      o2_re += B2_re;
      o2_im += B2_im;      
    }
  } // hasNaik
}


{  //direction: -T
#if (DD_RECON < 2)
  int sign = ( ((x4+kparam.X4m3)%kparam.X4)>= kparam.X4m3 ) ? -1 : 1;
#endif
  int dir = 7;
#ifdef MULTI_GPU
  int space_con = (x3*kparam.X2X1+x2*kparam.X1+x1)/2;
  if ((kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[3]) || x4 >= 1)) || (kernel_type == EXTERIOR_KERNEL_T && x4 < 1))
#endif
  {
    int sp_idx_1st_nbr = ((x4==0)    ? X+kparam.X4X3X2X1mX3X2X1 : X-kparam.X3X2X1) >> 1;
    int fat_idx = sp_idx_1st_nbr;    
    int nbr_idx1 = sp_idx_1st_nbr;
    int stride1 = kparam.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx = nbr_idx1;
#endif
#ifdef MULTI_GPU
    if (kernel_type == EXTERIOR_KERNEL_T){
      if (x4==0){
        fat_idx = kparam.Vh + space_con;
        nbr_idx1 = param.ghostOffset[3] + (Nface-1)*kparam.ghostFace[3]+ space_con;
        stride1 = Nface*kparam.ghostFace[3];
#if (DD_PREC == 2) //half precision
        norm_idx = param.ghostNormOffset[3]  + (Nface-1)*kparam.ghostFace[3]+ space_con;
#endif		    
      }        	
    }
#endif
    READ_FAT_MATRIX(FAT, FATLINK1TEX, dir, fat_idx, fatlinkStride);
    READ_KS_NBR_SPINOR(I, SPINORTEX, nbr_idx1, stride1);
    ADJ_MAT_MUL_V(A, fat, i);

    if(kernel_type == EXTERIOR_KERNEL_T){
      if(x4 == 0){
        printf("-T x = (%d, %d, %d, %d), I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }
    }else if(kparam.X4 == 12){
      if(x4 == 2){
        printf("-T x = (%d, %d, %d, %d), I = (%lf, %lf, %lf, %lf, %lf, %lf), f = %lf, %lf\n", x1, x2, x3, x4, i00_re, i00_im, i01_re, i01_im, i02_re, i02_im, fat00_re, fat22_im);
      }
    }


    o0_re -= A0_re;
    o0_im -= A0_im;
    o1_re -= A1_re;
    o1_im -= A1_im;
    o2_re -= A2_re;
    o2_im -= A2_im;
  }

  if(hasNaik){  
#ifdef MULTI_GPU
    if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[3]) || x4 >= 3)) || (kernel_type == EXTERIOR_KERNEL_T && x4 < 3))
#endif
    {
      int sp_idx_3rd_nbr = ((x4<3) ? X + kparam.X4m3*kparam.X3X2X1: X - 3*kparam.X3X2X1) >> 1;
      int long_idx = sp_idx_3rd_nbr;
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = kparam.sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx = nbr_idx3;
#endif	    
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_T){
        if (x4<3){                      
          long_idx = kparam.Vh + x4*kparam.ghostFace[3]+ space_con;
          nbr_idx3 = param.ghostOffset[3] + (Nface-3 + x4)*kparam.ghostFace[3]+ space_con;
          stride3 = Nface*kparam.ghostFace[3];
#if (DD_PREC == 2) //half precision
          norm_idx = param.ghostNormOffset[3]  + (Nface-3 + x4)*kparam.ghostFace[3]+ space_con;
#endif		    
        }
      }
#endif	    
      READ_LONG_MATRIX(LONG, LONGLINK1TEX, dir, long_idx, longlinkStride);
      READ_KS_NBR_SPINOR(T, SPINORTEX, nbr_idx3, stride3);       
      RECONSTRUCT_GAUGE_MATRIX(7, long, sp_idx_3rd_nbr, sign);    
      ADJ_MAT_MUL_V(B, long, t);    
      o0_re -= B0_re;
      o0_im -= B0_im;
      o1_re -= B1_re;
      o1_im -= B1_im;
      o2_re -= B2_re;
      o2_im -= B2_im;
    }
  } // hasNaik        
} // -T


#if (DD_DAG == 1)
{
  o0_re = - o0_re;
  o0_im = - o0_im;
  o1_re = - o1_re;
  o1_im = - o1_im;
  o2_re = - o2_re;
  o2_im = - o2_im;
}

#endif

#ifdef DSLASH_AXPY
#ifdef MULTI_GPU
if (kernel_type == INTERIOR_KERNEL){
  READ_ACCUM(ACCUMTEX, sid, kparam.sp_stride);
  o0_re = -o0_re + a*accum0.x;
  o0_im = -o0_im + a*accum0.y;
  o1_re = -o1_re + a*accum1.x;
  o1_im = -o1_im + a*accum1.y;
  o2_re = -o2_re + a*accum2.x;
  o2_im = -o2_im + a*accum2.y;
}else{
  o0_re = -o0_re;
  o0_im = -o0_im;
  o1_re = -o1_re;
  o1_im = -o1_im;
  o2_re = -o2_re;
  o2_im = -o2_im;
}
#else
READ_ACCUM(ACCUMTEX, sid, kparam.sp_stride);
o0_re = -o0_re + a*accum0.x;
o0_im = -o0_im + a*accum0.y;
o1_re = -o1_re + a*accum1.x;
o1_im = -o1_im + a*accum1.y;
o2_re = -o2_re + a*accum2.x;
o2_im = -o2_im + a*accum2.y;
#endif //MULTI_GPU
#endif // DSLASH_AXPY

#ifdef MULTI_GPU
//if (kernel_type == EXTERIOR_KERNEL_T){
if (kernel_type != INTERIOR_KERNEL){
  READ_AND_SUM_SPINOR(INTERTEX);
}
#endif


// write spinor field back to device memory
WRITE_SPINOR(out, sid, kparam.sp_stride);


// undefine to prevent warning when precision is changed
#undef spinorFloat
#undef SHARED_STRIDE

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
#undef kernel_type

#undef o0_re
#undef o0_im
#undef o1_re
#undef o1_im
#undef o2_re
#undef o2_im

#undef VOLATILE
