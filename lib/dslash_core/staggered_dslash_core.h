// *** CUDA DSLASH ***
#undef SHARED_FLOATS_PER_THREAD 
#define SHARED_FLOATS_PER_THREAD 6

//#define kernel_type param.kernel_type

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

#if (DD_RECON == 18) //18 (no) reconstruct
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
#define o00_re s[0*SHARED_STRIDE]
#define o00_im s[1*SHARED_STRIDE]
#define o01_re s[2*SHARED_STRIDE]
#define o01_im s[3*SHARED_STRIDE]
#define o02_re s[4*SHARED_STRIDE]
#define o02_im s[5*SHARED_STRIDE]


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


#if (DD_IMPROVED==1)
#define NFACE 3
#else
#define NFACE 1
#endif

#if ((DD_RECON==9 || DD_RECON==13) && DD_IMPROVED==1)
#if (DD_PREC==0) // double precision
double PHASE = 0.;
#else
float PHASE = 0.f;
#endif
#endif

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if(sid >= param.threads) return;

int za,zb; 
int x1h, x2h;
int x1,x2,x3,x4;
int af;
int x1odd,x2odd;
int X;

if(kernel_type == INTERIOR_KERNEL){
  //data order: X4 X3 X2 X1h
  za = sid / X1h;
  x1h = sid - za*X1h;
  zb = za / X2;
  x2 = za - zb*X2;
  x4 = zb / X3;
  x3 = zb - x4*X3;
  x1odd = (x2 + x3 + x4 + param.parity) & 1;
  x1 = 2*x1h + x1odd;
  X = 2*sid + x1odd;
 }else if (kernel_type == EXTERIOR_KERNEL_X){
  //data order: X1 X4 X3 X2h
  za = sid / X2h;
  x2h = sid - za*X2h;
  zb = za / X3;
  x3 = za - zb*X3;
  x1 = zb / X4;
  x4 = zb - x1*X4;
  af = (x1 >= NFACE)?(X1-2*NFACE):0;
  x1 = x1 + af;
  x2odd = (x3 + x4 + x1 + param.parity) & 1;
  x2 = 2*x2h + x2odd;
  X = x4*X3X2X1+x3*X2X1+x2*X1+x1;
  sid = X>>1;
 }else if (kernel_type == EXTERIOR_KERNEL_Y){
  //data order: X2 X4 X3 X1h
  za = sid / X1h;
  x1h = sid - za*X1h;
  zb = za / X3;
  x3 = za - zb*X3;
  x2 = zb / X4;
  x4 = zb - x2*X4;
  af = (x2 >= NFACE)?(X2-2*NFACE):0;
  x2 = x2 + af;
  x1odd = (x3 + x4 + x2 + param.parity) & 1;
  x1 = 2*x1h + x1odd;
  X = x4*X3X2X1+x3*X2X1+x2*X1+x1;
  sid = X>>1;

 }else if (kernel_type == EXTERIOR_KERNEL_Z){
  //data order: X3 X4 X2 X1h
  za = sid / X1h;
  x1h = sid - za*X1h;
  zb = za / X2;
  x2 = za - zb*X2;
  x3 = zb / X4;
  x4 = zb - x3*X4;
  af = (x3 >= NFACE)?(X3-2*NFACE):0;
  x3 = x3 + af;
  x1odd = (x2 + x4 + x3 + param.parity) & 1;
  x1 = 2*x1h + x1odd;
  X = x4*X3X2X1+x3*X2X1+x2*X1+x1;
  sid = X>>1;
 }else if (kernel_type == EXTERIOR_KERNEL_T){
  //data order: X4 X3 X2 X1h
  za = sid / X1h;
  x1h = sid - za*X1h;
  zb = za / X2;
  x2 = za - zb*X2;
  x4 = zb / X3;
  x3 = zb - x4*X3;
  af = (x4 >= NFACE)?(X4-2*NFACE):0;
  int x4_new = x4 + af;
  sid +=Vsh*(x4_new -x4);
  x4=x4_new;
  x1odd = (x2 + x3 + x4 + param.parity) & 1;
  x1 = 2*x1h + x1odd;
  X = 2*sid + x1odd;
 }

#if (DD_PREC == 0) // double precision
o00_re = o00_im = 0.0;
o01_re = o01_im = 0.0;
o02_re = o02_im = 0.0;
#else 
o00_re = o00_im = 0.f;
o01_re = o01_im = 0.f;
o02_re = o02_im = 0.f;
#endif
#if((DD_RECON == 13 || DD_RECON == 9) && DD_IMPROVED==1)
int sign = 1;
#endif

{
  //direction: +X


#if ((DD_RECON == 12 || DD_RECON == 8) && DD_IMPROVED==1)
  int sign = (x4%2 == 1) ? -1 : 1;
#endif

  int ga_idx = sid;

#ifdef MULTI_GPU
  if ( (kernel_type == INTERIOR_KERNEL && ( (!param.ghostDim[0]) || x1 < X1m1) )|| (kernel_type == EXTERIOR_KERNEL_X && x1 >= X1m1))
#endif
    {
      int sp_idx_1st_nbr = ((x1==X1m1) ? X-X1m1 : X+1) >> 1;
      READ_FAT_MATRIX(FATLINK0TEX, 0, ga_idx);
      int nbr_idx1 = sp_idx_1st_nbr;
      int stride1 = sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx1 = nbr_idx1;
#endif	    
#ifdef MULTI_GPU
      if ( (kernel_type == EXTERIOR_KERNEL_X)){
	int space_con = (x4*X3X2+x3*X2+x2)/2;	
	if (x1 >= X1m1){
	  nbr_idx1 = param.ghostOffset[0] + 3*NFACE*ghostFace[0] +(x1-X1m1)*ghostFace[0]+ space_con;
	  stride1 = NFACE*ghostFace[0];
#if (DD_PREC == 2) //half precision
	  norm_idx1 = param.ghostNormOffset[0] + NFACE*ghostFace[0] + (x1-X1m1)*ghostFace[0]+ space_con;
#endif		    
	}
      } 
#endif
      READ_1ST_NBR_SPINOR(SPINORTEX, nbr_idx1, stride1);
      MAT_MUL_V(A, fat, i);    
      o00_re += A0_re;
      o00_im += A0_im;
      o01_re += A1_re;
      o01_im += A1_im;
      o02_re += A2_re;
      o02_im += A2_im;
      }
    
#if (DD_IMPROVED==1)
#ifdef MULTI_GPU
  if ( (kernel_type == INTERIOR_KERNEL && ( (!param.ghostDim[0]) || x1 < X1m3) )|| (kernel_type == EXTERIOR_KERNEL_X && x1 >= X1m3))
#endif
    {
      int sp_idx_3rd_nbr = ((x1 >= X1m3) ? X -X1m3 : X+3) >> 1;
      READ_LONG_MATRIX(LONGLINK0TEX, 0, ga_idx);        
      READ_LONG_PHASE(LONGPHASE0TEX, 0, ga_idx);
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = sp_stride;    
#if (DD_PREC == 2) //half precision
      int norm_idx3 = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
      if ( (kernel_type == EXTERIOR_KERNEL_X)){
	int space_con = (x4*X3X2+x3*X2+x2)/2;		
	if (x1  >= X1m3){
	  nbr_idx3 = param.ghostOffset[0] + 3*NFACE*ghostFace[0] +(x1-X1m3)*ghostFace[0]+ space_con;
	  stride3 = NFACE*ghostFace[0];
#if (DD_PREC == 2) //half precision
	  norm_idx3 = param.ghostNormOffset[0] + NFACE*ghostFace[0] + (x1-X1m3)*ghostFace[0]+ space_con;
#endif	
	}
      }
#endif
      READ_3RD_NBR_SPINOR(SPINORTEX, nbr_idx3, stride3);
	
      RECONSTRUCT_GAUGE_MATRIX(0, long, ga_idx, sign);
      MAT_MUL_V(B, long, t);        
      o00_re += B0_re;
      o00_im += B0_im;
      o01_re += B1_re;
      o01_im += B1_im;
      o02_re += B2_re;
      o02_im += B2_im;  
    }
    
#endif

} // direction: +X


{
  // direction: -X
#if ((DD_RECON == 12 || DD_RECON == 8) && DD_IMPROVED==1)
  int sign = (x4%2 == 1) ? -1 : 1;
#endif
  int dir =1;

#ifdef MULTI_GPU
  int space_con = (x4*X3X2 + x3*X2+ x2) >>1;
  if ( (kernel_type == INTERIOR_KERNEL && ( (!param.ghostDim[0]) || x1 >= 1)) || (kernel_type == EXTERIOR_KERNEL_X && x1 < 1))
#endif
    {
      int sp_idx_1st_nbr = ((x1==0) ? X+X1m1 : X-1) >> 1;
      int fat_idx = sp_idx_1st_nbr;
#ifdef MULTI_GPU
      if ((x1 -1) < 0){
	fat_idx = Vh + space_con;
      }
#endif
      READ_FAT_MATRIX(FATLINK1TEX, dir, fat_idx);
      int nbr_idx1 = sp_idx_1st_nbr;
      int stride1 = sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx1 = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_X){
	if (x1 - 1 < 0){
	  nbr_idx1 = param.ghostOffset[0] + (x1+NFACE-1)*ghostFace[0]+ space_con;
	  stride1 = NFACE*ghostFace[0];
#if (DD_PREC == 2) //half precision
	  norm_idx1 = param.ghostNormOffset[0]  + (x1+NFACE-1)*ghostFace[0]+ space_con;
#endif	
	}        
      }
#endif
      READ_1ST_NBR_SPINOR(SPINORTEX, nbr_idx1, stride1);
      ADJ_MAT_MUL_V(A, fat, i);       
      o00_re -= A0_re;
      o00_im -= A0_im;
      o01_re -= A1_re;
      o01_im -= A1_im;
      o02_re -= A2_re;
      o02_im -= A2_im;
    }

#if (DD_IMPROVED==1)

#ifdef MULTI_GPU    
  if ( (kernel_type == INTERIOR_KERNEL && ( (!param.ghostDim[0]) || x1 >= 3)) || (kernel_type == EXTERIOR_KERNEL_X && x1 < 3))
#endif
    {
      int sp_idx_3rd_nbr = ((x1<3) ? X + X1m3: X -3)>>1; 
      int long_idx = sp_idx_3rd_nbr;
#ifdef MULTI_GPU
      if ((x1 -3) < 0){
	long_idx =Vh + x1*X4X3X2h + space_con;
      }    
#endif
      READ_LONG_MATRIX(LONGLINK1TEX, dir, long_idx); 		
      READ_LONG_PHASE(LONGPHASE1TEX, dir, long_idx); 		
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx3 = nbr_idx3;
#endif	     
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_X){
	if (x1 - 3 < 0){
	  nbr_idx3 = param.ghostOffset[0] + x1*ghostFace[0]+ space_con;
	  stride3 = NFACE*ghostFace[0];
#if (DD_PREC == 2) //half precision
	  norm_idx3 = param.ghostNormOffset[0]  + x1*ghostFace[0]+ space_con;
#endif
	}
      }
#endif

      READ_3RD_NBR_SPINOR(SPINORTEX, nbr_idx3, stride3);  
      RECONSTRUCT_GAUGE_MATRIX(1, long, sp_idx_3rd_nbr, sign);
      ADJ_MAT_MUL_V(B, long, t);    
      o00_re -= B0_re;
      o00_im -= B0_im;
      o01_re -= B1_re;
      o01_im -= B1_im;
      o02_re -= B2_re;
      o02_im -= B2_im;  
    }
#endif
    
}



{
  //direction: +Y
#if ((DD_RECON == 12 || DD_RECON == 8) && DD_IMPROVED==1)
  int sign = ((x4+x1)%2 == 1) ? -1 : 1;
#endif
   
  int ga_idx = sid;

#ifdef MULTI_GPU
  int space_con = (x4*X3X1+x3*X1+x1)/2;
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[1]) || x2 < X2m1))|| (kernel_type == EXTERIOR_KERNEL_Y && x2 >= X2m1))
#endif
    {
      int sp_idx_1st_nbr = ((x2==X2m1) ? X-X2X1mX1 : X+X1) >> 1;
      READ_FAT_MATRIX(FATLINK0TEX, 2, ga_idx);
      int nbr_idx1 = sp_idx_1st_nbr;
      int stride1 = sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx1 = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_Y){	    
	if (x2 >= X2m1){
	  nbr_idx1 = param.ghostOffset[1] + 3*NFACE*ghostFace[1] +(x2-X2m1)*ghostFace[1]+ space_con;
	  stride1 = NFACE*ghostFace[1];
#if (DD_PREC == 2) //half precision
	  norm_idx1 = param.ghostNormOffset[1] + NFACE*ghostFace[1] + (x2-X2m1)*ghostFace[1]+ space_con;
#endif		    
	}      
      }
#endif 
      READ_1ST_NBR_SPINOR(SPINORTEX, nbr_idx1, stride1);
      MAT_MUL_V(A, fat, i);
      o00_re += A0_re;
      o00_im += A0_im;
      o01_re += A1_re;
      o01_im += A1_im;
      o02_re += A2_re;
      o02_im += A2_im;
    }
    
#if (DD_IMPROVED==1)

#ifdef MULTI_GPU
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[1]) || x2 < X2m3))|| (kernel_type == EXTERIOR_KERNEL_Y && x2 >= X2m3))    
#endif
    {
      int sp_idx_3rd_nbr = ((x2 >= X2m3 ) ? X-X2m3*X1 : X+3*X1) >> 1;    
      READ_LONG_MATRIX(LONGLINK0TEX, 2, ga_idx);
      READ_LONG_PHASE(LONGPHASE0TEX, 2, ga_idx);
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = sp_stride;        
#if (DD_PREC == 2) //half precision
      int norm_idx3 = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_Y){
	if (x2>= X2m3){
	  nbr_idx3 = param.ghostOffset[1] + 3*NFACE*ghostFace[1] +(x2-X2m3)*ghostFace[1]+ space_con;
	  stride3 = NFACE*ghostFace[1];
#if (DD_PREC == 2) //half precision
	  norm_idx3 = param.ghostNormOffset[1] + NFACE*ghostFace[1] + (x2-X2m3)*ghostFace[1]+ space_con;
#endif		    
	}
      }
#endif    
      READ_3RD_NBR_SPINOR(SPINORTEX, nbr_idx3, stride3);

      RECONSTRUCT_GAUGE_MATRIX(2, long, ga_idx, sign);
      MAT_MUL_V(B, long, t);            
      o00_re += B0_re;
      o00_im += B0_im;
      o01_re += B1_re;
      o01_im += B1_im;
      o02_re += B2_re;
      o02_im += B2_im;  
    }
#endif
}

{
  //direction: -Y

#if ((DD_RECON == 12 || DD_RECON == 8) && DD_IMPROVED==1)
  int sign = ((x4+x1)%2 == 1) ? -1 : 1;
#endif

  int dir=3;
#ifdef MULTI_GPU
  int space_con = (x4*X3X1 + x3*X1+ x1) >>1;    
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[1]) || x2 >= 1)) || (kernel_type == EXTERIOR_KERNEL_Y && x2 < 1))
#endif
    {
      int sp_idx_1st_nbr = ((x2==0)    ? X+X2X1mX1 : X-X1) >> 1;
      int fat_idx=sp_idx_1st_nbr;
#ifdef MULTI_GPU
      if ((x2 -1) < 0){
	fat_idx = Vh + space_con;
      }    
#endif
      READ_FAT_MATRIX(FATLINK1TEX, dir, fat_idx);
      int nbr_idx1 = sp_idx_1st_nbr;
      int stride1 = sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx1 = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_Y){
	if (x2 - 1 < 0){
	  nbr_idx1 = param.ghostOffset[1] + (x2+NFACE-1)*ghostFace[1]+ space_con;
	  stride1 = NFACE*ghostFace[1];
#if (DD_PREC == 2) //half precision
	  norm_idx1 = param.ghostNormOffset[1]  + (x2+NFACE-1)*ghostFace[1]+ space_con;
#endif	
	}              
      }
#endif
      READ_1ST_NBR_SPINOR(SPINORTEX, nbr_idx1, stride1);
      ADJ_MAT_MUL_V(A, fat, i);
      o00_re -= A0_re;
      o00_im -= A0_im;
      o01_re -= A1_re;
      o01_im -= A1_im;
      o02_re -= A2_re;
      o02_im -= A2_im;
    }
    
#if (DD_IMPROVED==1)

#ifdef MULTI_GPU
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[1]) || x2 >= 3)) || (kernel_type == EXTERIOR_KERNEL_Y && x2 < 3))
#endif
    {
      int sp_idx_3rd_nbr = ((x2 < 3) ? X + X2m3*X1: X -3*X1 )>> 1; 
      int long_idx = sp_idx_3rd_nbr;
#ifdef MULTI_GPU
      if ((x2-3) < 0){
	long_idx = Vh+ x2*X4X3X1h + space_con;
      }    
#endif
      READ_LONG_MATRIX(LONGLINK1TEX, dir, long_idx); 
      READ_LONG_PHASE(LONGPHASE1TEX, dir, long_idx); 
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = sp_stride;    
#if (DD_PREC == 2) //half precision
      int norm_idx3 = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_Y){
	if (x2 - 3 < 0){
	  nbr_idx3 = param.ghostOffset[1] + x2*ghostFace[1]+ space_con;
	  stride3 = NFACE*ghostFace[1];
#if (DD_PREC == 2) //half precision
	  norm_idx3 = param.ghostNormOffset[1]  + x2*ghostFace[1]+ space_con;
#endif
	}
      }
#endif
      READ_3RD_NBR_SPINOR(SPINORTEX, nbr_idx3, stride3);

      RECONSTRUCT_GAUGE_MATRIX(3, long, sp_idx_3rd_nbr,sign);	    
      ADJ_MAT_MUL_V(B, long, t);    
      o00_re -= B0_re;
      o00_im -= B0_im;
      o01_re -= B1_re;
      o01_im -= B1_im;
      o02_re -= B2_re;
      o02_im -= B2_im;  
    }    
#endif
}

{
  //direction: +Z

#if ((DD_RECON == 12 || DD_RECON == 8) && DD_IMPROVED==1)
  int sign = ((x4+x1+x2)%2 == 1) ? -1 : 1;
#endif

  int ga_idx = sid;

#ifdef MULTI_GPU
  int space_con = (x4*X2X1+x2*X1+x1)/2;
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[2]) || x3 < X3m1))|| (kernel_type == EXTERIOR_KERNEL_Z && x3 >= X3m1))
#endif
    {
      int sp_idx_1st_nbr = ((x3==X3m1) ? X-X3X2X1mX2X1 : X+X2X1) >> 1;
      READ_FAT_MATRIX(FATLINK0TEX, 4, ga_idx);
      int nbr_idx1 = sp_idx_1st_nbr;
      int stride1 = sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx1 = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_Z){	
	if (x3 >= X3m1){
	  nbr_idx1 = param.ghostOffset[2] + 3*NFACE*ghostFace[2] +(x3-X3m1)*ghostFace[2]+ space_con;
	  stride1 = NFACE*ghostFace[2];	    
#if (DD_PREC == 2) //half precision
	  norm_idx1 = param.ghostNormOffset[2] + NFACE*ghostFace[2] + (x3-X3m1)*ghostFace[2]+ space_con;
#endif		
	}      
      }
#endif
      READ_1ST_NBR_SPINOR(SPINORTEX, nbr_idx1, stride1);
      MAT_MUL_V(A, fat, i);	 
      o00_re += A0_re;
      o00_im += A0_im;
      o01_re += A1_re;
      o01_im += A1_im;
      o02_re += A2_re;
      o02_im += A2_im;
    }

#if (DD_IMPROVED==1)

#ifdef MULTI_GPU
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[2]) || x3 < X3m3))|| (kernel_type == EXTERIOR_KERNEL_Z && x3 >= X3m3))
#endif
    {
      int sp_idx_3rd_nbr = ((x3>= X3m3)? X -X3m3*X2X1: X + 3*X2X1)>> 1;    
      READ_LONG_MATRIX(LONGLINK0TEX, 4, ga_idx);
      READ_LONG_PHASE(LONGPHASE0TEX, 4, ga_idx);
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx3 = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_Z){
	if (x3 >= X3m3){
	  nbr_idx3 = param.ghostOffset[2] + 3*NFACE*ghostFace[2] +(x3-X3m3)*ghostFace[2]+ space_con;
	  stride3 = NFACE*ghostFace[2];
#if (DD_PREC == 2) //half precision
	  norm_idx3 = param.ghostNormOffset[2] + NFACE*ghostFace[2] + (x3-X3m3)*ghostFace[2]+ space_con;
#endif
	}
      }
#endif
      READ_3RD_NBR_SPINOR(SPINORTEX, nbr_idx3, stride3);

      RECONSTRUCT_GAUGE_MATRIX(4, long, ga_idx, sign);    
      MAT_MUL_V(B, long, t);        
      o00_re += B0_re;
      o00_im += B0_im;
      o01_re += B1_re;
      o01_im += B1_im;
      o02_re += B2_re;
      o02_im += B2_im;      
    }
#endif
 
}

{
  //direction: -Z

#if ((DD_RECON == 12 || DD_RECON == 8) && DD_IMPROVED==1)
  int sign = ((x4+x1+x2)%2 == 1) ? -1 : 1;
#endif

  int dir = 5;

#ifdef MULTI_GPU
  int space_con = (x4*X2X1 + x2*X1+ x1) >>1;    
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[2]) || x3 >= 1)) || (kernel_type == EXTERIOR_KERNEL_Z && x3 < 1))
#endif
    {
      int sp_idx_1st_nbr = ((x3==0)    ? X+X3X2X1mX2X1 : X-X2X1) >> 1;
      int fat_idx = sp_idx_1st_nbr;
#ifdef MULTI_GPU
      if ((x3 -1) < 0){
	fat_idx = Vh + space_con;
      }    
#endif
      READ_FAT_MATRIX(FATLINK1TEX, dir, fat_idx);
      int nbr_idx1 = sp_idx_1st_nbr;
      int stride1 = sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx1 = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_Z){
	if (x3 - 1 < 0){
	  nbr_idx1 = param.ghostOffset[2] + (x3+NFACE-1)*ghostFace[2]+ space_con;
	  stride1 = NFACE*ghostFace[2];
#if (DD_PREC == 2) //half precision
	  norm_idx1 = param.ghostNormOffset[2]  + (x3+NFACE-1)*ghostFace[2]+ space_con;
#endif			    
	}        
      }
#endif
      READ_1ST_NBR_SPINOR(SPINORTEX, nbr_idx1, stride1);
      ADJ_MAT_MUL_V(A, fat, i);
      o00_re -= A0_re;
      o00_im -= A0_im;
      o01_re -= A1_re;
      o01_im -= A1_im;
      o02_re -= A2_re;
      o02_im -= A2_im;
    }
    
#if (DD_IMPROVED==1)

#ifdef MULTI_GPU
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[2]) || x3 >= 3)) || (kernel_type == EXTERIOR_KERNEL_Z && x3 < 3))
#endif
    {
      int sp_idx_3rd_nbr = ((x3 <3) ? X + X3m3*X2X1: X - 3*X2X1)>>1;
      int long_idx = sp_idx_3rd_nbr;
#ifdef MULTI_GPU
      if ((x3 -3) < 0){
	long_idx = Vh + x3*X4X2X1h + space_con;
      }    
#endif
      READ_LONG_MATRIX(LONGLINK1TEX, dir, long_idx);         
      READ_LONG_PHASE(LONGPHASE1TEX, dir, long_idx);         
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = sp_stride;    
#if (DD_PREC == 2) //half precision
      int norm_idx3 = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_Z){
	if (x3 - 3 < 0){
	  nbr_idx3 = param.ghostOffset[2] + x3*ghostFace[2]+ space_con;
	  stride3 = NFACE*ghostFace[2];
#if (DD_PREC == 2) //half precision
	  norm_idx3 = param.ghostNormOffset[2]  + x3*ghostFace[2]+ space_con;
#endif			    
	}
      }
#endif
      READ_3RD_NBR_SPINOR(SPINORTEX, nbr_idx3, stride3);

      RECONSTRUCT_GAUGE_MATRIX(5, long, sp_idx_3rd_nbr,sign);
      ADJ_MAT_MUL_V(B, long, t);    	    
      o00_re -= B0_re;
      o00_im -= B0_im;
      o01_re -= B1_re;
      o01_im -= B1_im;
      o02_re -= B2_re;
      o02_im -= B2_im;    
    }
#endif
}

{
  //direction: +T
#if ((DD_RECON == 12 || DD_RECON == 8) && DD_IMPROVED==1)
  int sign = (x4 >= (X4-3)) ? -1 : 1;
#endif

  int ga_idx = sid;

#ifdef MULTI_GPU
  int space_con = (x3*X2X1+x2*X1+x1)/2;
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[3]) || x4 < X4m1))|| (kernel_type == EXTERIOR_KERNEL_T && x4 >= X4m1))
#endif
    {    
      int sp_idx_1st_nbr = ((x4==X4m1) ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1;
      READ_FAT_MATRIX(FATLINK0TEX, 6, ga_idx);
      int nbr_idx1 = sp_idx_1st_nbr;
      int stride1 = sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx1 = nbr_idx1;
#endif
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_T){      
	if (x4 >= X4m1){
	  nbr_idx1 = param.ghostOffset[3] + 3*NFACE*ghostFace[3] +(x4-X4m1)*ghostFace[3]+ space_con;
	  stride1 = NFACE*ghostFace[3];
#if (DD_PREC == 2) //half precision
	  norm_idx1 = param.ghostNormOffset[3] + NFACE*ghostFace[3] + (x4-X4m1)*ghostFace[3]+ space_con;
#endif
	}
      }
#endif
      READ_1ST_NBR_SPINOR(SPINORTEX, nbr_idx1, stride1);    
      MAT_MUL_V(A, fat, i);
      o00_re += A0_re;
      o00_im += A0_im;
      o01_re += A1_re;
      o01_im += A1_im;
      o02_re += A2_re;
      o02_im += A2_im;
    }

    
#if (DD_IMPROVED==1)

#ifdef MULTI_GPU
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[3]) || x4 < X4m3))|| (kernel_type == EXTERIOR_KERNEL_T && x4 >= X4m3))
#endif
    {
      int sp_idx_3rd_nbr = ((x4>=X4m3)? X -X4m3*X3X2X1 : X + 3*X3X2X1)>> 1;     
      READ_LONG_MATRIX(LONGLINK0TEX, 6, ga_idx);    
      READ_LONG_PHASE(LONGPHASE0TEX, 6, ga_idx);    
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = sp_stride;    
#if (DD_PREC == 2) //half precision
      int norm_idx3 = nbr_idx3;
#endif
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_T){
	if (x4  >= X4m3){
	  nbr_idx3 = param.ghostOffset[3] + 3*NFACE*ghostFace[3] +(x4-X4m3)*ghostFace[3]+ space_con;
	  stride3 = NFACE*ghostFace[3];
#if (DD_PREC == 2) //half precision
	  norm_idx3 = param.ghostNormOffset[3] + NFACE*ghostFace[3] + (x4-X4m3)*ghostFace[3]+ space_con;
#endif
	}
      }
#endif
	    
      READ_3RD_NBR_SPINOR(SPINORTEX, nbr_idx3, stride3); 

      RECONSTRUCT_GAUGE_MATRIX(6, long, ga_idx, sign);
      MAT_MUL_V(B, long, t);    
      o00_re += B0_re;
      o00_im += B0_im;
      o01_re += B1_re;
      o01_im += B1_im;
      o02_re += B2_re;
      o02_im += B2_im;      
    }
#endif
}

{
  //direction: -T
#if ((DD_RECON == 12 || DD_RECON == 8) && DD_IMPROVED==1)
  int sign = ( ((x4+X4m3)%X4)>= X4m3 ) ? -1 : 1;
#endif
    
  int dir = 7;

#ifdef MULTI_GPU
  int space_con = (x3*X2X1+x2*X1+x1)/2;
  if ((kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[3]) || x4 >= 1)) || (kernel_type == EXTERIOR_KERNEL_T && x4 < 1))
#endif
    {
      int sp_idx_1st_nbr = ((x4==0)    ? X+X4X3X2X1mX3X2X1 : X-X3X2X1) >> 1;
      int fat_idx = sp_idx_1st_nbr;    
      int nbr_idx1 = sp_idx_1st_nbr;
      int stride1 = sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx1 = nbr_idx1;
#endif
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_T){
	if ( (x4 - 1) < 0){
	  fat_idx = Vh + space_con;
	}
		
	if (x4 - 1 < 0){
	  nbr_idx1 = param.ghostOffset[3] + (x4+NFACE-1)*ghostFace[3]+ space_con;
	  stride1 = NFACE*ghostFace[3];
#if (DD_PREC == 2) //half precision
	  norm_idx1 = param.ghostNormOffset[3]  + (x4+NFACE-1)*ghostFace[3]+ space_con;
#endif		    
	}        	
      }
#endif
      READ_FAT_MATRIX(FATLINK1TEX, dir, fat_idx);
      READ_1ST_NBR_SPINOR(SPINORTEX, nbr_idx1, stride1);
      ADJ_MAT_MUL_V(A, fat, i);
      o00_re -= A0_re;
      o00_im -= A0_im;
      o01_re -= A1_re;
      o01_im -= A1_im;
      o02_re -= A2_re;
      o02_im -= A2_im;
    }
    
#if (DD_IMPROVED==1)

#ifdef MULTI_GPU
  if ( (kernel_type == INTERIOR_KERNEL && ((!param.ghostDim[3]) || x4 >= 3)) || (kernel_type == EXTERIOR_KERNEL_T && x4 < 3))
#endif
    {
      int sp_idx_3rd_nbr = ((x4<3) ? X + X4m3*X3X2X1: X - 3*X3X2X1) >> 1;
      int long_idx = sp_idx_3rd_nbr;
      int nbr_idx3 = sp_idx_3rd_nbr;
      int stride3 = sp_stride;
#if (DD_PREC == 2) //half precision
      int norm_idx3 = nbr_idx3;
#endif	    
#ifdef MULTI_GPU
      if (kernel_type == EXTERIOR_KERNEL_T){
	if ( (x4 - 3) < 0){
	  long_idx = Vh + x4*ghostFace[3]+ space_con;
	}	
	if (x4 - 3 < 0){
	  nbr_idx3 = param.ghostOffset[3] + x4*ghostFace[3]+ space_con;
	  stride3 = NFACE*ghostFace[3];
#if (DD_PREC == 2) //half precision
	  norm_idx3 = param.ghostNormOffset[3]  + x4*ghostFace[3]+ space_con;
#endif		    
	}
      }
#endif	    
      READ_LONG_MATRIX(LONGLINK1TEX, dir, long_idx);
      READ_LONG_PHASE(LONGPHASE1TEX, dir, long_idx);
      READ_3RD_NBR_SPINOR(SPINORTEX, nbr_idx3, stride3);       

      RECONSTRUCT_GAUGE_MATRIX(7, long, sp_idx_3rd_nbr, sign);    
      ADJ_MAT_MUL_V(B, long, t);    
      o00_re -= B0_re;
      o00_im -= B0_im;
      o01_re -= B1_re;
      o01_im -= B1_im;
      o02_re -= B2_re;
      o02_im -= B2_im;
    }        
#endif
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

#ifdef DSLASH_AXPY
#ifdef MULTI_GPU
if (kernel_type == INTERIOR_KERNEL){
  READ_ACCUM(ACCUMTEX);
  o00_re = -o00_re + a*accum0.x;
  o00_im = -o00_im + a*accum0.y;
  o01_re = -o01_re + a*accum1.x;
  o01_im = -o01_im + a*accum1.y;
  o02_re = -o02_re + a*accum2.x;
  o02_im = -o02_im + a*accum2.y;
 }else{
  o00_re = -o00_re;
  o00_im = -o00_im;
  o01_re = -o01_re;
  o01_im = -o01_im;
  o02_re = -o02_re;
  o02_im = -o02_im;
 }
#else
READ_ACCUM(ACCUMTEX);
o00_re = -o00_re + a*accum0.x;
o00_im = -o00_im + a*accum0.y;
o01_re = -o01_re + a*accum1.x;
o01_im = -o01_im + a*accum1.y;
o02_re = -o02_re + a*accum2.x;
o02_im = -o02_im + a*accum2.y;
#endif //MULTI_GPU
#endif // DSLASH_AXPY

#ifdef MULTI_GPU
//if (kernel_type == EXTERIOR_KERNEL_T){
if (kernel_type != INTERIOR_KERNEL){
  READ_AND_SUM_SPINOR(INTERTEX);
 }
#endif


// write spinor field back to device memory
WRITE_SPINOR(out);


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

#undef o00_re
#undef o00_im
#undef o01_re
#undef o01_im
#undef o02_re
#undef o02_im

#undef NFACE

#undef VOLATILE
