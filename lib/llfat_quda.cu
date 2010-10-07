#include <stdio.h>

#include <quda_internal.h>
#include <llfat_quda.h>
#include <read_gauge.h>
#include <gauge_quda.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define WRITE_FAT_MATRIX(gauge, dir, idx)do {		\
    gauge[idx + dir*Vhx9] = FAT0;			\
    gauge[idx + dir*Vhx9 + Vh  ] = FAT1;		\
    gauge[idx + dir*Vhx9 + Vhx2] = FAT2;		\
    gauge[idx + dir*Vhx9 + Vhx3] = FAT3;		\
    gauge[idx + dir*Vhx9 + Vhx4] = FAT4;		\
    gauge[idx + dir*Vhx9 + Vhx5] = FAT5;		\
    gauge[idx + dir*Vhx9 + Vhx6] = FAT6;		\
    gauge[idx + dir*Vhx9 + Vhx7] = FAT7;		\
    gauge[idx + dir*Vhx9 + Vhx8] = FAT8;} while(0)			


#define WRITE_STAPLE_MATRIX(gauge, idx)		\
  gauge[idx] = STAPLE0;				\
  gauge[idx + Vh] = STAPLE1;			\
  gauge[idx + Vhx2] = STAPLE2;			\
  gauge[idx + Vhx3] = STAPLE3;			\
  gauge[idx + Vhx4] = STAPLE4;			\
  gauge[idx + Vhx5] = STAPLE5;			\
  gauge[idx + Vhx6] = STAPLE6;			\
  gauge[idx + Vhx7] = STAPLE7;			\
  gauge[idx + Vhx8] = STAPLE8;					
    

#define SCALAR_MULT_SU3_MATRIX(a, b, c) \
  c##00_re = a*b##00_re;		\
  c##00_im = a*b##00_im;		\
  c##01_re = a*b##01_re;		\
  c##01_im = a*b##01_im;		\
  c##02_re = a*b##02_re;		\
  c##02_im = a*b##02_im;		\
  c##10_re = a*b##10_re;		\
  c##10_im = a*b##10_im;		\
  c##11_re = a*b##11_re;		\
  c##11_im = a*b##11_im;		\
  c##12_re = a*b##12_re;		\
  c##12_im = a*b##12_im;		\
  c##20_re = a*b##20_re;		\
  c##20_im = a*b##20_im;		\
  c##21_re = a*b##21_re;		\
  c##21_im = a*b##21_im;		\
  c##22_re = a*b##22_re;		\
  c##22_im = a*b##22_im;		\
    

#define LOAD_MATRIX_18_SINGLE(gauge, dir, idx, var)			\
  float2 var##0 = gauge[idx + dir*Vhx9];				\
  float2 var##1 = gauge[idx + dir*Vhx9 + Vh];				\
  float2 var##2 = gauge[idx + dir*Vhx9 + Vhx2];				\
  float2 var##3 = gauge[idx + dir*Vhx9 + Vhx3];				\
  float2 var##4 = gauge[idx + dir*Vhx9 + Vhx4];				\
  float2 var##5 = gauge[idx + dir*Vhx9 + Vhx5];				\
  float2 var##6 = gauge[idx + dir*Vhx9 + Vhx6];				\
  float2 var##7 = gauge[idx + dir*Vhx9 + Vhx7];				\
  float2 var##8 = gauge[idx + dir*Vhx9 + Vhx8];

#define LOAD_MATRIX_18_SINGLE_TEX(gauge, dir, idx, var)			\
  float2 var##0 = tex1Dfetch(gauge, idx + dir*Vhx9);			\
  float2 var##1 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vh);		\
  float2 var##2 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx2);		\
  float2 var##3 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx3);		\
  float2 var##4 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx4);		\
  float2 var##5 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx5);		\
  float2 var##6 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx6);		\
  float2 var##7 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx7);		\
  float2 var##8 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx8); 

#define LOAD_MATRIX_18_DOUBLE(gauge, dir, idx, var)			\
  double2 var##0 = gauge[idx + dir*Vhx9];				\
  double2 var##1 = gauge[idx + dir*Vhx9 + Vh];				\
  double2 var##2 = gauge[idx + dir*Vhx9 + Vhx2];				\
  double2 var##3 = gauge[idx + dir*Vhx9 + Vhx3];				\
  double2 var##4 = gauge[idx + dir*Vhx9 + Vhx4];				\
  double2 var##5 = gauge[idx + dir*Vhx9 + Vhx5];				\
  double2 var##6 = gauge[idx + dir*Vhx9 + Vhx6];				\
  double2 var##7 = gauge[idx + dir*Vhx9 + Vhx7];				\
  double2 var##8 = gauge[idx + dir*Vhx9 + Vhx8];

#define LOAD_MATRIX_18_DOUBLE_TEX(gauge, dir, idx, var)			\
  double2 var##0 = fetch_double2(gauge, idx + dir*Vhx9);			\
  double2 var##1 = fetch_double2(gauge, idx + dir*Vhx9 + Vh);		\
  double2 var##2 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx2);		\
  double2 var##3 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx3);		\
  double2 var##4 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx4);		\
  double2 var##5 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx5);		\
  double2 var##6 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx6);		\
  double2 var##7 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx7);		\
  double2 var##8 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx8); 


#define SITE_MATRIX_LOAD_TEX 0
#define MULINK_LOAD_TEX 0
#define FATLINK_LOAD_TEX 1


#define LOAD_MATRIX_12_SINGLE_DECLARE(gauge, dir, idx, var)		\
  float4 var##0 = gauge[idx + dir*Vhx3];				\
  float4 var##1 = gauge[idx + dir*Vhx3 + Vh];				\
  float4 var##2 = gauge[idx + dir*Vhx3 + Vhx2];				\
  float4 var##3, var##4;

#define LOAD_MATRIX_12_SINGLE_TEX_DECLARE(gauge, dir, idx, var)		\
  float4 var##0 = tex1Dfetch(gauge, idx + dir* Vhx3);			\
  float4 var##1 = tex1Dfetch(gauge, idx + dir*Vhx3 + Vh);		\
  float4 var##2 = tex1Dfetch(gauge, idx + dir*Vhx3 + Vhx2);		\
  float4 var##3, var##4;

#define LOAD_MATRIX_18_SINGLE_DECLARE(gauge, dir, idx, var)		\
  float2 var##0 = gauge[idx + dir*Vhx9];				\
  float2 var##1 = gauge[idx + dir*Vhx9 + Vh];				\
  float2 var##2 = gauge[idx + dir*Vhx9 + Vhx2];				\
  float2 var##3 = gauge[idx + dir*Vhx9 + Vhx3];				\
  float2 var##4 = gauge[idx + dir*Vhx9 + Vhx4];				\
  float2 var##5 = gauge[idx + dir*Vhx9 + Vhx5];				\
  float2 var##6 = gauge[idx + dir*Vhx9 + Vhx6];				\
  float2 var##7 = gauge[idx + dir*Vhx9 + Vhx7];				\
  float2 var##8 = gauge[idx + dir*Vhx9 + Vhx8];				


#define LOAD_MATRIX_18_SINGLE_TEX_DECLARE(gauge, dir, idx, var)		\
  float2 var##0 = tex1Dfetch(gauge, idx + dir*Vhx9);			\
  float2 var##1 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vh);		\
  float2 var##2 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx2);		\
  float2 var##3 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx3);		\
  float2 var##4 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx4);		\
  float2 var##5 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx5);		\
  float2 var##6 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx6);		\
  float2 var##7 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx7);		\
  float2 var##8 = tex1Dfetch(gauge, idx + dir*Vhx9 + Vhx8);		



#define LOAD_MATRIX_18_DOUBLE_DECLARE(gauge, dir, idx, var)		\
  double2 var##0 = gauge[idx + dir*Vhx9];				\
  double2 var##1 = gauge[idx + dir*Vhx9 + Vh];				\
  double2 var##2 = gauge[idx + dir*Vhx9 + Vhx2];			\
  double2 var##3 = gauge[idx + dir*Vhx9 + Vhx3];			\
  double2 var##4 = gauge[idx + dir*Vhx9 + Vhx4];			\
  double2 var##5 = gauge[idx + dir*Vhx9 + Vhx5];			\
  double2 var##6 = gauge[idx + dir*Vhx9 + Vhx6];			\
  double2 var##7 = gauge[idx + dir*Vhx9 + Vhx7];			\
  double2 var##8 = gauge[idx + dir*Vhx9 + Vhx8];				


#define LOAD_MATRIX_18_DOUBLE_TEX_DECLARE(gauge, dir, idx, var)		\
  double2 var##0 = fetch_double2(gauge, idx + dir*Vhx9);		\
  double2 var##1 = fetch_double2(gauge, idx + dir*Vhx9 + Vh);		\
  double2 var##2 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx2);		\
  double2 var##3 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx3);		\
  double2 var##4 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx4);		\
  double2 var##5 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx5);		\
  double2 var##6 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx6);		\
  double2 var##7 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx7);		\
  double2 var##8 = fetch_double2(gauge, idx + dir*Vhx9 + Vhx8);		


#define LOAD_MATRIX_12_DOUBLE_DECLARE(gauge, dir, idx, var)		\
  double2 var##0 = gauge[idx + dir*Vhx6];				\
  double2 var##1 = gauge[idx + dir*Vhx6 + Vh];				\
  double2 var##2 = gauge[idx + dir*Vhx6 + Vhx2];			\
  double2 var##3 = gauge[idx + dir*Vhx6 + Vhx3];			\
  double2 var##4 = gauge[idx + dir*Vhx6 + Vhx4];			\
  double2 var##5 = gauge[idx + dir*Vhx6 + Vhx5];			\
  double2 var##6, var##7, var##8;


#define LOAD_MATRIX_12_DOUBLE_TEX_DECLARE(gauge, dir, idx, var)		\
  double2 var##0 = fetch_double2(gauge, idx + dir*Vhx6);		\
  double2 var##1 = fetch_double2(gauge, idx + dir*Vhx6 + Vh);		\
  double2 var##2 = fetch_double2(gauge, idx + dir*Vhx6 + Vhx2);		\
  double2 var##3 = fetch_double2(gauge, idx + dir*Vhx6 + Vhx3);		\
  double2 var##4 = fetch_double2(gauge, idx + dir*Vhx6 + Vhx4);		\
  double2 var##5 = fetch_double2(gauge, idx + dir*Vhx6 + Vhx5);		\
  double2 var##6, var##7, var##8;

#define LLFAT_ADD_SU3_MATRIX(ma, mb, mc)	\
  mc##00_re = ma##00_re + mb##00_re;		\
  mc##00_im = ma##00_im + mb##00_im;		\
  mc##01_re = ma##01_re + mb##01_re;		\
  mc##01_im = ma##01_im + mb##01_im;		\
  mc##02_re = ma##02_re + mb##02_re;		\
  mc##02_im = ma##02_im + mb##02_im;		\
  mc##10_re = ma##10_re + mb##10_re;		\
  mc##10_im = ma##10_im + mb##10_im;		\
  mc##11_re = ma##11_re + mb##11_re;		\
  mc##11_im = ma##11_im + mb##11_im;		\
  mc##12_re = ma##12_re + mb##12_re;		\
  mc##12_im = ma##12_im + mb##12_im;		\
  mc##20_re = ma##20_re + mb##20_re;		\
  mc##20_im = ma##20_im + mb##20_im;		\
  mc##21_re = ma##21_re + mb##21_re;		\
  mc##21_im = ma##21_im + mb##21_im;		\
  mc##22_re = ma##22_re + mb##22_re;		\
  mc##22_im = ma##22_im + mb##22_im;		



void
llfat_init_cuda(QudaGaugeParam* param)
{
  static int llfat_init_cuda_flag = 0;
  if (llfat_init_cuda_flag){
    return;
  }
    
  llfat_init_cuda_flag = 1;
  
  init_kernel_cuda(param);

}


 
#define LLFAT_COMPUTE_NEW_IDX_LOWER_STAPLE(mydir1, mydir2) do {		\
    new_x1 = x1;							\
    new_x2 = x2;							\
    new_x3 = x3;							\
    new_x4 = x4;							\
    switch(mydir1){							\
    case 0:								\
      new_mem_idx = ( (x1==0)?X+X1m1:X-1);				\
      new_x1 = (x1==0)?X1m1:x1 - 1;					\
      break;								\
    case 1:								\
      new_mem_idx = ( (x2==0)?X+X2X1mX1:X-X1);				\
      new_x2 = (x2==0)?X2m1:x2 - 1;					\
      break;								\
    case 2:								\
      new_mem_idx = ( (x3==0)?X+X3X2X1mX2X1:X-X2X1);			\
      new_x3 = (x3==0)?X3m1:x3 - 1;					\
      break;								\
    case 3:								\
      new_mem_idx = ( (x4==0)?X+X4X3X2X1mX3X2X1:X-X3X2X1);		\
      new_x4 = (x4==0)?X4m1:x4 - 1;					\
      break;								\
    }									\
    switch(mydir2){							\
    case 0:								\
      new_mem_idx = ( (x1==X1m1)?new_mem_idx-X1m1:new_mem_idx+1)>> 1;	\
      new_x1 = (x1==X1m1)?0:x1+1;					\
      break;								\
    case 1:								\
      new_mem_idx = ( (x2==X2m1)?new_mem_idx-X2X1mX1:new_mem_idx+X1) >> 1; \
      new_x2 = (x2==X2m1)?0:x2+1;					\
      break;								\
    case 2:								\
      new_mem_idx = ( (x3==X3m1)?new_mem_idx-X3X2X1mX2X1:new_mem_idx+X2X1) >> 1; \
      new_x3 = (x3==X3m1)?0:x3+1;					\
      break;								\
    case 3:								\
      new_mem_idx = ( (x4==X4m1)?new_mem_idx-X4X3X2X1mX3X2X1:new_mem_idx+X3X2X1) >> 1; \
      new_x4 = (x4==X4m1)?0:x4+1;					\
      break;								\
    }									\
  }while(0)



#define LLFAT_COMPUTE_NEW_IDX_PLUS(mydir, idx) do {			\
    new_x1 = x1;							\
    new_x2 = x2;							\
    new_x3 = x3;							\
    new_x4 = x4;							\
    switch(mydir){							\
    case 0:								\
      new_mem_idx = ( (x1==X1m1)?idx-X1m1:idx+1)>>1;			\
      new_x1 = (x1==X1m1)?0:x1+1;					\
      break;								\
    case 1:								\
      new_mem_idx = ( (x2==X2m1)?idx-X2X1mX1:idx+X1)>>1;		\
      new_x2 = (x2==X2m1)?0:x2+1;					\
      break;								\
    case 2:								\
      new_mem_idx = ( (x3==X3m1)?idx-X3X2X1mX2X1:idx+X2X1)>>1;		\
      new_x3 = (x3==X3m1)?0:x3+1;					\
      break;								\
    case 3:								\
      new_mem_idx = ( (x4==X4m1)?idx-X4X3X2X1mX3X2X1:idx+X3X2X1)>>1;	\
      new_x4 = (x4==X4m1)?0:x4+1;					\
      break;								\
    }									\
  }while(0)

#define LLFAT_COMPUTE_NEW_IDX_MINUS(mydir, idx) do {			\
    new_x1 = x1;							\
    new_x2 = x2;							\
    new_x3 = x3;							\
    new_x4 = x4;							\
    switch(mydir){							\
    case 0:								\
      new_mem_idx = ( (x1==0)?idx+X1m1:idx-1) >> 1;			\
      new_x1 = (x1==0)?X1m1:x1 - 1;					\
      break;								\
    case 1:								\
      new_mem_idx = ( (x2==0)?idx+X2X1mX1:idx-X1) >> 1;			\
      new_x2 = (x2==0)?X2m1:x2 - 1;					\
      break;								\
    case 2:								\
      new_mem_idx = ( (x3==0)?idx+X3X2X1mX2X1:idx-X2X1) >> 1;		\
      new_x3 = (x3==0)?X3m1:x3 - 1;					\
      break;								\
    case 3:								\
      new_mem_idx = ( (x4==0)?idx+X4X3X2X1mX3X2X1:idx-X3X2X1) >> 1;	\
      new_x4 = (x4==0)?X4m1:x4 - 1;					\
      break;								\
    }									\
  }while(0)

    

#define COMPUTE_RECONSTRUCT_SIGN(sign, dir, i1,i2,i3,i4) do {	\
    sign =1;							\
    switch(dir){						\
    case XUP:							\
      if ( (i4 & 1) == 1){					\
	sign = -1;						\
      }								\
      break;							\
    case YUP:							\
      if ( ((i4+i1) & 1) == 1){					\
	sign = -1;						\
      }								\
      break;							\
    case ZUP:							\
      if ( ((i4+i1+i2) & 1) == 1){				\
	sign = -1;						\
      }								\
      break;							\
    case TUP:							\
      if (i4 == X4m1 ){						\
	sign = -1;						\
      }								\
      break;							\
    }								\
  }while (0)


#define LLFAT_CONCAT(a,b) a##b##Kernel
#define LLFAT_KERNEL(a,b) LLFAT_CONCAT(a,b)

//precision: 0 is for double, 1 is for single

//single precision, common macro
#define PRECISION 1
#define Float  float
#define LOAD_FAT_MATRIX(gauge, dir, idx) LOAD_MATRIX_18_SINGLE(gauge, dir, idx, FAT)
#if (MULINK_LOAD_TEX == 1)
#define LOAD_EVEN_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX(muLink0TexSingle, dir, idx, var)
#define LOAD_ODD_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX(muLink1TexSingle, dir, idx, var)
#else
#define LOAD_EVEN_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE(mulink_even, dir, idx, var)
#define LOAD_ODD_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE(mulink_odd, dir, idx, var)
#endif

#if (FATLINK_LOAD_TEX == 1)
#define LOAD_EVEN_FAT_MATRIX(dir, idx) LOAD_MATRIX_18_SINGLE_TEX(fatGauge0TexSingle, dir, idx, FAT)
#define LOAD_ODD_FAT_MATRIX(dir, idx) LOAD_MATRIX_18_SINGLE_TEX(fatGauge1TexSingle, dir, idx, FAT)
#else
#define LOAD_EVEN_FAT_MATRIX(dir, idx) LOAD_MATRIX_18_SINGLE(fatlink_even, dir, idx, FAT)
#define LOAD_ODD_FAT_MATRIX(dir, idx)  LOAD_MATRIX_18_SINGLE(fatlink_odd, dir, idx, FAT)
#endif


//single precision, 12-reconstruct
#define SITELINK0TEX siteLink0TexSingle
#define SITELINK1TEX siteLink1TexSingle
#if (SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX_DECLARE(SITELINK0TEX, dir, idx, var)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX_DECLARE(SITELINK1TEX, dir, idx, var)
#else
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_DECLARE(sitelink_even, dir, idx, var)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_DECLARE(sitelink_odd, dir, idx, var)
#endif
#define LOAD_SITE_MATRIX(sitelink, dir, idx, var) LOAD_MATRIX_12_SINGLE_DECLARE(sitelink, dir, idx, var)

#define RECONSTRUCT_SITE_LINK(dir, idx, sign, var)  RECONSTRUCT_LINK_12(dir, idx, sign, var);
#define FloatN float4
#define FloatM float2
#define RECONSTRUCT 12
#include "llfat_core.h"
#undef SITELINK0TEX
#undef SITELINK1TEX
#undef LOAD_EVEN_SITE_MATRIX
#undef LOAD_ODD_SITE_MATRIX
#undef LOAD_SITE_MATRIX
#undef RECONSTRUCT_SITE_LINK
#undef FloatN
#undef FloatM
#undef RECONSTRUCT

//single precision, 18-reconstruct
#define SITELINK0TEX siteLink0TexSingle_norecon
#define SITELINK1TEX siteLink1TexSingle_norecon
#if (SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX_DECLARE(SITELINK0TEX, dir, idx, var)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX_DECLARE(SITELINK1TEX, dir, idx, var)
#else
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_DECLARE(sitelink_even, dir, idx, var)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_DECLARE(sitelink_odd, dir, idx, var)
#endif
#define LOAD_SITE_MATRIX(sitelink, dir, idx, var) LOAD_MATRIX_18_SINGLE(sitelink, dir, idx, var)
#define RECONSTRUCT_SITE_LINK(dir, idx, sign, var)  
#define FloatN float2
#define FloatM float2
#define RECONSTRUCT 18
#include "llfat_core.h"
#undef SITELINK0TEX
#undef SITELINK1TEX
#undef LOAD_EVEN_SITE_MATRIX
#undef LOAD_ODD_SITE_MATRIX
#undef LOAD_SITE_MATRIX
#undef RECONSTRUCT_SITE_LINK
#undef FloatN
#undef FloatM
#undef RECONSTRUCT


#undef PRECISION
#undef Float
#undef LOAD_FAT_MATRIX
#undef LOAD_EVEN_MULINK_MATRIX
#undef LOAD_ODD_MULINK_MATRIX
#undef LOAD_EVEN_FAT_MATRIX
#undef LOAD_ODD_FAT_MATRIX


//double precision, common macro
#define PRECISION 0
#define Float double
#define LOAD_FAT_MATRIX(gauge, dir, idx) LOAD_MATRIX_18_DOUBLE(gauge, dir, idx, FAT)
#if (MULINK_LOAD_TEX == 1)
#define LOAD_EVEN_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_TEX(muLink0TexDouble, dir, idx, var)
#define LOAD_ODD_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_TEX(muLink1TexDouble, dir, idx, var)
#else
#define LOAD_EVEN_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE(mulink_even, dir, idx, var)
#define LOAD_ODD_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE(mulink_odd, dir, idx, var)
#endif

#if (FATLINK_LOAD_TEX == 1)
#define LOAD_EVEN_FAT_MATRIX(dir, idx) LOAD_MATRIX_18_DOUBLE_TEX(fatGauge0TexDouble, dir, idx, FAT)
#define LOAD_ODD_FAT_MATRIX(dir, idx) LOAD_MATRIX_18_DOUBLE_TEX(fatGauge1TexDouble, dir, idx, FAT)
#else
#define LOAD_EVEN_FAT_MATRIX(dir, idx) LOAD_MATRIX_18_DOUBLE(fatlink_even, dir, idx, FAT)
#define LOAD_ODD_FAT_MATRIX(dir, idx)  LOAD_MATRIX_18_DOUBLE(fatlink_odd, dir, idx, FAT)
#endif

//double precision,  18-reconstruct
#define SITELINK0TEX siteLink0TexDouble
#define SITELINK1TEX siteLink1TexDouble
#if (SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_TEX_DECLARE(SITELINK0TEX, dir, idx, var)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_TEX_DECLARE(SITELINK1TEX, dir, idx, var)
#else
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_DECLARE(sitelink_even, dir, idx, var)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_DECLARE(sitelink_odd, dir, idx, var)
#endif
#define LOAD_SITE_MATRIX(sitelink, dir, idx, var) LOAD_MATRIX_18_DOUBLE(sitelink, dir, idx, var)
#define RECONSTRUCT_SITE_LINK(dir, idx, sign, var)  
#define FloatN double2
#define FloatM double2
#define RECONSTRUCT 18
#include "llfat_core.h"
#undef SITELINK0TEX
#undef SITELINK1TEX
#undef LOAD_EVEN_SITE_MATRIX
#undef LOAD_ODD_SITE_MATRIX
#undef LOAD_SITE_MATRIX
#undef RECONSTRUCT_SITE_LINK
#undef FloatN
#undef FloatM
#undef RECONSTRUCT

#if 1
//double precision, 12-reconstruct
#define SITELINK0TEX siteLink0TexDouble
#define SITELINK1TEX siteLink1TexDouble
#if (SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE_TEX_DECLARE(SITELINK0TEX, dir, idx, var)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE_TEX_DECLARE(SITELINK1TEX, dir, idx, var)
#else
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE_DECLARE(sitelink_even, dir, idx, var)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE_DECLARE(sitelink_odd, dir, idx, var)
#endif
#define LOAD_SITE_MATRIX(sitelink, dir, idx, var) LOAD_MATRIX_12_DOUBLE_DECLARE(sitelink, dir, idx, var)
#define RECONSTRUCT_SITE_LINK(dir, idx, sign, var)  RECONSTRUCT_LINK_12(dir, idx, sign, var);
#define FloatN double2
#define FloatM double2
#define RECONSTRUCT 12
#include "llfat_core.h"
#undef SITELINK0TEX
#undef SITELINK1TEX
#undef LOAD_EVEN_SITE_MATRIX
#undef LOAD_ODD_SITE_MATRIX
#undef LOAD_SITE_MATRIX
#undef RECONSTRUCT_SITE_LINK
#undef FloatN
#undef FloatM
#undef RECONSTRUCT
#endif

#undef PRECISION
#undef Float
#undef LOAD_FAT_MATRIX
#undef LOAD_EVEN_MULINK_MATRIX
#undef LOAD_ODD_MULINK_MATRIX
#undef LOAD_EVEN_FAT_MATRIX
#undef LOAD_ODD_FAT_MATRIX

#undef LLFAT_CONCAT
#undef LLFAT_KERNEL

#define UNBIND_ALL_TEXTURE do{						\
    if(prec ==QUDA_DOUBLE_PRECISION){					\
      cudaUnbindTexture(siteLink0TexDouble);				\
      cudaUnbindTexture(siteLink1TexDouble);				\
      cudaUnbindTexture(fatGauge0TexDouble);				\
      cudaUnbindTexture(fatGauge1TexDouble);				\
      cudaUnbindTexture(muLink0TexDouble);				\
      cudaUnbindTexture(muLink1TexDouble);				\
    }else{								\
      if(cudaSiteLink.reconstruct == QUDA_RECONSTRUCT_NO){		\
	cudaUnbindTexture(siteLink0TexSingle_norecon);			\
	cudaUnbindTexture(siteLink1TexSingle_norecon);			\
      }else{								\
	cudaUnbindTexture(siteLink0TexSingle);				\
	cudaUnbindTexture(siteLink1TexSingle);				\
      }									\
      cudaUnbindTexture(fatGauge0TexSingle);				\
      cudaUnbindTexture(fatGauge1TexSingle);				\
      cudaUnbindTexture(muLink0TexSingle);				\
      cudaUnbindTexture(muLink1TexSingle);				\
    }									\
  }while(0)

#define UNBIND_SITE_AND_FAT_LINK do{					\
    if(prec == QUDA_DOUBLE_PRECISION){					\
      cudaUnbindTexture(siteLink0TexDouble);				\
      cudaUnbindTexture(siteLink1TexDouble);				\
      cudaUnbindTexture(fatGauge0TexDouble);				\
      cudaUnbindTexture(fatGauge1TexDouble);				\
    }else {								\
      if(cudaSiteLink.reconstruct == QUDA_RECONSTRUCT_NO){		\
	cudaUnbindTexture(siteLink0TexSingle_norecon);			\
	cudaUnbindTexture(siteLink1TexSingle_norecon);			\
      }else{								\
	cudaUnbindTexture(siteLink0TexSingle);				\
	cudaUnbindTexture(siteLink1TexSingle);				\
      }									\
      cudaUnbindTexture(fatGauge0TexSingle);				\
      cudaUnbindTexture(fatGauge1TexSingle);				\
    }									\
  }while(0)

#define BIND_SITE_AND_FAT_LINK do {					\
  if(prec == QUDA_DOUBLE_PRECISION){					\
    cudaBindTexture(0, siteLink0TexDouble, cudaSiteLink.even, cudaSiteLink.bytes); \
    cudaBindTexture(0, siteLink1TexDouble, cudaSiteLink.odd, cudaSiteLink.bytes); \
    cudaBindTexture(0, fatGauge0TexDouble, cudaFatLink.even, cudaFatLink.bytes); \
    cudaBindTexture(0, fatGauge1TexDouble, cudaFatLink.odd,  cudaFatLink.bytes); \
  }else{								\
    if(cudaSiteLink.reconstruct == QUDA_RECONSTRUCT_NO){		\
      cudaBindTexture(0, siteLink0TexSingle_norecon, cudaSiteLink.even, cudaSiteLink.bytes); \
      cudaBindTexture(0, siteLink1TexSingle_norecon, cudaSiteLink.odd, cudaSiteLink.bytes); \
    }else{								\
      cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.even, cudaSiteLink.bytes); \
      cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes); \
    }									\
    cudaBindTexture(0, fatGauge0TexSingle, cudaFatLink.even, cudaFatLink.bytes); \
    cudaBindTexture(0, fatGauge1TexSingle, cudaFatLink.odd,  cudaFatLink.bytes); \
    }									\
  }while(0)

#define BIND_SITE_AND_FAT_LINK_REVERSE do {				\
    if(prec == QUDA_DOUBLE_PRECISION){					\
      cudaBindTexture(0, siteLink1TexDouble, cudaSiteLink.even, cudaSiteLink.bytes); \
      cudaBindTexture(0, siteLink0TexDouble, cudaSiteLink.odd, cudaSiteLink.bytes); \
      cudaBindTexture(0, fatGauge1TexDouble, cudaFatLink.even, cudaFatLink.bytes); \
      cudaBindTexture(0, fatGauge0TexDouble, cudaFatLink.odd,  cudaFatLink.bytes); \
    }else{								\
      if(cudaSiteLink.reconstruct == QUDA_RECONSTRUCT_NO){		\
	cudaBindTexture(0, siteLink1TexSingle_norecon, cudaSiteLink.even, cudaSiteLink.bytes); \
	cudaBindTexture(0, siteLink0TexSingle_norecon, cudaSiteLink.odd, cudaSiteLink.bytes); \
      }else{								\
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes); \
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes); \
      }									\
      cudaBindTexture(0, fatGauge1TexSingle, cudaFatLink.even, cudaFatLink.bytes); \
      cudaBindTexture(0, fatGauge0TexSingle, cudaFatLink.odd,  cudaFatLink.bytes); \
    }									\
  }while(0)



#define ENUMERATE_FUNCS(mu,nu,odd_bit)	switch(mu) {			\
  case 0:								\
    switch(nu){								\
    case 0:								\
      printf("ERROR: invalid direction combination\n"); exit(1);	\
      break;								\
    case 1:								\
      if (!odd_bit) { CALL_FUNCTION(0,1,0); }				\
      else {CALL_FUNCTION(0,1,1); }					\
      break;								\
    case 2:								\
      if (!odd_bit) { CALL_FUNCTION(0,2,0); }				\
      else {CALL_FUNCTION(0,2,1); }					\
      break;								\
    case 3:								\
      if (!odd_bit) { CALL_FUNCTION(0,3,0); }				\
      else {CALL_FUNCTION(0,3,1); }					\
      break;								\
    }									\
    break;								\
  case 1:								\
    switch(nu){								\
    case 0:								\
      if (!odd_bit) { CALL_FUNCTION(1,0,0); }				\
      else {CALL_FUNCTION(1,0,1); }					\
      break;								\
    case 1:								\
      printf("ERROR: invalid direction combination\n"); exit(1);	\
      break;								\
    case 2:								\
      if (!odd_bit) { CALL_FUNCTION(1,2,0); }				\
      else {CALL_FUNCTION(1,2,1); }					\
      break;								\
    case 3:								\
      if (!odd_bit) { CALL_FUNCTION(1,3,0); }				\
      else {CALL_FUNCTION(1,3,1); }					\
      break;								\
    }									\
    break;								\
  case 2:								\
    switch(nu){								\
    case 0:								\
      if (!odd_bit) { CALL_FUNCTION(2,0,0); }				\
      else {CALL_FUNCTION(2,0,1); }					\
      break;								\
    case 1:								\
      if (!odd_bit) { CALL_FUNCTION(2,1,0); }				\
      else {CALL_FUNCTION(2,1,1); }					\
      break;								\
    case 2:								\
      printf("ERROR: invalid direction combination\n"); exit(1);	\
      break;								\
    case 3:								\
      if (!odd_bit) { CALL_FUNCTION(2,3,0); }				\
      else {CALL_FUNCTION(2,3,1); }					\
      break;								\
    }									\
    break;								\
  case 3:								\
    switch(nu){								\
    case 0:								\
      if (!odd_bit) { CALL_FUNCTION(3,0,0); }				\
      else {CALL_FUNCTION(3,0,1); }					\
      break;								\
    case 1:								\
      if (!odd_bit) { CALL_FUNCTION(3,1,0); }				\
      else {CALL_FUNCTION(3,1,1); }					\
      break;								\
    case 2:								\
      if (!odd_bit) { CALL_FUNCTION(3,2,0); }				\
      else {CALL_FUNCTION(3,2,1); }					\
      break;								\
    case 3:								\
      printf("ERROR: invalid direction combination\n"); exit(1);	\
      break;								\
    }									\
    break;								\
  }

void siteComputeGenStapleParityKernel(void* staple_even, void* staple_odd, 
				      void* sitelink_even, void* sitelink_odd, 
				      void* fatlink_even, void* fatlink_odd,	
				      int mu, int nu,int odd_bit,
				      double mycoeff,
				      dim3 halfGridDim, dim3 blockDim, 
				      QudaReconstructType recon, QudaPrecision prec)
{
    
#define  CALL_FUNCTION(mu, nu, odd_bit)					\
  if (prec == QUDA_DOUBLE_PRECISION){					\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_siteComputeGenStapleParity18Kernel<mu,nu, odd_bit>		\
	<<<halfGridDim, blockDim>>>((double2*)staple_even, (double2*)staple_odd, \
				    (double2*)sitelink_even, (double2*)sitelink_odd, \
				    (double2*)fatlink_even, (double2*)fatlink_odd, \
				    (double)mycoeff);			\
    }else{								\
      do_siteComputeGenStapleParity12Kernel<mu,nu, odd_bit>		\
	<<<halfGridDim, blockDim>>>((double2*)staple_even, (double2*)staple_odd, \
				    (double2*)sitelink_even, (double2*)sitelink_odd, \
				    (double2*)fatlink_even, (double2*)fatlink_odd, \
				    (double)mycoeff);			\
    }									\
  }else {								\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_siteComputeGenStapleParity18Kernel<mu,nu, odd_bit>		\
	<<<halfGridDim, blockDim>>>((float2*)staple_even, (float2*)staple_odd, \
				    (float2*)sitelink_even, (float2*)sitelink_odd, \
				    (float2*)fatlink_even, (float2*)fatlink_odd, \
				    (float)mycoeff);			\
    }else{								\
      do_siteComputeGenStapleParity12Kernel<mu,nu, odd_bit>		\
	<<<halfGridDim, blockDim>>>((float2*)staple_even, (float2*)staple_odd, \
				    (float4*)sitelink_even, (float4*)sitelink_odd, \
				    (float2*)fatlink_even, (float2*)fatlink_odd, \
				    (float)mycoeff);			\
    }									\
  }
  
  ENUMERATE_FUNCS(mu,nu,odd_bit);
  
#undef CALL_FUNCTION
    
    
}

void
computeGenStapleFieldParityKernel(void* sitelink_even, void* sitelink_odd,
				  void* fatlink_even, void* fatlink_odd,			    
				  void* mulink_even, void* mulink_odd, 
				  int mu, int nu, int odd_bit,
				  double mycoeff,
				  dim3 halfGridDim, dim3 blockDim, 
				  QudaReconstructType recon, QudaPrecision prec)
{    
#define  CALL_FUNCTION(mu, nu, odd_bit)					\
  if (prec == QUDA_DOUBLE_PRECISION){					\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_computeGenStapleFieldParity18Kernel<mu,nu, odd_bit>		\
	<<<halfGridDim, blockDim>>>((double2*)sitelink_even, (double2*)sitelink_odd, \
				    (double2*)fatlink_even, (double2*)fatlink_odd, \
				    (double2*)mulink_even, (double2*)mulink_odd, \
				    (double)mycoeff);			\
    }else{								\
      do_computeGenStapleFieldParity12Kernel<mu,nu, odd_bit>		\
	<<<halfGridDim, blockDim>>>((double2*)sitelink_even, (double2*)sitelink_odd, \
				    (double2*)fatlink_even, (double2*)fatlink_odd, \
				    (double2*)mulink_even, (double2*)mulink_odd, \
				    (double)mycoeff);			\
    }									\
  }else{								\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_computeGenStapleFieldParity18Kernel<mu,nu, odd_bit>		\
	<<<halfGridDim, blockDim>>>((float2*)sitelink_even, (float2*)sitelink_odd, \
				    (float2*)fatlink_even, (float2*)fatlink_odd, \
				    (float2*)mulink_even, (float2*)mulink_odd, \
				    (float)mycoeff);			\
    }else{								\
      do_computeGenStapleFieldParity12Kernel<mu,nu, odd_bit>		\
	<<<halfGridDim, blockDim>>>((float4*)sitelink_even, (float4*)sitelink_odd, \
				    (float2*)fatlink_even, (float2*)fatlink_odd, \
				    (float2*)mulink_even, (float2*)mulink_odd, \
				    (float)mycoeff);			\
    }									\
  }
  
  ENUMERATE_FUNCS(mu,nu,odd_bit);

#undef CALL_FUNCTION 
    
}

void
computeGenStapleFieldSaveParityKernel(void* staple_even, void* staple_odd, 
				      void* sitelink_even, void* sitelink_odd,
				      void* fatlink_even, void* fatlink_odd,			    
				      void* mulink_even, void* mulink_odd, 
				      int mu, int nu, int odd_bit,
				      double mycoeff,
				      dim3 halfGridDim, dim3 blockDim,
				      QudaReconstructType recon, QudaPrecision prec)
{
#define  CALL_FUNCTION(mu, nu, odd_bit)					\
  if (prec == QUDA_DOUBLE_PRECISION){					\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_computeGenStapleFieldSaveParity18Kernel<mu,nu, odd_bit>	\
	<<<halfGridDim, blockDim>>>((double2*)staple_even, (double2*)staple_odd, \
				    (double2*)sitelink_even, (double2*)sitelink_odd, \
				    (double2*)fatlink_even, (double2*)fatlink_odd, \
				    (double2*)mulink_even, (double2*)mulink_odd, \
				    (double)mycoeff);			\
    }else{								\
      do_computeGenStapleFieldSaveParity12Kernel<mu,nu, odd_bit>	\
	<<<halfGridDim, blockDim>>>((double2*)staple_even, (double2*)staple_odd, \
				    (double2*)sitelink_even, (double2*)sitelink_odd, \
				    (double2*)fatlink_even, (double2*)fatlink_odd, \
				    (double2*)mulink_even, (double2*)mulink_odd, \
				    (double)mycoeff);			\
    }									\
  }else{								\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_computeGenStapleFieldSaveParity18Kernel<mu,nu, odd_bit>	\
	<<<halfGridDim, blockDim>>>((float2*)staple_even, (float2*)staple_odd, \
				    (float2*)sitelink_even, (float2*)sitelink_odd, \
				    (float2*)fatlink_even, (float2*)fatlink_odd, \
				    (float2*)mulink_even, (float2*)mulink_odd, \
				    (float)mycoeff);			\
  }else{								\
      do_computeGenStapleFieldSaveParity12Kernel<mu,nu, odd_bit>	\
	<<<halfGridDim, blockDim>>>((float2*)staple_even, (float2*)staple_odd, \
				    (float4*)sitelink_even, (float4*)sitelink_odd, \
				    (float2*)fatlink_even, (float2*)fatlink_odd, \
				    (float2*)mulink_even, (float2*)mulink_odd, \
				    (float)mycoeff);			\
    }									\
  }

  ENUMERATE_FUNCS(mu,nu,odd_bit);

#undef CALL_FUNCTION 
    
}

void
llfat_cuda(void* fatLink, void* siteLink,
	   FullGauge cudaFatLink, FullGauge cudaSiteLink, 
	   FullStaple cudaStaple, FullStaple cudaStaple1,
	   QudaGaugeParam* param, double* act_path_coeff)
{

  int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
  dim3 gridDim(volume/BLOCK_DIM,1,1);
  dim3 halfGridDim(volume/(2*BLOCK_DIM),1,1);
  dim3 blockDim(BLOCK_DIM , 1, 1);
  
  QudaPrecision prec = cudaSiteLink.precision;
  QudaReconstructType recon = cudaSiteLink.reconstruct;
  BIND_SITE_AND_FAT_LINK;
  if(prec == QUDA_DOUBLE_PRECISION){
    if(recon == QUDA_RECONSTRUCT_NO){
      llfatOneLink18Kernel<<<gridDim, blockDim>>>((double2*)cudaSiteLink.even, (double2*)cudaSiteLink.odd,
						  (double2*)cudaFatLink.even, (double2*)cudaFatLink.odd,
						  (double)act_path_coeff[0], (double)act_path_coeff[5]);    
    }else{
      
      llfatOneLink12Kernel<<<gridDim, blockDim>>>((double2*)cudaSiteLink.even, (double2*)cudaSiteLink.odd,
						  (double2*)cudaFatLink.even, (double2*)cudaFatLink.odd,
						  (double)act_path_coeff[0], (double)act_path_coeff[5]);    
      
    }
  }else{ //single precision
    if(recon == QUDA_RECONSTRUCT_NO){    
      llfatOneLink18Kernel<<<gridDim, blockDim>>>((float2*)cudaSiteLink.even, (float2*)cudaSiteLink.odd,
						  (float2*)cudaFatLink.even, (float2*)cudaFatLink.odd,
						  (float)act_path_coeff[0], (float)act_path_coeff[5]);    						  
    }else{
      llfatOneLink12Kernel<<<gridDim, blockDim>>>((float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd,
						  (float2*)cudaFatLink.even, (float2*)cudaFatLink.odd,
						  (float)act_path_coeff[0], (float)act_path_coeff[5]);    
    }
  }
UNBIND_SITE_AND_FAT_LINK;

  for(int dir = 0;dir < 4; dir++){
    for(int nu = 0; nu < 4; nu++){
      if (nu != dir){
		
	//even
	BIND_SITE_AND_FAT_LINK;		
	siteComputeGenStapleParityKernel((void*)cudaStaple.even, (void*)cudaStaple.odd,
					 (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					 (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
					 dir, nu,0,
					 act_path_coeff[2],
					 halfGridDim, blockDim, recon, prec);
	UNBIND_SITE_AND_FAT_LINK;
	
	//odd
	BIND_SITE_AND_FAT_LINK_REVERSE;
	siteComputeGenStapleParityKernel((void*)cudaStaple.odd, (void*)cudaStaple.even,
					 (void*)cudaSiteLink.odd, (void*)cudaSiteLink.even,
					 (void*)cudaFatLink.odd, (void*)cudaFatLink.even, 
					 dir, nu,1,
					 act_path_coeff[2],
					 halfGridDim, blockDim, recon, prec);	
	UNBIND_SITE_AND_FAT_LINK;	
	
	
	//even
	BIND_SITE_AND_FAT_LINK;		
	cudaBindTexture(0, muLink0TexSingle, cudaStaple.even, cudaStaple.bytes);
	cudaBindTexture(0, muLink1TexSingle, cudaStaple.odd, cudaStaple.bytes);
	computeGenStapleFieldParityKernel((void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					  (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
					  (void*)cudaStaple.even, (void*)cudaStaple.odd,
					  dir, nu,0,
					  act_path_coeff[5],
					  halfGridDim, blockDim, recon, prec);							  
	UNBIND_ALL_TEXTURE;
	
	//odd
	BIND_SITE_AND_FAT_LINK_REVERSE;		
	cudaBindTexture(0, muLink1TexSingle, cudaStaple.even, cudaStaple.bytes);
	cudaBindTexture(0, muLink0TexSingle, cudaStaple.odd, cudaStaple.bytes);
	computeGenStapleFieldParityKernel((void*)cudaSiteLink.odd, (void*)cudaSiteLink.even,
					  (void*)cudaFatLink.odd, (void*)cudaFatLink.even, 
					  (void*)cudaStaple.odd, (void*)cudaStaple.even,
					  dir, nu,1,
					  act_path_coeff[5],
					  halfGridDim, blockDim, recon, prec);	
	UNBIND_ALL_TEXTURE;
	
	
	for(int rho = 0; rho < 4; rho++){
	  if (rho != dir && rho != nu){
	    
	    //even
	    BIND_SITE_AND_FAT_LINK;		
	    cudaBindTexture(0, muLink0TexSingle, cudaStaple.even, cudaStaple.bytes);
	    cudaBindTexture(0, muLink1TexSingle, cudaStaple.odd, cudaStaple.bytes);			
	    computeGenStapleFieldSaveParityKernel((void*)cudaStaple1.even, (void*)cudaStaple1.odd,
						  (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
						  (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
						  (void*)cudaStaple.even, (void*)cudaStaple.odd,
						  dir, rho,0,
						  act_path_coeff[3],
						  halfGridDim, blockDim, recon, prec);								      
	    
	    UNBIND_ALL_TEXTURE;
	    
	    //odd
	    BIND_SITE_AND_FAT_LINK_REVERSE;		
	    cudaBindTexture(0, muLink1TexSingle, cudaStaple.even, cudaStaple.bytes);
	    cudaBindTexture(0, muLink0TexSingle, cudaStaple.odd, cudaStaple.bytes);						
	    computeGenStapleFieldSaveParityKernel((void*)cudaStaple1.odd, (void*)cudaStaple1.even,
						  (void*)cudaSiteLink.odd, (void*)cudaSiteLink.even,
						  (void*)cudaFatLink.odd, (void*)cudaFatLink.even, 
						  (void*)cudaStaple.odd, (void*)cudaStaple.even,
						  dir, rho,1,
						  act_path_coeff[3],
						  halfGridDim, blockDim, recon, prec);								      
	    UNBIND_ALL_TEXTURE;

			
	    for(int sig = 0; sig < 4; sig++){
	      if (sig != dir && sig != nu && sig != rho){				
				
		//even				
		BIND_SITE_AND_FAT_LINK;		
		cudaBindTexture(0, muLink0TexSingle, cudaStaple1.even, cudaStaple1.bytes);
		cudaBindTexture(0, muLink1TexSingle, cudaStaple1.odd,  cudaStaple1.bytes);
		computeGenStapleFieldParityKernel((void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
						  (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
						  (void*)cudaStaple1.even, (void*)cudaStaple1.odd,
						  dir, sig, 0, 
						  act_path_coeff[4],
						  halfGridDim, blockDim, recon, prec);	
		UNBIND_ALL_TEXTURE;
		
		//odd
		BIND_SITE_AND_FAT_LINK_REVERSE;		
		cudaBindTexture(0, muLink1TexSingle, cudaStaple1.even, cudaStaple1.bytes);
		cudaBindTexture(0, muLink0TexSingle, cudaStaple1.odd,  cudaStaple1.bytes);
		computeGenStapleFieldParityKernel((void*)cudaSiteLink.odd, (void*)cudaSiteLink.even,
						  (void*)cudaFatLink.odd, (void*)cudaFatLink.even, 
						  (void*)cudaStaple1.odd, (void*)cudaStaple1.even,
						  dir, sig, 1, 
						  act_path_coeff[4],
						  halfGridDim, blockDim, recon, prec);	
		UNBIND_ALL_TEXTURE;				
	      }			    
	    }//sig
	  }
	}//rho



      }
    }//nu
  }//dir
  
  checkCudaError();
  return;
}

