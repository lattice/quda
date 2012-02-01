#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <quda_internal.h>
#include <llfat_quda.h>
#include <read_gauge.h>
#include "gauge_field.h"
#include <force_common.h>

#if (__COMPUTE_CAPABILITY__ >= 200)
#define SITE_MATRIX_LOAD_TEX 1
#define MULINK_LOAD_TEX 1
#define FATLINK_LOAD_TEX 1
#else
#define SITE_MATRIX_LOAD_TEX 0
#define MULINK_LOAD_TEX 1
#define FATLINK_LOAD_TEX 1
#endif

#define WRITE_FAT_MATRIX(gauge, dir, idx)do {			\
    gauge[idx + dir*9*llfat_ga_stride] = FAT0;			\
    gauge[idx + (dir*9+1) * llfat_ga_stride] = FAT1;			\
    gauge[idx + (dir*9+2) * llfat_ga_stride] = FAT2;			\
    gauge[idx + (dir*9+3) * llfat_ga_stride] = FAT3;			\
    gauge[idx + (dir*9+4) * llfat_ga_stride] = FAT4;		\
    gauge[idx + (dir*9+5) * llfat_ga_stride] = FAT5;		\
    gauge[idx + (dir*9+6) * llfat_ga_stride] = FAT6;		\
    gauge[idx + (dir*9+7) * llfat_ga_stride] = FAT7;		\
    gauge[idx + (dir*9+8) * llfat_ga_stride] = FAT8;} while(0)			


#define WRITE_STAPLE_MATRIX(gauge, idx)				\
  gauge[idx] = STAPLE0;						\
  gauge[idx + staple_stride] = STAPLE1;				\
  gauge[idx + 2*staple_stride] = STAPLE2;			\
  gauge[idx + 3*staple_stride] = STAPLE3;			\
  gauge[idx + 4*staple_stride] = STAPLE4;			\
  gauge[idx + 5*staple_stride] = STAPLE5;			\
  gauge[idx + 6*staple_stride] = STAPLE6;			\
  gauge[idx + 7*staple_stride] = STAPLE7;			\
  gauge[idx + 8*staple_stride] = STAPLE8;					
    

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
  

#define LOAD_MATRIX_12_SINGLE_DECLARE(gauge, dir, idx, var, stride)	\
  float2 var##0 = gauge[idx + dir*6*stride];				\
  float2 var##1 = gauge[idx + dir*6*stride + stride];			\
  float2 var##2 = gauge[idx + dir*6*stride + 2*stride];			\
  float2 var##3 = gauge[idx + dir*6*stride + 3*stride];			\
  float2 var##4 = gauge[idx + dir*6*stride + 4*stride];			\
  float2 var##5 = gauge[idx + dir*6*stride + 5*stride];			\
  float2 var##6, var##7, var##8;

#define LOAD_MATRIX_12_SINGLE_TEX_DECLARE(gauge, dir, idx, var, stride)	\
  float2 var##0 = tex1Dfetch(gauge, idx + dir*6*stride);		\
  float2 var##1 = tex1Dfetch(gauge, idx + dir*6*stride + stride);	\
  float2 var##2 = tex1Dfetch(gauge, idx + dir*6*stride + 2*stride);	\
  float2 var##3 = tex1Dfetch(gauge, idx + dir*6*stride + 3*stride);	\
  float2 var##4 = tex1Dfetch(gauge, idx + dir*6*stride + 4*stride);	\
  float2 var##5 = tex1Dfetch(gauge, idx + dir*6*stride + 5*stride);	\
  float2 var##6, var##7, var##8;

#define LOAD_MATRIX_18_SINGLE_DECLARE(gauge, dir, idx, var, stride)	\
  float2 var##0 = gauge[idx + dir*9*stride];				\
  float2 var##1 = gauge[idx + dir*9*stride + stride];			\
  float2 var##2 = gauge[idx + dir*9*stride + 2*stride];			\
  float2 var##3 = gauge[idx + dir*9*stride + 3*stride];			\
  float2 var##4 = gauge[idx + dir*9*stride + 4*stride];			\
  float2 var##5 = gauge[idx + dir*9*stride + 5*stride];			\
  float2 var##6 = gauge[idx + dir*9*stride + 6*stride];			\
  float2 var##7 = gauge[idx + dir*9*stride + 7*stride];			\
  float2 var##8 = gauge[idx + dir*9*stride + 8*stride];			


#define LOAD_MATRIX_18_SINGLE_TEX_DECLARE(gauge, dir, idx, var, stride)	\
  float2 var##0 = tex1Dfetch(gauge, idx + dir*9*stride);		\
  float2 var##1 = tex1Dfetch(gauge, idx + dir*9*stride + stride);	\
  float2 var##2 = tex1Dfetch(gauge, idx + dir*9*stride + 2*stride);	\
  float2 var##3 = tex1Dfetch(gauge, idx + dir*9*stride + 3*stride);	\
  float2 var##4 = tex1Dfetch(gauge, idx + dir*9*stride + 4*stride);	\
  float2 var##5 = tex1Dfetch(gauge, idx + dir*9*stride + 5*stride);	\
  float2 var##6 = tex1Dfetch(gauge, idx + dir*9*stride + 6*stride);	\
  float2 var##7 = tex1Dfetch(gauge, idx + dir*9*stride + 7*stride);	\
  float2 var##8 = tex1Dfetch(gauge, idx + dir*9*stride + 8*stride);			



#define LOAD_MATRIX_18_DOUBLE_DECLARE(gauge, dir, idx, var, stride)	\
  double2 var##0 = gauge[idx + dir*9*stride];				\
  double2 var##1 = gauge[idx + dir*9*stride + stride];			\
  double2 var##2 = gauge[idx + dir*9*stride + 2*stride];		\
  double2 var##3 = gauge[idx + dir*9*stride + 3*stride];		\
  double2 var##4 = gauge[idx + dir*9*stride + 4*stride];		\
  double2 var##5 = gauge[idx + dir*9*stride + 5*stride];		\
  double2 var##6 = gauge[idx + dir*9*stride + 6*stride];		\
  double2 var##7 = gauge[idx + dir*9*stride + 7*stride];		\
  double2 var##8 = gauge[idx + dir*9*stride + 8*stride];			


#define LOAD_MATRIX_18_DOUBLE_TEX_DECLARE(gauge_tex, gauge, dir, idx, var, stride) \
  double2 var##0 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*9*stride); \
  double2 var##1 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*9*stride + stride);	\
  double2 var##2 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*9*stride + 2*stride); \
  double2 var##3 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*9*stride + 3*stride); \
  double2 var##4 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*9*stride + 4*stride); \
  double2 var##5 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*9*stride + 5*stride); \
  double2 var##6 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*9*stride + 6*stride); \
  double2 var##7 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*9*stride + 7*stride); \
  double2 var##8 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*9*stride + 8*stride);	


#define LOAD_MATRIX_12_DOUBLE_DECLARE(gauge, dir, idx, var, stride)		\
  double2 var##0 = gauge[idx + dir*6*stride];				\
  double2 var##1 = gauge[idx + dir*6*stride + stride];			\
  double2 var##2 = gauge[idx + dir*6*stride + 2*stride];		\
  double2 var##3 = gauge[idx + dir*6*stride + 3*stride];		\
  double2 var##4 = gauge[idx + dir*6*stride + 4*stride];		\
  double2 var##5 = gauge[idx + dir*6*stride + 5*stride];		\
  double2 var##6, var##7, var##8;


#define LOAD_MATRIX_12_DOUBLE_TEX_DECLARE(gauge_tex, gauge, dir, idx, var, stride) \
  double2 var##0 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*6*stride); \
  double2 var##1 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*6*stride + stride);	\
  double2 var##2 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*6*stride + 2*stride); \
  double2 var##3 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*6*stride + 3*stride); \
  double2 var##4 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*6*stride + 4*stride); \
  double2 var##5 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*6*stride + 5*stride); \
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

__constant__ int dir1_array[16];
__constant__ int dir2_array[16];
__constant__ int last_proc_in_tdim;
__constant__ int first_proc_in_tdim;
__constant__ int E1, E2, E3, E4, E1h;
__constant__ int Vh_ex;
__constant__ int E2E1;
__constant__ int E3E1;
__constant__ int E3E2;
__constant__ int E3E2E1;
__constant__ int E4E2E1;
__constant__ int E4E3E1;
__constant__ int E4E3E2;

unsigned long staple_bytes=0;

void
llfat_init_cuda(QudaGaugeParam* param)
{
  static int llfat_init_cuda_flag = 0;
  if (llfat_init_cuda_flag){
    return;
  }
  
  llfat_init_cuda_flag = 1;
  
  init_kernel_cuda(param);
  int Vh = param->X[0]*param->X[1]*param->X[2]*param->X[3]/2;
  int site_ga_stride = param->site_ga_pad + Vh;
  int staple_stride = param->staple_pad + Vh;
  int llfat_ga_stride = param->llfat_ga_pad + Vh;
  
  cudaMemcpyToSymbol("site_ga_stride", &site_ga_stride, sizeof(int));  
  cudaMemcpyToSymbol("staple_stride", &staple_stride, sizeof(int));  
  cudaMemcpyToSymbol("llfat_ga_stride", &llfat_ga_stride, sizeof(int));
  int dir1[16];
  int dir2[16];
  for(int nu =0; nu < 4; nu++)
    for(int mu=0; mu < 4; mu++){
      if(nu == mu) continue;
      int d1, d2;
      for(d1=0; d1 < 4; d1 ++){
        if(d1 != nu && d1 != mu){
          break;
        }
      }
      dir1[nu*4+mu] = d1;

      for(d2=0; d2 < 4; d2 ++){
        if(d2 != nu && d2 != mu && d2 != d1){
          break;
        }
      }

      dir2[nu*4+mu] = d2;
    }
  
  cudaMemcpyToSymbol("dir1_array", &dir1, sizeof(dir1));
  cudaMemcpyToSymbol("dir2_array", &dir2, sizeof(dir2));   
  
  int first_proc_in_tdim = 0;
  int last_proc_in_tdim = 0;
  if(commCoords(3) == (commDim(3) -1)){
    last_proc_in_tdim =  1;
  }
  
  if(commCoords(3) == 0){
    first_proc_in_tdim =  1;    
  }

  cudaMemcpyToSymbol("last_proc_in_tdim", &last_proc_in_tdim, sizeof(int));
  cudaMemcpyToSymbol("first_proc_in_tdim", &first_proc_in_tdim, sizeof(int));
  
}


void
llfat_init_cuda_ex(QudaGaugeParam* param_ex)
{
  static int llfat_init_cuda_flag = 0;
  if (llfat_init_cuda_flag){
    return;
  }

  llfat_init_cuda_flag = 1;
  
  init_kernel_cuda(param_ex);
  int Vh_ex = param_ex->X[0]*param_ex->X[1]*param_ex->X[2]*param_ex->X[3]/2;
  int Vh = (param_ex->X[0]-4)*(param_ex->X[1]-4)*(param_ex->X[2]-4)*(param_ex->X[3]-4)/2;
  int site_ga_stride = param_ex->site_ga_pad + Vh_ex;
  int staple_stride = param_ex->staple_pad + Vh_ex;
  int llfat_ga_stride = param_ex->llfat_ga_pad + Vh;
  
  cudaMemcpyToSymbol("site_ga_stride", &site_ga_stride, sizeof(int));
  cudaMemcpyToSymbol("staple_stride", &staple_stride, sizeof(int));
  cudaMemcpyToSymbol("llfat_ga_stride", &llfat_ga_stride, sizeof(int));
  
  int first_proc_in_tdim = 0;
  int last_proc_in_tdim = 0;
  if(commCoords(3) == (commDim(3) -1)){
    last_proc_in_tdim =  1;
  }
  
  if(commCoords(3) == 0){
    first_proc_in_tdim =  1;
  }
  
  cudaMemcpyToSymbol("last_proc_in_tdim", &last_proc_in_tdim, sizeof(int));
  cudaMemcpyToSymbol("first_proc_in_tdim", &first_proc_in_tdim, sizeof(int));
  
  int E1 = param_ex->X[0];
  int E1h = E1/2;
  int E2 = param_ex->X[1];
  int E3 = param_ex->X[2];
  int E4 = param_ex->X[3];
  int E2E1 =E2*E1;
  int E3E2E1=E3*E2*E1;
  
  cudaMemcpyToSymbol("E1", &E1, sizeof(int));
  cudaMemcpyToSymbol("E1h", &E1h, sizeof(int));
  cudaMemcpyToSymbol("E2", &E2, sizeof(int));
  cudaMemcpyToSymbol("E3", &E3, sizeof(int));
  cudaMemcpyToSymbol("E4", &E4, sizeof(int));
  cudaMemcpyToSymbol("E2E1", &E2E1, sizeof(int));
  cudaMemcpyToSymbol("E3E2E1", &E3E2E1, sizeof(int));

  cudaMemcpyToSymbol("Vh_ex", &Vh_ex, sizeof(int));
  
}




#define LLFAT_CONCAT(a,b) a##b##Kernel
#define LLFAT_CONCAT_EX(a,b) a##b##Kernel_ex
#define LLFAT_KERNEL(a,b) LLFAT_CONCAT(a,b)
#define LLFAT_KERNEL_EX(a,b) LLFAT_CONCAT_EX(a,b)

//precision: 0 is for double, 1 is for single

//single precision, common macro
#define PRECISION 1
#define Float  float
#define LOAD_FAT_MATRIX(gauge, dir, idx) LOAD_MATRIX_18_SINGLE_DECLARE(gauge, dir, idx, FAT, llfat_ga_stride)
#if (MULINK_LOAD_TEX == 1)
#define LOAD_EVEN_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX_DECLARE((odd_bit?muLink1TexSingle:muLink0TexSingle), dir, idx, var, staple_stride)
#define LOAD_ODD_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX_DECLARE((odd_bit?muLink0TexSingle:muLink1TexSingle), dir, idx, var, staple_stride)
#else
#define LOAD_EVEN_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_DECLARE(mulink_even, dir, idx, var, staple_stride)
#define LOAD_ODD_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_DECLARE(mulink_odd, dir, idx, var, staple_stride)
#endif

#if (FATLINK_LOAD_TEX == 1)
#define LOAD_EVEN_FAT_MATRIX(dir, idx) LOAD_MATRIX_18_SINGLE_TEX_DECLARE((odd_bit?fatGauge1TexSingle:fatGauge0TexSingle), dir, idx, FAT, llfat_ga_stride);
#define LOAD_ODD_FAT_MATRIX(dir, idx) LOAD_MATRIX_18_SINGLE_TEX_DECLARE((odd_bit?fatGauge0TexSingle:fatGauge1TexSingle), dir, idx, FAT, llfat_ga_stride);
#else
#define LOAD_EVEN_FAT_MATRIX(dir, idx) LOAD_MATRIX_18_SINGLE_DECLARE(fatlink_even, dir, idx, FAT, llfat_ga_stride)
#define LOAD_ODD_FAT_MATRIX(dir, idx)  LOAD_MATRIX_18_SINGLE_DECLARE(fatlink_odd, dir, idx, FAT, llfat_ga_stride)
#endif


//single precision, 12-reconstruct
#define DECLARE_VAR_SIGN short sign=1
#define SITELINK0TEX siteLink0TexSingle
#define SITELINK1TEX siteLink1TexSingle
#if (SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX_DECLARE((odd_bit?SITELINK1TEX:SITELINK0TEX), dir, idx, var, site_ga_stride)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX_DECLARE((odd_bit?SITELINK0TEX:SITELINK1TEX), dir, idx, var, site_ga_stride)
#else
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_DECLARE(sitelink_even, dir, idx, var, site_ga_stride)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_DECLARE(sitelink_odd, dir, idx, var, site_ga_stride)
#endif
#define LOAD_SITE_MATRIX(sitelink, dir, idx, var) LOAD_MATRIX_12_SINGLE_DECLARE(sitelink, dir, idx, var, site_ga_stride)

#define RECONSTRUCT_SITE_LINK(dir, idx, sign, var)  RECONSTRUCT_LINK_12(dir, idx, sign, var);
#define FloatN float2
#define FloatM float2
#define RECONSTRUCT 12
#define sd_data float_12_sd_data
#include "llfat_core.h"
#undef DECLARE_VAR_SIGN
#undef SITELINK0TEX
#undef SITELINK1TEX
#undef LOAD_EVEN_SITE_MATRIX
#undef LOAD_ODD_SITE_MATRIX
#undef LOAD_SITE_MATRIX
#undef RECONSTRUCT_SITE_LINK
#undef FloatN
#undef FloatM
#undef RECONSTRUCT
#undef sd_data

//single precision, 18-reconstruct
#define SITELINK0TEX siteLink0TexSingle_norecon
#define SITELINK1TEX siteLink1TexSingle_norecon
#if (SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var)  LOAD_MATRIX_18_SINGLE_TEX_DECLARE((odd_bit?SITELINK1TEX:SITELINK0TEX), dir, idx, var, site_ga_stride)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX_DECLARE((odd_bit?SITELINK0TEX:SITELINK1TEX), dir, idx, var, site_ga_stride)
#else
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_DECLARE(sitelink_even, dir, idx, var, site_ga_stride)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_DECLARE(sitelink_odd, dir, idx, var, site_ga_stride)
#endif
#define LOAD_SITE_MATRIX(sitelink, dir, idx, var) LOAD_MATRIX_18_SINGLE_DECLARE(sitelink, dir, idx, var, site_ga_stride)
#define RECONSTRUCT_SITE_LINK(dir, idx, sign, var)  
#define FloatN float2
#define FloatM float2
#define RECONSTRUCT 18
#define sd_data float_18_sd_data
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
#undef sd_data


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
#define LOAD_FAT_MATRIX(gauge, dir, idx) LOAD_MATRIX_18_DOUBLE_DECLARE(gauge, dir, idx, FAT, llfat_ga_stride)
#if (MULINK_LOAD_TEX == 1)
#define LOAD_EVEN_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_TEX_DECLARE(odd_bit?muLink1TexDouble:muLink0TexDouble), mulink_even, dir, idx, var, staple_stride)
#define LOAD_ODD_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_TEX_DECLARE((odd_bit?muLink0TexDouble:muLink1TexDouble), mulink_odd, dir, idx, var, staple_stride)
#else
#define LOAD_EVEN_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE(mulink_even, dir, idx, var, staple_stride)
#define LOAD_ODD_MULINK_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE(mulink_odd, dir, idx, var, staple_stride)
#endif

#if (FATLINK_LOAD_TEX == 1)
#define LOAD_EVEN_FAT_MATRIX(dir, idx) LOAD_MATRIX_18_DOUBLE_TEX_DECLARE((odd_bit?fatGauge1TexDouble:fatGauge0TexDouble), fatlink_even, dir, idx, FAT, llfat_ga_stride)
#define LOAD_ODD_FAT_MATRIX(dir, idx) LOAD_MATRIX_18_DOUBLE_TEX_DECLARE((odd_bit?fatGauge0TexDouble:fatGauge1TexDouble), fatlink_odd, dir, idx, FAT, llfat_ga_stride)
#else
#define LOAD_EVEN_FAT_MATRIX(dir, idx) LOAD_MATRIX_18_DOUBLE_DECLARE(fatlink_even, dir, idx, FAT, llfat_ga_stride)
#define LOAD_ODD_FAT_MATRIX(dir, idx)  LOAD_MATRIX_18_DOUBLE_DECLARE(fatlink_odd, dir, idx, FAT, llfat_ga_stride)
#endif

//double precision,  18-reconstruct
#define SITELINK0TEX siteLink0TexDouble
#define SITELINK1TEX siteLink1TexDouble
#if (SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_TEX_DECLARE((odd_bit?SITELINK1TEX:SITELINK0TEX), sitelink_even, dir, idx, var, site_ga_stride)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_TEX_DECLARE((odd_bit?SITELINK0TEX:SITELINK1TEX), sitelink_odd, dir, idx, var, site_ga_stride)
#else
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_DECLARE(sitelink_even, dir, idx, var, site_ga_stride)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_DECLARE(sitelink_odd, dir, idx, var, site_ga_stride)
#endif
#define LOAD_SITE_MATRIX(sitelink, dir, idx, var) LOAD_MATRIX_18_DOUBLE_DECLARE(sitelink, dir, idx, var, site_ga_stride)
#define RECONSTRUCT_SITE_LINK(dir, idx, sign, var)  
#define FloatN double2
#define FloatM double2
#define RECONSTRUCT 18
#define sd_data double_18_sd_data
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
#undef sd_data



#if 1
//double precision, 12-reconstruct
#define SITELINK0TEX siteLink0TexDouble
#define SITELINK1TEX siteLink1TexDouble
#if (SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE_TEX_DECLARE((odd_bit?SITELINK1TEX:SITELINK0TEX), sitelink_even, dir, idx, var, site_ga_stride)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE_TEX_DECLARE((odd_bit?SITELINK0TEX:SITELINK1TEX), sitelink_odd, dir, idx, var, site_ga_stride)
#else
#define LOAD_EVEN_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE_DECLARE(sitelink_even, dir, idx, var, site_ga_stride)
#define LOAD_ODD_SITE_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE_DECLARE(sitelink_odd, dir, idx, var, site_ga_stride)
#endif
#define LOAD_SITE_MATRIX(sitelink, dir, idx, var) LOAD_MATRIX_12_DOUBLE_DECLARE(sitelink, dir, idx, var, site_ga_stride)
#define RECONSTRUCT_SITE_LINK(dir, idx, sign, var)  RECONSTRUCT_LINK_12(dir, idx, sign, var);
#define FloatN double2
#define FloatM double2
#define RECONSTRUCT 12
#define sd_data double_12_sd_data
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
#undef sd_data
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


#define BIND_MU_LINK() do{						\
    if(prec == QUDA_DOUBLE_PRECISION){					\
      cudaBindTexture(0, muLink0TexDouble, mulink_even, staple_bytes);  \
      cudaBindTexture(0, muLink1TexDouble, mulink_odd, staple_bytes);	\
    }else{								\
      cudaBindTexture(0, muLink0TexSingle, mulink_even, staple_bytes);  \
      cudaBindTexture(0, muLink1TexSingle, mulink_odd, staple_bytes);	\
    }									\
  }while(0)

#define UNBIND_MU_LINK() do{			  \
    if(prec == QUDA_DOUBLE_PRECISION){		  \
      cudaUnbindTexture(muLink0TexSingle);        \
      cudaUnbindTexture(muLink1TexSingle);        \
    }else{					  \
      cudaUnbindTexture(muLink0TexDouble);        \
      cudaUnbindTexture(muLink1TexDouble);        \
    }						  \
  }while(0)                


#define BIND_SITE_AND_FAT_LINK do {					\
  if(prec == QUDA_DOUBLE_PRECISION){					\
    cudaBindTexture(0, siteLink0TexDouble, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()); \
    cudaBindTexture(0, siteLink1TexDouble, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()); \
    cudaBindTexture(0, fatGauge0TexDouble, cudaFatLink.Even_p(), cudaFatLink.Bytes()); \
    cudaBindTexture(0, fatGauge1TexDouble, cudaFatLink.Odd_p(),  cudaFatLink.Bytes()); \
  }else{								\
    if(cudaSiteLink.Reconstruct() == QUDA_RECONSTRUCT_NO){		\
      cudaBindTexture(0, siteLink0TexSingle_norecon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()); \
      cudaBindTexture(0, siteLink1TexSingle_norecon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()); \
    }else{								\
      cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()); \
      cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()); \
    }									\
    cudaBindTexture(0, fatGauge0TexSingle, cudaFatLink.Even_p(), cudaFatLink.Bytes()); \
    cudaBindTexture(0, fatGauge1TexSingle, cudaFatLink.Odd_p(),  cudaFatLink.Bytes()); \
    }									\
  }while(0)

#define BIND_MU_LINK() do{						\
    if(prec == QUDA_DOUBLE_PRECISION){					\
      cudaBindTexture(0, muLink0TexDouble, mulink_even, staple_bytes);	\
      cudaBindTexture(0, muLink1TexDouble, mulink_odd, staple_bytes);	\
    }else{								\
      cudaBindTexture(0, muLink0TexSingle, mulink_even, staple_bytes);	\
      cudaBindTexture(0, muLink1TexSingle, mulink_odd, staple_bytes);	\
    }									\
  }while(0)

#define UNBIND_MU_LINK() do{						\
    if(prec == QUDA_DOUBLE_PRECISION){					\
      cudaUnbindTexture(muLink0TexSingle);				\
      cudaUnbindTexture(muLink1TexSingle);				\
    }else{								\
      cudaUnbindTexture(muLink0TexDouble);				\
      cudaUnbindTexture(muLink1TexDouble);				\
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



#define ENUMERATE_FUNCS(mu,nu)	switch(mu) {				\
  case 0:								\
    switch(nu){								\
    case 0:								\
      printf("ERROR: invalid direction combination\n"); exit(1);	\
      break;								\
    case 1:								\
      CALL_FUNCTION(0,1);						\
      break;								\
    case 2:								\
      CALL_FUNCTION(0,2);						\
      break;								\
    case 3:								\
      CALL_FUNCTION(0,3); 						\
      break;								\
    }									\
    break;								\
  case 1:								\
    switch(nu){								\
    case 0:								\
      CALL_FUNCTION(1,0);						\
      break;								\
    case 1:								\
      printf("ERROR: invalid direction combination\n"); exit(1);	\
      break;								\
    case 2:								\
      CALL_FUNCTION(1,2);						\
      break;								\
    case 3:								\
      CALL_FUNCTION(1,3);						\
      break;								\
    }									\
    break;								\
  case 2:								\
    switch(nu){								\
    case 0:								\
      CALL_FUNCTION(2,0);						\
      break;								\
    case 1:								\
      CALL_FUNCTION(2,1);						\
      break;								\
    case 2:								\
      printf("ERROR: invalid direction combination\n"); exit(1);	\
      break;								\
    case 3:								\
      CALL_FUNCTION(2,3);						\
      break;								\
    }									\
    break;								\
  case 3:								\
    switch(nu){								\
    case 0:								\
      CALL_FUNCTION(3,0);						\
      break;								\
    case 1:								\
      CALL_FUNCTION(3,1);						\
      break;								\
    case 2:								\
      CALL_FUNCTION(3,2);						\
      break;								\
    case 3:								\
      printf("ERROR: invalid direction combination\n"); exit(1);	\
      break;								\
    }									\
    break;								\
  }

#define ENUMERATE_FUNCS_SAVE(mu,nu, save_staple) if(save_staple){ \
    switch(mu) {							\
    case 0:								\
      switch(nu){							\
      case 0:								\
	printf("ERROR: invalid direction combination\n"); exit(1);	\
	break;								\
      case 1:								\
	CALL_FUNCTION(0,1,1); 						\
	break;								\
      case 2:								\
	CALL_FUNCTION(0,2,1);						\
	break;								\
      case 3:								\
	CALL_FUNCTION(0,3,1);						\
	break;								\
      }									\
      break;								\
    case 1:								\
      switch(nu){							\
      case 0:								\
	CALL_FUNCTION(1,0,1);						\
	break;								\
      case 1:								\
	printf("ERROR: invalid direction combination\n"); exit(1);	\
	break;								\
      case 2:								\
	CALL_FUNCTION(1,2,1);						\
	break;								\
      case 3:								\
	CALL_FUNCTION(1,3,1); 						\
	break;								\
      }									\
      break;								\
    case 2:								\
      switch(nu){							\
      case 0:								\
	CALL_FUNCTION(2,0,1);						\
	break;								\
      case 1:								\
	CALL_FUNCTION(2,1,1); 						\
	break;								\
      case 2:								\
	printf("ERROR: invalid direction combination\n"); exit(1);	\
	break;								\
      case 3:								\
	CALL_FUNCTION(2,3,1);						\
	break;								\
      }									\
      break;								\
    case 3:								\
      switch(nu){							\
      case 0:								\
	CALL_FUNCTION(3,0,1);						\
	break;								\
      case 1:								\
	CALL_FUNCTION(3,1,1);						\
	break;								\
      case 2:								\
	CALL_FUNCTION(3,2,1);						\
	break;								\
      case 3:								\
	printf("ERROR: invalid direction combination\n"); exit(1);	\
	break;								\
      }									\
      break;								\
    }									\
  }else{								\
    switch(mu) {							\
    case 0:								\
      switch(nu){							\
      case 0:								\
	printf("ERROR: invalid direction combination\n"); exit(1);	\
	break;								\
      case 1:								\
	CALL_FUNCTION(0,1,0);						\
	break;								\
      case 2:								\
	CALL_FUNCTION(0,2,0);						\
	break;								\
      case 3:								\
	CALL_FUNCTION(0,3,0);						\
	break;								\
      }									\
      break;								\
    case 1:								\
      switch(nu){							\
      case 0:								\
	CALL_FUNCTION(1,0,0);						\
	break;								\
      case 1:								\
	printf("ERROR: invalid direction combination\n"); exit(1);	\
	break;								\
      case 2:								\
	CALL_FUNCTION(1,2,0);						\
	break;								\
      case 3:								\
	CALL_FUNCTION(1,3,0); 						\
	break;								\
      }									\
      break;								\
    case 2:								\
      switch(nu){							\
      case 0:								\
	CALL_FUNCTION(2,0,0);						\
	break;								\
      case 1:								\
	CALL_FUNCTION(2,1,0);						\
	break;								\
      case 2:								\
	printf("ERROR: invalid direction combination\n"); exit(1);	\
	break;								\
      case 3:								\
	CALL_FUNCTION(2,3,0);						\
	break;								\
      }									\
      break;								\
    case 3:								\
      switch(nu){							\
      case 0:								\
	CALL_FUNCTION(3,0,0);						\
	break;								\
      case 1:								\
	CALL_FUNCTION(3,1,0);						\
	break;								\
      case 2:								\
	CALL_FUNCTION(3,2,0); 						\
	break;								\
      case 3:								\
	printf("ERROR: invalid direction combination\n"); exit(1);	\
	break;								\
      }									\
      break;								\
    }									\
  }

void siteComputeGenStapleParityKernel(void* staple_even, void* staple_odd, 
				      void* sitelink_even, void* sitelink_odd, 
				      void* fatlink_even, void* fatlink_odd,	
				      int mu, int nu, double mycoeff,
				      QudaReconstructType recon, QudaPrecision prec,
				      dim3 halfGridDim,  llfat_kernel_param_t kparam,
				      cudaStream_t* stream)
{

  //compute even and odd
  
#define  CALL_FUNCTION(mu, nu)						\
  if (prec == QUDA_DOUBLE_PRECISION){					\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_siteComputeGenStapleParity18Kernel<mu,nu, 0>		\
	<<<halfGridDim, blockDim, 0, *stream>>>((double2*)staple_even, (double2*)staple_odd, \
						(double2*)sitelink_even, (double2*)sitelink_odd, \
						(double2*)fatlink_even, (double2*)fatlink_odd, \
						(double)mycoeff, kparam);	\
      do_siteComputeGenStapleParity18Kernel<mu,nu, 1>		\
	<<<halfGridDim, blockDim, 0, *stream>>>((double2*)staple_odd, (double2*)staple_even, \
						(double2*)sitelink_odd, (double2*)sitelink_even, \
						(double2*)fatlink_odd, (double2*)fatlink_even, \
						(double)mycoeff, kparam);	\
    }else{								\
      do_siteComputeGenStapleParity12Kernel<mu,nu, 0>		\
	<<<halfGridDim, blockDim, 0, *stream>>>((double2*)staple_even, (double2*)staple_odd, \
						(double2*)sitelink_even, (double2*)sitelink_odd, \
						(double2*)fatlink_even, (double2*)fatlink_odd, \
						(double)mycoeff, kparam);	\
      do_siteComputeGenStapleParity12Kernel<mu,nu, 1>		\
	<<<halfGridDim, blockDim, 0, *stream>>>((double2*)staple_odd, (double2*)staple_even, \
						(double2*)sitelink_odd, (double2*)sitelink_even, \
						(double2*)fatlink_odd, (double2*)fatlink_even, \
						(double)mycoeff, kparam);	\
    }									\
  }else {								\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_siteComputeGenStapleParity18Kernel<mu,nu, 0>		\
	<<<halfGridDim, blockDim, 0, *stream>>>((float2*)staple_even, (float2*)staple_odd, \
						(float2*)sitelink_even, (float2*)sitelink_odd, \
						(float2*)fatlink_even, (float2*)fatlink_odd, \
						(float)mycoeff, kparam);	\
      do_siteComputeGenStapleParity18Kernel<mu,nu, 1>		\
	<<<halfGridDim, blockDim, 0, *stream>>>((float2*)staple_odd, (float2*)staple_even, \
						(float2*)sitelink_odd, (float2*)sitelink_even, \
						(float2*)fatlink_odd, (float2*)fatlink_even, \
						(float)mycoeff, kparam); \
    }else{								\
      do_siteComputeGenStapleParity12Kernel<mu,nu, 0>		\
	<<<halfGridDim, blockDim, 0, *stream>>>((float2*)staple_even, (float2*)staple_odd, \
						(float2*)sitelink_even, (float2*)sitelink_odd, \
						(float2*)fatlink_even, (float2*)fatlink_odd, \
						(float)mycoeff, kparam); \
      do_siteComputeGenStapleParity12Kernel<mu,nu, 1>		\
	<<<halfGridDim, blockDim, 0, *stream>>>((float2*)staple_odd, (float2*)staple_even, \
						(float2*)sitelink_odd, (float2*)sitelink_even, \
						(float2*)fatlink_odd, (float2*)fatlink_even, \
						(float)mycoeff, kparam); \
    }									\
  }
  

  dim3 blockDim(BLOCK_DIM , 1, 1);  
  ENUMERATE_FUNCS(mu,nu);  

#undef CALL_FUNCTION
    
    
}


void
computeGenStapleFieldParityKernel(void* staple_even, void* staple_odd, 
				  void* sitelink_even, void* sitelink_odd,
				  void* fatlink_even, void* fatlink_odd,			    
				  void* mulink_even, void* mulink_odd, 
				  int mu, int nu, int save_staple,
				  double mycoeff,
				  QudaReconstructType recon, QudaPrecision prec,
				  dim3 halfGridDim, llfat_kernel_param_t kparam,
				  cudaStream_t* stream)
{

#define  CALL_FUNCTION(mu, nu, save_staple)				\
  if (prec == QUDA_DOUBLE_PRECISION){					\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_computeGenStapleFieldParity18Kernel<mu,nu, 0, save_staple>	\
	<<<halfGridDim, blockDim, 0, *stream>>>((double2*)staple_even, (double2*)staple_odd, \
						(double2*)sitelink_even, (double2*)sitelink_odd, \
						(double2*)fatlink_even, (double2*)fatlink_odd, \
						(double2*)mulink_even, (double2*)mulink_odd, \
						(double)mycoeff, kparam); \
      do_computeGenStapleFieldParity18Kernel<mu,nu, 1, save_staple> \
	<<<halfGridDim, blockDim, 0, *stream>>>((double2*)staple_odd, (double2*)staple_even, \
						(double2*)sitelink_odd, (double2*)sitelink_even, \
						(double2*)fatlink_odd, (double2*)fatlink_even, \
						(double2*)mulink_odd, (double2*)mulink_even, \
						(double)mycoeff, kparam); \
    }else{								\
      do_computeGenStapleFieldParity12Kernel<mu,nu, 0, save_staple> \
	<<<halfGridDim, blockDim, 0, *stream>>>((double2*)staple_even, (double2*)staple_odd, \
						(double2*)sitelink_even, (double2*)sitelink_odd, \
						(double2*)fatlink_even, (double2*)fatlink_odd, \
						(double2*)mulink_even, (double2*)mulink_odd, \
						(double)mycoeff, kparam); \
      do_computeGenStapleFieldParity12Kernel<mu,nu, 1, save_staple> \
	<<<halfGridDim, blockDim, 0, *stream>>>((double2*)staple_odd, (double2*)staple_even, \
						(double2*)sitelink_odd, (double2*)sitelink_even, \
						(double2*)fatlink_odd, (double2*)fatlink_even, \
						(double2*)mulink_odd, (double2*)mulink_even, \
						(double)mycoeff, kparam); \
    }									\
  }else{								\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_computeGenStapleFieldParity18Kernel<mu,nu, 0, save_staple> \
	<<<halfGridDim, blockDim, 0, *stream>>>((float2*)staple_even, (float2*)staple_odd, \
						(float2*)sitelink_even, (float2*)sitelink_odd, \
						(float2*)fatlink_even, (float2*)fatlink_odd, \
						(float2*)mulink_even, (float2*)mulink_odd, \
						(float)mycoeff, kparam); \
      do_computeGenStapleFieldParity18Kernel<mu,nu, 1, save_staple> \
	<<<halfGridDim, blockDim, 0, *stream>>>((float2*)staple_odd, (float2*)staple_even, \
						(float2*)sitelink_odd, (float2*)sitelink_even, \
						(float2*)fatlink_odd, (float2*)fatlink_even, \
						(float2*)mulink_odd, (float2*)mulink_even, \
						(float)mycoeff, kparam); \
    }else{								\
      do_computeGenStapleFieldParity12Kernel<mu,nu, 0, save_staple>	\
	<<<halfGridDim, blockDim, 0, *stream>>>((float2*)staple_even, (float2*)staple_odd, \
						(float2*)sitelink_even, (float2*)sitelink_odd, \
						(float2*)fatlink_even, (float2*)fatlink_odd, \
						(float2*)mulink_even, (float2*)mulink_odd, \
						(float)mycoeff, kparam); \
      do_computeGenStapleFieldParity12Kernel<mu,nu, 1, save_staple> \
	<<<halfGridDim, blockDim, 0, *stream>>>((float2*)staple_odd, (float2*)staple_even, \
						(float2*)sitelink_odd, (float2*)sitelink_even, \
						(float2*)fatlink_odd, (float2*)fatlink_even, \
						(float2*)mulink_odd, (float2*)mulink_even, \
						(float)mycoeff, kparam); \
    }									\
  }
  
  BIND_MU_LINK();
  dim3 blockDim(BLOCK_DIM , 1, 1);
  ENUMERATE_FUNCS_SAVE(mu,nu,save_staple);

  UNBIND_MU_LINK();

#undef CALL_FUNCTION 
    
}


void siteComputeGenStapleParityKernel_ex(void* staple_even, void* staple_odd, 
					 void* sitelink_even, void* sitelink_odd, 
					 void* fatlink_even, void* fatlink_odd,	
					 int mu, int nu, double mycoeff,
					 QudaReconstructType recon, QudaPrecision prec,
					 llfat_kernel_param_t kparam)
{
  
  //compute even and odd
  dim3 blockDim = kparam.blockDim;
  dim3 halfGridDim = kparam.halfGridDim;
  int sbytes_dp = blockDim.x*5*sizeof(double2);
  int sbytes_sp = blockDim.x*5*sizeof(float2);
  
#define  CALL_FUNCTION(mu, nu)						\
  if (prec == QUDA_DOUBLE_PRECISION){					\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_siteComputeGenStapleParity18Kernel_ex<mu,nu, 0>		\
	<<<halfGridDim, blockDim, sbytes_dp>>>((double2*)staple_even, (double2*)staple_odd, \
					       (double2*)sitelink_even, (double2*)sitelink_odd, \
					       (double2*)fatlink_even, (double2*)fatlink_odd, \
					       (double)mycoeff, kparam); \
      do_siteComputeGenStapleParity18Kernel_ex<mu,nu, 1>		\
	<<<halfGridDim, blockDim, sbytes_dp>>>((double2*)staple_odd, (double2*)staple_even, \
					       (double2*)sitelink_odd, (double2*)sitelink_even, \
					       (double2*)fatlink_odd, (double2*)fatlink_even, \
					       (double)mycoeff, kparam); \
    }else{								\
      do_siteComputeGenStapleParity12Kernel_ex<mu,nu, 0>		\
	<<<halfGridDim, blockDim, sbytes_dp>>>((double2*)staple_even, (double2*)staple_odd, \
					       (double2*)sitelink_even, (double2*)sitelink_odd, \
					       (double2*)fatlink_even, (double2*)fatlink_odd, \
					       (double)mycoeff, kparam); \
      do_siteComputeGenStapleParity12Kernel_ex<mu,nu, 1>		\
	<<<halfGridDim, blockDim, sbytes_dp>>>((double2*)staple_odd, (double2*)staple_even, \
					       (double2*)sitelink_odd, (double2*)sitelink_even, \
					       (double2*)fatlink_odd, (double2*)fatlink_even, \
					       (double)mycoeff, kparam); \
    }									\
  }else {								\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_siteComputeGenStapleParity18Kernel_ex<mu,nu, 0>		\
	<<<halfGridDim, blockDim, sbytes_sp>>>((float2*)staple_even, (float2*)staple_odd, \
					       (float2*)sitelink_even, (float2*)sitelink_odd, \
					       (float2*)fatlink_even, (float2*)fatlink_odd, \
					       (float)mycoeff, kparam);	\
      do_siteComputeGenStapleParity18Kernel_ex<mu,nu, 1>		\
	<<<halfGridDim, blockDim, sbytes_sp>>>((float2*)staple_odd, (float2*)staple_even, \
					       (float2*)sitelink_odd, (float2*)sitelink_even, \
					       (float2*)fatlink_odd, (float2*)fatlink_even, \
					       (float)mycoeff, kparam);	\
    }else{								\
      do_siteComputeGenStapleParity12Kernel_ex<mu,nu, 0>		\
	<<<halfGridDim, blockDim, sbytes_sp>>>((float2*)staple_even, (float2*)staple_odd, \
					       (float2*)sitelink_even, (float2*)sitelink_odd, \
					       (float2*)fatlink_even, (float2*)fatlink_odd, \
					       (float)mycoeff, kparam);	\
      do_siteComputeGenStapleParity12Kernel_ex<mu,nu, 1>		\
	<<<halfGridDim, blockDim, sbytes_sp>>>((float2*)staple_odd, (float2*)staple_even, \
					       (float2*)sitelink_odd, (float2*)sitelink_even, \
					       (float2*)fatlink_odd, (float2*)fatlink_even, \
					       (float)mycoeff, kparam);	\
    }									\
  }
  
  
  ENUMERATE_FUNCS(mu,nu);  

#undef CALL_FUNCTION
    
    
}



void
computeGenStapleFieldParityKernel_ex(void* staple_even, void* staple_odd, 
				     void* sitelink_even, void* sitelink_odd,
				     void* fatlink_even, void* fatlink_odd,			    
				     void* mulink_even, void* mulink_odd, 
				     int mu, int nu, int save_staple,
				     double mycoeff,
				     QudaReconstructType recon, QudaPrecision prec,
				     llfat_kernel_param_t kparam)
{
  
  dim3 blockDim = kparam.blockDim;
  dim3 halfGridDim= kparam.halfGridDim;
  
  int sbytes_dp = blockDim.x*5*sizeof(double2);
  int sbytes_sp = blockDim.x*5*sizeof(float2);

#define  CALL_FUNCTION(mu, nu, save_staple)				\
  if (prec == QUDA_DOUBLE_PRECISION){					\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_computeGenStapleFieldParity18Kernel_ex<mu,nu, 0, save_staple>	\
	<<<halfGridDim, blockDim, sbytes_dp>>>((double2*)staple_even, (double2*)staple_odd, \
					       (double2*)sitelink_even, (double2*)sitelink_odd, \
					       (double2*)fatlink_even, (double2*)fatlink_odd, \
					       (double2*)mulink_even, (double2*)mulink_odd, \
					       (double)mycoeff, kparam); \
      do_computeGenStapleFieldParity18Kernel_ex<mu,nu, 1, save_staple>	\
	<<<halfGridDim, blockDim, sbytes_dp>>>((double2*)staple_odd, (double2*)staple_even, \
					       (double2*)sitelink_odd, (double2*)sitelink_even, \
					       (double2*)fatlink_odd, (double2*)fatlink_even, \
					       (double2*)mulink_odd, (double2*)mulink_even, \
					       (double)mycoeff, kparam); \
    }else{								\
      do_computeGenStapleFieldParity12Kernel_ex<mu,nu, 0, save_staple>	\
	<<<halfGridDim, blockDim, sbytes_dp>>>((double2*)staple_even, (double2*)staple_odd, \
					       (double2*)sitelink_even, (double2*)sitelink_odd, \
					       (double2*)fatlink_even, (double2*)fatlink_odd, \
					       (double2*)mulink_even, (double2*)mulink_odd, \
					       (double)mycoeff, kparam); \
      do_computeGenStapleFieldParity12Kernel_ex<mu,nu, 1, save_staple>	\
	<<<halfGridDim, blockDim, sbytes_dp>>>((double2*)staple_odd, (double2*)staple_even, \
					       (double2*)sitelink_odd, (double2*)sitelink_even, \
					       (double2*)fatlink_odd, (double2*)fatlink_even, \
					       (double2*)mulink_odd, (double2*)mulink_even, \
					       (double)mycoeff, kparam); \
    }									\
  }else{								\
    if(recon == QUDA_RECONSTRUCT_NO){					\
      do_computeGenStapleFieldParity18Kernel_ex<mu,nu, 0, save_staple>	\
	<<<halfGridDim, blockDim, sbytes_sp>>>((float2*)staple_even, (float2*)staple_odd, \
					       (float2*)sitelink_even, (float2*)sitelink_odd, \
					       (float2*)fatlink_even, (float2*)fatlink_odd, \
					       (float2*)mulink_even, (float2*)mulink_odd, \
					       (float)mycoeff, kparam);	\
      do_computeGenStapleFieldParity18Kernel_ex<mu,nu, 1, save_staple>	\
	<<<halfGridDim, blockDim, sbytes_sp>>>((float2*)staple_odd, (float2*)staple_even, \
					       (float2*)sitelink_odd, (float2*)sitelink_even, \
					       (float2*)fatlink_odd, (float2*)fatlink_even, \
					       (float2*)mulink_odd, (float2*)mulink_even, \
					       (float)mycoeff, kparam);	\
    }else{								\
      do_computeGenStapleFieldParity12Kernel_ex<mu,nu, 0, save_staple>	\
	<<<halfGridDim, blockDim, sbytes_sp>>>((float2*)staple_even, (float2*)staple_odd, \
					       (float2*)sitelink_even, (float2*)sitelink_odd, \
					       (float2*)fatlink_even, (float2*)fatlink_odd, \
					       (float2*)mulink_even, (float2*)mulink_odd, \
					       (float)mycoeff, kparam);	\
      do_computeGenStapleFieldParity12Kernel_ex<mu,nu, 1, save_staple>	\
	<<<halfGridDim, blockDim, sbytes_sp>>>((float2*)staple_odd, (float2*)staple_even, \
					       (float2*)sitelink_odd, (float2*)sitelink_even, \
					       (float2*)fatlink_odd, (float2*)fatlink_even, \
					       (float2*)mulink_odd, (float2*)mulink_even, \
					       (float)mycoeff, kparam);	\
    }									\
  }
  
  BIND_MU_LINK();
  ENUMERATE_FUNCS_SAVE(mu,nu,save_staple);

  UNBIND_MU_LINK();

#undef CALL_FUNCTION 
    
}




void llfatOneLinkKernel(cudaGaugeField& cudaFatLink, cudaGaugeField& cudaSiteLink,
			cudaGaugeField& cudaStaple, cudaGaugeField& cudaStaple1,
			QudaGaugeParam* param, double* act_path_coeff)
{  
  QudaPrecision prec = cudaSiteLink.Precision();
  QudaReconstructType recon = cudaSiteLink.Reconstruct();
  
  BIND_SITE_AND_FAT_LINK;
  int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];  
  dim3 gridDim(volume/BLOCK_DIM,1,1);
  dim3 blockDim(BLOCK_DIM , 1, 1);

  staple_bytes = cudaStaple.Bytes();

  if(prec == QUDA_DOUBLE_PRECISION){
    if(recon == QUDA_RECONSTRUCT_NO){
      llfatOneLink18Kernel<<<gridDim, blockDim>>>((double2*)cudaSiteLink.Even_p(), (double2*)cudaSiteLink.Odd_p(),
						  (double2*)cudaFatLink.Even_p(), (double2*)cudaFatLink.Odd_p(),
						  (double)act_path_coeff[0], (double)act_path_coeff[5]);    
    }else{
      
      llfatOneLink12Kernel<<<gridDim, blockDim>>>((double2*)cudaSiteLink.Even_p(), (double2*)cudaSiteLink.Odd_p(),
						  (double2*)cudaFatLink.Even_p(), (double2*)cudaFatLink.Odd_p(),
						  (double)act_path_coeff[0], (double)act_path_coeff[5]);    
      
    }
  }else{ //single precision
    if(recon == QUDA_RECONSTRUCT_NO){    
      llfatOneLink18Kernel<<<gridDim, blockDim>>>((float2*)cudaSiteLink.Even_p(), (float2*)cudaSiteLink.Odd_p(),
						  (float2*)cudaFatLink.Even_p(), (float2*)cudaFatLink.Odd_p(),
						  (float)act_path_coeff[0], (float)act_path_coeff[5]);    						  
    }else{
      llfatOneLink12Kernel<<<gridDim, blockDim>>>((float2*)cudaSiteLink.Even_p(), (float2*)cudaSiteLink.Odd_p(),
						  (float2*)cudaFatLink.Even_p(), (float2*)cudaFatLink.Odd_p(),
						  (float)act_path_coeff[0], (float)act_path_coeff[5]);    
    }
  }
}



void llfatOneLinkKernel_ex(cudaGaugeField& cudaFatLink, cudaGaugeField& cudaSiteLink,
                           cudaGaugeField& cudaStaple, cudaGaugeField& cudaStaple1,
                           QudaGaugeParam* param, double* act_path_coeff,
                           llfat_kernel_param_t kparam)
{
  QudaPrecision prec = cudaSiteLink.Precision();
  QudaReconstructType recon = cudaSiteLink.Reconstruct();

  BIND_SITE_AND_FAT_LINK;

  dim3 gridDim;
  dim3 blockDim = kparam.blockDim;
  gridDim.x = 2* kparam.halfGridDim.x;
  gridDim.y = 1;
  gridDim.z = 1;
  staple_bytes = cudaStaple.Bytes();

  if(prec == QUDA_DOUBLE_PRECISION){
    if(recon == QUDA_RECONSTRUCT_NO){
      llfatOneLink18Kernel_ex<<<gridDim, blockDim>>>((double2*)cudaSiteLink.Even_p(), (double2*)cudaSiteLink.Odd_p(),
                                                     (double2*)cudaFatLink.Even_p(), (double2*)cudaFatLink.Odd_p(),
                                                     (double)act_path_coeff[0], (double)act_path_coeff[5], kparam);
    }else{
      
      llfatOneLink12Kernel_ex<<<gridDim, blockDim>>>((double2*)cudaSiteLink.Even_p(), (double2*)cudaSiteLink.Odd_p(),
                                                     (double2*)cudaFatLink.Even_p(), (double2*)cudaFatLink.Odd_p(),
                                                     (double)act_path_coeff[0], (double)act_path_coeff[5], kparam);

    }
  }else{ //single precision
    if(recon == QUDA_RECONSTRUCT_NO){
      llfatOneLink18Kernel_ex<<<gridDim, blockDim>>>((float2*)cudaSiteLink.Even_p(), (float2*)cudaSiteLink.Odd_p(),
                                                     (float2*)cudaFatLink.Even_p(), (float2*)cudaFatLink.Odd_p(),
                                                     (float)act_path_coeff[0], (float)act_path_coeff[5], kparam);
    }else{
      llfatOneLink12Kernel_ex<<<gridDim, blockDim>>>((float2*)cudaSiteLink.Even_p(), (float2*)cudaSiteLink.Odd_p(),
                                                     (float2*)cudaFatLink.Even_p(), (float2*)cudaFatLink.Odd_p(),
                                                     (float)act_path_coeff[0], (float)act_path_coeff[5], kparam);
    }
  }
}

