#include "llfat_quda.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include "read_gauge.h"
#include "gauge_quda.h"
#include "force_kernel_common_macro.h"


#define CUERR  do{ cudaError_t cuda_err;				\
        if ((cuda_err = cudaGetLastError()) != cudaSuccess) {		\
            fprintf(stderr, "ERROR: CUDA error: %s, line %d, function %s, file %s\n", \
		    cudaGetErrorString(cuda_err),  __LINE__, __FUNCTION__, __FILE__); \
            exit(cuda_err);}}while(0) 


#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#define READ_SITE_LINK_MATRIX READ_SITE_LINK_12_SINGLE

#define SITELINK0TEX siteLink0TexSingle
#define SITELINK1TEX siteLink1TexSingle


//assuming 12 construct for the site link
#define site00_re SITE0.x
#define site00_im SITE0.y
#define site01_re SITE0.z
#define site01_im SITE0.w
#define site02_re SITE1.x
#define site02_im SITE1.y
#define site10_re SITE1.z
#define site10_im SITE1.w
#define site11_re SITE2.x
#define site11_im SITE2.y
#define site12_re SITE2.z
#define site12_im SITE2.w
#define site20_re SITE3.x
#define site20_im SITE3.y
#define site21_re SITE3.z
#define site21_im SITE3.w
#define site22_re SITE4.x
#define site22_im SITE4.y


#define a00_re A0.x
#define a00_im A0.y
#define a01_re A0.z
#define a01_im A0.w
#define a02_re A1.x
#define a02_im A1.y
#define a10_re A1.z
#define a10_im A1.w
#define a11_re A2.x
#define a11_im A2.y
#define a12_re A2.z
#define a12_im A2.w
#define a20_re A3.x
#define a20_im A3.y
#define a21_re A3.z
#define a21_im A3.w
#define a22_re A4.x
#define a22_im A4.y

#define b00_re B0.x
#define b00_im B0.y
#define b01_re B0.z
#define b01_im B0.w
#define b02_re B1.x
#define b02_im B1.y
#define b10_re B1.z
#define b10_im B1.w
#define b11_re B2.x
#define b11_im B2.y
#define b12_re B2.z
#define b12_im B2.w
#define b20_re B3.x
#define b20_im B3.y
#define b21_re B3.z
#define b21_im B3.w
#define b22_re B4.x
#define b22_im B4.y

#define bb00_re BB0.x
#define bb00_im BB0.y
#define bb01_re BB1.x
#define bb01_im BB1.y
#define bb02_re BB2.x
#define bb02_im BB2.y
#define bb10_re BB3.x
#define bb10_im BB3.y
#define bb11_re BB4.x
#define bb11_im BB4.y
#define bb12_re BB5.x
#define bb12_im BB5.y
#define bb20_re BB6.x
#define bb20_im BB6.y
#define bb21_re BB7.x
#define bb21_im BB7.y
#define bb22_re BB8.x
#define bb22_im BB8.y

#define c00_re C0.x
#define c00_im C0.y
#define c01_re C0.z
#define c01_im C0.w
#define c02_re C1.x
#define c02_im C1.y
#define c10_re C1.z
#define c10_im C1.w
#define c11_re C2.x
#define c11_im C2.y
#define c12_re C2.z
#define c12_im C2.w
#define c20_re C3.x
#define c20_im C3.y
#define c21_re C3.z
#define c21_im C3.w
#define c22_re C4.x
#define c22_im C4.y

#define aT00_re (+a00_re)
#define aT00_im (-a00_im)
#define aT01_re (+a10_re)
#define aT01_im (-a10_im)
#define aT02_re (+a20_re)
#define aT02_im (-a20_im)
#define aT10_re (+a01_re)
#define aT10_im (-a01_im)
#define aT11_re (+a11_re)
#define aT11_im (-a11_im)
#define aT12_re (+a21_re)
#define aT12_im (-a21_im)
#define aT20_re (+a02_re)
#define aT20_im (-a02_im)
#define aT21_re (+a12_re)
#define aT21_im (-a12_im)
#define aT22_re (+a22_re)
#define aT22_im (-a22_im)

#define bT00_re (+b00_re)
#define bT00_im (-b00_im)
#define bT01_re (+b10_re)
#define bT01_im (-b10_im)
#define bT02_re (+b20_re)
#define bT02_im (-b20_im)
#define bT10_re (+b01_re)
#define bT10_im (-b01_im)
#define bT11_re (+b11_re)
#define bT11_im (-b11_im)
#define bT12_re (+b21_re)
#define bT12_im (-b21_im)
#define bT20_re (+b02_re)
#define bT20_im (-b02_im)
#define bT21_re (+b12_re)
#define bT21_im (-b12_im)
#define bT22_re (+b22_re)
#define bT22_im (-b22_im)

#define cT00_re (+c00_re)
#define cT00_im (-c00_im)
#define cT01_re (+c10_re)
#define cT01_im (-c10_im)
#define cT02_re (+c20_re)
#define cT02_im (-c20_im)
#define cT10_re (+c01_re)
#define cT10_im (-c01_im)
#define cT11_re (+c11_re)
#define cT11_im (-c11_im)
#define cT12_re (+c21_re)
#define cT12_im (-c21_im)
#define cT20_re (+c02_re)
#define cT20_im (-c02_im)
#define cT21_re (+c12_re)
#define cT21_im (-c12_im)
#define cT22_re (+c22_re)
#define cT22_im (-c22_im)


#define tempa00_re TEMPA0.x
#define tempa00_im TEMPA0.y
#define tempa01_re TEMPA0.z
#define tempa01_im TEMPA0.w
#define tempa02_re TEMPA1.x
#define tempa02_im TEMPA1.y
#define tempa10_re TEMPA1.z
#define tempa10_im TEMPA1.w
#define tempa11_re TEMPA2.x
#define tempa11_im TEMPA2.y
#define tempa12_re TEMPA2.z
#define tempa12_im TEMPA2.w
#define tempa20_re TEMPA3.x
#define tempa20_im TEMPA3.y
#define tempa21_re TEMPA3.z
#define tempa21_im TEMPA3.w
#define tempa22_re TEMPA4.x
#define tempa22_im TEMPA4.y

#define tempb00_re TEMPB0.x
#define tempb00_im TEMPB0.y
#define tempb01_re TEMPB0.z
#define tempb01_im TEMPB0.w
#define tempb02_re TEMPB1.x
#define tempb02_im TEMPB1.y
#define tempb10_re TEMPB1.z
#define tempb10_im TEMPB1.w
#define tempb11_re TEMPB2.x
#define tempb11_im TEMPB2.y
#define tempb12_re TEMPB2.z
#define tempb12_im TEMPB2.w
#define tempb20_re TEMPB3.x
#define tempb20_im TEMPB3.y
#define tempb21_re TEMPB3.z
#define tempb21_im TEMPB3.w
#define tempb22_re TEMPB4.x
#define tempb22_im TEMPB4.y

#define temps00_re TEMPS0[threadIdx.x].x
#define temps00_im TEMPS0[threadIdx.x].y
#define temps01_re TEMPS0[threadIdx.x].z
#define temps01_im TEMPS0[threadIdx.x].w
#define temps02_re TEMPS1[threadIdx.x].x
#define temps02_im TEMPS1[threadIdx.x].y
#define temps10_re TEMPS1[threadIdx.x].z
#define temps10_im TEMPS1[threadIdx.x].w
#define temps11_re TEMPS2[threadIdx.x].x
#define temps11_im TEMPS2[threadIdx.x].y
#define temps12_re TEMPS2[threadIdx.x].z
#define temps12_im TEMPS2[threadIdx.x].w
#define temps20_re TEMPS3[threadIdx.x].x
#define temps20_im TEMPS3[threadIdx.x].y
#define temps21_re TEMPS3[threadIdx.x].z
#define temps21_im TEMPS3[threadIdx.x].w
#define temps22_re TEMPS4[threadIdx.x].x
#define temps22_im TEMPS4[threadIdx.x].y

#define tempsT00_re (+temps00_re)
#define tempsT00_im (-temps00_im)
#define tempsT01_re (+temps10_re)
#define tempsT01_im (-temps10_im)
#define tempsT02_re (+temps20_re)
#define tempsT02_im (-temps20_im)
#define tempsT10_re (+temps01_re)
#define tempsT10_im (-temps01_im)
#define tempsT11_re (+temps11_re)
#define tempsT11_im (-temps11_im)
#define tempsT12_re (+temps21_re)
#define tempsT12_im (-temps21_im)
#define tempsT20_re (+temps02_re)
#define tempsT20_im (-temps02_im)
#define tempsT21_re (+temps12_re)
#define tempsT21_im (-temps12_im)
#define tempsT22_re (+temps22_re)
#define tempsT22_im (-temps22_im)

#define tempt00_re TEMPT0[threadIdx.x].x
#define tempt00_im TEMPT0[threadIdx.x].y
#define tempt01_re TEMPT0[threadIdx.x].z
#define tempt01_im TEMPT0[threadIdx.x].w
#define tempt02_re TEMPT1[threadIdx.x].x
#define tempt02_im TEMPT1[threadIdx.x].y
#define tempt10_re TEMPT1[threadIdx.x].z
#define tempt10_im TEMPT1[threadIdx.x].w
#define tempt11_re TEMPT2[threadIdx.x].x
#define tempt11_im TEMPT2[threadIdx.x].y
#define tempt12_re TEMPT2[threadIdx.x].z
#define tempt12_im TEMPT2[threadIdx.x].w
#define tempt20_re TEMPT3[threadIdx.x].x
#define tempt20_im TEMPT3[threadIdx.x].y
#define tempt21_re TEMPT3[threadIdx.x].z
#define tempt21_im TEMPT3[threadIdx.x].w
#define tempt22_re TEMPT4[threadIdx.x].x
#define tempt22_im TEMPT4[threadIdx.x].y

#define temptT00_re (+tempt00_re)
#define temptT00_im (-tempt00_im)
#define temptT01_re (+tempt10_re)
#define temptT01_im (-tempt10_im)
#define temptT02_re (+tempt20_re)
#define temptT02_im (-tempt20_im)
#define temptT10_re (+tempt01_re)
#define temptT10_im (-tempt01_im)
#define temptT11_re (+tempt11_re)
#define temptT11_im (-tempt11_im)
#define temptT12_re (+tempt21_re)
#define temptT12_im (-tempt21_im)
#define temptT20_re (+tempt02_re)
#define temptT20_im (-tempt02_im)
#define temptT21_re (+tempt12_re)
#define temptT21_im (-tempt12_im)
#define temptT22_re (+tempt22_re)
#define temptT22_im (-tempt22_im)

// temporaries
#define A_re SITE4.z
#define A_im SITE4.w


//fat link is not compressible
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


#define SITE_MATRIX_LOAD_TEX 1
#define MULINK_LOAD_TEX 0
#define FATLINK_LOAD_TEX 1



#if (SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_SITE_MATRIX_12_SINGLE(dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX(siteLink0TexSingle, dir, idx, var)
#define LOAD_ODD_SITE_MATRIX_12_SINGLE(dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX(siteLink1TexSingle, dir, idx, var)
#else
#define LOAD_EVEN_SITE_MATRIX_12_SINGLE(dir, idx, var) LOAD_MATRIX_12_SINGLE(sitelink_even, dir, idx, var)
#define LOAD_ODD_SITE_MATRIX_12_SINGLE(dir, idx, var) LOAD_MATRIX_12_SINGLE(sitelink_odd, dir, idx, var)
#endif



#define WRITE_FAT_MATRIX(gauge, dir, idx)do {				\
	int start_idx = idx + dir*Vhx9;					\
	gauge[start_idx] = FAT0;					\
	gauge[start_idx + Vh  ] = FAT1;					\
	gauge[start_idx + Vhx2] = FAT2;					\
	gauge[start_idx + Vhx3] = FAT3;					\
	gauge[start_idx + Vhx4] = FAT4;					\
	gauge[start_idx + Vhx5] = FAT5;					\
	gauge[start_idx + Vhx6] = FAT6;					\
	gauge[start_idx + Vhx7] = FAT7;					\
	gauge[start_idx + Vhx8] = FAT8;} while(0)			
	

#define WRITE_STAPLE_MATRIX(gauge, idx)					\
    gauge[idx] = STAPLE0;						\
    gauge[idx + Vh] = STAPLE1;						\
    gauge[idx + Vhx2] = STAPLE2;					\
    gauge[idx + Vhx3] = STAPLE3;					\
    gauge[idx + Vhx4] = STAPLE4;					\
    gauge[idx + Vhx5] = STAPLE5;					\
    gauge[idx + Vhx6] = STAPLE6;					\
    gauge[idx + Vhx7] = STAPLE7;					\
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
    


    
#define LOAD_MATRIX_18_SINGLE_FLOAT4(gauge, dir, idx, var)do{		\
	int pos0= idx + dir*Vhx9;					\
	int pos1=pos0+Vh;						\
	int pos2=pos0+Vhx2;						\
	int pos3=pos0+Vhx3;						\
	int pos4=pos0+Vhx4;						\
	int pos5=pos0+Vhx5;						\
	int pos6=pos0+Vhx6;						\
	int pos7=pos0+Vhx7;						\
	int pos8=pos0+Vhx8;						\
	var##0 = make_float4(gauge[pos0].x, gauge[pos0].y, gauge[pos1].x, gauge[pos1].y); \
	var##1 = make_float4(gauge[pos2].x, gauge[pos2].y, gauge[pos3].x, gauge[pos3].y); \
	var##2 = make_float4(gauge[pos4].x, gauge[pos4].y, gauge[pos5].x, gauge[pos5].y); \
	var##3 = make_float4(gauge[pos6].x, gauge[pos6].y, gauge[pos7].x, gauge[pos7].y); \
	var##4 = make_float4(gauge[pos8].x, gauge[pos8].y, 0, 0);	\
    } while(0)


#define LOAD_MATRIX_18_SINGLE(gauge, dir, idx, var)do{			\
	int start_pos= idx + dir*Vhx9;					\
	var##0 = gauge[start_pos];					\
	var##1 = gauge[start_pos + Vh];					\
	var##2 = gauge[start_pos + Vhx2];				\
	var##3 = gauge[start_pos + Vhx3];				\
	var##4 = gauge[start_pos + Vhx4];				\
	var##5 = gauge[start_pos + Vhx5];				\
	var##6 = gauge[start_pos + Vhx6];				\
	var##7 = gauge[start_pos + Vhx7];				\
	var##8 = gauge[start_pos + Vhx8]; } while(0)

#define LOAD_MATRIX_18_SINGLE_TEX(gauge, dir, idx, var)do{		\
	int start_pos= idx + dir*Vhx9;					\
	var##0 = tex1Dfetch(gauge, start_pos);				\
	var##1 = tex1Dfetch(gauge, start_pos + Vh);			\
	var##2 = tex1Dfetch(gauge, start_pos + Vhx2);			\
	var##3 = tex1Dfetch(gauge, start_pos + Vhx3);			\
	var##4 = tex1Dfetch(gauge, start_pos + Vhx4);			\
	var##5 = tex1Dfetch(gauge, start_pos + Vhx5);			\
	var##6 = tex1Dfetch(gauge, start_pos + Vhx6);			\
	var##7 = tex1Dfetch(gauge, start_pos + Vhx7);			\
	var##8 = tex1Dfetch(gauge, start_pos + Vhx8); } while(0)


#if (MULINK_LOAD_TEX == 1)
#define LOAD_EVEN_MULINK_MATRIX_18_SINGLE(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX(muLink0TexSingle, dir, idx, var)
#define LOAD_ODD_MULINK_MATRIX_18_SINGLE(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX(muLink1TexSingle, dir, idx, var)
#else
#define LOAD_EVEN_MULINK_MATRIX_18_SINGLE(dir, idx, var) LOAD_MATRIX_18_SINGLE(mulink_even, dir, idx, var)
#define LOAD_ODD_MULINK_MATRIX_18_SINGLE(dir, idx, var) LOAD_MATRIX_18_SINGLE(mulink_odd, dir, idx, var)
#endif

#if (FATLINK_LOAD_TEX == 1)
#define LOAD_EVEN_FAT_MATRIX_18_SINGLE(dir, idx) LOAD_MATRIX_18_SINGLE_TEX(fatGauge0TexSingle, dir, idx, FAT)
#define LOAD_ODD_FAT_MATRIX_18_SINGLE(dir, idx) LOAD_MATRIX_18_SINGLE_TEX(fatGauge1TexSingle, dir, idx, FAT)
#else
#define LOAD_EVEN_FAT_MATRIX_18_SINGLE(dir, idx) LOAD_MATRIX_18_SINGLE(fatlink_even, dir, idx, FAT)
#define LOAD_ODD_FAT_MATRIX_18_SINGLE(dir, idx)  LOAD_MATRIX_18_SINGLE(fatlink_odd, dir, idx, FAT)
#endif


#define LOAD_FAT_MATRIX_18_SINGLE(gauge, dir, idx) LOAD_MATRIX_18_SINGLE(gauge, dir, idx, FAT)




#define MULT_SU3_NN_TEST(ma, mb) do{				\
	float fa_re,fa_im, fb_re, fb_im, fc_re, fc_im;		\
	fa_re =							\
	    ma##00_re * mb##00_re - ma##00_im * mb##00_im +	\
	    ma##01_re * mb##10_re - ma##01_im * mb##10_im +	\
	    ma##02_re * mb##20_re - ma##02_im * mb##20_im;	\
	fa_im =							\
	    ma##00_re * mb##00_im + ma##00_im * mb##00_re +	\
	    ma##01_re * mb##10_im + ma##01_im * mb##10_re +	\
	    ma##02_re * mb##20_im + ma##02_im * mb##20_re;	\
	fb_re =							\
	    ma##00_re * mb##01_re - ma##00_im * mb##01_im +	\
	    ma##01_re * mb##11_re - ma##01_im * mb##11_im +	\
	    ma##02_re * mb##21_re - ma##02_im * mb##21_im;	\
	fb_im =							\
	    ma##00_re * mb##01_im + ma##00_im * mb##01_re +	\
	    ma##01_re * mb##11_im + ma##01_im * mb##11_re +	\
	    ma##02_re * mb##21_im + ma##02_im * mb##21_re;	\
	fc_re =							\
	    ma##00_re * mb##02_re - ma##00_im * mb##02_im +	\
	    ma##01_re * mb##12_re - ma##01_im * mb##12_im +	\
	    ma##02_re * mb##22_re - ma##02_im * mb##22_im;	\
	fc_im =							\
	    ma##00_re * mb##02_im + ma##00_im * mb##02_re +	\
	    ma##01_re * mb##12_im + ma##01_im * mb##12_re +	\
	    ma##02_re * mb##22_im + ma##02_im * mb##22_re;	\
	ma##00_re = fa_re;					\
	ma##00_im = fa_im;					\
	ma##01_re = fb_re;					\
	ma##01_im = fb_im;					\
	ma##02_re = fc_re;					\
	ma##02_im = fc_im;					\
	fa_re =							\
	    ma##10_re * mb##00_re - ma##10_im * mb##00_im +	\
	    ma##11_re * mb##10_re - ma##11_im * mb##10_im +	\
	    ma##12_re * mb##20_re - ma##12_im * mb##20_im;	\
	fa_im =							\
	    ma##10_re * mb##00_im + ma##10_im * mb##00_re +	\
	    ma##11_re * mb##10_im + ma##11_im * mb##10_re +	\
	    ma##12_re * mb##20_im + ma##12_im * mb##20_re;	\
	fb_re =							\
	    ma##10_re * mb##01_re - ma##10_im * mb##01_im +	\
	    ma##11_re * mb##11_re - ma##11_im * mb##11_im +	\
	    ma##12_re * mb##21_re - ma##12_im * mb##21_im;	\
	fb_im =							\
	    ma##10_re * mb##01_im + ma##10_im * mb##01_re +	\
	    ma##11_re * mb##11_im + ma##11_im * mb##11_re +	\
	    ma##12_re * mb##21_im + ma##12_im * mb##21_re;	\
	fc_re =							\
	    ma##10_re * mb##02_re - ma##10_im * mb##02_im +	\
	    ma##11_re * mb##12_re - ma##11_im * mb##12_im +	\
	    ma##12_re * mb##22_re - ma##12_im * mb##22_im;	\
	fc_im =							\
	    ma##10_re * mb##02_im + ma##10_im * mb##02_re +	\
	    ma##11_re * mb##12_im + ma##11_im * mb##12_re +	\
	    ma##12_re * mb##22_im + ma##12_im * mb##22_re;	\
	ma##10_re = fa_re;					\
	ma##10_im = fa_im;					\
	ma##11_re = fb_re;					\
	ma##11_im = fb_im;					\
	ma##12_re = fc_re;					\
	ma##12_im = fc_im;					\
	fa_re =							\
	    ma##20_re * mb##00_re - ma##20_im * mb##00_im +	\
	    ma##21_re * mb##10_re - ma##21_im * mb##10_im +	\
	    ma##22_re * mb##20_re - ma##22_im * mb##20_im;	\
	fa_im =							\
	    ma##20_re * mb##00_im + ma##20_im * mb##00_re +	\
	    ma##21_re * mb##10_im + ma##21_im * mb##10_re +	\
	    ma##22_re * mb##20_im + ma##22_im * mb##20_re;	\
	fb_re =							\
	    ma##20_re * mb##01_re - ma##20_im * mb##01_im +	\
	    ma##21_re * mb##11_re - ma##21_im * mb##11_im +	\
	    ma##22_re * mb##21_re - ma##22_im * mb##21_im;	\
	fb_im =							\
	    ma##20_re * mb##01_im + ma##20_im * mb##01_re +	\
	    ma##21_re * mb##11_im + ma##21_im * mb##11_re +	\
	    ma##22_re * mb##21_im + ma##22_im * mb##21_re;	\
	fc_re =							\
	    ma##20_re * mb##02_re - ma##20_im * mb##02_im +	\
	    ma##21_re * mb##12_re - ma##21_im * mb##12_im +	\
	    ma##22_re * mb##22_re - ma##22_im * mb##22_im;	\
	fc_im =							\
	    ma##20_re * mb##02_im + ma##20_im * mb##02_re +	\
	    ma##21_re * mb##12_im + ma##21_im * mb##12_re +	\
	    ma##22_re * mb##22_im + ma##22_im * mb##22_re;	\
	ma##20_re = fa_re;					\
	ma##20_im = fa_im;					\
	ma##21_re = fb_re;					\
	ma##21_im = fb_im;					\
	ma##22_re = fc_re;					\
	ma##22_im = fc_im;					\
    }while(0)



#define ADD_MULT_SU3_NN(ma, mb, mc, md)				\
    md##00_re =							\
	ma##00_re * mb##00_re - ma##00_im * mb##00_im +		\
	ma##01_re * mb##10_re - ma##01_im * mb##10_im +		\
	ma##02_re * mb##20_re - ma##02_im * mb##20_im +		\
	mc##00_re;						\
    md##00_im =							\
	ma##00_re * mb##00_im + ma##00_im * mb##00_re +		\
	ma##01_re * mb##10_im + ma##01_im * mb##10_re +		\
	ma##02_re * mb##20_im + ma##02_im * mb##20_re +		\
	mc##00_im;						\
    md##10_re =							\
	ma##10_re * mb##00_re - ma##10_im * mb##00_im +		\
	ma##11_re * mb##10_re - ma##11_im * mb##10_im +		\
	ma##12_re * mb##20_re - ma##12_im * mb##20_im +		\
	mc##10_re;						\
    md##10_im =							\
	ma##10_re * mb##00_im + ma##10_im * mb##00_re +		\
	ma##11_re * mb##10_im + ma##11_im * mb##10_re +		\
	ma##12_re * mb##20_im + ma##12_im * mb##20_re +		\
	mc##10_im;						\
    md##20_re =							\
	ma##20_re * mb##00_re - ma##20_im * mb##00_im +		\
	ma##21_re * mb##10_re - ma##21_im * mb##10_im +		\
	ma##22_re * mb##20_re - ma##22_im * mb##20_im +		\
	mc##20_re;						\
    md##20_im =							\
	ma##20_re * mb##00_im + ma##20_im * mb##00_re +		\
	ma##21_re * mb##10_im + ma##21_im * mb##10_re +		\
	ma##22_re * mb##20_im + ma##22_im * mb##20_re +		\
	mc##20_im;						\
    md##01_re =							\
	ma##00_re * mb##01_re - ma##00_im * mb##01_im +		\
	ma##01_re * mb##11_re - ma##01_im * mb##11_im +		\
	ma##02_re * mb##21_re - ma##02_im * mb##21_im +		\
	mc##01_re;						\
    md##01_im =							\
	ma##00_re * mb##01_im + ma##00_im * mb##01_re +		\
	ma##01_re * mb##11_im + ma##01_im * mb##11_re +		\
	ma##02_re * mb##21_im + ma##02_im * mb##21_re +		\
	mc##01_im;						\
    md##11_re =							\
	ma##10_re * mb##01_re - ma##10_im * mb##01_im +		\
	ma##11_re * mb##11_re - ma##11_im * mb##11_im +		\
	ma##12_re * mb##21_re - ma##12_im * mb##21_im +		\
	mc##11_re;						\
    md##11_im =							\
	ma##10_re * mb##01_im + ma##10_im * mb##01_re +		\
	ma##11_re * mb##11_im + ma##11_im * mb##11_re +		\
	ma##12_re * mb##21_im + ma##12_im * mb##21_re +		\
	mc##11_im;						\
    md##21_re =							\
	ma##20_re * mb##01_re - ma##20_im * mb##01_im +		\
	ma##21_re * mb##11_re - ma##21_im * mb##11_im +		\
	ma##22_re * mb##21_re - ma##22_im * mb##21_im +		\
	mc##21_re;						\
    md##21_im =							\
	ma##20_re * mb##01_im + ma##20_im * mb##01_re +		\
	ma##21_re * mb##11_im + ma##21_im * mb##11_re +		\
	ma##22_re * mb##21_im + ma##22_im * mb##21_re +		\
	mc##21_im;						\
    md##02_re =							\
	ma##00_re * mb##02_re - ma##00_im * mb##02_im +		\
	ma##01_re * mb##12_re - ma##01_im * mb##12_im +		\
	ma##02_re * mb##22_re - ma##02_im * mb##22_im +		\
	mc##02_re;						\
    md##02_im =							\
	ma##00_re * mb##02_im + ma##00_im * mb##02_re +		\
	ma##01_re * mb##12_im + ma##01_im * mb##12_re +		\
	ma##02_re * mb##22_im + ma##02_im * mb##22_re +		\
	mc##02_im;						\
    md##12_re =							\
	ma##10_re * mb##02_re - ma##10_im * mb##02_im +		\
	ma##11_re * mb##12_re - ma##11_im * mb##12_im +		\
	ma##12_re * mb##22_re - ma##12_im * mb##22_im +		\
	mc##12_re;						\
    md##12_im =							\
	ma##10_re * mb##02_im + ma##10_im * mb##02_re +		\
	ma##11_re * mb##12_im + ma##11_im * mb##12_re +		\
	ma##12_re * mb##22_im + ma##12_im * mb##22_re +		\
	mc##12_im;						\
    md##22_re =							\
	ma##20_re * mb##02_re - ma##20_im * mb##02_im +		\
	ma##21_re * mb##12_re - ma##21_im * mb##12_im +		\
	ma##22_re * mb##22_re - ma##22_im * mb##22_im +		\
	mc##22_re;						\
    md##22_im =							\
	ma##20_re * mb##02_im + ma##20_im * mb##02_re +		\
	ma##21_re * mb##12_im + ma##21_im * mb##12_re +		\
	ma##22_re * mb##22_im + ma##22_im * mb##22_re +		\
	mc##22_im;						\


#define MULT_SU3_NA_TEST(ma, mb)	do{				\
	float fa_re, fa_im, fb_re, fb_im, fc_re, fc_im;			\
	fa_re =								\
	    ma##00_re * mb##T00_re - ma##00_im * mb##T00_im +		\
	    ma##01_re * mb##T10_re - ma##01_im * mb##T10_im +		\
	    ma##02_re * mb##T20_re - ma##02_im * mb##T20_im;		\
	fa_im =								\
	    ma##00_re * mb##T00_im + ma##00_im * mb##T00_re +		\
	    ma##01_re * mb##T10_im + ma##01_im * mb##T10_re +		\
	    ma##02_re * mb##T20_im + ma##02_im * mb##T20_re;		\
	fb_re =								\
	    ma##00_re * mb##T01_re - ma##00_im * mb##T01_im +		\
	    ma##01_re * mb##T11_re - ma##01_im * mb##T11_im +		\
	    ma##02_re * mb##T21_re - ma##02_im * mb##T21_im;		\
	fb_im =								\
	    ma##00_re * mb##T01_im + ma##00_im * mb##T01_re +		\
	    ma##01_re * mb##T11_im + ma##01_im * mb##T11_re +		\
	    ma##02_re * mb##T21_im + ma##02_im * mb##T21_re;		\
	fc_re =								\
	    ma##00_re * mb##T02_re - ma##00_im * mb##T02_im +		\
	    ma##01_re * mb##T12_re - ma##01_im * mb##T12_im +		\
	    ma##02_re * mb##T22_re - ma##02_im * mb##T22_im;		\
	fc_im =								\
	    ma##00_re * mb##T02_im + ma##00_im * mb##T02_re +		\
	    ma##01_re * mb##T12_im + ma##01_im * mb##T12_re +		\
	    ma##02_re * mb##T22_im + ma##02_im * mb##T22_re;		\
	ma##00_re = fa_re;						\
	ma##00_im = fa_im;						\
	ma##01_re = fb_re;						\
	ma##01_im = fb_im;						\
	ma##02_re = fc_re;						\
	ma##02_im = fc_im;						\
	fa_re =								\
	    ma##10_re * mb##T00_re - ma##10_im * mb##T00_im +		\
	    ma##11_re * mb##T10_re - ma##11_im * mb##T10_im +		\
	    ma##12_re * mb##T20_re - ma##12_im * mb##T20_im;		\
	fa_im =								\
	    ma##10_re * mb##T00_im + ma##10_im * mb##T00_re +		\
	    ma##11_re * mb##T10_im + ma##11_im * mb##T10_re +		\
	    ma##12_re * mb##T20_im + ma##12_im * mb##T20_re;		\
	fb_re =								\
	    ma##10_re * mb##T01_re - ma##10_im * mb##T01_im +		\
	    ma##11_re * mb##T11_re - ma##11_im * mb##T11_im +		\
	    ma##12_re * mb##T21_re - ma##12_im * mb##T21_im;		\
	fb_im =								\
	    ma##10_re * mb##T01_im + ma##10_im * mb##T01_re +		\
	    ma##11_re * mb##T11_im + ma##11_im * mb##T11_re +		\
	    ma##12_re * mb##T21_im + ma##12_im * mb##T21_re;		\
	fc_re =								\
	    ma##10_re * mb##T02_re - ma##10_im * mb##T02_im +		\
	    ma##11_re * mb##T12_re - ma##11_im * mb##T12_im +		\
	    ma##12_re * mb##T22_re - ma##12_im * mb##T22_im;		\
	fc_im =								\
	    ma##10_re * mb##T02_im + ma##10_im * mb##T02_re +		\
	    ma##11_re * mb##T12_im + ma##11_im * mb##T12_re +		\
	    ma##12_re * mb##T22_im + ma##12_im * mb##T22_re;		\
	ma##10_re = fa_re;						\
	ma##10_im = fa_im;						\
	ma##11_re = fb_re;						\
	ma##11_im = fb_im;						\
	ma##12_re = fc_re;						\
	ma##12_im = fc_im;						\
	fa_re =								\
	    ma##20_re * mb##T00_re - ma##20_im * mb##T00_im +		\
	    ma##21_re * mb##T10_re - ma##21_im * mb##T10_im +		\
	    ma##22_re * mb##T20_re - ma##22_im * mb##T20_im;		\
	fa_im =								\
	    ma##20_re * mb##T00_im + ma##20_im * mb##T00_re +		\
	    ma##21_re * mb##T10_im + ma##21_im * mb##T10_re +		\
	    ma##22_re * mb##T20_im + ma##22_im * mb##T20_re;		\
	fb_re =								\
	    ma##20_re * mb##T01_re - ma##20_im * mb##T01_im +		\
	    ma##21_re * mb##T11_re - ma##21_im * mb##T11_im +		\
	    ma##22_re * mb##T21_re - ma##22_im * mb##T21_im;		\
	fb_im =								\
	    ma##20_re * mb##T01_im + ma##20_im * mb##T01_re +		\
	    ma##21_re * mb##T11_im + ma##21_im * mb##T11_re +		\
	    ma##22_re * mb##T21_im + ma##22_im * mb##T21_re;		\
	fc_re =								\
	    ma##20_re * mb##T02_re - ma##20_im * mb##T02_im +		\
	    ma##21_re * mb##T12_re - ma##21_im * mb##T12_im +		\
	    ma##22_re * mb##T22_re - ma##22_im * mb##T22_im;		\
	fc_im =								\
	    ma##20_re * mb##T02_im + ma##20_im * mb##T02_re +		\
	    ma##21_re * mb##T12_im + ma##21_im * mb##T12_re +		\
	    ma##22_re * mb##T22_im + ma##22_im * mb##T22_re;		\
	ma##20_re = fa_re;						\
	ma##20_im = fa_im;						\
	ma##21_re = fb_re;						\
	ma##21_im = fb_im;						\
	ma##22_re = fc_re;						\
	ma##22_im = fc_im;						\
    }while(0)



#define MULT_SU3_AN_TEST(ma, mb)	do{				\
	float fa_re, fa_im, fb_re, fb_im, fc_re, fc_im;			\
	fa_re =								\
	    ma##T00_re * mb##00_re - ma##T00_im * mb##00_im +		\
	    ma##T01_re * mb##10_re - ma##T01_im * mb##10_im +		\
	    ma##T02_re * mb##20_re - ma##T02_im * mb##20_im;		\
	fa_im =								\
	    ma##T00_re * mb##00_im + ma##T00_im * mb##00_re +		\
	    ma##T01_re * mb##10_im + ma##T01_im * mb##10_re +		\
	    ma##T02_re * mb##20_im + ma##T02_im * mb##20_re;		\
	fb_re =								\
	    ma##T10_re * mb##00_re - ma##T10_im * mb##00_im +		\
	    ma##T11_re * mb##10_re - ma##T11_im * mb##10_im +		\
	    ma##T12_re * mb##20_re - ma##T12_im * mb##20_im;		\
	fb_im =								\
	    ma##T10_re * mb##00_im + ma##T10_im * mb##00_re +		\
	    ma##T11_re * mb##10_im + ma##T11_im * mb##10_re +		\
	    ma##T12_re * mb##20_im + ma##T12_im * mb##20_re;		\
	fc_re =								\
	    ma##T20_re * mb##00_re - ma##T20_im * mb##00_im +		\
	    ma##T21_re * mb##10_re - ma##T21_im * mb##10_im +		\
	    ma##T22_re * mb##20_re - ma##T22_im * mb##20_im;		\
	fc_im =								\
	    ma##T20_re * mb##00_im + ma##T20_im * mb##00_re +		\
	    ma##T21_re * mb##10_im + ma##T21_im * mb##10_re +		\
	    ma##T22_re * mb##20_im + ma##T22_im * mb##20_re;		\
	mb##00_re = fa_re;						\
	mb##00_im = fa_im;						\
	mb##10_re = fb_re;						\
	mb##10_im = fb_im;						\
	mb##20_re = fc_re;						\
	mb##20_im = fc_im;						\
	fa_re =								\
	    ma##T00_re * mb##01_re - ma##T00_im * mb##01_im +		\
	    ma##T01_re * mb##11_re - ma##T01_im * mb##11_im +		\
	    ma##T02_re * mb##21_re - ma##T02_im * mb##21_im;		\
	fa_im =								\
	    ma##T00_re * mb##01_im + ma##T00_im * mb##01_re +		\
	    ma##T01_re * mb##11_im + ma##T01_im * mb##11_re +		\
	    ma##T02_re * mb##21_im + ma##T02_im * mb##21_re;		\
	fb_re =								\
	    ma##T10_re * mb##01_re - ma##T10_im * mb##01_im +		\
	    ma##T11_re * mb##11_re - ma##T11_im * mb##11_im +		\
	    ma##T12_re * mb##21_re - ma##T12_im * mb##21_im;		\
	fb_im =								\
	    ma##T10_re * mb##01_im + ma##T10_im * mb##01_re +		\
	    ma##T11_re * mb##11_im + ma##T11_im * mb##11_re +		\
	    ma##T12_re * mb##21_im + ma##T12_im * mb##21_re;		\
	fc_re =								\
	    ma##T20_re * mb##01_re - ma##T20_im * mb##01_im +		\
	    ma##T21_re * mb##11_re - ma##T21_im * mb##11_im +		\
	    ma##T22_re * mb##21_re - ma##T22_im * mb##21_im;		\
	fc_im =								\
	    ma##T20_re * mb##01_im + ma##T20_im * mb##01_re +		\
	    ma##T21_re * mb##11_im + ma##T21_im * mb##11_re +		\
	    ma##T22_re * mb##21_im + ma##T22_im * mb##21_re;		\
	mb##01_re = fa_re;						\
	mb##01_im = fa_im;						\
	mb##11_re = fb_re;						\
	mb##11_im = fb_im;						\
	mb##21_re = fc_re;						\
	mb##21_im = fc_im;						\
	fa_re =								\
	    ma##T00_re * mb##02_re - ma##T00_im * mb##02_im +		\
	    ma##T01_re * mb##12_re - ma##T01_im * mb##12_im +		\
	    ma##T02_re * mb##22_re - ma##T02_im * mb##22_im;		\
	fa_im =								\
	    ma##T00_re * mb##02_im + ma##T00_im * mb##02_re +		\
	    ma##T01_re * mb##12_im + ma##T01_im * mb##12_re +		\
	    ma##T02_re * mb##22_im + ma##T02_im * mb##22_re;		\
	fb_re =								\
	    ma##T10_re * mb##02_re - ma##T10_im * mb##02_im +		\
	    ma##T11_re * mb##12_re - ma##T11_im * mb##12_im +		\
	    ma##T12_re * mb##22_re - ma##T12_im * mb##22_im;		\
	fb_im =								\
	    ma##T10_re * mb##02_im + ma##T10_im * mb##02_re +		\
	    ma##T11_re * mb##12_im + ma##T11_im * mb##12_re +		\
	    ma##T12_re * mb##22_im + ma##T12_im * mb##22_re;		\
	fc_re =								\
	    ma##T20_re * mb##02_re - ma##T20_im * mb##02_im +		\
	    ma##T21_re * mb##12_re - ma##T21_im * mb##12_im +		\
	    ma##T22_re * mb##22_re - ma##T22_im * mb##22_im;		\
	fc_im =								\
	    ma##T20_re * mb##02_im + ma##T20_im * mb##02_re +		\
	    ma##T21_re * mb##12_im + ma##T21_im * mb##12_re +		\
	    ma##T22_re * mb##22_im + ma##T22_im * mb##22_re;		\
	mb##02_re = fa_re;						\
	mb##02_im = fa_im;						\
	mb##12_re = fb_re;						\
	mb##12_im = fb_im;						\
	mb##22_re = fc_re;						\
	mb##22_im = fc_im;						\
    }while(0)

     
#define tmpMULT_SU3_AN_TEST(ma, mb)	do{				\
	float fa_re, fa_im, fb_re, fb_im, fc_re, fc_im;			\
	fa_re =								\
	    ma##T00_re * mb##00_re - ma##T00_im * mb##00_im +		\
	    ma##T01_re * mb##10_re - ma##T01_im * mb##10_im +		\
	    ma##T02_re * mb##20_re - ma##T02_im * mb##20_im;		\
	fa_im =								\
	    ma##T00_re * mb##00_im + ma##T00_im * mb##00_re +		\
	    ma##T01_re * mb##10_im + ma##T01_im * mb##10_re +		\
	    ma##T02_re * mb##20_im + ma##T02_im * mb##20_re;		\
	fb_re =								\
	    ma##T10_re * mb##00_re - ma##T10_im * mb##00_im +		\
	    ma##T11_re * mb##10_re - ma##T11_im * mb##10_im +		\
	    ma##T12_re * mb##20_re - ma##T12_im * mb##20_im;		\
	fb_im =								\
	    ma##T10_re * mb##00_im + ma##T10_im * mb##00_re +		\
	    ma##T11_re * mb##10_im + ma##T11_im * mb##10_re +		\
	    ma##T12_re * mb##20_im + ma##T12_im * mb##20_re;		\
	fc_re =								\
	    ma##T20_re * mb##00_re - ma##T20_im * mb##00_im +		\
	    ma##T21_re * mb##10_re - ma##T21_im * mb##10_im +		\
	    ma##T22_re * mb##20_re - ma##T22_im * mb##20_im;		\
	fc_im =								\
	    ma##T20_re * mb##00_im + ma##T20_im * mb##00_re +		\
	    ma##T21_re * mb##10_im + ma##T21_im * mb##10_re +		\
	    ma##T22_re * mb##20_im + ma##T22_im * mb##20_re;		\
	mb##00_re = fa_re;						\
	mb##00_im = fa_im;						\
	mb##10_re = fb_re;						\
	mb##10_im = fb_im;						\
	mb##20_re = fc_re;						\
	mb##20_im = fc_im;						\
	fa_re =								\
	    ma##T00_re * mb##01_re - ma##T00_im * mb##01_im +		\
	    ma##T01_re * mb##11_re - ma##T01_im * mb##11_im +		\
	    ma##T02_re * mb##21_re - ma##T02_im * mb##21_im;		\
	fa_im =								\
	    ma##T00_re * mb##01_im + ma##T00_im * mb##01_re +		\
	    ma##T01_re * mb##11_im + ma##T01_im * mb##11_re +		\
	    ma##T02_re * mb##21_im + ma##T02_im * mb##21_re;		\
	fb_re =								\
	    ma##T10_re * mb##01_re - ma##T10_im * mb##01_im +		\
	    ma##T11_re * mb##11_re - ma##T11_im * mb##11_im +		\
	    ma##T12_re * mb##21_re - ma##T12_im * mb##21_im;		\
	fb_im =								\
	    ma##T10_re * mb##01_im + ma##T10_im * mb##01_re +		\
	    ma##T11_re * mb##11_im + ma##T11_im * mb##11_re +		\
	    ma##T12_re * mb##21_im + ma##T12_im * mb##21_re;		\
	fc_re =								\
	    ma##T20_re * mb##01_re - ma##T20_im * mb##01_im +		\
	    ma##T21_re * mb##11_re - ma##T21_im * mb##11_im +		\
	    ma##T22_re * mb##21_re - ma##T22_im * mb##21_im;		\
	fc_im =								\
	    ma##T20_re * mb##01_im + ma##T20_im * mb##01_re +		\
	    ma##T21_re * mb##11_im + ma##T21_im * mb##11_re +		\
	    ma##T22_re * mb##21_im + ma##T22_im * mb##21_re;		\
	mb##01_re = fa_re;						\
	mb##01_im = fa_im;						\
	mb##11_re = fb_re;						\
	mb##11_im = fb_im;						\
	mb##21_re = fc_re;						\
	mb##21_im = fc_im;						\
	fa_re =								\
	    ma##T00_re * mb##02_re - ma##T00_im * mb##02_im +		\
	    ma##T01_re * mb##12_re - ma##T01_im * mb##12_im +		\
	    ma##T02_re * mb##22_re - ma##T02_im * mb##22_im;		\
	fa_im =								\
	    ma##T00_re * mb##02_im + ma##T00_im * mb##02_re +		\
	    ma##T01_re * mb##12_im + ma##T01_im * mb##12_re +		\
	    ma##T02_re * mb##22_im + ma##T02_im * mb##22_re;		\
	fb_re =								\
	    ma##T10_re * mb##02_re - ma##T10_im * mb##02_im +		\
	    ma##T11_re * mb##12_re - ma##T11_im * mb##12_im +		\
	    ma##T12_re * mb##22_re - ma##T12_im * mb##22_im;		\
	fb_im =								\
	    ma##T10_re * mb##02_im + ma##T10_im * mb##02_re +		\
	    ma##T11_re * mb##12_im + ma##T11_im * mb##12_re +		\
	    ma##T12_re * mb##22_im + ma##T12_im * mb##22_re;		\
	fc_re =								\
	    ma##T20_re * mb##02_re - ma##T20_im * mb##02_im +		\
	    ma##T21_re * mb##12_re - ma##T21_im * mb##12_im +		\
	    ma##T22_re * mb##22_re - ma##T22_im * mb##22_im;		\
	fc_im =								\
	    ma##T20_re * mb##02_im + ma##T20_im * mb##02_re +		\
	    ma##T21_re * mb##12_im + ma##T21_im * mb##12_re +		\
	    ma##T22_re * mb##22_im + ma##T22_im * mb##22_re;		\
	ma##02_re = fa_re;						\
	ma##02_im = fa_im;						\
	ma##12_re = fb_re;						\
	ma##12_im = fb_im;						\
	ma##22_re = fc_re;						\
	ma##22_im = fc_im;						\
	ma##00_re = mb##00_re;						\
	ma##00_im = mb##00_im;						\
	ma##10_re = mb##10_re;						\
	ma##10_im = mb##10_im;						\
	ma##20_re = mb##20_re;						\
	ma##20_im = mb##20_im;						\
	ma##01_re = mb##01_re;						\
	ma##01_im = mb##01_im;						\
	ma##11_re = mb##11_re;						\
	ma##11_im = mb##11_im;						\
	ma##21_re = mb##21_re;						\
	ma##21_im = mb##21_im;						\
    }while(0)




#define LLFAT_SCALAR_MULT_SU3_MATRIX(mb, s, mc)		\
    mc##00_re = mb##00_re * s;				\
    mc##00_im = mb##00_im * s;				\
    mc##01_re = mb##01_re * s;				\
    mc##01_im = mb##01_im * s;				\
    mc##02_re = mb##02_re * s;				\
    mc##02_im = mb##02_im * s;				\
    mc##10_re = mb##10_re * s;				\
    mc##10_im = mb##10_im * s;				\
    mc##11_re = mb##11_re * s;				\
    mc##11_im = mb##11_im * s;				\
    mc##12_re = mb##12_re * s;				\
    mc##12_im = mb##12_im * s;				\
    mc##20_re = mb##20_re * s;				\
    mc##20_im = mb##20_im * s;				\
    mc##21_re = mb##21_re * s;				\
    mc##21_im = mb##21_im * s;				\
    mc##22_re = mb##22_re * s;				\
    mc##22_im = mb##22_im * s;	

#define LLFAT_ADD_SU3_MATRIX(ma, mb, mc)		\
    mc##00_re = ma##00_re + mb##00_re;			\
    mc##00_im = ma##00_im + mb##00_im;			\
    mc##01_re = ma##01_re + mb##01_re;			\
    mc##01_im = ma##01_im + mb##01_im;			\
    mc##02_re = ma##02_re + mb##02_re;			\
    mc##02_im = ma##02_im + mb##02_im;			\
    mc##10_re = ma##10_re + mb##10_re;			\
    mc##10_im = ma##10_im + mb##10_im;			\
    mc##11_re = ma##11_re + mb##11_re;			\
    mc##11_im = ma##11_im + mb##11_im;			\
    mc##12_re = ma##12_re + mb##12_re;			\
    mc##12_im = ma##12_im + mb##12_im;			\
    mc##20_re = ma##20_re + mb##20_re;			\
    mc##20_im = ma##20_im + mb##20_im;			\
    mc##21_re = ma##21_re + mb##21_re;			\
    mc##21_im = ma##21_im + mb##21_im;			\
    mc##22_re = ma##22_re + mb##22_re;			\
    mc##22_im = ma##22_im + mb##22_im;		


__constant__ float act_path_coeff_f0;
__constant__ float act_path_coeff_f1;
__constant__ float act_path_coeff_f2;
__constant__ float act_path_coeff_f3;
__constant__ float act_path_coeff_f4;
__constant__ float act_path_coeff_f5;

__constant__ double act_path_coeff0;
__constant__ double act_path_coeff1;
__constant__ double act_path_coeff2;
__constant__ double act_path_coeff3;
__constant__ double act_path_coeff4;
__constant__ double act_path_coeff5;

void
llfat_init_cuda(QudaGaugeParam* param, void* act_path_coeff)
{
    static int llfat_init_cuda_flag = 0;
    if (llfat_init_cuda_flag){
	return;
    }
    
    llfat_init_cuda_flag = 1;
    if (param->cuda_prec == QUDA_DOUBLE_PRECISION){
	double* tmp = (double*)act_path_coeff;
	cudaMemcpyToSymbol("act_path_coeff0", tmp, sizeof(double)); CUERR;
	cudaMemcpyToSymbol("act_path_coeff1", tmp + 1, sizeof(double)); CUERR;
	cudaMemcpyToSymbol("act_path_coeff2", tmp + 2, sizeof(double)); CUERR;
	cudaMemcpyToSymbol("act_path_coeff3", tmp + 3, sizeof(double)); CUERR;
	cudaMemcpyToSymbol("act_path_coeff4", tmp + 4, sizeof(double)); CUERR;
	cudaMemcpyToSymbol("act_path_coeff5", tmp + 5, sizeof(double)); CUERR; 	
    }else if (param->cuda_prec == QUDA_SINGLE_PRECISION){
	float* tmp = (float*)act_path_coeff;
	cudaMemcpyToSymbol("act_path_coeff_f0", tmp, sizeof(float)); CUERR;
	cudaMemcpyToSymbol("act_path_coeff_f1", tmp + 1, sizeof(float)); CUERR;
	cudaMemcpyToSymbol("act_path_coeff_f2", tmp + 2, sizeof(float)); CUERR;
	cudaMemcpyToSymbol("act_path_coeff_f3", tmp + 3, sizeof(float)); CUERR;
	cudaMemcpyToSymbol("act_path_coeff_f4", tmp + 4, sizeof(float)); CUERR;
	cudaMemcpyToSymbol("act_path_coeff_f5", tmp + 5, sizeof(float)); CUERR;	
    }else{
	fprintf(stderr ,"ERROR: %s: half precision is not supported\n", __FUNCTION__);
	exit(1);
	
    }

    init_kernel_cuda(param);

}


 
#define LLFAT_COMPUTE_NEW_IDX_LOWER_STAPLE(mydir1, mydir2) do {		\
	new_x1 = x1;							\
	new_x2 = x2;							\
	new_x3 = x3;							\
	new_x4 = x4;							\
	switch(mydir1){							\
	case 0:								\
	    new_mem_idx = ( (x1==0)?X+X1m1:X-1);			\
	    new_x1 = (x1==0)?X1m1:x1 - 1;				\
	    break;							\
	case 1:								\
	    new_mem_idx = ( (x2==0)?X+X2X1mX1:X-X1);			\
	    new_x2 = (x2==0)?X2m1:x2 - 1;				\
	    break;							\
	case 2:								\
	    new_mem_idx = ( (x3==0)?X+X3X2X1mX2X1:X-X2X1);		\
	    new_x3 = (x3==0)?X3m1:x3 - 1;				\
	    break;							\
	case 3:								\
	    new_mem_idx = ( (x4==0)?X+X4X3X2X1mX3X2X1:X-X3X2X1);	\
	    new_x4 = (x4==0)?X4m1:x4 - 1;				\
	    break;							\
	}								\
	switch(mydir2){							\
	case 0:								\
	    new_mem_idx = ( (x1==X1m1)?new_mem_idx-X1m1:new_mem_idx+1)>> 1; \
	    new_x1 = (x1==X1m1)?0:x1+1;					\
	    break;							\
	case 1:								\
	    new_mem_idx = ( (x2==X2m1)?new_mem_idx-X2X1mX1:new_mem_idx+X1) >> 1; \
	    new_x2 = (x2==X2m1)?0:x2+1;					\
	    break;							\
	case 2:								\
	    new_mem_idx = ( (x3==X3m1)?new_mem_idx-X3X2X1mX2X1:new_mem_idx+X2X1) >> 1; \
	    new_x3 = (x3==X3m1)?0:x3+1;					\
	    break;							\
	case 3:								\
	    new_mem_idx = ( (x4==X4m1)?new_mem_idx-X4X3X2X1mX3X2X1:new_mem_idx+X3X2X1) >> 1; \
	    new_x4 = (x4==X4m1)?0:x4+1;					\
	    break;							\
	}								\
    }while(0)



#define LLFAT_COMPUTE_NEW_IDX_PLUS(mydir, idx) do {			\
	new_x1 = x1;							\
	new_x2 = x2;							\
	new_x3 = x3;							\
	new_x4 = x4;							\
        switch(mydir){                                                  \
        case 0:                                                         \
            new_mem_idx = ( (x1==X1m1)?idx-X1m1:idx+1)>>1;		\
            new_x1 = (x1==X1m1)?0:x1+1;					\
            break;                                                      \
        case 1:                                                         \
            new_mem_idx = ( (x2==X2m1)?idx-X2X1mX1:idx+X1)>>1;		\
            new_x2 = (x2==X2m1)?0:x2+1;					\
            break;                                                      \
        case 2:                                                         \
            new_mem_idx = ( (x3==X3m1)?idx-X3X2X1mX2X1:idx+X2X1)>>1;	\
            new_x3 = (x3==X3m1)?0:x3+1;					\
            break;                                                      \
        case 3:                                                         \
            new_mem_idx = ( (x4==X4m1)?idx-X4X3X2X1mX3X2X1:idx+X3X2X1)>>1; \
            new_x4 = (x4==X4m1)?0:x4+1;					\
            break;                                                      \
        }                                                               \
    }while(0)

#define LLFAT_COMPUTE_NEW_IDX_MINUS(mydir, idx) do {			\
	new_x1 = x1;							\
	new_x2 = x2;							\
	new_x3 = x3;							\
	new_x4 = x4;							\
        switch(mydir){                                                  \
        case 0:                                                         \
            new_mem_idx = ( (x1==0)?idx+X1m1:idx-1) >> 1;		\
            new_x1 = (x1==0)?X1m1:x1 - 1;				\
            break;                                                      \
        case 1:                                                         \
            new_mem_idx = ( (x2==0)?idx+X2X1mX1:idx-X1) >> 1;		\
            new_x2 = (x2==0)?X2m1:x2 - 1;				\
            break;                                                      \
        case 2:                                                         \
            new_mem_idx = ( (x3==0)?idx+X3X2X1mX2X1:idx-X2X1) >> 1;	\
            new_x3 = (x3==0)?X3m1:x3 - 1;				\
            break;                                                      \
        case 3:                                                         \
            new_mem_idx = ( (x4==0)?idx+X4X3X2X1mX3X2X1:idx-X3X2X1) >> 1; \
            new_x4 = (x4==0)?X4m1:x4 - 1;				\
            break;                                                      \
        }                                                               \
    }while(0)

    

#define COMPUTE_RECONSTRUCT_SIGN(sign, dir, i1,i2,i3,i4) do {	\
	sign =1;						\
	switch(dir){						\
	case XUP:						\
	    if ( (i4 & 1) == 1){				\
		sign = -1;					\
	    }							\
	    break;						\
	case YUP:						\
	    if ( ((i4+i1) & 1) == 1){				\
		sign = -1;					\
	    }							\
	    break;						\
	case ZUP:						\
	    if ( ((i4+i1+i2) & 1) == 1){			\
		sign = -1;					\
	    }							\
	    break;						\
	case TUP:						\
	    if (i4 == X4m1 ){					\
		sign = -1;					\
	    }							\
	    break;						\
	}							\
    }while (0)



template<int mu, int nu, int odd_bit>
__global__ void
do_siteComputeGenStapleParityKernel(float2* staple_even, float2* staple_odd, 
				 float4* sitelink_even, float4* sitelink_odd, 
				 float2* fatlink_even, float2* fatlink_odd,	
				 float mycoeff)
{
    float4 TEMPA0, TEMPA1, TEMPA2, TEMPA3, TEMPA4;
    float4 A0, A1, A2, A3, A4;
    float4 B0, B1, B2, B3, B4;
    float4 C0, C1, C2, C3, C4;
    float2 STAPLE0, STAPLE1, STAPLE2, STAPLE3, STAPLE4, STAPLE5, STAPLE6, STAPLE7, STAPLE8;
    float2 FAT0, FAT1, FAT2, FAT3, FAT4, FAT5, FAT6, FAT7, FAT8;
    
    int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    int z1 = FAST_INT_DIVIDE(mem_idx, X1h);
    short x1h = mem_idx - z1*X1h;
    int z2 = FAST_INT_DIVIDE(z1, X2);
    short x2 = z1 - z2*X2;
    short x4 = FAST_INT_DIVIDE(z2, X3);
    short x3 = z2 - x4*X3;
    short x1odd = (x2 + x3 + x4 + odd_bit) & 1;
    short x1 = 2*x1h + x1odd;
    int X = 2*mem_idx + x1odd;    
    float sign =1;    
    int new_mem_idx;
    int new_x1 = x1;
    int new_x2 = x2;
    int new_x3 = x3;
    int new_x4 = x4;
    
    
    /* Upper staple */
    /* Computes the staple :
     *                 mu (B)
     *             +-------+
     *       nu	   |	   | 
     *	     (A)   |	   |(C)
     *		   X	   X
     *
     */
    
    /* load matrix A*/
    LOAD_EVEN_SITE_MATRIX_12_SINGLE(nu, mem_idx, A);   
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, x1, x2, x3, x4);
    RECONSTRUCT_LINK_12_SINGLE(nu, mem_idx, sign, a);
    
    /* load matrix B*/  
    LLFAT_COMPUTE_NEW_IDX_PLUS(nu, X);    
    LOAD_ODD_SITE_MATRIX_12_SINGLE(mu, new_mem_idx, B);
    COMPUTE_RECONSTRUCT_SIGN(sign, mu, new_x1, new_x2, new_x3, new_x4);    
    RECONSTRUCT_LINK_12_SINGLE(mu, new_mem_idx, sign, b);
    
    MULT_SU3_NN(a, b, tempa);    
    
    /* load matrix C*/
    
    
    LLFAT_COMPUTE_NEW_IDX_PLUS(mu, X);    
    LOAD_ODD_SITE_MATRIX_12_SINGLE(nu, new_mem_idx, C);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, new_x1, new_x2, new_x3, new_x4);    
    RECONSTRUCT_LINK_12_SINGLE(nu, new_mem_idx, sign, c);
    
    MULT_SU3_NA(tempa, c, staple);		


    /***************lower staple****************
     *
     *                 X       X
     *           nu    |       | 
     *	         (A)   |       | (C)
     *		       +-------+
     *                  mu (B)
     *
     *********************************************/
    /* load matrix A*/
    LLFAT_COMPUTE_NEW_IDX_MINUS(nu,X);    
    
    LOAD_ODD_SITE_MATRIX_12_SINGLE(nu, (new_mem_idx), A);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, new_x1, new_x2, new_x3, new_x4);        
    RECONSTRUCT_LINK_12_SINGLE(nu, (new_mem_idx), sign, a);
    
    /* load matrix B*/				
    LOAD_ODD_SITE_MATRIX_12_SINGLE(mu, (new_mem_idx), B);
    COMPUTE_RECONSTRUCT_SIGN(sign, mu, new_x1, new_x2, new_x3, new_x4);    
    RECONSTRUCT_LINK_12_SINGLE(mu, (new_mem_idx), sign, b);
    
    MULT_SU3_AN(a, b, tempa);
    
    /* load matrix C*/
    //COMPUTE_NEW_IDX_PLUS(mu, new_mem_idx);
    LLFAT_COMPUTE_NEW_IDX_LOWER_STAPLE(nu, mu);
    LOAD_EVEN_SITE_MATRIX_12_SINGLE(nu, new_mem_idx, C);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, new_x1, new_x2, new_x3, new_x4);        
    RECONSTRUCT_LINK_12_SINGLE(nu, new_mem_idx, sign, c);
				
    
    MULT_SU3_NN(tempa, c, b);		
    LLFAT_ADD_SU3_MATRIX(b, staple, staple);

    //ADD_MULT_SU3_NN(tempa, c, tempb, staple);
    
    LOAD_EVEN_FAT_MATRIX_18_SINGLE(mu, mem_idx);
    SCALAR_MULT_ADD_SU3_MATRIX(fat, staple, mycoeff, fat);
    WRITE_FAT_MATRIX(fatlink_even,mu,  mem_idx);	

    WRITE_STAPLE_MATRIX(staple_even, mem_idx);	
    
    return;
}


template<int mu, int nu, int odd_bit>
__global__ void
do_computeGenStapleFieldParityKernel(float4* sitelink_even, float4* sitelink_odd,
				  float2* fatlink_even, float2* fatlink_odd,			    
				  float2* mulink_even, float2* mulink_odd, 
				  float mycoeff)
{
    float4 A0, A1, A2, A3, A4;
    float4 B0, B1, B2, B3, B4;
    float2 BB0, BB1, BB2, BB3, BB4, BB5, BB6, BB7, BB8;
    float4 C0, C1, C2, C3, C4;
    float4 TEMPA0, TEMPA1, TEMPA2, TEMPA3, TEMPA4;
    float2 FAT0, FAT1, FAT2, FAT3, FAT4, FAT5, FAT6, FAT7, FAT8;

    int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    int z1 = FAST_INT_DIVIDE(mem_idx, X1h);
    int x1h = mem_idx - z1*X1h;
    int z2 = FAST_INT_DIVIDE(z1, X2);
    int x2 = z1 - z2*X2;
    int x4 = FAST_INT_DIVIDE(z2, X3);
    int x3 = z2 - x4*X3;
    int x1odd = (x2 + x3 + x4 + odd_bit) & 1;
    int x1 = 2*x1h + x1odd;
    int X = 2*mem_idx + x1odd;
    
    int sign =1;

    int new_mem_idx;
   int new_x1 = x1;
    int new_x2 = x2;
    int new_x3 = x3;
    int new_x4 = x4;

    
    /* Upper staple */
    /* Computes the staple :
     *                mu (BB)
     *             +-------+
     *       nu	   |	   | 
     *	     (A)   |	   |(C)
     *		   X	   X
     *
     */
    
    /* load matrix A*/
    LOAD_EVEN_SITE_MATRIX_12_SINGLE(nu, mem_idx, A);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, x1, x2, x3, x4);
    RECONSTRUCT_LINK_12_SINGLE(nu, mem_idx, sign, a);
		
    /* load matrix BB*/
 
    LLFAT_COMPUTE_NEW_IDX_PLUS(nu, X);
    LOAD_ODD_MULINK_MATRIX_18_SINGLE(0, new_mem_idx, BB);
    //LOAD_MATRIX_18_SINGLE_FLOAT4(mulink_odd, 0, new_mem_idx, B);
    
    MULT_SU3_NN(a, bb, tempa);
    //MULT_SU3_NN_TEST(a, bb);
    //COPY_SU3_MATRIX(a, bb);	
    /* load matrix C*/
    
    LLFAT_COMPUTE_NEW_IDX_PLUS(mu, X);    
    LOAD_ODD_SITE_MATRIX_12_SINGLE(nu, new_mem_idx, C);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, new_x1, new_x2, new_x3, new_x4);
    RECONSTRUCT_LINK_12_SINGLE(nu, new_mem_idx, sign, c);
    
    
    //MULT_SU3_NA(tempa, c, tempb);		
    MULT_SU3_NA_TEST(tempa, c);
    //COPY_SU3_MATRIX(a, tempb);

    //LOAD_FAT_MATRIX_18_SINGLE(my_fatlink, mu, mem_idx);
    //SCALAR_MULT_ADD_SU3_MATRIX(fat, tempb, mycoeff, fat);
    
    /***************lower staple****************
     *
     *                 X       X
     *           nu    |       | 
     *	         (A)   |       | (C)
     *		       +-------+
     *                  mu (B)
     *
     *********************************************/
		

		
    /* load matrix A*/
    LLFAT_COMPUTE_NEW_IDX_MINUS(nu, X);
    
    LOAD_ODD_SITE_MATRIX_12_SINGLE(nu, (new_mem_idx), A);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, new_x1, new_x2, new_x3, new_x4);
    RECONSTRUCT_LINK_12_SINGLE(nu, (new_mem_idx), sign, a);
    
    /* load matrix B*/				
    LOAD_ODD_MULINK_MATRIX_18_SINGLE(0, (new_mem_idx), BB);
    //LOAD_MATRIX_18_SINGLE_FLOAT4(mulink_odd, 0, (new_mem_idx), C);
	
    MULT_SU3_AN(a, bb, b);
    //MULT_SU3_AN_TEST(b, c);

    /* load matrix C*/
    //COMPUTE_NEW_IDX_PLUS(mu, new_mem_idx);
    LLFAT_COMPUTE_NEW_IDX_LOWER_STAPLE(nu, mu);
    LOAD_EVEN_SITE_MATRIX_12_SINGLE(nu, new_mem_idx, C);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, new_x1, new_x2, new_x3, new_x4);
    RECONSTRUCT_LINK_12_SINGLE(nu, new_mem_idx, sign, c);
    
    
    //MULT_SU3_NN(tempa, c, a);
    MULT_SU3_NN_TEST(b, c);

    
    LLFAT_ADD_SU3_MATRIX(b, tempa, b);
    
    LOAD_EVEN_FAT_MATRIX_18_SINGLE(mu, mem_idx);
    SCALAR_MULT_ADD_SU3_MATRIX(fat, b, mycoeff, fat);	
    
    WRITE_FAT_MATRIX(fatlink_even, mu,  mem_idx);	
    
    return;
}

template<int mu, int nu, int odd_bit>
__global__ void
do_computeGenStapleFieldSaveParityKernel(float2* staple_even, float2* staple_odd, 
					 float4* sitelink_even, float4* sitelink_odd,
					 float2* fatlink_even, float2* fatlink_odd,			    
					 float2* mulink_even, float2* mulink_odd, 
					 float mycoeff)
{
    float4 TEMPA0, TEMPA1, TEMPA2, TEMPA3, TEMPA4;
    float4 A0, A1, A2, A3, A4;
    float2 BB0, BB1, BB2, BB3, BB4, BB5, BB6, BB7, BB8;
    float4 C0, C1, C2, C3, C4;
    float2 STAPLE0, STAPLE1, STAPLE2, STAPLE3, STAPLE4, STAPLE5, STAPLE6, STAPLE7, STAPLE8;
    float2 FAT0, FAT1, FAT2, FAT3, FAT4, FAT5, FAT6, FAT7, FAT8;
    
    int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    int z1 = FAST_INT_DIVIDE(mem_idx, X1h);
    int x1h = mem_idx - z1*X1h;
    int z2 = FAST_INT_DIVIDE(z1, X2);
    int x2 = z1 - z2*X2;
    int x4 = FAST_INT_DIVIDE(z2, X3);
    int x3 = z2 - x4*X3;
    int x1odd = (x2 + x3 + x4 + odd_bit) & 1;
    int x1 = 2*x1h + x1odd;
    int X = 2*mem_idx + x1odd;
    
    int sign =1;
    
    int new_mem_idx;
    int new_x1 = x1;
    int new_x2 = x2;
    int new_x3 = x3;
    int new_x4 = x4;

    
    /* Upper staple */
    /* Computes the staple :
     *                mu (BB)
     *             +-------+
     *       nu	   |	   | 
     *	     (A)   |	   |(C)
     *		   X	   X
     *
     */
		
    /* load matrix A*/
    LOAD_EVEN_SITE_MATRIX_12_SINGLE(nu, mem_idx, A);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, x1, x2, x3, x4);
    RECONSTRUCT_LINK_12_SINGLE(nu, mem_idx, sign, a);
    
    /* load matrix BB*/
    LLFAT_COMPUTE_NEW_IDX_PLUS(nu, X);    
    LOAD_ODD_MULINK_MATRIX_18_SINGLE(0, new_mem_idx, BB);
	
    MULT_SU3_NN(a, bb, tempa);    

    /* load matrix C*/
    LLFAT_COMPUTE_NEW_IDX_PLUS(mu, X);    
    LOAD_ODD_SITE_MATRIX_12_SINGLE(nu, new_mem_idx, C);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, new_x1, new_x2, new_x3, new_x4);
    RECONSTRUCT_LINK_12_SINGLE(nu, new_mem_idx, sign, c);
    
    MULT_SU3_NA(tempa, c, staple);

    /***************lower staple****************
     *
     *                 X       X
     *           nu    |       | 
     *	         (A)   |       | (C)
     *		       +-------+
     *                  mu (B)
     *
     *********************************************/
    

    
    /* load matrix A*/
    LLFAT_COMPUTE_NEW_IDX_MINUS(nu, X);

    LOAD_ODD_SITE_MATRIX_12_SINGLE(nu, new_mem_idx, A);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, new_x1, new_x2, new_x3, new_x4);
    RECONSTRUCT_LINK_12_SINGLE(nu, new_mem_idx, sign, a);
		
    /* load matrix B*/				
    LOAD_ODD_MULINK_MATRIX_18_SINGLE(0, new_mem_idx, BB);
    
    MULT_SU3_AN(a, bb, tempa);
    
    /* load matrix C*/
    //COMPUTE_NEW_IDX_PLUS(mu, new_mem_idx);
    LLFAT_COMPUTE_NEW_IDX_LOWER_STAPLE(nu, mu);
    
    LOAD_EVEN_SITE_MATRIX_12_SINGLE(nu, new_mem_idx, C);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, new_x1, new_x2, new_x3, new_x4);
    RECONSTRUCT_LINK_12_SINGLE(nu, new_mem_idx, sign, c);				
    
    MULT_SU3_NN(tempa, c, a);	
    LLFAT_ADD_SU3_MATRIX(staple, a, staple);

    LOAD_EVEN_FAT_MATRIX_18_SINGLE(mu, mem_idx);
    SCALAR_MULT_ADD_SU3_MATRIX(fat, staple, mycoeff, fat);
    
    WRITE_FAT_MATRIX(fatlink_even, mu,  mem_idx);	
    WRITE_STAPLE_MATRIX(staple_even, mem_idx);		    
    
    return;
}

__global__ void 
computeFatLinkKernel(float4* sitelink_even, float4* sitelink_odd,
		     float2* fatlink_even, float2* fatlink_odd)
{

    float4 TEMPA0, TEMPA1, TEMPA2, TEMPA3, TEMPA4;
    float2 FAT0, FAT1, FAT2, FAT3, FAT4, FAT5, FAT6, FAT7, FAT8;

    float4* my_sitelink;
    float2* my_fatlink;
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    int mem_idx = sid;

    int odd_bit= 0;
    my_sitelink = sitelink_even;
    my_fatlink = fatlink_even;
    if (mem_idx >= Vh){
	odd_bit=1;
	mem_idx = mem_idx - Vh;
	my_sitelink = sitelink_odd;
	my_fatlink = fatlink_odd;
    }
   
    int z1 = FAST_INT_DIVIDE(mem_idx, X1h);
    int x1h = mem_idx - z1*X1h;
    int z2 = FAST_INT_DIVIDE(z1, X2);
    int x2 = z1 - z2*X2;
    int x4 = FAST_INT_DIVIDE(z2, X3);
    int x3 = z2 - x4*X3;
    int x1odd = (x2 + x3 + x4 + odd_bit) & 1;
    int x1 = 2*x1h + x1odd;
    int sign =1;   	

    for(int dir=0;dir < 4; dir++){
	LOAD_MATRIX_12_SINGLE(my_sitelink, dir, mem_idx, TEMPA);
	COMPUTE_RECONSTRUCT_SIGN(sign, dir, x1, x2, x3, x4);
	RECONSTRUCT_LINK_12_SINGLE(dir, mem_idx, sign, tempa);

	LOAD_FAT_MATRIX_18_SINGLE(my_fatlink, dir, mem_idx);
	
	SCALAR_MULT_SU3_MATRIX((act_path_coeff_f0 - 6.0*act_path_coeff_f5), tempa, fat); 
	
	WRITE_FAT_MATRIX(my_fatlink,dir, mem_idx);	
    }
    
    return;
}


#define UNBIND_ALL_TEXTURE do{			\
	cudaUnbindTexture(siteLink0TexSingle);	\
	cudaUnbindTexture(siteLink1TexSingle);	\
	cudaUnbindTexture(fatGauge0TexSingle);	\
	cudaUnbindTexture(fatGauge1TexSingle);	\
	cudaUnbindTexture(muLink0TexSingle);	\
	cudaUnbindTexture(muLink1TexSingle);	\
    }while(0)

#define UNBIND_SITE_AND_FAT_LINK do{		\
	cudaUnbindTexture(siteLink0TexSingle);	\
	cudaUnbindTexture(siteLink1TexSingle);	\
	cudaUnbindTexture(fatGauge0TexSingle);	\
	cudaUnbindTexture(fatGauge1TexSingle);	\
    }while(0)

#define BIND_SITE_AND_FAT_LINK do {					\
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.even, cudaSiteLink.bytes); \
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes); \
	cudaBindTexture(0, fatGauge0TexSingle, cudaFatLink.even, cudaFatLink.bytes); \
	cudaBindTexture(0, fatGauge1TexSingle, cudaFatLink.odd,  cudaFatLink.bytes); \
    }while(0)

#define BIND_SITE_AND_FAT_LINK_REVERSE do {				\
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes); \
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes); \
	cudaBindTexture(0, fatGauge1TexSingle, cudaFatLink.even, cudaFatLink.bytes); \
	cudaBindTexture(0, fatGauge0TexSingle, cudaFatLink.odd,  cudaFatLink.bytes); \
    }while(0)



#define ENUMERATE_FUNCS(mu,nu,odd_bit)	switch(mu) {			\
    case 0:								\
	switch(nu){							\
	case 0:								\
	    printf("ERROR: invalid direction combination\n"); exit(1);	\
	    break;							\
	case 1:								\
	    if (!odd_bit) { CALL_FUNCTION(0,1,0); }			\
	    else {CALL_FUNCTION(0,1,1); }				\
	    break;							\
	case 2:								\
	    if (!odd_bit) { CALL_FUNCTION(0,2,0); }			\
	    else {CALL_FUNCTION(0,2,1); }				\
	    break;							\
	case 3:								\
	    if (!odd_bit) { CALL_FUNCTION(0,3,0); }			\
	    else {CALL_FUNCTION(0,3,1); }				\
	    break;							\
	}								\
	break;								\
    case 1:								\
	switch(nu){							\
	case 0:								\
	    if (!odd_bit) { CALL_FUNCTION(1,0,0); }			\
	    else {CALL_FUNCTION(1,0,1); }				\
	    break;							\
	case 1:								\
	    printf("ERROR: invalid direction combination\n"); exit(1);	\
	    break;							\
	case 2:								\
	    if (!odd_bit) { CALL_FUNCTION(1,2,0); }			\
	    else {CALL_FUNCTION(1,2,1); }				\
	    break;							\
	case 3:								\
	    if (!odd_bit) { CALL_FUNCTION(1,3,0); }			\
	    else {CALL_FUNCTION(1,3,1); }				\
	    break;							\
	}								\
	break;								\
    case 2:								\
	switch(nu){							\
	case 0:								\
	    if (!odd_bit) { CALL_FUNCTION(2,0,0); }			\
	    else {CALL_FUNCTION(2,0,1); }				\
	    break;							\
	case 1:								\
	    if (!odd_bit) { CALL_FUNCTION(2,1,0); }			\
	    else {CALL_FUNCTION(2,1,1); }				\
	    break;							\
	case 2:								\
	    printf("ERROR: invalid direction combination\n"); exit(1);	\
	    break;							\
	case 3:								\
	    if (!odd_bit) { CALL_FUNCTION(2,3,0); }			\
	    else {CALL_FUNCTION(2,3,1); }				\
	    break;							\
	}								\
	break;								\
    case 3:								\
	switch(nu){							\
	case 0:								\
	    if (!odd_bit) { CALL_FUNCTION(3,0,0); }			\
	    else {CALL_FUNCTION(3,0,1); }				\
	    break;							\
	case 1:								\
	    if (!odd_bit) { CALL_FUNCTION(3,1,0); }			\
	    else {CALL_FUNCTION(3,1,1); }				\
	    break;							\
	case 2:								\
	    if (!odd_bit) { CALL_FUNCTION(3,2,0); }			\
	    else {CALL_FUNCTION(3,2,1); }				\
	    break;							\
	case 3:								\
	    printf("ERROR: invalid direction combination\n"); exit(1);	\
	    break;							\
	}								\
	break;								\
    }


static void
siteComputeGenStapleParityKernel(float2* staple_even, float2* staple_odd, 
				 float4* sitelink_even, float4* sitelink_odd, 
				 float2* fatlink_even, float2* fatlink_odd,	
				 int mu, int nu,int odd_bit,
				 float mycoeff,
				 dim3 halfGridDim, dim3 blockDim)
{
    
#define  CALL_FUNCTION(mu, nu, odd_bit)					\
    do_siteComputeGenStapleParityKernel<mu,nu, odd_bit><<<halfGridDim, blockDim>>>(staple_even, staple_odd, \
										   sitelink_even, sitelink_odd,	\
										   fatlink_even, fatlink_odd, \
										   mycoeff) 
    ENUMERATE_FUNCS(mu,nu,odd_bit);
#undef CALL_FUNCTION
    
    
}

static void
computeGenStapleFieldParityKernel(float4* sitelink_even, float4* sitelink_odd,
				  float2* fatlink_even, float2* fatlink_odd,			    
				  float2* mulink_even, float2* mulink_odd, 
				  int mu, int nu, int odd_bit,
				  float mycoeff,
				  dim3 halfGridDim, dim3 blockDim)
{    
#define  CALL_FUNCTION(mu, nu, odd_bit)					\
    do_computeGenStapleFieldParityKernel<mu,nu, odd_bit><<<halfGridDim, blockDim>>>(sitelink_even, sitelink_odd, \
										    fatlink_even, fatlink_odd, \
										    mulink_even, mulink_odd, \
										    mycoeff) 
    ENUMERATE_FUNCS(mu,nu,odd_bit);
#undef CALL_FUNCTION 
    
}

static void
computeGenStapleFieldSaveParityKernel(float2* staple_even, float2* staple_odd, 
				      float4* sitelink_even, float4* sitelink_odd,
				      float2* fatlink_even, float2* fatlink_odd,			    
				      float2* mulink_even, float2* mulink_odd, 
				      int mu, int nu, int odd_bit,
				      float mycoeff,
				      dim3 halfGridDim, dim3 blockDim)
{
#define  CALL_FUNCTION(mu, nu, odd_bit)					\
    do_computeGenStapleFieldSaveParityKernel<mu,nu, odd_bit><<<halfGridDim, blockDim>>>(staple_even, staple_odd, \
											sitelink_even, sitelink_odd, \
											fatlink_even, fatlink_odd, \
											mulink_even, mulink_odd, \
											mycoeff) 
    ENUMERATE_FUNCS(mu,nu,odd_bit);
#undef CALL_FUNCTION 
    
}

void
llfat_cuda(void* fatLink, void* siteLink,
	   FullGauge cudaFatLink, FullGauge cudaSiteLink, 
	   FullStaple cudaStaple, FullStaple cudaStaple1,
	   QudaGaugeParam* param, void* _act_path_coeff)
{
        
    float* act_path_coeff = (float*) _act_path_coeff;
    int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
    dim3 gridDim(volume/BLOCK_DIM,1,1);
    dim3 halfGridDim(volume/(2*BLOCK_DIM),1,1);
    dim3 blockDim(BLOCK_DIM , 1, 1);
    
    BIND_SITE_AND_FAT_LINK;
    computeFatLinkKernel<<<gridDim, blockDim>>>((float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd,
						(float2*)cudaFatLink.even, (float2*)cudaFatLink.odd);     
    UNBIND_SITE_AND_FAT_LINK;
    
    for(int dir = 0;dir < 4; dir++){
	for(int nu = 0; nu < 4; nu++){
	    if (nu != dir){
		
		//even
		BIND_SITE_AND_FAT_LINK;		
		siteComputeGenStapleParityKernel((float2*)cudaStaple.even, (float2*)cudaStaple.odd,
						 (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd,
						 (float2*)cudaFatLink.even, (float2*)cudaFatLink.odd, 
						 dir, nu,0,
						 act_path_coeff[2],
						 halfGridDim, blockDim);
		UNBIND_SITE_AND_FAT_LINK;

		//odd
		BIND_SITE_AND_FAT_LINK_REVERSE;
		siteComputeGenStapleParityKernel((float2*)cudaStaple.odd, (float2*)cudaStaple.even,
						 (float4*)cudaSiteLink.odd, (float4*)cudaSiteLink.even,
						 (float2*)cudaFatLink.odd, (float2*)cudaFatLink.even, 
						 dir, nu,1,
						 act_path_coeff[2],
						 halfGridDim, blockDim);	
		UNBIND_SITE_AND_FAT_LINK;	
		
		
		//even
		BIND_SITE_AND_FAT_LINK;		
		cudaBindTexture(0, muLink0TexSingle, cudaStaple.even, cudaStaple.bytes);
		cudaBindTexture(0, muLink1TexSingle, cudaStaple.odd, cudaStaple.bytes);
		computeGenStapleFieldParityKernel((float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd,
						  (float2*)cudaFatLink.even, (float2*)cudaFatLink.odd, 
						  (float2*)cudaStaple.even, (float2*)cudaStaple.odd,
						  dir, nu,0,
						  act_path_coeff[5],
						  halfGridDim, blockDim);							  
		UNBIND_ALL_TEXTURE;
		
		//odd
		BIND_SITE_AND_FAT_LINK_REVERSE;		
		cudaBindTexture(0, muLink1TexSingle, cudaStaple.even, cudaStaple.bytes);
		cudaBindTexture(0, muLink0TexSingle, cudaStaple.odd, cudaStaple.bytes);
		computeGenStapleFieldParityKernel((float4*)cudaSiteLink.odd, (float4*)cudaSiteLink.even,
						  (float2*)cudaFatLink.odd, (float2*)cudaFatLink.even, 
						  (float2*)cudaStaple.odd, (float2*)cudaStaple.even,
						  dir, nu,1,
						  act_path_coeff[5],
						  halfGridDim, blockDim);	
		UNBIND_ALL_TEXTURE;
		

		for(int rho = 0; rho < 4; rho++){
		    if (rho != dir && rho != nu){

			//even
			BIND_SITE_AND_FAT_LINK;		
			cudaBindTexture(0, muLink0TexSingle, cudaStaple.even, cudaStaple.bytes);
			cudaBindTexture(0, muLink1TexSingle, cudaStaple.odd, cudaStaple.bytes);			
			computeGenStapleFieldSaveParityKernel((float2*)cudaStaple1.even, (float2*)cudaStaple1.odd,
							      (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd,
							      (float2*)cudaFatLink.even, (float2*)cudaFatLink.odd, 
							      (float2*)cudaStaple.even, (float2*)cudaStaple.odd,
							      dir, rho,0,
							      act_path_coeff[3],
							      halfGridDim, blockDim);								      
			
			UNBIND_ALL_TEXTURE;

			//odd
			BIND_SITE_AND_FAT_LINK_REVERSE;		
			cudaBindTexture(0, muLink1TexSingle, cudaStaple.even, cudaStaple.bytes);
			cudaBindTexture(0, muLink0TexSingle, cudaStaple.odd, cudaStaple.bytes);						
			computeGenStapleFieldSaveParityKernel((float2*)cudaStaple1.odd, (float2*)cudaStaple1.even,
							      (float4*)cudaSiteLink.odd, (float4*)cudaSiteLink.even,
							      (float2*)cudaFatLink.odd, (float2*)cudaFatLink.even, 
							      (float2*)cudaStaple.odd, (float2*)cudaStaple.even,
							      dir, rho,1,
							      act_path_coeff[3],
							      halfGridDim, blockDim);								      
			UNBIND_ALL_TEXTURE;

			
			for(int sig = 0; sig < 4; sig++){
			    if (sig != dir && sig != nu && sig != rho){				
				
				//even				
				BIND_SITE_AND_FAT_LINK;		
				cudaBindTexture(0, muLink0TexSingle, cudaStaple1.even, cudaStaple1.bytes);
				cudaBindTexture(0, muLink1TexSingle, cudaStaple1.odd,  cudaStaple1.bytes);
				computeGenStapleFieldParityKernel((float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd,
								  (float2*)cudaFatLink.even, (float2*)cudaFatLink.odd, 
								  (float2*)cudaStaple1.even, (float2*)cudaStaple1.odd,
								  dir, sig, 0, 
								  act_path_coeff[4],
								  halfGridDim, blockDim);	
				UNBIND_ALL_TEXTURE;
				
				//odd
				BIND_SITE_AND_FAT_LINK_REVERSE;		
				cudaBindTexture(0, muLink1TexSingle, cudaStaple1.even, cudaStaple1.bytes);
				cudaBindTexture(0, muLink0TexSingle, cudaStaple1.odd,  cudaStaple1.bytes);
				computeGenStapleFieldParityKernel((float4*)cudaSiteLink.odd, (float4*)cudaSiteLink.even,
								  (float2*)cudaFatLink.odd, (float2*)cudaFatLink.even, 
								  (float2*)cudaStaple1.odd, (float2*)cudaStaple1.even,
								  dir, sig, 1, 
								  act_path_coeff[4],
								  halfGridDim, blockDim);	
				UNBIND_ALL_TEXTURE;				
			    }			    
			}//sig
		    }
		}//rho



	    }
	}//nu
    }//dir


    CUERR;
    return;
}


