#include <read_gauge.h>
#include <gauge_quda.h>

#include "gauge_force_quda.h"

#define MULT_SU3_NN_TEST(ma, mb) do{				\
    float fa_re,fa_im, fb_re, fb_im, fc_re, fc_im;		\
    fa_re =							\
      ma##00_re * mb##00_re - ma##00_im * mb##00_im +		\
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


#define GF_SITE_MATRIX_LOAD_TEX 1

#if (GF_SITE_MATRIX_LOAD_TEX == 1)

#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX(siteLink0TexSingle_recon, dir, idx, var)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_12_SINGLE_TEX(siteLink1TexSingle_recon, dir, idx, var)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE(linkEven, dir, idx, var)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE(linkOdd, dir, idx, var)
#endif


#define LOAD_MATRIX LOAD_MATRIX_12_SINGLE
#define LOAD_ANTI_HERMITIAN LOAD_ANTI_HERMITIAN_SINGLE
#define WRITE_ANTI_HERMITIAN WRITE_ANTI_HERMITIAN_SINGLE
#define RECONSTRUCT_MATRIX RECONSTRUCT_LINK_12


__constant__ int path_max_length;

void
gauge_force_init_cuda(QudaGaugeParam* param, int path_max_length)
{    
  
#ifdef MULTI_GPU
#error "multi gpu is not supported for gauge force computation"  
#endif
  
    static int gauge_force_init_cuda_flag = 0;
    if (gauge_force_init_cuda_flag){
	return;
    }
    gauge_force_init_cuda_flag=1;

    init_kernel_cuda(param);
    
    cudaMemcpyToSymbol("path_max_length", &path_max_length, sizeof(int));

}

#define COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(mydir, idx) do {		\
        switch(mydir){                                                  \
        case 0:                                                         \
            new_mem_idx = ( (new_x1==X1m1)?idx-X1m1:idx+1);		\
	    new_x1 = (new_x1==X1m1)?0:new_x1+1;				\
            break;                                                      \
        case 1:                                                         \
            new_mem_idx = ( (new_x2==X2m1)?idx-X2X1mX1:idx+X1);		\
	    new_x2 = (new_x2==X2m1)?0:new_x2+1;				\
            break;                                                      \
        case 2:                                                         \
            new_mem_idx = ( (new_x3==X3m1)?idx-X3X2X1mX2X1:idx+X2X1);	\
	    new_x3 = (new_x3==X3m1)?0:new_x3+1;				\
            break;                                                      \
        case 3:                                                         \
            new_mem_idx = ( (new_x4==X4m1)?idx-X4X3X2X1mX3X2X1:idx+X3X2X1); \
	    new_x4 = (new_x4==X4m1)?0:new_x4+1;				\
            break;                                                      \
        }                                                               \
    }while(0)

#define COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mydir, idx) do {		\
        switch(mydir){                                                  \
        case 0:                                                         \
            new_mem_idx = ( (new_x1==0)?idx+X1m1:idx-1);		\
	    new_x1 = (new_x1==0)?X1m1:new_x1 - 1;			\
            break;                                                      \
        case 1:                                                         \
            new_mem_idx = ( (new_x2==0)?idx+X2X1mX1:idx-X1);		\
	    new_x2 = (new_x2==0)?X2m1:new_x2 - 1;			\
            break;                                                      \
        case 2:                                                         \
            new_mem_idx = ( (new_x3==0)?idx+X3X2X1mX2X1:idx-X2X1);	\
	    new_x3 = (new_x3==0)?X3m1:new_x3 - 1;			\
            break;                                                      \
        case 3:                                                         \
            new_mem_idx = ( (new_x4==0)?idx+X4X3X2X1mX3X2X1:idx-X3X2X1); \
	    new_x4 = (new_x4==0)?X4m1:new_x4 - 1;			\
            break;                                                      \
        }                                                               \
    }while(0)

#define GF_COMPUTE_RECONSTRUCT_SIGN(sign, dir, i1,i2,i3,i4) do {	\
        sign =1;							\
        switch(dir){							\
        case XUP:							\
            if ( (i4 & 1) == 1){					\
	      sign = 1;							\
            }								\
            break;							\
        case YUP:							\
            if ( ((i4+i1) & 1) == 1){					\
                sign = 1;						\
            }								\
            break;							\
        case ZUP:							\
            if ( ((i4+i1+i2) & 1) == 1){				\
                sign = 1;						\
            }								\
            break;							\
        case TUP:							\
            if (i4 == X4m1 ){						\
                sign = 1;						\
            }								\
            break;							\
        }								\
    }while (0)



//for now we only consider 12-reconstruct and single precision

template<int oddBit>
__global__ void
parity_compute_gauge_force_kernel(float2* momEven, float2* momOdd,
				  int dir, double eb3,
				  float4* linkEven, float4* linkOdd,
				  int* input_path, 
				  int* length, float* path_coeff, int num_paths)
{
    int i,j=0;
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int z1 = sid / X1h;
    int x1h = sid - z1*X1h;
    int z2 = z1 / X2;
    int x2 = z1 - z2*X2;
    int x4 = z2 / X3;
    int x3 = z2 - x4*X3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;  
    int X = 2*sid + x1odd;
    
    int sign = 1;
    
    float2* mymom=momEven;
    if (oddBit){
	mymom = momOdd;
    }

    float4 LINKA0, LINKA1, LINKA2, LINKA3, LINKA4;
    float4 LINKB0, LINKB1, LINKB2, LINKB3, LINKB4;
    float2 STAPLE0, STAPLE1, STAPLE2, STAPLE3,STAPLE4, STAPLE5, STAPLE6, STAPLE7, STAPLE8;
    float2 AH0, AH1, AH2, AH3, AH4;

    int new_mem_idx;
    
    
    SET_SU3_MATRIX(staple, 0);
    for(i=0;i < num_paths; i++){
	int nbr_oddbit = (oddBit^1 );
	
	int new_x1 =x1;
	int new_x2 =x2;
	int new_x3 =x3;
	int new_x4 =x4;
	COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(dir, X);
	
	//linka: current matrix
	//linkb: the loaded matrix in this round	
	SET_UNIT_SU3_MATRIX(linka);	
	int* path = input_path + i*path_max_length;
	
	int lnkdir;
	int path0 = path[0];
	if (GOES_FORWARDS(path0)){
	    lnkdir=path0;
	}else{
	    lnkdir=OPP_DIR(path0);
	    COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(path0), new_mem_idx);
	    nbr_oddbit = nbr_oddbit^1;
	    
	}
	
	int nbr_idx = new_mem_idx >>1;
	if (nbr_oddbit){
	    LOAD_ODD_MATRIX( lnkdir, nbr_idx, LINKB);
	}else{
	    LOAD_EVEN_MATRIX( lnkdir, nbr_idx, LINKB);
	}
	
	GF_COMPUTE_RECONSTRUCT_SIGN(sign, lnkdir, new_x1, new_x2, new_x3, new_x4);
	RECONSTRUCT_MATRIX(lnkdir, nbr_idx, sign, linkb);
	if (GOES_FORWARDS(path0)){
	    COPY_SU3_MATRIX(linkb, linka);
	    COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(path0, new_mem_idx);
	    nbr_oddbit = nbr_oddbit^1;
	}else{
	    SU3_ADJOINT(linkb, linka);
	}	
	
	for(j=1; j < length[i]; j++){
	    
	    int lnkdir;
	    int pathj = path[j];
	    if (GOES_FORWARDS(pathj)){
		lnkdir=pathj;
	    }else{
		lnkdir=OPP_DIR(pathj);
		COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(pathj), new_mem_idx);
		nbr_oddbit = nbr_oddbit^1;

	    }
	    
	    int nbr_idx = new_mem_idx >>1;
	    if (nbr_oddbit){
		LOAD_ODD_MATRIX(lnkdir, nbr_idx, LINKB);
	    }else{
		LOAD_EVEN_MATRIX(lnkdir, nbr_idx, LINKB);
	    }
	    GF_COMPUTE_RECONSTRUCT_SIGN(sign, lnkdir, new_x1, new_x2, new_x3, new_x4);
	    RECONSTRUCT_MATRIX(lnkdir, nbr_idx, sign, linkb);
	    if (GOES_FORWARDS(pathj)){
	      MULT_SU3_NN_TEST(linka, linkb);
		
		COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(pathj, new_mem_idx);
		nbr_oddbit = nbr_oddbit^1;
		
		
	    }else{
		MULT_SU3_NA_TEST(linka, linkb);		
	    }
	    
	}//j
	SCALAR_MULT_ADD_SU3_MATRIX(staple, linka, path_coeff[i], staple);
    }//i
    

    //update mom 
    if (oddBit){
	LOAD_ODD_MATRIX(dir, sid, LINKA);
    }else{
	LOAD_EVEN_MATRIX(dir, sid, LINKA);
    }
    GF_COMPUTE_RECONSTRUCT_SIGN(sign, dir, x1, x2, x3, x4);
    RECONSTRUCT_MATRIX(dir, sid, sign, linka);
    MULT_SU3_NN_TEST(linka, staple);
    LOAD_ANTI_HERMITIAN(mymom, dir, sid, AH);
    UNCOMPRESS_ANTI_HERMITIAN(ah, linkb);
    SCALAR_MULT_SUB_SU3_MATRIX(linkb, linka, eb3, linka);
    MAKE_ANTI_HERMITIAN(linka, ah);
    
    WRITE_ANTI_HERMITIAN(mymom, dir, sid, AH);

    return;
}

void
gauge_force_cuda(FullMom  cudaMom, int dir, double eb3, FullGauge cudaSiteLink,
                 QudaGaugeParam* param, int** input_path, 
		 int* length, void* path_coeff, int num_paths, int max_length)
{

    int i, j;
    //input_path
    int bytes = num_paths*max_length* sizeof(int);
    int* input_path_d;
    cudaMalloc((void**)&input_path_d, bytes); checkCudaError();    
    cudaMemset(input_path_d, 0, bytes);checkCudaError();

    int* input_path_h = (int*)malloc(bytes);
    if (input_path_h == NULL){
	printf("ERROR: malloc failed for input_path_h in function %s\n", __FUNCTION__);
	exit(1);
    }
        
    memset(input_path_h, 0, bytes);
    for(i=0;i < num_paths;i++){
	for(j=0; j < length[i]; j++){
	    input_path_h[i*max_length + j] =input_path[i][j];
	}
    }

    cudaMemcpy(input_path_d, input_path_h, bytes, cudaMemcpyHostToDevice); checkCudaError();
    
    //length
    int* length_d;
    cudaMalloc((void**)&length_d, num_paths*sizeof(int)); checkCudaError();
    cudaMemcpy(length_d, length, num_paths*sizeof(int), cudaMemcpyHostToDevice); checkCudaError();
    
    //path_coeff
    int gsize;
    if (param->cuda_prec == QUDA_DOUBLE_PRECISION){
	gsize = sizeof(double);
    }else{
	gsize= sizeof(float);
    }     
    void* path_coeff_d;
    cudaMalloc((void**)&path_coeff_d, num_paths*gsize); checkCudaError();
    cudaMemcpy(path_coeff_d, path_coeff, num_paths*gsize, cudaMemcpyHostToDevice); checkCudaError();

    //compute the gauge forces
    int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
    dim3 blockDim(BLOCK_DIM, 1,1);
    dim3 gridDim(volume/blockDim.x, 1, 1);
    dim3 halfGridDim(volume/(2*blockDim.x), 1, 1);
    
    float2* momEven = (float2*)cudaMom.even;
    float2* momOdd = (float2*)cudaMom.odd;
    float4* linkEven = (float4*)cudaSiteLink.even;
    float4* linkOdd = (float4*)cudaSiteLink.odd;        

    cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
    cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
    parity_compute_gauge_force_kernel<0><<<halfGridDim, blockDim>>>(momEven, momOdd,
								  dir, eb3,
								  linkEven, linkOdd, 
								  input_path_d, length_d, (float*)path_coeff_d,
								  num_paths);   
    //odd
    /* The reason we do not switch the even/odd function input paramemters and the texture binding
     * is that we use the oddbit to decided where to load, in the kernel function
     */
    parity_compute_gauge_force_kernel<1><<<halfGridDim, blockDim>>>(momEven, momOdd,
								  dir, eb3,
								  linkEven, linkOdd, 
								  input_path_d, length_d, (float*)path_coeff_d,
								  num_paths);  
    

    
    cudaUnbindTexture(siteLink0TexSingle_recon);
    cudaUnbindTexture(siteLink1TexSingle_recon);
    
    checkCudaError();
    
    cudaFree(input_path_d); checkCudaError();
    free(input_path_h);
    cudaFree(length_d);
    cudaFree(path_coeff_d);

    

}


#undef LOAD_EVEN_MATRIX
#undef LOAD_ODD_MATRIX
#undef LOAD_MATRIX 
#undef LOAD_ANTI_HERMITIAN 
#undef WRITE_ANTI_HERMITIAN
#undef RECONSTRUCT_MATRIX
