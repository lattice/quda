#include <read_gauge.h>
#include <gauge_field.h>
#include <hisq_force_quda.h>
#include <hw_quda.h>
#include <hisq_force_macros.h>
#include<utility>


//DEBUG : control compile 
#define COMPILE_HISQ_DP_18 
#define COMPILE_HISQ_DP_12 
#define COMPILE_HISQ_SP_18 
#define COMPILE_HISQ_SP_12

// Disable texture read for now. Need to revisit this.
#define HISQ_SITE_MATRIX_LOAD_TEX 1
#define HISQ_NEW_OPROD_LOAD_TEX 1

namespace quda {
  namespace fermion_force {

    typedef struct hisq_kernel_param_s{
      unsigned long threads;
      int D1, D2,D3, D4, D1h;
      int base_idx[4];
      int ghostDim[4];
    }hisq_kernel_param_t;

    texture<int4, 1> newOprod0TexDouble;
    texture<int4, 1> newOprod1TexDouble;
    texture<float2, 1, cudaReadModeElementType>  newOprod0TexSingle;
    texture<float2, 1, cudaReadModeElementType> newOprod1TexSingle;


    void hisqForceInitCuda(QudaGaugeParam* param)
    {
      static int hisq_force_init_cuda_flag = 0; 
      
      if (hisq_force_init_cuda_flag){
	return;
      }
      hisq_force_init_cuda_flag=1;
	
      int Vh = param->X[0]*param->X[1]*param->X[2]*param->X[3]/2;
	
      fat_force_const_t hf;
#ifdef MULTI_GPU
      int Vh_ex = (param->X[0]+4)*(param->X[1]+4)*(param->X[2]+4)*(param->X[3]+4)/2;
      hf.site_ga_stride = Vh_ex + param->site_ga_pad;;
      hf.color_matrix_stride = Vh_ex;
#else
      hf.site_ga_stride = Vh + param->site_ga_pad;
      hf.color_matrix_stride = Vh;
#endif
      hf.mom_ga_stride = Vh + param->mom_ga_pad;
	
      cudaMemcpyToSymbol("hf", &hf, sizeof(fat_force_const_t));
    }
    




    // struct for holding the fattening path coefficients
    template<class Real>
    struct PathCoefficients
    {
      Real one; 
      Real three;
      Real five;
      Real seven;
      Real naik;
      Real lepage;
    };


    inline __device__ float2 operator*(float a, const float2 & b)
    {
      return make_float2(a*b.x,a*b.y);
    }

    inline __device__ double2 operator*(double a, const double2 & b)
    {
      return make_double2(a*b.x,a*b.y);
    }

    inline __device__ const float2 & operator+=(float2 & a, const float2 & b)
    {
      a.x += b.x;
      a.y += b.y;
      return a;
    }

    inline __device__ const double2 & operator+=(double2 & a, const double2 & b)
    {
      a.x += b.x;
      a.y += b.y;
      return a;
    }

    inline __device__ const float4 & operator+=(float4 & a, const float4 & b)
    {
      a.x += b.x;
      a.y += b.y;
      a.z += b.z;
      a.w += b.w;
      return a;
    }

    // Replication of code 
    // This structure is already defined in 
    // unitarize_utilities.h

    template<class T>
    struct RealTypeId; 

    template<>
    struct RealTypeId<float2>
    {
      typedef float Type;
    };

    template<>
    struct RealTypeId<double2>
    {
      typedef double Type;
    };


    template<class T>
    inline __device__
    void adjointMatrix(T* mat)
    {
#define CONJ_INDEX(i,j) j*3 + i

      T tmp;
      mat[CONJ_INDEX(0,0)] = conj(mat[0]);
      mat[CONJ_INDEX(1,1)] = conj(mat[4]);
      mat[CONJ_INDEX(2,2)] = conj(mat[8]);
      tmp  = conj(mat[1]);
      mat[CONJ_INDEX(1,0)] = conj(mat[3]);
      mat[CONJ_INDEX(0,1)] = tmp;	
      tmp = conj(mat[2]);
      mat[CONJ_INDEX(2,0)] = conj(mat[6]);
      mat[CONJ_INDEX(0,2)] = tmp;
      tmp = conj(mat[5]);
      mat[CONJ_INDEX(2,1)] = conj(mat[7]);
      mat[CONJ_INDEX(1,2)] = tmp;

#undef CONJ_INDEX
      return;
    }


    template<int N, class T>
    inline __device__
    void loadMatrixFromField(const T* const field_even, const T* const field_odd,
			     int dir, int idx, T* const mat, int oddness, int stride)
    {
      const T* const field = (oddness)?field_odd:field_even;
      for(int i = 0;i < N ;i++){
	mat[i] = field[idx + dir*N*stride + i*stride];          
      }
      return;
    }

    template<class T>
    inline __device__
    void loadMatrixFromField(const T* const field_even, const T* const field_odd,
			     int dir, int idx, T* const mat, int oddness, int stride)
    {
      loadMatrixFromField<9> (field_even, field_odd, dir, idx, mat, oddness, stride);
      return;
    }
    
    

    inline __device__
    void loadMatrixFromField(const float4* const field_even, const float4* const field_odd, 
			     int dir, int idx, float2* const mat, int oddness, int stride)
    {
      const float4* const field = oddness?field_odd: field_even;
      float4 tmp;
      tmp = field[idx + dir*stride*3];
      mat[0] = make_float2(tmp.x, tmp.y);
      mat[1] = make_float2(tmp.z, tmp.w);
      tmp = field[idx + dir*stride*3 + stride];
      mat[2] = make_float2(tmp.x, tmp.y);
      mat[3] = make_float2(tmp.z, tmp.w);
      tmp = field[idx + dir*stride*3 + 2*stride];
      mat[4] = make_float2(tmp.x, tmp.y);
      mat[5] = make_float2(tmp.z, tmp.w);
      return;
    }

    template<class T>
    inline __device__
    void loadMatrixFromField(const T* const field_even, const T* const field_odd, int idx, T* const mat, int oddness, int stride)
    {
      const T* const field = (oddness)?field_odd:field_even;
      mat[0] = field[idx];
      mat[1] = field[idx + stride];
      mat[2] = field[idx + stride*2];
      mat[3] = field[idx + stride*3];
      mat[4] = field[idx + stride*4];
      mat[5] = field[idx + stride*5];
      mat[6] = field[idx + stride*6];
      mat[7] = field[idx + stride*7];
      mat[8] = field[idx + stride*8];

      return;
    }
    

#define  addMatrixToNewOprod(mat,  dir, idx, coeff, field_even, field_odd, oddness)     do { \
      RealA* const field = (oddness)?field_odd: field_even;		\
      RealA value[9];							\
      value[0] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*hf.color_matrix_stride*9); \
      value[1] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*hf.color_matrix_stride*9 + hf.color_matrix_stride); \
      value[2] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*hf.color_matrix_stride*9 + 2*hf.color_matrix_stride); \
      value[3] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*hf.color_matrix_stride*9 + 3*hf.color_matrix_stride); \
      value[4] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*hf.color_matrix_stride*9 + 4*hf.color_matrix_stride); \
      value[5] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*hf.color_matrix_stride*9 + 5*hf.color_matrix_stride); \
      value[6] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*hf.color_matrix_stride*9 + 6*hf.color_matrix_stride); \
      value[7] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*hf.color_matrix_stride*9 + 7*hf.color_matrix_stride); \
      value[8] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*hf.color_matrix_stride*9 + 8*hf.color_matrix_stride); \
      field[idx + dir*hf.color_matrix_stride*9]          = value[0] + coeff*mat[0]; \
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride]     = value[1] + coeff*mat[1];	\
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*2]   = value[2] + coeff*mat[2];	\
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*3]   = value[3] + coeff*mat[3];	\
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*4]   = value[4] + coeff*mat[4];	\
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*5]   = value[5] + coeff*mat[5];	\
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*6]   = value[6] + coeff*mat[6];	\
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*7]   = value[7] + coeff*mat[7];	\
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*8]   = value[8] + coeff*mat[8];	\
    }while(0)					
     


    // only works if Promote<T,U>::Type = T

    template<class T, class U>   
    inline __device__
    void addMatrixToField(const T* const mat, int dir, int idx, U coeff, 
			  T* const field_even, T* const field_odd, int oddness)
    {
      T* const field = (oddness)?field_odd: field_even;
      field[idx + dir*hf.color_matrix_stride*9]          += coeff*mat[0];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride]     += coeff*mat[1];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*2]   += coeff*mat[2];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*3]   += coeff*mat[3];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*4]   += coeff*mat[4];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*5]   += coeff*mat[5];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*6]   += coeff*mat[6];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*7]   += coeff*mat[7];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*8]   += coeff*mat[8];

      return;
    }


    template<class T, class U>
    inline __device__
    void addMatrixToField(const T* const mat, int idx, U coeff, T* const field_even,
			  T* const field_odd, int oddness)
    {
      T* const field = (oddness)?field_odd: field_even;
      field[idx ]         += coeff*mat[0];
      field[idx + hf.color_matrix_stride]     += coeff*mat[1];
      field[idx + hf.color_matrix_stride*2]   += coeff*mat[2];
      field[idx + hf.color_matrix_stride*3]   += coeff*mat[3];
      field[idx + hf.color_matrix_stride*4]   += coeff*mat[4];
      field[idx + hf.color_matrix_stride*5]   += coeff*mat[5];
      field[idx + hf.color_matrix_stride*6]   += coeff*mat[6];
      field[idx + hf.color_matrix_stride*7]   += coeff*mat[7];
      field[idx + hf.color_matrix_stride*8]   += coeff*mat[8];

      return;
    }

    template<class T, class U>
    inline __device__
    void addMatrixToField_test(const T* const mat, int idx, U coeff, T* const field_even,
			       T* const field_odd, int oddness)
    {
      T* const field = (oddness)?field_odd: field_even;
      //T oldvalue=field[idx];
      field[idx ]         += coeff*mat[0];
      field[idx + hf.color_matrix_stride]     += coeff*mat[1];
      field[idx + hf.color_matrix_stride*2]   += coeff*mat[2];
      field[idx + hf.color_matrix_stride*3]   += coeff*mat[3];
      field[idx + hf.color_matrix_stride*4]   += coeff*mat[4];
      field[idx + hf.color_matrix_stride*5]   += coeff*mat[5];
      field[idx + hf.color_matrix_stride*6]   += coeff*mat[6];
      field[idx + hf.color_matrix_stride*7]   += coeff*mat[7];
      field[idx + hf.color_matrix_stride*8]   += coeff*mat[8];

#if (!defined(__CUDA_ARCH__) || (__COMPUTE_CAPABILITY__>=200))
      printf("value is  coeff(%f) * mat[0].x(%f)=%f\n", coeff, mat[0].x, field[idx].x);
#endif
      return;
    }

    template<class T>
    inline __device__
    void storeMatrixToField(const T* const mat, int dir, int idx, T* const field_even, T* const field_odd, int oddness)
    {
      T* const field = (oddness)?field_odd: field_even;
      field[idx + dir*hf.color_matrix_stride*9]          = mat[0];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride]     = mat[1];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*2]   = mat[2];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*3]   = mat[3];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*4]   = mat[4];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*5]   = mat[5];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*6]   = mat[6];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*7]   = mat[7];
      field[idx + dir*hf.color_matrix_stride*9 + hf.color_matrix_stride*8]   = mat[8];

      return;
    }


    template<class T>
    inline __device__
    void storeMatrixToField(const T* const mat, int idx, T* const field_even, T* const field_odd, int oddness)
    {
      T* const field = (oddness)?field_odd: field_even;
      field[idx]          = mat[0];
      field[idx + hf.color_matrix_stride]     = mat[1];
      field[idx + hf.color_matrix_stride*2]   = mat[2];
      field[idx + hf.color_matrix_stride*3]   = mat[3];
      field[idx + hf.color_matrix_stride*4]   = mat[4];
      field[idx + hf.color_matrix_stride*5]   = mat[5];
      field[idx + hf.color_matrix_stride*6]   = mat[6];
      field[idx + hf.color_matrix_stride*7]   = mat[7];
      field[idx + hf.color_matrix_stride*8]   = mat[8];

      return;
    }


    template<class T, class U> 
    inline __device__
    void storeMatrixToMomentumField(const T* const mat, int dir, int idx, U coeff, 
				    T* const mom_even, T* const mom_odd, int oddness)
    {
      T* const mom_field = (oddness)?mom_odd:mom_even;
      T temp2;
      temp2.x = (mat[1].x - mat[3].x)*0.5*coeff;
      temp2.y = (mat[1].y + mat[3].y)*0.5*coeff;
      mom_field[idx + dir*hf.mom_ga_stride*5] = temp2;	
	  
      temp2.x = (mat[2].x - mat[6].x)*0.5*coeff;
      temp2.y = (mat[2].y + mat[6].y)*0.5*coeff;
      mom_field[idx + dir*hf.mom_ga_stride*5 + hf.mom_ga_stride] = temp2;
	  
      temp2.x = (mat[5].x - mat[7].x)*0.5*coeff;
      temp2.y = (mat[5].y + mat[7].y)*0.5*coeff;
      mom_field[idx + dir*hf.mom_ga_stride*5 + hf.mom_ga_stride*2] = temp2;

      const typename RealTypeId<T>::Type temp = (mat[0].y + mat[4].y + mat[8].y)*0.3333333333333333333333333;
      temp2.x =  (mat[0].y-temp)*coeff; 
      temp2.y =  (mat[4].y-temp)*coeff;
      mom_field[idx + dir*hf.mom_ga_stride*5 + hf.mom_ga_stride*3] = temp2;
	  
      temp2.x = (mat[8].y - temp)*coeff;
      temp2.y = 0.0;
      mom_field[idx + dir*hf.mom_ga_stride*5 + hf.mom_ga_stride*4] = temp2;
 
      return;
    }

     // Struct to determine the coefficient sign at compile time
    template<int pos_dir, int odd_lattice>
    struct CoeffSign
    {
      static const int result = -1;
    };

    template<>
    struct CoeffSign<0,1>
    {
      static const int result = -1;
    }; 

    template<>
    struct CoeffSign<0,0>
    {
      static const int result = 1;
    };

    template<>
    struct CoeffSign<1,1>
    {
      static const int result = 1;
    };

    template<int odd_lattice>
    struct Sign
    {
      static const int result = 1;
    };

    template<>
    struct Sign<1>
    {
      static const int result = -1;
    };

    template<class RealX>
    struct ArrayLength
    {
      static const int result=9;
    };

    template<>
    struct ArrayLength<float4>
    {
      static const int result=5;
    };
 


     

    // reconstructSign doesn't do anything right now, 
    // but it will, soon.
    template<typename T>
    __device__ void reconstructSign(int* const sign, int dir, const T i[4]){

 
      *sign=1;
      
      switch(dir){
      case XUP:
	if( (i[3]&1)==1) *sign=-1;
	break;	  

      case YUP:
	if( ((i[3]+i[0])&1) == 1) *sign=-1; 
	break;
	
      case ZUP:
	if( ((i[3]+i[0]+i[1])&1) == 1) *sign=-1; 
	break;
	
      case TUP:
#ifdef MULTI_GPU	
	if( (i[3] == X4+1 && PtNm1)
	    || (i[3] == 1 && Pt0)) {
	  *sign=-1; 
	}
#else
	if(i[3] == X4m1) *sign=-1; 
#endif
	break;
	
      default:
#if (!defined(__CUDA_ARCH__) || (__COMPUTE_CAPABILITY__>=200))
	printf("Error: invalid dir\n");
#endif
	break;
      }
      return;
    }






    template<class RealA, int oddBit>
    __global__ void 
    do_one_link_term_kernel(const RealA* const oprodEven, const RealA* const oprodOdd,
			    int sig, typename RealTypeId<RealA>::Type coeff,
			    RealA* const outputEven, RealA* const outputOdd, const int threads)
    {
      int sid = blockIdx.x * blockDim.x + threadIdx.x;
      if (sid >= threads) return;
#ifdef MULTI_GPU
      int x[4];
      int z1 = sid/X1h;
      int x1h = sid - z1*X1h;
      int z2 = z1/X2;
      x[1] = z1 - z2*X2;
      x[3] = z2/X3;
      x[2] = z2 - x[3]*X3;
      int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
      x[0] = 2*x1h + x1odd;
      //int X = 2*sid + x1odd;

      int new_sid = ( (x[3]+2)*E3E2E1+(x[2]+2)*E2E1+(x[1]+2)*E1+(x[0]+2))>>1 ;
#else
      int new_sid = sid;
#endif
      RealA COLOR_MAT_W[ArrayLength<RealA>::result];
      if(GOES_FORWARDS(sig)){
	loadMatrixFromField(oprodEven, oprodOdd, sig, new_sid, COLOR_MAT_W, oddBit, hf.color_matrix_stride);
	addMatrixToField(COLOR_MAT_W, sig, new_sid, coeff, outputEven, outputOdd, oddBit);
      }
      return;
    }


#define DD_CONCAT(n,r) n ## r ## kernel

#define HISQ_KERNEL_NAME(a,b) DD_CONCAT(a,b)
    //precision: 0 is for double, 1 is for single

#define NEWOPROD_EVEN_TEX newOprod0TexDouble
#define NEWOPROD_ODD_TEX newOprod1TexDouble
#if (HISQ_NEW_OPROD_LOAD_TEX == 1)
#define LOAD_TEX_ENTRY(tex, field, idx)  READ_DOUBLE2_TEXTURE(tex, field, idx)
#else
#define LOAD_TEX_ENTRY(tex, field, idx) field[idx]
#endif

    //double precision, recon=18
#define PRECISION 0
#define RECON 18
#if (HISQ_SITE_MATRIX_LOAD_TEX == 1)
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   HISQ_LOAD_MATRIX_18_DOUBLE_TEX((oddness)?siteLink1TexDouble:siteLink0TexDouble,  (oddness)?linkOdd:linkEven, dir, idx, var, hf.site_ga_stride)        
#else
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   loadMatrixFromField(linkEven, linkOdd, dir, idx, var, oddness, hf.site_ga_stride)  
#endif
#define COMPUTE_LINK_SIGN(sign, dir, x) 
#define RECONSTRUCT_SITE_LINK(var, sign)
#include "hisq_paths_force_core.h"
#undef PRECISION
#undef RECON
#undef HISQ_LOAD_LINK
#undef COMPUTE_LINK_SIGN
#undef RECONSTRUCT_SITE_LINK

    //double precision, recon=12
#define PRECISION 0
#define RECON 12
#if (HISQ_SITE_MATRIX_LOAD_TEX == 1)
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   HISQ_LOAD_MATRIX_12_DOUBLE_TEX((oddness)?siteLink1TexDouble:siteLink0TexDouble,  (oddness)?linkOdd:linkEven,dir, idx, var, hf.site_ga_stride)        
#else
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   loadMatrixFromField<6>(linkEven, linkOdd, dir, idx, var, oddness, hf.site_ga_stride)  
#endif
#define COMPUTE_LINK_SIGN(sign, dir, x) reconstructSign(sign, dir, x)
#define RECONSTRUCT_SITE_LINK(var, sign)  FF_RECONSTRUCT_LINK_12(var, sign)
#include "hisq_paths_force_core.h"
#undef PRECISION
#undef RECON
#undef HISQ_LOAD_LINK
#undef COMPUTE_LINK_SIGN
#undef RECONSTRUCT_SITE_LINK       
#undef NEWOPROD_EVEN_TEX 
#undef NEWOPROD_ODD_TEX 
#undef LOAD_TEX_ENTRY


#define NEWOPROD_EVEN_TEX newOprod0TexSingle
#define NEWOPROD_ODD_TEX newOprod1TexSingle

#if (HISQ_NEW_OPROD_LOAD_TEX==1)
#define LOAD_TEX_ENTRY(tex, field, idx)  tex1Dfetch(tex,idx)
#else
#define LOAD_TEX_ENTRY(tex, field, idx) field[idx]
#endif

    //single precision, recon=18  
#define PRECISION 1
#define RECON 18
#if (HISQ_SITE_MATRIX_LOAD_TEX == 1)
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   HISQ_LOAD_MATRIX_18_SINGLE_TEX((oddness)?siteLink1TexSingle:siteLink0TexSingle, dir, idx, var, hf.site_ga_stride)        
#else
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   loadMatrixFromField(linkEven, linkOdd, dir, idx, var, oddness, hf.site_ga_stride)  
#endif
#define COMPUTE_LINK_SIGN(sign, dir, x) 
#define RECONSTRUCT_SITE_LINK(var, sign)
#include "hisq_paths_force_core.h"
#undef PRECISION
#undef RECON
#undef HISQ_LOAD_LINK
#undef COMPUTE_LINK_SIGN
#undef RECONSTRUCT_SITE_LINK

    //single precision, recon=12
#define PRECISION 1
#define RECON 12
#if (HISQ_SITE_MATRIX_LOAD_TEX == 1)
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   HISQ_LOAD_MATRIX_12_SINGLE_TEX((oddness)?siteLink1TexSingle_recon:siteLink0TexSingle_recon, dir, idx, var, hf.site_ga_stride)        
#else
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   loadMatrixFromField(linkEven, linkOdd, dir, idx, var, oddness, hf.site_ga_stride)  
#endif
#define COMPUTE_LINK_SIGN(sign, dir, x) reconstructSign(sign, dir, x)
#define RECONSTRUCT_SITE_LINK(var, sign)  FF_RECONSTRUCT_LINK_12(var, sign)
#include "hisq_paths_force_core.h"
#undef PRECISION
#undef RECON
#undef HISQ_LOAD_LINK
#undef COMPUTE_LINK_SIGN
#undef RECONSTRUCT_SITE_LINK
#undef NEWOPROD_EVEN_TEX 
#undef NEWOPROD_ODD_TEX 
#undef LOAD_TEX_ENTRY




    template<class RealA, class RealB>
    class MiddleLink : public Tunable {

    private:
      const cudaGaugeField &link;
      const cudaGaugeField &oprod;
      const cudaGaugeField &Qprev;
      const int sig;
      const int mu;
      typename RealTypeId<RealA>::Type &coeff; 
      cudaGaugeField &Pmu;
      cudaGaugeField &P3;
      cudaGaugeField &Qmu;
      cudaGaugeField &newOprod;
      const hisq_kernel_param_t &kparam;

      int sharedBytesPerThread() const { return 0; }
      int sharedBytesPerBlock(const TuneParam &) const { return 0; }


      // don't tune the grid dimension
      bool advanceGridDim(TuneParam &param) const { return false; }

      // generalize Tunable::advanceBlockDim() to also set gridDim, with extra checking to ensure that gridDim isn't too large for the device
      bool advanceBlockDim(TuneParam &param) const
      {
	const unsigned int max_threads = deviceProp.maxThreadsDim[0];
	const unsigned int max_blocks = deviceProp.maxGridSize[0];
	const unsigned int max_shared = 16384; // FIXME: use deviceProp.sharedMemPerBlock;
	const int step = deviceProp.warpSize;
	bool ret;
	param.block.x += step;
	if (param.block.x > max_threads || sharedBytesPerThread()*param.block.x > max_shared) {
	  param.block = dim3((kparam.threads+max_blocks-1)/max_blocks, 1, 1); // ensure the blockDim is large enough, given the limit on gridDim
	  param.block.x = ((param.block.x+step-1) / step) * step; // round up to the nearest "step"
	  if (param.block.x > max_threads) errorQuda("Local lattice volume is too large for device");
	  ret = false;
	} else {
	  ret = true;
	}
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
	return ret;
      }

    public:
      MiddleLink(const cudaGaugeField &link, 
		 const cudaGaugeField &oprod,
		 const cudaGaugeField &Qprev,
		 int sig, int mu,
		 typename RealTypeId<RealA>::Type coeff, 
		 cudaGaugeField &Pmu, // write only
		 cudaGaugeField &P3,  // write only
		 cudaGaugeField &Qmu,
		 cudaGaugeField &newOprod,
		 const hisq_kernel_param_t &kparam) :
	link(link), oprod(oprod), Qprev(Qprev), sig(sig), mu(mu), 
	coeff(coeff), Pmu(Pmu), P3(P3), Qmu(Qmu), newOprod(newOprod), kparam(kparam)
      {	; }
      // need alternative constructor to hack around null pointer passing
      MiddleLink(const cudaGaugeField &link, 
		 const cudaGaugeField &oprod,
		 int sig, int mu,
		 typename RealTypeId<RealA>::Type coeff, 
		 cudaGaugeField &Pmu, // write only
		 cudaGaugeField &P3,  // write only
		 cudaGaugeField &Qmu,
		 cudaGaugeField &newOprod,
		 const hisq_kernel_param_t &kparam) :
	link(link), oprod(oprod), Qprev(link), sig(sig), mu(mu), 
	coeff(coeff), Pmu(Pmu), P3(P3), Qmu(Qmu), newOprod(newOprod), kparam(kparam)
      {	; }
      virtual ~MiddleLink() { ; }

      TuneKey tuneKey() const {
	std::stringstream vol, aux;
	vol << kparam.D1 << "x";
	vol << kparam.D2 << "x";
	vol << kparam.D3 << "x";
	vol << kparam.D4;    
	aux << "threads=" << kparam.threads << ",prec=" << link.Precision();
	aux << ",recon=" << link.Reconstruct() << ",sig=" << sig << ",mu=" << mu;
	return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }  

      
#define CALL_ARGUMENTS(typeA, typeB) <<<tp.grid, tp.block>>>		\
      ((typeA*)oprod.Even_p(), (typeA*)oprod.Odd_p(),			\
       (typeA*)Qprev_even, (typeA*)Qprev_odd,				\
       (typeB*)link.Even_p(), (typeB*)link.Odd_p(),			\
       sig, mu, (typename RealTypeId<typeA>::Type)coeff,		\
       (typeA*)Pmu.Even_p(), (typeA*)Pmu.Odd_p(),			\
       (typeA*)P3.Even_p(), (typeA*)P3.Odd_p(),				\
       (typeA*)Qmu.Even_p(), (typeA*)Qmu.Odd_p(),			\
       (typeA*)newOprod.Even_p(), (typeA*)newOprod.Odd_p(), kparam)
      

#define CALL_MIDDLE_LINK_KERNEL(sig_sign, mu_sign)			\
	if(oddness_change ==0 ){					\
	  if(sizeof(RealA) == sizeof(float2)){				\
	    if(recon  == QUDA_RECONSTRUCT_NO){				\
	      do_middle_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(float2, float2); \
	      do_middle_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(float2, float2); \
	    }else{							\
	      do_middle_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(float2, float4); \
	      do_middle_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(float2, float4); \
	    }								\
	  }else{							\
	    if(recon  == QUDA_RECONSTRUCT_NO){				\
	      do_middle_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(double2, double2); \
	      do_middle_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(double2, double2); \
	    }else{							\
	      do_middle_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(double2, double2); \
	      do_middle_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(double2, double2); \
	    }								\
	  }								\
	}else{								\
	  if(sizeof(RealA) == sizeof(float2)){				\
	    if(recon  == QUDA_RECONSTRUCT_NO){				\
	      do_middle_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(float2, float2); \
	      do_middle_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(float2, float2); \
	    }else{							\
	      do_middle_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(float2, float4); \
	      do_middle_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(float2, float4); \
	    }								\
	  }else{							\
	    if(recon  == QUDA_RECONSTRUCT_NO){				\
	      do_middle_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(double2, double2); \
	      do_middle_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(double2, double2); \
	    }else{							\
	      do_middle_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(double2, double2); \
	      do_middle_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(double2, double2); \
	    }								\
	  }								\
	}



      void apply(const cudaStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
	QudaReconstructType recon = link.Reconstruct();
	int oddness_change = (kparam.base_idx[0] + kparam.base_idx[1]
			      + kparam.base_idx[2] + kparam.base_idx[3])&1;

	const void *Qprev_even = (&Qprev == &link) ? NULL : Qprev.Even_p();
	const void *Qprev_odd = (&Qprev == &link) ? NULL : Qprev.Odd_p();
		

	if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){	
	  CALL_MIDDLE_LINK_KERNEL(1,1);
	}else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	  CALL_MIDDLE_LINK_KERNEL(1,0);
	}else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	  CALL_MIDDLE_LINK_KERNEL(0,1);
	}else{
	  CALL_MIDDLE_LINK_KERNEL(0,0);
	}
      }
      
#undef CALL_ARGUMENTS	
#undef CALL_MIDDLE_LINK_KERNEL
      
      void preTune() {
	Pmu.backup();
	P3.backup();
	Qmu.backup();
	newOprod.backup();
      }

      void postTune() {
	Pmu.restore();
	P3.restore();
	Qmu.restore();
	newOprod.restore();
      }

      virtual void initTuneParam(TuneParam &param) const
      {
	const unsigned int max_threads = deviceProp.maxThreadsDim[0];
	const unsigned int max_blocks = deviceProp.maxGridSize[0];
	const int step = deviceProp.warpSize;
	param.block = dim3((kparam.threads+max_blocks-1)/max_blocks, 1, 1); // ensure the blockDim is large enough, given the limit on gridDim
	param.block.x = ((param.block.x+step-1) / step) * step; // round up to the nearest "step"
	if (param.block.x > max_threads) errorQuda("Local lattice volume is too large for device");
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
	param.shared_bytes = sharedBytesPerThread()*param.block.x > sharedBytesPerBlock(param) ?
	  sharedBytesPerThread()*param.block.x : sharedBytesPerBlock(param);
      }
      
      /** sets default values for when tuning is disabled */
      void defaultTuneParam(TuneParam &param) const {
	initTuneParam(param);
      }

      long long flops() const { return 0; }
    };


    template<class RealA, class RealB>
    class LepageMiddleLink : public Tunable {

    private:
      const cudaGaugeField &link;
      const cudaGaugeField &oprod;
      const cudaGaugeField &Qprev;
      const int sig;
      const int mu;
      typename RealTypeId<RealA>::Type &coeff; 
      cudaGaugeField &P3; // write only
      cudaGaugeField &newOprod;
      const hisq_kernel_param_t &kparam;

      int sharedBytesPerThread() const { return 0; }
      int sharedBytesPerBlock(const TuneParam &) const { return 0; }

      // don't tune the grid dimension
      bool advanceGridDim(TuneParam &param) const { return false; }
      bool advanceBlockDim(TuneParam &param) const {
	bool rtn = Tunable::advanceBlockDim(param);
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
	return rtn;
      }

    public:
      LepageMiddleLink(const cudaGaugeField &link, 
		       const cudaGaugeField &oprod, 
		       const cudaGaugeField &Qprev,
		       int sig, int mu,
		       typename RealTypeId<RealA>::Type coeff, 
		       cudaGaugeField &P3, cudaGaugeField &newOprod,
		       const hisq_kernel_param_t &kparam) :
	link(link), oprod(oprod), Qprev(Qprev), sig(sig), mu(mu), 
	coeff(coeff), P3(P3), newOprod(newOprod), kparam(kparam)
      {	; }
      virtual ~LepageMiddleLink() { ; }

      TuneKey tuneKey() const {
	std::stringstream vol, aux;
	vol << kparam.D1 << "x";
	vol << kparam.D2 << "x";
	vol << kparam.D3 << "x";
	vol << kparam.D4;    
	aux << "threads=" << kparam.threads << ",prec=" << link.Precision();
	aux << ",recon=" << link.Reconstruct() << ",sig=" << sig << ",mu=" << mu;
	return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }  
      
#define CALL_ARGUMENTS(typeA, typeB) <<<tp.grid, tp.block>>>		\
      ((typeA*)oprod.Even_p(), (typeA*)oprod.Odd_p(),			\
       (typeA*)Qprev.Even_p(), (typeA*)Qprev.Odd_p(),			\
       (typeB*)link.Even_p(), (typeB*)link.Odd_p(),			\
       sig, mu, (typename RealTypeId<typeA>::Type)coeff,		\
       (typeA*)P3.Even_p(), (typeA*)P3.Odd_p(),				\
       (typeA*)newOprod.Even_p(), (typeA*)newOprod.Odd_p(),		\
       kparam)
	
 #define CALL_MIDDLE_LINK_KERNEL(sig_sign, mu_sign)			\
      if(oddness_change == 0){						\
	if(sizeof(RealA) == sizeof(float2)){				\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_lepage_middle_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(float2, float2); \
	    do_lepage_middle_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(float2, float2); \
	    }else{							\
	      do_lepage_middle_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(float2, float4); \
	      do_lepage_middle_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(float2, float4); \
	    }								\
	  }else{							\
	    if(recon  == QUDA_RECONSTRUCT_NO){				\
	      do_lepage_middle_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(double2, double2); \
	      do_lepage_middle_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(double2, double2); \
	    }else{							\
	      do_lepage_middle_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(double2, double2); \
	      do_lepage_middle_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(double2, double2); \
	    }								\
	  }								\
	}else{								\
	  if(sizeof(RealA) == sizeof(float2)){				\
	    if(recon  == QUDA_RECONSTRUCT_NO){				\
	      do_lepage_middle_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(float2, float2); \
	      do_lepage_middle_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(float2, float2); \
	    }else{							\
	      do_lepage_middle_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(float2, float4); \
	      do_lepage_middle_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(float2, float4); \
	    }								\
	  }else{							\
	    if(recon  == QUDA_RECONSTRUCT_NO){				\
	      do_lepage_middle_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(double2, double2); \
	      do_lepage_middle_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(double2, double2); \
	    }else{							\
	      do_lepage_middle_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(double2, double2); \
	      do_lepage_middle_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(double2, double2); \
	    }								\
	  }								\
	}     
      
      void apply(const cudaStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
	QudaReconstructType recon = link.Reconstruct();
	int oddness_change = (kparam.base_idx[0] + kparam.base_idx[1]
			      + kparam.base_idx[2] + kparam.base_idx[3])&1;
	
	
	if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){	
	  CALL_MIDDLE_LINK_KERNEL(1,1);
	}else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	  CALL_MIDDLE_LINK_KERNEL(1,0);
	}else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	  CALL_MIDDLE_LINK_KERNEL(0,1);
	}else{
	  CALL_MIDDLE_LINK_KERNEL(0,0);
	}
	
      }
      
#undef CALL_ARGUMENTS	
#undef CALL_MIDDLE_LINK_KERNEL
      
      void preTune() {
	P3.backup();
	newOprod.backup();
      }

      void postTune() {
	P3.restore();
	newOprod.restore();
      }

      virtual void initTuneParam(TuneParam &param) const
      {
	Tunable::initTuneParam(param);
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
      }
      
      /** sets default values for when tuning is disabled */
      void defaultTuneParam(TuneParam &param) const
      {
	Tunable::defaultTuneParam(param);
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
      }

      long long flops() const { return 0; }
    };
    
    template<class RealA, class RealB>
    class SideLink : public Tunable {

    private:
      const cudaGaugeField &link;
      const cudaGaugeField &P3;
      const cudaGaugeField &oprod;
      const int sig;
      const int mu;
      typename RealTypeId<RealA>::Type &coeff; 
      typename RealTypeId<RealA>::Type &accumu_coeff;
      cudaGaugeField &shortP;
      cudaGaugeField &newOprod;
      const hisq_kernel_param_t &kparam;

      int sharedBytesPerThread() const { return 0; }
      int sharedBytesPerBlock(const TuneParam &) const { return 0; }

      // don't tune the grid dimension
      bool advanceGridDim(TuneParam &param) const { return false; }
      bool advanceBlockDim(TuneParam &param) const {
	bool rtn = Tunable::advanceBlockDim(param);
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
	return rtn;
      }

    public:
      SideLink(const cudaGaugeField &link, 
	       const cudaGaugeField &P3,
	       const cudaGaugeField &oprod,
	       int sig, int mu, 
	       typename RealTypeId<RealA>::Type coeff, 
	       typename RealTypeId<RealA>::Type accumu_coeff,
	       cudaGaugeField &shortP,
	       cudaGaugeField &newOprod,
	       const hisq_kernel_param_t &kparam) :
	link(link), P3(P3), oprod(oprod), 
	sig(sig), mu(mu), coeff(coeff), accumu_coeff(accumu_coeff), 
	shortP(shortP), newOprod(newOprod), kparam(kparam)
      {	; }
      virtual ~SideLink() { ; }

      TuneKey tuneKey() const {
	std::stringstream vol, aux;
	vol << kparam.D1 << "x";
	vol << kparam.D2 << "x";
	vol << kparam.D3 << "x";
	vol << kparam.D4;    
	aux << "threads=" << kparam.threads << ",prec=" << link.Precision();
	aux << ",recon=" << link.Reconstruct() << ",sig=" << sig << ",mu=" << mu;
	return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }  
      
#define CALL_ARGUMENTS(typeA, typeB) <<<tp.grid, tp.block>>>		\
      ((typeA*)P3.Even_p(), (typeA*)P3.Odd_p(),				\
       (typeA*)oprod.Even_p(),  (typeA*)oprod.Odd_p(),			\
       (typeB*)link.Even_p(), (typeB*)link.Odd_p(),			\
       sig, mu,								\
       (typename RealTypeId<typeA>::Type) coeff,			\
       (typename RealTypeId<typeA>::Type) accumu_coeff,			\
       (typeA*)shortP.Even_p(), (typeA*)shortP.Odd_p(),			\
       (typeA*)newOprod.Even_p(), (typeA*)newOprod.Odd_p(),		\
       kparam)

#define CALL_SIDE_LINK_KERNEL(sig_sign, mu_sign)			\
      if(oddness_change == 0){						\
	if(sizeof(RealA) == sizeof(float2)){				\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_side_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(float2, float2); \
	    do_side_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(float2, float2); \
	  }else{							\
	    do_side_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(float2, float4); \
	    do_side_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(float2, float4); \
	  }								\
	}else{								\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_side_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(double2, double2); \
	    do_side_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(double2, double2); \
	  }else{							\
	    do_side_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(double2, double2); \
	    do_side_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(double2, double2); \
	  }								\
	}								\
      }else{								\
	if(sizeof(RealA) == sizeof(float2)){				\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_side_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(float2, float2); \
	    do_side_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(float2, float2); \
	  }else{							\
	    do_side_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(float2, float4); \
	    do_side_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(float2, float4); \
	  }								\
	}else{								\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_side_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(double2, double2); \
	    do_side_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(double2, double2); \
	  }else{							\
	    do_side_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(double2, double2); \
	    do_side_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(double2, double2); \
	  }								\
	}								\
      }
      
      void apply(const cudaStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
	QudaReconstructType recon = link.Reconstruct();
	int oddness_change = (kparam.base_idx[0] + kparam.base_idx[1]
			      + kparam.base_idx[2] + kparam.base_idx[3])&1;
	
	if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
	  CALL_SIDE_LINK_KERNEL(1,1);
	}else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	  CALL_SIDE_LINK_KERNEL(1,0); 
	}else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	  CALL_SIDE_LINK_KERNEL(0,1);
	}else{
	  CALL_SIDE_LINK_KERNEL(0,0);
	}
      }
      
#undef CALL_SIDE_LINK_KERNEL
#undef CALL_ARGUMENTS      
      
      void preTune() {
	shortP.backup();
	newOprod.backup();
      }

      void postTune() {
	shortP.restore();
	newOprod.restore();
      }

      virtual void initTuneParam(TuneParam &param) const
      {
	Tunable::initTuneParam(param);
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
      }
      
      /** sets default values for when tuning is disabled */
      void defaultTuneParam(TuneParam &param) const
      {
	Tunable::defaultTuneParam(param);
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
      }

      long long flops() const { return 0; }
    };


    template<class RealA, class RealB>
    class SideLinkShort : public Tunable {

    private:
      const cudaGaugeField &link;
      const cudaGaugeField &P3; 
      const int sig;
      const int mu;
      typename RealTypeId<RealA>::Type &coeff; 
      cudaGaugeField &newOprod;
      const hisq_kernel_param_t &kparam;

      int sharedBytesPerThread() const { return 0; }
      int sharedBytesPerBlock(const TuneParam &) const { return 0; }

      // don't tune the grid dimension
      bool advanceGridDim(TuneParam &param) const { return false; }
      bool advanceBlockDim(TuneParam &param) const {
	bool rtn = Tunable::advanceBlockDim(param);
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
	return rtn;
      }

    public:
      SideLinkShort(const cudaGaugeField &link, const cudaGaugeField &P3, int sig, int mu, 
		    typename RealTypeId<RealA>::Type coeff, cudaGaugeField &newOprod,
		    const hisq_kernel_param_t &kparam) :
	link(link), P3(P3), sig(sig), mu(mu), coeff(coeff), newOprod(newOprod), kparam(kparam)
      {	; }
      virtual ~SideLinkShort() { ; }

      TuneKey tuneKey() const {
	std::stringstream vol, aux;
	vol << kparam.D1 << "x";
	vol << kparam.D2 << "x";
	vol << kparam.D3 << "x";
	vol << kparam.D4;    
	aux << "threads=" << kparam.threads << ",prec=" << link.Precision();
	aux << ",recon=" << link.Reconstruct() << ",sig=" << sig << ",mu=" << mu;
	return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }  
      
#define CALL_ARGUMENTS(typeA, typeB) <<<tp.grid, tp.block>>>		\
      ((typeA*)P3.Even_p(), (typeA*)P3.Odd_p(),				\
       (typeB*)link.Even_p(), (typeB*)link.Odd_p(),			\
       sig, mu,	(typename RealTypeId<typeA>::Type) coeff,		\
       (typeA*)newOprod.Even_p(), (typeA*)newOprod.Odd_p(), kparam)
									      

#define CALL_SIDE_LINK_KERNEL(sig_sign, mu_sign)			\
      if(oddness_change == 0){						\
	if(sizeof(RealA) == sizeof(float2)){				\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_side_link_short_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(float2, float2); \
	    do_side_link_short_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(float2, float2); \
	  }else{							\
	    do_side_link_short_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(float2, float4); \
	    do_side_link_short_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(float2, float4); \
	  }								\
	}else{								\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_side_link_short_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(double2, double2); \
	    do_side_link_short_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(double2, double2); \
	  }else{							\
	    do_side_link_short_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(double2, double2); \
	    do_side_link_short_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(double2, double2); \
	  }								\
	}								\
      }else{								\
	if(sizeof(RealA) == sizeof(float2)){				\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_side_link_short_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(float2, float2); \
	    do_side_link_short_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(float2, float2); \
	  }else{							\
	    do_side_link_short_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(float2, float4); \
	    do_side_link_short_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(float2, float4); \
	  }								\
	}else{								\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_side_link_short_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(double2, double2); \
	    do_side_link_short_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(double2, double2); \
	  }else{							\
	    do_side_link_short_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(double2, double2); \
	    do_side_link_short_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(double2, double2); \
	  }								\
	}								\
      }
      
      void apply(const cudaStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
	QudaReconstructType recon = link.Reconstruct();
	int oddness_change = (kparam.base_idx[0] + kparam.base_idx[1]
			      + kparam.base_idx[2] + kparam.base_idx[3])&1;
	
	if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
	  CALL_SIDE_LINK_KERNEL(1,1);
	}else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	  CALL_SIDE_LINK_KERNEL(1,0);
	  
	}else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	  CALL_SIDE_LINK_KERNEL(0,1);
	}else{
	  CALL_SIDE_LINK_KERNEL(0,0);
	}	
      }
      
#undef CALL_SIDE_LINK_KERNEL
#undef CALL_ARGUMENTS      
      
      
      void preTune() {
	newOprod.backup();
      }

      void postTune() {
	newOprod.restore();
      }

      virtual void initTuneParam(TuneParam &param) const
      {
	Tunable::initTuneParam(param);
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
      }
      
      /** sets default values for when tuning is disabled */
      void defaultTuneParam(TuneParam &param) const
      {
	Tunable::defaultTuneParam(param);
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
      }

      long long flops() const { return 0; }
    };

    template<class RealA, class RealB>
    class AllLink : public Tunable {

    private:
      const cudaGaugeField &link;
      const cudaGaugeField &oprod;
      const cudaGaugeField &Qprev;
      const int sig;
      const int mu;
      typename RealTypeId<RealA>::Type &coeff; 
      typename RealTypeId<RealA>::Type &accumu_coeff;
      cudaGaugeField &shortP;
      cudaGaugeField &newOprod;
      const hisq_kernel_param_t &kparam;

      int sharedBytesPerThread() const { return 0; }
      int sharedBytesPerBlock(const TuneParam &) const { return 0; }

      // don't tune the grid dimension
      bool advanceGridDim(TuneParam &param) const { return false; }
      bool advanceBlockDim(TuneParam &param) const {
	bool rtn = Tunable::advanceBlockDim(param);
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
	return rtn;
      }

    public:
      AllLink(const cudaGaugeField &link, 
	      const cudaGaugeField &oprod, 
	      const cudaGaugeField &Qprev, 
	      int sig, int mu,
	      typename RealTypeId<RealA>::Type coeff, 
	      typename RealTypeId<RealA>::Type  accumu_coeff,
	      cudaGaugeField &shortP, cudaGaugeField &newOprod, 
	      const hisq_kernel_param_t &kparam) : 
	link(link), oprod(oprod), Qprev(Qprev), sig(sig), mu(mu), 
	coeff(coeff), accumu_coeff(accumu_coeff), shortP(shortP), 
	newOprod(newOprod), kparam(kparam)
      { ; }
      virtual ~AllLink() { ; }

      TuneKey tuneKey() const {
	std::stringstream vol, aux;
	vol << kparam.D1 << "x";
	vol << kparam.D2 << "x";
	vol << kparam.D3 << "x";
	vol << kparam.D4;    
	aux << "threads=" << kparam.threads << ",prec=" << link.Precision();
	aux << ",recon=" << link.Reconstruct() << ",sig=" << sig << ",mu=" << mu;
	return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }  
      
#define CALL_ARGUMENTS(typeA, typeB) <<<tp.grid, tp.block>>>		\
      ((typeA*)oprod.Even_p(), (typeA*)oprod.Odd_p(),			\
       (typeA*)Qprev.Even_p(), (typeA*)Qprev.Odd_p(),			\
       (typeB*)link.Even_p(), (typeB*)link.Odd_p(), sig,  mu,		\
       (typename RealTypeId<typeA>::Type)coeff,				\
       (typename RealTypeId<typeA>::Type)accumu_coeff,			\
       (typeA*)shortP.Even_p(),(typeA*)shortP.Odd_p(),			\
       (typeA*)newOprod.Even_p(), (typeA*)newOprod.Odd_p(), kparam)
      
#define CALL_ALL_LINK_KERNEL(sig_sign, mu_sign)				\
      if(oddness_change == 0){						\
	if(sizeof(RealA) == sizeof(float2)){				\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_all_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(float2, float2); \
	    do_all_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(float2, float2); \
	  }else{							\
	    do_all_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(float2, float4); \
	    do_all_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(float2, float4); \
	  }								\
	}else{								\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_all_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(double2, double2); \
	    do_all_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(double2, double2); \
	  }else{							\
	    do_all_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0, 0> CALL_ARGUMENTS(double2, double2); \
	    do_all_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1, 0> CALL_ARGUMENTS(double2, double2); \
	  }								\
	}								\
      }else{								\
	if(sizeof(RealA) == sizeof(float2)){				\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_all_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(float2, float2); \
	    do_all_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(float2, float2); \
	  }else{							\
	    do_all_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(float2, float4); \
	    do_all_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(float2, float4); \
	  }								\
	}else{								\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_all_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(double2, double2); \
	    do_all_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(double2, double2); \
	  }else{							\
	    do_all_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0, 1> CALL_ARGUMENTS(double2, double2); \
	    do_all_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1, 1> CALL_ARGUMENTS(double2, double2); \
	  }								\
	}								\
      }     
      void apply(const cudaStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
	QudaReconstructType recon = link.Reconstruct();
	int oddness_change = (kparam.base_idx[0] + kparam.base_idx[1]
			      + kparam.base_idx[2] + kparam.base_idx[3])&1;

	if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
	  CALL_ALL_LINK_KERNEL(1, 1);
	}else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	  CALL_ALL_LINK_KERNEL(1, 0);
	}else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	  CALL_ALL_LINK_KERNEL(0, 1);
	}else{
	  CALL_ALL_LINK_KERNEL(0, 0);
	}
	
	return;
      }
      
#undef CALL_ARGUMENTS
#undef CALL_ALL_LINK_KERNEL	    
      
      void preTune() {
	shortP.backup();
	newOprod.backup();
      }

      void postTune() {
	shortP.restore();
	newOprod.restore();
      }

      virtual void initTuneParam(TuneParam &param) const
      {
	Tunable::initTuneParam(param);
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
      }
      
      /** sets default values for when tuning is disabled */
      void defaultTuneParam(TuneParam &param) const
      {
	Tunable::defaultTuneParam(param);
	param.grid = dim3((kparam.threads+param.block.x-1)/param.block.x, 1, 1);
      }

      long long flops() const { return 0; }
    };


    template<class RealA, class RealB>
      class OneLinkTerm : public Tunable {

    private:
      const cudaGaugeField &oprod;
      const int sig;
      typename RealTypeId<RealA>::Type &coeff; 
      typename RealTypeId<RealA>::Type &naik_coeff;
      cudaGaugeField &ForceMatrix;
      const int* X;
      
      int sharedBytesPerThread() const { return 0; }
      int sharedBytesPerBlock(const TuneParam &) const { return 0; }

      // don't tune the grid dimension
      bool advanceGridDim(TuneParam &param) const { return false; }
      bool advanceBlockDim(TuneParam &param) const {
	bool rtn = Tunable::advanceBlockDim(param);
	int threads = X[0]*X[1]*X[2]*X[3]/2;

	param.grid = dim3((threads + param.block.x-1)/param.block.x, 1, 1);
	return rtn;
      }

    public:
      OneLinkTerm(const cudaGaugeField &oprod, int sig, 
		  typename RealTypeId<RealA>::Type coeff, 
		  typename RealTypeId<RealA>::Type naik_coeff,
		  cudaGaugeField &ForceMatrix, const int* _X) :
      oprod(oprod), sig(sig), coeff(coeff), naik_coeff(naik_coeff), ForceMatrix(ForceMatrix),
	X(_X)
      { ; }

      virtual ~OneLinkTerm() { ; }

      TuneKey tuneKey() const {
	std::stringstream vol, aux;
	vol << X[0] << "x";
	vol << X[1] << "x";
	vol << X[2] << "x";
	vol << X[3];    
	int threads = X[0]*X[1]*X[2]*X[3]/2;
	aux << "threads=" << threads << ",prec=" << oprod.Precision();
	aux << ",sig=" << sig << ",coeff=" << coeff;
	return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }  

      void apply(const cudaStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);

	int threads = X[0]*X[1]*X[2]*X[3]/2;
	
        if(GOES_FORWARDS(sig)){
          do_one_link_term_kernel<RealA,0><<<tp.grid,tp.block>>>(static_cast<const RealA*>(oprod.Even_p()), 
								 static_cast<const RealA*>(oprod.Odd_p()), 
								 sig, coeff,
								 static_cast<RealA*>(ForceMatrix.Even_p()), 
								 static_cast<RealA*>(ForceMatrix.Odd_p()),
								 threads);
          do_one_link_term_kernel<RealA,1><<<tp.grid,tp.block>>>(static_cast<const RealA*>(oprod.Even_p()), 
								 static_cast<const RealA*>(oprod.Odd_p()), 
								 sig, coeff,
								 static_cast<RealA*>(ForceMatrix.Even_p()), 
								 static_cast<RealA*>(ForceMatrix.Odd_p()),
								 threads);

	}
      }

      void preTune() {
	ForceMatrix.backup();
      }

      void postTune() {
	ForceMatrix.restore();
      }

      virtual void initTuneParam(TuneParam &param) const
      {
	Tunable::initTuneParam(param);

	int threads = X[0]*X[1]*X[2]*X[3]/2;
	param.grid = dim3((threads+param.block.x-1)/param.block.x, 1, 1);
      }
      
      /** sets default values for when tuning is disabled */
      void defaultTuneParam(TuneParam &param) const
      {
	Tunable::defaultTuneParam(param);
	int threads = X[0]*X[1]*X[2]*X[3]/2;
	param.grid = dim3((threads+param.block.x-1)/param.block.x, 1, 1);
      }

      long long flops() const { return 0; }
    };
    

    template<class RealA, class RealB>
    class LongLinkTerm : public Tunable {

    private:
      const cudaGaugeField &link;
      const cudaGaugeField &naikOprod;
      const int sig;
      typename RealTypeId<RealA>::Type naik_coeff;
      cudaGaugeField &output;
      const int * X;
      const hisq_kernel_param_t &kparam;

      int sharedBytesPerThread() const { return 0; }
      int sharedBytesPerBlock(const TuneParam &) const { return 0; }

      // don't tune the grid dimension
      bool advanceGridDim(TuneParam &param) const { return false; }
      bool advanceBlockDim(TuneParam &param) const {
	bool rtn = Tunable::advanceBlockDim(param);
	int threads = X[0]*X[1]*X[2]*X[3]/2;
	param.grid = dim3((threads + param.block.x-1)/param.block.x, 1, 1);
	return rtn;
      }

    public:
    LongLinkTerm(const cudaGaugeField &link, const cudaGaugeField &naikOprod,
		   int sig, typename RealTypeId<RealA>::Type naik_coeff,
		 cudaGaugeField &output, const int* _X, const hisq_kernel_param_t &kparam) :
      link(link), naikOprod(naikOprod), sig(sig), naik_coeff(naik_coeff), output(output),
	X(_X), kparam(kparam)
      { ; }

      virtual ~LongLinkTerm() { ; }

      TuneKey tuneKey() const {
	std::stringstream vol, aux;
	vol << X[0] << "x";
	vol << X[1] << "x";
	vol << X[2] << "x";
	vol << X[3];    
	int threads = X[0]*X[1]*X[2]*X[3]/2;
	aux << "threads=" << threads << ",prec=" << link.Precision();
	aux << ",sig=" << sig;
	return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }  

#define CALL_ARGUMENTS(typeA, typeB) <<<tp.grid,tp.block>>>		\
	((typeB*)link.Even_p(), (typeB*)link.Odd_p(),			\
       (typeA*)naikOprod.Even_p(),  (typeA*)naikOprod.Odd_p(),		\
       sig, naik_coeff,							\
       (typeA*)output.Even_p(), (typeA*)output.Odd_p(),			\
       kparam);		
      
      void apply(const cudaStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
	QudaReconstructType recon = link.Reconstruct();
	
        if(GOES_BACKWARDS(sig)) errorQuda("sig does not go forward\n");
	if(sizeof(RealA) == sizeof(float2)){
	  if(recon == QUDA_RECONSTRUCT_NO){
	    do_longlink_sp_18_kernel<float2,float2, 0> CALL_ARGUMENTS(float2, float2);
	    do_longlink_sp_18_kernel<float2,float2, 1> CALL_ARGUMENTS(float2, float2);
	  }else{
	    do_longlink_sp_12_kernel<float2,float4, 0> CALL_ARGUMENTS(float2, float4);
	    do_longlink_sp_12_kernel<float2,float4, 1> CALL_ARGUMENTS(float2, float4);
	  }
	}else{
	  if(recon == QUDA_RECONSTRUCT_NO){
	    do_longlink_dp_18_kernel<double2,double2, 0> CALL_ARGUMENTS(double2, double2);
	    do_longlink_dp_18_kernel<double2,double2, 1> CALL_ARGUMENTS(double2, double2);
	  }else{
	    do_longlink_dp_12_kernel<double2,double2, 0> CALL_ARGUMENTS(double2, double2);
	    do_longlink_dp_12_kernel<double2,double2, 1> CALL_ARGUMENTS(double2, double2);	    
	  }
	}
      }

#undef CALL_ARGUMENTS	

      void preTune() {
	output.backup();
      }

      void postTune() {
	output.restore();
      }

      virtual void initTuneParam(TuneParam &param) const
      {
	Tunable::initTuneParam(param);
	int threads = X[0]*X[1]*X[2]*X[3]/2;
	param.grid = dim3((threads+param.block.x-1)/param.block.x, 1, 1);
      }
      
      /** sets default values for when tuning is disabled */
      void defaultTuneParam(TuneParam &param) const
      {
	Tunable::defaultTuneParam(param);
	int threads = X[0]*X[1]*X[2]*X[3]/2;
	param.grid = dim3((threads+param.block.x-1)/param.block.x, 1, 1);
      }

      long long flops() const { return 0; }
    };




    template<class RealA, class RealB>
    class CompleteForce : public Tunable {

    private:
      const cudaGaugeField &link;
      const cudaGaugeField &oprod;
      const int sig;
      cudaGaugeField &mom;
      const int* X;
      
      int sharedBytesPerThread() const { return 0; }
      int sharedBytesPerBlock(const TuneParam &) const { return 0; }

      // don't tune the grid dimension
      bool advanceGridDim(TuneParam &param) const { return false; }
      bool advanceBlockDim(TuneParam &param) const {
	bool rtn = Tunable::advanceBlockDim(param);
	int threads = X[0]*X[1]*X[2]*X[3]/2;
	param.grid = dim3((threads + param.block.x-1)/param.block.x, 1, 1);
	return rtn;
      }

    public:
    CompleteForce(const cudaGaugeField &link, const cudaGaugeField &oprod, 
		    int sig, cudaGaugeField &mom, const int* _X) :
      link(link), oprod(oprod), sig(sig), mom(mom), X(_X)
      { ; }

      virtual ~CompleteForce() { ; }

      TuneKey tuneKey() const {
	std::stringstream vol, aux;
	vol << X[0] << "x";
	vol << X[1] << "x";
	vol << X[2] << "x";
	vol << X[3];    
	int threads = X[0]*X[1]*X[2]*X[3]/2;
	aux << "threads=" << threads << ",prec=" << link.Precision() << ",sig=" << sig;
	return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }  

#define CALL_ARGUMENTS(typeA, typeB)  <<<tp.grid, tp.block>>>		\
      ((typeB*)link.Even_p(), (typeB*)link.Odd_p(),			\
       (typeA*)oprod.Even_p(), (typeA*)oprod.Odd_p(),			\
       sig,								\
       (typeA*)mom.Even_p(), (typeA*)mom.Odd_p(),			\
       X[0] * X[1] * X[2] * X[3]/2);		
      
      void apply(const cudaStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
	QudaReconstructType recon = link.Reconstruct();;
      
	if(sizeof(RealA) == sizeof(float2)){
	  if(recon == QUDA_RECONSTRUCT_NO){
	    do_complete_force_sp_18_kernel<float2,float2, 0> CALL_ARGUMENTS(float2, float2);
	    do_complete_force_sp_18_kernel<float2,float2, 1> CALL_ARGUMENTS(float2, float2);
	  }else{
	    do_complete_force_sp_12_kernel<float2,float4, 0> CALL_ARGUMENTS(float2, float4);
	    do_complete_force_sp_12_kernel<float2,float4, 1> CALL_ARGUMENTS(float2, float4);
	  }
	}else{
	  if(recon == QUDA_RECONSTRUCT_NO){
	    do_complete_force_dp_18_kernel<double2,double2, 0> CALL_ARGUMENTS(double2, double2);
	    do_complete_force_dp_18_kernel<double2,double2, 1> CALL_ARGUMENTS(double2, double2);
	  }else{
	    do_complete_force_dp_12_kernel<double2,double2, 0> CALL_ARGUMENTS(double2, double2);
	    do_complete_force_dp_12_kernel<double2,double2, 1> CALL_ARGUMENTS(double2, double2);	    
	  }
	}
      }

#undef CALL_ARGUMENTS	

      void preTune() {
	mom.backup();
      }

      void postTune() {
	mom.restore();
      }

      virtual void initTuneParam(TuneParam &param) const
      {
	Tunable::initTuneParam(param);
	int threads = X[0]*X[1]*X[2]*X[3]/2;
	param.grid = dim3((threads+param.block.x-1)/param.block.x, 1, 1);
      }
      
      /** sets default values for when tuning is disabled */
      void defaultTuneParam(TuneParam &param) const
      {
	Tunable::defaultTuneParam(param);
	int threads = X[0]*X[1]*X[2]*X[3]/2;
	param.grid = dim3((threads+param.block.x-1)/param.block.x, 1, 1);
      }
      
      long long flops() const { return 0; }
    };


    static void 
    bind_tex_link(const cudaGaugeField& link, const cudaGaugeField& newOprod)
    {
      if(link.Precision() == QUDA_DOUBLE_PRECISION){
	cudaBindTexture(0, siteLink0TexDouble, link.Even_p(), link.Bytes()/2);
	cudaBindTexture(0, siteLink1TexDouble, link.Odd_p(), link.Bytes()/2);
    
	cudaBindTexture(0, newOprod0TexDouble, newOprod.Even_p(), newOprod.Bytes()/2);
	cudaBindTexture(0, newOprod1TexDouble, newOprod.Odd_p(), newOprod.Bytes()/2);
      }else{
	if(link.Reconstruct() == QUDA_RECONSTRUCT_NO){
	  cudaBindTexture(0, siteLink0TexSingle, link.Even_p(), link.Bytes()/2);      
	  cudaBindTexture(0, siteLink1TexSingle, link.Odd_p(), link.Bytes()/2);      
	}else{
	  cudaBindTexture(0, siteLink0TexSingle_recon, link.Even_p(), link.Bytes()/2);      
	  cudaBindTexture(0, siteLink1TexSingle_recon, link.Odd_p(), link.Bytes()/2);            
	}
	cudaBindTexture(0, newOprod0TexSingle, newOprod.Even_p(), newOprod.Bytes()/2);
	cudaBindTexture(0, newOprod1TexSingle, newOprod.Odd_p(), newOprod.Bytes()/2);
    
      }
    }

    static void 
    unbind_tex_link(const cudaGaugeField& link, const cudaGaugeField& newOprod)
    {
      if(link.Precision() == QUDA_DOUBLE_PRECISION){
	cudaUnbindTexture(siteLink0TexDouble);
	cudaUnbindTexture(siteLink1TexDouble);
	cudaUnbindTexture(newOprod0TexDouble);
	cudaUnbindTexture(newOprod1TexDouble);
      }else{
	if(link.Reconstruct() == QUDA_RECONSTRUCT_NO){
	  cudaUnbindTexture(siteLink0TexSingle);
	  cudaUnbindTexture(siteLink1TexSingle);      
	}else{
	  cudaUnbindTexture(siteLink0TexSingle_recon);
	  cudaUnbindTexture(siteLink1TexSingle_recon);      
	}
	cudaUnbindTexture(newOprod0TexSingle);
	cudaUnbindTexture(newOprod1TexSingle);
      }
    }

    template<class Real, class RealA, class RealB>
    static void
    do_hisq_staples_force_cuda( PathCoefficients<Real> act_path_coeff,
				const QudaGaugeParam& param,
				const cudaGaugeField &oprod, 
				const cudaGaugeField &link,
				cudaGaugeField &Pmu,
				cudaGaugeField &P3,
				cudaGaugeField &P5,
				cudaGaugeField &Pnumu,
				cudaGaugeField &Qmu,
				cudaGaugeField &Qnumu,
				cudaGaugeField &newOprod)
    {

      Real coeff;
      Real OneLink, Lepage, FiveSt, ThreeSt, SevenSt;
      Real mLepage, mFiveSt, mThreeSt;

      OneLink = act_path_coeff.one;
      ThreeSt = act_path_coeff.three; mThreeSt = -ThreeSt;
      FiveSt  = act_path_coeff.five; mFiveSt  = -FiveSt;
      SevenSt = act_path_coeff.seven; 
      Lepage  = act_path_coeff.lepage; mLepage  = -Lepage;


      for(int sig=0; sig<8; ++sig){
	if(GOES_FORWARDS(sig)){
	  OneLinkTerm<RealA, RealB> oneLink(oprod, sig, OneLink, 0.0, newOprod, param.X);
	  oneLink.apply(0);
	  checkCudaError();
	} // GOES_FORWARDS(sig)
      }
      

      int ghostDim[4]={
	commDimPartitioned(0),
	commDimPartitioned(1),
	commDimPartitioned(2),
	commDimPartitioned(3)
      };
      
      hisq_kernel_param_t kparam_1g, kparam_2g;
      
      
#ifdef MULTI_GPU
      kparam_1g.D1 = commDimPartitioned(0)?(param.X[0]+2):(param.X[0]);
      kparam_1g.D2 = commDimPartitioned(1)?(param.X[1]+2):(param.X[1]);
      kparam_1g.D3 = commDimPartitioned(2)?(param.X[2]+2):(param.X[2]);
      kparam_1g.D4 = commDimPartitioned(3)?(param.X[3]+2):(param.X[3]);
      kparam_1g.D1h =  kparam_1g.D1/2;
      kparam_1g.base_idx[0]=commDimPartitioned(0)?1:2;
      kparam_1g.base_idx[1]=commDimPartitioned(1)?1:2;
      kparam_1g.base_idx[2]=commDimPartitioned(2)?1:2;
      kparam_1g.base_idx[3]=commDimPartitioned(3)?1:2;
      kparam_1g.threads = kparam_1g.D1*kparam_1g.D2*kparam_1g.D3*kparam_1g.D4/2;
      
      kparam_2g.D1 = commDimPartitioned(0)?(param.X[0]+4):(param.X[0]);
      kparam_2g.D2 = commDimPartitioned(1)?(param.X[1]+4):(param.X[1]);
      kparam_2g.D3 = commDimPartitioned(2)?(param.X[2]+4):(param.X[2]);
      kparam_2g.D4 = commDimPartitioned(3)?(param.X[3]+4):(param.X[3]);
      kparam_2g.D1h = kparam_2g.D1/2;
      kparam_2g.base_idx[0]=commDimPartitioned(0)?0:2;
      kparam_2g.base_idx[1]=commDimPartitioned(1)?0:2;
      kparam_2g.base_idx[2]=commDimPartitioned(2)?0:2;
      kparam_2g.base_idx[3]=commDimPartitioned(3)?0:2;
      kparam_2g.threads = kparam_2g.D1*kparam_2g.D2*kparam_2g.D3*kparam_2g.D4/2;
      
      
      for(int i=0;i < 4; i++){
	kparam_1g.ghostDim[i] = kparam_2g.ghostDim[i]=kparam_1g.ghostDim[i]=kparam_2g.ghostDim[i] = ghostDim[i];
      }
#else
      hisq_kernel_param_t kparam;
      kparam.D1 = param.X[0];
      kparam.D2 = param.X[1];
      kparam.D3 = param.X[2];
      kparam.D4 = param.X[3];
      kparam.D1h = param.X[0]/2;
      kparam.threads=param.X[0]*param.X[1]*param.X[2]*param.X[3]/2;
      kparam.base_idx[0]=0;
      kparam.base_idx[1]=0;
      kparam.base_idx[2]=0;
      kparam.base_idx[3]=0;
      kparam_2g = kparam_1g = kparam;
      
#endif
      for(int sig=0; sig<8; sig++){
	for(int mu=0; mu<8; mu++){
	  if ( (mu == sig) || (mu == OPP_DIR(sig))){
	    continue;
	  }
	  //3-link
	  //Kernel A: middle link

	  MiddleLink<RealA,RealB> middleLink( link, oprod,  // read only
					      sig, mu, mThreeSt,
					      Pmu, P3, Qmu, // write only
					      newOprod, kparam_2g);
	  middleLink.apply(0);
	  checkCudaError();

	  for(int nu=0; nu < 8; nu++){
	    if (nu == sig || nu == OPP_DIR(sig)
		|| nu == mu || nu == OPP_DIR(mu)){
	      continue;
	    }
	    //5-link: middle link
	    //Kernel B
	    MiddleLink<RealA,RealB> middleLink( link, Pmu, Qmu, // read only
						sig, nu, FiveSt,
						Pnumu, P5, Qnumu, // write only
						newOprod, kparam_1g);
	    middleLink.apply(0);
	    checkCudaError();


	    for(int rho = 0; rho < 8; rho++){
	      if (rho == sig || rho == OPP_DIR(sig)
		  || rho == mu || rho == OPP_DIR(mu)
		  || rho == nu || rho == OPP_DIR(nu)){
		continue;
	      }
	      //7-link: middle link and side link
	      if(FiveSt != 0)coeff = SevenSt/FiveSt; else coeff = 0;
	      AllLink<RealA,RealB> allLink(link, Pnumu, Qnumu, sig, rho, SevenSt, coeff,
					   P5, newOprod, kparam_1g);

	      allLink.apply(0);

	      checkCudaError();
	      //return;
	    }//rho  		

	    //5-link: side link
	    if(ThreeSt != 0)coeff = FiveSt/ThreeSt; else coeff = 0;
	    SideLink<RealA,RealB> sideLink(link, P5, Qmu, //read only
					   sig, nu, mFiveSt, coeff,
					   P3, // write only
					   newOprod, kparam_1g);
	    sideLink.apply(0);
	    checkCudaError();

	  } //nu 

            //lepage
	  if(Lepage != 0.){
	    LepageMiddleLink<RealA,RealB> 
	      lepageMiddleLink ( link, Pmu, Qmu, // read only
				 sig, mu, Lepage,
				 P5, // write only
				 newOprod, kparam_2g);
	    lepageMiddleLink.apply(0);
	    checkCudaError();

	    if(ThreeSt != 0)coeff = Lepage/ThreeSt ; else coeff = 0;

	    SideLink<RealA, RealB> sideLink(link, P5, Qmu, // read only
					    sig, mu, mLepage, coeff,
					    P3, //write only
					    newOprod, kparam_2g);
	      
	    sideLink.apply(0);
	    checkCudaError();		
	  } // Lepage != 0.0

            //3-link side link
	  SideLinkShort<RealA,RealB> sideLinkShort(link, P3, // read only
						   sig, mu, ThreeSt,
						   newOprod, kparam_1g);
	  sideLinkShort.apply(0);
	  checkCudaError();			    
	    
	}//mu
      }//sig

      
      return; 
    } // do_hisq_staples_force_cuda


#undef Pmu
#undef Pnumu
#undef P3
#undef P5
#undef Qmu
#undef Qnumu


    void hisqCompleteForceCuda(const QudaGaugeParam &param,
			       const cudaGaugeField &oprod,
			       const cudaGaugeField &link,
			       cudaGaugeField* force)
    {
      bind_tex_link(link, oprod);

      for(int sig=0; sig<4; sig++){
	if(param.cuda_prec == QUDA_DOUBLE_PRECISION){
	  CompleteForce<double2,double2> completeForce(link, oprod, sig, *force, param.X);
	  completeForce.apply(0);
	  checkCudaError();
	}else if(param.cuda_prec == QUDA_SINGLE_PRECISION){
	  CompleteForce<float2,float2> completeForce(link, oprod, sig, *force, param.X);
	  completeForce.apply(0);
	  checkCudaError();
	}else{
	  errorQuda("Unsupported precision");
	}
      } // loop over directions

      unbind_tex_link(link, oprod);
      return;
    }

  
    void hisqLongLinkForceCuda(double coeff,
			       const QudaGaugeParam &param,
			       const cudaGaugeField &oldOprod,
			       const cudaGaugeField &link,
			       cudaGaugeField  *newOprod)
    {
      bind_tex_link(link, *newOprod);
      const int volume = param.X[0]*param.X[1]*param.X[2]*param.X[3];
      hisq_kernel_param_t kparam;
      for(int i =0;i < 4;i++){
	kparam.ghostDim[i] = commDimPartitioned(i);
      }
      kparam.threads = volume/2;
      
      for(int sig=0; sig<4; ++sig){
	if(param.cuda_prec == QUDA_DOUBLE_PRECISION){
	  LongLinkTerm<double2,double2> longLink(link, oldOprod, sig, coeff, *newOprod, param.X, kparam);
	  longLink.apply(0);
	  checkCudaError();
	}else if(param.cuda_prec == QUDA_SINGLE_PRECISION){
	  LongLinkTerm<float2,float2> longLink(link, oldOprod, sig, static_cast<float>(coeff), 
					       *newOprod, param.X, kparam);
	  longLink.apply(0);
	  checkCudaError();
	}else{
	  errorQuda("Unsupported precision");
	}
      } // loop over directions
      unbind_tex_link(link, *newOprod);
      return;
    }





    void
    hisqStaplesForceCuda(const double path_coeff_array[6],
			 const QudaGaugeParam &param,
			 const cudaGaugeField &oprod, 
			 const cudaGaugeField &link, 
			 cudaGaugeField* newOprod)
    {

#ifdef MULTI_GPU
      int X[4] = {
	param.X[0]+4,  param.X[1]+4,  param.X[2]+4,  param.X[3]+4
      };
#else
      int X[4] = {
	param.X[0],  param.X[1],  param.X[2],  param.X[3]
      };
#endif	

      // create color matrix fields with zero padding
      int pad = 0;
      GaugeFieldParam gauge_param(X, param.cuda_prec, QUDA_RECONSTRUCT_NO, pad, QUDA_SCALAR_GEOMETRY);

      cudaGaugeField Pmu(gauge_param);
      cudaGaugeField P3(gauge_param);
      cudaGaugeField P5(gauge_param);
      cudaGaugeField Pnumu(gauge_param);
      cudaGaugeField Qmu(gauge_param);
      cudaGaugeField Qnumu(gauge_param);

      bind_tex_link(link, *newOprod);
	
      cudaEvent_t start, end;
	
      cudaEventCreate(&start);
      cudaEventCreate(&end);
	
      cudaEventRecord(start);
      if (param.cuda_prec == QUDA_DOUBLE_PRECISION){
	  
	PathCoefficients<double> act_path_coeff;
	act_path_coeff.one    = path_coeff_array[0];
	act_path_coeff.naik   = path_coeff_array[1];
	act_path_coeff.three  = path_coeff_array[2];
	act_path_coeff.five   = path_coeff_array[3];
	act_path_coeff.seven  = path_coeff_array[4];
	act_path_coeff.lepage = path_coeff_array[5];
	do_hisq_staples_force_cuda<double,double2,double2>( act_path_coeff,
							    param,
							    oprod,
							    link, 
							    Pmu,
							    P3,
							    P5,
							    Pnumu,
							    Qmu,
							    Qnumu,
							    *newOprod);
							   

      }else if(param.cuda_prec == QUDA_SINGLE_PRECISION){	
	PathCoefficients<float> act_path_coeff;
	act_path_coeff.one    = path_coeff_array[0];
	act_path_coeff.naik   = path_coeff_array[1];
	act_path_coeff.three  = path_coeff_array[2];
	act_path_coeff.five   = path_coeff_array[3];
	act_path_coeff.seven  = path_coeff_array[4];
	act_path_coeff.lepage = path_coeff_array[5];

	do_hisq_staples_force_cuda<float,float2,float2>( act_path_coeff,
							 param,
							 oprod,
							 link, 
							 Pmu,
							 P3,
							 P5,
							 Pnumu,
							 Qmu,
							 Qnumu,
							 *newOprod);
      }else{
	errorQuda("Unsupported precision");
      }
	
	
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      float runtime;
      cudaEventElapsedTime(&runtime, start, end);
	
      unbind_tex_link(link, *newOprod);

      return; 
    }

  } // namespace fermion_force
} // namespace quda
