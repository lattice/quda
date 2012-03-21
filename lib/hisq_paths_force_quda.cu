#include <read_gauge.h>
#include <gauge_field.h>

#include <hisq_force_quda.h>
#include <hw_quda.h>
#include <hisq_force_macros.h>
#include<utility>


//DEBUG : control conpile 
#define COMPILE_HISQ_DP_18 
#define COMPILE_HISQ_DP_12 
#define COMPILE_HISQ_SP_18 
#define COMPILE_HISQ_SP_12

// Disable texture read for now. Need to revisit this.
#define HISQ_SITE_MATRIX_LOAD_TEX 1
#define HISQ_NEW_OPROD_LOAD_TEX 1

namespace hisq {
  namespace fermion_force {



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
        init_kernel_cuda(param);    
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
			       int dir, int idx, T* const mat, int oddness)
    {
      const T* const field = (oddness)?field_odd:field_even;
      for(int i = 0;i < N ;i++){
	  mat[i] = field[idx + dir*N*Vh + i*Vh];          
      }
      return;
    }

    template<class T>
      inline __device__
      void loadMatrixFromField(const T* const field_even, const T* const field_odd,
			       int dir, int idx, T* const mat, int oddness)
      {
	loadMatrixFromField<9> (field_even, field_odd, dir, idx, mat, oddness);
        return;
      }
    
    

    inline __device__
      void loadMatrixFromField(const float4* const field_even, const float4* const field_odd, 
			       int dir, int idx, float2* const mat, int oddness)
    {
      const float4* const field = oddness?field_odd: field_even;
      float4 tmp;
      tmp = field[idx + dir*Vhx3];
      mat[0] = make_float2(tmp.x, tmp.y);
      mat[1] = make_float2(tmp.z, tmp.w);
      tmp = field[idx + dir*Vhx3 + Vh];
      mat[2] = make_float2(tmp.x, tmp.y);
      mat[3] = make_float2(tmp.z, tmp.w);
      tmp = field[idx + dir*Vhx3 + 2*Vh];
      mat[4] = make_float2(tmp.x, tmp.y);
      mat[5] = make_float2(tmp.z, tmp.w);
      return;
    }

    template<class T>
      inline __device__
      void loadMatrixFromField(const T* const field_even, const T* const field_odd, int idx, T* const mat, int oddness)
      {
	const T* const field = (oddness)?field_odd:field_even;
        mat[0] = field[idx];
        mat[1] = field[idx + Vh];
        mat[2] = field[idx + Vhx2];
        mat[3] = field[idx + Vhx3];
        mat[4] = field[idx + Vhx4];
        mat[5] = field[idx + Vhx5];
        mat[6] = field[idx + Vhx6];
        mat[7] = field[idx + Vhx7];
        mat[8] = field[idx + Vhx8];

        return;
      }
    

#define  addMatrixToNewOprod(mat,  dir, idx, coeff, field_even, field_odd, oddness)     do { \
      RealA* const field = (oddness)?field_odd: field_even;		\
      RealA value[9];							\
      value[0] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*Vhx9); \
      value[1] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*Vhx9 + Vh);	\
      value[2] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*Vhx9 + 2*Vh); \
      value[3] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*Vhx9 + 3*Vh); \
      value[4] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*Vhx9 + 4*Vh); \
      value[5] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*Vhx9 + 5*Vh); \
      value[6] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*Vhx9 + 6*Vh); \
      value[7] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*Vhx9 + 7*Vh); \
      value[8] = LOAD_TEX_ENTRY( ((oddness)?NEWOPROD_ODD_TEX:NEWOPROD_EVEN_TEX), field, idx+dir*Vhx9 + 8*Vh); \
      field[idx + dir*Vhx9]          = value[0] + coeff*mat[0];		\
      field[idx + dir*Vhx9 + Vh]     = value[1] + coeff*mat[1];		\
      field[idx + dir*Vhx9 + Vhx2]   = value[2] + coeff*mat[2];		\
      field[idx + dir*Vhx9 + Vhx3]   = value[3] + coeff*mat[3];		\
      field[idx + dir*Vhx9 + Vhx4]   = value[4] + coeff*mat[4];		\
      field[idx + dir*Vhx9 + Vhx5]   = value[5] + coeff*mat[5];		\
      field[idx + dir*Vhx9 + Vhx6]   = value[6] + coeff*mat[6];		\
      field[idx + dir*Vhx9 + Vhx7]   = value[7] + coeff*mat[7];		\
      field[idx + dir*Vhx9 + Vhx8]   = value[8] + coeff*mat[8];		\
  }while(0)					
     


    // only works if Promote<T,U>::Type = T

    template<class T, class U>   
    inline __device__
      void addMatrixToField(const T* const mat, int dir, int idx, U coeff, 
			     T* const field_even, T* const field_odd, int oddness)
      {
	T* const field = (oddness)?field_odd: field_even;
        field[idx + dir*Vhx9]          += coeff*mat[0];
        field[idx + dir*Vhx9 + Vh]     += coeff*mat[1];
        field[idx + dir*Vhx9 + Vhx2]   += coeff*mat[2];
        field[idx + dir*Vhx9 + Vhx3]   += coeff*mat[3];
        field[idx + dir*Vhx9 + Vhx4]   += coeff*mat[4];
        field[idx + dir*Vhx9 + Vhx5]   += coeff*mat[5];
        field[idx + dir*Vhx9 + Vhx6]   += coeff*mat[6];
        field[idx + dir*Vhx9 + Vhx7]   += coeff*mat[7];
        field[idx + dir*Vhx9 + Vhx8]   += coeff*mat[8];

        return;
      }


    template<class T, class U>
    inline __device__
      void addMatrixToField(const T* const mat, int idx, U coeff, T* const field_even,
			     T* const field_odd, int oddness)
      {
	T* const field = (oddness)?field_odd: field_even;
        field[idx ]         += coeff*mat[0];
        field[idx + Vh]     += coeff*mat[1];
        field[idx + Vhx2]   += coeff*mat[2];
        field[idx + Vhx3]   += coeff*mat[3];
        field[idx + Vhx4]   += coeff*mat[4];
        field[idx + Vhx5]   += coeff*mat[5];
        field[idx + Vhx6]   += coeff*mat[6];
        field[idx + Vhx7]   += coeff*mat[7];
        field[idx + Vhx8]   += coeff*mat[8];

        return;
      }


   template<class T>
    inline __device__
     void storeMatrixToField(const T* const mat, int dir, int idx, T* const field_even, T* const field_odd, int oddness)
      {
	T* const field = (oddness)?field_odd: field_even;
        field[idx + dir*Vhx9]          = mat[0];
        field[idx + dir*Vhx9 + Vh]     = mat[1];
        field[idx + dir*Vhx9 + Vhx2]   = mat[2];
        field[idx + dir*Vhx9 + Vhx3]   = mat[3];
        field[idx + dir*Vhx9 + Vhx4]   = mat[4];
        field[idx + dir*Vhx9 + Vhx5]   = mat[5];
        field[idx + dir*Vhx9 + Vhx6]   = mat[6];
        field[idx + dir*Vhx9 + Vhx7]   = mat[7];
        field[idx + dir*Vhx9 + Vhx8]   = mat[8];

        return;
      }


    template<class T>
    inline __device__
      void storeMatrixToField(const T* const mat, int idx, T* const field_even, T* const field_odd, int oddness)
      {
	T* const field = (oddness)?field_odd: field_even;
        field[idx]          = mat[0];
        field[idx + Vh]     = mat[1];
        field[idx + Vhx2]   = mat[2];
        field[idx + Vhx3]   = mat[3];
        field[idx + Vhx4]   = mat[4];
        field[idx + Vhx5]   = mat[5];
        field[idx + Vhx6]   = mat[6];
        field[idx + Vhx7]   = mat[7];
        field[idx + Vhx8]   = mat[8];

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
	  mom_field[idx + dir*Vhx5] = temp2;	

	  temp2.x = (mat[2].x - mat[6].x)*0.5*coeff;
	  temp2.y = (mat[2].y + mat[6].y)*0.5*coeff;
	  mom_field[idx + dir*Vhx5 + Vh] = temp2;

	  temp2.x = (mat[5].x - mat[7].x)*0.5*coeff;
	  temp2.y = (mat[5].y + mat[7].y)*0.5*coeff;
	  mom_field[idx + dir*Vhx5 + Vhx2] = temp2;

	  const typename RealTypeId<T>::Type temp = (mat[0].y + mat[4].y + mat[8].y)*0.3333333333333333333333333;
	  temp2.x =  (mat[0].y-temp)*coeff; 
	  temp2.y =  (mat[4].y-temp)*coeff;
	  mom_field[idx + dir*Vhx5 + Vhx3] = temp2;
		  
	  temp2.x = (mat[8].y - temp)*coeff;
	  temp2.y = 0.0;
	  mom_field[idx + dir*Vhx5 + Vhx4] = temp2;
 
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
	if(i[3] == X4m1) *sign=-1; 
	break;
	
      default:
	printf("Error: invalid dir\n");
	break;
      }
      
      return;
    }






template<class RealA, int oddBit>
  __global__ void 
  do_one_link_term_kernel(const RealA* const oprodEven, const RealA* const oprodOdd,
			  int sig, typename RealTypeId<RealA>::Type coeff,
			  RealA* const outputEven, RealA* const outputOdd)
{
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  
  RealA COLOR_MAT_W[ArrayLength<RealA>::result];
  if(GOES_FORWARDS(sig)){
    loadMatrixFromField(oprodEven, oprodOdd, sig, sid, COLOR_MAT_W, oddBit);
    addMatrixToField(COLOR_MAT_W, sig, sid, coeff, outputEven, outputOdd, oddBit);
  }
  return;
}


#define DD_CONCAT(n,r) n ## r ## kernel

#define HISQ_KERNEL_NAME(a,b) DD_CONCAT(a,b)
//precision: 0 is for double, 1 is for single

#define NEWOPROD_EVEN_TEX newOprod0TexDouble
#define NEWOPROD_ODD_TEX newOprod1TexDouble
#ifdef HISQ_NEW_OPROD_LOAD_TEX
#define LOAD_TEX_ENTRY(tex, field, idx)  READ_DOUBLE2_TEXTURE(tex, field, idx)
#else
#define LOAD_TEX_ENTRY(tex, field, idx) field[idx]
#endif

//double precision, recon=18
#define PRECISION 0
#define RECON 18
#if (HISQ_SITE_MATRIX_LOAD_TEX == 1)
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   HISQ_LOAD_MATRIX_18_DOUBLE_TEX((oddness)?siteLink1TexDouble:siteLink0TexDouble,  (oddness)?linkOdd:linkEven, dir, idx, var, Vh)        
#else
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   loadMatrixFromField(linkEven, linkOdd, dir, idx, var, oddness)  
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
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   HISQ_LOAD_MATRIX_12_DOUBLE_TEX((oddness)?siteLink1TexDouble:siteLink0TexDouble,  (oddness)?linkOdd:linkEven,dir, idx, var, Vh)        
#else
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   loadMatrixFromField<6>(linkEven, linkOdd, dir, idx, var, oddness)  
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

#ifdef HISQ_NEW_OPROD_LOAD_TEX
#define LOAD_TEX_ENTRY(tex, field, idx)  tex1Dfetch(tex,idx)
#else
#define LOAD_TEX_ENTRY(tex, field, idx) field[idx]
#endif

//single precision, recon=18  
#define PRECISION 1
#define RECON 18
#if (HISQ_SITE_MATRIX_LOAD_TEX == 1)
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   HISQ_LOAD_MATRIX_18_SINGLE_TEX((oddness)?siteLink1TexSingle:siteLink0TexSingle, dir, idx, var, Vh)        
#else
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   loadMatrixFromField(linkEven, linkOdd, dir, idx, var, oddness)  
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
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   HISQ_LOAD_MATRIX_12_SINGLE_TEX((oddness)?siteLink1TexSingle_recon:siteLink0TexSingle_recon, dir, idx, var, Vh)        
#else
#define HISQ_LOAD_LINK(linkEven, linkOdd, dir, idx, var, oddness)   loadMatrixFromField(linkEven, linkOdd, dir, idx, var, oddness)  
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
      static void
      middle_link_kernel(
          const RealA* const oprodEven, const RealA* const oprodOdd, 
          const RealA* const QprevEven, const RealA* const QprevOdd,
          const RealB* const linkEven,  const RealB* const linkOdd, 
          const cudaGaugeField &link, int sig, int mu, 
          typename RealTypeId<RealA>::Type coeff,
          dim3 gridDim, dim3 BlockDim,
          RealA* const PmuEven,  RealA* const PmuOdd, // write only
          RealA* const P3Even,   RealA* const P3Odd,  // write only
          RealA* const QmuEven,  RealA* const QmuOdd,   // write only
          RealA* const newOprodEven,  RealA* const newOprodOdd)
      {
	QudaReconstructType recon = link.Reconstruct();
        dim3 halfGridDim(gridDim.x/2, 1,1);
	
#define CALL_ARGUMENTS(typeA, typeB) <<<halfGridDim, BlockDim>>>((typeA*)oprodEven, (typeA*)oprodOdd, \
								 (typeA*)QprevEven, (typeA*)QprevOdd, \
								 (typeB*)linkEven, (typeB*)linkOdd, \
								 sig, mu, (typename RealTypeId<typeA>::Type)coeff, \
								 (typeA*)PmuEven, (typeA*)PmuOdd, \
								 (typeA*)P3Even, (typeA*)P3Odd,	\
								 (typeA*)QmuEven, (typeA*)QmuOdd, \
								 (typeA*)newOprodEven, (typeA*)newOprodOdd)
	
#define CALL_MIDDLE_LINK_KERNEL(sig_sign, mu_sign)			\
	if(sizeof(RealA) == sizeof(float2)){				\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_middle_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0> CALL_ARGUMENTS(float2, float2); \
	    do_middle_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1> CALL_ARGUMENTS(float2, float2); \
	  }else{							\
	    do_middle_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0> CALL_ARGUMENTS(float2, float4); \
	    do_middle_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1> CALL_ARGUMENTS(float2, float4); \
	  }								\
	}else{								\
	  if(recon  == QUDA_RECONSTRUCT_NO){				\
	    do_middle_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0> CALL_ARGUMENTS(double2, double2); \
	    do_middle_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1> CALL_ARGUMENTS(double2, double2); \
	  }else{							\
	    do_middle_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0> CALL_ARGUMENTS(double2, double2); \
	    do_middle_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1> CALL_ARGUMENTS(double2, double2); \
	  }								\
	}
	
        if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){	
	  CALL_MIDDLE_LINK_KERNEL(1,1);
        }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	  CALL_MIDDLE_LINK_KERNEL(1,0);
        }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	  CALL_MIDDLE_LINK_KERNEL(0,1);
        }else{
	  CALL_MIDDLE_LINK_KERNEL(0,0);
        }
	
#undef CALL_ARGUMENTS	
#undef CALL_MIDDLE_LINK_KERNEL
        return;
      }




    template<class RealA, class RealB>
      static void
      side_link_kernel(
          const RealA* const P3Even, const RealA* const P3Odd, 
          const RealA* const oprodEven, const RealA* const oprodOdd,
          const RealB* const linkEven,  const RealB* const linkOdd, 
          const cudaGaugeField &link, int sig, int mu, 
          typename RealTypeId<RealA>::Type coeff, 
          typename RealTypeId<RealA>::Type accumu_coeff,
          dim3 gridDim, dim3 blockDim,
          RealA* shortPEven,  RealA* shortPOdd,
          RealA* newOprodEven, RealA* newOprodOdd)
    {
      QudaReconstructType recon =link.Reconstruct();
      
      dim3 halfGridDim(gridDim.x/2,1,1);

#define CALL_ARGUMENTS(typeA, typeB) 	<<<halfGridDim, blockDim>>>((typeA*)P3Even, (typeA*)P3Odd, \
								    (typeA*)oprodEven,  (typeA*)oprodOdd, \
								    (typeB*)linkEven, (typeB*)linkOdd, \
								    sig, mu, \
								    (typename RealTypeId<typeA>::Type) coeff, \
								    (typename RealTypeId<typeA>::Type) accumu_coeff, \
								    (typeA*)shortPEven, (typeA*)shortPOdd, \
								    (typeA*)newOprodEven, (typeA*)newOprodOdd)
      
#define CALL_SIDE_LINK_KERNEL(sig_sign, mu_sign)			\
      if(sizeof(RealA) == sizeof(float2)){				\
	if(recon  == QUDA_RECONSTRUCT_NO){				\
	  do_side_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0> CALL_ARGUMENTS(float2, float2); \
	  do_side_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1> CALL_ARGUMENTS(float2, float2); \
	}else{								\
	  do_side_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0> CALL_ARGUMENTS(float2, float4); \
	  do_side_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1> CALL_ARGUMENTS(float2, float4); \
	}								\
      }else{								\
	if(recon  == QUDA_RECONSTRUCT_NO){				\
	  do_side_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0> CALL_ARGUMENTS(double2, double2); \
	  do_side_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1> CALL_ARGUMENTS(double2, double2); \
	}else{								\
	  do_side_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0> CALL_ARGUMENTS(double2, double2); \
	  do_side_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1> CALL_ARGUMENTS(double2, double2); \
	}								\
      }
      
      if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
	CALL_SIDE_LINK_KERNEL(1,1);
      }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	CALL_SIDE_LINK_KERNEL(1,0);
	
      }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	CALL_SIDE_LINK_KERNEL(0,1);
      }else{
	CALL_SIDE_LINK_KERNEL(0,0);
      }
      
#undef CALL_SIDE_LINK_KERNEL
#undef CALL_ARGUMENTS      
      return;
    }

   

    template<class RealA, class RealB>
      static void
      all_link_kernel(
          const RealA* const oprodEven, const RealA* const oprodOdd,
          const RealA* const QprevEven, const RealA* const QprevOdd, 
          const RealB* const linkEven,  const RealB* const linkOdd, 
          const cudaGaugeField &link, int sig, int mu,
          typename RealTypeId<RealA>::Type coeff, 
          typename RealTypeId<RealA>::Type  accumu_coeff,
          dim3 gridDim, dim3 blockDim,
          RealA* const shortPEven, RealA* const shortPOdd,
          RealA* const newOprodEven, RealA* const newOprodOdd)
    {
      QudaReconstructType recon = link.Reconstruct();
      dim3 halfGridDim(gridDim.x/2, 1,1);
      
#define CALL_ARGUMENTS(typeA, typeB) <<<halfGridDim, blockDim>>>((typeA*)oprodEven, (typeA*)oprodOdd, \
								 (typeA*)QprevEven, (typeA*)QprevOdd, \
								 (typeB*)linkEven, (typeB*)linkOdd, \
								 sig,  mu, \
								 (typename RealTypeId<typeA>::Type)coeff, \
								 (typename RealTypeId<typeA>::Type)accumu_coeff, \
								 (typeA*)shortPEven,(typeA*)shortPOdd, \
								 (typeA*)newOprodEven, (typeA*)newOprodOdd)

#define CALL_ALL_LINK_KERNEL(sig_sign, mu_sign)				\
      if(sizeof(RealA) == sizeof(float2)){				\
	if(recon  == QUDA_RECONSTRUCT_NO){				\
	  do_all_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 0> CALL_ARGUMENTS(float2, float2); \
	  do_all_link_sp_18_kernel<float2, float2, sig_sign, mu_sign, 1> CALL_ARGUMENTS(float2, float2); \
	}else{								\
	  do_all_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 0> CALL_ARGUMENTS(float2, float4); \
	  do_all_link_sp_12_kernel<float2, float4, sig_sign, mu_sign, 1> CALL_ARGUMENTS(float2, float4); \
	}								\
      }else{								\
	if(recon  == QUDA_RECONSTRUCT_NO){				\
	  do_all_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 0> CALL_ARGUMENTS(double2, double2); \
	  do_all_link_dp_18_kernel<double2, double2, sig_sign, mu_sign, 1> CALL_ARGUMENTS(double2, double2); \
	}else{								\
	  do_all_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 0> CALL_ARGUMENTS(double2, double2); \
	  do_all_link_dp_12_kernel<double2, double2, sig_sign, mu_sign, 1> CALL_ARGUMENTS(double2, double2); \
	}								\
      }
      
      if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
	CALL_ALL_LINK_KERNEL(1, 1);
      }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	CALL_ALL_LINK_KERNEL(1, 0);
      }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	CALL_ALL_LINK_KERNEL(0, 1);
      }else{
	CALL_ALL_LINK_KERNEL(0, 0);
      }
      
#undef CALL_ARGUMENTS
#undef CALL_ALL_LINK_KERNEL	    
      
      return;
    }
    

    template<class RealA>
      static void
      one_link_term(
          const RealA* const oprodEven, 
          const RealA* const oprodOdd,
          int sig, 
          typename RealTypeId<RealA>::Type coeff, 
          typename RealTypeId<RealA>::Type naik_coeff,
          dim3 gridDim, dim3 blockDim,
          RealA* const ForceMatrixEven,
	  RealA* const ForceMatrixOdd)
      {

        dim3 halfGridDim(gridDim.x/2,1,1);

        if(GOES_FORWARDS(sig)){

          do_one_link_term_kernel<RealA,0><<<halfGridDim,blockDim>>>(oprodEven, oprodOdd,
								     sig, coeff,
								     ForceMatrixEven, ForceMatrixOdd);
          do_one_link_term_kernel<RealA,1><<<halfGridDim,blockDim>>>(oprodEven, oprodOdd,
								     sig, coeff,
								     ForceMatrixEven, ForceMatrixOdd);
	  	  
        } // GOES_FORWARDS(sig)

        return;
      }

      template<class RealA,class RealB>
      void longlink_terms(const RealB* const linkEven, const RealB* const linkOdd,
			  const RealA* const naikOprodEven, const RealA* const naikOprodOdd,
			  int sig, typename RealTypeId<RealA>::Type naik_coeff,
			  dim3 gridDim, dim3 blockDim, const cudaGaugeField& link, 
			  RealA* const outputEven, RealA* const outputOdd)
      {
	
        dim3 halfGridDim(gridDim.x/2,1,1);
	
	QudaReconstructType recon = link.Reconstruct();;
	
#define CALL_ARGUMENTS(typeA, typeB)	<<<halfGridDim,blockDim>>>((typeB*)linkEven, (typeB*)linkOdd, \
								   (typeA*)naikOprodEven,  (typeA*)naikOprodOdd, \
								   sig, naik_coeff, \
								   (typeA*)outputEven, (typeA*)outputOdd); \
	
	
        if(GOES_BACKWARDS(sig)){
          errorQuda("sig does not go forward\n");
	}
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
#undef CALL_ARGUMENTS	
        return;
      }



          
    template<class RealA, class RealB>
      static void 
      complete_force_kernel(const RealA* const oprodEven, 
			    const RealA* const oprodOdd,
			    const RealB* const linkEven, 
			    const RealB* const linkOdd, 
			    const cudaGaugeField &link,
			    int sig, dim3 gridDim, dim3 blockDim,
			    RealA* const momEven, 
			    RealA* const momOdd)
    {
      dim3 halfGridDim(gridDim.x/2, 1, 1);
#define CALL_ARGUMENTS(typeA, typeB)  <<<halfGridDim, blockDim>>>((typeB*)linkEven, (typeB*)linkOdd, \
								  (typeA*)oprodEven, (typeA*)oprodOdd, \
								  sig,	\
								  (typeA*)momEven, (typeA*)momOdd); 

      QudaReconstructType recon = link.Reconstruct();
      
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
	
      
#undef CALL_ARGUMENTS   
      return;
    }


    
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



#define Pmu 	  tempmat[0]
#define P3        tempmat[1]
#define P5	  tempmat[2]
#define Pnumu     tempmat[3]

#define Qmu      tempCmat[0]
#define Qnumu    tempCmat[1]


    template<class Real, class  RealA, class RealB>
      static void
      do_hisq_staples_force_cuda( PathCoefficients<Real> act_path_coeff,
                          	 const QudaGaugeParam& param,
                                 const cudaGaugeField &oprod, 
                          	 const cudaGaugeField &link,
                          	 FullMatrix tempmat[4], 
                         	 FullMatrix tempCmat[2], 
                          	 cudaGaugeField &newOprod)
      {

	QudaReconstructType recon = link.Reconstruct();
        Real coeff;
        Real OneLink, Lepage, FiveSt, ThreeSt, SevenSt;
        Real mLepage, mFiveSt, mThreeSt;



        OneLink = act_path_coeff.one;
        ThreeSt = act_path_coeff.three; mThreeSt = -ThreeSt;
        FiveSt  = act_path_coeff.five; mFiveSt  = -FiveSt;
        SevenSt = act_path_coeff.seven; 
        Lepage  = act_path_coeff.lepage; mLepage  = -Lepage;
	
	
        const int volume = param.X[0]*param.X[1]*param.X[2]*param.X[3];
        dim3 blockDim(BLOCK_DIM,1,1);
        dim3 gridDim(volume/blockDim.x, 1, 1);

        for(int sig=0; sig<8; sig++){
          for(int mu=0; mu<8; mu++){
            if ( (mu == sig) || (mu == OPP_DIR(sig))){
              continue;
            }
            //3-link
            //Kernel A: middle link


            middle_link_kernel( 
                (RealA*)oprod.Even_p(), (RealA*)oprod.Odd_p(),                            // read only
                (RealA*)NULL,         (RealA*)NULL,                                       // read only
                (RealB*)link.Even_p(), (RealB*)link.Odd_p(),	                          // read only 
                link,  // read only
                sig, mu, mThreeSt,
                gridDim, blockDim,
                (RealA*)Pmu.even.data, (RealA*)Pmu.odd.data,                               // write only
                (RealA*)P3.even.data, (RealA*)P3.odd.data,                                 // write only
                (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,                               // write only     
                (RealA*)newOprod.Even_p(), (RealA*)newOprod.Odd_p());

            checkCudaError();

            for(int nu=0; nu < 8; nu++){
              if (nu == sig || nu == OPP_DIR(sig)
                  || nu == mu || nu == OPP_DIR(mu)){
                continue;
              }

              //5-link: middle link
              //Kernel B
              middle_link_kernel( 
                  (RealA*)Pmu.even.data, (RealA*)Pmu.odd.data,      // read only
                  (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,      // read only
                  (RealB*)link.Even_p(), (RealB*)link.Odd_p(), 
                  link, 
                  sig, nu, FiveSt,
                  gridDim, blockDim,
                  (RealA*)Pnumu.even.data, (RealA*)Pnumu.odd.data,  // write only
                  (RealA*)P5.even.data, (RealA*)P5.odd.data,        // write only
                  (RealA*)Qnumu.even.data, (RealA*)Qnumu.odd.data,  // write only
                  (RealA*)newOprod.Even_p(), (RealA*)newOprod.Odd_p());

              checkCudaError();

              for(int rho = 0; rho < 8; rho++){
                if (rho == sig || rho == OPP_DIR(sig)
                    || rho == mu || rho == OPP_DIR(mu)
                    || rho == nu || rho == OPP_DIR(nu)){
                  continue;
                }
                //7-link: middle link and side link
                if(FiveSt != 0)coeff = SevenSt/FiveSt; else coeff = 0;
                all_link_kernel(
                    (RealA*)Pnumu.even.data, (RealA*)Pnumu.odd.data,
                    (RealA*)Qnumu.even.data, (RealA*)Qnumu.odd.data,
                    (RealB*)link.Even_p(), (RealB*)link.Odd_p(), 
                    link,
                    sig, rho, SevenSt, coeff,
                    gridDim, blockDim,
                    (RealA*)P5.even.data, (RealA*)P5.odd.data, 
                    (RealA*)newOprod.Even_p(), (RealA*)newOprod.Odd_p());

                checkCudaError();

              }//rho  		


              //5-link: side link
              if(ThreeSt != 0)coeff = FiveSt/ThreeSt; else coeff = 0;
              side_link_kernel(
                  (RealA*)P5.even.data, (RealA*)P5.odd.data,    // read only
                  (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,  // read only
                  (RealB*)link.Even_p(), (RealB*)link.Odd_p(), 
                  link,
                  sig, nu, mFiveSt, coeff,
                  gridDim, blockDim,
                  (RealA*)P3.even.data, (RealA*)P3.odd.data,    // write
                  (RealA*)newOprod.Even_p(), (RealA*)newOprod.Odd_p());
              checkCudaError();

            } //nu 

            //lepage
	    if(Lepage != 0.){
              middle_link_kernel( 
                  (RealA*)Pmu.even.data, (RealA*)Pmu.odd.data,     // read only
                  (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,     // read only
                  (RealB*)link.Even_p(), (RealB*)link.Odd_p(), 
                  link, 
                  sig, mu, Lepage,
                  gridDim, blockDim,
                  (RealA*)NULL, (RealA*)NULL,                      // write only
                  (RealA*)P5.even.data, (RealA*)P5.odd.data,       // write only
                  (RealA*)NULL, (RealA*)NULL,                      // write only
		  (RealA*)newOprod.Even_p(), (RealA*)newOprod.Odd_p());


              if(ThreeSt != 0)coeff = Lepage/ThreeSt ; else coeff = 0;

              side_link_kernel(
                  (RealA*)P5.even.data, (RealA*)P5.odd.data,           // read only
                  (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,         // read only
                  (RealB*)link.Even_p(), (RealB*)link.Odd_p(), 
                  link,
                  sig, mu, mLepage, coeff,
                  gridDim, blockDim,
                  (RealA*)P3.even.data, (RealA*)P3.odd.data,           // write only
                  (RealA*)newOprod.Even_p(), (RealA*)newOprod.Odd_p());

                  checkCudaError();		
            } // Lepage != 0.0


            //3-link side link
            coeff=0.;
            side_link_kernel(
                (RealA*)P3.even.data, (RealA*)P3.odd.data, // read only
                (RealA*)NULL, (RealA*)NULL,                // read only
                (RealB*)link.Even_p(), (RealB*)link.Odd_p(), 
                link,
                sig, mu, ThreeSt, coeff,
                gridDim, blockDim, 
                (RealA*)NULL, (RealA*)NULL,                // write
		(RealA*)newOprod.Even_p(), (RealA*)newOprod.Odd_p());

            checkCudaError();			    

          }//mu
        }//sig


        for(int sig=0; sig<8; ++sig){
          if(GOES_FORWARDS(sig)){
            one_link_term(
                (RealA*)oprod.Even_p(), (RealA*)oprod.Odd_p(),
                sig, OneLink, 0.0,
                gridDim, blockDim,
                (RealA*)newOprod.Even_p(), (RealA*)newOprod.Odd_p());
          } // GOES_FORWARDS(sig)
          checkCudaError();
        }
      
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

	   const int volume = param.X[0]*param.X[1]*param.X[2]*param.X[3];
	   dim3 blockDim(BLOCK_DIM,1,1);
	   dim3 gridDim(volume/blockDim.x, 1, 1);

	   bind_tex_link(link, oprod);
	   for(int sig=0; sig<4; sig++){
		   if(param.cuda_prec == QUDA_DOUBLE_PRECISION){
		     complete_force_kernel((double2*)oprod.Even_p(), (double2*)oprod.Odd_p(),
					   (double2*)link.Even_p(), (double2*)link.Odd_p(), 
					   link,
					   sig, gridDim, blockDim,
					   (double2*)force->Even_p(), (double2*)force->Odd_p());
		   }else if(param.cuda_prec == QUDA_SINGLE_PRECISION){
		     complete_force_kernel((float2*)oprod.Even_p(), (float2*)oprod.Odd_p(),
					   (float2*)link.Even_p(), (float2*)link.Odd_p(), 
					   link,
					   sig, gridDim, blockDim,
					   (float2*)force->Even_p(), (float2*)force->Odd_p());
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
     const int volume = param.X[0]*param.X[1]*param.X[2]*param.X[3];
     dim3 blockDim(BLOCK_DIM,1,1);
     dim3 gridDim(volume/blockDim.x, 1, 1);

     bind_tex_link(link, *newOprod);
     
     for(int sig=0; sig<4; ++sig){
       if(param.cuda_prec == QUDA_DOUBLE_PRECISION){
	 longlink_terms((double2*)link.Even_p(), (double2*)link.Odd_p(),
			(double2*)oldOprod.Even_p(), (double2*)oldOprod.Odd_p(),
			sig, coeff, 
			gridDim, blockDim, link, 
			(double2*)newOprod->Even_p(), (double2*)newOprod->Odd_p());
       }else if(param.cuda_prec == QUDA_SINGLE_PRECISION){
	 longlink_terms((float2*)link.Even_p(), (float2*)link.Odd_p(),
			(float2*)oldOprod.Even_p(), (float2*)oldOprod.Odd_p(),
			sig, static_cast<float>(coeff), 
			gridDim, blockDim, link,
			(float2*)newOprod->Even_p(), (float2*)newOprod->Odd_p());
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

        FullMatrix tempmat[4];
        for(int i=0; i<4; i++){
          tempmat[i]  = createMatQuda(param.X, param.cuda_prec);
        }

        FullMatrix tempCompmat[2];
        for(int i=0; i<2; i++){
          tempCompmat[i] = createMatQuda(param.X, param.cuda_prec);
        }	

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
							   tempmat, 
							   tempCompmat, 
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
							   tempmat, 
							   tempCompmat, 
							   *newOprod);
        }else{
	  errorQuda("Unsupported precision");
	}
	
	
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float runtime;
	cudaEventElapsedTime(&runtime, start, end);
	
	//printfQuda("hisq staple time=%.2f ms\n", runtime);

	unbind_tex_link(link, *newOprod);

        for(int i=0; i<4; i++){
          freeMatQuda(tempmat[i]);
        }

        for(int i=0; i<2; i++){
          freeMatQuda(tempCompmat[i]);
        }
        return; 
      }

  } // namespace fermion_force
} // namespace hisq
