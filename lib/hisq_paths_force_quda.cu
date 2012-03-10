#include <read_gauge.h>
#include <gauge_field.h>

#include <hisq_force_quda.h>
#include <hw_quda.h>
#include <hisq_force_macros.h>
#include<utility>


// Disable texture read for now. Need to revisit this.
//#define FF_SITE_MATRIX_LOAD_TEX 1

#if (FF_SITE_MATRIX_LOAD_TEX == 1)
#define linkEvenTex siteLink0TexSingle_recon
#define linkOddTex siteLink1TexSingle_recon
#endif




namespace hisq {
  namespace fermion_force {

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
      void loadMatrixFromField(const T* const field_even, const T* const field_odd,
				int dir, int idx, T* const mat, int oddness)
      {
	const T* const field = (oddness)?field_odd:field_even;
        mat[0] = field[idx + dir*Vhx9];
        mat[1] = field[idx + dir*Vhx9 + Vh];
        mat[2] = field[idx + dir*Vhx9 + Vhx2];
        mat[3] = field[idx + dir*Vhx9 + Vhx3];
        mat[4] = field[idx + dir*Vhx9 + Vhx4];
        mat[5] = field[idx + dir*Vhx9 + Vhx5];
        mat[6] = field[idx + dir*Vhx9 + Vhx6];
        mat[7] = field[idx + dir*Vhx9 + Vhx7];
        mat[8] = field[idx + dir*Vhx9 + Vhx8];

        return;
      }



    template<class T>
      inline __device__
      void loadAdjointMatrixFromField(const T* const field_even, const T* const field_odd, int dir, int idx, T* const mat, int oddness)
      {
	const T* const field = (oddness)?field_odd: field_even;
#define CONJ_INDEX(i,j) j*3 + i
        mat[CONJ_INDEX(0,0)] = conj(field[idx + dir*Vhx9]);
        mat[CONJ_INDEX(0,1)] = conj(field[idx + dir*Vhx9 + Vh]);
        mat[CONJ_INDEX(0,2)] = conj(field[idx + dir*Vhx9 + Vhx2]);
        mat[CONJ_INDEX(1,0)] = conj(field[idx + dir*Vhx9 + Vhx3]);
        mat[CONJ_INDEX(1,1)] = conj(field[idx + dir*Vhx9 + Vhx4]);
        mat[CONJ_INDEX(1,2)] = conj(field[idx + dir*Vhx9 + Vhx5]);
        mat[CONJ_INDEX(2,0)] = conj(field[idx + dir*Vhx9 + Vhx6]);
        mat[CONJ_INDEX(2,1)] = conj(field[idx + dir*Vhx9 + Vhx7]);
        mat[CONJ_INDEX(2,2)] = conj(field[idx + dir*Vhx9 + Vhx8]);
#undef CONJ_INDEX
        return;
      }

    inline __device__
      void loadMatrixFromField(const float4* const field, int dir, int idx, float4* const mat)
      {
        mat[0] = field[idx + dir*Vhx3];
        mat[1] = field[idx + dir*Vhx3 + Vh];
        mat[2] = field[idx + dir*Vhx3 + Vhx2];
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
    __device__ void reconstructSign(int* const sign, int dir, const int i[4]){

 /*
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
      }
*/
      return;
    }



      void
      hisqForceInitCuda(QudaGaugeParam* param)
      {
        static int hisq_force_init_cuda_flag = 0; 

        if (hisq_force_init_cuda_flag){
          return;
        }
        hisq_force_init_cuda_flag=1;
        init_kernel_cuda(param);    
      }

  
    
#include "hisq_paths_force_core.h"

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
        dim3 halfGridDim(gridDim.x/2, 1,1);

#define CALL_MIDDLE_LINK_KERNEL(sig_sign, mu_sign)			\
	do_middle_link_kernel<RealA, RealB, sig_sign, mu_sign, 0><<<halfGridDim, BlockDim>>>(oprodEven, oprodOdd, \
											     QprevEven, QprevOdd, \
											     linkEven, linkOdd, \
											     sig, mu, coeff, \
											     PmuEven, PmuOdd, \
											     P3Even, P3Odd, \
											     QmuEven, QmuOdd, \
											     newOprodEven, newOprodOdd); \
	do_middle_link_kernel<RealA, RealB, sig_sign, mu_sign, 1><<<halfGridDim, BlockDim>>>(oprodEven, oprodOdd, \
											     QprevEven, QprevOdd, \
											     linkEven, linkOdd, \
											     sig, mu, coeff, \
											     PmuEven, PmuOdd, \
											     P3Even, P3Odd, \
											     QmuEven, QmuOdd, \
											     newOprodEven, newOprodOdd);
	
        if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){	
	  CALL_MIDDLE_LINK_KERNEL(1,1);
        }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	  CALL_MIDDLE_LINK_KERNEL(1,0);
        }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	  CALL_MIDDLE_LINK_KERNEL(0,1);
        }else{
	  CALL_MIDDLE_LINK_KERNEL(0,0);
        }
	
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
      dim3 halfGridDim(gridDim.x/2,1,1);
	
#define CALL_SIDE_LINK_KERNEL(sig_sign, mu_sign)			\
      do_side_link_kernel<RealA, RealB, sig_sign, mu_sign, 0><<<halfGridDim, blockDim>>>(P3Even, P3Odd, \
											 oprodEven,  oprodOdd, \
											 linkEven, linkOdd, \
											 sig, mu, coeff, accumu_coeff, \
											 shortPEven, shortPOdd, \
											 newOprodEven, newOprodOdd); \
      do_side_link_kernel<RealA, RealB, sig_sign, mu_sign, 1><<<halfGridDim, blockDim>>>(P3Even, P3Odd, \
											 oprodEven,  oprodOdd, \
											 linkEven, linkOdd, \
											 sig, mu, coeff, accumu_coeff, \
											 shortPEven, shortPOdd, \
											 newOprodEven, newOprodOdd);
	
      
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
            dim3 halfGridDim(gridDim.x/2, 1,1);
	    
#define CALL_ALL_LINK_KERNEL(sig_sign, mu_sign)				\
            do_all_link_kernel<RealA, RealB, sig_sign, mu_sign, 0><<<halfGridDim, blockDim>>>(oprodEven, oprodOdd, \
											      QprevEven, QprevOdd, \
											      linkEven, linkOdd, \
											      sig,  mu, \
											      coeff, accumu_coeff, \
											      shortPEven,shortPOdd, \
											      newOprodEven, newOprodOdd); \
	    do_all_link_kernel<RealA, RealB, sig_sign, mu_sign, 1><<<halfGridDim, blockDim>>>(oprodEven, oprodOdd, \
											      QprevEven, QprevOdd, \
											      linkEven, linkOdd, \
											      sig,  mu, \
											      coeff, accumu_coeff, \
											      shortPEven,shortPOdd, \
											      newOprodEven, newOprodOdd);
	    
            if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
	      CALL_ALL_LINK_KERNEL(1, 1);
            }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	      CALL_ALL_LINK_KERNEL(1, 0);
            }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	      CALL_ALL_LINK_KERNEL(0, 1);
            }else{
	      CALL_ALL_LINK_KERNEL(0, 0);
            }

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

    template<class RealA>
      void longlink_terms(const RealA* const linkEven, const RealA* const linkOdd,
			  const RealA* const naikOprodEven, const RealA* const naikOprodOdd,
			  int sig, typename RealTypeId<RealA>::Type naik_coeff,
			  dim3 gridDim, dim3 blockDim,
			  RealA* const outputEven, RealA* const outputOdd)
      {
	
        dim3 halfGridDim(gridDim.x/2,1,1);
	
        if(GOES_FORWARDS(sig)){
          do_longlink_kernel<RealA,0><<<halfGridDim,blockDim>>>(linkEven, linkOdd,
								naikOprodEven, naikOprodOdd,
								sig, naik_coeff,
								outputEven, outputOdd);
          do_longlink_kernel<RealA,1><<<halfGridDim,blockDim>>>(linkEven, linkOdd,
								naikOprodEven, naikOprodOdd,
								sig, naik_coeff,
								outputEven, outputOdd);
        }
        else {
          errorQuda("sig does not go forward\n");
        }
	
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
      
      cudaBindTexture(0, siteLink0TexSingle_recon, link.Even_p(), link.Bytes()/2);
      cudaBindTexture(0, siteLink1TexSingle_recon, link.Odd_p(),  link.Bytes()/2);
      
      do_complete_force_kernel<RealA, RealB, 0><<<halfGridDim, blockDim>>>(linkEven, linkOdd,
									   oprodEven, oprodOdd,
									   sig,
									   momEven, momOdd);
      do_complete_force_kernel<RealA, RealB, 1><<<halfGridDim, blockDim>>>(linkEven, linkOdd,
									   oprodEven, oprodOdd,
									   sig,
									   momEven, momOdd);			
      cudaUnbindTexture(siteLink0TexSingle_recon);
      cudaUnbindTexture(siteLink1TexSingle_recon);
      
      return;
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


        Real coeff;
        Real OneLink, Lepage, FiveSt, ThreeSt, SevenSt;
        Real mLepage, mFiveSt, mThreeSt;



        OneLink = act_path_coeff.one;
        ThreeSt = act_path_coeff.three; mThreeSt = -ThreeSt;
        FiveSt  = act_path_coeff.five; mFiveSt  = -FiveSt;
        SevenSt = act_path_coeff.seven; 
        Lepage  = act_path_coeff.lepage; mLepage  = -Lepage;

	cudaBindTexture(0, siteLink0TexSingle_recon, link.Even_p(), link.Bytes()/2);
	cudaBindTexture(0, siteLink1TexSingle_recon, link.Odd_p(), link.Bytes()/2);   


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

        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);   


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
     
     for(int sig=0; sig<4; ++sig){
       if(param.cuda_prec == QUDA_DOUBLE_PRECISION){
	 longlink_terms((double2*)link.Even_p(), (double2*)link.Odd_p(),
			(double2*)oldOprod.Even_p(), (double2*)oldOprod.Odd_p(),
			sig, coeff, 
			gridDim, blockDim,
			(double2*)newOprod->Even_p(), (double2*)newOprod->Odd_p());
       }else if(param.cuda_prec == QUDA_SINGLE_PRECISION){
	 longlink_terms((float2*)link.Even_p(), (float2*)link.Odd_p(),
			(float2*)oldOprod.Even_p(), (float2*)oldOprod.Odd_p(),
			sig, static_cast<float>(coeff), 
			gridDim, blockDim,
			(float2*)newOprod->Even_p(), (float2*)newOprod->Odd_p());
       }else{
	 errorQuda("Unsupported precision");
       }
     } // loop over directions
     
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
