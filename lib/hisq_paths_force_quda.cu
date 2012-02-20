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
#define FF_LOAD_MATRIX(src, dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX(src##Tex, dir, idx, var)
#define FF_LOAD_ARRAY(src, dir, idx, var) LOAD_ARRAY_12_SINGLE_TEX(src##Tex, dir, idx, var)    
#else
#define FF_LOAD_MATRIX(src, dir, idx, var) LOAD_MATRIX_12_SINGLE(src, dir, idx, var)
#define FF_LOAD_ARRAY(src, dir, idx, var) LOAD_ARRAY_12_SINGLE(src, dir, idx, var)    
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
      void loadMatrixFromField(const T* const field, int dir, int idx, T* const mat)
      {
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
      void loadAdjointMatrixFromField(const T* const field, int dir, int idx, T* const mat)
      {
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
      void loadMatrixFromField(const T* const field, int idx, T* const mat)
      {
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
      void addMatrixToField(const T* const mat, int dir, int idx, U coeff, T* const field)
      {
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
      void addMatrixToField(const T* const mat, int idx, U coeff, T* const field)
      {
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
      void storeMatrixToField(const T* const mat, int dir, int idx, T* const field)
      {
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
      void storeMatrixToField(const T* const mat, int idx, T* const field)
      {
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
	void storeMatrixToMomentumField(const T* const mat, int dir, int idx, U coeff, T* const mom_field)
 	{
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




    template<class RealA, class RealB, int oddBit>
      __global__ void 
      do_complete_force_kernel(const RealB* const linkEven, 
                                 const RealA* const oprodEven,      
                                 int sig,
                                 RealA* const forceEven)
      {
        int sid = blockIdx.x * blockDim.x + threadIdx.x;

        int x[4];
        int z1 = sid/X1h;
        int x1h = sid - z1*X1h;
        int z2 = z1/X2;
        x[1] = z1 - z2*X2;
        x[3] = z2/X3;
        x[2] = z2 - x[3]*X3;
        int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
        x[0] = 2*x1h + x1odd;

        int link_sign;

        RealB LINK_W[ArrayLength<RealB>::result];
        RealA COLOR_MAT_W[ArrayLength<RealA>::result];
        RealA COLOR_MAT_X[ArrayLength<RealA>::result];


        loadMatrixFromField(linkEven, sig, sid, LINK_W);
        reconstructSign(&link_sign, sig, x);	

        loadMatrixFromField(oprodEven, sig, sid, COLOR_MAT_X);

	typename RealTypeId<RealA>::Type coeff = (oddBit==1) ? -1 : 1;
        MAT_MUL_MAT(LINK_W, COLOR_MAT_X, COLOR_MAT_W);
	
	storeMatrixToMomentumField(COLOR_MAT_W, sig, sid, coeff, forceEven); 
        return;
      }

    template<class RealA, int oddBit>
      __global__ void 
      do_one_link_term_kernel(
          const RealA* const oprodEven, 
          int sig, 
          typename RealTypeId<RealA>::Type coeff,
          RealA* const outputEven
          )
      {
        int sid = blockIdx.x * blockDim.x + threadIdx.x;

        RealA COLOR_MAT_W[ArrayLength<RealA>::result];
        if(GOES_FORWARDS(sig)){
          loadMatrixFromField(oprodEven, sig, sid, COLOR_MAT_W);
          addMatrixToField(COLOR_MAT_W, sig, sid, coeff, outputEven);
        }
        return;
      }
 
    template<class RealA, int oddBit>
      __global__ void 
      do_longlink_kernel(
          const RealA* const linkEven,
          const RealA* const linkOdd,
          const RealA* const naikOprodEven,
          const RealA* const naikOprodOdd,
          int sig, typename RealTypeId<RealA>::Type coeff,
	  RealA* const outputEven)
      {
       
        int sid = blockIdx.x * blockDim.x + threadIdx.x;

        int x[4];
        int z1 = sid/X1h;
        int x1h = sid - z1*X1h;
        int z2 = z1/X2;
        x[1] = z1 - z2*X2;
        x[3] = z2/X3;
        x[2] = z2 - x[3]*X3;
        int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
        x[0] = 2*x1h + x1odd;

        int new_x[4];
        new_x[0] = x[0];
        new_x[1] = x[1];
        new_x[2] = x[2];
        new_x[3] = x[3];


        RealA LINK_W[ArrayLength<RealA>::result];
        RealA LINK_X[ArrayLength<RealA>::result];
        RealA LINK_Y[ArrayLength<RealA>::result];
        RealA LINK_Z[ArrayLength<RealA>::result];

        RealA COLOR_MAT_U[ArrayLength<RealA>::result];
        RealA COLOR_MAT_V[ArrayLength<RealA>::result];
        RealA COLOR_MAT_W[ArrayLength<RealA>::result]; // used as a temporary
        RealA COLOR_MAT_X[ArrayLength<RealA>::result];
        RealA COLOR_MAT_Y[ArrayLength<RealA>::result];
        RealA COLOR_MAT_Z[ArrayLength<RealA>::result];


        const int & point_c = sid;
        int point_a, point_b, point_d, point_e;
        // need to work these indices
        int X[4];
        X[0] = X1;
        X[1] = X2;
        X[2] = X3;
        X[3] = X4;

       // compute the force for forward long links
        if(GOES_FORWARDS(sig))
        {
          new_x[sig] = (x[sig] + 1 + X[sig])%X[sig];
          point_d = (new_x[3]*X3X2X1+new_x[2]*X2X1+new_x[1]*X1+new_x[0]) >> 1;

          new_x[sig] = (new_x[sig] + 1 + X[sig])%X[sig];
          point_e = (new_x[3]*X3X2X1+new_x[2]*X2X1+new_x[1]*X1+new_x[0]) >> 1;

          new_x[sig] = (x[sig] - 1 + X[sig])%X[sig];
          point_b = (new_x[3]*X3X2X1+new_x[2]*X2X1+new_x[1]*X1+new_x[0]) >> 1;

          new_x[sig] = (new_x[sig] - 1 + X[sig])%X[sig];
          point_a = (new_x[3]*X3X2X1+new_x[2]*X2X1+new_x[1]*X1+new_x[0]) >> 1;

          loadMatrixFromField(linkEven, sig, point_a, LINK_W);
          loadMatrixFromField(linkOdd, sig, point_b, LINK_X);
          loadMatrixFromField(linkOdd, sig, point_d, LINK_Y);
          loadMatrixFromField(linkEven, sig, point_e, LINK_Z);

          loadMatrixFromField(naikOprodEven, sig, point_c, COLOR_MAT_Z);
          loadMatrixFromField(naikOprodOdd, sig, point_b, COLOR_MAT_Y);
          loadMatrixFromField(naikOprodEven, sig, point_a, COLOR_MAT_X);


          MAT_MUL_MAT(LINK_Z, COLOR_MAT_Z, COLOR_MAT_W); // link(d)*link(e)*Naik(c)
          MAT_MUL_MAT(LINK_Y, COLOR_MAT_W, COLOR_MAT_V);

          MAT_MUL_MAT(LINK_Y, COLOR_MAT_Y, COLOR_MAT_W);  // link(d)*Naik(b)*link(b)
          MAT_MUL_MAT(COLOR_MAT_W, LINK_X, COLOR_MAT_U);
	  SCALAR_MULT_ADD_MATRIX(COLOR_MAT_V, COLOR_MAT_U, -1, COLOR_MAT_V);

          MAT_MUL_MAT(COLOR_MAT_X, LINK_W, COLOR_MAT_W); // Naik(a)*link(a)*link(b)
          MAT_MUL_MAT(COLOR_MAT_W, LINK_X, COLOR_MAT_U);
          SCALAR_MULT_ADD_MATRIX(COLOR_MAT_V, COLOR_MAT_U, 1, COLOR_MAT_V);

          addMatrixToField(COLOR_MAT_V, sig, sid,  coeff, outputEven);
        }

        return;
      }




    template<class RealA>
      void longlink_terms(
          const RealA* const linkEven,
          const RealA* const linkOdd,
          const RealA* const naikOprodEven,
          const RealA* const naikOprodOdd,
          int sig, typename RealTypeId<RealA>::Type naik_coeff,
          dim3 gridDim, dim3 blockDim,
	  RealA* const outputEven,
	  RealA* const outputOdd)
      {

        dim3 halfGridDim(gridDim.x/2,1,1);



        if(GOES_FORWARDS(sig)){
          // Even half lattice
          do_longlink_kernel<RealA,0><<<halfGridDim,blockDim>>>(linkEven,
                                                               linkOdd,
                                                               naikOprodEven,
                                                               naikOprodOdd,
                                                               sig, naik_coeff,
							       outputEven);

          // Odd half lattice
          do_longlink_kernel<RealA,1><<<halfGridDim,blockDim>>>(linkOdd,
                                                                linkEven,
                                                                naikOprodOdd,
                                                                naikOprodEven,
                                                                sig, naik_coeff,
								outputOdd);
        }
        else {
          errorQuda("sig does not go forward\n");
        }

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

          do_one_link_term_kernel<RealA,0><<<halfGridDim,blockDim>>>(
              oprodEven,
              sig, coeff,
              ForceMatrixEven
              );

          do_one_link_term_kernel<RealA, 1><<<halfGridDim,blockDim>>>(
              oprodOdd,
              sig, coeff,
              ForceMatrixOdd
              );

        } // GOES_FORWARDS(sig)

        return;
      }



    template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit> 
      __global__ void
      do_middle_link_kernel(
          const RealA* const oprodEven, 
          const RealA* const oprodOdd,
          const RealA* const QprevOdd, 		
          const RealB* const linkEven, 
          const RealB* const linkOdd,
          int sig, int mu, 
          typename RealTypeId<RealA>::Type coeff,
          RealA* const PmuOdd, 
          RealA* const P3Even,
          RealA* const QmuEven, 
          RealA* const newOprodEven 
          ) 
      {		
        int sid = blockIdx.x * blockDim.x + threadIdx.x;

        int x[4];
        int z1 = sid/X1h;
        int x1h = sid - z1*X1h;
        int z2 = z1/X2;
        x[1] = z1 - z2*X2;
        x[3] = z2/X3;
        x[2] = z2 - x[3]*X3;
        int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
        x[0] = 2*x1h + x1odd;
        int X = 2*sid + x1odd;

        int new_x[4];
        int new_mem_idx;
        int ad_link_sign=1;
        int ab_link_sign=1;
        int bc_link_sign=1;

        RealB LINK_W[ArrayLength<RealB>::result];
        RealB LINK_X[ArrayLength<RealB>::result];
        RealB LINK_Y[ArrayLength<RealB>::result];


        RealA COLOR_MAT_W[ArrayLength<RealA>::result];
        RealA COLOR_MAT_Y[ArrayLength<RealA>::result];
        RealA COLOR_MAT_X[ArrayLength<RealA>::result];

        //        A________B
        //    mu   |      |
        // 	  D|      |C
        //	  
        //	  A is the current point (sid)
        int point_b, point_c, point_d;
        int ad_link_nbr_idx, ab_link_nbr_idx, bc_link_nbr_idx;
        int mymu;

        new_x[0] = x[0];
        new_x[1] = x[1];
        new_x[2] = x[2];
        new_x[3] = x[3];

        if(mu_positive){
          mymu = mu;
          FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mu, X, new_mem_idx);
        }else{
          mymu = OPP_DIR(mu);
          FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(OPP_DIR(mu), X, new_mem_idx);	
        }
        point_d = (new_mem_idx >> 1);
        if (mu_positive){
          ad_link_nbr_idx = point_d;
          reconstructSign(&ad_link_sign, mymu, new_x);
        }else{
          ad_link_nbr_idx = sid;
          reconstructSign(&ad_link_sign, mymu, x);	
        }

        int mysig; 
        if(sig_positive){
          mysig = sig;
          FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, new_mem_idx, new_mem_idx);
        }else{
          mysig = OPP_DIR(sig);
          FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), new_mem_idx, new_mem_idx);	
        }
        point_c = (new_mem_idx >> 1);
        if (mu_positive){
          bc_link_nbr_idx = point_c;	
          reconstructSign(&bc_link_sign, mymu, new_x);
        }

        new_x[0] = x[0];
        new_x[1] = x[1];
        new_x[2] = x[2];
        new_x[3] = x[3];

        if(sig_positive){
          FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, X, new_mem_idx);
        }else{
          FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), X, new_mem_idx);	
        }
        point_b = (new_mem_idx >> 1); 

        if (!mu_positive){
          bc_link_nbr_idx = point_b;
          reconstructSign(&bc_link_sign, mymu, new_x);
        }   

        if(sig_positive){
          ab_link_nbr_idx = sid;
          reconstructSign(&ab_link_sign, mysig, x);	
        }else{	
          ab_link_nbr_idx = point_b;
          reconstructSign(&ab_link_sign, mysig, new_x);
        }
        // now we have ab_link_nbr_idx


        // load the link variable connecting a and b 
        // Store in LINK_W 
        if(sig_positive){
          loadMatrixFromField(linkEven, mysig, ab_link_nbr_idx, LINK_W);
        }else{
          loadMatrixFromField(linkOdd, mysig, ab_link_nbr_idx, LINK_W);
        }

        // load the link variable connecting b and c 
        // Store in LINK_X
        if(mu_positive){
          loadMatrixFromField(linkEven, mymu, bc_link_nbr_idx, LINK_X);
        }else{ 
          loadMatrixFromField(linkOdd, mymu, bc_link_nbr_idx, LINK_X);
        }


        if(QprevOdd == NULL){
          if(sig_positive){
            loadMatrixFromField(oprodOdd, sig, point_d, COLOR_MAT_Y);
          }else{
	    loadAdjointMatrixFromField(oprodEven, OPP_DIR(sig), point_c, COLOR_MAT_Y);
          }
        }else{ // QprevOdd != NULL
          loadMatrixFromField(oprodEven, point_c, COLOR_MAT_Y);
        }
       

        MATRIX_PRODUCT(COLOR_MAT_W, LINK_X, COLOR_MAT_Y, !mu_positive);
        if(PmuOdd){
	  storeMatrixToField(COLOR_MAT_W, point_b, PmuOdd);
        }
        MATRIX_PRODUCT(COLOR_MAT_Y, LINK_W, COLOR_MAT_W, sig_positive);
	storeMatrixToField(COLOR_MAT_Y, sid, P3Even);


        if(mu_positive){
          loadMatrixFromField(linkOdd, mymu, ad_link_nbr_idx, LINK_Y);
        }else{
          loadAdjointMatrixFromField(linkEven, mymu, ad_link_nbr_idx, LINK_Y);
        }


        if(QprevOdd == NULL){
          if(sig_positive){
            MAT_MUL_MAT(COLOR_MAT_W, LINK_Y, COLOR_MAT_Y);
          }
          if(QmuEven){
            ASSIGN_MAT(LINK_Y, COLOR_MAT_X); 
	    storeMatrixToField(COLOR_MAT_X, sid, QmuEven);
          }
        }else{ 
          loadMatrixFromField(QprevOdd, point_d, COLOR_MAT_Y);
          MAT_MUL_MAT(COLOR_MAT_Y, LINK_Y, COLOR_MAT_X);
          if(QmuEven){
	    storeMatrixToField(COLOR_MAT_X, sid, QmuEven);
          }
          if(sig_positive){
            MAT_MUL_MAT(COLOR_MAT_W, COLOR_MAT_X, COLOR_MAT_Y);
          }	
        }


        if(sig_positive){
          addMatrixToField(COLOR_MAT_Y, sig, sid, coeff, newOprodEven);
        }

        return;
      }




    template<class RealA, class RealB>
      static void 
      complete_force_kernel(
          const RealA* const oprodEven, 
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

        do_complete_force_kernel<RealA, RealB, 0><<<halfGridDim, blockDim>>>(linkEven,
                                                                               oprodEven,
                                                                               sig,
                                                                               momEven);

        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);

        cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
        cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);

        do_complete_force_kernel<RealA, RealB, 1><<<halfGridDim, blockDim>>>(linkOdd,
                                                                               oprodOdd,
                                                                               sig,
                                                                               momOdd);

        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);

        return;
      }



    template<class RealA, class RealB>
      static void
      middle_link_kernel(
          const RealA* const oprodEven, 
          const RealA* const oprodOdd, 
          const RealA* const QprevEven, 
          const RealA* const QprevOdd,
          const RealB* const linkEven, 
          const RealB* const linkOdd, 
          const cudaGaugeField &link,
          int sig, int mu, 
          typename RealTypeId<RealA>::Type coeff,
          dim3 gridDim, dim3 BlockDim,
          RealA* const PmuEven, // write only  
          RealA* const PmuOdd, // write only
          RealA* const P3Even, // write only   
          RealA* const P3Odd,  // write only
          RealA* const QmuEven,  // write only
          RealA* const QmuOdd,   // write only
          RealA* const newOprodEven, 
          RealA* const newOprodOdd)
      {
        dim3 halfGridDim(gridDim.x/2, 1,1);

        cudaBindTexture(0, siteLink0TexSingle_recon, link.Even_p(), link.Bytes()/2);
        cudaBindTexture(0, siteLink1TexSingle_recon, link.Odd_p(), link.Bytes()/2);

        if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){	
          do_middle_link_kernel<RealA, RealB, 1, 1, 0><<<halfGridDim, BlockDim>>>( oprodEven, oprodOdd,
              QprevOdd,
              linkEven, linkOdd,
              sig, mu, coeff,
              PmuOdd,  P3Even,
              QmuEven, 
              newOprodEven);

          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);
          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);

          do_middle_link_kernel<RealA, RealB, 1, 1, 1><<<halfGridDim, BlockDim>>>( oprodOdd, oprodEven,
              QprevEven,
              linkOdd, linkEven,
              sig, mu, coeff,
              PmuEven,  P3Odd,
              QmuOdd, 
              newOprodOdd);

        }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
          do_middle_link_kernel<RealA, RealB, 1, 0, 0><<<halfGridDim, BlockDim>>>( oprodEven, oprodOdd,
              QprevOdd,
              linkEven, linkOdd,
              sig, mu, coeff,
              PmuOdd,  P3Even,
              QmuEven,
              newOprodEven);
	
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);

          do_middle_link_kernel<RealA, RealB, 1, 0, 1><<<halfGridDim, BlockDim>>>( oprodOdd, oprodEven,
              QprevEven,
              linkOdd, linkEven,
              sig, mu, coeff,
              PmuEven,  P3Odd,
              QmuOdd,  
              newOprodOdd);

        }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){

          do_middle_link_kernel<RealA, RealB, 0, 1, 0><<<halfGridDim, BlockDim>>>( oprodEven, oprodOdd,
              QprevOdd,
              linkEven, linkOdd,
              sig, mu, coeff,
              PmuOdd,  P3Even,
              QmuEven, 
              newOprodEven);
	
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);

          do_middle_link_kernel<RealA, RealB, 0, 1, 1><<<halfGridDim, BlockDim>>>( oprodOdd, oprodEven,
              QprevEven, 
              linkOdd, linkEven,
              sig, mu, coeff,
              PmuEven,  P3Odd,
              QmuOdd, 
              newOprodOdd);

        }else{

          do_middle_link_kernel<RealA, RealB, 0, 0, 0><<<halfGridDim, BlockDim>>>( oprodEven, oprodOdd,
              QprevOdd,
              linkEven, linkOdd,
              sig, mu, coeff,
              PmuOdd, P3Even,
              QmuEven, 
              newOprodEven);		

          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);

          do_middle_link_kernel<RealA, RealB, 0, 0, 1><<<halfGridDim, BlockDim>>>( oprodOdd, oprodEven,
              QprevEven,
              linkOdd, linkEven,
              sig, mu, coeff,
              PmuEven,  P3Odd,
              QmuOdd,  
              newOprodOdd);		
        }
        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);    

        return;
      }



    template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit>
      __global__ void
      do_side_link_kernel(
          const RealA* const P3Even, 
          const RealA* const oprodEven, 
          const RealA* const oprodOdd,
          const RealB* const linkEven, 
          const RealB* const linkOdd,
          int sig, int mu, 
          typename RealTypeId<RealA>::Type coeff, 
          typename RealTypeId<RealA>::Type accumu_coeff,
          RealA* const shortPOdd,
          RealA* const newOprodEven, 
          RealA* const newOprodOdd)
      {

        int sid = blockIdx.x * blockDim.x + threadIdx.x;

        int x[4];
        int z1 = sid/X1h;
        int x1h = sid - z1*X1h;
        int z2 = z1/X2;
        x[1] = z1 - z2*X2;
        x[3] = z2/X3;
        x[2] = z2 - x[3]*X3;
        int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
        x[0] = 2*x1h + x1odd;
        int X = 2*sid + x1odd;

        int ad_link_sign = 1;

        RealB LINK_W[ArrayLength<RealB>::result];

        RealA COLOR_MAT_W[ArrayLength<RealA>::result];
        RealA COLOR_MAT_X[ArrayLength<RealA>::result]; 
        RealA COLOR_MAT_Y[ArrayLength<RealA>::result]; 
        // The compiler probably knows to reorder so that loads are done early on
        loadMatrixFromField(P3Even, sid, COLOR_MAT_Y);

//      compute the side link contribution to the momentum
//
//             sig
//          A________B
//           |      |   mu
//         D |      |C
//
//      A is the current point (sid)

        typename RealTypeId<RealA>::Type mycoeff;
        int point_d;
        int ad_link_nbr_idx;
        int mymu;
        int new_mem_idx;

        int new_x[4];
        new_x[0] = x[0];
        new_x[1] = x[1];
        new_x[2] = x[2];
        new_x[3] = x[3];

        if(mu_positive){
          mymu=mu;
          FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mymu,X, new_mem_idx);
        }else{
          mymu = OPP_DIR(mu);
          FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(mymu, X, new_mem_idx);
        }
        point_d = (new_mem_idx >> 1);


        if (mu_positive){
          ad_link_nbr_idx = point_d;
          reconstructSign(&ad_link_sign, mymu, new_x);
        }else{
          ad_link_nbr_idx = sid;
          reconstructSign(&ad_link_sign, mymu, x);	
        }


        if(mu_positive){
          loadMatrixFromField(linkOdd, mymu, ad_link_nbr_idx, LINK_W);
        }else{
          loadMatrixFromField(linkEven, mymu, ad_link_nbr_idx, LINK_W);
        }


        // Should all be inside if (shortPOdd)
        if (shortPOdd){
          MATRIX_PRODUCT(COLOR_MAT_W, LINK_W, COLOR_MAT_Y, mu_positive);
          addMatrixToField(COLOR_MAT_W, point_d, accumu_coeff, shortPOdd);
        }


        mycoeff = CoeffSign<sig_positive,oddBit>::result*coeff;

        if(oprodOdd){
          loadMatrixFromField(oprodOdd, point_d, COLOR_MAT_X);
          if(mu_positive){
            MAT_MUL_MAT(COLOR_MAT_Y, COLOR_MAT_X, COLOR_MAT_W);

            // Added by J.F.
            if(!oddBit){ mycoeff = -mycoeff; }
            addMatrixToField(COLOR_MAT_W, mu, point_d, mycoeff, newOprodOdd);
          }else{
            ADJ_MAT_MUL_ADJ_MAT(COLOR_MAT_X, COLOR_MAT_Y, COLOR_MAT_W);
            if(oddBit){ mycoeff = -mycoeff; }
            addMatrixToField(COLOR_MAT_W, OPP_DIR(mu), sid, mycoeff, newOprodEven);
          } 
        }

        if(!oprodOdd){
          if(mu_positive){
            if(!oddBit){ mycoeff = -mycoeff;}
            addMatrixToField(COLOR_MAT_Y, mu, point_d, mycoeff, newOprodOdd);
          }else{
            if(oddBit){ mycoeff = -mycoeff; }
            ADJ_MAT(COLOR_MAT_Y, COLOR_MAT_W);
            addMatrixToField(COLOR_MAT_W, OPP_DIR(mu), sid, mycoeff, newOprodEven);
          }
        }

        return;
      }




    template<class RealA, class RealB>
      static void
      side_link_kernel(
          const RealA* const P3Even, 
          const RealA* const P3Odd, 
          const RealA* const oprodEven, 
          const RealA* const oprodOdd,
          const RealB* const linkEven, 
          const RealB* const linkOdd, 
          const cudaGaugeField &link,
          int sig, int mu, 
          typename RealTypeId<RealA>::Type coeff, 
          typename RealTypeId<RealA>::Type accumu_coeff,
          dim3 gridDim, dim3 blockDim,
          RealA* shortPEven,  
          RealA* shortPOdd,
          RealA* newOprodEven, 
          RealA* newOprodOdd)
      {
        dim3 halfGridDim(gridDim.x/2,1,1);

        cudaBindTexture(0, siteLink0TexSingle_recon, link.Even_p(), link.Bytes()/2);
        cudaBindTexture(0, siteLink1TexSingle_recon, link.Odd_p(), link.Bytes()/2);   

        if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
          do_side_link_kernel<RealA, RealB, 1, 1, 0><<<halfGridDim, blockDim>>>( P3Even, 
              oprodEven,  oprodOdd,
              linkEven, linkOdd,
              sig, mu, coeff, accumu_coeff,
              shortPOdd,
              newOprodEven, newOprodOdd);

          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);

          do_side_link_kernel<RealA, RealB, 1, 1, 1><<<halfGridDim, blockDim>>>( P3Odd, 
              oprodOdd,  oprodEven,
              linkOdd, linkEven,
              sig, mu, coeff, accumu_coeff,
              shortPEven,
              newOprodOdd, newOprodEven);

        }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
          do_side_link_kernel<RealA, RealB, 1, 0, 0><<<halfGridDim, blockDim>>>( P3Even, 
              oprodEven,  oprodOdd,
              linkEven,  linkOdd,
              sig, mu, coeff, accumu_coeff,
              shortPOdd,
              newOprodEven, newOprodOdd);		

          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);

          do_side_link_kernel<RealA, RealB, 1, 0, 1><<<halfGridDim, blockDim>>>( P3Odd, 
              oprodOdd,  oprodEven,
              linkOdd, linkEven,
              sig, mu, coeff, accumu_coeff,
              shortPEven,
              newOprodOdd, newOprodEven);		

        }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
          do_side_link_kernel<RealA, RealB, 0, 1, 0><<<halfGridDim, blockDim>>>( P3Even,
              oprodEven,  oprodOdd,
              linkEven,  linkOdd,
              sig, mu, coeff, accumu_coeff,
              shortPOdd,
              newOprodEven, newOprodOdd);
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);

          do_side_link_kernel<RealA, RealB, 0, 1, 1><<<halfGridDim, blockDim>>>( P3Odd,
              oprodOdd,  oprodEven,
              linkOdd, linkEven,
              sig, mu, coeff, accumu_coeff,
              shortPEven,
              newOprodOdd, newOprodEven);

        }else{
          do_side_link_kernel<RealA, RealB, 0, 0, 0><<<halfGridDim, blockDim>>>( P3Even,
              oprodEven,  oprodOdd,
              linkEven, linkOdd,
              sig, mu, coeff, accumu_coeff,
              shortPOdd,
              newOprodEven, newOprodOdd);
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);

          do_side_link_kernel<RealA, RealB, 0, 0, 1><<<halfGridDim, blockDim>>>( P3Odd, 
              oprodOdd,  oprodEven,
              linkOdd, linkEven,
              sig, mu, coeff, accumu_coeff,
              shortPEven,
              newOprodOdd, newOprodEven);
        }

        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);    

        return;
      }


    template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit>
      __global__ void
      do_all_link_kernel(
          const RealA* const oprodEven, 
          const RealA* const QprevOdd,
          const RealB* const linkEven, 
          const RealB* const linkOdd,
          int sig, int mu, 
          typename RealTypeId<RealA>::Type coeff, 
          typename RealTypeId<RealA>::Type accumu_coeff,
          RealA* const shortPOdd,
          RealA* const newOprodEven,
          RealA* const newOprodOdd)
      {
        int sid = blockIdx.x * blockDim.x + threadIdx.x;

        int x[4];

        int z1 = sid/X1h;
        int x1h = sid - z1*X1h;
        int z2 = z1/X2;
        x[1] = z1 - z2*X2;
        x[3] = z2/X3;
        x[2] = z2 - x[3]*X3;
        int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
        x[0] = 2*x1h + x1odd;
        int X = 2*sid + x1odd;

        int new_x[4];
        int ad_link_sign=1;
        int ab_link_sign=1;
        int bc_link_sign=1;   

        RealB LINK_W[ArrayLength<RealB>::result];
        RealB LINK_X[ArrayLength<RealB>::result];
        RealB LINK_Y[ArrayLength<RealB>::result];

        RealA COLOR_MAT_W[ArrayLength<RealA>::result]; 
        RealA COLOR_MAT_Y[ArrayLength<RealA>::result]; 
        RealA COLOR_MAT_X[ArrayLength<RealA>::result]; 
        RealA COLOR_MAT_Z[ArrayLength<RealA>::result];


        //            sig
        //         A________B
        //      mu  |      |
        //        D |      |C
        //
        //   A is the current point (sid)
        //

        int point_b, point_c, point_d;
        int ad_link_nbr_idx, ab_link_nbr_idx, bc_link_nbr_idx;
        int mymu;
        int new_mem_idx;
        new_x[0] = x[0];
        new_x[1] = x[1];
        new_x[2] = x[2];
        new_x[3] = x[3];

        if(mu_positive){
          mymu =mu;
          FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mu, X, new_mem_idx);
        }else{
          mymu = OPP_DIR(mu);
          FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(OPP_DIR(mu), X, new_mem_idx);	
        }
        point_d = (new_mem_idx >> 1);

        if (mu_positive){
          ad_link_nbr_idx = point_d;
          reconstructSign(&ad_link_sign, mymu, new_x);
        }else{
          ad_link_nbr_idx = sid;
          reconstructSign(&ad_link_sign, mymu, x);	
        }


        int mysig; 
        if(sig_positive){
          mysig = sig;
          FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, new_mem_idx, new_mem_idx);
        }else{
          mysig = OPP_DIR(sig);
          FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), new_mem_idx, new_mem_idx);	
        }
        point_c = (new_mem_idx >> 1);
        if (mu_positive){
          bc_link_nbr_idx = point_c;	
          reconstructSign(&bc_link_sign, mymu, new_x);
        }

        new_x[0] = x[0];
        new_x[1] = x[1];
        new_x[2] = x[2];
        new_x[3] = x[3];

        if(sig_positive){
          FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, X, new_mem_idx);
        }else{
          FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), X, new_mem_idx);	
        }
        point_b = (new_mem_idx >> 1);
        if (!mu_positive){
          bc_link_nbr_idx = point_b;
          reconstructSign(&bc_link_sign, mymu, new_x);
        }      

        if(sig_positive){
          ab_link_nbr_idx = sid;
          reconstructSign(&ab_link_sign, mysig, x);	
        }else{	
          ab_link_nbr_idx = point_b;
          reconstructSign(&ab_link_sign, mysig, new_x);
        }

        loadMatrixFromField(QprevOdd, point_d, COLOR_MAT_X);

        if (mu_positive){
          loadMatrixFromField(linkOdd, mymu, ad_link_nbr_idx, LINK_Y);
        }else{
          loadMatrixFromField(linkEven, mymu, ad_link_nbr_idx, LINK_Y);
        }

        if(sig_positive){
          if (mu_positive){
            MAT_MUL_MAT(COLOR_MAT_X, LINK_Y, COLOR_MAT_W);
          }else{
            MAT_MUL_ADJ_MAT(COLOR_MAT_X, LINK_Y, COLOR_MAT_W);
          }
        }
        loadMatrixFromField(oprodEven, point_c, COLOR_MAT_Y);



        if (mu_positive){
          loadMatrixFromField(linkEven, mymu, bc_link_nbr_idx, LINK_W);
        }else{
          loadMatrixFromField(linkOdd, mymu, bc_link_nbr_idx, LINK_W);
        }


        MATRIX_PRODUCT(LINK_X, LINK_W, COLOR_MAT_Y, !mu_positive);

        // I can use a pointer to the even and odd link fields 
        // to avoid all the if statements
        if (sig_positive){
          loadMatrixFromField(linkEven, mysig, ab_link_nbr_idx, LINK_W);
        }else{
          loadMatrixFromField(linkOdd, mysig, ab_link_nbr_idx, LINK_W);
        }
        MATRIX_PRODUCT(COLOR_MAT_Y, LINK_W, LINK_X, sig_positive);

        const typename RealTypeId<RealA>::Type & mycoeff = CoeffSign<sig_positive,oddBit>::result*coeff;
        if (sig_positive)
        {	
          MAT_MUL_MAT(LINK_X, COLOR_MAT_W, COLOR_MAT_Z);
          if(oddBit){
            addMatrixToField(COLOR_MAT_Z, sig, sid, -mycoeff, newOprodEven);
          }else{
            addMatrixToField(COLOR_MAT_Z, sig, sid, mycoeff, newOprodEven);
          }
        }

        if (mu_positive)
        {
          MAT_MUL_MAT(COLOR_MAT_Y, COLOR_MAT_X, COLOR_MAT_Z);
          if(oddBit){
            addMatrixToField(COLOR_MAT_Z, mu, point_d, mycoeff, newOprodOdd);
          }else{
            addMatrixToField(COLOR_MAT_Z, mu, point_d, -mycoeff, newOprodOdd);
          }
        }else{
          ADJ_MAT_MUL_ADJ_MAT(COLOR_MAT_X, COLOR_MAT_Y, COLOR_MAT_Z);	
          if(oddBit){
            addMatrixToField(COLOR_MAT_Z, OPP_DIR(mu), sid, -mycoeff, newOprodEven);
          }else{
            addMatrixToField(COLOR_MAT_Z, OPP_DIR(mu), sid, mycoeff, newOprodEven);
          }
        }

        MATRIX_PRODUCT(COLOR_MAT_W, LINK_Y, COLOR_MAT_Y, mu_positive);
        addMatrixToField(COLOR_MAT_W, point_d, accumu_coeff, shortPOdd);
        return;
      }


    template<class RealA, class RealB>
      static void
      all_link_kernel(
          const RealA* const oprodEven, 
          const RealA* const oprodOdd,
          const RealA* const QprevEven, 
          const RealA* const QprevOdd, 
          const RealB* const linkEven, 
          const RealB* const linkOdd, 
          const cudaGaugeField &link,
          int sig, int mu,
          typename RealTypeId<RealA>::Type coeff, 
          typename RealTypeId<RealA>::Type  accumu_coeff,
          dim3 gridDim, dim3 blockDim,
          RealA* const shortPEven, 
          RealA* const shortPOdd,
          RealA* const newOprodEven, 
          RealA* const newOprodOdd)
          {
            dim3 halfGridDim(gridDim.x/2, 1,1);

            cudaBindTexture(0, siteLink0TexSingle_recon, link.Even_p(), link.Bytes()/2);
            cudaBindTexture(0, siteLink1TexSingle_recon, link.Odd_p(), link.Bytes()/2);

            if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){		
              do_all_link_kernel<RealA, RealB, 1, 1, 0><<<halfGridDim, blockDim>>>( 
                  oprodEven,  
                  QprevOdd, 
                  linkEven, linkOdd,
                  sig,  mu,
                  coeff, accumu_coeff,
                  shortPOdd,
                  newOprodEven, newOprodOdd);

              cudaUnbindTexture(siteLink0TexSingle_recon);
              cudaUnbindTexture(siteLink1TexSingle_recon);

              //opposite binding
              cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
              cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);
              do_all_link_kernel<RealA, RealB, 1, 1, 1><<<halfGridDim, blockDim>>>( 
                  oprodOdd,  
                  QprevEven,
                  linkOdd, linkEven,
                  sig,  mu,
                  coeff, accumu_coeff,
                  shortPEven,
                  newOprodOdd, newOprodEven);

            }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){

              do_all_link_kernel<RealA, RealB, 1, 0, 0><<<halfGridDim, blockDim>>>( 
                  oprodEven,   
                  QprevOdd,
                  linkEven, linkOdd,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  shortPOdd,
                  newOprodEven, newOprodOdd);

              cudaUnbindTexture(siteLink0TexSingle_recon);
              cudaUnbindTexture(siteLink1TexSingle_recon);

              //opposite binding
              cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
              cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);

              do_all_link_kernel<RealA, RealB, 1, 0, 1><<<halfGridDim, blockDim>>>( 
                  oprodOdd,  
                  QprevEven, 
                  linkOdd, linkEven,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  shortPEven,
                  newOprodOdd, newOprodEven);

            }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
              do_all_link_kernel<RealA, RealB, 0, 1, 0><<<halfGridDim, blockDim>>>( 
                  oprodEven,  
                  QprevOdd, 
                  linkEven, linkOdd,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  shortPOdd,
                  newOprodEven, newOprodOdd);

              cudaUnbindTexture(siteLink0TexSingle_recon);
              cudaUnbindTexture(siteLink1TexSingle_recon);

              //opposite binding
              cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
              cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);


              do_all_link_kernel<RealA, RealB, 0, 1, 1><<<halfGridDim, blockDim>>>( 
                  oprodOdd,  
                  QprevEven, 
                  linkOdd, linkEven,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  shortPEven,
                  newOprodOdd, newOprodEven);

            }else{
              do_all_link_kernel<RealA, RealB, 0, 0, 0><<<halfGridDim, blockDim>>>( 
                  oprodEven, 
                  QprevOdd, 
                  linkEven, linkOdd,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  shortPOdd,
                  newOprodEven, newOprodOdd);

              cudaUnbindTexture(siteLink0TexSingle_recon);
              cudaUnbindTexture(siteLink1TexSingle_recon);

              //opposite binding
              cudaBindTexture(0, siteLink0TexSingle_recon, link.Odd_p(), link.Bytes()/2);
              cudaBindTexture(0, siteLink1TexSingle_recon, link.Even_p(), link.Bytes()/2);

              do_all_link_kernel<RealA, RealB, 0, 0, 1><<<halfGridDim, blockDim>>>( 
                  oprodOdd,  
                  QprevEven, 
                  linkOdd, linkEven,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  shortPEven,
                  newOprodOdd, newOprodEven);
            }

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

            checkCudaError();		

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
			   longlink_terms( 
					   (float2*)link.Even_p(), (float2*)link.Odd_p(),
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
