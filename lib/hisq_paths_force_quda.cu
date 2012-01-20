#include <read_gauge.h>
#include <gauge_field.h>

#include <hisq_force_quda.h>
#include <hw_quda.h>
#include <hisq_force_macros.h>

#include<utility>

#define LOAD_ANTI_HERMITIAN LOAD_ANTI_HERMITIAN_SINGLE
#define LOAD_MATRIX(src, dir, idx, var) LOAD_MATRIX_12_SINGLE(src, dir, idx, var)


// Disable texture read for now, need to revisit this.
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


#define SIMPLE_MAT_FORCE_TO_MOM(mat, mom, idx, dir, temp_mat, coeff) do { \
  {                                                             \
    float2 AH0, AH1, AH2, AH3, AH4;                             \
    LOAD_ANTI_HERMITIAN(mom, dir, idx, AH);			\
    UNCOMPRESS_ANTI_HERMITIAN(ah, temp_mat);			\
    SCALAR_MULT_ADD_SU3_MATRIX(temp_mat, mat, coeff, link_W);	\
    MAKE_ANTI_HERMITIAN(temp_mat, ah);				\
    WRITE_ANTI_HERMITIAN_SINGLE(mom, dir, idx, AH);		\
  }                                                             \
}while(0)


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
        static const bool compressed=false;
      };

    template<>
      struct ArrayLength<float4>
      {
        static const int result=5;
        static const bool compressed=true;
      };
  


    __device__ void reconstructSign(int* const sign, int dir, int i[4]){
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
    }



    void
      hisq_force_init_cuda(QudaGaugeParam* param)
      {
        static int fermion_force_init_cuda_flag = 0; 

        if (fermion_force_init_cuda_flag){
          return;
        }
        fermion_force_init_cuda_flag=1;
        init_kernel_cuda(param);    
      }




    template<class RealA, class RealB, int oddBit>
      __global__ void 
      do_compute_force_kernel(const RealB* const linkEven, 
                              const RealA* const momMatrixEven,      
                              int sig,
                              RealA* const momEven)
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
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(sig, sid, link_sign, link_W);
        }

        loadMatrixFromField(momMatrixEven, sig, sid, COLOR_MAT_X);
        MAT_MUL_MAT(link_W, color_mat_X, color_mat_W);
   
	typename RealTypeId<RealA>::Type coeff = (oddBit==1) ? -1 : 1;
			 
        SIMPLE_MAT_FORCE_TO_MOM(color_mat_W, momEven, sid, sig, link_W, coeff);
          
        return;
      }

    template<class RealA, int oddBit>
      __global__ void 
      do_one_link_term_kernel(
          const RealA* const oprodEven, 
          int sig, 
          typename RealTypeId<RealA>::Type coeff,
          RealA* const momMatrixEven
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
        new_x[0] = x[0];
        new_x[1] = x[1];
        new_x[2] = x[2];
        new_x[3] = x[3];

        int new_mem_idx;
        int point_b;

        if(GOES_FORWARDS(sig)){
          FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, X, new_mem_idx);
        }else{
          FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), X, new_mem_idx);	
        }
        point_b = (new_mem_idx >> 1); 
        const int & point_a = sid;

        RealA COLOR_MAT_W[ArrayLength<RealA>::result];
        if(GOES_FORWARDS(sig)){
          loadMatrixFromField(oprodEven, sig, point_a, COLOR_MAT_W);
          addMatrixToField(COLOR_MAT_W, sig, point_a, coeff, momMatrixEven);
        }
        return;
      }

    template<class RealA, int oddBit>
      __global__ void 
      do_naik_terms_kernel(
          RealA* const momMatrixEven,
          const RealA* const linkEven,
          const RealA* const linkOdd,
          const RealA* const naikOprodEven,
          const RealA* const naikOprodOdd,
          int sig, typename RealTypeId<RealA>::Type naik_coeff)
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

        RealA COLOR_MAT_T[ArrayLength<RealA>::result];
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

        typename RealTypeId<RealA>::Type my_naik_coeff;


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

          MAT_MUL_MAT(link_Z, color_mat_Z, color_mat_W); // link(d)*link(e)*Naik(c)
          MAT_MUL_MAT(link_Y, color_mat_W, color_mat_V);

          MAT_MUL_MAT(link_Y, color_mat_Y, color_mat_W);  // link(d)*Naik(b)*link(b)
          MAT_MUL_MAT(color_mat_W, link_X, color_mat_U);

          MAT_MUL_MAT(color_mat_X, link_W, color_mat_W); // Naik(a)*link(a)*link(b)
          MAT_MUL_MAT(color_mat_W, link_X, color_mat_T);


          addMatrixToField(COLOR_MAT_T, sig, sid, naik_coeff, momMatrixEven);
          addMatrixToField(COLOR_MAT_U, sig, sid, -naik_coeff, momMatrixEven);
          addMatrixToField(COLOR_MAT_V, sig, sid, naik_coeff, momMatrixEven);
        }

        return;
      }




    template<class RealA>
      void naik_terms(
          RealA* const momMatrixEven,
          RealA* const momMatrixOdd,
          const RealA* const linkEven,
          const RealA* const linkOdd,
          const RealA* const naikOprodEven,
          const RealA* const naikOprodOdd,
          int sig, typename RealTypeId<RealA>::Type naik_coeff,
          dim3 gridDim, dim3 blockDim)
      {

        dim3 halfGridDim(gridDim.x/2,1,1);


        printf("naik_coeff = %lf\n", naik_coeff);

        if(GOES_FORWARDS(sig)){
          // Even half lattice
          do_naik_terms_kernel<RealA,0><<<halfGridDim,blockDim>>>(momMatrixEven,
                                                               linkEven,
                                                               linkOdd,
                                                               naikOprodEven,
                                                               naikOprodOdd,
                                                               sig, naik_coeff);

          // Odd half lattice
          do_naik_terms_kernel<RealA,1><<<halfGridDim,blockDim>>>(momMatrixOdd,
                                                                linkOdd,
                                                                linkEven,
                                                                naikOprodOdd,
                                                                naikOprodEven,
                                                                sig, naik_coeff);
        }
        else {
          printf("sig does not go forward\n");
        }

        return;
      }      
          
        





    template<class RealA>
      static void
      one_link_term(
          RealA* const ForceMatrixEven,   
          RealA* const  ForceMatrixOdd,
          const RealA* const oprodEven, 
          const RealA* const oprodOdd,
          int sig, 
          typename RealTypeId<RealA>::Type coeff, 
          typename RealTypeId<RealA>::Type naik_coeff,
          dim3 gridDim, dim3 blockDim)
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
          RealA* const PmuOdd, 
          RealA* const P3Even,
          const RealA* const QprevOdd, 		
          RealA* const QmuEven, 
          int sig, int mu, 
         typename RealTypeId<RealA>::Type coeff,
          const RealB* const linkEven, 
          const RealB* const linkOdd,
          RealA* const momMatrixEven 
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
        RealA COLOR_MAT_Z[ArrayLength<RealA>::result];


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
        // Store in link_W 
        if(sig_positive){
          //loadMatrixFromField(LINK_W, mysig, ab_link_nbr_idx, linkEven);
          loadMatrixFromField(linkEven, mysig, ab_link_nbr_idx, LINK_W);
        }else{
          //loadMatrixFromField(LINK_W, mysig, ab_link_nbr_idx, linkOdd);
          loadMatrixFromField(linkOdd, mysig, ab_link_nbr_idx, LINK_W);
        }
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(mysig, ab_link_nbr_idx, ab_link_sign, link_W);
        }

        // load the link variable connecting b and c 
        // Store in link_X
        if(mu_positive){
          //loadMatrixFromField(LINK_X, mymu, bc_link_nbr_idx, linkEven);
          loadMatrixFromField(linkEven, mymu, bc_link_nbr_idx, LINK_X);
        }else{ 
          //loadMatrixFromField(LINK_X, mymu, bc_link_nbr_idx, linkOdd);
          loadMatrixFromField(linkOdd, mymu, bc_link_nbr_idx, LINK_X);
        }
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(mymu, bc_link_nbr_idx, bc_link_sign, link_X);
        }




        if(QprevOdd == NULL){
          if(sig_positive){
            //loadMatrixFromField(COLOR_MAT_Y, sig, point_d, oprodOdd);
            loadMatrixFromField(oprodOdd, sig, point_d, COLOR_MAT_Y);
          }else{
            //loadMatrixFromField(COLOR_MAT_Z, OPP_DIR(sig), point_c, oprodEven);
            loadMatrixFromField(oprodEven, OPP_DIR(sig), point_c, COLOR_MAT_Z);
            ADJ_MAT(color_mat_Z, color_mat_Y);
          }
        }else{ // QprevOdd != NULL
          //loadMatrixFromField(COLOR_MAT_Y, point_c, oprodEven);
          loadMatrixFromField(oprodEven, point_c, COLOR_MAT_Y);
        }
       

        MATRIX_PRODUCT(color_mat_W, link_X, color_mat_Y, !mu_positive);
        if(PmuOdd){
          WRITE_MATRIX_18_SINGLE(PmuOdd, point_b, COLOR_MAT_W);
        }

        MATRIX_PRODUCT(color_mat_Y, link_W, color_mat_W, sig_positive);
        WRITE_MATRIX_18_SINGLE(P3Even, sid, COLOR_MAT_Y);


        if(mu_positive){
          // loadMatrixFromField(LINK_Y, mymu, ad_link_nbr_idx, linkOdd);
          loadMatrixFromField(linkOdd, mymu, ad_link_nbr_idx, LINK_Y);
          if(ArrayLength<RealB>::compressed){
            RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, link_Y);
          }
        }else{
          // loadMatrixFromField(LINK_X, mymu, ad_link_nbr_idx, linkEven);
          loadMatrixFromField(linkEven, mymu, ad_link_nbr_idx, LINK_X);
          if(ArrayLength<RealB>::compressed){
            RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, link_X);
          }
          ADJ_MAT(link_X, link_Y);
        }


        if(QprevOdd == NULL){
          if(sig_positive){
            MAT_MUL_MAT(color_mat_W, link_Y, color_mat_Y);
          }
          if(QmuEven){
            ASSIGN_MAT(link_Y, color_mat_X); 
            WRITE_MATRIX_18_SINGLE(QmuEven, sid, COLOR_MAT_X);
          }
        }else{ 
          // loadMatrixFromField(COLOR_MAT_Y, point_d, QprevOdd);
          loadMatrixFromField(QprevOdd, point_d, COLOR_MAT_Y);
          MAT_MUL_MAT(color_mat_Y, link_Y, color_mat_X);
          if(QmuEven){
            WRITE_MATRIX_18_SINGLE(QmuEven, sid, COLOR_MAT_X);
          }
          if(sig_positive){
            MAT_MUL_MAT(color_mat_W, color_mat_X, color_mat_Y);
          }	
        }


        if(sig_positive){
          //addMatrixToField(momMatrixEven, sig, sid, coeff, COLOR_MAT_Y);
          addMatrixToField(COLOR_MAT_Y, sig, sid, coeff, momMatrixEven);
        }

        return;
      }




    template<class RealA, class RealB>
      static void 
      compute_force_kernel(
          const RealA* const momMatrixEven, 
          const RealA* const momMatrixOdd,
          const RealB* const linkEven, 
          const RealB* const linkOdd, 
          cudaGaugeField &cudaSiteLink,
          int sig, dim3 gridDim, dim3 blockDim,
          RealA* const momEven, 
          RealA* const momOdd)
      {
        dim3 halfGridDim(gridDim.x/2, 1, 1);

        cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);
        cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Odd_p(),  cudaSiteLink.Bytes()/2);

        do_compute_force_kernel<RealA, RealB, 0><<<halfGridDim, blockDim>>>(linkEven,
                                                                          momMatrixEven,
                                                                          sig,
                                                                          momEven);

        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);

        cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
        cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);

        do_compute_force_kernel<RealA, RealB, 1><<<halfGridDim, blockDim>>>(linkOdd,
                                                                          momMatrixOdd,
                                                                          sig,
                                                                          momOdd);

        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);

        return;
      }



    template<class RealA, class RealB>
      static void
      middle_link_kernel(
          RealA* const momMatrixEven, 
          RealA* const momMatrixOdd,
          RealA* const PmuEven, // write only  
          RealA* const PmuOdd, // write only
          RealA* const P3Even, // write only   
          RealA* const P3Odd,  // write only
          RealA* const QmuEven,  // write only
          RealA* const QmuOdd,   // write only
          const RealA* const oprodEven, 
          const RealA* const oprodOdd, 
          const RealA* const QprevEven, 
          const RealA* const QprevOdd,
          const RealB* const linkEven, 
          const RealB* const linkOdd, 
          cudaGaugeField &cudaSiteLink,
          int sig, int mu, 
          typename RealTypeId<RealA>::Type coeff,
          dim3 gridDim, dim3 BlockDim)
      {
        dim3 halfGridDim(gridDim.x/2, 1,1);

        cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);
        cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);

        if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){	
          do_middle_link_kernel<RealA, RealB, 1, 1, 0><<<halfGridDim, BlockDim>>>( oprodEven, oprodOdd,
              PmuOdd,  P3Even,
              QprevOdd,
              QmuEven, 
              sig, mu, coeff,
              linkEven, linkOdd,
              momMatrixEven);
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);
          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);

          do_middle_link_kernel<RealA, RealB, 1, 1, 1><<<halfGridDim, BlockDim>>>( oprodOdd, oprodEven,
              PmuEven,  P3Odd,
              QprevEven,
              QmuOdd, 
              sig, mu, coeff,
              linkOdd, linkEven,
              momMatrixOdd);
        }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
          do_middle_link_kernel<RealA, RealB, 1, 0, 0><<<halfGridDim, BlockDim>>>( oprodEven, oprodOdd,
              PmuOdd,  P3Even,
              QprevOdd,
              QmuEven,
              sig, mu, coeff,
              linkEven, linkOdd,
              momMatrixEven);	
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);

          do_middle_link_kernel<RealA, RealB, 1, 0, 1><<<halfGridDim, BlockDim>>>( oprodOdd, oprodEven,
              PmuEven,  P3Odd,
              QprevEven,
              QmuOdd,  
              sig, mu, coeff,
              linkOdd, linkEven,
              momMatrixOdd);

        }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
          do_middle_link_kernel<RealA, RealB, 0, 1, 0><<<halfGridDim, BlockDim>>>( oprodEven, oprodOdd,
              PmuOdd,  P3Even,
              QprevOdd,
              QmuEven, 
              sig, mu, coeff,
              linkEven, linkOdd,
              momMatrixEven);	
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);

          do_middle_link_kernel<RealA, RealB, 0, 1, 1><<<halfGridDim, BlockDim>>>( oprodOdd, oprodEven,
              PmuEven,  P3Odd,
              QprevEven, 
              QmuOdd, 
              sig, mu, coeff,
              linkOdd, linkEven,
              momMatrixOdd);
        }else{
          do_middle_link_kernel<RealA, RealB, 0, 0, 0><<<halfGridDim, BlockDim>>>( oprodEven, oprodOdd,
              PmuOdd, P3Even,
              QprevOdd,
              QmuEven, 
              sig, mu, coeff,
              linkEven, linkOdd,
              momMatrixEven);		

          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);

          do_middle_link_kernel<RealA, RealB, 0, 0, 1><<<halfGridDim, BlockDim>>>( oprodOdd, oprodEven,
              PmuEven,  P3Odd,
              QprevEven,
              QmuOdd,  
              sig, mu, coeff,
              linkOdd, linkEven,
              momMatrixOdd);		
        }
        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);    

        return;
      }



    template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit>
      __global__ void
      do_side_link_kernel(
          const RealA* const P3Even, 
          const RealA* const TempxEven, 
          const RealA* const TempxOdd,
          RealA* const shortPOdd,
          const RealB* const linkEven, 
          const RealB* const linkOdd,
          int sig, int mu, 
          typename RealTypeId<RealA>::Type coeff, 
          typename RealTypeId<RealA>::Type accumu_coeff,
          RealA* const momMatrixEven, 
          RealA* const momMatrixOdd)
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


      //  LOAD_MATRIX_18_SINGLE(P3Even, sid, COLOR_MAT_Y);
        //loadMatrixFromField(COLOR_MAT_Y, sid, P3Even);
        if(mu_positive){
          //loadMatrixFromField(LINK_W, mymu, ad_link_nbr_idx, linkOdd);
          loadMatrixFromField(linkOdd, mymu, ad_link_nbr_idx, LINK_W);
        }else{
          //loadMatrixFromField(LINK_W, mymu, ad_link_nbr_idx, linkEven);
          loadMatrixFromField(linkEven, mymu, ad_link_nbr_idx, LINK_W);
        }
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, link_W);	
        }


        // Should all be inside if (shortPOdd)
        if (shortPOdd){
          MATRIX_PRODUCT(color_mat_W, link_W, color_mat_Y, mu_positive);
          //ADD_MATRIX_18_TO_FIELD(shortPOdd, point_d, accumu_coeff, COLOR_MAT_W);
          // addMatrixToField(shortPOdd, point_d, accumu_coeff, COLOR_MAT_W);
          addMatrixToField(COLOR_MAT_W, point_d, accumu_coeff, shortPOdd);
        }


        mycoeff = CoeffSign<sig_positive,oddBit>::result*coeff;

        if(TempxOdd){
          //LOAD_MATRIX_18_SINGLE(TempxOdd, point_d, COLOR_MAT_X);
          //loadMatrixFromField(COLOR_MAT_X, point_d, TempxOdd);
          loadMatrixFromField(TempxOdd, point_d, COLOR_MAT_X);
          if(mu_positive){
            MAT_MUL_MAT(color_mat_Y, color_mat_X, color_mat_W);

            // Added by J.F.
            if(!oddBit){ mycoeff = -mycoeff; }
            // addMatrixToField(momMatrixOdd, mu, point_d, mycoeff, COLOR_MAT_W);
            addMatrixToField(COLOR_MAT_W, mu, point_d, mycoeff, momMatrixOdd);
          }else{
            ADJ_MAT_MUL_ADJ_MAT(color_mat_X, color_mat_Y, color_mat_W);
            // Added by J.F.
            if(oddBit){ mycoeff = -mycoeff; }
            //addMatrixToField(momMatrixEven, OPP_DIR(mu), sid, mycoeff, COLOR_MAT_W);
            addMatrixToField(COLOR_MAT_W, OPP_DIR(mu), sid, mycoeff, momMatrixEven);
          } 
        }

        if(!TempxOdd){
          if(mu_positive){
            // Added by J.F.
            if(!oddBit){ mycoeff = -mycoeff;}
            // addMatrixToField(momMatrixOdd, mu, point_d, mycoeff, COLOR_MAT_Y);
            addMatrixToField(COLOR_MAT_Y, mu, point_d, mycoeff, momMatrixOdd);
          }else{
            // Added by J.F.
            if(oddBit){ mycoeff = -mycoeff; }
            ADJ_MAT(color_mat_Y, color_mat_W);
            //addMatrixToField(momMatrixEven, OPP_DIR(mu), sid, mycoeff, COLOR_MAT_W);
            addMatrixToField(COLOR_MAT_W, OPP_DIR(mu), sid, mycoeff, momMatrixEven);
          }
        }

        return;
      }




    template<class RealA, class RealB>
      static void
      side_link_kernel(
          RealA* momMatrixEven, 
          RealA* momMatrixOdd,
          RealA* shortPEven,  
          RealA* shortPOdd,
          const RealA* const P3Even, 
          const RealA* const P3Odd, 
          const RealA* const TempxEven, 
          const RealA* const TempxOdd,
          const RealB* const linkEven, 
          const RealB* const linkOdd, 
          cudaGaugeField &cudaSiteLink,
          int sig, int mu, 
          typename RealTypeId<RealA>::Type coeff, 
          typename RealTypeId<RealA>::Type accumu_coeff,
          dim3 gridDim, dim3 blockDim)
      {
        dim3 halfGridDim(gridDim.x/2,1,1);

        cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);
        cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);   

        if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
          do_side_link_kernel<RealA, RealB, 1, 1, 0><<<halfGridDim, blockDim>>>( P3Even, 
              TempxEven,  TempxOdd,
              shortPOdd,
              linkEven, linkOdd,
              sig, mu, coeff, accumu_coeff,
              momMatrixEven, momMatrixOdd);
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);

          do_side_link_kernel<RealA, RealB, 1, 1, 1><<<halfGridDim, blockDim>>>( P3Odd, 
              TempxOdd,  TempxEven,
              shortPEven,
              linkOdd, linkEven,
              sig, mu, coeff, accumu_coeff,
              momMatrixOdd, momMatrixEven);

        }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
          do_side_link_kernel<RealA, RealB, 1, 0, 0><<<halfGridDim, blockDim>>>( P3Even, 
              TempxEven,  TempxOdd,
              shortPOdd,
              linkEven,  linkOdd,
              sig, mu, coeff, accumu_coeff,
              momMatrixEven, momMatrixOdd);		
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);

          do_side_link_kernel<RealA, RealB, 1, 0, 1><<<halfGridDim, blockDim>>>( P3Odd, 
              TempxOdd,  TempxEven,
              shortPEven,
              linkOdd, linkEven,
              sig, mu, coeff, accumu_coeff,
              momMatrixOdd, momMatrixEven);		

        }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
          do_side_link_kernel<RealA, RealB, 0, 1, 0><<<halfGridDim, blockDim>>>( P3Even,
              TempxEven,  TempxOdd,
              shortPOdd,
              linkEven,  linkOdd,
              sig, mu, coeff, accumu_coeff,
              momMatrixEven, momMatrixOdd);
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);

          do_side_link_kernel<RealA, RealB, 0, 1, 1><<<halfGridDim, blockDim>>>( P3Odd,
              TempxOdd,  TempxEven,
              shortPEven,
              linkOdd, linkEven,
              sig, mu, coeff, accumu_coeff,
              momMatrixOdd, momMatrixEven);

        }else{
          do_side_link_kernel<RealA, RealB, 0, 0, 0><<<halfGridDim, blockDim>>>( P3Even,
              TempxEven,  TempxOdd,
              shortPOdd,
              linkEven, linkOdd,
              sig, mu, coeff, accumu_coeff,
              momMatrixEven, momMatrixOdd);
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);

          do_side_link_kernel<RealA, RealB, 0, 0, 1><<<halfGridDim, blockDim>>>( P3Odd, 
              TempxOdd,  TempxEven,
              shortPEven,
              linkOdd, linkEven,
              sig, mu, coeff, accumu_coeff,
              momMatrixOdd, momMatrixEven);
        }

        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);    

        return;
      }


    template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit>
      __global__ void
      do_all_link_kernel(
          RealA* const momMatrixEven,
          RealA* const momMatrixOdd,
          const RealA* const oprodEven, 
          const RealA* const QprevOdd,
          RealA* const shortPOdd,
          int sig, int mu, 
          typename RealTypeId<RealA>::Type coeff, 
          typename RealTypeId<RealA>::Type accumu_coeff,
          const RealB* const linkEven, 
          const RealB* const linkOdd)
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

        //loadMatrixFromField(COLOR_MAT_X, point_d, QprevOdd);
        loadMatrixFromField(QprevOdd, point_d, COLOR_MAT_X);

        if (mu_positive){
          //loadMatrixFromField(LINK_Y, mymu, ad_link_nbr_idx, linkOdd);
          loadMatrixFromField(linkOdd, mymu, ad_link_nbr_idx, LINK_Y);
        }else{
          //loadMatrixFromField(LINK_Y, mymu, ad_link_nbr_idx, linkEven);
          loadMatrixFromField(linkEven, mymu, ad_link_nbr_idx, LINK_Y);
        }
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, link_Y);
        }

        if(sig_positive){
          if (mu_positive){
            MAT_MUL_MAT(color_mat_X, link_Y, color_mat_W);
          }else{
            MAT_MUL_ADJ_MAT(color_mat_X, link_Y, color_mat_W);
          }
        }
        // loadMatrixFromField(COLOR_MAT_Y, point_c, oprodEven);
        loadMatrixFromField(oprodEven, point_c, COLOR_MAT_Y);



        if (mu_positive){
          // loadMatrixFromField(LINK_W, mymu, bc_link_nbr_idx, linkEven);
          loadMatrixFromField(linkEven, mymu, bc_link_nbr_idx, LINK_W);
        }else{
          //loadMatrixFromField(LINK_W, mymu, bc_link_nbr_idx, linkOdd);
          loadMatrixFromField(linkOdd, mymu, bc_link_nbr_idx, LINK_W);
        }
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(mymu, bc_link_nbr_idx, bc_link_sign, link_W);
        }


        MATRIX_PRODUCT(link_X, link_W, color_mat_Y, !mu_positive);

        // I can use a pointer to the even and odd link fields 
        // to avoid all the if statements
        if (sig_positive){
          //loadMatrixFromField(LINK_W, mysig, ab_link_nbr_idx, linkEven);
          loadMatrixFromField(linkEven, mysig, ab_link_nbr_idx, LINK_W);
        }else{
          //loadMatrixFromField(LINK_W, mysig, ab_link_nbr_idx, linkOdd);
          loadMatrixFromField(linkOdd, mysig, ab_link_nbr_idx, LINK_W);
        }
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(mysig, ab_link_nbr_idx, ab_link_sign, link_W);
        }


        MATRIX_PRODUCT(color_mat_Y, link_W, link_X, sig_positive);

        const typename RealTypeId<RealA>::Type & mycoeff = CoeffSign<sig_positive,oddBit>::result*coeff;
        if (sig_positive)
        {	
          MAT_MUL_MAT(link_X, color_mat_W, color_mat_Z);
          if(oddBit){
            //addMatrixToField(momMatrixEven, sig, sid, -mycoeff, COLOR_MAT_Z);
            addMatrixToField(COLOR_MAT_Z, sig, sid, -mycoeff, momMatrixEven);
          }else{
            // addMatrixToField(momMatrixEven, sig, sid, mycoeff, COLOR_MAT_Z);
            addMatrixToField(COLOR_MAT_Z, sig, sid, mycoeff, momMatrixEven);
          }
        }

        if (mu_positive)
        {
          MAT_MUL_MAT(color_mat_Y, color_mat_X, color_mat_Z);
          if(oddBit){
            // addMatrixToField(momMatrixOdd, mu, point_d, mycoeff, COLOR_MAT_Z);
            addMatrixToField(COLOR_MAT_Z, mu, point_d, mycoeff, momMatrixOdd);
          }else{
            // addMatrixToField(momMatrixOdd, mu, point_d, -mycoeff, COLOR_MAT_Z);
            addMatrixToField(COLOR_MAT_Z, mu, point_d, -mycoeff, momMatrixOdd);
          }
        }else{
          ADJ_MAT_MUL_ADJ_MAT(color_mat_X, color_mat_Y, color_mat_Z);	
          if(oddBit){
            // addMatrixToField(momMatrixEven, OPP_DIR(mu), sid, -mycoeff, COLOR_MAT_Z);
            addMatrixToField(COLOR_MAT_Z, OPP_DIR(mu), sid, -mycoeff, momMatrixEven);
          }else{
            // addMatrixToField(momMatrixEven, OPP_DIR(mu), sid, mycoeff, COLOR_MAT_Z);
            addMatrixToField(COLOR_MAT_Z, OPP_DIR(mu), sid, mycoeff, momMatrixEven);
          }
        }

        MATRIX_PRODUCT(color_mat_W, link_Y, color_mat_Y, mu_positive);
        // addMatrixToField(shortPOdd, point_d, accumu_coeff, COLOR_MAT_W);
        addMatrixToField(COLOR_MAT_W, point_d, accumu_coeff, shortPOdd);
        return;
      }


    template<class RealA, class RealB>
      static void
      all_link_kernel(
          RealA* const momMatrixEven, 
          RealA* const momMatrixOdd,
          RealA* const shortPEven, 
          RealA* const shortPOdd,
          const RealA* const oprodEven, 
          const RealA* const oprodOdd,
          const RealA* const QprevEven, 
          const RealA* const QprevOdd, 
          const RealB* const linkEven, 
          const RealB* const linkOdd, 
          cudaGaugeField &cudaSiteLink,
          int sig, int mu,
          typename RealTypeId<RealA>::Type coeff, 
          typename RealTypeId<RealA>::Type  accumu_coeff,
          dim3 gridDim, dim3 blockDim)
          {
            dim3 halfGridDim(gridDim.x/2, 1,1);

            cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);
            cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);

            if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){		
              do_all_link_kernel<RealA, RealB, 1, 1, 0><<<halfGridDim, blockDim>>>( 
                  momMatrixEven, momMatrixOdd,
                  oprodEven,  
                  QprevOdd, 
                  shortPOdd,
                  sig,  mu,
                  coeff, accumu_coeff,
                  linkEven, linkOdd);
              cudaUnbindTexture(siteLink0TexSingle_recon);
              cudaUnbindTexture(siteLink1TexSingle_recon);

              //opposite binding
              cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
              cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);
              do_all_link_kernel<RealA, RealB, 1, 1, 1><<<halfGridDim, blockDim>>>( 
                  momMatrixOdd, momMatrixEven,
                  oprodOdd,  
                  QprevEven,
                  shortPEven,
                  sig,  mu,
                  coeff, accumu_coeff,
                  linkOdd, linkEven);

            }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){

              do_all_link_kernel<RealA, RealB, 1, 0, 0><<<halfGridDim, blockDim>>>( 
                  momMatrixEven, momMatrixOdd,
                  oprodEven,   
                  QprevOdd,
                  shortPOdd,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  linkEven, linkOdd);

              cudaUnbindTexture(siteLink0TexSingle_recon);
              cudaUnbindTexture(siteLink1TexSingle_recon);

              //opposite binding
              cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
              cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);

              do_all_link_kernel<RealA, RealB, 1, 0, 1><<<halfGridDim, blockDim>>>( 
                  momMatrixOdd, momMatrixEven,
                  oprodOdd,  
                  QprevEven, 
                  shortPEven,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  linkOdd, linkEven);

            }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
              do_all_link_kernel<RealA, RealB, 0, 1, 0><<<halfGridDim, blockDim>>>( 
                  momMatrixEven, momMatrixOdd,
                  oprodEven,  
                  QprevOdd, 
                  shortPOdd,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  linkEven, linkOdd);
              cudaUnbindTexture(siteLink0TexSingle_recon);
              cudaUnbindTexture(siteLink1TexSingle_recon);

              //opposite binding
              cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
              cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);


              do_all_link_kernel<RealA, RealB, 0, 1, 1><<<halfGridDim, blockDim>>>( 
                  momMatrixOdd, momMatrixEven,
                  oprodOdd,  
                  QprevEven, 
                  shortPEven,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  linkOdd, linkEven);
            }else{
              do_all_link_kernel<RealA, RealB, 0, 0, 0><<<halfGridDim, blockDim>>>( 
                  momMatrixEven, momMatrixOdd,
                  oprodEven, 
                  QprevOdd, 
                  shortPOdd,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  linkEven, linkOdd);

              cudaUnbindTexture(siteLink0TexSingle_recon);
              cudaUnbindTexture(siteLink1TexSingle_recon);

              //opposite binding
              cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);
              cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);

              do_all_link_kernel<RealA, RealB, 0, 0, 1><<<halfGridDim, blockDim>>>( 
                  momMatrixOdd, momMatrixEven,
                  oprodOdd,  
                  QprevEven, 
                  shortPEven,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  linkOdd, linkEven);
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
      do_hisq_force_cuda( PathCoefficients<Real> act_path_coeff,
                          cudaGaugeField &cudaOprod, 
                          cudaGaugeField &cudaSiteLink, 
                          cudaGaugeField &cudaForceMatrix, 
                          FullMatrix tempmat[4], 
                          FullMatrix tempCmat[2], 
                          QudaGaugeParam* param)
      {


        Real coeff;
        Real OneLink, Lepage, Naik, FiveSt, ThreeSt, SevenSt;
        Real mLepage, mFiveSt, mThreeSt;



        OneLink = act_path_coeff.one;
        Naik    = act_path_coeff.naik;
        ThreeSt = act_path_coeff.three; mThreeSt = -ThreeSt;
        FiveSt  = act_path_coeff.five; mFiveSt  = -FiveSt;
        SevenSt = act_path_coeff.seven; 
        Lepage  = act_path_coeff.lepage; mLepage  = -Lepage;


        int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
        dim3 blockDim(BLOCK_DIM,1,1);
        dim3 gridDim(volume/blockDim.x, 1, 1);

        for(int sig=0; sig<8; sig++){
          for(int mu=0; mu<8; mu++){
            if ( (mu == sig) || (mu == OPP_DIR(sig))){
              continue;
            }
            //3-link
            //Kernel A: middle link

            int new_sig;
            if(GOES_BACKWARDS(sig)){ new_sig = OPP_DIR(sig); }else{ new_sig = sig; }

            middle_link_kernel( 
                (RealA*)cudaForceMatrix.Even_p(), (RealA*)cudaForceMatrix.Odd_p(),
                (RealA*)Pmu.even.data, (RealA*)Pmu.odd.data,                               // write only
                (RealA*)P3.even.data, (RealA*)P3.odd.data,                                 // write only
                (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,                               // write only     
                (RealA*)cudaOprod.Even_p(), (RealA*)cudaOprod.Odd_p(),                     // read only
                (RealA*)NULL,         (RealA*)NULL,                                        // read only
                (RealB*)cudaSiteLink.Even_p(), (RealB*)cudaSiteLink.Odd_p(), 
                cudaSiteLink,  // read only
                sig, mu, mThreeSt,
                gridDim, blockDim);

            checkCudaError();

            for(int nu=0; nu < 8; nu++){
              if (nu == sig || nu == OPP_DIR(sig)
                  || nu == mu || nu == OPP_DIR(mu)){
                continue;
              }

              //5-link: middle link
              //Kernel B
              middle_link_kernel( 
                  (RealA*)cudaForceMatrix.Even_p(), (RealA*)cudaForceMatrix.Odd_p(),
                  (RealA*)Pnumu.even.data, (RealA*)Pnumu.odd.data,  // write only
                  (RealA*)P5.even.data, (RealA*)P5.odd.data,        // write only
                  (RealA*)Qnumu.even.data, (RealA*)Qnumu.odd.data,  // write only
                  (RealA*)Pmu.even.data, (RealA*)Pmu.odd.data,      // read only
                  (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,      // read only
                  (RealB*)cudaSiteLink.Even_p(), (RealB*)cudaSiteLink.Odd_p(), 
                  cudaSiteLink, 
                  sig, nu, FiveSt,
                  gridDim, blockDim);

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
                    (RealA*)cudaForceMatrix.Even_p(), (RealA*)cudaForceMatrix.Odd_p(),
                    (RealA*)P5.even.data, (RealA*)P5.odd.data, 
                    (RealA*)Pnumu.even.data, (RealA*)Pnumu.odd.data,
                    (RealA*)Qnumu.even.data, (RealA*)Qnumu.odd.data,
                    (RealB*)cudaSiteLink.Even_p(), (RealB*)cudaSiteLink.Odd_p(), 
                    cudaSiteLink,
                    sig, rho, SevenSt, coeff,
                    gridDim, blockDim);
                checkCudaError();

              }//rho  		


              //5-link: side link
              if(ThreeSt != 0)coeff = FiveSt/ThreeSt; else coeff = 0;
              side_link_kernel(
                  (RealA*)cudaForceMatrix.Even_p(), (RealA*)cudaForceMatrix.Odd_p(),
                  (RealA*)P3.even.data, (RealA*)P3.odd.data,    // write
                  (RealA*)P5.even.data, (RealA*)P5.odd.data,    // read only
                  (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,  // read only
                  (RealB*)cudaSiteLink.Even_p(), (RealB*)cudaSiteLink.Odd_p(), 
                  cudaSiteLink,
                  sig, nu, mFiveSt, coeff,
                  gridDim, blockDim);
              checkCudaError();

            } //nu 

            //lepage
            middle_link_kernel( 
                (RealA*)cudaForceMatrix.Even_p(), (RealA*)cudaForceMatrix.Odd_p(),
                (RealA*)NULL, (RealA*)NULL,                      // write only
                (RealA*)P5.even.data, (RealA*)P5.odd.data,       // write only
                (RealA*)NULL, (RealA*)NULL,                      // write only
                (RealA*)Pmu.even.data, (RealA*)Pmu.odd.data,     // read only
                (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,     // read only
                (RealB*)cudaSiteLink.Even_p(), (RealB*)cudaSiteLink.Odd_p(), 
                cudaSiteLink, 
                sig, mu, Lepage,
                gridDim, blockDim);
            checkCudaError();		

            if(ThreeSt != 0)coeff = Lepage/ThreeSt ; else coeff = 0;

            side_link_kernel(
                (RealA*)cudaForceMatrix.Even_p(), (RealA*)cudaForceMatrix.Odd_p(),
                (RealA*)P3.even.data, (RealA*)P3.odd.data,           // write only
                (RealA*)P5.even.data, (RealA*)P5.odd.data,           // read only
                (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,         // read only
                (RealB*)cudaSiteLink.Even_p(), (RealB*)cudaSiteLink.Odd_p(), 
                cudaSiteLink,
                sig, mu, mLepage, coeff,
                gridDim, blockDim);
            checkCudaError();		


            //3-link side link
            coeff=0.;
            side_link_kernel(
                (RealA*)cudaForceMatrix.Even_p(), (RealA*)cudaForceMatrix.Odd_p(),
                (RealA*)NULL, (RealA*)NULL,                // write
                (RealA*)P3.even.data, (RealA*)P3.odd.data, // read only
                (RealA*)NULL, (RealA*)NULL,                // read only
                (RealB*)cudaSiteLink.Even_p(), (RealB*)cudaSiteLink.Odd_p(), 
                cudaSiteLink,
                sig, mu, ThreeSt, coeff,
                gridDim, blockDim);

            checkCudaError();			    

          }//mu
        }//sig


        for(int sig=0; sig<8; ++sig){
          if(GOES_FORWARDS(sig)){
            one_link_term(
                (RealA*)cudaForceMatrix.Even_p(), (RealA*)cudaForceMatrix.Odd_p(),
                (RealA*)cudaOprod.Even_p(), (RealA*)cudaOprod.Odd_p(),
                sig, OneLink, 0.0,
                gridDim, blockDim);
          } // GOES_FORWARDS(sig)
          checkCudaError();
        }

        return; 
   }


   template<class RealA>
     __global__ void
     do_rewrite_oprod_kernel(RealA* const momMatrix,
                             const RealA* const Oprod,
                             int sig)
     {
       if(GOES_FORWARDS(sig)){
         int sid = blockIdx.x*blockDim.x + threadIdx.x;
         RealA COLOR_MAT_W[ArrayLength<RealA>::result];
         //loadMatrixFromField(COLOR_MAT_W, sig, sid, Oprod);
         loadMatrixFromField(Oprod, sig, sid, COLOR_MAT_W);
         //storeMatrixToField(momMatrix, sig, sid, COLOR_MAT_W);
         storeMatrixToField(COLOR_MAT_W, sig, sid, momMatrix);
       }
       return;
     }

   // rewrite this function so it multiplies the odd matrices 
   // by a minus sign -> easy to do

   template<class RealA>
     static void 
     rewrite_oprod_kernel(RealA* const momMatrixEven,
                          RealA* const momMatrixOdd,
                          const RealA* const OprodEven,
                          const RealA* const OprodOdd,
                          int sig, dim3 gridDim, dim3 blockDim)
     {
       dim3 halfGridDim(gridDim.x/2, 1, 1);
   
       do_rewrite_oprod_kernel<RealA><<<halfGridDim, blockDim>>>(momMatrixEven, OprodEven, sig);
       do_rewrite_oprod_kernel<RealA><<<halfGridDim, blockDim>>>(momMatrixOdd, OprodOdd, sig);
       return;
     }


   template<class RealA>
     void rewriteOprod(cudaGaugeField &cudaForceMatrix, cudaGaugeField &cudaOprod, QudaGaugeParam* param)
     {
       int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
       dim3 blockDim(BLOCK_DIM,1,1);
       dim3 gridDim(volume/blockDim.x, 1, 1);
       for(int sig=0; sig<4; ++sig){
         rewrite_oprod_kernel<RealA>((RealA*)cudaForceMatrix.Even_p(), (RealA*)cudaForceMatrix.Odd_p(),
                                     (RealA*)cudaOprod.Even_p(), 
                                     (RealA*)cudaOprod.Odd_p(),
                                     sig, gridDim, blockDim);  
       }
        
       return;   
     }


   void rewriteOprodCuda(cudaGaugeField &cudaForceMatrix, cudaGaugeField &cudaOprod, QudaGaugeParam* param)
   {
     rewriteOprod<float2>(cudaForceMatrix, cudaOprod, param);
     return;     
   }


   template<class RealA>
   __global__ void 
   do_rescaleHalfField(RealA* const halfField, typename  RealTypeId<RealA>::Type coeff)
   {

     // Note: 
     // At the moment, this will only work for float2, double2 
     RealA matrix[ArrayLength<RealA>::result];    
     int sid = blockIdx.x*blockDim.x + threadIdx.x; 
     for(int sig=0; sig<4; ++sig){
      // loadMatrixFromField(matrix, sig, sid, halfField);
       loadMatrixFromField(halfField, sig, sid, matrix);
       for(int i=0; i<ArrayLength<RealA>::result; ++i){
        matrix[i].x = coeff*matrix[i].x;
        matrix[i].y = coeff*matrix[i].y;
       }
       //storeMatrixToField(halfField, sig, sid, matrix);
       storeMatrixToField(matrix, sig, sid, halfField);
     }
     
     return;
   }



   template<class RealA>
    static void rescaleHalfField(RealA* const halfField, const QudaGaugeParam& param, typename RealTypeId<RealA>::Type coeff)
     {
        int volume = param.X[0]*param.X[1]*param.X[2]*param.X[3];
        dim3 blockDim(BLOCK_DIM,1,1);
        dim3 gridDim(volume/blockDim.x, 1, 1);
        dim3 halfGridDim(gridDim.x/2, 1, 1);
        do_rescaleHalfField<RealA><<<halfGridDim, blockDim>>>(halfField,coeff);
        return;
     }

   void rescaleHalfFieldCuda(cudaGaugeField &cudaField, const QudaGaugeParam& param, int oddBit, double coeff)
   {
     if(oddBit){
       rescaleHalfField<float2>((float2*)cudaField.Odd_p(), param, coeff);   
     }else{
       rescaleHalfField<float2>((float2*)cudaField.Even_p(), param, coeff);   
     } 
     return; 
   }

#undef Pmu
#undef Pnumu
#undef P3
#undef P5
#undef Qmu
#undef Qnumu


   void hisq_contract_cuda(cudaGaugeField &cudaOprod,
													 cudaGaugeField &cudaLink,
													 QudaGaugeParam* param,
													 cudaGaugeField &cudaForce)
  {

    int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
    dim3 blockDim(BLOCK_DIM,1,1);
    dim3 gridDim(volume/blockDim.x, 1, 1);
    
    for(int sig=0; sig<4; sig++){
          if(GOES_FORWARDS(sig)){
            compute_force_kernel( 
                (float2*)cudaOprod.Even_p(), (float2*)cudaOprod.Odd_p(),
                (float2*)cudaLink.Even_p(), (float2*)cudaLink.Odd_p(), 
                cudaLink,
                sig, gridDim, blockDim,
                (float2*)cudaForce.Even_p(), (float2*)cudaForce.Odd_p());
          } 
        }
    return;
  }





   void hisq_naik_force_cuda(void* path_coeff_array,
                             cudaGaugeField &cudaOprod,
                             cudaGaugeField &cudaLink,
                             QudaGaugeParam* param,
                             cudaGaugeField &cudaForce)
   {
     int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
     dim3 blockDim(BLOCK_DIM,1,1);
     dim3 gridDim(volume/blockDim.x, 1, 1);

     if(param->cuda_prec == QUDA_DOUBLE_PRECISION){
       for(int sig=0; sig<4; ++sig){
         naik_terms((double2*)cudaForce.Even_p(), (double2*)cudaForce.Odd_p(), 
             (double2*)cudaLink.Even_p(), (double2*)cudaLink.Odd_p(),
             (double2*)cudaOprod.Even_p(), (double2*)cudaOprod.Odd_p(),
             sig, ((double*)path_coeff_array)[1], 
             gridDim, blockDim);
       }
     }else if(param->cuda_prec == QUDA_SINGLE_PRECISION){
       for(int sig=0; sig<4; ++sig){
         naik_terms((float2*)cudaForce.Even_p(), (float2*)cudaForce.Odd_p(), 
             (float2*)cudaLink.Even_p(), (float2*)cudaLink.Odd_p(),
             (float2*)cudaOprod.Even_p(), (float2*)cudaOprod.Odd_p(),
             sig, ((float*)path_coeff_array)[1], 
             gridDim, blockDim);
       }
     }else{
       errorQuda("Unsupported precision\n");
     }

     return;
   }





    void
      hisq_force_cuda(void* path_coeff_array,
                      cudaGaugeField &cudaOprod, 
                      cudaGaugeField &cudaSiteLink, 
                      QudaGaugeParam* param,
                      cudaGaugeField &cudaForceMatrix)
      {

        FullMatrix tempmat[4];
        for(int i=0; i<4; i++){
          tempmat[i]  = createMatQuda(param->X, param->cuda_prec);
        }

        FullMatrix tempCompmat[2];
        for(int i=0; i<2; i++){
          tempCompmat[i] = createMatQuda(param->X, param->cuda_prec);
        }	


        if (param->cuda_prec == QUDA_DOUBLE_PRECISION){
          errorQuda("Double precision not supported\n");
        }else{	

          PathCoefficients<float> act_path_coeff;
          act_path_coeff.one    = ((float*)path_coeff_array)[0];
          act_path_coeff.naik   = ((float*)path_coeff_array)[1];
          act_path_coeff.three  = ((float*)path_coeff_array)[2];
          act_path_coeff.five   = ((float*)path_coeff_array)[3];
          act_path_coeff.seven  = ((float*)path_coeff_array)[4];
          act_path_coeff.lepage = ((float*)path_coeff_array)[5];

          do_hisq_force_cuda<float,float2,float2>( act_path_coeff,
                                                   cudaOprod,
                                                   cudaSiteLink, cudaForceMatrix, tempmat, tempCompmat, param);
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
