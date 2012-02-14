#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <gauge_field.h>

#include "quda_matrix.h"
#include "svd_quda.h"



#define HISQ_UNITARIZE_PI 3.14159265358979323846
#define HISQ_UNITARIZE_PI23 HISQ_UNITARIZE_PI*2.0/3.0

// constants - File scope only
__constant__ double DEV_HISQ_UNITARIZE_EPS;
__constant__ double DEV_HISQ_FORCE_FILTER;
__constant__ double DEV_MAX_DET_ERROR;
__constant__ bool DEV_REUNIT_ALLOW_SVD;
__constant__ bool DEV_REUNIT_SVD_ONLY;
__constant__ double DEV_REUNIT_SVD_REL_ERROR;
__constant__ double DEV_REUNIT_SVD_ABS_ERROR;

static double HOST_HISQ_UNITARIZE_EPS;
static double HOST_HISQ_FORCE_FILTER;
static double HOST_MAX_DET_ERROR;
static bool   HOST_REUNIT_ALLOW_SVD;
static bool   HOST_REUNIT_SVD_ONLY;
static double HOST_REUNIT_SVD_REL_ERROR;
static double HOST_REUNIT_SVD_ABS_ERROR;

 
namespace hisq{
  namespace fermion_force{

     template<class Cmplx> 
     __host__ __device__ 	
     void printLink(Matrix<Cmplx,3>& link);



      void set_unitarize_force_constants(double unitarize_eps, double hisq_force_filter, double max_det_error, 
					bool allow_svd, bool svd_only,
					double svd_rel_error, double svd_abs_error)
      {

	// not_set is only initialised once
	static bool not_set=true;
		
	if(not_set){
          cudaMemcpyToSymbol("DEV_HISQ_UNITARIZE_EPS", &unitarize_eps, sizeof(double));
          cudaMemcpyToSymbol("DEV_HISQ_FORCE_FILTER", &hisq_force_filter, sizeof(double));
          cudaMemcpyToSymbol("DEV_MAX_DET_ERROR", &max_det_error, sizeof(double));
	  cudaMemcpyToSymbol("DEV_REUNIT_ALLOW_SVD", &allow_svd, sizeof(bool));
          cudaMemcpyToSymbol("DEV_REUNIT_SVD_ONLY", &svd_only, sizeof(bool));
	  cudaMemcpyToSymbol("DEV_REUNIT_SVD_REL_ERROR", &svd_rel_error, sizeof(double));
          cudaMemcpyToSymbol("DEV_REUNIT_SVD_ABS_ERROR", &svd_abs_error, sizeof(double));
	
	  HOST_HISQ_UNITARIZE_EPS = unitarize_eps;
          HOST_HISQ_FORCE_FILTER = hisq_force_filter;
          HOST_MAX_DET_ERROR = max_det_error;     
	  HOST_REUNIT_ALLOW_SVD = allow_svd;
          HOST_REUNIT_SVD_ONLY = svd_only;
	  HOST_REUNIT_SVD_REL_ERROR = svd_rel_error;
          HOST_REUNIT_SVD_ABS_ERROR = svd_abs_error;

          not_set = false;
	}
        checkCudaError();
	return;
      }


    template<class Real>
     class DerivativeCoefficients{
       private:
         Real b[6]; 
         __device__ __host__       
         Real computeC00(const Real &, const Real &, const Real &);
         __device__ __host__
         Real computeC01(const Real &, const Real &, const Real &);
         __device__ __host__
         Real computeC02(const Real &, const Real &, const Real &);
         __device__ __host__
         Real computeC11(const Real &, const Real &, const Real &);
         __device__ __host__
         Real computeC12(const Real &, const Real &, const Real &);
         __device__ __host__
         Real computeC22(const Real &, const Real &, const Real &);

       public:
         __device__ __host__ void set(const Real & u, const Real & v, const Real & w);
         __device__ __host__
         Real getB00() const { return b[0]; }
         __device__ __host__
         Real getB01() const { return b[1]; }
         __device__ __host__
         Real getB02() const { return b[2]; }
         __device__ __host__
         Real getB11() const { return b[3]; }
         __device__ __host__
         Real getB12() const { return b[4]; }
         __device__ __host__
         Real getB22() const { return b[5]; }
     };

    template<class Real>
      __device__ __host__
      Real DerivativeCoefficients<Real>::computeC00(const Real & u, const Real & v, const Real & w){
        Real result =   -pow(w,3)*pow(u,6)
          + 3*v*pow(w,3)*pow(u,4)
          + 3*pow(v,4)*w*pow(u,4)
          -   pow(v,6)*pow(u,3)
          - 4*pow(w,4)*pow(u,3)
          - 12*pow(v,3)*pow(w,2)*pow(u,3)
          + 16*pow(v,2)*pow(w,3)*pow(u,2)
          + 3*pow(v,5)*w*pow(u,2)
          - 8*v*pow(w,4)*u
          - 3*pow(v,4)*pow(w,2)*u
          + pow(w,5)
          + pow(v,3)*pow(w,3);

        return result;
      }

    template<class Real>
      __device__ __host__
      Real DerivativeCoefficients<Real>::computeC01(const Real & u, const Real & v, const Real & w){
        Real result =  - pow(w,2)*pow(u,7)
          - pow(v,2)*w*pow(u,6)
          + pow(v,4)*pow(u,5)   // This was corrected!
          + 6*v*pow(w,2)*pow(u,5)
          - 5*pow(w,3)*pow(u,4)    // This was corrected!
          - pow(v,3)*w*pow(u,4)
          - 2*pow(v,5)*pow(u,3)
          - 6*pow(v,2)*pow(w,2)*pow(u,3)
          + 10*v*pow(w,3)*pow(u,2)
          + 6*pow(v,4)*w*pow(u,2)
          - 3*pow(w,4)*u
          - 6*pow(v,3)*pow(w,2)*u
          + 2*pow(v,2)*pow(w,3);
        return result;
      }

    template<class Real>
      __device__ __host__
      Real DerivativeCoefficients<Real>::computeC02(const Real & u, const Real & v, const Real & w){
        Real result =   pow(w,2)*pow(u,5)
          + pow(v,2)*w*pow(u,4)
          - pow(v,4)*pow(u,3)
          - 4*v*pow(w,2)*pow(u,3)
          + 4*pow(w,3)*pow(u,2)
          + 3*pow(v,3)*w*pow(u,2)
          - 3*pow(v,2)*pow(w,2)*u
          + v*pow(w,3);
        return result;
      }

    template<class Real>
      __device__ __host__
      Real DerivativeCoefficients<Real>::computeC11(const Real & u, const Real & v, const Real & w){
        Real result = - w*pow(u,8)
          - pow(v,2)*pow(u,7)
          + 7*v*w*pow(u,6)
          + 4*pow(v,3)*pow(u,5)
          - 5*pow(w,2)*pow(u,5)
          - 16*pow(v,2)*w*pow(u,4)
          - 4*pow(v,4)*pow(u,3)
          + 16*v*pow(w,2)*pow(u,3)
          - 3*pow(w,3)*pow(u,2)
          + 12*pow(v,3)*w*pow(u,2)
          - 12*pow(v,2)*pow(w,2)*u
          + 3*v*pow(w,3);
        return result;
      }

    template<class Real>
      __device__ __host__
      Real DerivativeCoefficients<Real>::computeC12(const Real & u, const Real & v, const Real & w){
        Real result =  w*pow(u,6)
          + pow(v,2)*pow(u,5) // Fixed this!
          - 5*v*w*pow(u,4)  // Fixed this!
          - 2*pow(v,3)*pow(u,3)
          + 4*pow(w,2)*pow(u,3)
          + 6*pow(v,2)*w*pow(u,2)
          - 6*v*pow(w,2)*u
          + pow(w,3);
        return result;
      }

    template<class Real>
      __device__ __host__
      Real DerivativeCoefficients<Real>::computeC22(const Real & u, const Real & v, const Real & w){
        Real result = - w*pow(u,4)
          - pow(v,2)*pow(u,3)
          + 3*v*w*pow(u,2)
          - 3*pow(w,2)*u;
        return result;
      }

    template <class Real>
      __device__ __host__
      void  DerivativeCoefficients<Real>::set(const Real & u, const Real & v, const Real & w){
        const Real & denominator = 2.0*pow(w*(u*v-w),3); 
        b[0] = computeC00(u,v,w)/denominator;
        b[1] = computeC01(u,v,w)/denominator;
        b[2] = computeC02(u,v,w)/denominator;
        b[3] = computeC11(u,v,w)/denominator;
        b[4] = computeC12(u,v,w)/denominator;
        b[5] = computeC22(u,v,w)/denominator;
        return;
      }


    template<class Cmplx>
      __device__ __host__
      void accumBothDerivatives(Matrix<Cmplx,3>* result, const Matrix<Cmplx,3> & left, const Matrix<Cmplx,3> & right, const Matrix<Cmplx,3> & outer_prod)
      {
        const typename RealTypeId<Cmplx>::Type temp = 2.0*getTrace(left*outer_prod).x;
        for(int k=0; k<3; ++k){
          for(int l=0; l<3; ++l){
            // Need to write it this way to get it to work 
            // on the CPU. Not sure why.
            result->operator()(k,l).x += temp*right(k,l).x;
            result->operator()(k,l).y += temp*right(k,l).y;
          }
        }
        return;
      }


    template<class Cmplx>
      __device__ __host__
      void accumDerivatives(Matrix<Cmplx,3>* result, const Matrix<Cmplx,3> & left, const Matrix<Cmplx,3> & right, const Matrix<Cmplx,3> & outer_prod)
      {
        Cmplx temp = getTrace(left*outer_prod);
        for(int k=0; k<3; ++k){
          for(int l=0; l<3; ++l){
            result->operator()(k,l) = temp*right(k,l);
          }
        }
        return;
      }


    template<class T>
      __device__ __host__
    T getAbsMin(const T* const array, int size){
      T min = fabs(array[0]);
      for(int i=1; i<size; ++i){
        T abs_val = fabs(array[i]);
        if((abs_val) < min){ min = abs_val; }   
      }
      return min;
    }


	  template<class Real>
    __device__ __host__
    inline bool checkAbsoluteError(Real a, Real b, Real epsilon)
	  {
			if( fabs(a-b) <  epsilon) return true;
		 	return false;
		}


    template<class Real>
    __device__ __host__ 
    inline bool checkRelativeError(Real a, Real b, Real epsilon)
    {
      if( fabs((a-b)/b)  < epsilon ) return true;
			return false;
    }
    



    // Compute the reciprocal square root of the matrix q
    // Also modify q if the eigenvalues are dangerously small.
    template<class Cmplx> 
      __device__  __host__ 
      void reciprocalRoot(Matrix<Cmplx,3>* res, DerivativeCoefficients<typename RealTypeId<Cmplx>::Type>* deriv_coeffs, 
								typename RealTypeId<Cmplx>::Type f[3], Matrix<Cmplx,3> & q, int *unitarization_failed){

        Matrix<Cmplx,3> qsq, tempq;

        typename RealTypeId<Cmplx>::Type c[3];
        typename RealTypeId<Cmplx>::Type g[3];

#ifdef __CUDA_ARCH__
#define REUNIT_SVD_ONLY DEV_REUNIT_SVD_ONLY
#else
#define REUNIT_SVD_ONLY HOST_REUNIT_SVD_ONLY
#endif
	if(!REUNIT_SVD_ONLY){
		qsq = q*q;
		tempq = qsq*q;

		c[0] = getTrace(q).x;
		c[1] = getTrace(qsq).x/2.0;
		c[2] = getTrace(tempq).x/3.0;

		g[0] = g[1] = g[2] = c[0]/3.;
		typename RealTypeId<Cmplx>::Type r,s,theta;
		s = c[1]/3. - c[0]*c[0]/18;
		r = c[2]/2. - (c[0]/3.)*(c[1] - c[0]*c[0]/9.);

#ifdef __CUDA_ARCH__
#define HISQ_UNITARIZE_EPS DEV_HISQ_UNITARIZE_EPS
#else
#define HISQ_UNITARIZE_EPS HOST_HISQ_UNITARIZE_EPS
#endif

		typename RealTypeId<Cmplx>::Type cosTheta = r/sqrt(s*s*s);
		if(fabs(s) < HISQ_UNITARIZE_EPS){
			cosTheta = 1.;
			s = 0.0; 
		}
		if(fabs(cosTheta)>1.0){ r>0 ? theta=0.0 : theta=HISQ_UNITARIZE_PI/3.0; }
		else{ theta = acos(cosTheta)/3.0; }

		s = 2.0*sqrt(s);
		for(int i=0; i<3; ++i){
			g[i] += s*cos(theta + (i-1)*HISQ_UNITARIZE_PI23);
		}

	} // !REUNIT_SVD_ONLY?

	//
	// Compare the product of the eigenvalues computed thus far to the 
	// absolute value of the determinant. 
	// If the error is greater than some predefined value 
	// then recompute the eigenvalues using a singular-value decomposition.
	// We can compare either the absolute or relative error on the determinant. 
	// and the choice of error is a run-time parameter.
	// Note that this particular calculation contains multiple branches, 
	// so it doesn't appear to be particularly well-suited to the GPU 
	// programming model. However, the analytic calculation of the 
	// unitarization is extremely fast, and if the SVD routine is not called 
	// too often, we expect reasonable performance.
	//

#ifdef __CUDA_ARCH__
#define REUNIT_ALLOW_SVD DEV_REUNIT_ALLOW_SVD
#define REUNIT_SVD_REL_ERROR DEV_REUNIT_SVD_REL_ERROR
#define REUNIT_SVD_ABS_ERROR DEV_REUNIT_SVD_ABS_ERROR
#else // cpu
#define REUNIT_ALLOW_SVD HOST_REUNIT_ALLOW_SVD
#define REUNIT_SVD_REL_ERROR HOST_REUNIT_SVD_REL_ERROR
#define REUNIT_SVD_ABS_ERROR HOST_REUNIT_SVD_ABS_ERROR
#endif

	if(REUNIT_ALLOW_SVD){
		bool perform_svd = true;
		if(!REUNIT_SVD_ONLY){
		  const typename RealTypeId<Cmplx>::Type det = getDeterminant(q).x;
		  if( fabs(det) >= REUNIT_SVD_ABS_ERROR){  
		    if( checkRelativeError(g[0]*g[1]*g[2],det,REUNIT_SVD_REL_ERROR) ) perform_svd = false;
		  }
		}	

		if(perform_svd){	
			Matrix<Cmplx,3> tmp2;
			// compute the eigenvalues using the singular value decomposition
			computeSVD<Cmplx>(q,tempq,tmp2,g);
			// The array g contains the eigenvalues of the matrix q
			// The determinant is the product of the eigenvalues, and I can use this
			// to check the SVD
			const typename RealTypeId<Cmplx>::Type determinant = getDeterminant(q).x;
			const typename RealTypeId<Cmplx>::Type gprod = g[0]*g[1]*g[2];
			// Check the svd result for errors
#ifdef __CUDA_ARCH__
#define MAX_DET_ERROR DEV_MAX_DET_ERROR
#else
#define MAX_DET_ERROR HOST_MAX_DET_ERROR
#endif
			if(fabs(gprod - determinant) > MAX_DET_ERROR){
#if  (!defined(__CUDA_ARCH__) || (__COMPUTE_CAPABILITY__ >= 200))
				printf("Warning: Error in determinant computed by SVD : %g > %g\n", fabs(gprod-determinant), MAX_DET_ERROR);
				printLink(q);
#endif

#ifdef __CUDA_ARCH__
				atomicAdd(unitarization_failed,1);
#else
				(*unitarization_failed)++;
#endif
			} 
		} // perform_svd?

	} // REUNIT_ALLOW_SVD?

#ifdef __CUDA_ARCH__
#define HISQ_FORCE_FILTER DEV_HISQ_FORCE_FILTER
#else
#define HISQ_FORCE_FILTER HOST_HISQ_FORCE_FILTER
#endif	


	typename RealTypeId<Cmplx>::Type delta = getAbsMin(g,3);
	if(delta < HISQ_FORCE_FILTER){
		for(int i=0; i<3; ++i){ 
			g[i]     += HISQ_FORCE_FILTER; 
			q(i,i).x += HISQ_FORCE_FILTER;
		}
		qsq = q*q; // recalculate Q^2
	}


        // At this point we have finished with the c's 
        // use these to store sqrt(g)
        for(int i=0; i<3; ++i) c[i] = sqrt(g[i]);

        // done with the g's, use these to store u, v, w
        g[0] = c[0]+c[1]+c[2];
        g[1] = c[0]*c[1] + c[0]*c[2] + c[1]*c[2];
        g[2] = c[0]*c[1]*c[2];
        
        // set the derivative coefficients!
        deriv_coeffs->set(g[0], g[1], g[2]);

        const typename RealTypeId<Cmplx>::Type & denominator  = g[2]*(g[0]*g[1]-g[2]); 
        c[0] = (g[0]*g[1]*g[1] - g[2]*(g[0]*g[0]+g[1]))/denominator;
        c[1] = (-g[0]*g[0]*g[0] - g[2] + 2.*g[0]*g[1])/denominator;
        c[2] =  g[0]/denominator;

        tempq = c[1]*q + c[2]*qsq;
        // Add a real scalar
        tempq(0,0).x += c[0];
        tempq(1,1).x += c[0];
        tempq(2,2).x += c[0];

        f[0] = c[0];
        f[1] = c[1];
        f[2] = c[2];

        *res = tempq;
        return;
      }



      // "v" denotes a "fattened" link variable
      template<class Cmplx>
      __device__ __host__
        void getUnitarizeForceSite(const Matrix<Cmplx,3> & v, const Matrix<Cmplx,3> & outer_prod, Matrix<Cmplx,3>* result, int *unitarization_failed)
        {
          typename RealTypeId<Cmplx>::Type f[3]; 
          typename RealTypeId<Cmplx>::Type b[6];

          Matrix<Cmplx,3> v_dagger = conj(v);  // okay!
          Matrix<Cmplx,3> q   = v_dagger*v;    // okay!

          Matrix<Cmplx,3> rsqrt_q;

          DerivativeCoefficients<typename RealTypeId<Cmplx>::Type> deriv_coeffs;

          reciprocalRoot<Cmplx>(&rsqrt_q, &deriv_coeffs, f, q, unitarization_failed);

          // Pure hack here
          b[0] = deriv_coeffs.getB00();
          b[1] = deriv_coeffs.getB01();
          b[2] = deriv_coeffs.getB02();
          b[3] = deriv_coeffs.getB11();
          b[4] = deriv_coeffs.getB12();
          b[5] = deriv_coeffs.getB22();


          Matrix<Cmplx,3> & local_result = *result;

          local_result = rsqrt_q*outer_prod;

          // We are now finished with rsqrt_q
          Matrix<Cmplx,3> qv_dagger  = q*v_dagger;
          Matrix<Cmplx,3> vv_dagger  = v*v_dagger; 
          Matrix<Cmplx,3> vqv_dagger = v*qv_dagger;
          Matrix<Cmplx,3> temp = f[1]*vv_dagger + f[2]*vqv_dagger;


          temp = f[1]*v_dagger + f[2]*qv_dagger;
          Matrix<Cmplx,3> conj_outer_prod = conj(outer_prod);


          temp = f[1]*v + f[2]*v*q;
          local_result = local_result + outer_prod*temp*v_dagger + f[2]*q*outer_prod*vv_dagger;

          local_result = local_result + v_dagger*conj_outer_prod*conj(temp) + f[2]*qv_dagger*conj_outer_prod*v_dagger;


          // now done with vv_dagger, I think
          Matrix<Cmplx,3> qsqv_dagger = q*qv_dagger;
          Matrix<Cmplx,3> pv_dagger   = b[0]*v_dagger + b[1]*qv_dagger + b[2]*qsqv_dagger;
          accumBothDerivatives(&local_result, v, pv_dagger, outer_prod);

          Matrix<Cmplx,3> rv_dagger = b[1]*v_dagger + b[3]*qv_dagger + b[4]*qsqv_dagger;
          Matrix<Cmplx,3> vq = v*q;
          accumBothDerivatives(&local_result, vq, rv_dagger, outer_prod);

          Matrix<Cmplx,3> sv_dagger = b[2]*v_dagger + b[4]*qv_dagger + b[5]*qsqv_dagger;
          Matrix<Cmplx,3> vqsq = vq*q;
          accumBothDerivatives(&local_result, vqsq, sv_dagger, outer_prod);
          return;
        } // get unit force term



      // I don't need to swap between odd and even half lattices, do I
        template<class Cmplx>
          __global__ void getUnitarizeForceField(Cmplx* link_even, Cmplx* link_odd,
                                                 Cmplx* old_force_even, Cmplx* old_force_odd,
						 																		 Cmplx* force_even, Cmplx* force_odd,
																								 int* unitarization_failed)
          {
            int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;
	
            Cmplx* force;
            Cmplx* link;
            Cmplx* old_force;

            force = force_even;
            link = link_even;
            old_force = old_force_even;
            if(mem_idx >= Vh){
              mem_idx = mem_idx - Vh;
              force = force_odd;
              link = link_odd;
              old_force = old_force_odd;
            }


            // This part of the calculation is always done in double precision
            Matrix<double2,3> v, result, oprod;
           
            for(int dir=0; dir<4; ++dir){
              loadLinkVariableFromArray(old_force, dir, mem_idx, Vh, &oprod);
              loadLinkVariableFromArray(link, dir, mem_idx, Vh, &v);

              getUnitarizeForceSite<double2>(v, oprod, &result, unitarization_failed); 

              writeLinkVariableToArray(result, dir, mem_idx, Vh, force); 
            }
            return;
          } // getUnitarizeForceField


/*
        // template this! 
        void copyArrayToLink(Matrix<float2,3>* link, float* array){
          for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
              (*link)(i,j).x = array[(i*3+j)*2];
              (*link)(i,j).y = array[(i*3+j)*2 + 1];
            }
          }
          return;
        }
	
	template<class Cmplx, class Real>
        void copyArrayToLink(Matrix<Cmplx,3>* link, Real* array){
          for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
              (*link)(i,j).x = array[(i*3+j)*2];
              (*link)(i,j).y = array[(i*3+j)*2 + 1];
            }
          }
          return;
        }
	
        
        // and this!
        void copyLinkToArray(float* array, const Matrix<float2,3>& link){
          for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
              array[(i*3+j)*2] = link(i,j).x;
              array[(i*3+j)*2 + 1] = link(i,j).y;
            }
          }
          return;
        }

        // and this!
	      template<class Cmplx, class Real>
        void copyLinkToArray(Real* array, const Matrix<Cmplx,3>& link){
          for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
              array[(i*3+j)*2] = link(i,j).x;
              array[(i*3+j)*2 + 1] = link(i,j).y;
            }
          }
          return;
        }



        // and this!
	template<class Cmplx>
	__host__ __device__
        void printLink(Matrix<Cmplx,3>& link){
          printf("(%lf, %lf)\t", link(0,0).x, link(0,0).y);
          printf("(%lf, %lf)\t", link(0,1).x, link(0,1).y);
          printf("(%lf, %lf)\n", link(0,2).x, link(0,2).y);
          printf("(%lf, %lf)\t", link(1,0).x, link(1,0).y);
          printf("(%lf, %lf)\t", link(1,1).x, link(1,1).y);
          printf("(%lf, %lf)\n", link(1,2).x, link(1,2).y);
          printf("(%lf, %lf)\t", link(2,0).x, link(2,0).y);
          printf("(%lf, %lf)\t", link(2,1).x, link(2,1).y);
          printf("(%lf, %lf)\n", link(2,2).x, link(2,2).y);
          printf("\n");
        }

*/
	void unitarize_force_cpu(const QudaGaugeParam& param, cpuGaugeField& cpuOldForce, cpuGaugeField& cpuGauge, cpuGaugeField* cpuNewForce)
	{

		int num_failures = 0;	
		Matrix<double2,3> old_force, new_force, v;
		for(int i=0; i<cpuGauge.Volume(); ++i){
			for(int dir=0; dir<4; ++dir){
				if(param.cpu_prec == QUDA_SINGLE_PRECISION){
					copyArrayToLink(&old_force, ((float*)(cpuOldForce.Gauge_p()) + (i*4 + dir)*18)); 
					copyArrayToLink(&v, ((float*)(cpuGauge.Gauge_p()) + (i*4 + dir)*18)); 
					getUnitarizeForceSite<double2>(v, old_force, &new_force, &num_failures);
					copyLinkToArray(((float*)(cpuNewForce->Gauge_p()) + (i*4 + dir)*18), new_force); 
				}else if(param.cpu_prec == QUDA_DOUBLE_PRECISION){
					copyArrayToLink(&old_force, ((double*)(cpuOldForce.Gauge_p()) + (i*4 + dir)*18)); 
					copyArrayToLink(&v, ((double*)(cpuGauge.Gauge_p()) + (i*4 + dir)*18)); 
					getUnitarizeForceSite<double2>(v, old_force, &new_force, &num_failures);
					copyLinkToArray(((double*)(cpuNewForce->Gauge_p()) + (i*4 + dir)*18), new_force); 
				} // precision?
			} // dir
		} // i
		return;
	} // unitarize_force_cpu


        void unitarize_force_cuda(const QudaGaugeParam& param, cudaGaugeField& cudaOldForce, cudaGaugeField& cudaGauge,  cudaGaugeField* cudaNewForce, int* unitarization_failed)
        {

          dim3 gridDim(cudaGauge.Volume()/BLOCK_DIM,1,1);
          dim3 blockDim(BLOCK_DIM,1,1);

	  if(param.cuda_prec == QUDA_SINGLE_PRECISION){
		  getUnitarizeForceField<<<gridDim,blockDim>>>((float2*)cudaGauge.Even_p(), (float2*)cudaGauge.Odd_p(),
				  (float2*)cudaOldForce.Even_p(), (float2*)cudaOldForce.Odd_p(),
				  (float2*)cudaNewForce->Even_p(), (float2*)cudaNewForce->Odd_p(),
				  unitarization_failed);
	  }else if(param.cuda_prec == QUDA_DOUBLE_PRECISION){
		  getUnitarizeForceField<<<gridDim,blockDim>>>((double2*)cudaGauge.Even_p(), (double2*)cudaGauge.Odd_p(),
				  (double2*)cudaOldForce.Even_p(), (double2*)cudaOldForce.Odd_p(),
				  (double2*)cudaNewForce->Even_p(), (double2*)cudaNewForce->Odd_p(),
				  unitarization_failed);
	  } // precision?
          return;
        } // unitarize_force_cuda


  } // namespace fermion_force
} // namespace hisq


