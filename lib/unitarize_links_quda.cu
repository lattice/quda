#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <gauge_field.h>

#include "quda_matrix.h"
#include "svd_quda.h"
#include <hisq_links_quda.h>

#define HISQ_UNITARIZE_PI 3.14159265358979323846
#define HISQ_UNITARIZE_PI23 HISQ_UNITARIZE_PI*2.0/3.0

__constant__ int INPUT_PADDING=0;
__constant__ int OUTPUT_PADDING=0;
__constant__ int DEV_MAX_ITER = 20;

static int HOST_MAX_ITER = 20;

__constant__ double DEV_HISQ_UNITARIZE_EPS = 1e-6;
__constant__ double DEV_MAX_DET_ERROR = 1e-9;
__constant__ bool DEV_REUNIT_ALLOW_SVD = true;
__constant__ bool DEV_REUNIT_SVD_ONLY = false;
__constant__ double DEV_REUNIT_SVD_REL_ERROR = 1e-6;
__constant__ double DEV_REUNIT_SVD_ABS_ERROR = 1e-6;


static double HOST_HISQ_UNITARIZE_EPS;
static double HOST_MAX_DET_ERROR;
static bool   HOST_REUNIT_ALLOW_SVD;
static bool   HOST_REUNIT_SVD_ONLY;
static double HOST_REUNIT_SVD_REL_ERROR;
static double HOST_REUNIT_SVD_ABS_ERROR;

namespace hisq{


  void setUnitarizeLinksPadding(int input_padding, int output_padding)
  {
    cudaMemcpyToSymbol("INPUT_PADDING", &input_padding, sizeof(int));
    cudaMemcpyToSymbol("OUTPUT_PADDING", &output_padding, sizeof(int));
    return;
  }


  template<class Cmplx>
    __device__ __host__
    bool isUnitary(const Matrix<Cmplx,3>& matrix, double max_error)
    {
      const Matrix<Cmplx,3> identity = conj(matrix)*matrix;

      for(int i=0; i<3; ++i){
        if( fabs(identity(i,i).x - 1.0) > max_error || fabs(identity(i,i).y) > max_error) return false;
          for(int j=i+1; j<3; ++j){
	    if( fabs(identity(i,j).x) > max_error || fabs(identity(i,j).y) > max_error
	    ||  fabs(identity(j,i).x) > max_error || fabs(identity(j,i).y) > max_error ){
	      return false;
	    }
	  }
	}
        return true;
    }




template<class Cmplx>
__device__ __host__
bool isUnitarizedLinkConsistent(const Matrix<Cmplx,3>& initial_matrix,
			        const Matrix<Cmplx,3>& unitary_matrix,
			        double max_error)	
{
  Matrix<Cmplx,3> temporary; 
  //temporary = conj(initial_matrix)*unitary_matrix*unitary_matrix - initial_matrix;
  temporary = conj(initial_matrix)*unitary_matrix;

  temporary = temporary*temporary - conj(initial_matrix)*initial_matrix;
   
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j){
	if( fabs(temporary(i,j).x) > max_error || fabs(temporary(i,j).y) > max_error){
	 return false;
	}
    }
  }
  return true;
}



 void setUnitarizeLinksConstants(double unitarize_eps, double max_det_error, 
				 bool allow_svd, bool svd_only,
				 double svd_rel_error, double svd_abs_error)
      {

	// not_set is only initialised once
	static bool not_set=true;
		
	if(not_set){
          cudaMemcpyToSymbol("DEV_HISQ_UNITARIZE_EPS", &unitarize_eps, sizeof(double));
          cudaMemcpyToSymbol("DEV_MAX_DET_ERROR", &max_det_error, sizeof(double));
	  cudaMemcpyToSymbol("DEV_REUNIT_ALLOW_SVD", &allow_svd, sizeof(bool));
          cudaMemcpyToSymbol("DEV_REUNIT_SVD_ONLY", &svd_only, sizeof(bool));
	  cudaMemcpyToSymbol("DEV_REUNIT_SVD_REL_ERROR", &svd_rel_error, sizeof(double));
          cudaMemcpyToSymbol("DEV_REUNIT_SVD_ABS_ERROR", &svd_abs_error, sizeof(double));
	
	  HOST_HISQ_UNITARIZE_EPS = unitarize_eps;
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
      bool reciprocalRoot(const Matrix<Cmplx,3>& q, Matrix<Cmplx,3>* res){

        	Matrix<Cmplx,3> qsq, tempq;

       		typename RealTypeId<Cmplx>::Type c[3];
        	typename RealTypeId<Cmplx>::Type g[3];

#ifdef __CUDA_ARCH__
#define REUNIT_SVD_ONLY DEV_REUNIT_SVD_ONLY
#else
#define REUNIT_SVD_ONLY HOST_REUNIT_SVD_ONLY
#endif
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


#ifdef __CUDA_ARCH__
#define REUNIT_SVD_REL_ERROR DEV_REUNIT_SVD_REL_ERROR
#define REUNIT_SVD_ABS_ERROR DEV_REUNIT_SVD_ABS_ERROR
#else // cpu
#define REUNIT_SVD_REL_ERROR HOST_REUNIT_SVD_REL_ERROR
#define REUNIT_SVD_ABS_ERROR HOST_REUNIT_SVD_ABS_ERROR
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

                
		// Check the eigenvalues, if the determinant does not match the product of the eigenvalues
                // return false. Then call SVD instead.
                typename RealTypeId<Cmplx>::Type det = getDeterminant(q).x;
		if( fabs(det) >= REUNIT_SVD_ABS_ERROR){  
		  if( checkRelativeError(g[0]*g[1]*g[2],det,REUNIT_SVD_REL_ERROR) == false ) return false;
		}else{
		  return false;
		}

        	// At this point we have finished with the c's 
        	// use these to store sqrt(g)
        	for(int i=0; i<3; ++i) c[i] = sqrt(g[i]);

        	// done with the g's, use these to store u, v, w
        	g[0] = c[0]+c[1]+c[2];
        	g[1] = c[0]*c[1] + c[0]*c[2] + c[1]*c[2];
        	g[2] = c[0]*c[1]*c[2];
        
        	const typename RealTypeId<Cmplx>::Type & denominator  = g[2]*(g[0]*g[1]-g[2]); 
        	c[0] = (g[0]*g[1]*g[1] - g[2]*(g[0]*g[0]+g[1]))/denominator;
        	c[1] = (-g[0]*g[0]*g[0] - g[2] + 2.*g[0]*g[1])/denominator;
       	 	c[2] =  g[0]/denominator;

        	tempq = c[1]*q + c[2]*qsq;
        	// Add a real scalar
        	tempq(0,0).x += c[0];
        	tempq(1,1).x += c[0];
        	tempq(2,2).x += c[0];

        	*res = tempq;
        	
		return true;
      }




  template<class Cmplx>
  __host__ __device__
  bool unitarizeLinkMILC(const Matrix<Cmplx,3>& in, Matrix<Cmplx,3>* const result)
  {
    Matrix<Cmplx,3> u;
#ifdef __CUDA_ARCH__
#define REUNIT_SVD_ONLY DEV_REUNIT_SVD_ONLY
#else
#define REUNIT_SVD_ONLY HOST_REUNIT_SVD_ONLY
#endif
    if( !REUNIT_SVD_ONLY ){
      if( reciprocalRoot<Cmplx>(conj(in)*in,&u) ){
	*result = in*u;
	return true;
      }
    }

    Matrix<Cmplx,3> v;
    typename RealTypeId<Cmplx>::Type singular_values[3];
    computeSVD<Cmplx>(in, u, v, singular_values); // should pass pointers to u, v I guess
    *result = u*conj(v);
#ifdef __CUDA_ARCH__
#define MAX_DET_ERROR DEV_MAX_DET_ERROR
#else
#define MAX_DET_ERROR HOST_MAX_DET_ERROR
#endif
    // Finally, check that the absolute value of the determinant does not 
    // differ significantly from one.
    // We could be more rigorous by explicitly checking the unitarity of the 
    // matrix, or, even more stringently, by verifying that W = V/sqrt(V^{dagger} V).
    const Cmplx det = getDeterminant(*result);
    if( cabs(det)-1.0 > MAX_DET_ERROR){ 
      return false;
    }
    return true;
#undef MAX_DET_ERROR
  } // unitarizeMILC
    

   template<class Cmplx>
   __host__ __device__
   bool unitarizeLinkSVD(const Matrix<Cmplx,3>& in, Matrix<Cmplx,3>* const result)
   {
	Matrix<Cmplx,3> u, v;
	typename RealTypeId<Cmplx>::Type singular_values[3];
	computeSVD<Cmplx>(in, u, v, singular_values); // should pass pointers to u,v I guess	

	*result = u*conj(v);

	if(isUnitary(*result,0.0000001)==false)
	{
#if (!defined(__CUDA_ARCH__) || (__COMPUTE_CAPABILITY__>=200))
          printf("ERROR: Unitarized link is not consistent with incoming link\n");
#endif
	  return false;
	}
	return true;
   }



   template<class Cmplx>
   __host__ __device__
   bool unitarizeLinkNewton(const Matrix<Cmplx,3>& in, Matrix<Cmplx,3>* const result)
   {
      Matrix<Cmplx,3> u, uinv;
      u = in;

#ifdef __CUDA_ARCH__
#define MAX_ITER DEV_MAX_ITER
#else
#define MAX_ITER HOST_MAX_ITER
#endif
      for(int i=0; i<MAX_ITER; ++i){
         computeMatrixInverse(u, &uinv);
	 u = 0.5*(u + conj(uinv));
      }

#undef MAX_ITER	
      if(isUnitarizedLinkConsistent(in,u,0.0000001)==false)
      {
#if (!defined(__CUDA_ARCH__) || (__COMPUTE_CAPABILITY__>=200))
        printf("ERROR: Unitarized link is not consistent with incoming link\n");
#endif
	return false;
      }
      *result = u;

      return true;
   }   


  




  template<class Cmplx>
  __global__ void getUnitarizedField(const Cmplx* inlink_even, const Cmplx*  inlink_odd,
				    Cmplx*  outlink_even, Cmplx*  outlink_odd,
				    int* num_failures)
  {
    int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const Cmplx* inlink;
    Cmplx* outlink;

    inlink  = inlink_even;
    outlink = outlink_even;
    
    if(mem_idx >= Vh){
      mem_idx = mem_idx - Vh;
      inlink  = inlink_odd;
      outlink = outlink_odd;
    }

    // Unitarization is always done in double precision
    Matrix<double2,3> v, result;
    for(int dir=0; dir<4; ++dir){
      loadLinkVariableFromArray(inlink, dir, mem_idx, Vh+INPUT_PADDING, &v); 
       if( !unitarizeLinkMILC(v, &result) ){
#if __CUDA_ARCH__
	atomicAdd(num_failures, 1);
#else 
	(*num_failures)++;
#endif
      }
      writeLinkVariableToArray(result, dir, mem_idx, Vh+OUTPUT_PADDING, outlink); 
    }
    
    return;
  }


  void unitarizeLinksCuda(const QudaGaugeParam& param, cudaGaugeField& infield, cudaGaugeField* outfield, int* num_failures)
  {
	
    dim3 gridDim(infield.Volume()/BLOCK_DIM,1,1);
    dim3 blockDim(BLOCK_DIM,1,1); 


    checkCudaError();

    if(param.cuda_prec == QUDA_SINGLE_PRECISION){
      getUnitarizedField<<<gridDim,blockDim>>>((float2*)infield.Even_p(), (float2*)infield.Odd_p(),
					       (float2*)outfield->Even_p(), (float2*)outfield->Odd_p(),
					       num_failures);
						
    }else if(param.cuda_prec == QUDA_DOUBLE_PRECISION){
      getUnitarizedField<<<gridDim,blockDim>>>((double2*)infield.Even_p(), (double2*)infield.Odd_p(),
					       (double2*)outfield->Even_p(), (double2*)outfield->Odd_p(),
					       num_failures);
    }

    checkCudaError();
    return;
  } // unitarize_links_cuda


 
  void unitarizeLinksCPU(const QudaGaugeParam& param, cpuGaugeField& infield, cpuGaugeField* outfield)
  {
    int num_failures = 0;
    Matrix<double2,3> inlink, outlink;

    for(int i=0; i<infield.Volume(); ++i){
	for(int dir=0; dir<4; ++dir){
	  if(param.cpu_prec == QUDA_SINGLE_PRECISION){
	    copyArrayToLink(&inlink, ((float*)(infield.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	    if( unitarizeLinkNewton<double2>(inlink, &outlink) == false ) num_failures++; 
	    copyLinkToArray(((float*)(outfield->Gauge_p()) + (i*4 + dir)*18), outlink); 
	  }else if(param.cpu_prec == QUDA_DOUBLE_PRECISION){
	    copyArrayToLink(&inlink, ((double*)(infield.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	    if( unitarizeLinkNewton<double2>(inlink, &outlink) == false ) num_failures++; 
	    copyLinkToArray(((double*)(outfield->Gauge_p()) + (i*4 + dir)*18), outlink); 
	  } // precision?
	} // dir
    }  // return
    return;
  }



} // namespace hisq
