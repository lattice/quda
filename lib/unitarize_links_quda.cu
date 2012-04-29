#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <gauge_field.h>

#include "quda_matrix.h"
#include "svd_quda.h"
#include <hisq_links_quda.h>

#ifndef FL_UNITARIZE_PI
#define FL_UNITARIZE_PI 3.14159265358979323846
#endif
#ifndef FL_UNITARIZE_PI23
#define FL_UNITARIZE_PI23 FL_UNITARIZE_PI*2.0/3.0
#endif 
 
__constant__ int INPUT_PADDING=0;
__constant__ int OUTPUT_PADDING=0;
__constant__ int DEV_MAX_ITER = 20;

static int HOST_MAX_ITER = 20;

__constant__ double DEV_FL_MAX_ERROR;
__constant__ double DEV_FL_UNITARIZE_EPS;
__constant__ bool   DEV_FL_REUNIT_ALLOW_SVD;
__constant__ bool   DEV_FL_REUNIT_SVD_ONLY;
__constant__ double DEV_FL_REUNIT_SVD_REL_ERROR;
__constant__ double DEV_FL_REUNIT_SVD_ABS_ERROR;
__constant__ bool   DEV_FL_CHECK_UNITARIZATION;

static double HOST_FL_MAX_ERROR;
static double HOST_FL_UNITARIZE_EPS;
static bool   HOST_FL_REUNIT_ALLOW_SVD;
static bool   HOST_FL_REUNIT_SVD_ONLY;
static double HOST_FL_REUNIT_SVD_REL_ERROR;
static double HOST_FL_REUNIT_SVD_ABS_ERROR;
static bool   HOST_FL_CHECK_UNITARIZATION;
namespace quda{

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



 void setUnitarizeLinksConstants(double unitarize_eps, double max_error, 
				 bool allow_svd, bool svd_only,
				 double svd_rel_error, double svd_abs_error, bool check_unitarization)
      {

	// not_set is only initialised once
	static bool not_set=true;
		
	if(not_set){
          cudaMemcpyToSymbol("DEV_FL_UNITARIZE_EPS", &unitarize_eps, sizeof(double));
	  cudaMemcpyToSymbol("DEV_FL_REUNIT_ALLOW_SVD", &allow_svd, sizeof(bool));
          cudaMemcpyToSymbol("DEV_FL_REUNIT_SVD_ONLY", &svd_only, sizeof(bool));
	  cudaMemcpyToSymbol("DEV_FL_REUNIT_SVD_REL_ERROR", &svd_rel_error, sizeof(double));
          cudaMemcpyToSymbol("DEV_FL_REUNIT_SVD_ABS_ERROR", &svd_abs_error, sizeof(double));
          cudaMemcpyToSymbol("DEV_FL_MAX_ERROR", &max_error, sizeof(double));
	  cudaMemcpyToSymbol("DEV_FL_CHECK_UNITARIZATION", &check_unitarization, sizeof(bool));
	  

	  HOST_FL_UNITARIZE_EPS = unitarize_eps;
	  HOST_FL_REUNIT_ALLOW_SVD = allow_svd;
          HOST_FL_REUNIT_SVD_ONLY = svd_only;
	  HOST_FL_REUNIT_SVD_REL_ERROR = svd_rel_error;
          HOST_FL_REUNIT_SVD_ABS_ERROR = svd_abs_error;
          HOST_FL_MAX_ERROR = max_error;     
          HOST_FL_CHECK_UNITARIZATION = check_unitarization;

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

		qsq = q*q;
		tempq = qsq*q;

		c[0] = getTrace(q).x;
		c[1] = getTrace(qsq).x/2.0;
		c[2] = getTrace(tempq).x/3.0;

		g[0] = g[1] = g[2] = c[0]/3.;
		typename RealTypeId<Cmplx>::Type r,s,theta;
		s = c[1]/3. - c[0]*c[0]/18;

#ifdef __CUDA_ARCH__
#define FL_UNITARIZE_EPS DEV_FL_UNITARIZE_EPS
#else
#define FL_UNITARIZE_EPS HOST_FL_UNITARIZE_EPS
#endif


#ifdef __CUDA_ARCH__
#define FL_REUNIT_SVD_REL_ERROR DEV_FL_REUNIT_SVD_REL_ERROR
#define FL_REUNIT_SVD_ABS_ERROR DEV_FL_REUNIT_SVD_ABS_ERROR
#else // cpu
#define FL_REUNIT_SVD_REL_ERROR HOST_FL_REUNIT_SVD_REL_ERROR
#define FL_REUNIT_SVD_ABS_ERROR HOST_FL_REUNIT_SVD_ABS_ERROR
#endif


		typename RealTypeId<Cmplx>::Type cosTheta; 
		if(fabs(s) >= FL_UNITARIZE_EPS){
		  const typename RealTypeId<Cmplx>::Type sqrt_s = sqrt(s);
		  r = c[2]/2. - (c[0]/3.)*(c[1] - c[0]*c[0]/9.);
		  cosTheta = r/(sqrt_s*sqrt_s*sqrt_s);
		  if(fabs(cosTheta) >= 1.0){
		    if( r > 0 ){ 
			theta = 0.0;
		    }else{
			theta = FL_UNITARIZE_PI;
		    }
		  }else{ 
			theta = acos(cosTheta);
		  }
		  g[0] = c[0]/3 + 2*sqrt_s*cos( theta/3 );
		  g[1] = c[0]/3 + 2*sqrt_s*cos( theta/3 + FL_UNITARIZE_PI23 );
 		  g[2] = c[0]/3 + 2*sqrt_s*cos( theta/3 + 2*FL_UNITARIZE_PI23 );
		}
                
		// Check the eigenvalues, if the determinant does not match the product of the eigenvalues
                // return false. Then call SVD instead.
                typename RealTypeId<Cmplx>::Type det = getDeterminant(q).x;
	        if( fabs(det) < FL_REUNIT_SVD_ABS_ERROR ){ 
			return false;
		}
		if( checkRelativeError(g[0]*g[1]*g[2],det,FL_REUNIT_SVD_REL_ERROR) == false ) return false;


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
#define FL_REUNIT_SVD_ONLY  DEV_FL_REUNIT_SVD_ONLY
#define FL_REUNIT_ALLOW_SVD DEV_FL_REUNIT_ALLOW_SVD
#else
#define FL_REUNIT_SVD_ONLY  HOST_FL_REUNIT_SVD_ONLY
#define FL_REUNIT_ALLOW_SVD HOST_FL_REUNIT_ALLOW_SVD
#endif
    if( !FL_REUNIT_SVD_ONLY ){
      if( reciprocalRoot<Cmplx>(conj(in)*in,&u) ){
	*result = in*u;
	return true;
      }
    }

   // If we've got this far, then the Caley-Hamilton unitarization 
   // has failed. If SVD is not allowed, the unitarization has failed.
   if( !FL_REUNIT_ALLOW_SVD ) return false;

    Matrix<Cmplx,3> v;
    typename RealTypeId<Cmplx>::Type singular_values[3];
    computeSVD<Cmplx>(in, u, v, singular_values); // should pass pointers to u, v I guess
    *result = u*conj(v);
    return true;
  } // unitarizeMILC
    

   template<class Cmplx>
   __host__ __device__
   bool unitarizeLinkSVD(const Matrix<Cmplx,3>& in, Matrix<Cmplx,3>* const result)
   {
	Matrix<Cmplx,3> u, v;
	typename RealTypeId<Cmplx>::Type singular_values[3];
	computeSVD<Cmplx>(in, u, v, singular_values); // should pass pointers to u,v I guess	

	*result = u*conj(v);

#ifdef __CUDA_ARCH__ 
#define FL_MAX_ERROR  DEV_FL_MAX_ERROR
#else 
#define FL_MAX_ERROR  HOST_FL_MAX_ERROR
#endif
	if(isUnitary(*result,FL_MAX_ERROR)==false)
	{
#if (!defined(__CUDA_ARCH__) || (__COMPUTE_CAPABILITY__>=200))
          printf("ERROR: Link unitarity test failed\n");
          printf("TOLERANCE: %g\n", FL_MAX_ERROR);
#endif
	  return false;
	}
	return true;
   }
#undef FL_MAX_ERROR


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
				     int* num_failures, const int threads)
  {
    int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (mem_idx >= threads) return;

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
       unitarizeLinkMILC(v, &result);
#ifdef __CUDA_ARCH__
#define FL_MAX_ERROR DEV_FL_MAX_ERROR
#define FL_CHECK_UNITARIZATION DEV_FL_CHECK_UNITARIZATION
#else
#define FL_MAX_ERROR HOST_FL_MAX_ERROR
#define FL_CHECK_UNITARIZATION HOST_FL_CHECK_UNITARIZATION
#endif
     if(FL_CHECK_UNITARIZATION){
        if(isUnitary(result,FL_MAX_ERROR) == false)
        {
#ifdef __CUDA_ARCH__
	  atomicAdd(num_failures, 1);
#else 
	  (*num_failures)++;
#endif
        }
      }
      writeLinkVariableToArray(result, dir, mem_idx, Vh+OUTPUT_PADDING, outlink); 
    }
    return;
  }

  class UnitarizeLinksCuda : public Tunable {
  private:
    const cudaGaugeField &inField;
    cudaGaugeField &outField;
    int *fails;
    
    int sharedBytesPerThread() const { return 0; }
    int sharedBytesPerBlock(const TuneParam &) const { return 0; }
    
    // don't tune the grid dimension
    bool advanceGridDim(TuneParam &param) const { return false; }
    bool advanceBlockDim(TuneParam &param) const {
      bool rtn = Tunable::advanceBlockDim(param);
      const int threads = inField.Volume();
      param.grid = dim3((threads+param.block.x-1)/param.block.x, 1, 1);
      return rtn;
    }
  public:
    UnitarizeLinksCuda(const cudaGaugeField& inField, cudaGaugeField& outField,  int* fails) : 
      inField(inField), outField(outField), fails(fails) { ; }
    virtual ~UnitarizeLinksCuda() { ; }
    
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);
      
      if(inField.Precision() == QUDA_SINGLE_PRECISION){
	getUnitarizedField<<<tp.grid,tp.block>>>((float2*)inField.Even_p(), (float2*)inField.Odd_p(),
						 (float2*)outField.Even_p(), (float2*)outField.Odd_p(),
						 fails, inField.Volume());
      }else if(inField.Precision() == QUDA_DOUBLE_PRECISION){
	getUnitarizedField<<<tp.grid,tp.block>>>((double2*)inField.Even_p(), (double2*)inField.Odd_p(),
						 (double2*)outField.Even_p(), (double2*)outField.Odd_p(),
						 fails, inField.Volume());
      } else {
	errorQuda("UnitarizeLinks not implemented for precision %d", inField.Precision());
      }
      
    }
    void preTune() { ; }
    void postTune() { cudaMemset(fails, 0, sizeof(int)); } // reset fails counter
    
    void initTuneParam(TuneParam &param) const {
      Tunable::initTuneParam(param);
      const int threads = inField.Volume();
      param.grid = dim3((threads+param.block.x-1)/param.block.x, 1, 1);
    }
    
      
    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const {
      Tunable::defaultTuneParam(param);
      const int threads = inField.Volume();
      param.grid = dim3((threads+param.block.x-1)/param.block.x, 1, 1);
    }
      
    long long flops() const { return 0; } // FIXME: add flops counter

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << inField.X()[0] << "x";
      vol << inField.X()[1] << "x";
      vol << inField.X()[2] << "x";
      vol << inField.X()[3] << "x";
      aux << "threads=" << inField.Volume() << ",prec=" << inField.Precision();
      aux << "stride=" << inField.Stride();
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }  
  }; // UnitarizeLinksCuda
    
  void unitarizeLinksCuda(const QudaGaugeParam& param,
			  cudaGaugeField& inField,
			  cudaGaugeField* outField, 
			  int* fails) { 
    UnitarizeLinksCuda unitarizeLinks(inField, *outField, fails);
    unitarizeLinks.apply(0);
  }

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
    }  // loop over volume
    return;
  }
    
  // CPU function which checks that the gauge field is unitary
  bool isUnitary(const QudaGaugeParam& param, cpuGaugeField& field, double max_error)
  {
    Matrix<double2,3> link, identity;
      
    for(int i=0; i<field.Volume(); ++i){
      for(int dir=0; dir<4; ++dir){
	if(param.cpu_prec == QUDA_SINGLE_PRECISION){
	  copyArrayToLink(&link, ((float*)(field.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	}else if(param.cpu_prec == QUDA_DOUBLE_PRECISION){     
	  copyArrayToLink(&link, ((double*)(field.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	}else{
	  errorQuda("Unsupported precision\n");
	}
	if(isUnitary(link,max_error) == false){ 
	  printf("Unitarity failure\n");
	  printf("site index = %d,\t direction = %d\n", i, dir);
	  printLink(link);
	  identity = conj(link)*link;
	  printLink(identity);
	  return false;
	}
      } // dir
    } // i	  
    return true;
  } // is unitary

    
    
} // namespace quda
