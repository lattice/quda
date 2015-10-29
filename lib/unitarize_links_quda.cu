#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>

#include <tune_quda.h>
#include <quda_matrix.h>
#include <unitarization_links.h>

#include <su3_project.cuh>
#include <index_helper.cuh>


namespace quda{
#ifdef GPU_UNITARIZE

namespace{
  #include <svd_quda.h>
}

#ifndef FL_UNITARIZE_PI
#define FL_UNITARIZE_PI 3.14159265358979323846
#endif
#ifndef FL_UNITARIZE_PI23
#define FL_UNITARIZE_PI23 FL_UNITARIZE_PI*0.66666666666666666666
#endif 
 
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

    const typename RealTypeId<Cmplx>::Type one_third = 0.333333333333333333333;
    const typename RealTypeId<Cmplx>::Type one_ninth = 0.111111111111111111111;
    const typename RealTypeId<Cmplx>::Type one_eighteenth = 0.055555555555555555555;

    qsq = q*q;
    tempq = qsq*q;

    c[0] = getTrace(q).x;
    c[1] = getTrace(qsq).x * 0.5;
    c[2] = getTrace(tempq).x * one_third;;

    g[0] = g[1] = g[2] = c[0] * one_third;
    typename RealTypeId<Cmplx>::Type r,s,theta;
    s = c[1]*one_third - c[0]*c[0]*one_eighteenth;

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
    if(fabs(s) >= FL_UNITARIZE_EPS){ // faster when this conditional is removed?
      const typename RealTypeId<Cmplx>::Type rsqrt_s = rsqrt(s);
      r = c[2]*0.5 - (c[0]*one_third)*(c[1] - c[0]*c[0]*one_ninth);
      cosTheta = r*rsqrt_s*rsqrt_s*rsqrt_s;

      if(fabs(cosTheta) >= 1.0){
	theta = (r > 0) ? 0.0 : FL_UNITARIZE_PI;
      }else{ 
	theta = acos(cosTheta); // this is the primary performance limiter
      }

      const typename RealTypeId<Cmplx>::Type sqrt_s = s*rsqrt_s;

#if 0 // experimental version
      typename RealTypeId<Cmplx>::Type as, ac;
      sincos( theta*one_third, &as, &ac );
      g[0] = c[0]*one_third + 2*sqrt_s*ac;
      //g[1] = c[0]*one_third + 2*sqrt_s*(ac*cos(1*FL_UNITARIZE_PI23) - as*sin(1*FL_UNITARIZE_PI23));
      g[1] = c[0]*one_third - 2*sqrt_s*(0.5*ac + as*0.8660254037844386467637);
      //g[2] = c[0]*one_third + 2*sqrt_s*(ac*cos(2*FL_UNITARIZE_PI23) - as*sin(2*FL_UNITARIZE_PI23));
      g[2] = c[0]*one_third + 2*sqrt_s*(-0.5*ac + as*0.8660254037844386467637);
#else
      g[0] = c[0]*one_third + 2*sqrt_s*cos( theta*one_third );
      g[1] = c[0]*one_third + 2*sqrt_s*cos( theta*one_third + FL_UNITARIZE_PI23 );
      g[2] = c[0]*one_third + 2*sqrt_s*cos( theta*one_third + 2*FL_UNITARIZE_PI23 );
#endif
    }
                
    // Check the eigenvalues, if the determinant does not match the product of the eigenvalues
    // return false. Then call SVD instead.
    typename RealTypeId<Cmplx>::Type det = getDeterminant(q).x;
    if( fabs(det) < FL_REUNIT_SVD_ABS_ERROR ) return false;
    if( checkRelativeError(g[0]*g[1]*g[2],det,FL_REUNIT_SVD_REL_ERROR) == false ) return false;


    // At this point we have finished with the c's 
    // use these to store sqrt(g)
    for(int i=0; i<3; ++i) c[i] = sqrt(g[i]);

    // done with the g's, use these to store u, v, w
    g[0] = c[0]+c[1]+c[2];
    g[1] = c[0]*c[1] + c[0]*c[2] + c[1]*c[2];
    g[2] = c[0]*c[1]*c[2];
        
    const typename RealTypeId<Cmplx>::Type & denominator  = 1.0 / ( g[2]*(g[0]*g[1]-g[2]) ); 
    c[0] = (g[0]*g[1]*g[1] - g[2]*(g[0]*g[0]+g[1])) * denominator;
    c[1] = (-g[0]*g[0]*g[0] - g[2] + 2.*g[0]*g[1]) * denominator;
    c[2] =  g[0] * denominator;

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
    computeSVD<Cmplx>(in, u, v, singular_values);
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

  void unitarizeLinksCPU(cpuGaugeField &outfield, const cpuGaugeField& infield)
  {
    if (infield.Precision() != outfield.Precision())
      errorQuda("Precisions must match (out=%d != in=%d)", outfield.Precision(), infield.Precision());
    
    int num_failures = 0;
    Matrix<double2,3> inlink, outlink;
      
    for (int i=0; i<infield.Volume(); ++i){
      for (int dir=0; dir<4; ++dir){
	if (infield.Precision() == QUDA_SINGLE_PRECISION){
	  copyArrayToLink(&inlink, ((float*)(infield.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	  if( unitarizeLinkNewton<double2>(inlink, &outlink) == false ) num_failures++; 
	  copyLinkToArray(((float*)(outfield.Gauge_p()) + (i*4 + dir)*18), outlink); 
	} else if (infield.Precision() == QUDA_DOUBLE_PRECISION){
	  copyArrayToLink(&inlink, ((double*)(infield.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	  if( unitarizeLinkNewton<double2>(inlink, &outlink) == false ) num_failures++; 
	  copyLinkToArray(((double*)(outfield.Gauge_p()) + (i*4 + dir)*18), outlink); 
	} // precision?
      } // dir
    }  // loop over volume
    return;
  }
    
  // CPU function which checks that the gauge field is unitary
  bool isUnitary(const cpuGaugeField& field, double max_error)
  {
    Matrix<double2,3> link, identity;
      
    for(int i=0; i<field.Volume(); ++i){
      for(int dir=0; dir<4; ++dir){
	if(field.Precision() == QUDA_SINGLE_PRECISION){
	  copyArrayToLink(&link, ((float*)(field.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	}else if(field.Precision() == QUDA_DOUBLE_PRECISION){     
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

  template <typename Out, typename In>
  struct UnitarizeLinksQudaArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    Out output;
    const In input;
    int *fails;
    UnitarizeLinksQudaArg(Out &output, const In &input, const GaugeField &data,  int* fails) 
      : output(output), input(input), fails(fails) {
      for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
      threads = X[0]*X[1]*X[2]*X[3];
    }
  };


  template<typename Float, typename Out, typename In>
  __global__ void DoUnitarizedLink(UnitarizeLinksQudaArg<Out,In> arg){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx >= arg.threads) return;
    typedef typename ComplexTypeId<Float>::Type Cmplx;
    int parity = 0;
    if(idx >= arg.threads/2) {
      parity = 1;
      idx -= arg.threads/2;
    }
    int X[4]; 
    for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];
    int x[4];
    getCoords(x, idx, X, parity);
    
    idx = linkIndex(x,X);
    Matrix<double2,3> v, result;
    Matrix<Cmplx,3> tmp;
    for (int mu = 0; mu < 4; mu++) { 
      arg.input.load((Float*)(tmp.data),idx, mu, parity);
      for(int i = 0; i < 9;i++) {
        v.data[i].x = (double)tmp.data[i].x;
        v.data[i].y = (double)tmp.data[i].y;
      }
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
	    atomicAdd(arg.fails, 1);
#else 
	    (*arg.fails)++;
#endif
	  }
      }
      //WRITE BACK IF FAIL??????????
      for(int i = 0; i < 9;i++) {
	tmp.data[i].x = (Float)result.data[i].x;
	tmp.data[i].y = (Float)result.data[i].y;
      }
      arg.output.save((Float*)(tmp.data),idx, mu, parity); 
    }
  }



  template<typename Float, typename Out, typename In>
  class UnitarizeLinksQuda : Tunable {    
    UnitarizeLinksQudaArg<Out,In> arg;
    
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }
    
    // don't tune the grid dimension
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }
    
  public:
    UnitarizeLinksQuda(UnitarizeLinksQudaArg<Out,In> &arg) : arg(arg) { }
    
    
    void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      DoUnitarizedLink<Float,Out,In><<<tp.grid, tp.block, 0, stream>>>(arg);
    }
    void preTune() { if (arg.input.gauge == arg.output.gauge) arg.output.save(); }
    void postTune() {
      if (arg.input.gauge == arg.output.gauge) arg.output.load();
      cudaMemset(arg.fails, 0, sizeof(int)); // reset fails counter
    }
    
    long long flops() const { 
	  // Accounted only the minimum flops for the case FL_REUNIT_SVD_ONLY=0
      return 4588LL*arg.threads; 
    }
    long long bytes() const { return 4ll * arg.threads * (arg.input.Bytes() + arg.output.Bytes()); }
    
    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << arg.X[0] << "x";
      vol << arg.X[1] << "x";
      vol << arg.X[2] << "x";
      vol << arg.X[3];
      aux << "threads=" << arg.threads << ",prec=" << sizeof(Float);
      return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
    }  
  }; 
  
  
  template<typename Float, typename Out, typename In>
  void unitarizeLinksQuda(Out output,  const In input, const cudaGaugeField& meta, int* fails) {
    UnitarizeLinksQudaArg<Out,In> arg(output, input, meta, fails);
    UnitarizeLinksQuda<Float, Out, In> unitlinks(arg) ;
    unitlinks.apply(0);
    cudaDeviceSynchronize(); // need to synchronize to ensure failure write has completed
  }
  
template<typename Float>
void unitarizeLinksQuda(cudaGaugeField& output, const cudaGaugeField &input, int* fails) {

  if( output.isNative() && input.isNative() ) {
    if(output.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type Out;

      if(input.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type In;
	unitarizeLinksQuda<Float>(Out(output), In(input), input, fails) ;
      } else if(input.Reconstruct() == QUDA_RECONSTRUCT_12) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type In;
	unitarizeLinksQuda<Float>(Out(output), In(input), input, fails) ;
      } else if(input.Reconstruct() == QUDA_RECONSTRUCT_8) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type In;
	unitarizeLinksQuda<Float>(Out(output), In(input), input, fails) ;
      } else {
	errorQuda("Reconstruction type %d of gauge field not supported", input.Reconstruct());
      }

    } else if(output.Reconstruct() == QUDA_RECONSTRUCT_12){
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type Out;

      if(input.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type In;
	unitarizeLinksQuda<Float>(Out(output), In(input), input, fails) ;
      } else if(input.Reconstruct() == QUDA_RECONSTRUCT_12) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type In;
	unitarizeLinksQuda<Float>(Out(output), In(input), input, fails) ;
      } else if(input.Reconstruct() == QUDA_RECONSTRUCT_8) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type In;
	unitarizeLinksQuda<Float>(Out(output), In(input), input, fails) ;
      } else {
	errorQuda("Reconstruction type %d of gauge field not supported", input.Reconstruct());
      }


    } else if(output.Reconstruct() == QUDA_RECONSTRUCT_8){
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type Out;

      if(input.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type In;
	unitarizeLinksQuda<Float>(Out(output), In(input), input, fails) ;
      } else if(input.Reconstruct() == QUDA_RECONSTRUCT_12) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type In;
	unitarizeLinksQuda<Float>(Out(output), In(input), input, fails) ;
      } else if(input.Reconstruct() == QUDA_RECONSTRUCT_8) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type In;
	unitarizeLinksQuda<Float>(Out(output), In(input), input, fails) ;
      } else {
	errorQuda("Reconstruction type %d of gauge field not supported", input.Reconstruct());
      }


    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", output.Reconstruct());
    }
  } else {
    errorQuda("Invalid Gauge Order (output=%d, input=%d)", output.Order(), input.Order());
  }
}
  
#endif
  
  void unitarizeLinksQuda(cudaGaugeField& output, const cudaGaugeField &input, int* fails) {
#ifdef GPU_UNITARIZE
    if (input.Precision() != output.Precision()) 
      errorQuda("input (%d) and output (%d) precisions must match", output.Precision(), input.Precision());

    if (input.Precision() == QUDA_SINGLE_PRECISION) {
      unitarizeLinksQuda<float>(output, input, fails);
    } else if(input.Precision() == QUDA_DOUBLE_PRECISION) {
      unitarizeLinksQuda<double>(output, input, fails);
    } else {
      errorQuda("Precision %d not supported", input.Precision());
    }
#else
    errorQuda("Unitarization has not been built");
#endif
  }

  void unitarizeLinksQuda(cudaGaugeField &links, int* fails) {
    unitarizeLinksQuda(links, links, fails);
  }

  void setUnitarizeLinksConstants(double unitarize_eps_h, double max_error_h, 
				  bool allow_svd_h, bool svd_only_h,
				  double svd_rel_error_h, double svd_abs_error_h, 
				  bool check_unitarization_h)
  {
#ifdef GPU_UNITARIZE
    // not_set is only initialised once
    static bool not_set=true;
		
    if(not_set){
      cudaMemcpyToSymbol(DEV_FL_UNITARIZE_EPS, &unitarize_eps_h, sizeof(double));
      cudaMemcpyToSymbol(DEV_FL_REUNIT_ALLOW_SVD, &allow_svd_h, sizeof(bool));
      cudaMemcpyToSymbol(DEV_FL_REUNIT_SVD_ONLY, &svd_only_h, sizeof(bool));
      cudaMemcpyToSymbol(DEV_FL_REUNIT_SVD_REL_ERROR, &svd_rel_error_h, sizeof(double));
      cudaMemcpyToSymbol(DEV_FL_REUNIT_SVD_ABS_ERROR, &svd_abs_error_h, sizeof(double));
      cudaMemcpyToSymbol(DEV_FL_MAX_ERROR, &max_error_h, sizeof(double));
      cudaMemcpyToSymbol(DEV_FL_CHECK_UNITARIZATION, &check_unitarization_h, sizeof(bool));
	  

      HOST_FL_UNITARIZE_EPS = unitarize_eps_h;
      HOST_FL_REUNIT_ALLOW_SVD = allow_svd_h;
      HOST_FL_REUNIT_SVD_ONLY = svd_only_h;
      HOST_FL_REUNIT_SVD_REL_ERROR = svd_rel_error_h;
      HOST_FL_REUNIT_SVD_ABS_ERROR = svd_abs_error_h;
      HOST_FL_MAX_ERROR = max_error_h;     
      HOST_FL_CHECK_UNITARIZATION = check_unitarization_h;

      not_set = false;
    }
    checkCudaError();
#else
    errorQuda("Unitarization has not been built");
#endif
    return;
  }


  template <typename Float, typename G>
  struct ProjectSU3Arg {
    int threads; // number of active threads required
    G u;
    Float tol;
    int *fails;
    int X[4];
    ProjectSU3Arg(G u, const GaugeField &meta, Float tol, int *fails) 
      : u(u), tol(tol), fails(fails) {
      for(int dir=0; dir<4; ++dir) X[dir] = meta.X()[dir];
      threads = meta.VolumeCB();
    }
  };

  template<typename Float, typename G>
  __global__ void ProjectSU3kernel(ProjectSU3Arg<Float,G> arg){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = blockIdx.y;
    if(idx >= arg.threads) return;
    
    typedef typename ComplexTypeId<Float>::Type Cmplx;
    Matrix<Cmplx,3> u;

    for (int mu = 0; mu < 4; mu++) { 
      arg.u.load((Float*)(u.data),idx, mu, parity);
      polarSu3<Cmplx,Float>(u, arg.tol);

      // count number of failures
      if(isUnitary(u, arg.tol) == false) atomicAdd(arg.fails, 1);

      arg.u.save((Float*)(u.data),idx, mu, parity); 
    }
  }

  template<typename Float, typename G>
  class ProjectSU3 : Tunable {    
    ProjectSU3Arg<Float,G> arg;
    
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }
    
    // don't tune the grid dimension
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }
    
  public:
    ProjectSU3(ProjectSU3Arg<Float,G> &arg) : arg(arg) { }
    
    void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ProjectSU3kernel<Float,G><<<tp.grid, tp.block, 0, stream>>>(arg);
    }
    void preTune() { arg.u.save(); }
    void postTune() { arg.u.load(); }
  
    long long flops() const { return 0; } // depends on number of iterations
    long long bytes() const { return 4ll * arg.threads * arg.u.Bytes(); }
    
    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << arg.X[0] << "x" << arg.X[1] << "x" << arg.X[2] << "x" << arg.X[3];
      aux << "threads=" << arg.threads << ",prec=" << sizeof(Float);
      return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
    }
  };
  

  template <typename Float>
  void projectSU3(cudaGaugeField &u, double tol, int *fails) {
    if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
      ProjectSU3Arg<Float,G> arg(G(u), u, static_cast<Float>(tol), fails);
      ProjectSU3<Float,G> project(arg);
      project.apply(0);
      cudaDeviceSynchronize();
      checkCudaError();
    } else {
      errorQuda("Reconstruct %d not supported", u.Reconstruct());
    }
  }
  
  void projectSU3(cudaGaugeField &u, double tol, int *fails) {
#ifdef GPU_UNITARIZE
    // check the the field doesn't have staggered phases applied
    if (u.StaggeredPhaseApplied()) 
      errorQuda("Cannot project gauge field with staggered phases applied");

    if (u.Precision() == QUDA_DOUBLE_PRECISION) {
      projectSU3<double>(u, tol, fails);
    } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
      projectSU3<float>(u, tol, fails);      
    } else {
      errorQuda("Precision %d not supported", u.Precision());
    }
#else
    errorQuda("Unitarization has not been built");
#endif
  }

} // namespace quda

