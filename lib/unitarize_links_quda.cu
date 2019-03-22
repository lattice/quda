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
 
  static const int max_iter_newton = 20;
  static const int max_iter = 20;

  static double unitarize_eps = 1e-14;
  static double max_error = 1e-10;
  static int reunit_allow_svd = 1;
  static int reunit_svd_only  = 0;
  static double svd_rel_error = 1e-6;
  static double svd_abs_error = 1e-6;

  template <typename Out, typename In>
  struct UnitarizeLinksArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    Out output;
    const In input;
    int *fails;
    const int max_iter;
    const double unitarize_eps;
    const double max_error;
    const int reunit_allow_svd;
    const int reunit_svd_only;
    const double svd_rel_error;
    const double svd_abs_error;
    const static bool check_unitarization = true;

    UnitarizeLinksArg(Out &output, const In &input, const GaugeField &data, int* fails,
		      int max_iter, double unitarize_eps, double max_error,
		      int reunit_allow_svd, int reunit_svd_only, double svd_rel_error,
		      double svd_abs_error)
      : threads(data.VolumeCB()), output(output), input(input), fails(fails), unitarize_eps(unitarize_eps),
	max_iter(max_iter), max_error(max_error), reunit_allow_svd(reunit_allow_svd),
	reunit_svd_only(reunit_svd_only), svd_rel_error(svd_rel_error),
	svd_abs_error(svd_abs_error)
    {
      for (int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
    }
  };

#endif // GPU_UNITARIZE

  void setUnitarizeLinksConstants(double unitarize_eps_, double max_error_,
				  bool reunit_allow_svd_, bool reunit_svd_only_,
				  double svd_rel_error_, double svd_abs_error_) {
#ifdef GPU_UNITARIZE
    unitarize_eps = unitarize_eps_;
    max_error = max_error_;
    reunit_allow_svd = reunit_allow_svd_;
    reunit_svd_only = reunit_svd_only_;
    svd_rel_error = svd_rel_error_;
    svd_abs_error = svd_abs_error_;
#else
    errorQuda("Unitarization has not been built");
#endif
  }

#ifdef GPU_UNITARIZE
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
  template<class Float, typename Arg>
  __device__  __host__ 
  bool reciprocalRoot(const Matrix<complex<Float>,3>& q, Matrix<complex<Float>,3>* res, Arg &arg){

    Matrix<complex<Float>,3> qsq, tempq;

    Float c[3];
    Float g[3];

    const Float one_third = 0.333333333333333333333;
    const Float one_ninth = 0.111111111111111111111;
    const Float one_eighteenth = 0.055555555555555555555;

    qsq = q*q;
    tempq = qsq*q;

    c[0] = getTrace(q).x;
    c[1] = getTrace(qsq).x * 0.5;
    c[2] = getTrace(tempq).x * one_third;;

    g[0] = g[1] = g[2] = c[0] * one_third;
    Float r,s,theta;
    s = c[1]*one_third - c[0]*c[0]*one_eighteenth;

    Float cosTheta;
    if(fabs(s) >= arg.unitarize_eps){ // faster when this conditional is removed?
      const Float rsqrt_s = rsqrt(s);
      r = c[2]*0.5 - (c[0]*one_third)*(c[1] - c[0]*c[0]*one_ninth);
      cosTheta = r*rsqrt_s*rsqrt_s*rsqrt_s;

      if(fabs(cosTheta) >= 1.0){
	theta = (r > 0) ? 0.0 : FL_UNITARIZE_PI;
      }else{ 
	theta = acos(cosTheta); // this is the primary performance limiter
      }

      const Float sqrt_s = s*rsqrt_s;

#if 0 // experimental version
      Float as, ac;
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
    Float det = getDeterminant(q).x;
    if( fabs(det) < arg.svd_abs_error) return false;
    if( checkRelativeError(g[0]*g[1]*g[2],det,arg.svd_rel_error) == false ) return false;


    // At this point we have finished with the c's 
    // use these to store sqrt(g)
    for(int i=0; i<3; ++i) c[i] = sqrt(g[i]);

    // done with the g's, use these to store u, v, w
    g[0] = c[0]+c[1]+c[2];
    g[1] = c[0]*c[1] + c[0]*c[2] + c[1]*c[2];
    g[2] = c[0]*c[1]*c[2];
        
    const Float denominator  = 1.0 / ( g[2]*(g[0]*g[1]-g[2]) );
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




  template<class Float, typename Arg>
  __host__ __device__
  bool unitarizeLinkMILC(const Matrix<complex<Float>,3>& in, Matrix<complex<Float>,3>* const result, Arg &arg)
  {
    Matrix<complex<Float>,3> u;
    if( !arg.reunit_svd_only ){
      if( reciprocalRoot<Float>(conj(in)*in,&u,arg) ){
	*result = in*u;
	return true;
      }
    }

    // If we've got this far, then the Caley-Hamilton unitarization 
    // has failed. If SVD is not allowed, the unitarization has failed.
    if( !arg.reunit_allow_svd ) return false;

    Matrix<complex<Float>,3> v;
    Float singular_values[3];
    computeSVD<Float>(in, u, v, singular_values);
    *result = u*conj(v);
    return true;
  } // unitarizeMILC
    

  template<class Float>
  __host__ __device__
  bool unitarizeLinkSVD(const Matrix<complex<Float>,3>& in, Matrix<complex<Float>,3>* const result,
			const double max_error)
  {
    Matrix<complex<Float>,3> u, v;
    Float singular_values[3];
    computeSVD<Float>(in, u, v, singular_values); // should pass pointers to u,v I guess

    *result = u*conj(v);

    if (isUnitary(*result,max_error)==false)
      {
	printf("ERROR: Link unitarity test failed\n");
	printf("TOLERANCE: %g\n", max_error);
	return false;
      }
    return true;
  }


  template<class Float>
  __host__ __device__
  bool unitarizeLinkNewton(const Matrix<complex<Float>,3>& in, Matrix<complex<Float>,3>* const result, int max_iter)
  {
    Matrix<complex<Float>,3> u, uinv;
    u = in;

    for(int i=0; i<max_iter; ++i){
      uinv = inverse(u);
      u = 0.5*(u + conj(uinv));
    }

    if(isUnitarizedLinkConsistent(in,u,0.0000001)==false)
      {
        printf("ERROR: Unitarized link is not consistent with incoming link\n");
	return false;
      }
    *result = u;

    return true;
  }   

#endif // GPU_UNITARIZE

  void unitarizeLinksCPU(cpuGaugeField &outfield, const cpuGaugeField& infield)
  {
#ifdef GPU_UNITARIZE
    if (infield.Precision() != outfield.Precision())
      errorQuda("Precisions must match (out=%d != in=%d)", outfield.Precision(), infield.Precision());
    
    int num_failures = 0;
    Matrix<complex<double>,3> inlink, outlink;
      
    for (int i=0; i<infield.Volume(); ++i){
      for (int dir=0; dir<4; ++dir){
	if (infield.Precision() == QUDA_SINGLE_PRECISION){
	  copyArrayToLink(&inlink, ((float*)(infield.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	  if( unitarizeLinkNewton<double>(inlink, &outlink, max_iter_newton) == false ) num_failures++;
	  copyLinkToArray(((float*)(outfield.Gauge_p()) + (i*4 + dir)*18), outlink); 
	} else if (infield.Precision() == QUDA_DOUBLE_PRECISION){
	  copyArrayToLink(&inlink, ((double*)(infield.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	  if( unitarizeLinkNewton<double>(inlink, &outlink, max_iter_newton) == false ) num_failures++;
	  copyLinkToArray(((double*)(outfield.Gauge_p()) + (i*4 + dir)*18), outlink); 
	} // precision?
      } // dir
    }  // loop over volume
    return;
#else
    errorQuda("Unitarization has not been built");
#endif
  }

    
  // CPU function which checks that the gauge field is unitary
  bool isUnitary(const cpuGaugeField& field, double max_error)
  {
#ifdef GPU_UNITARIZE
    Matrix<complex<double>,3> link, identity;
      
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
#else
    errorQuda("Unitarization has not been built");
    return false;
#endif
  } // is unitary


#ifdef GPU_UNITARIZE

  template<typename Float, typename Out, typename In>
  __global__ void DoUnitarizedLink(UnitarizeLinksArg<Out,In> arg){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y + blockIdx.y*blockDim.y;
    int mu = threadIdx.z + blockIdx.z*blockDim.z;
    if (idx >= arg.threads) return;
    if (mu >= 4) return;

    // result is always in double precision
    Matrix<complex<double>,3> v, result;
    Matrix<complex<Float>,3> tmp = arg.input(mu, idx, parity);

    v = tmp;
    unitarizeLinkMILC(v, &result, arg);
    if (arg.check_unitarization) {
      if (isUnitary(result,arg.max_error) == false) atomicAdd(arg.fails, 1);
    }
    tmp = result;

    arg.output(mu, idx, parity) = tmp;
  }



  template<typename Float, typename Out, typename In>
  class UnitarizeLinks : TunableVectorYZ {
    UnitarizeLinksArg<Out,In> arg;
    const GaugeField &meta;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    // don't tune the grid dimension
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

  public:
    UnitarizeLinks(UnitarizeLinksArg<Out,In> &arg, const GaugeField &meta)
      : TunableVectorYZ(2,4), arg(arg), meta(meta) { }
    
    void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      DoUnitarizedLink<Float,Out,In><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
    }
    void preTune() { if (arg.input.gauge == arg.output.gauge) arg.output.save(); }
    void postTune() {
      if (arg.input.gauge == arg.output.gauge) arg.output.load();
      cudaMemset(arg.fails, 0, sizeof(int)); // reset fails counter
    }
    
    long long flops() const { 
      // Accounted only the minimum flops for the case reunitarize_svd_only=0
      return 4ll * 2 * arg.threads * 1147;
    }
    long long bytes() const { return 4ll * 2 * arg.threads * (arg.input.Bytes() + arg.output.Bytes()); }
    
    TuneKey tuneKey() const {
      std::stringstream aux;
      aux << "threads=" << arg.threads << ",prec=" << sizeof(Float);
      return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
    }  
  }; 
  
  
  template<typename Float, typename Out, typename In>
  void unitarizeLinks(Out output, const In input, const cudaGaugeField& meta, int* fails) {
    UnitarizeLinksArg<Out,In> arg(output, input, meta, fails, max_iter, unitarize_eps, max_error,
                                  reunit_allow_svd, reunit_svd_only, svd_rel_error, svd_abs_error);
    UnitarizeLinks<Float, Out, In> unitlinks(arg, meta);
    unitlinks.apply(0);
    qudaDeviceSynchronize(); // need to synchronize to ensure failure write has completed
  }
  
template<typename Float>
void unitarizeLinks(cudaGaugeField& output, const cudaGaugeField &input, int* fails) {

  if( output.isNative() && input.isNative() ) {
    if(output.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type Out;

      if(input.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type In;
	unitarizeLinks<Float>(Out(output), In(input), input, fails) ;
      } else if(input.Reconstruct() == QUDA_RECONSTRUCT_12) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type In;
	unitarizeLinks<Float>(Out(output), In(input), input, fails) ;
      } else if(input.Reconstruct() == QUDA_RECONSTRUCT_8) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type In;
	unitarizeLinks<Float>(Out(output), In(input), input, fails) ;
      } else {
	errorQuda("Reconstruction type %d of gauge field not supported", input.Reconstruct());
      }

    } else if(output.Reconstruct() == QUDA_RECONSTRUCT_12){
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type Out;

      if(input.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type In;
	unitarizeLinks<Float>(Out(output), In(input), input, fails) ;
      } else if(input.Reconstruct() == QUDA_RECONSTRUCT_12) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type In;
	unitarizeLinks<Float>(Out(output), In(input), input, fails) ;
      } else if(input.Reconstruct() == QUDA_RECONSTRUCT_8) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type In;
	unitarizeLinks<Float>(Out(output), In(input), input, fails) ;
      } else {
	errorQuda("Reconstruction type %d of gauge field not supported", input.Reconstruct());
      }


    } else if(output.Reconstruct() == QUDA_RECONSTRUCT_8){
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type Out;

      if(input.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type In;
	unitarizeLinks<Float>(Out(output), In(input), input, fails) ;
      } else if(input.Reconstruct() == QUDA_RECONSTRUCT_12) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type In;
	unitarizeLinks<Float>(Out(output), In(input), input, fails) ;
      } else if(input.Reconstruct() == QUDA_RECONSTRUCT_8) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type In;
	unitarizeLinks<Float>(Out(output), In(input), input, fails) ;
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
  
#endif // GPU_UNITARIZE
  
  void unitarizeLinks(cudaGaugeField& output, const cudaGaugeField &input, int* fails) {
#ifdef GPU_UNITARIZE
    if (input.Precision() != output.Precision()) 
      errorQuda("input (%d) and output (%d) precisions must match", output.Precision(), input.Precision());

    if (input.Precision() == QUDA_SINGLE_PRECISION) {
      unitarizeLinks<float>(output, input, fails);
    } else if(input.Precision() == QUDA_DOUBLE_PRECISION) {
      unitarizeLinks<double>(output, input, fails);
    } else {
      errorQuda("Precision %d not supported", input.Precision());
    }
#else
    errorQuda("Unitarization has not been built");
#endif
  }

  void unitarizeLinks(cudaGaugeField &links, int* fails) {
    unitarizeLinks(links, links, fails);
  }


  template <typename Float, typename G>
  struct ProjectSU3Arg {
    int threads; // number of active threads required
    G u;
    Float tol;
    int *fails;
    ProjectSU3Arg(G u, const GaugeField &meta, Float tol, int *fails) 
      : threads(meta.VolumeCB()), u(u), tol(tol), fails(fails) { }
  };

  template<typename Float, typename G>
  __global__ void ProjectSU3kernel(ProjectSU3Arg<Float,G> arg){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y + blockIdx.y*blockDim.y;
    int mu = threadIdx.z + blockIdx.z*blockDim.z;
    if (idx >= arg.threads) return;
    if (mu >= 4) return;

    Matrix<complex<Float>,3> u = arg.u(mu, idx, parity);

    polarSu3<Float>(u, arg.tol);

    // count number of failures
    if (isUnitary(u, arg.tol) == false) {
      atomicAdd(arg.fails, 1);
    }

    arg.u(mu, idx, parity) = u;
  }

  template<typename Float, typename G>
  class ProjectSU3 : TunableVectorYZ {
    ProjectSU3Arg<Float,G> arg;
    const GaugeField &meta;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }
    
    // don't tune the grid dimension
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }
    
  public:
    ProjectSU3(ProjectSU3Arg<Float,G> &arg, const GaugeField &meta)
      : TunableVectorYZ(2, 4), arg(arg), meta(meta) { }
    
    void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE); //getVerbosity());
      ProjectSU3kernel<Float,G><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
    }
    void preTune() { arg.u.save(); }
    void postTune() {
      arg.u.load();
      cudaMemset(arg.fails, 0, sizeof(int)); // reset fails counter
    }
  
    long long flops() const { return 0; } // depends on number of iterations
    long long bytes() const { return 4ll * 2 * arg.threads * 2 * arg.u.Bytes(); }
    
    TuneKey tuneKey() const {
      std::stringstream aux;
      aux << "threads=" << arg.threads << ",prec=" << sizeof(Float);
      return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
    }
  };
  

  template <typename Float>
  void projectSU3(cudaGaugeField &u, double tol, int *fails) {
    if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
      ProjectSU3Arg<Float,G> arg(G(u), u, static_cast<Float>(tol), fails);
      ProjectSU3<Float,G> project(arg, u);
      project.apply(0);
      qudaDeviceSynchronize();
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

