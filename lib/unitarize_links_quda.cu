#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>

#include <tune_quda.h>
#include <quda_matrix.h>
#include <hisq_links_quda.h>

namespace quda{
#ifdef GPU_UNITARIZE

namespace{
  #include <svd_quda.h>
}

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

  void setUnitarizeLinksPadding(int input_padding_h, int output_padding_h)
  {
    cudaMemcpyToSymbol(INPUT_PADDING, &input_padding_h, sizeof(int));
    cudaMemcpyToSymbol(OUTPUT_PADDING, &output_padding_h, sizeof(int));
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
    
    if(mem_idx >= threads/2){
      mem_idx = mem_idx - (threads/2);
      inlink  = inlink_odd;
      outlink = outlink_odd;
    }

    // Unitarization is always done in double precision
    Matrix<double2,3> v, result;
    for(int dir=0; dir<4; ++dir){
      loadLinkVariableFromArray(inlink, dir, mem_idx, (threads/2)+INPUT_PADDING, &v); 
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
      writeLinkVariableToArray(result, dir, mem_idx, (threads/2)+OUTPUT_PADDING, outlink); 
    }
    return;
  }

  class UnitarizeLinksCuda : public Tunable {
  private:
    const cudaGaugeField &inField;
    cudaGaugeField &outField;
    int *fails;
    
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }
    
    // don't tune the grid dimension
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return inField.Volume(); }

  public:
    UnitarizeLinksCuda(const cudaGaugeField& inField, cudaGaugeField& outField,  int* fails) : 
      inField(inField), outField(outField), fails(fails) { ; }
    virtual ~UnitarizeLinksCuda() { ; }
    
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      
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
    
    long long flops() const { return 0; } // FIXME: add flops counter

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << inField.X()[0] << "x";
      vol << inField.X()[1] << "x";
      vol << inField.X()[2] << "x";
      vol << inField.X()[3] << "x";
      aux << "threads=" << inField.Volume() << ",prec=" << inField.Precision();
      aux << "stride=" << inField.Stride();
      return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
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





  


















  template <typename Gauge>
  struct UnitarizeLinksQudaArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    Gauge links;
    int *fails;
    UnitarizeLinksQudaArg(const Gauge &links, const GaugeField &data,  int* fails) : links(links), fails(fails) {
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];  
    threads = X[0]*X[1]*X[2]*X[3];
    }
  };


__device__ __host__ inline int linkIndex(int x[], const int X[4]) {
  int idx = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
  return idx;
}

__device__ __host__ inline void getCoords3(int x[4], int cb_index, const int X[4], int parity) {
  x[3] = cb_index/(X[2]*X[1]*X[0]/2);
  x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
  x[1] = (cb_index/(X[0]/2)) % X[1];
  x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

  return;
}


template<typename Float, typename Gauge>
__global__ void DoUnitarizedLink(UnitarizeLinksQudaArg<Gauge> arg){
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
  getCoords3(x, idx, X, parity);
 
  idx = linkIndex(x,X);
  Matrix<double2,3> v, result;
  Matrix<Cmplx,3> tmp;
  for (int mu = 0; mu < 4; mu++) { 
    arg.links.load((Float*)(tmp.data),idx, mu, parity);
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
        arg.links.save((Float*)(tmp.data),idx, mu, parity); 
  }
}



template<typename Float, typename Gauge>
  class UnitarizeLinksQuda : Tunable {    
    UnitarizeLinksQudaArg<Gauge> arg;
    
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }
    
    // don't tune the grid dimension
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

  public:
    UnitarizeLinksQuda(UnitarizeLinksQudaArg<Gauge> &arg) : arg(arg) { }


      void apply(const cudaStream_t &stream){
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          DoUnitarizedLink<Float,Gauge><<<tp.grid, tp.block, 0, stream>>>(arg);
      }
    void preTune() { ; }
    void postTune() { cudaMemset(arg.fails, 0, sizeof(int)); } // reset fails counter
    
    long long flops() const { return 0; } // FIXME: add flops counter

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << arg.X[0] << "x";
      vol << arg.X[1] << "x";
      vol << arg.X[2] << "x";
      vol << arg.X[3] << "x";
      aux << "threads=" << arg.threads << ",prec=" << sizeof(Float);
      return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
    }  
}; 


template<typename Float, typename Gauge>
void unitarizeLinksQuda( Gauge links,  cudaGaugeField& data, int* fails) {

      UnitarizeLinksQudaArg<Gauge> arg(links, data, fails);
      UnitarizeLinksQuda<Float, Gauge> unitlinks(arg) ;
      unitlinks.apply(0);
}

template<typename Float>
void unitarizeLinksQuda( cudaGaugeField& links, int* fails) {

        if(links.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
        if(links.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        //printf("QUDA_RECONSTRUCT_NO\n");
          unitarizeLinksQuda<Float>(FloatNOrder<Float, 18, 2, 18>(links), links, fails) ;
        } else if(links.Reconstruct() == QUDA_RECONSTRUCT_12){
        //printf("QUDA_RECONSTRUCT_12\n");
          unitarizeLinksQuda<Float>(FloatNOrder<Float, 18, 2, 12>(links), links, fails) ;
        } else if(links.Reconstruct() == QUDA_RECONSTRUCT_8){
        //printf("QUDA_RECONSTRUCT_8\n");
          unitarizeLinksQuda<Float>(FloatNOrder<Float, 18, 2, 8>(links), links, fails) ;
        
        } else {
          errorQuda("Reconstruction type %d of gauge field not supported", links.Reconstruct());
        }
      } else if(links.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
        if(links.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        //printf("QUDA_RECONSTRUCT_NO\n");
          unitarizeLinksQuda<Float>(FloatNOrder<Float, 18, 4, 18>(links), links, fails) ;
        } else if(links.Reconstruct() == QUDA_RECONSTRUCT_12){
        //printf("QUDA_RECONSTRUCT_12\n");
          unitarizeLinksQuda<Float>(FloatNOrder<Float, 18, 4, 12>(links), links, fails) ;
        } else if(links.Reconstruct() == QUDA_RECONSTRUCT_8){
        //printf("QUDA_RECONSTRUCT_8\n");
          unitarizeLinksQuda<Float>(FloatNOrder<Float, 18, 4, 8>(links), links, fails) ;
        } else {
          errorQuda("Reconstruction type %d of gauge field not supported", links.Reconstruct());
        }
      } else {
        errorQuda("Invalid Gauge Order\n");
      }
    }
#endif
  
  void unitarizeLinksQuda(cudaGaugeField& links, int* fails) {
#ifdef GPU_UNITARIZE
    if(links.Precision() == QUDA_HALF_PRECISION) {
      errorQuda("Half precision not supported\n");
    }
    if (links.Precision() == QUDA_SINGLE_PRECISION) {
      unitarizeLinksQuda<float>(links, fails);
    } else if(links.Precision() == QUDA_DOUBLE_PRECISION) {
      unitarizeLinksQuda<double>(links, fails);
    } else {
      errorQuda("Precision %d not supported", links.Precision());
    }
#else
    errorQuda("Unitarization has not been built");
#endif
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
  
} // namespace quda

