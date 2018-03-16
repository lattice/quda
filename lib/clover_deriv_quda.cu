#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <cassert>

/**
   @file clover_deriv_quda.cu

   @brief This kernel has been a bit of a pain to optimize since it is
   excessively register bound.  To reduce register pressure we use
   shared memory to help offload some of this pressure.  Annoyingly,
   the optimal approach for CUDA 8.0 is not the same as CUDA 7.5, so
   implementation is compiler version dependent.  The CUDA 8.0 optimal
   code runs 10x slower on 7.5, though the 7.5 code runs fine on 8.0.

   CUDA >= 8.0
   - Used shared memory for force accumulator matrix
   - Template mu / nu to prevent register spilling of indexing arrays
   - Force the computation routine to inline

   CUDA <= 7.5
   - Used shared memory for force accumulator matrix
   - Keep mu/nu dynamic and use shared memory to store indexing arrays
   - Do not inline computation routine

   For the shared-memory dynamic indexing arrays, we use chars, since
   the array is 4-d, a 4-d coordinate can be stored in a single word
   which means that we will not have to worry about bank conflicts,
   and the shared array can be passed to the usual indexing routines
   (getCoordsExtended and linkIndexShift) with no code changes.  This
   strategy works as long as each local lattice coordinate is less
   than 256.
 */


#if (CUDA_VERSION < 8000)
#define DYNAMIC_MU_NU
#endif

// Use shared memory for the force accumulator matrix
#define SHARED_ACCUMULATOR


#ifdef DYNAMIC_MU_NU

// When using dynamic mu/nu indexing, to avoid local spills use shared
// memory for the per thread indexing arrays.
// FIXME for reasons I don't understand, the shared array breaks in multi-GPU mode
//#define SHARED_ARRAY

#endif // DYNAMIC_MU_NU


namespace quda {

#ifdef SHARED_ACCUMULATOR

#define DECLARE_LINK(U)							\
  extern __shared__ int s[];						\
  real *U = (real*)s;							\
  {									\
    const int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x; \
    const int block = blockDim.x * blockDim.y * blockDim.z;		\
    for (int i=0; i<18; i++) force[i*block + tid] = 0.0;		\
  }

#define LINK real*

  template <typename real, typename Link>
  __device__ inline void axpy(real a, const real *x, Link &y) {
    const int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
    const int block = blockDim.x * blockDim.y * blockDim.z;
#pragma unroll
    for (int i=0; i<9; i++) {
      y.data[i] += a * complex<real>(x[(2*i+0)*block + tid], x[(2*i+1)*block + tid]);
    }
  }

  template <typename real, typename Link>
  __device__ inline void operator+=(real *y, const Link &x) {
    const int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
    const int block = blockDim.x * blockDim.y * blockDim.z;
#pragma unroll
    for (int i=0; i<9; i++) {
      y[(2*i+0)*block + tid] += x.data[i].real();
      y[(2*i+1)*block + tid] += x.data[i].imag();
    }
  }

  template <typename real, typename Link>
  __device__ inline void operator-=(real *y, const Link &x) {
    const int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
    const int block = blockDim.x * blockDim.y * blockDim.z;
#pragma unroll
    for (int i=0; i<9; i++) {
      y[(2*i+0)*block + tid] -= x.data[i].real();
      y[(2*i+1)*block + tid] -= x.data[i].imag();
    }
  }

#else

#define DECLARE_LINK(U) Link U;

#define LINK Link &

  template <typename real, typename Link>
  __device__ inline void axpy(real a, const Link &x, Link &y) { y += a*x;  }

#endif

#if defined(SHARED_ARRAY) && defined(SHARED_ACCUMULATOR)

#define DECLARE_ARRAY(d, idx)						\
  unsigned char *d;							\
  {									\
    extern __shared__ int s[];						\
    int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x; \
    int block = blockDim.x*blockDim.y*blockDim.z;			\
    int offset = 18*block*sizeof(real)/sizeof(int) + idx*block + tid;	\
    s[offset] = 0;							\
    d = (unsigned char*)&s[offset];					\
  }
#elif defined(SHARED_ARRAY)
#error Cannot use SHARED_ARRAY with SHARED_ACCUMULATOR
#else
#define DECLARE_ARRAY(d, idx)						\
  int d[4] = {0, 0, 0, 0};
#endif


#ifdef GPU_CLOVER_DIRAC

  template<class Float, typename Force, typename Gauge, typename Oprod>
  struct CloverDerivArg
  {
    int X[4];
    int E[4];
    int border[4];
    Float coeff;
    int parity;
    int volumeCB;

    Force force;
    Gauge gauge;
    Oprod oprod;

    CloverDerivArg(const Force& force, const Gauge& gauge, const Oprod& oprod,
		   const int *X_, const int *E_,
		   double coeff, int parity) :
      coeff(coeff), parity(parity), volumeCB(force.volumeCB),
      force(force), gauge(gauge), oprod(oprod)
    {
      for(int dir=0; dir<4; ++dir) {
	this->X[dir] = X_[dir];
	this->E[dir] = E_[dir];
	this->border[dir] = (E_[dir] - X_[dir])/2;
      }
    }
  };


#ifdef DYNAMIC_MU_NU
  template <typename real, typename Arg, typename Link>
  __device__ void computeForce(LINK force, Arg &arg, int xIndex, int yIndex, int mu, int nu) {
#else
  template <typename real, typename Arg, int mu, int nu, typename Link>
  __device__ __forceinline__ void computeForce(LINK force, Arg &arg, int xIndex, int yIndex) {
#endif

    int otherparity = (1-arg.parity);

    const int tidx = mu > nu ? (mu-1)*mu/2 + nu : (nu-1)*nu/2 + mu;

    if (yIndex == 0) { // do "this" force

      DECLARE_ARRAY(x, 1);
      getCoordsExtended(x, xIndex, arg.X, arg.parity, arg.border);

      // U[mu](x) U[nu](x+mu) U[*mu](x+nu) U[*nu](x) Oprod(x)
      {
	DECLARE_ARRAY(d,0);

	// load U(x)_(+mu)
	Link U1 = arg.gauge(mu, linkIndexShift(x, d, arg.E), arg.parity);

	// load U(x+mu)_(+nu)
	d[mu]++;
	Link U2 = arg.gauge(nu, linkIndexShift(x, d, arg.E), otherparity);
	d[mu]--;

	// load U(x+nu)_(+mu)
	d[nu]++;
	Link U3 = arg.gauge(mu, linkIndexShift(x, d, arg.E), otherparity);
	d[nu]--;
      
	// load U(x)_(+nu)
	Link U4 = arg.gauge(nu, linkIndexShift(x, d, arg.E), arg.parity);

	// load Oprod
	Link Oprod1 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), arg.parity);

        if (nu < mu) force -= U1*U2*conj(U3)*conj(U4)*Oprod1;
	else   	     force += U1*U2*conj(U3)*conj(U4)*Oprod1;

	d[mu]++; d[nu]++;
	Link Oprod2 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), arg.parity);
	d[mu]--; d[nu]--;

        if (nu < mu) force -= U1*U2*Oprod2*conj(U3)*conj(U4);
	else         force += U1*U2*Oprod2*conj(U3)*conj(U4);
      }
 
      {
	DECLARE_ARRAY(d,0);

	// load U(x-nu)(+nu)
	d[nu]--;
	Link U1 = arg.gauge(nu, linkIndexShift(x, d, arg.E), otherparity);
	d[nu]++;

	// load U(x-nu)(+mu)
	d[nu]--;
	Link U2 = arg.gauge(mu, linkIndexShift(x, d, arg.E), otherparity);
	d[nu]++;

	// load U(x+mu-nu)(nu)
	d[mu]++; d[nu]--;
	Link U3 = arg.gauge(nu, linkIndexShift(x, d, arg.E), arg.parity);
	d[mu]--; d[nu]++;

	// load U(x)_(+mu)
	Link U4 = arg.gauge(mu, linkIndexShift(x, d, arg.E), arg.parity);

	d[mu]++; d[nu]--;
	Link Oprod1 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), arg.parity);
	d[mu]--; d[nu]++;

        if (nu < mu) force += conj(U1)*U2*Oprod1*U3*conj(U4);
	else         force -= conj(U1)*U2*Oprod1*U3*conj(U4);

	Link Oprod4 = arg.oprod(tidx, linkIndexShift(x, d, arg.E), arg.parity);

        if (nu < mu) force += Oprod4*conj(U1)*U2*U3*conj(U4);
	else         force -= Oprod4*conj(U1)*U2*U3*conj(U4);
      }

    } else { // else do other force

      DECLARE_ARRAY(y, 1);
      getCoordsExtended(y, xIndex, arg.X, otherparity, arg.border);

      {
	DECLARE_ARRAY(d,0);

	// load U(x)_(+mu)
	Link U1 = arg.gauge(mu, linkIndexShift(y, d, arg.E), otherparity);

	// load U(x+mu)_(+nu)
	d[mu]++;
	Link U2 = arg.gauge(nu, linkIndexShift(y, d, arg.E), arg.parity);
	d[mu]--;

	// load U(x+nu)_(+mu)
	d[nu]++;
	Link U3 = arg.gauge(mu, linkIndexShift(y, d, arg.E), arg.parity);
	d[nu]--;

	// load U(x)_(+nu)
	Link U4 = arg.gauge(nu, linkIndexShift(y, d, arg.E), otherparity);

	// load opposite parity Oprod
	d[nu]++;
	Link Oprod3 = arg.oprod(tidx, linkIndexShift(y, d, arg.E), arg.parity);
	d[nu]--;

	if (nu < mu) force -= U1*U2*conj(U3)*Oprod3*conj(U4);
	else         force += U1*U2*conj(U3)*Oprod3*conj(U4);

	// load Oprod(x+mu)
	d[mu]++;
	Link Oprod4 = arg.oprod(tidx, linkIndexShift(y, d, arg.E), arg.parity);
	d[mu]--;

	if (nu < mu) force -= U1*Oprod4*U2*conj(U3)*conj(U4);
	else         force += U1*Oprod4*U2*conj(U3)*conj(U4);
      }

      // Lower leaf
      // U[nu*](x-nu) U[mu](x-nu) U[nu](x+mu-nu) Oprod(x+mu) U[*mu](x)
      {
	DECLARE_ARRAY(d,0);

	// load U(x-nu)(+nu)
	d[nu]--;
	Link U1 = arg.gauge(nu, linkIndexShift(y, d, arg.E), arg.parity);
	d[nu]++;

	// load U(x-nu)(+mu)
	d[nu]--;
	Link U2 = arg.gauge(mu, linkIndexShift(y, d, arg.E), arg.parity);
	d[nu]++;

	// load U(x+mu-nu)(nu)
	d[mu]++; d[nu]--;
	Link U3 = arg.gauge(nu, linkIndexShift(y, d, arg.E), otherparity);
	d[mu]--; d[nu]++;

	// load U(x)_(+mu)
	Link U4 = arg.gauge(mu, linkIndexShift(y, d, arg.E), otherparity);

	// load Oprod(x+mu)
	d[mu]++;
	Link Oprod1 = arg.oprod(tidx, linkIndexShift(y, d, arg.E), arg.parity);
	d[mu]--;

	if (nu < mu) force += conj(U1)*U2*U3*Oprod1*conj(U4);
	else         force -= conj(U1)*U2*U3*Oprod1*conj(U4);

	d[nu]--;
	Link Oprod2 = arg.oprod(tidx, linkIndexShift(y, d, arg.E), arg.parity);
	d[nu]++;

	if (nu < mu) force += conj(U1)*Oprod2*U2*U3*conj(U4);
	else         force -= conj(U1)*Oprod2*U2*U3*conj(U4);
      }

    }

  }

  template<typename real, typename Arg>
  __global__ void cloverDerivativeKernel(Arg arg)
  {
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= arg.volumeCB) return;

    // y index determines whether we're updating arg.parity or (1-arg.parity)
    int yIndex = threadIdx.y + blockIdx.y*blockDim.y;
    if (yIndex >= 2) return;

    // mu index is mapped from z thread index
    int mu = threadIdx.z + blockIdx.z*blockDim.z;
    if (mu >= 4) return;

    typedef complex<real> Complex;
    typedef Matrix<Complex,3> Link;

    DECLARE_LINK(force);

#ifdef DYNAMIC_MU_NU
    for (int nu=0; nu<4; nu++) {
      if (mu==nu) continue;
      computeForce<real,Arg,Link>(force, arg, index, yIndex, mu, nu);
    }
#else
    switch(mu) {
    case 0:
      computeForce<real,Arg,0,1,Link>(force, arg, index, yIndex);
      computeForce<real,Arg,0,2,Link>(force, arg, index, yIndex);
      computeForce<real,Arg,0,3,Link>(force, arg, index, yIndex);
      break;
    case 1:
      computeForce<real,Arg,1,0,Link>(force, arg, index, yIndex);
      computeForce<real,Arg,1,3,Link>(force, arg, index, yIndex);
      computeForce<real,Arg,1,2,Link>(force, arg, index, yIndex);
      break;
    case 2:
      computeForce<real,Arg,2,3,Link>(force, arg, index, yIndex);
      computeForce<real,Arg,2,0,Link>(force, arg, index, yIndex);
      computeForce<real,Arg,2,1,Link>(force, arg, index, yIndex);
      break;
    case 3:
      computeForce<real,Arg,3,2,Link>(force, arg, index, yIndex);
      computeForce<real,Arg,3,1,Link>(force, arg, index, yIndex);
      computeForce<real,Arg,3,0,Link>(force, arg, index, yIndex);
      break;
    }
#endif

    // Write to array
    Link F;
    arg.force.load((real*)(F.data), index, mu, yIndex == 0 ? arg.parity : 1-arg.parity);
    axpy(arg.coeff, force, F);
    arg.force.save((real*)(F.data), index, mu, yIndex == 0 ? arg.parity : 1-arg.parity);

    return;
  } // cloverDerivativeKernel
  
  
  template<typename Float, typename Arg>
  class CloverDerivative : public TunableVectorY {
    
  private:
    Arg arg;
    const GaugeField &meta;

#if defined(SHARED_ACCUMULATOR) && defined(SHARED_ARRAY)
    unsigned int sharedBytesPerThread() const { return 18*sizeof(Float) + 8; }
#elif defined(SHARED_ACCUMULATOR)
    unsigned int sharedBytesPerThread() const { return 18*sizeof(Float); }
#else
    unsigned int sharedBytesPerThread() const { return 0; }
#endif
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    unsigned int minThreads() const { return arg.volumeCB; }
    bool tuneGridDim() const { return false; } // don't tune the grid dimension

  public:
    CloverDerivative(const Arg &arg, const GaugeField &meta) : TunableVectorY(2), arg(arg), meta(meta) {
      writeAuxString("threads=%d,prec=%lu,fstride=%d,gstride=%d,ostride=%d",
		     arg.volumeCB,sizeof(Float),arg.force.stride,
		     arg.gauge.stride,arg.oprod.stride);
    }
    virtual ~CloverDerivative() {}

    void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      cloverDerivativeKernel<Float><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
    } // apply

    bool advanceBlockDim(TuneParam &param) const {
      dim3 block = param.block;
      dim3 grid = param.grid;
      bool rtn = TunableVectorY::advanceBlockDim(param);
      param.block.z = block.z;
      param.grid.z = grid.z;

      if (!rtn) {
	if (param.block.z < 4) {
	  param.block.z++;
	  param.grid.z = (4 + param.block.z - 1) / param.block.z;
	  rtn = true;
	} else {
	  param.block.z = 1;
	  param.grid.z = 4;
	  rtn = false;
	}
      }
      return rtn;
    }

    void initTuneParam(TuneParam &param) const {
      TunableVectorY::initTuneParam(param);
      param.block.y = 1;
      param.block.z = 1;
      param.grid.y = 2;
      param.grid.z = 4;
    }

    void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

    // The force field is updated so we must preserve its initial state
    void preTune() { arg.force.save(); } 
    void postTune(){ arg.force.load(); } 

    long long flops() const { return 16 * 198 * 3 * 4 * 2 * (long long)arg.volumeCB; }
    long long bytes() const { return ((8*arg.gauge.Bytes() + 4*arg.oprod.Bytes())*3 + 2*arg.force.Bytes()) * 4 * 2 * arg.volumeCB; }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };

  
  template<typename Float>
  void cloverDerivative(cudaGaugeField &force,
			cudaGaugeField &gauge,
			cudaGaugeField &oprod,
			double coeff, int parity) {
 
    if (oprod.Reconstruct() != QUDA_RECONSTRUCT_NO) 
      errorQuda("Force field does not support reconstruction");
    
    if (force.Order() != oprod.Order()) 
      errorQuda("Force and Oprod orders must match");
    
    if (force.Reconstruct() != QUDA_RECONSTRUCT_NO) 
      errorQuda("Force field does not support reconstruction");

    if (force.Order() == QUDA_FLOAT2_GAUGE_ORDER){
      typedef gauge::FloatNOrder<Float, 18, 2, 18> F;
      typedef gauge::FloatNOrder<Float, 18, 2, 18> O;

      if (gauge.isNative()) {
	if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	  typedef CloverDerivArg<Float,F,G,O> Arg;
	  Arg arg(F(force), G(gauge), O(oprod), force.X(), oprod.X(), coeff, parity);
	  CloverDerivative<Float, Arg> deriv(arg, gauge);
	  deriv.apply(0);
#if 0
	} else if (gauge.Reconstruct() == QUDA_RECONSTRUCT_12) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
	  typedef CloverDerivArg<Float,F,G,O> Arg;
	  Arg arg(F(force), G(gauge), O(oprod), force.X(), oprod.X(), coeff, parity);
	  CloverDerivative<Float, Arg> deriv(arg, gauge);
	  deriv.apply(0);
#endif
	} else {
	  errorQuda("Reconstruction type %d not supported",gauge.Reconstruct());
	}
      } else {
	errorQuda("Gauge order %d not supported", gauge.Order());
      }
    } else {
      errorQuda("Force order %d not supported", force.Order());
    } // force / oprod order

    cudaDeviceSynchronize();
  }
#endif // GPU_CLOVER

void cloverDerivative(cudaGaugeField &force,   
		      cudaGaugeField &gauge,
		      cudaGaugeField &oprod,
		      double coeff, QudaParity parity)
{
#ifdef GPU_CLOVER_DIRAC
  assert(oprod.Geometry() == QUDA_TENSOR_GEOMETRY);
  assert(force.Geometry() == QUDA_VECTOR_GEOMETRY);

  for (int d=0; d<4; d++) {
    if (oprod.X()[d] != gauge.X()[d])
      errorQuda("Incompatible extended dimensions d=%d gauge=%d oprod=%d", d, gauge.X()[d], oprod.X()[d]);
  }

  int device_parity = (parity == QUDA_EVEN_PARITY) ? 0 : 1;

  if(force.Precision() == QUDA_DOUBLE_PRECISION){
    cloverDerivative<double>(force, gauge, oprod, coeff, device_parity);
#if 0
  } else if (force.Precision() == QUDA_SINGLE_PRECISION){
    cloverDerivative<float>(force, gauge, oprod, coeff, device_parity);
#endif
  } else {
    errorQuda("Precision %d not supported", force.Precision());
  }

  return;
#else
  errorQuda("Clover has not been built");
#endif
}              


} // namespace quda
