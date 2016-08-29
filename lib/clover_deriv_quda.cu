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

namespace quda {
  
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

    bool conjugate;      

    CloverDerivArg(const Force& force, const Gauge& gauge, const Oprod& oprod,
		   const int *X_, const int *E_,
		   double coeff, int parity, bool conjugate) :
      coeff(coeff), parity(parity), volumeCB(force.volumeCB),
      force(force), gauge(gauge), oprod(oprod), conjugate(conjugate)
    {
      for(int dir=0; dir<4; ++dir) {
	this->X[dir] = X_[dir];
	this->E[dir] = E_[dir];
	this->border[dir] = (E_[dir] - X_[dir])/2;
      }
    }
  };


#define SHARED_ACCUMULATOR

#ifdef SHARED_ACCUMULATOR

#define DECLARE_FORCE							\
  extern __shared__ int s[];						\
  real *force = (real*)s;						\
  {									\
    const int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x; \
    const int block = blockDim.x * blockDim.y * blockDim.z;		\
    for (int i=0; i<18; i++) force[i*block + tid] = 0.0;		\
  }

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
  __device__ inline void add(real *y, const Link &x) {
    const int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
    const int block = blockDim.x * blockDim.y * blockDim.z;
#pragma unroll
    for (int i=0; i<9; i++) {
      y[(2*i+0)*block + tid] += x.data[i].real();
      y[(2*i+1)*block + tid] += x.data[i].imag();
    }
  }

  template <typename real, typename Link>
  __device__ inline void sub(real *y, const Link &x) {
    const int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
    const int block = blockDim.x * blockDim.y * blockDim.z;
#pragma unroll
    for (int i=0; i<9; i++) {
      y[(2*i+0)*block + tid] -= x.data[i].real();
      y[(2*i+1)*block + tid] -= x.data[i].imag();
    }
  }

#else

  template <typename real, typename Link>
  __device__ inline void axpy(real a, const Link &x, Link &y) { y += a*x;  }

  template <typename Link>
  __device__ inline void add(Link &y, const Link &x) { y += x; }

  template <typename Link>
  __device__ inline void sub(Link &y, const Link &x) { y -= x; }

#endif

#ifdef SHARED_ACCUMULATOR
  template <typename real, bool isConjugate, typename Arg, int mu, int nu, typename Link>
  __device__ __forceinline__ void computeForce(real *force, Arg &arg, int xIndex, int yIndex) {
#else
  template <typename real, bool isConjugate, typename Arg, int mu, int nu, typename Link>
  __device__ __forceinline__ void computeForce(Link &force, Arg &arg, int xIndex, int yIndex) {
#endif

    int otherparity = (1-arg.parity);

    constexpr int tidx = mu > nu ? (mu-1)*mu/2 + nu : (nu-1)*nu/2 + mu;

    if (yIndex == 0) { // do "this" force

      int x[4] = {0, 0, 0, 0};
      getCoordsExtended(x, xIndex, arg.X, arg.parity, arg.border);

      // U[mu](x) U[nu](x+mu) U[*mu](x+nu) U[*nu](x) Oprod(x)
      {
	int d[4] = {0, 0, 0, 0};
	Link U1, U2, U3, U4, Oprod1, Oprod2;

	// load U(x)_(+mu)
	arg.gauge.load((real*)(U1.data), linkIndexShift(x, d, arg.E), mu, arg.parity);

	// load U(x+mu)_(+nu)
	d[mu]++;
	arg.gauge.load((real*)(U2.data), linkIndexShift(x, d, arg.E), nu, otherparity);
	d[mu]--;

	// load U(x+nu)_(+mu)
	d[nu]++;
	arg.gauge.load((real*)(U3.data), linkIndexShift(x, d, arg.E), mu, otherparity);
	d[nu]--;
      
	// load U(x)_(+nu)
	arg.gauge.load((real*)(U4.data), linkIndexShift(x, d, arg.E), nu, arg.parity);

	// load Oprod
	arg.oprod.load((real*)(Oprod1.data), linkIndexShift(x, d, arg.E), tidx, arg.parity);

	if (isConjugate) Oprod1 -= conj(Oprod1);

        if (nu < mu) sub(force, U1*U2*conj(U3)*conj(U4)*Oprod1);
	else   	     add(force, U1*U2*conj(U3)*conj(U4)*Oprod1);

	d[mu]++; d[nu]++;
	arg.oprod.load((real*)(Oprod2.data), linkIndexShift(x, d, arg.E), tidx, arg.parity);
	d[mu]--; d[nu]--;

	if (isConjugate) Oprod2 -= conj(Oprod2);

        if (nu < mu) sub(force, U1*U2*Oprod2*conj(U3)*conj(U4));
	else         add(force, U1*U2*Oprod2*conj(U3)*conj(U4));
      }
 
      {
	int d[4] = {0, 0, 0, 0};
	Link U1, U2, U3, U4, Oprod1, Oprod4;

	// load U(x-nu)(+nu)
	d[nu]--;
	arg.gauge.load((real*)(U1.data), linkIndexShift(x, d, arg.E), nu, otherparity);
	d[nu]++;

	// load U(x-nu)(+mu)
	d[nu]--;
	arg.gauge.load((real*)(U2.data), linkIndexShift(x, d, arg.E), mu, otherparity);
	d[nu]++;

	// load U(x+mu-nu)(nu)
	d[mu]++; d[nu]--;
	arg.gauge.load((real*)(U3.data), linkIndexShift(x, d, arg.E), nu, arg.parity);
	d[mu]--; d[nu]++;

	// load U(x)_(+mu)
	arg.gauge.load((real*)(U4.data), linkIndexShift(x, d, arg.E), mu, arg.parity);

	d[mu]++; d[nu]--;
	arg.oprod.load((real*)(Oprod1.data), linkIndexShift(x, d, arg.E), tidx, arg.parity);
	d[mu]--; d[nu]++;

	if (isConjugate) Oprod1 -= conj(Oprod1);

        if (nu < mu) add(force, conj(U1)*U2*Oprod1*U3*conj(U4));
	else         sub(force, conj(U1)*U2*Oprod1*U3*conj(U4));

	arg.oprod.load((real*)(Oprod4.data), linkIndexShift(x, d, arg.E), tidx, arg.parity);

	if (isConjugate) Oprod4 -= conj(Oprod4);

        if (nu < mu) add(force, Oprod4*conj(U1)*U2*U3*conj(U4));
	else         sub(force, Oprod4*conj(U1)*U2*U3*conj(U4));
      }

    } else { // else do other force

      int y[4] = {0, 0, 0, 0};
      getCoordsExtended(y, xIndex, arg.X, otherparity, arg.border);

      {
	int d[4] = {0, 0, 0, 0};
	Link U1, U2, U3, U4, Oprod3, Oprod4;

	// load U(x)_(+mu)
	arg.gauge.load((real*)(U1.data), linkIndexShift(y, d, arg.E), mu, otherparity);

	// load U(x+mu)_(+nu)
	d[mu]++;
	arg.gauge.load((real*)(U2.data), linkIndexShift(y, d, arg.E), nu, arg.parity);
	d[mu]--;

	// load U(x+nu)_(+mu)
	d[nu]++;
	arg.gauge.load((real*)(U3.data), linkIndexShift(y, d, arg.E), mu, arg.parity);
	d[nu]--;

	// load U(x)_(+nu)
	arg.gauge.load((real*)(U4.data), linkIndexShift(y, d, arg.E), nu, otherparity);

	// load opposite parity Oprod
	d[nu]++;
	arg.oprod.load((real*)(Oprod3.data), linkIndexShift(y, d, arg.E), tidx, arg.parity);
	d[nu]--;

	if (isConjugate) Oprod3 -= conj(Oprod3);

	if (nu < mu) sub(force, U1*U2*conj(U3)*Oprod3*conj(U4));
	else         add(force, U1*U2*conj(U3)*Oprod3*conj(U4));

	// load Oprod(x+mu)
	d[mu]++;
	arg.oprod.load((real*)(Oprod4.data), linkIndexShift(y, d, arg.E), tidx, arg.parity);
	d[mu]--;

	if (isConjugate) Oprod4 -= conj(Oprod4);

	if (nu < mu) sub(force, U1*Oprod4*U2*conj(U3)*conj(U4));
	else         add(force, U1*Oprod4*U2*conj(U3)*conj(U4));
      }

      // Lower leaf
      // U[nu*](x-nu) U[mu](x-nu) U[nu](x+mu-nu) Oprod(x+mu) U[*mu](x)
      {
	int d[4] = {0, 0, 0, 0};
	Link U1, U2, U3, U4, Oprod1, Oprod2;

	// load U(x-nu)(+nu)
	d[nu]--;
	arg.gauge.load((real*)(U1.data), linkIndexShift(y, d, arg.E), nu, arg.parity);
	d[nu]++;

	// load U(x-nu)(+mu)
	d[nu]--;
	arg.gauge.load((real*)(U2.data), linkIndexShift(y, d, arg.E), mu, arg.parity);
	d[nu]++;

	// load U(x+mu-nu)(nu)
	d[mu]++; d[nu]--;
	arg.gauge.load((real*)(U3.data), linkIndexShift(y, d, arg.E), nu, otherparity);
	d[mu]--; d[nu]++;

	// load U(x)_(+mu)
	arg.gauge.load((real*)(U4.data), linkIndexShift(y, d, arg.E), mu, otherparity);

	// load Oprod(x+mu)
	d[mu]++;
	arg.oprod.load((real*)(Oprod1.data), linkIndexShift(y, d, arg.E), tidx, arg.parity);
	d[mu]--;

	if (isConjugate) Oprod1 -= conj(Oprod1);

	if (nu < mu) add(force, conj(U1)*U2*U3*Oprod1*conj(U4));
	else         sub(force, conj(U1)*U2*U3*Oprod1*conj(U4));

	d[nu]--;
	arg.oprod.load((real*)(Oprod2.data), linkIndexShift(y, d, arg.E), tidx, arg.parity);
	d[nu]++;

	if (isConjugate) Oprod2 -= conj(Oprod2);

	if (nu < mu) add(force, conj(U1)*Oprod2*U2*U3*conj(U4));
	else         sub(force, conj(U1)*Oprod2*U2*U3*conj(U4));
      }

    }

  }


  template<typename real, bool isConjugate, typename Arg>
  __global__ void cloverDerivativeKernel(Arg arg)
  {
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= arg.volumeCB) return;

    // y index determines whether we're updating arg.parity or (1-arg.parity)
    int yIndex = threadIdx.y + blockIdx.y*blockDim.y;
    if (yIndex >= 2) return;

    // mu index is mapped from z thread index
    int mu = threadIdx.z + blockIdx.z*blockDim.z;

    typedef complex<real> Complex;
    typedef Matrix<Complex,3> Link;

    DECLARE_FORCE;

    switch(mu) {
    case 0:
      computeForce<real,isConjugate,Arg,0,1,Link>(force, arg, index, yIndex);
      computeForce<real,isConjugate,Arg,0,2,Link>(force, arg, index, yIndex);
      computeForce<real,isConjugate,Arg,0,3,Link>(force, arg, index, yIndex);
      break;
    case 1:
      computeForce<real,isConjugate,Arg,1,0,Link>(force, arg, index, yIndex);
      computeForce<real,isConjugate,Arg,1,3,Link>(force, arg, index, yIndex);
      computeForce<real,isConjugate,Arg,1,2,Link>(force, arg, index, yIndex);
      break;
    case 2:
      computeForce<real,isConjugate,Arg,2,3,Link>(force, arg, index, yIndex);
      computeForce<real,isConjugate,Arg,2,0,Link>(force, arg, index, yIndex);
      computeForce<real,isConjugate,Arg,2,1,Link>(force, arg, index, yIndex);
      break;
    case 3:
      computeForce<real,isConjugate,Arg,3,2,Link>(force, arg, index, yIndex);
      computeForce<real,isConjugate,Arg,3,1,Link>(force, arg, index, yIndex);
      computeForce<real,isConjugate,Arg,3,0,Link>(force, arg, index, yIndex);
      break;
    }

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

    unsigned int sharedBytesPerThread() const { return 18*sizeof(Float); }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    unsigned int minThreads() const { return arg.volumeCB; }
    bool tuneGridDim() const { return false; }

  public:
    CloverDerivative(const Arg &arg, const GaugeField &meta) : TunableVectorY(2), arg(arg), meta(meta) {
      writeAuxString("conj=%d,threads=%d,prec=%lu,fstride=%d,gstride=%d,ostride=%d",
		     arg.conjugate,arg.volumeCB,sizeof(Float),arg.force.stride,
		     arg.gauge.stride,arg.oprod.stride);
    }
    virtual ~CloverDerivative() {}

    void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (arg.conjugate) {
	cloverDerivativeKernel<Float,true><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      } else {
	cloverDerivativeKernel<Float,false><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      }
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
			double coeff, int parity,
			int conjugate) {
 
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
	  Arg arg(F(force), G(gauge), O(oprod), force.X(), oprod.X(), coeff, parity, conjugate);
	  CloverDerivative<Float, Arg> deriv(arg, gauge);
	  deriv.apply(0);
#if 0
	} else if (gauge.Reconstruct() == QUDA_RECONSTRUCT_12) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
	  typedef CloverDerivArg<Float,F,G,O> Arg;
	  Arg arg(F(force), G(gauge), O(oprod), force.X(), oprod.X(), coeff, parity, conjugate);
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
		      double coeff, QudaParity parity, int conjugate)
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
    cloverDerivative<double>(force, gauge, oprod, coeff, device_parity, conjugate);
#if 0
  } else if (force.Precision() == QUDA_SINGLE_PRECISION){
    cloverDerivative<float>(force, gauge, oprod, coeff, device_parity, conjugate);
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
