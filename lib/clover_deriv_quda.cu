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
  
  template<class Cmplx, typename Force, typename Gauge, typename Oprod>
  struct CloverDerivArg
  {
    int X[4];
    int border[4];
    int mu;
    int nu;
    typename RealTypeId<Cmplx>::Type coeff;
    int parity;
    int volumeCB;

    Force force;
    Gauge gauge;
    Oprod oprod;

    bool conjugate;      

    CloverDerivArg(const Force& force, const Gauge& gauge, const Oprod& oprod, const int *X, 
		   int mu, int nu, double coeff, int parity, bool conjugate) :  
      mu(mu), nu(nu), coeff(coeff), parity(parity), volumeCB(force.volumeCB), 
      force(force), gauge(gauge), oprod(oprod), conjugate(conjugate)
    {
      for(int dir=0; dir<4; ++dir) this->X[dir] = X[dir];
      //for(int dir=0; dir<4; ++dir) border[dir] =  commDimPartitioned(dir) ? 2 : 0;
      for(int dir=0; dir<4; ++dir) border[dir] = 2;
    }
  };

  
  template<typename Cmplx, bool isConjugate, typename Arg>
  __global__ void 
  cloverDerivativeKernel(Arg arg)
  {
    typedef typename RealTypeId<Cmplx>::Type real;

    int index = threadIdx.x + blockIdx.x*blockDim.x;

    if(index >= arg.volumeCB) return;

    int x[4];
    int y[4];
    int otherparity = (1-arg.parity);
    getCoords(x, index, arg.X, arg.parity);
    getCoords(y, index, arg.X, otherparity);
    int X[4]; 
    for(int dir=0; dir<4; ++dir) X[dir] = arg.X[dir];

    for(int dir=0; dir<4; ++dir){
      x[dir] += arg.border[dir];
      y[dir] += arg.border[dir];
      X[dir] += 2*arg.border[dir];
    }

    const int& mu = arg.mu;
    const int& nu = arg.nu;

    Matrix<Cmplx,3> thisForce;
    Matrix<Cmplx,3> otherForce;

    // U[mu](x) U[nu](x+mu) U[*mu](x+nu) U[*nu](x) Oprod(x)
    {
      int d[4] = {0, 0, 0, 0};

      // load U(x)_(+mu)
      Matrix<Cmplx,3> U1;
      arg.gauge.load((real*)(U1.data), linkIndexShift(x, d, X), mu, arg.parity);

      // load U(x+mu)_(+nu)
      Matrix<Cmplx,3> U2;
      d[mu]++;
      arg.gauge.load((real*)(U2.data), linkIndexShift(x, d, X), nu, otherparity);
      d[mu]--;


      // load U(x+nu)_(+mu) 
      Matrix<Cmplx,3> U3;
      d[nu]++;
      arg.gauge.load((real*)(U3.data), linkIndexShift(x, d, X), mu, otherparity);
      d[nu]--;
      
      // load U(x)_(+nu)
      Matrix<Cmplx,3> U4;
      arg.gauge.load((real*)(U4.data), linkIndexShift(x, d, X), nu, arg.parity);

      // load Oprod
      Matrix<Cmplx,3> Oprod1;
      arg.oprod.load((real*)(Oprod1.data), linkIndexShift(x, d, X), 0, arg.parity);

      if(isConjugate) Oprod1 -= conj(Oprod1);
      thisForce = U1*U2*conj(U3)*conj(U4)*Oprod1;

      Matrix<Cmplx,3> Oprod2;
      d[mu]++; d[nu]++;
      arg.oprod.load((real*)(Oprod2.data), linkIndexShift(x, d, X), 0, arg.parity);
      d[mu]--; d[nu]--;

      if(isConjugate) Oprod2 -= conj(Oprod2);

      thisForce += U1*U2*Oprod2*conj(U3)*conj(U4);
    } 
 
    { 
      int d[4] = {0, 0, 0, 0};
      // load U(x)_(+mu)
      Matrix<Cmplx,3> U1;
      arg.gauge.load((real*)(U1.data), linkIndexShift(y, d, X), mu, otherparity);

      // load U(x+mu)_(+nu)
      Matrix<Cmplx,3> U2;
      d[mu]++;
      arg.gauge.load((real*)(U2.data), linkIndexShift(y, d, X), nu, arg.parity);
      d[mu]--;

      // load U(x+nu)_(+mu) 
      Matrix<Cmplx,3> U3;
      d[nu]++;
      arg.gauge.load((real*)(U3.data), linkIndexShift(y, d, X), mu, arg.parity);
      d[nu]--;

      // load U(x)_(+nu)
      Matrix<Cmplx,3> U4;
      arg.gauge.load((real*)(U4.data), linkIndexShift(y, d, X), nu, otherparity);

      // load opposite parity Oprod
      Matrix<Cmplx,3> Oprod3;
      d[nu]++;
      arg.oprod.load((real*)(Oprod3.data), linkIndexShift(y, d, X), 0, arg.parity);
      d[nu]--;

      if(isConjugate) Oprod3 -= conj(Oprod3);
      otherForce = U1*U2*conj(U3)*Oprod3*conj(U4);

      // load Oprod(x+mu)
      Matrix<Cmplx, 3> Oprod4;
      d[mu]++;
      arg.oprod.load((real*)(Oprod4.data), linkIndexShift(y, d, X), 0, arg.parity);
      d[mu]--;

      if(isConjugate) Oprod4 -= conj(Oprod4);

      otherForce += U1*Oprod4*U2*conj(U3)*conj(U4);
    }


    // Lower leaf
    // U[nu*](x-nu) U[mu](x-nu) U[nu](x+mu-nu) Oprod(x+mu) U[*mu](x)
    {
      int d[4] = {0, 0, 0, 0};
      // load U(x-nu)(+nu)
      Matrix<Cmplx,3> U1;
      d[nu]--;
      arg.gauge.load((real*)(U1.data), linkIndexShift(y, d, X), nu, arg.parity);
      d[nu]++;

      // load U(x-nu)(+mu) 
      Matrix<Cmplx, 3> U2;
      d[nu]--;
      arg.gauge.load((real*)(U2.data), linkIndexShift(y, d, X), mu, arg.parity);
      d[nu]++;

      // load U(x+mu-nu)(nu)
      Matrix<Cmplx, 3> U3;
      d[mu]++; d[nu]--;
      arg.gauge.load((real*)(U3.data), linkIndexShift(y, d, X), nu, otherparity);
      d[mu]--; d[nu]++;

      // load U(x)_(+mu)
      Matrix<Cmplx,3> U4;
      arg.gauge.load((real*)(U4.data), linkIndexShift(y, d, X), mu, otherparity);

      // load Oprod(x+mu)
      Matrix<Cmplx, 3> Oprod1;
      d[mu]++;
      arg.oprod.load((real*)(Oprod1.data), linkIndexShift(y, d, X), 0, arg.parity);
      d[mu]--;    

      if(isConjugate) Oprod1 -= conj(Oprod1);

      otherForce -= conj(U1)*U2*U3*Oprod1*conj(U4);

      Matrix<Cmplx,3> Oprod2;
      d[nu]--;
      arg.oprod.load((real*)(Oprod2.data), linkIndexShift(y, d, X), 0, arg.parity);
      d[nu]++;

      if(isConjugate) Oprod2 -= conj(Oprod2);
      otherForce -= conj(U1)*Oprod2*U2*U3*conj(U4);
    }

    {
      int d[4] = {0, 0, 0, 0};
      // load U(x-nu)(+nu)
      Matrix<Cmplx,3> U1;
      d[nu]--;
      arg.gauge.load((real*)(U1.data), linkIndexShift(x, d, X), nu, otherparity);
      d[nu]++;
	
      // load U(x-nu)(+mu) 
      Matrix<Cmplx, 3> U2;
      d[nu]--;
      arg.gauge.load((real*)(U2.data), linkIndexShift(x, d, X), mu, otherparity);
      d[nu]++;

      // load U(x+mu-nu)(nu)
      Matrix<Cmplx, 3> U3;
      d[mu]++; d[nu]--;
      arg.gauge.load((real*)(U3.data), linkIndexShift(x, d, X), nu, arg.parity);
      d[mu]--; d[nu]++;

      // load U(x)_(+mu)
      Matrix<Cmplx,3> U4;
      arg.gauge.load((real*)(U4.data), linkIndexShift(x, d, X), mu, arg.parity);

      Matrix<Cmplx,3> Oprod1;
      d[mu]++; d[nu]--;
      arg.oprod.load((real*)(Oprod1.data), linkIndexShift(x, d, X), 0, arg.parity);
      d[mu]--; d[nu]++;

      if(isConjugate) Oprod1 -= conj(Oprod1);
      thisForce -= conj(U1)*U2*Oprod1*U3*conj(U4);

      Matrix<Cmplx, 3> Oprod4;
      arg.oprod.load((real*)(Oprod4.data), linkIndexShift(x, d, X), 0, arg.parity);

      if(isConjugate) Oprod4 -= conj(Oprod4);
      thisForce -= Oprod4*conj(U1)*U2*U3*conj(U4);
    }
    
    thisForce *= arg.coeff;
    otherForce *= arg.coeff;

    // Write to array
    {
      Matrix<Cmplx, 3> F;
      arg.force.load((real*)(F.data), index, mu, arg.parity);
      F += thisForce;
      arg.force.save((real*)(F.data), index, mu, arg.parity);
    }
      
    {
      Matrix<Cmplx, 3> F;
      arg.force.load((real*)(F.data), index, mu, otherparity);
      F += otherForce;
      arg.force.save((real*)(F.data), index, mu, otherparity);
    }
      
    return;
  } // cloverDerivativeKernel
  
  
  template<typename Complex, typename Arg>
  class CloverDerivative : public Tunable {
    
  private:
    Arg arg;
    const GaugeField &meta;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    unsigned int minThreads() const { return arg.volumeCB; }
    bool tuneGridDim() const { return false; }

  public:
    CloverDerivative(const Arg &arg, const GaugeField &meta)
      : arg(arg), meta(meta) {
      writeAuxString("threads=%d,prec=%lu,fstride=%d,gstride=%d,ostride=%d",
		     arg.volumeCB,sizeof(Complex)/2,arg.force.stride,arg.gauge.stride,arg.oprod.stride);
    }
    virtual ~CloverDerivative() {}

    void apply(const cudaStream_t &stream){
#if __COMPUTE_CAPABILITY__ >= 200
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#else // tuning not supported on Tesla architecture
      TuneParam tp = tuneLaunch(*this, QUDA_TUNE_NO, getVerbosity());
#endif
      if(arg.conjugate){
	cloverDerivativeKernel<Complex,true><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      }else{
	cloverDerivativeKernel<Complex,false><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      }
    } // apply

    // The force field is updated so we must preserve its initial state
#if __COMPUTE_CAPABILITY__ >= 200
    void preTune() { arg.force.save(); } 
    void postTune(){ arg.force.load(); } 
#endif

    long long flops() const { return 0; }
    long long bytes() const { return (16*arg.gauge.Bytes() + 8*arg.oprod.Bytes() + 4*arg.force.Bytes()) * arg.volumeCB; }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };

  
  template<typename Float>
  void cloverDerivative(cudaGaugeField &force,
			cudaGaugeField &gauge,
			cudaGaugeField &oprod,
			int mu, int nu, double coeff, int parity,
			int conjugate) {
 
    if (oprod.Reconstruct() != QUDA_RECONSTRUCT_NO) 
      errorQuda("Force field does not support reconstruction");
    
    if (force.Order() != oprod.Order()) 
      errorQuda("Force and Oprod orders must match");
    
    if (force.Reconstruct() != QUDA_RECONSTRUCT_NO) 
      errorQuda("Force field does not support reconstruction");
  
    typedef typename ComplexTypeId<Float>::Type Complex;

    if (force.Order() == QUDA_FLOAT2_GAUGE_ORDER){
      typedef FloatNOrder<Float, 18, 2, 18> F;
      typedef FloatNOrder<Float, 18, 2, 18> O;

      if (gauge.isNative()) {
	if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	  typedef CloverDerivArg<Complex,F,G,O> Arg;
	  Arg arg(F(force), G(gauge), O(oprod), force.X(), mu, nu, coeff, parity, conjugate);
	  CloverDerivative<Complex, Arg> deriv(arg, gauge);
	  deriv.apply(0);
	} else if(gauge.Reconstruct() == QUDA_RECONSTRUCT_12) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
	  typedef CloverDerivArg<Complex,F,G,O> Arg;
	  Arg arg(F(force), G(gauge), O(oprod), force.X(), mu, nu, coeff, parity, conjugate);
	  CloverDerivative<Complex, Arg> deriv(arg, gauge);
	  deriv.apply(0);
	}else{
	  errorQuda("Reconstruction type %d not supported",gauge.Reconstruct());
	}
      } else {
	errorQuda("Gauge order %d not supported", gauge.Order());
      }
    } else {
      errorQuda("Force order %d not supported", force.Order());
    } // force / oprod order
  }
#endif // GPU_CLOVER

void cloverDerivative(cudaGaugeField &force,   
		      cudaGaugeField &gauge,
		      cudaGaugeField &oprod,
		      int mu, int nu, double coeff, QudaParity parity, int conjugate)
{
#ifdef GPU_CLOVER_DIRAC
  assert(oprod.Geometry() == QUDA_SCALAR_GEOMETRY);
  assert(force.Geometry() == QUDA_VECTOR_GEOMETRY);

  int device_parity = (parity == QUDA_EVEN_PARITY) ? 0 : 1;

  if(force.Precision() == QUDA_DOUBLE_PRECISION){
    cloverDerivative<double>(force, gauge, oprod, mu, nu, coeff, device_parity, conjugate);   
  } else if (force.Precision() == QUDA_SINGLE_PRECISION){
    cloverDerivative<float>(force, gauge, oprod, mu, nu, coeff, device_parity, conjugate);
  } else {
    errorQuda("Precision %d not supported", force.Precision());
  }

  return;
#else
  errorQuda("Clover has not been built");
#endif
}              


} // namespace quda
