#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <quda_matrix.h>
#include <cassert>

namespace quda {

  template<class Cmplx>
    struct CloverDerivArg
    {
      int X[4];
      int border[4];
      int mu;
      int nu;
      typename RealTypeId<Cmplx>::Type coeff;
      int parity;
      int volumeCB;

      Cmplx* gauge;
      Cmplx* force;
      Cmplx* oprod;

      int forceStride;
      int gaugeStride;
      int oprodStride;

      int forceOffset;
      int gaugeOffset;
      int oprodOffset;
     
      bool conjugate;      

      CloverDerivArg(cudaGaugeField& force, cudaGaugeField& gauge, cudaGaugeField& oprod, int mu, int nu, double coeff, int parity, bool conjugate) :  
        mu(mu), nu(nu), coeff(coeff), parity(parity), volumeCB(force.VolumeCB()), 
        force(reinterpret_cast<Cmplx*>(force.Gauge_p())),  gauge(reinterpret_cast<Cmplx*>(gauge.Gauge_p())), oprod(reinterpret_cast<Cmplx*>(oprod.Gauge_p())),
        forceStride(force.Stride()), gaugeStride(gauge.Stride()), oprodStride(oprod.Stride()),
        forceOffset(force.Bytes()/(2*sizeof(Cmplx))), gaugeOffset(gauge.Bytes()/(2*sizeof(Cmplx))), oprodOffset(oprod.Bytes()/(2*sizeof(Cmplx)))
      {
        for(int dir=0; dir<4; ++dir) X[dir] = force.X()[dir];
        //for(int dir=0; dir<4; ++dir) border[dir] =  commDimPartitioned(dir) ? 2 : 0;
        for(int dir=0; dir<4; ++dir) border[dir] = 2;
      }
    };

  __device__ void getCoords(int x[4], int cb_index, const int X[4], int parity)
  {
    x[3] = cb_index/(X[2]*X[1]*X[0]/2);
    x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    x[1] = (cb_index/(X[0]/2)) % X[1];
    x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    return;
  }

  __device__ int linkIndex(const int x[4], const int dx[4], const int X[4])
  {
    int y[4];
    for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
    return (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0])/2;
  }




  template<typename Cmplx, bool isConjugate>
    __global__ void 
    cloverDerivativeKernel(const CloverDerivArg<Cmplx> arg)
    {
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

      Cmplx* thisGauge = arg.gauge + arg.parity*arg.gaugeOffset;
      Cmplx* otherGauge = arg.gauge + (otherparity)*arg.gaugeOffset;

      Cmplx* thisOprod = arg.oprod + arg.parity*arg.oprodOffset;

      const int& mu = arg.mu;
      const int& nu = arg.nu;

      Matrix<Cmplx,3> thisForce;
      Matrix<Cmplx,3> otherForce;

      // U[mu](x) U[nu](x+mu) U[*mu](x+nu) U[*nu](x) Oprod(x)
      {
        int d[4] = {0, 0, 0, 0};

        // load U(x)_(+mu)
        Matrix<Cmplx,3> U1;
        loadLinkVariableFromArray(thisGauge, mu, linkIndex(x, d, X), 
            arg.gaugeStride, &U1);


        // load U(x+mu)_(+nu)
        Matrix<Cmplx,3> U2;
        d[mu]++;
        loadLinkVariableFromArray(otherGauge, nu, linkIndex(x, d, X), 
            arg.gaugeStride, &U2);
        d[mu]--;


        // load U(x+nu)_(+mu) 
        Matrix<Cmplx,3> U3;
        d[nu]++;
        loadLinkVariableFromArray(otherGauge, mu, linkIndex(x, d, X),
            arg.gaugeStride, &U3);
        d[nu]--;
      
        // load U(x)_(+nu)
        Matrix<Cmplx,3> U4;
        loadLinkVariableFromArray(thisGauge, nu, linkIndex(x, d, X),
            arg.gaugeStride, &U4);

        // load Oprod
        Matrix<Cmplx,3> Oprod1;
        loadMatrixFromArray(thisOprod, linkIndex(x, d, X), arg.oprodStride, &Oprod1);

        if(isConjugate) Oprod1 -= conj(Oprod1);
        thisForce = U1*U2*conj(U3)*conj(U4)*Oprod1;

        Matrix<Cmplx,3> Oprod2;
        d[mu]++; d[nu]++;
        loadMatrixFromArray(thisOprod, linkIndex(x, d, X), arg.oprodStride, &Oprod2);
        d[mu]--; d[nu]--;

        if(isConjugate) Oprod2 -= conj(Oprod2);

        thisForce += U1*U2*Oprod2*conj(U3)*conj(U4);

      } 
 
      { 
        int d[4] = {0, 0, 0, 0};
        // load U(x)_(+mu)
        Matrix<Cmplx,3> U1;
        loadLinkVariableFromArray(otherGauge, mu, linkIndex(y, d, X),
            arg.gaugeStride, &U1);

        // load U(x+mu)_(+nu)
        Matrix<Cmplx,3> U2;
        d[mu]++;
        loadLinkVariableFromArray(thisGauge, nu, linkIndex(y, d, X),
            arg.gaugeStride, &U2);
        d[mu]--;

        // load U(x+nu)_(+mu) 
        Matrix<Cmplx,3> U3;
        d[nu]++;
        loadLinkVariableFromArray(thisGauge, mu, linkIndex(y, d, X),
            arg.gaugeStride, &U3);
        d[nu]--;

        // load U(x)_(+nu)
        Matrix<Cmplx,3> U4;
        loadLinkVariableFromArray(otherGauge, nu, linkIndex(y, d, X),
            arg.gaugeStride, &U4);

        // load opposite parity Oprod
        Matrix<Cmplx,3> Oprod3;
        d[nu]++;
        loadMatrixFromArray(thisOprod, linkIndex(y, d, X), arg.oprodStride, &Oprod3);
        d[nu]--;

        if(isConjugate) Oprod3 -= conj(Oprod3);
        otherForce = U1*U2*conj(U3)*Oprod3*conj(U4);

        // load Oprod(x+mu)
        Matrix<Cmplx, 3> Oprod4;
        d[mu]++;
        loadMatrixFromArray(thisOprod, linkIndex(y, d, X), arg.oprodStride, &Oprod4);
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
        loadLinkVariableFromArray(thisGauge, nu, linkIndex(y, d, X),
            arg.gaugeStride, &U1);
        d[nu]++;

        // load U(x-nu)(+mu) 
        Matrix<Cmplx, 3> U2;
        d[nu]--;
        loadLinkVariableFromArray(thisGauge, mu, linkIndex(y, d, X),
            arg.gaugeStride, &U2);
        d[nu]++;

        // load U(x+mu-nu)(nu)
        Matrix<Cmplx, 3> U3;
        d[mu]++; d[nu]--;
        loadLinkVariableFromArray(otherGauge, nu, linkIndex(y, d, X),
            arg.gaugeStride, &U3);
        d[mu]--; d[nu]++;

        // load U(x)_(+mu)
        Matrix<Cmplx,3> U4;
        loadLinkVariableFromArray(otherGauge, mu, linkIndex(y, d, X),
            arg.gaugeStride, &U4);

        // load Oprod(x+mu)
        Matrix<Cmplx, 3> Oprod1;
        d[mu]++;
        loadMatrixFromArray(thisOprod, linkIndex(y, d, X), arg.oprodStride, &Oprod1);
        d[mu]--;    

        if(isConjugate) Oprod1 -= conj(Oprod1);

        otherForce -= conj(U1)*U2*U3*Oprod1*conj(U4);

        Matrix<Cmplx,3> Oprod2;
        d[nu]--;
        loadMatrixFromArray(thisOprod, linkIndex(y, d, X), arg.oprodStride, &Oprod2);
        d[nu]++;

        if(isConjugate) Oprod2 -= conj(Oprod2);
        otherForce -= conj(U1)*Oprod2*U2*U3*conj(U4);
      }

      {
        int d[4] = {0, 0, 0, 0};
        // load U(x-nu)(+nu)
        Matrix<Cmplx,3> U1;
        d[nu]--;
        loadLinkVariableFromArray(otherGauge, nu, linkIndex(x, d, X), 
            arg.gaugeStride, &U1);
        d[nu]++;

        // load U(x-nu)(+mu) 
        Matrix<Cmplx, 3> U2;
        d[nu]--;
        loadLinkVariableFromArray(otherGauge, mu, linkIndex(x, d, X),
            arg.gaugeStride, &U2);
        d[nu]++;

        // load U(x+mu-nu)(nu)
        Matrix<Cmplx, 3> U3;
        d[mu]++; d[nu]--;
        loadLinkVariableFromArray(thisGauge, nu, linkIndex(x, d, X),
            arg.gaugeStride, &U3);
        d[mu]--; d[nu]++;

        // load U(x)_(+mu)
        Matrix<Cmplx,3> U4;
        loadLinkVariableFromArray(thisGauge, mu, linkIndex(x, d, X),
            arg.gaugeStride, &U4);


        Matrix<Cmplx,3> Oprod1;
        d[mu]++; d[nu]--;
        loadMatrixFromArray(thisOprod, linkIndex(x, d, X), arg.oprodStride, &Oprod1);
        d[mu]--; d[nu]++;

        if(isConjugate) Oprod1 -= conj(Oprod1);
        thisForce -= conj(U1)*U2*Oprod1*U3*conj(U4);

        Matrix<Cmplx, 3> Oprod4;
        loadMatrixFromArray(thisOprod, linkIndex(x, d, X), arg.oprodStride, &Oprod4);

        if(isConjugate) Oprod4 -= conj(Oprod4);
        thisForce -= Oprod4*conj(U1)*U2*U3*conj(U4);
      }
    
      thisForce *= arg.coeff;
      otherForce *= arg.coeff;


      // Write to array
      {
        appendMatrixToArray(thisForce, index, arg.forceStride, arg.force + arg.parity*arg.forceOffset);
        appendMatrixToArray(otherForce, index, arg.forceStride, arg.force + otherparity*arg.forceOffset); 
      }
      return;
    } // cloverDerivativeKernel


  template<typename Complex>
    class CloverDerivative : public Tunable {

      private:
        CloverDerivArg<Complex> arg;

        unsigned int sharedBytesPerThread() const { return 0; }
        unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

        unsigned int minThreads() const { return arg.volumeCB; }
        bool tuneGridDim() const { return false; }

      public:
        CloverDerivative(const CloverDerivArg<Complex> &arg)
          : arg(arg) {}
        virtual ~CloverDerivative() {}

        void apply(const cudaStream_t &stream){
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          if(arg.conjugate){
            cloverDerivativeKernel<Complex,true><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);   
          }else{
            cloverDerivativeKernel<Complex,false><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);   
          }      
        } // apply

        void preTune(){}
        void postTune(){}

        long long flops() const {
          return 0;
        }    

        long long bytes() const { return 0; }

        TuneKey tuneKey() const {
          std::stringstream vol, aux;
          vol << arg.X[0] << "x";
          vol << arg.X[1] << "x";
          vol << arg.X[2] << "x";
          vol << arg.X[3] << "x";
          aux << "threads=" << arg.volumeCB << ",prec=" << sizeof(Complex)/2;
          aux << "stride=" << arg.forceOffset;
          return TuneKey(vol.str(), typeid(*this).name(), aux.str());
        }
    };


  template<typename Float>
    void cloverDerivative(cudaGaugeField &out,
        cudaGaugeField& gauge,
        cudaGaugeField& oprod,
        int mu, int nu, double coeff, int parity,
        int conjugate)
    {
      typedef typename ComplexTypeId<Float>::Type Complex;
      CloverDerivArg<Complex> arg(out, gauge, oprod, mu, nu, coeff, parity, conjugate);
//      CloverDerivative<Complex> cloverDerivative(arg);
//      cloverDerivative.apply(0);
      dim3 blockDim(128, 1, 1);
      dim3 gridDim((arg.volumeCB + blockDim.x-1)/blockDim.x, 1, 1);
      if(conjugate){
        cloverDerivativeKernel<Complex,true><<<gridDim,blockDim,0>>>(arg);
      }else{
        cloverDerivativeKernel<Complex,false><<<gridDim,blockDim,0>>>(arg);
      }
    }    


  void cloverDerivative(cudaGaugeField &out,   
      cudaGaugeField& gauge,
      cudaGaugeField& oprod,
      int mu, int nu, double coeff, QudaParity parity, int conjugate)
  {
    assert(oprod.Geometry() == QUDA_SCALAR_GEOMETRY);
    assert(out.Geometry() == QUDA_SCALAR_GEOMETRY);

    int device_parity = (parity == QUDA_EVEN_PARITY) ? 0 : 1;

    if(out.Precision() == QUDA_DOUBLE_PRECISION){
      cloverDerivative<double>(out, gauge, oprod, mu, nu, coeff, device_parity, conjugate);   
    } else if (out.Precision() == QUDA_SINGLE_PRECISION){
      cloverDerivative<float>(out, gauge, oprod, mu, nu, coeff, device_parity, conjugate);
    } else {
      errorQuda("Precision %d not supported", out.Precision());
    }
    return;
  }              


} // namespace quda
