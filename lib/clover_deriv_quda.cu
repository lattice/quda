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
    int mu;
    int nu;
    int parity;
    int threads;
    int volumeCB;

    Cmplx* gauge;
    Cmplx* force;
    Cmplx* oprod;

    int gaugeStride;
    int oprodStride;
    int forceStride;
    int gaugeLengthCB;
    int oprodLengthCB;
    int forceLengthCB;
  };

  __device__ void getCoords(int x[4], int cb_index, const int X[4], int parity)
  {
    x[3] = cb_index/(X[2]*X[1]*X[0]/2);
    x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    x[1] = (cb_index/(X[0]/2)) % X[1];
    x[0] = 2*(cb_index/(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    return;
  }

  __device__ int linkIndex(const int x[4], const int dx[4], const int X[4])
  {
    int y[4] = {x[0]+dx[0], x[1]+dx[1], x[2]+dx[2], x[3]+dx[3]};
    return (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0])/2;
  }




  template<typename Cmplx>
    __global__ void 
    cloverDerivativeKernel(const CloverDerivArg<Cmplx> arg)
    {
      int index = threadIdx.x + blockIdx.x*blockDim.x;

      if(index > arg.threads) return;


      int x[4];
      int y[4];
      int otherparity = (arg.parity^1);
      getCoords(x, index, arg.X, arg.parity);
      getCoords(y, index, arg.X, otherparity);

      const Cmplx* thisGauge = arg.gauge + arg.parity*arg.gaugeLengthCB;
      const Cmplx* otherGauge = arg.gauge + (otherparity)*arg.gaugeLengthCB;

      const Cmplx* thisOprod = arg.oprod + arg.parity*arg.oprodLengthCB;

      const int& mu = arg.mu;
      const int& nu = arg.nu;

      Matrix<Cmplx,3> thisForce;
      Matrix<Cmplx,3> otherForce;

      // U[mu](x) U[nu](x+mu) U[*mu](x+nu) U[*nu](x) Oprod(x)
      {
        int d[4] = {0, 0, 0, 0};

        // load U(x)_(+mu)
        Matrix<Cmplx,3> U1;
        loadLinkVariableFromArray(thisGauge, mu, linkIndex(x, d, arg.X), 
            arg.gaugeStride, &U1);

        // load U(x+mu)_(+nu)
        Matrix<Cmplx,3> U2;
        d[mu]++;
        loadLinkVariableFromArray(otherGauge, nu, linkIndex(x, d, arg.X), 
            arg.gaugeStride, &U2);
        d[mu]--;

        // load U(x+nu)_(+mu) 
        Matrix<Cmplx,3> U3;
        d[nu]++;
        loadLinkVariableFromArray(otherGauge, mu, linkIndex(x, d, arg.X),
            arg.gaugeStride, &U3);
        d[nu]--;

        // load U(x)_(+nu)
        Matrix<Cmplx,3> U4;
        loadLinkVariableFromArray(thisGauge, nu, linkIndex(x, d, arg.X),
            arg.gaugeStride, &U4);

        // load Oprod
        Matrix<Cmplx,3> Oprod1;
        loadMatrixFromArray(thisOprod, linkIndex(x, d, arg.X), arg.oprodStride, &Oprod1);

        thisForce = U1*U2*conj(U3)*Oprod1*conj(U4);

        Matrix<Cmplx,3> Oprod2;
        d[mu]++; d[nu]++;
        loadMatrixFromArray(thisOprod, linkIndex(x, d, arg.X), arg.oprodStride, &Oprod2);
        d[mu]--; d[nu]--;

        thisForce += U1*U2*Oprod2*conj(U3)*conj(U4);
      }  


      { 
        int d[4] = {0, 0, 0, 0};
        // load U(x)_(+mu)
        Matrix<Cmplx,3> U1;
        loadLinkVariableFromArray(otherGauge, mu, linkIndex(y, d, arg.X),
            arg.gaugeStride, &U1);

        // load U(x+mu)_(+nu)
        Matrix<Cmplx,3> U2;
        d[mu]++;
        loadLinkVariableFromArray(thisGauge, nu, linkIndex(y, d, arg.X),
            arg.gaugeStride, &U2);
        d[mu]--;

        // load U(x+nu)_(+mu) 
        Matrix<Cmplx,3> U3;
        d[nu]++;
        loadLinkVariableFromArray(thisGauge, mu, linkIndex(y, d, arg.X),
            arg.gaugeStride, &U3);
        d[nu]--;

        // load U(x)_(+nu)
        Matrix<Cmplx,3> U4;
        loadLinkVariableFromArray(otherGauge, nu, linkIndex(y, d, arg.X),
            arg.gaugeStride, &U4);

        // load opposite parity Oprod
        Matrix<Cmplx,3> Oprod3;
        d[nu]++;
        loadMatrixFromArray(thisOprod, linkIndex(y, d, arg.X), arg.oprodStride, &Oprod3);
        d[nu]--;

        otherForce = U1*U2*conj(U3)*Oprod3*conj(U4);

        // load Oprod(x+mu)
        Matrix<Cmplx, 3> Oprod4;
        d[mu]++;
        loadMatrixFromArray(thisOprod, linkIndex(y, d, arg.X), arg.oprodStride, &Oprod4);
        d[nu]++;
        otherForce += U1*Oprod4*U2*conj(U3)*conj(U4);
      }


      // Lower leaf
      // U[nu*](x-nu) U[mu](x-nu) U[nu](x+mu-nu) Oprod(x+mu) U[*mu](x)
      {
        int d[4] = {0, 0, 0, 0};
        // load U(x-nu)(+nu)
        Matrix<Cmplx,3> U1;
        d[nu]--;
        loadLinkVariableFromArray(thisGauge, nu, linkIndex(y, d, arg.X),
            arg.gaugeStride, &U1);
        d[nu]++;

        // load U(x-nu)(+mu) 
        Matrix<Cmplx, 3> U2;
        d[nu]--;
        loadLinkVariableFromArray(thisGauge, mu, linkIndex(y, d, arg.X),
            arg.gaugeStride, &U2);
        d[nu]++;

        // load U(x+mu-nu)(nu)
        Matrix<Cmplx, 3> U3;
        d[mu]++; d[nu]--;
        loadLinkVariableFromArray(otherGauge, nu, linkIndex(y, d, arg.X),
            arg.gaugeStride, &U3);
        d[mu]--; d[nu]++;

        // load U(x)_(+mu)
        Matrix<Cmplx,3> U4;
        loadLinkVariableFromArray(otherGauge, mu, linkIndex(x, d, arg.X),
            arg.gaugeStride, &U4);

        // load Oprod(x+mu)
        Matrix<Cmplx, 3> Oprod1;
        d[mu]++;
        loadMatrixFromArray(thisOprod, linkIndex(x, d, arg.X), arg.oprodStride, &Oprod1);
        d[mu]--;    


        otherForce -= conj(U1)*U2*U3*Oprod1*conj(U4);

        Matrix<Cmplx,3> Oprod2;
        d[nu]--;
        loadMatrixFromArray(thisOprod, linkIndex(x, d, arg.X), arg.oprodStride, &Oprod2);
        d[nu]++;

        otherForce -= conj(U1)*Oprod2*U2*U3*conj(U4);
      }

      {
        int d[4] = {0, 0, 0, 0};
        // load U(x-nu)(+nu)
        Matrix<Cmplx,3> U1;
        d[nu]--;
        loadLinkVariableFromArray(otherGauge, nu, linkIndex(y, d, arg.X), 
            arg.gaugeStride, &U1);
        d[nu]++;

        // load U(x-nu)(+mu) 
        Matrix<Cmplx, 3> U2;
        d[nu]--;
        loadLinkVariableFromArray(otherGauge, mu, linkIndex(y, d, arg.X),
            arg.gaugeStride, &U2);
        d[nu]++;

        // load U(x+mu-nu)(nu)
        Matrix<Cmplx, 3> U3;
        d[mu]++; d[nu]--;
        loadLinkVariableFromArray(thisGauge, nu, linkIndex(y, d, arg.X),
            arg.gaugeStride, &U3);
        d[mu]--; d[nu]++;

        // load U(x)_(+mu)
        Matrix<Cmplx,3> U4;
        loadLinkVariableFromArray(thisGauge, mu, linkIndex(x, d, arg.X),
            arg.gaugeStride, &U4);


        Matrix<Cmplx,3> Oprod1;
        d[mu]++; d[nu]--;
        loadMatrixFromArray(thisOprod, linkIndex(x, d, arg.X), arg.oprodStride, &Oprod1);
        d[nu]--; d[mu]++;

        thisForce -= conj(U1)*U2*Oprod1*U3*conj(U4);

        Matrix<Cmplx, 3> Oprod4;
        loadMatrixFromArray(thisOprod, linkIndex(x, d, arg.X), arg.oprodStride, &Oprod4);

        thisForce -= Oprod4*conj(U1)*U2*U3*conj(U4);
      }

      // Write to array
      {
        writeMatrixToArray(thisForce, index, arg.forceStride, arg.force + arg.parity*arg.forceLengthCB);
        writeMatrixToArray(otherForce, index, arg.forceStride, arg.force + otherparity*arg.forceLengthCB); 
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
          cloverDerivativeKernel<Complex><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);         
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
          aux << "threads=" << 1 << ",prec=" << sizeof(Complex)/2;
          aux << "stride=" << 1;
          return TuneKey(vol.str(), typeid(*this).name(), aux.str());
        }
    };


   template<typename Float>
    void cloverDerivative(cudaGaugeField &out,
                          const cudaGaugeField& gauge,
                          const cudaGaugeField& oprod)
    {
      typedef typename ComplexTypeId<Float>::Type Complex;
      CloverDerivArg<Complex> arg;
      CloverDerivative<Complex> cloverDerivative(arg);
    }    


    void cloverDerivative(cudaGaugeField &out,   
                          const cudaGaugeField& gauge,
                          const cudaGaugeField& oprod)
    {
      assert(oprod.Geometry() == QUDA_SCALAR_GEOMETRY);
      assert(out.Geometry() == QUDA_SCALAR_GEOMETRY);
      
      if(out.Precision() == QUDA_DOUBLE_PRECISION){
        cloverDerivative<double>(out, gauge, oprod);   
      } else if (out.Precision() == QUDA_SINGLE_PRECISION){
        cloverDerivative<float>(out, gauge, oprod);
      } else {
        errorQuda("Precision %d not supported", out.Precision());
      }
      return;
    }              
                          

} // namespace quda
