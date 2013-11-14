#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <clover_field.h>
#include <gauge_field.h>

namespace CloverOrder {
  using namespace quda;
#include <clover_field_order.h>
} // CloverOrder


namespace quda {

  template<typename Float, typename Clover>
    struct CloverArg {
      int threads; // number of active threads required
      int X[4]; // grid dimensions
      double cloverCoeff;

      int gaugeStride; // stride used on gauge field
      int gaugeOffset; // parity offset 

      int FmunuStride; // stride used on Fmunu field
      int FmunuOffset; // parity offset 


      const typename ComplexTypeId<Float>::Type* gauge;
      typename ComplexTypeId<Float>::Type* Fmunu;
      Clover clover;

      CloverArg(Clover &clover, const GaugeField& gauge, GaugeField& Fmunu, double cloverCoeff)
        : threads(Fmunu.Volume()), 
        cloverCoeff(cloverCoeff),
        gaugeStride(gauge.Stride()), gaugeOffset(gauge.Bytes()/(4*sizeof(Float))),
        FmunuStride(Fmunu.Stride()), FmunuOffset(Fmunu.Bytes()/(4*sizeof(Float))),
        gauge(reinterpret_cast<const typename ComplexTypeId<Float>::Type*>(gauge.Gauge_p())),  
        Fmunu(reinterpret_cast<typename ComplexTypeId<Float>::Type*>(Fmunu.Gauge_p())),
        clover(clover) { 
          for(int dir=0; dir<4; ++dir) X[dir] = Fmunu.X()[dir];
        }

    };

  __device__ __host__ inline int linkIndex(int x[], int dx[], const int X[4]) {
    int y[4];
    for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
    int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
    return idx;
  }


  template <typename Float, typename Clover>
    __host__ __device__ void computeFmunuCore(CloverArg<Float,Clover> arg, int idx) {

      // compute spacetime dimensions and parity
      int aux1 = idx / (arg.X[0]/2);
      int x[4];
      x[0] = idx - aux1 * (arg.X[0]/2); // this is chbd x
      int aux2 = aux1 / arg.X[1];
      x[1] = aux1 - aux2 * arg.X[1];
      int aux3 = aux2 / arg.X[2];
      x[2] = aux2 - aux3 * arg.X[2];
      int parity = aux3 / arg.X[3];
      x[3] = aux3 - parity * arg.X[3];
      x[0] = 2*x[0] + parity; // now this is the full index

      int X[4]; 
      for(int dir=0; dir<4; ++dir) X[dir] = arg.X[dir];

      typedef typename ComplexTypeId<Float>::Type Cmplx;

      const int otherParity = (1-parity);
      const Cmplx* thisGauge = arg.gauge + parity*arg.gaugeOffset;
      const Cmplx* otherGauge = arg.gauge + (otherParity)*arg.gaugeOffset;


      for (int mu=0; mu<4; mu++) {
        for (int nu=0; nu<mu; nu++) {
          Matrix<Cmplx,3> F;
          setZero(&F);

          { // positive mu, nu

            // load U(x)_(+mu)
            Matrix<Cmplx,3> U1;
            int dx[4] = {0, 0, 0, 0};
            loadLinkVariableFromArray(thisGauge, mu, linkIndex(x, dx, X), 
                arg.gaugeStride, &U1);

            // load U(x+mu)_(+nu)
            Matrix<Cmplx,3> U2;
            dx[mu]++;
            loadLinkVariableFromArray(otherGauge, nu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U2);
            dx[mu]--;

            Matrix<Cmplx,3> Ftmp = U1 * U2;

            // load U(x+nu)_(+mu)
            Matrix<Cmplx,3> U3;
            dx[nu]++;
            loadLinkVariableFromArray(otherGauge, mu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U3);
            dx[nu]--;

            Ftmp = Ftmp * conj(U3) ;

            // load U(x)_(+nu)
            Matrix<Cmplx,3> U4;
            loadLinkVariableFromArray(thisGauge, nu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U4);

            // complete the plaquette
            Ftmp = Ftmp * conj(U4);

            // sum this contribution to Fmunu
            F += Ftmp - conj(Ftmp);
          }

          { // positive mu, negative nu

            // load U(x)_(+mu)
            Matrix<Cmplx,3> U1;
            int dx[4] = {0, 0, 0, 0};
            loadLinkVariableFromArray(thisGauge, mu, linkIndex(x, dx, X), 
                arg.gaugeStride, &U1);

            // load U(x+mu)_(-nu) = U(x+mu-nu)_(+nu)
            Matrix<Cmplx,3> U2;
            dx[mu]++;
            dx[nu]--;
            loadLinkVariableFromArray(thisGauge, nu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U2);
            dx[nu]++;
            dx[mu]--;

            Matrix<Cmplx,3> Ftmp =  U1 * conj(U2);

            // load U(x-nu)_mu
            Matrix<Cmplx,3> U3;
            dx[nu]--;
            loadLinkVariableFromArray(otherGauge, mu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U3);
            dx[nu]++;

            Ftmp =  Ftmp * conj(U3);

            // load U(x)_(-nu) = U(x-nu)_(+nu)
            Matrix<Cmplx,3> U4;
            dx[nu]--;
            loadLinkVariableFromArray(otherGauge, nu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U4);
            dx[nu]++;

            // complete the plaquette
            Ftmp = Ftmp * U4;

            // sum this contribution to Fmunu
            F += Ftmp - conj(Ftmp);
          }


          { // negative mu, positive nu

            // load U(x)_(-mu)
            Matrix<Cmplx,3> U1;
            int dx[4] = {0, 0, 0, 0};
            dx[mu]--;
            loadLinkVariableFromArray(otherGauge, mu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U1);
            dx[mu]++;

            // load U(x-mu)_(+nu)
            Matrix<Cmplx,3> U2;
            dx[mu]--;
            loadLinkVariableFromArray(otherGauge, nu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U2);
            dx[mu]++;

            Matrix<Cmplx,3> Ftmp = conj(U1) * U2;

            // load U(x+nu-mu)_(+mu)
            Matrix<Cmplx,3> U3;
            dx[nu]++;
            dx[mu]--;
            loadLinkVariableFromArray(thisGauge, mu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U3);
            dx[mu]++;
            dx[nu]--;

            Ftmp = Ftmp * U3;

            // load U(x)_(+nu)
            Matrix<Cmplx,3> U4;
            loadLinkVariableFromArray(thisGauge, nu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U4);

            // complete the plaquette
            Ftmp = Ftmp * conj(U4);

            // sum this contribution to Fmunu
            F += Ftmp - conj(Ftmp);
          }

          { // negative mu, negative nu

            // load U(x)_(-mu)
            Matrix<Cmplx,3> U1;
            int dx[4] = {0, 0, 0, 0};
            dx[mu]--;
            loadLinkVariableFromArray(otherGauge, mu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U1);
            dx[mu]++;

            // load U(x-mu)_(-nu) = U(x-mu-nu)_(+nu)
            Matrix<Cmplx,3> U2;
            dx[mu]--;
            dx[nu]--;
            loadLinkVariableFromArray(thisGauge, nu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U2);
            dx[nu]++;
            dx[mu]++;

            Matrix<Cmplx,3> Ftmp = conj(U1) * conj(U2);

            // load U(x-nu)_mu
            Matrix<Cmplx,3> U3;
            dx[mu]--;
            dx[nu]--;
            loadLinkVariableFromArray(thisGauge, mu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U3);
            dx[nu]++;
            dx[mu]++;

            Ftmp = Ftmp * U3;

            // load U(x)_(-nu) = U(x-nu)_(+nu)
            Matrix<Cmplx,3> U4;
            dx[nu]--;
            loadLinkVariableFromArray(otherGauge, nu, linkIndex(x,dx,X), 
                arg.gaugeStride, &U4);
            dx[nu]++;

            // complete the plaquette
            Ftmp = Ftmp * U4;

            // sum this contribution to Fmunu
            F += Ftmp - conj(Ftmp);
          }

          Cmplx* thisFmunu = arg.Fmunu + parity*arg.FmunuOffset;
          int munu_idx = (mu*(mu-1))/2 + nu; // lower-triangular indexing
          //writeLinkVariableToArray(F, munu_idx, X/2, arg.FmunuStride, arg.Fmunu+parity*arg.FmunuOffset);
          writeLinkVariableToArray(F, munu_idx, idx/2, arg.FmunuStride, thisFmunu);
        } // nu < mu
      } // mu
      // F[1,0], F[2,0], F[2,1], F[3,0], F[3,1], F[3,2]
      return;
    }


  template<typename Float, typename Clover>
    __global__ void computeFmunuKernel(CloverArg<Float,Clover> arg){
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      if(idx >= arg.threads) return;
      computeFmunuCore(arg,idx);
    }

  template<typename Float, typename Clover>
    void computeFmunuCPU(CloverArg<Float,Clover>& arg){
      errorQuda("computeFmunuCPU not yet supported\n");
      for(int idx=0; idx<arg.threads; idx++){
        computeFmunuCore(arg,idx);
      }
    }



  template<typename Float, typename Clover>
    class FmunuCompute : Tunable {
      CloverArg<Float,Clover> arg;
      const QudaFieldLocation location;

      private: 
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; } // Don't tune shared memory
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      FmunuCompute(CloverArg<Float,Clover> &arg, QudaFieldLocation location)
        : arg(arg), location(location) {}
      virtual ~FmunuCompute() {}

      void apply(const cudaStream_t &stream){
        // No tuning for the time being
        if(location == QUDA_CUDA_FIELD_LOCATION){
          dim3 blockDim(128, 1, 1);
          dim3 gridDim((arg.threads + blockDim.x - 1) / blockDim.x, 1, 1);
          computeFmunuKernel<<<gridDim,blockDim>>>(arg);
        }else{
          computeFmunuCPU(arg);
        }
      }

      TuneKey tuneKey() const {
        std::stringstream vol, aux;
        vol << arg.threads;
        aux << "stride=" << arg.FmunuStride;
        return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }

      std::string paramString(const TuneParam &param) const {
        std::stringstream ps;
        ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
        ps << "shared=" << param.shared_bytes;
        return ps.str();
      }

      long long flops() const { return 0; } // Fix this!
      long long bytes() const { return 0; } // Fix this!

    }; // FmunuCompute

  // Put into clover order 
  // Upper-left block (chirality index 0)
  //     /                                                                              \
  //     |  1 - c*(F[0,1] - F[2,3]) ,     -c*(F[1,2] - F[0,3]) + c*I*(F[0,2] + F[1,3])   |
  //     |                                                                              |
  //     |  -c*(F[1,2] - F[0,3]) - c*I*(F[0,2] + F[1,3]),   1 + c*(F[0,1] - F[2,3])      |
  //     |                                                                              |
  //     \                                                                              / 

  //     /
  //     | 1 - c*(F[0] - F[5]),   -c*(F[2] - F[3]) + c*I*(F[1] + F[4])  
  //     |
  //     |  -c*(F[2] -F[3]) - c*I*(F[1] + F[4]),   1 + c*(F[0] - F[5])  
  //     |
  //     \
  // 
  // Lower-right block (chirality index 1)
  //
  //     /                                                               \
  //     |  1 - c*(F[0] + F[5]),  -c*(F[2] + F[3]) + c*I*(F[1] - F[4])    |
  //     |                                                               |
  //     |    -c*(F[2]+F[3]) - c*I*(F[1]-F[4]),     1+ c*(F[0] + F[5])    |
  //     \                                                               / 
  //

  // Core routine for constructing clover term from field strength
  template<typename Float, typename Clover>
    __device__ __host__
    void cloverComputeCore(CloverArg<Float,Clover> arg, int idx){

      int parity = 0;  
      if(idx > arg.threads/2){
        parity = 1;
        idx -= arg.threads/2;
      }

      typedef typename ComplexTypeId<Float>::Type Cmplx;

      Float cloverCoeff = arg.cloverCoeff;

      // Load the field-strength tensor from global memory
      Matrix<Cmplx,3> F[5];
      for(int i=0; i<6; ++i){
        loadLinkVariableFromArray(arg.Fmunu + parity*arg.FmunuOffset, i, idx, arg.FmunuStride, &F[i]); 
      }

      Cmplx I; I.x = 0; I.y = 1.;
      Matrix<Cmplx,3> block1[2];
      Matrix<Cmplx,3> block2[2];
      block1[0] =  cloverCoeff*(F[0]-F[5]);
      block1[1] =  cloverCoeff*(F[0]+F[5]);
      block2[0] = -cloverCoeff*(F[2]-F[3]) - cloverCoeff*I*(F[1]+F[4]);
      block2[1] = -cloverCoeff*(F[2]+F[3]) - cloverCoeff*I*(F[1]-F[4]);


      const int idtab[15]={0,1,3,6,10,2,4,7,11,5,8,12,9,13,14};
      Float diag[6];
      Cmplx triangle[15]; 
      Float A[72];

      // This uses lots of unnecessary memory
      for(int ch=0; ch<2; ++ch){ 
        // c = 0(1) => positive(negative) chiral block
        // Compute real diagonal elements
        for(int i=0; i<3; ++i){
          diag[i]   = 1 - block1[ch](i,i).x;
          diag[i+3] = 1 + block1[ch](i,i).x;
        }

        // Compute off diagonal components
        // First row
        triangle[0]  = - block1[ch](1,0);
        // Second row
        triangle[1]  = - block1[ch](2,0);
        triangle[2]  = - block1[ch](2,1);
        // Third row
        triangle[3]  =   block2[ch](0,0);
        triangle[4]  =   block2[ch](0,1);
        triangle[5]  =   block2[ch](0,2);
        // Fourth row 
        triangle[6]  =   block2[ch](1,0);
        triangle[7]  =   block2[ch](1,1);
        triangle[8]  =   block2[ch](1,2);
        triangle[9]  =   block1[ch](1,0);
        // Fifth row
        triangle[10] =   block2[ch](2,0);
        triangle[11] =   block2[ch](2,1);
        triangle[12] =   block2[ch](2,2);
        triangle[13] =   block1[ch](2,0);
        triangle[14] =   block1[ch](2,1);

        for(int i=0; i<6; ++i){
          A[ch*36 + i] = diag[i];
        } 
        for(int i=0; i<15; ++i){
          A[ch*36+6+2*i]     = triangle[idtab[i]].x;
          A[ch*36+6+2*i + 1] = triangle[idtab[i]].y;
        } 
      } // ch

      arg.clover.save(A, idx, parity);
      return;
    }


  template<typename Float, typename Clover>
    __global__
    void cloverComputeKernel(CloverArg<Float,Clover> arg){
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      if(idx >= arg.threads) return;
      cloverComputeCore(arg, idx);
    }

  template<typename Float, typename Clover>
    void cloverComputeCPU(CloverArg<Float,Clover> arg){
      for(int idx=0; idx<arg.threads; ++idx){
        cloverComputeCore(arg, idx);
      }
    }


  template<typename Float, typename Clover>
    class CloverCompute : Tunable {
      CloverArg<Float, Clover> arg;
      const QudaFieldLocation location;

      private: 
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; } // Don't tune the shared memory.
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      CloverCompute(CloverArg<Float,Clover> &arg, QudaFieldLocation location) 
        : arg(arg), location(location) {}

      virtual ~CloverCompute() {}

      void apply(const cudaStream_t &stream) {
        if(location == QUDA_CUDA_FIELD_LOCATION){
          // Fix this
          dim3 blockDim(128, 1, 1);
          dim3 gridDim((arg.threads + blockDim.x - 1) / blockDim.x, 1, 1);
          cloverComputeKernel<<<gridDim,blockDim>>>(arg);
        }else{
          cloverComputeCPU(arg);
        }
      }

      TuneKey tuneKey() const {
        std::stringstream vol, aux;
        vol << arg.threads;
        aux << "stride=" << arg.clover.stride;
        return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }

      std::string paramString(const TuneParam &param) const { // Don't print the grid dim.
        std::stringstream ps;
        ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
        ps << "shared=" << param.shared_bytes;
        return ps.str();
      }

      long long flops() const { return 0; } // Fix this
      long long bytes() const { return 0; } // Fix this
    };



  template<typename Float,typename Clover>
    void computeClover(Clover clover, const GaugeField& gauge, GaugeField& Fmunu, Float cloverCoeff, QudaFieldLocation location){
      CloverArg<Float,Clover> arg(clover, gauge, Fmunu, cloverCoeff);

      FmunuCompute<Float,Clover> fmunuCompute(arg, location);
      fmunuCompute.apply(0);

      CloverCompute<Float,Clover> cloverCompute(arg, location);
      cloverCompute.apply(0);

      cudaDeviceSynchronize();
    }



  template<typename Float>
    void computeClover(CloverField &clover, const GaugeField& gauge, Float cloverCoeff, QudaFieldLocation location){
      int pad = 0;
      GaugeFieldParam tensorParam(gauge.X(), gauge.Precision(), QUDA_RECONSTRUCT_NO, pad, QUDA_TENSOR_GEOMETRY);

      GaugeField* Fmunu = NULL;
      if(location == QUDA_CPU_FIELD_LOCATION){
        Fmunu = new cpuGaugeField(tensorParam);
      } else if (location == QUDA_CUDA_FIELD_LOCATION){
        Fmunu = new cudaGaugeField(tensorParam); 
      } else {
        errorQuda("Invalid location\n");
      }

      if(clover.Order() == QUDA_FLOAT2_CLOVER_ORDER){
        computeClover(CloverOrder::quda::FloatNOrder<Float,72,2>(clover,0), gauge, *Fmunu, cloverCoeff, location);
      }else if(clover.Order() == QUDA_FLOAT4_CLOVER_ORDER){
        computeClover(CloverOrder::quda::FloatNOrder<Float,72,4>(clover,0), gauge, *Fmunu, cloverCoeff, location);
      }

      if(Fmunu) delete Fmunu;
    }


  void computeClover(CloverField &clover, const GaugeField& gauge, double cloverCoeff, QudaFieldLocation location){

    if(clover.Precision() == QUDA_HALF_PRECISION){
      errorQuda("Half precision not supported\n");
    }

    if (clover.Precision() == QUDA_SINGLE_PRECISION){
      computeClover<float>(clover, gauge, cloverCoeff, location);
    } else if(clover.Precision() == QUDA_DOUBLE_PRECISION) {
      computeClover<double>(clover, gauge, cloverCoeff, location);
    } else {
      errorQuda("Precision %d not supported", clover.Precision());
    }
    return;
  }

} // namespace quda

