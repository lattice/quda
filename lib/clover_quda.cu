#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <clover_field.h>
#include <gauge_field.h>
#include <gauge_field_order.h>

namespace CloverOrder {
  using namespace quda;
#include <clover_field_order.h>
} // CloverOrder



namespace quda {

  template<typename Float, typename Clover, typename Gauge>
    struct CloverArg {
      int threads; // number of active threads required
      int X[4]; // grid dimensions
#ifdef MULTI_GPU
      int border[4]; 
#endif
      double cloverCoeff;

      int FmunuStride; // stride used on Fmunu field
      int FmunuOffset; // parity offset 

      typename ComplexTypeId<Float>::Type* Fmunu;
      Gauge  gauge;
      Clover clover;

      CloverArg(Clover &clover, Gauge &gauge, GaugeField& Fmunu, double cloverCoeff)
        : threads(Fmunu.Volume()), 
        cloverCoeff(cloverCoeff),
        FmunuStride(Fmunu.Stride()), FmunuOffset(Fmunu.Bytes()/(4*sizeof(Float))),
        Fmunu(reinterpret_cast<typename ComplexTypeId<Float>::Type*>(Fmunu.Gauge_p())),
        gauge(gauge), clover(clover) { 
          for(int dir=0; dir<4; ++dir) X[dir] = Fmunu.X()[dir];

#ifdef MULTI_GPU
          for(int dir=0; dir<4; ++dir){
            border[dir] = 2;
          }
#endif
        }
    };

  __device__ __host__ inline int linkIndex(int x[], int dx[], const int X[4]) {
    int y[4];
    for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
    int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
    return idx;
  }


  __device__ __host__ inline void getCoords(int x[4], int cb_index, const int X[4], int parity)
  {
    x[3] = cb_index/(X[2]*X[1]*X[0]/2);
    x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    x[1] = (cb_index/(X[0]/2)) % X[1];
    x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    return;
  }





  template <typename Float, typename Clover, typename GaugeOrder>
    __host__ __device__ void computeFmunuCore(CloverArg<Float,Clover,GaugeOrder>& arg, int idx) {

      // compute spacetime dimensions and parity
      int parity = 0;
      if(idx >= arg.threads/2){
        parity = 1;
        idx -= arg.threads/2;
      }

      int X[4]; 
      for(int dir=0; dir<4; ++dir) X[dir] = arg.X[dir];

      int x[4];
      getCoords(x, idx, X, parity);
#ifdef MULTI_GPU
      for(int dir=0; dir<4; ++dir){
           x[dir] += arg.border[dir];
           X[dir] += 2*arg.border[dir];
      }
#endif

      typedef typename ComplexTypeId<Float>::Type Cmplx;



      for (int mu=0; mu<4; mu++) {
        for (int nu=0; nu<mu; nu++) {
          Matrix<Cmplx,3> F;
          setZero(&F);
          { // U(x,mu) U(x+mu,nu) U[dagger](x+nu,mu) U[dagger](x,nu)

            // load U(x)_(+mu)
            Matrix<Cmplx,3> U1;
            int dx[4] = {0, 0, 0, 0};
            arg.gauge.load((Float*)(U1.data),linkIndex(x,dx,X), mu, parity); 
            // load U(x+mu)_(+nu)
            Matrix<Cmplx,3> U2;
            dx[mu]++;
            arg.gauge.load((Float*)(U2.data),linkIndex(x,dx,X), nu, 1-parity); 
            dx[mu]--;
   

            Matrix<Cmplx,3> Ftmp = U1 * U2;

            // load U(x+nu)_(+mu)
            Matrix<Cmplx,3> U3;
            dx[nu]++;
            arg.gauge.load((Float*)(U3.data),linkIndex(x,dx,X), mu, 1-parity); 
            dx[nu]--;

            Ftmp = Ftmp * conj(U3) ;

            // load U(x)_(+nu)
            Matrix<Cmplx,3> U4;
            arg.gauge.load((Float*)(U4.data),linkIndex(x,dx,X), nu, parity); 

            // complete the plaquette
            F = Ftmp * conj(U4);
          }


          { // U(x,nu) U[dagger](x+nu-mu,mu) U[dagger](x-mu,nu) U(x-mu, mu)

            // load U(x)_(+nu)
            Matrix<Cmplx,3> U1;
            int dx[4] = {0, 0, 0, 0};
            arg.gauge.load((Float*)(U1.data), linkIndex(x,dx,X), nu, parity);

            // load U(x+nu)_(-mu) = U(x+nu-mu)_(+mu)
            Matrix<Cmplx,3> U2;
            dx[nu]++;
            dx[mu]--;
            arg.gauge.load((Float*)(U2.data), linkIndex(x,dx,X), mu, parity);
            dx[mu]++;
            dx[nu]--;

            Matrix<Cmplx,3> Ftmp =  U1 * conj(U2);

            // load U(x-mu)_nu
            Matrix<Cmplx,3> U3;
            dx[mu]--;
            arg.gauge.load((Float*)(U3.data), linkIndex(x,dx,X), nu, 1-parity);
            dx[mu]++;

            Ftmp =  Ftmp * conj(U3);

            // load U(x)_(-mu) = U(x-mu)_(+mu)
            Matrix<Cmplx,3> U4;
            dx[mu]--;
            arg.gauge.load((Float*)(U4.data), linkIndex(x,dx,X), mu, 1-parity);
            dx[mu]++;

            // complete the plaquette
            Ftmp = Ftmp * U4;

            // sum this contribution to Fmunu
            F += Ftmp;
          }

          { // U[dagger](x-nu,nu) U(x-nu,mu) U(x+mu-nu,nu) U[dagger](x,mu)


            // load U(x)_(-nu)
            Matrix<Cmplx,3> U1;
            int dx[4] = {0, 0, 0, 0};
            dx[nu]--;
            arg.gauge.load((Float*)(U1.data), linkIndex(x,dx,X), nu, 1-parity);
            dx[nu]++;

            // load U(x-nu)_(+mu)
            Matrix<Cmplx,3> U2;
            dx[nu]--;
            arg.gauge.load((Float*)(U2.data), linkIndex(x,dx,X), mu, 1-parity);
            dx[nu]++;

            Matrix<Cmplx,3> Ftmp = conj(U1) * U2;

            // load U(x+mu-nu)_(+nu)
            Matrix<Cmplx,3> U3;
            dx[mu]++;
            dx[nu]--;
            arg.gauge.load((Float*)(U3.data), linkIndex(x,dx,X), nu, parity);
            dx[nu]++;
            dx[mu]--;

            Ftmp = Ftmp * U3;

            // load U(x)_(+mu)
            Matrix<Cmplx,3> U4;
            arg.gauge.load((Float*)(U4.data), linkIndex(x,dx,X), mu, parity);

            Ftmp = Ftmp * conj(U4);

            // sum this contribution to Fmunu
            F += Ftmp;
          }

          { // U[dagger](x-mu,mu) U[dagger](x-mu-nu,nu) U(x-mu-nu,mu) U(x-nu,nu)


            // load U(x)_(-mu)
            Matrix<Cmplx,3> U1;
            int dx[4] = {0, 0, 0, 0};
            dx[mu]--;
            arg.gauge.load((Float*)(U1.data), linkIndex(x,dx,X), mu, 1-parity);
            dx[mu]++;



            // load U(x-mu)_(-nu) = U(x-mu-nu)_(+nu)
            Matrix<Cmplx,3> U2;
            dx[mu]--;
            dx[nu]--;
            arg.gauge.load((Float*)(U2.data), linkIndex(x,dx,X), nu, parity);
            dx[nu]++;
            dx[mu]++;

            Matrix<Cmplx,3> Ftmp = conj(U1) * conj(U2);

            // load U(x-nu)_mu
            Matrix<Cmplx,3> U3;
            dx[mu]--;
            dx[nu]--;
            arg.gauge.load((Float*)(U3.data), linkIndex(x,dx,X), mu, parity);
            dx[nu]++;
            dx[mu]++;

            Ftmp = Ftmp * U3;

            // load U(x)_(-nu) = U(x-nu)_(+nu)
            Matrix<Cmplx,3> U4;
            dx[nu]--;
            arg.gauge.load((Float*)(U4.data), linkIndex(x,dx,X), nu, 1-parity);
            dx[nu]++;

            // complete the plaquette
            Ftmp = Ftmp * U4;

            // sum this contribution to Fmunu
            F += Ftmp;

          }
          // 3 matrix additions, 12 matrix-matrix multiplications, 8 matrix conjugations
          // Each matrix conjugation involves 9 unary minus operations
          // Each matrix addition involves 18 real additions
          // Each matrix-matrix multiplication involves 9*3 complex multiplications and 9*2 complex additions 
          // = 9*3*6 + 9*2*2 = 198 floating-point ops
          // => Total number of floating point ops per site above is 
          // 8*9 + 3*18 + 12*198 = 72 + 54 + 2376 = 2502 
          
          { 
            F -= conj(F); // 18 real subtractions + one matrix conjugation (=9 unary minus ops)
            F *= 1.0/8.0; // 18 real multiplications
            // 45 floating point operations here
          }
          


          Cmplx* thisFmunu = arg.Fmunu + parity*arg.FmunuOffset;
          int munu_idx = (mu*(mu-1))/2 + nu; // lower-triangular indexing
  
          writeLinkVariableToArray(F, munu_idx, idx, arg.FmunuStride, thisFmunu);
        } // nu < mu
      } // mu
      // F[1,0], F[2,0], F[2,1], F[3,0], F[3,1], F[3,2]
      return;
    }



  template<typename Float, typename Clover, typename Gauge>
    __global__ void computeFmunuKernel(CloverArg<Float,Clover,Gauge> arg){
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      if(idx >= arg.threads) return;
      computeFmunuCore<Float,Clover,Gauge>(arg,idx);
    }

  template<typename Float, typename Clover, typename Gauge>
    void computeFmunuCPU(CloverArg<Float,Clover,Gauge>& arg){
      errorQuda("computeFmunuCPU not yet supported\n");
      for(int idx=0; idx<arg.threads; idx++){
        computeFmunuCore(arg,idx);
      }
    }



  template<typename Float, typename Clover, typename Gauge>
    class FmunuCompute : Tunable {
      CloverArg<Float,Clover,Gauge> arg;
      const QudaFieldLocation location;

      private: 
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; } // Don't tune shared memory
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      FmunuCompute(CloverArg<Float,Clover,Gauge> &arg, QudaFieldLocation location)
        : arg(arg), location(location) {}
      virtual ~FmunuCompute() {}

      void apply(const cudaStream_t &stream){
        if(location == QUDA_CUDA_FIELD_LOCATION){
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          computeFmunuKernel<<<tp.grid,tp.block,tp.shared_bytes>>>(arg);  
        }else{
          computeFmunuCPU(arg);
        }
      }

      TuneKey tuneKey() const {
        std::stringstream vol, aux;
        vol << arg.X[0] << "x";
        vol << arg.X[1] << "x";
        vol << arg.X[2] << "x";
        vol << arg.X[3];
        aux << "threads=" << arg.threads << ",prec="  << sizeof(Float);
        aux << ",stride=" << arg.clover.stride;
        return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }


      std::string paramString(const TuneParam &param) const {
        std::stringstream ps;
        ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
        ps << "shared=" << param.shared_bytes;
        return ps.str();
      }

      void preTune(){}
      void postTune(){}
      long long flops() const { return (2502 + 45)*6*arg.threads; }
      long long bytes() const { return (4*4*18 + 18)*6*arg.threads*sizeof(Float); } // Only correct if there is no link reconstruction

    }; // FmunuCompute

  // Put into clover order 
  // Upper-left block (chirality index 0)
  //     /                                                                                \
  //     |  1 + c*I*(F[0,1] - F[2,3]) ,     c*I*(F[1,2] - F[0,3]) + c*(F[0,2] + F[1,3])   |
  //     |                                                                                |
  //     |  c*I*(F[1,2] - F[0,3]) - c*(F[0,2] + F[1,3]),   1 - c*I*(F[0,1] - F[2,3])      |
  //     |                                                                                |
  //     \                                                                                / 

  //     /
  //     | 1 - c*I*(F[0] - F[5]),   -c*I*(F[2] - F[3]) - c*(F[1] + F[4])  
  //     |
  //     |  -c*I*(F[2] -F[3]) + c*(F[1] + F[4]),   1 + c*I*(F[0] - F[5])  
  //     |
  //     \
  // 
  // Lower-right block (chirality index 1)
  //
  //     /                                                                  \
  //     |  1 - c*I*(F[0] + F[5]),  -c*I*(F[2] + F[3]) - c*(F[1] - F[4])    |
  //     |                                                                  |
  //     |  -c*I*(F[2]+F[3]) + c*(F[1]-F[4]),     1 + c*I*(F[0] + F[5])     |
  //     \                                                                  / 
  //

  // Core routine for constructing clover term from field strength
  template<typename Float, typename Clover, typename Gauge>
    __device__ __host__
    void cloverComputeCore(CloverArg<Float,Clover,Gauge>& arg, int idx){

      int parity = 0;  
      if(idx >= arg.threads/2){
        parity = 1;
        idx -= arg.threads/2;
      }
      typedef typename ComplexTypeId<Float>::Type Cmplx;

      Float cloverCoeff = arg.cloverCoeff;

      // Load the field-strength tensor from global memory
      Matrix<Cmplx,3> F[6];
      for(int i=0; i<6; ++i){
        loadLinkVariableFromArray(arg.Fmunu + parity*arg.FmunuOffset, i, idx, arg.FmunuStride, &F[i]); 
      }

      Cmplx I; I.x = 0; I.y = 1.;
      Matrix<Cmplx,3> block1[2];
      Matrix<Cmplx,3> block2[2];
      block1[0] =  cloverCoeff*I*(F[0]-F[5]); // (18 + 6*9 + 18 =) 90 floating-point ops 
      block1[1] =  cloverCoeff*I*(F[0]+F[5]); // 90 floating-point ops 
      block2[0] =  cloverCoeff*(F[1]+F[4] - I*(F[2]-F[3])); // 108 floating-point ops
      block2[1] =  cloverCoeff*(F[1]-F[4] - I*(F[2]+F[3])); // 108 floating-point ops


      const int idtab[15]={0,1,3,6,10,2,4,7,11,5,8,12,9,13,14};
      Float diag[6];
      Cmplx triangle[15]; 
      Float A[72];

      // This uses lots of unnecessary memory
      for(int ch=0; ch<2; ++ch){ 
        // c = 0(1) => positive(negative) chiral block
        // Compute real diagonal elements
        for(int i=0; i<3; ++i){
          diag[i]   = 1.0 - block1[ch](i,i).x;
          diag[i+3] = 1.0 + block1[ch](i,i).x;
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
          A[ch*36 + i] = 0.5*diag[i];
        } 
        for(int i=0; i<15; ++i){
          A[ch*36+6+2*i]     = 0.5*triangle[idtab[i]].x;
          A[ch*36+6+2*i + 1] = 0.5*triangle[idtab[i]].y;
        } 
      } // ch
      // 96 floating-point ops


      arg.clover.save(A, idx, parity);
      return;
    }


  template<typename Float, typename Clover, typename Gauge>
    __global__
    void cloverComputeKernel(CloverArg<Float,Clover,Gauge> arg){
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      if(idx >= arg.threads) return;
      cloverComputeCore(arg, idx);
    }

  template<typename Float, typename Clover, typename Gauge>
    void cloverComputeCPU(CloverArg<Float,Clover,Gauge> arg){
      for(int idx=0; idx<arg.threads; ++idx){
        cloverComputeCore(arg, idx);
      }
    }


  template<typename Float, typename Clover, typename Gauge>
    class CloverCompute : Tunable {
      CloverArg<Float, Clover, Gauge> arg;
      const QudaFieldLocation location;

      private: 
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; } // Don't tune the shared memory.
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      CloverCompute(CloverArg<Float,Clover,Gauge> &arg, QudaFieldLocation location) 
        : arg(arg), location(location) {}

      virtual ~CloverCompute() {}

      void apply(const cudaStream_t &stream) {
        if(location == QUDA_CUDA_FIELD_LOCATION){
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          cloverComputeKernel<<<tp.grid,tp.block,tp.shared_bytes>>>(arg);  
        }else{ // run the CPU code
          cloverComputeCPU(arg);
        }
      }

      TuneKey tuneKey() const {
        std::stringstream vol, aux;
        vol << arg.X[0] << "x";
        vol << arg.X[1] << "x";
        vol << arg.X[2] << "x";
        vol << arg.X[3];
        aux << "threads=" << arg.threads << ",prec="  << sizeof(Float);
        aux << ",stride=" << arg.clover.stride;
        return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      }

      std::string paramString(const TuneParam &param) const { // Don't print the grid dim.
        std::stringstream ps;
        ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
        ps << "shared=" << param.shared_bytes;
        return ps.str();
      }

      void preTune(){}
      void postTune(){}
      long long flops() const { return 492*arg.threads; } 
      long long bytes() const { return arg.threads*(6*18 + 72)*sizeof(Float); } 
    };



  template<typename Float,typename Clover,typename Gauge>
    void computeClover(Clover clover, Gauge gauge, GaugeField& Fmunu, Float cloverCoeff, QudaFieldLocation location){
      CloverArg<Float,Clover,Gauge> arg(clover, gauge, Fmunu, cloverCoeff);
      FmunuCompute<Float,Clover,Gauge> fmunuCompute(arg, location);
      fmunuCompute.apply(0);
      CloverCompute<Float,Clover,Gauge> cloverCompute(arg, location);
      cloverCompute.apply(0);
      cudaDeviceSynchronize();
    }


  template<typename Float>
    void computeClover(CloverField &clover, const GaugeField& gauge, Float cloverCoeff, QudaFieldLocation location){
      int pad = 0;
      GaugeFieldParam tensorParam(clover.X(), clover.Precision(), QUDA_RECONSTRUCT_NO, pad, QUDA_TENSOR_GEOMETRY);
      tensorParam.siteSubset = QUDA_FULL_SITE_SUBSET;
      GaugeField* Fmunu = NULL;
      if(location == QUDA_CPU_FIELD_LOCATION){
        Fmunu = new cpuGaugeField(tensorParam);
      } else if (location == QUDA_CUDA_FIELD_LOCATION){
        Fmunu = new cudaGaugeField(tensorParam); 
      } else {
        errorQuda("Invalid location\n");
      }

      // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
      // Need to fix this!!

      if(clover.Order() == QUDA_FLOAT2_CLOVER_ORDER){
        if(gauge.Order() == QUDA_FLOAT2_GAUGE_ORDER){
          if(gauge.Reconstruct() == QUDA_RECONSTRUCT_NO){
            computeClover(CloverOrder::quda::FloatNOrder<Float,72,2>(clover,0), FloatNOrder<Float, 18, 2, 18>(gauge), *Fmunu, cloverCoeff, location);  
          }else if(gauge.Reconstruct() == QUDA_RECONSTRUCT_12){
            computeClover(CloverOrder::quda::FloatNOrder<Float,72,2>(clover,0), FloatNOrder<Float, 18, 2, 12>(gauge),  *Fmunu, cloverCoeff, location);

          }else{
            errorQuda("Reconstruction type %d not supported",gauge.Reconstruct());
          }

        }else if(gauge.Order() == QUDA_FLOAT4_GAUGE_ORDER){
          if(gauge.Reconstruct() == QUDA_RECONSTRUCT_12){
            computeClover(CloverOrder::quda::FloatNOrder<Float,72,2>(clover,0), FloatNOrder<Float,18,4,12>(gauge),  *Fmunu, cloverCoeff, location);
          }else{
            errorQuda("Reconstruction type %d not supported",gauge.Reconstruct());
          }
        }
      }else if(clover.Order() == QUDA_FLOAT4_CLOVER_ORDER){
        if(gauge.Order() == QUDA_FLOAT2_GAUGE_ORDER){
          if(gauge.Reconstruct() == QUDA_RECONSTRUCT_NO){
            computeClover(CloverOrder::quda::FloatNOrder<Float,72,4>(clover,0), FloatNOrder<Float,18,2,18>(gauge),  *Fmunu, cloverCoeff, location);
          }else if(gauge.Reconstruct() == QUDA_RECONSTRUCT_12){
            computeClover(CloverOrder::quda::FloatNOrder<Float,72,4>(clover,0), FloatNOrder<Float,18,2,12>(gauge),  *Fmunu, cloverCoeff, location);
          }else{
            errorQuda("Reconstruction type %d not supported",gauge.Reconstruct());
          }

        }else if(gauge.Order() == QUDA_FLOAT4_GAUGE_ORDER){
          if(gauge.Reconstruct() == QUDA_RECONSTRUCT_12){
            computeClover(CloverOrder::quda::FloatNOrder<Float,72,4>(clover,0), FloatNOrder<Float,18,4,12>(gauge), *Fmunu, cloverCoeff, location);
          }else{
            errorQuda("Reconstruction type %d not supported",gauge.Reconstruct());
          } // gauge order
        }
      } // clover order

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

