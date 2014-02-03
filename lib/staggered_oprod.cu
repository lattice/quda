#include <cstdio>
#include <cstdlib>
#include <staggered_oprod.h>

#include <tune_quda.h>
#include <quda_internal.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>

namespace quda {

  namespace { // anonymous
#include <texture.h>
  }

  static bool kernelPackT = true;

  template<int N>
    void createEventArray(cudaEvent_t (&event)[N], unsigned int flags=cudaEventDefault)
    {
      for(int i=0; i<N; ++i)
        cudaEventCreate(&event[i],flags);
      return;
    }

  template<int N>
    void destroyEventArray(cudaEvent_t (&event)[N])
    {
      for(int i=0; i<N; ++i)
        cudaEventDestroy(event[i]);
    }


  static cudaEvent_t packEnd;
  static cudaEvent_t gatherEnd[4];
  static cudaEvent_t scatterEnd[4];
  static cudaEvent_t oprodStart;
  static cudaEvent_t oprodEnd;


  void createStaggeredOprodEvents(){
#ifdef MULTI_GPU
    cudaEventCreate(&packEnd, cudaEventDisableTiming);
    createEventArray(gatherEnd, cudaEventDisableTiming);
    createEventArray(scatterEnd, cudaEventDisableTiming);
#endif
    cudaEventCreate(&oprodStart, cudaEventDisableTiming);
    cudaEventCreate(&oprodEnd, cudaEventDisableTiming);
    return;
  }

  void destroyStaggeredOprodEvents(){
#ifdef MULTI_GPU
    destroyEventArray(gatherEnd);
    destroyEventArray(scatterEnd);
    cudaEventDestroy(packEnd);
#endif
    cudaEventDestroy(oprodStart);
    cudaEventDestroy(oprodEnd);
    return;
  }


  enum KernelType {OPROD_INTERIOR_KERNEL, OPROD_EXTERIOR_KERNEL};

  template<typename Complex, typename Output, typename Input>
    struct StaggeredOprodArg {
      unsigned int length;
      unsigned int X[4];
      unsigned int parity;
      unsigned int dir;
      unsigned int ghostOffset;
      unsigned int displacement;
      KernelType kernelType;
      bool partitioned[4];
      Input inA;
      Input inB;
      Output outA;
      Output outB;
      typename RealTypeId<Complex>::Type coeff[2];

      StaggeredOprodArg(const unsigned int length,
          const unsigned int X[4],
          const unsigned int parity,
          const unsigned int dir,
          const unsigned int ghostOffset,
          const unsigned int displacement,   
          const KernelType& kernelType, 
          const double coeff[2],
          Input& inA,
          Input& inB,
          Output& outA,
          Output& outB) : length(length), parity(parity), ghostOffset(ghostOffset), 
      displacement(displacement), kernelType(kernelType), inA(inA), inB(inB), outA(outA), outB(outB) 
      {
        this->coeff[0] = coeff[0];
        this->coeff[1] = coeff[1];
        for(int i=0; i<4; ++i) this->X[i] = X[i];
        for(int i=0; i<4; ++i) this->partitioned[i] = commDimPartitioned(i) ? true : false;
      }
    };

  enum IndexType {
    EVEN_X = 0,
    EVEN_Y = 1,
    EVEN_Z = 2,
    EVEN_T = 3
  };

  template <IndexType idxType>
    static __device__ __forceinline__ void coordsFromIndex(int& idx, int c[4],  
        const unsigned int cb_idx, const unsigned int parity, const unsigned int X[4])
    {
      const unsigned int &LX = X[0];
      const unsigned int &LY = X[1];
      const unsigned int &LZ = X[2];
      const unsigned int XYZ = X[2]*X[1]*X[0];
      const unsigned int XY = X[1]*X[0];

      idx = 2*cb_idx;

      int x, y, z, t;

      if (idxType == EVEN_X /*!(LX & 1)*/) { // X even
        //   t = idx / XYZ;
        //   z = (idx / XY) % Z;
        //   y = (idx / X) % Y;
        //   idx += (parity + t + z + y) & 1;
        //   x = idx % X;
        // equivalent to the above, but with fewer divisions/mods:
        int aux1 = idx / LX;
        x = idx - aux1 * LX;
        int aux2 = aux1 / LY;
        y = aux1 - aux2 * LY;
        t = aux2 / LZ;
        z = aux2 - t * LZ;
        aux1 = (parity + t + z + y) & 1;
        x += aux1;
        idx += aux1;
      } else if (idxType == EVEN_Y /*!(LY & 1)*/) { // Y even
        t = idx / XYZ;
        z = (idx / XY) % LZ;
        idx += (parity + t + z) & 1;
        y = (idx / LX) % LY;
        x = idx % LX;
      } else if (idxType == EVEN_Z /*!(LZ & 1)*/) { // Z even
        t = idx / XYZ;
        idx += (parity + t) & 1;
        z = (idx / XY) % LZ;
        y = (idx / LX) % LY;
        x = idx % LX;
      } else {
        idx += parity;
        t = idx / XYZ;
        z = (idx / XY) % LZ;
        y = (idx / LX) % LY;
        x = idx % LX;
      }

      c[0] = x;
      c[1] = y;
      c[2] = z;
      c[3] = t;
    }




  // Get the  coordinates for the exterior kernels
  template<int Nspin>
    __device__ void coordsFromIndex(unsigned int x[4], const unsigned int cb_idx, const unsigned int X[4], const unsigned int dir, const int displacement, const unsigned int parity)
    {

      if(Nspin == 1){
        unsigned int Xh[2] = {X[0]/2, X[1]/2};
        switch(dir){
          case 0:
            x[2] = cb_idx/Xh[1] % X[2];
            x[3] = cb_idx/(Xh[1]*X[2]) % X[3];
            x[0] = cb_idx/(Xh[1]*X[2]*X[3]);
            x[0] += (X[0] - displacement);
            x[1] = 2*(cb_idx % Xh[1]) + ((x[0]+x[2]+x[3]+parity)&1);
            break;

          case 1:
            x[2] = cb_idx/Xh[0] % X[2];
            x[3] = cb_idx/(Xh[0]*X[2]) % X[3];
            x[1] = cb_idx/(Xh[0]*X[2]*X[3]);
            x[1] += (X[1] - displacement);
            x[0] = 2*(cb_idx % Xh[0]) + ((x[1]+x[2]+x[3]+parity)&1);

            break;

          case 2:
            x[1] = cb_idx/Xh[0] % X[1];
            x[3] = cb_idx/(Xh[0]*X[1]) % X[3];
            x[2] = cb_idx/(Xh[0]*X[1]*X[3]);
            x[2] += (X[2] - displacement);
            x[0] = 2*(cb_idx % Xh[0]) + ((x[1]+x[2]+x[3]+parity)&1);

            break;

          case 3:
            x[1] = cb_idx/Xh[0] % X[1];
            x[2] = cb_idx/(Xh[0]*X[1]) % X[2];
            x[3] = cb_idx/(Xh[0]*X[1]*X[2]);
            x[3] += (X[3] - displacement);
            x[0] = 2*(cb_idx % Xh[0]) + ((x[1]+x[2]+x[3]+parity)&1);

            break;
        }
      }else if(Nspin == 3){
        // currently unsupported
      }
      return;
    }


  template<int Nspin, int Nface> 
    __device__  int ghostIndexFromCoords(const unsigned int x[4], const unsigned int X[4], const unsigned int dir, const int shift){
      return 0;
    }



  template<>
    __device__  int ghostIndexFromCoords<3,3>(
        const unsigned int x[4],
        const unsigned int X[4], 
        unsigned int dir, 
        const int shift)
    {
      unsigned int ghost_idx;
      if(shift > 0){
        if((x[dir] + shift) >= X[dir]){
          switch(dir){
            case 0:
              ghost_idx = (3*3 + (x[0]-X[0]+shift))*(X[3]*X[2]*X[1])/2 + ((x[3]*X[2] + x[2])*X[1] + x[1])/2;
              break;          
            case 1:
              ghost_idx = (3*3 + (x[1]-X[1]+shift))*(X[3]*X[2]*X[0])/2 + (x[3]*X[2]*X[0] + x[2]*X[0] + x[0])/2;
              break;
            case 2:
              ghost_idx = (3*3 + (x[2]-X[2]+shift))*(X[3]*X[1]*X[0])/2 + (x[3]*X[1]*X[0] + x[1]*X[0] + x[0])/2;
              break;
            case 3:
              ghost_idx = (3*3 + (x[3]-X[3]+shift))*(X[2]*X[1]*X[0])/2 + (x[2]*X[1]*X[0] + x[1]*X[0] + x[0])/2;
              break;
            default:
              break;
          } // switch
        } // x[dir] + shift[dir] >= X[dir]
      }else{ // shift < 0
        if(static_cast<int>(x[dir]) + shift < 0){
          switch(dir){
            case 0:
              ghost_idx = (3 + shift)*(X[3]*X[2]*X[1])/2 + ((x[3]*X[2] + x[2])*X[1] + x[1])/2;
              break;
            case 1:
              ghost_idx = (3 + shift)*(X[3]*X[2]*X[0])/2 + ((x[3]*X[2] + x[2])*X[0] + x[0])/2;
              break;
            case 2:
              ghost_idx = (3 + shift)*(X[3]*X[1]*X[0])/2 + ((x[3]*X[1] + x[1])*X[0]  + x[0])/2;
              break;
            case 3:
              ghost_idx = (3 + shift)*(X[2]*X[1]*X[0])/2 + ((x[2]*X[1] + x[1])*X[0] + x[0])/2;
              break;
          } // switch(dir)
        }
      } // shift < 0

      return ghost_idx;
    }




  __device__ __forceinline__
    int neighborIndex(const unsigned int& cb_idx, const int shift[4],  const bool partitioned[4], const unsigned int& parity, 
        const unsigned int X[4]){

      int  full_idx;
      int x[4]; 


      coordsFromIndex<EVEN_X>(full_idx, x, cb_idx, parity, X);

#ifdef MULTI_GPU
      for(int dim = 0; dim<4; ++dim){
        if(partitioned[dim])
          if( (x[dim]+shift[dim])<0 || (x[dim]+shift[dim])>=X[dim]) return -1;
      }
#endif

      for(int dim=0; dim<4; ++dim){
        x[dim] = shift[dim] ? (x[dim]+shift[dim] + X[dim]) % X[dim] : x[dim];
      }
      return  (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
    }



  template<typename Complex, typename Output, typename Input>
    __global__ void interiorOprodKernel(StaggeredOprodArg<Complex, Output, Input> arg)
    {
      unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
      const unsigned int gridSize = gridDim.x*blockDim.x;

      typedef typename RealTypeId<Complex>::Type real;
      Complex x[3];
      Complex y[3];
      Complex z[3];
      Matrix<Complex,3> result;
      Matrix<Complex,3> tempA, tempB; // input


      while(idx<arg.length){
        arg.inA.load(x, idx);
        for(int dir=0; dir<4; ++dir){
          int shift[4] = {0,0,0,0};
          shift[dir] = 1;
          const int first_nbr_idx = neighborIndex(idx, shift, arg.partitioned, arg.parity, arg.X);
          if(first_nbr_idx >= 0){
            arg.inB.load(y, first_nbr_idx);
            outerProd(y,x,&result);
            arg.outA.load(reinterpret_cast<real*>(tempA.data), idx, dir, arg.parity); 
            result = tempA + result*arg.coeff[0];
            arg.outA.save(reinterpret_cast<real*>(result.data), idx, dir, arg.parity); 

            shift[dir] = 3;
            const int third_nbr_idx = neighborIndex(idx, shift, arg.partitioned, arg.parity, arg.X);
            if(third_nbr_idx >= 0){
              arg.inB.load(z, third_nbr_idx);
              outerProd(z, x, &result);
              arg.outB.load(reinterpret_cast<real*>(tempB.data), idx, dir, arg.parity); 
              result = tempB + result*arg.coeff[1];
              arg.outB.save(reinterpret_cast<real*>(result.data), idx, dir, arg.parity); 
            }
          }
        } // dir
        idx += gridSize;
      }
      return;
    } // interiorOprodKernel



  template<typename Complex, typename Output, typename Input> 
    __global__ void exteriorOprodKernel(StaggeredOprodArg<Complex, Output, Input> arg)
    {
      unsigned int cb_idx = blockIdx.x*blockDim.x + threadIdx.x;
      const unsigned int gridSize = gridDim.x*blockDim.x;

      Complex a[3];
      Complex b[3];
      Matrix<Complex,3> result;
      Matrix<Complex,3> inmatrix; // input
      typedef typename RealTypeId<Complex>::Type real;


      Output& out = (arg.displacement == 1) ? arg.outA : arg.outB;
      real coeff = (arg.displacement == 1) ? arg.coeff[0] : arg.coeff[1];

      unsigned int x[4];
      while(cb_idx<arg.length){
        coordsFromIndex<1>(x, cb_idx, arg.X, arg.dir, arg.displacement, arg.parity); 
        const unsigned int bulk_cb_idx = ((((x[3]*arg.X[2] + x[2])*arg.X[1] + x[1])*arg.X[0] + x[0]) >> 1);

        out.load(reinterpret_cast<real*>(inmatrix.data), bulk_cb_idx, arg.dir, arg.parity); 
        arg.inA.load(a, bulk_cb_idx);

        const unsigned int ghost_idx = arg.ghostOffset + ghostIndexFromCoords<3,3>(x, arg.X, arg.dir, arg.displacement);
        arg.inB.load(b, ghost_idx);

        outerProd(b,a,&result);
        result = inmatrix + result*coeff; 
        out.save(reinterpret_cast<real*>(result.data), bulk_cb_idx, arg.dir, arg.parity); 

        cb_idx += gridSize;
      }
      return;
    }



  template<typename Complex, typename Output, typename Input> 
    class StaggeredOprodField : public Tunable {

      private:
        StaggeredOprodArg<Complex,Output,Input> arg;
        QudaFieldLocation location; // location of the lattice fields

        unsigned int sharedBytesPerThread() const { return 0; }
        unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

        unsigned int minThreads() const { return arg.outA.volumeCB; }
        bool tunedGridDim() const { return false; }

      public:
        StaggeredOprodField(const StaggeredOprodArg<Complex,Output,Input> &arg,
            QudaFieldLocation location)
          : arg(arg), location(location) {} 

        virtual ~StaggeredOprodField() {}

        void set(const StaggeredOprodArg<Complex,Output,Input> &arg, QudaFieldLocation location){
          // This is a hack. Need to change this!
          this->arg.dir = arg.dir;
          this->arg.length = arg.length;
          this->arg.ghostOffset = arg.ghostOffset;
          this->arg.kernelType = arg.kernelType;
          this->location = location;
        } // set

        void apply(const cudaStream_t &stream){
          if(location == QUDA_CUDA_FIELD_LOCATION){
            // Disable tuning for the time being
            TuneParam tp;
            // TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
            if(arg.kernelType == OPROD_INTERIOR_KERNEL){
              //interiorOprodKernel<<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
              dim3 blockDim(128, 1, 1);
              const int gridSize = (arg.length + (blockDim.x-1))/blockDim.x;
              dim3 gridDim(gridSize, 1, 1);               
              interiorOprodKernel<<<gridDim,blockDim,0, stream>>>(arg);
            }else if(arg.kernelType == OPROD_EXTERIOR_KERNEL){
              const unsigned int volume = arg.X[0]*arg.X[1]*arg.X[2]*arg.X[3];
              arg.inB.setStride(3*volume/(2*arg.X[arg.dir]));
              exteriorOprodKernel<<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
              arg.inB.setStride(arg.inA.Stride());
            }else{
              errorQuda("Kernel type not supported\n");
            }
          }else{ // run the CPU code
            errorQuda("No CPU support for staggered outer-product calculation\n");
          }
        } // apply

        void preTune(){}
        void postTune(){}

        long long flops() const {
          return 0; // fix this
        }

        long long bytes() const { 
          return 0; // fix this
        }

        TuneKey tuneKey() const {
          std::stringstream vol, aux;
          vol << arg.X[0] << "x";
          vol << arg.X[1] << "x";
          vol << arg.X[2] << "x";
          vol << arg.X[3] << "x";

          aux << "threads=" << arg.length << ",prec=" << sizeof(Complex)/2;
          aux << "stride=" << arg.inA.Stride();
          return TuneKey(vol.str(), typeid(*this).name(), aux.str());
        }
    }; // StaggeredOprodField

  template<typename Complex, typename Output, typename Input>
    void computeStaggeredOprodCuda(Output outA, Output outB, Input& inA, Input& inB, cudaColorSpinorField& src, 
        FaceBuffer& faceBuffer,  const unsigned int parity, const int faceVolumeCB[4], 
        const unsigned int ghostOffset[4], const double coeff[2])
    {

      cudaEventRecord(oprodStart, streams[Nstream-1]);


      const unsigned int dim[4] = {src.X(0)*2, src.X(1), src.X(2), src.X(3)};
      // Create the arguments for the interior kernel 
      StaggeredOprodArg<Complex,Output,Input> arg(outA.volumeCB, dim, parity, 0, 0, 1, OPROD_INTERIOR_KERNEL, coeff, inA, inB, outA, outB);


      StaggeredOprodField<Complex,Output,Input> oprod(arg, QUDA_CUDA_FIELD_LOCATION);

#ifdef MULTI_GPU
      bool pack=false;
      for(int i=3; i>=0; i--){
        if(commDimPartitioned(i) && (i!=3 || kernelPackT)){
          pack = true;
          break;
        }
      } // i=3,..,0

      // source, dir(+/-1), parity, dagger, stream_ptr
      if(pack){
        faceBuffer.pack(src, -1, 1-parity, 0, streams); // packing is all done in streams[Nstream-1]
        //faceBuffer.pack(src, 1-parity, 0, streams); // packing is all done in streams[Nstream-1]
        cudaEventRecord(packEnd, streams[Nstream-1]);
      }

      for(int i=3; i>=0; i--){
        if(commDimPartitioned(i)){

          cudaEvent_t &event = (i!=3 || kernelPackT) ? packEnd : oprodStart;
          cudaStreamWaitEvent(streams[2*i], event, 0); // wait in stream 2*i for event to complete
      

          // Initialize the host transfer from the source spinor
          faceBuffer.gather(src, false, 2*i); 
          // record the end of the gathering 
          cudaEventRecord(gatherEnd[i], streams[2*i]);
        } // comDim(i)
      } // i=3,..,0
#endif
      oprod.apply(streams[Nstream-1]); 

#ifdef MULTI_GPU
      // compute gather completed 
      int gatherCompleted[5];
      int commsCompleted[5];
      int oprodCompleted[4];

      for(int i=0; i<4; ++i){
        gatherCompleted[i] = commsCompleted[i] = oprodCompleted[i] = 0;
      }
      gatherCompleted[4] = commsCompleted[4] = 1;

      // initialize commDimTotal 
      int commDimTotal = 0;
      for(int i=0; i<4; ++i){
        commDimTotal += commDimPartitioned(i);
      }
      commDimTotal *= 2;

      // initialize previousDir
      int previousDir[4];
      for(int i=3; i>=0; i--){
        if(commDimPartitioned(i)){
          int prev = 4;
          for(int j=3; j>i; j--){
            if(commDimPartitioned(j)){
              prev = j;
            }
          }
          previousDir[i] = prev;
        }
      } // set previous directions


      if(commDimTotal){
        arg.kernelType = OPROD_EXTERIOR_KERNEL;
        unsigned int completeSum=0;
        while(completeSum < commDimTotal){

          for(int i=3; i>=0; i--){
            if(!commDimPartitioned(i)) continue;

            if(!gatherCompleted[i] && gatherCompleted[previousDir[i]]){
              cudaError_t event_test = cudaEventQuery(gatherEnd[i]);

              if(event_test == cudaSuccess){
                gatherCompleted[i] = 1;
                completeSum++;
                faceBuffer.commsStart(2*i);
              }
            }

            // Query if comms has finished 
            if(!commsCompleted[i] && commsCompleted[previousDir[i]] && gatherCompleted[i]){
              int comms_test = faceBuffer.commsQuery(2*i);
              if(comms_test){
                commsCompleted[i] = 1;
                completeSum++;
                faceBuffer.scatter(src, false, 2*i);
              }
            }

            // enqueue the boundary oprod kernel as soon as the scatters have been enqueud
            if(!oprodCompleted[i] && commsCompleted[i]){
              cudaEventRecord(scatterEnd[i], streams[2*i]);
              cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[i],0);

              arg.dir = i;
              arg.ghostOffset = ghostOffset[i];
              const unsigned int volume = arg.X[0]*arg.X[1]*arg.X[2]*arg.X[3];
              arg.inB.setStride(3*volume/(2*arg.X[arg.dir]));
              // First, do the one hop term
              {

                arg.length = faceVolumeCB[i];
                arg.displacement = 1;
                dim3 blockDim(128, 1, 1);
                const int gridSize = (arg.length + (blockDim.x-1))/blockDim.x;
                dim3 gridDim(gridSize, 1, 1);               
                exteriorOprodKernel<<<gridDim, blockDim, 0, streams[Nstream-1]>>>(arg);              
              }
              // Now do the 3 hop term - Try putting this in a separate stream
              {

                arg.displacement = 3;                      
                arg.length = arg.displacement*faceVolumeCB[i];
                dim3 blockDim(128, 1, 1);
                const int gridSize = (arg.length + (blockDim.x-1))/blockDim.x;
                dim3 gridDim(gridSize, 1, 1);               
                exteriorOprodKernel<<<gridDim, blockDim, 0, streams[Nstream-1]>>>(arg);              
              } 
              arg.inB.setStride(arg.inA.Stride());

              oprodCompleted[i] = 1;
            }

          } // i=3,..,0 
        } // completeSum < commDimTotal
      } // if commDimTotal
#endif
    } // computeStaggeredOprodCuda


  // At the moment, I pass an instance of FaceBuffer in. 
  // Soon, faceBuffer will be subsumed into cudaColorSpinorField.
  void computeStaggeredOprod(cudaGaugeField& outA, cudaGaugeField& outB, cudaColorSpinorField& in,  
      FaceBuffer& faceBuffer,
      const unsigned int parity, const double coeff[2])
  {

    if(outA.Order() != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Unsupported output ordering: %d\n", outA.Order());    

    if(outB.Order() != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Unsupported output ordering: %d\n", outB.Order());    

    unsigned int ghostOffset[4] = {0,0,0,0};
#ifdef MULTI_GPU
    const unsigned int Npad = in.Ncolor()*in.Nspin()*2/in.FieldOrder();
    for(int dir=0; dir<4; ++dir){
      ghostOffset[dir] = Npad*(in.GhostOffset(dir) + in.Stride()); 
    }
#endif

    if(in.Precision() != outA.Precision()) errorQuda("Mixed precision not supported: %d %d\n", in.Precision(), outA.Precision());

    cudaColorSpinorField& inA = (parity&1) ? in.Odd() : in.Even();
    cudaColorSpinorField& inB = (parity&1) ? in.Even() : in.Odd();

    if(in.Precision() == QUDA_DOUBLE_PRECISION){

      Spinor<double2, double2, double2, 3, 0, 0> spinorA(inA);
      Spinor<double2, double2, double2, 3, 0, 0> spinorB(inB);
      computeStaggeredOprodCuda<double2>(FloatNOrder<double, 18, 2, 18>(outA), FloatNOrder<double, 18, 2, 18>(outB), 
          spinorA, spinorB, inB, faceBuffer, parity, inB.GhostFace(), ghostOffset, coeff);
    }else if(in.Precision() == QUDA_SINGLE_PRECISION){

      Spinor<float2, float2, float2, 3, 0, 0> spinorA(inA);
      Spinor<float2, float2, float2, 3, 0, 0> spinorB(inB);
      computeStaggeredOprodCuda<float2>(FloatNOrder<float, 18, 2, 18>(outA), FloatNOrder<float, 18, 2, 18>(outB), 
          spinorA, spinorB, inB, faceBuffer, parity, inB.GhostFace(), ghostOffset, coeff);
    }else{
      errorQuda("Unsupported precision: %d\n", in.Precision());
    }
    return;
  } // computeStaggeredOprod



} // namespace quda
