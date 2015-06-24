#include <cstdio>
#include <cstdlib>

#include <tune_quda.h>
#include <quda_internal.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <color_spinor.h>

namespace quda {

#ifdef GPU_CLOVER_DIRAC

  namespace { // anonymous
#include <texture.h>
  }
  
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


  void createCloverForceEvents(){
#ifdef MULTI_GPU
    cudaEventCreate(&packEnd, cudaEventDisableTiming);
    createEventArray(gatherEnd, cudaEventDisableTiming);
    createEventArray(scatterEnd, cudaEventDisableTiming);
#endif
    cudaEventCreate(&oprodStart, cudaEventDisableTiming);
    cudaEventCreate(&oprodEnd, cudaEventDisableTiming);
    return;
  }

  void destroyCloverForceEvents(){
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

  template<typename Complex, typename Output, typename Gauge, typename InputA, typename InputB>
    struct CloverForceArg {
      unsigned int length;
      int X[4];
      unsigned int parity;
      unsigned int dir;
      unsigned int ghostOffset[4];
      unsigned int displacement;
      KernelType kernelType;
      bool partitioned[4];
      InputA inA;
      InputB inB;
      Gauge  gauge;
      Output force;
      typename RealTypeId<Complex>::Type coeff;
      
    CloverForceArg(const unsigned int parity,
		   const unsigned int dir,
		   const unsigned int *ghostOffset,
		   const unsigned int displacement,   
		   const KernelType kernelType, 
		   const double coeff,
		   InputA& inA,
		   InputB& inB,
		   Gauge& gauge,
		   Output& force,
		   GaugeField &meta) : length(meta.VolumeCB()), parity(parity),
				       displacement(displacement), kernelType(kernelType), 
				       coeff(coeff), inA(inA), inB(inB), gauge(gauge), force(force)
      {
        for(int i=0; i<4; ++i) this->X[i] = meta.X()[i];
        for(int i=0; i<4; ++i) this->ghostOffset[i] = ghostOffset[i];
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
        const unsigned int cb_idx, const unsigned int parity, const int X[4])
    {
      const int &LX = X[0];
      const int &LY = X[1];
      const int &LZ = X[2];
      const int XYZ = X[2]*X[1]*X[0];
      const int XY = X[1]*X[0];

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
    __device__ void coordsFromIndex(int x[4], const unsigned int cb_idx, const int X[4], const unsigned int dir, const int displacement, const unsigned int parity)
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
    __device__  int ghostIndexFromCoords(const int x[4], const int X[4], const unsigned int dir, const int shift){
      return 0;
    }



  template<>
    __device__  int ghostIndexFromCoords<1,3>(
        const int x[4],
        const int X[4], 
        unsigned int dir, 
        const int shift)
    {
      int ghost_idx;
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
		    const int X[4]){
    
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



  template<typename Complex, typename Output, typename Gauge, typename InputA, typename InputB>
  __global__ void interiorOprodKernel(CloverForceArg<Complex, Output, Gauge, InputA, InputB> arg)
  {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int gridSize = gridDim.x*blockDim.x;
    
    typedef typename RealTypeId<Complex>::Type real;

    ColorSpinor<real,3,4> A;
    ColorSpinor<real,3,4> A_shift;
    ColorSpinor<real,3,4> B;
    ColorSpinor<real,3,4> B_shift;
    
    Matrix<Complex,3> result;
    Matrix<Complex,3> U;
    Matrix<Complex,3> temp;

    while(idx<arg.length){
      arg.inA.load(static_cast<Complex*>(A.data), idx);
      arg.inB.load(static_cast<Complex*>(B.data), idx);
      for(int dir=0; dir<4; ++dir){
	int shift[4] = {0,0,0,0};
	shift[dir] = 1;
	const int nbr_idx = neighborIndex(idx, shift, arg.partitioned, arg.parity, arg.X);
	if(nbr_idx >= 0){
	  arg.inB.load(static_cast<Complex*>(B_shift.data), nbr_idx); // need to do reconstruct
	  result = outerProdSpinTrace(B_shift,A);

	  arg.inA.load(static_cast<Complex*>(A_shift.data), nbr_idx);
	  result += outerProdSpinTrace(A_shift,B);

	  arg.force.load(reinterpret_cast<real*>(temp.data), idx, dir, arg.parity); 
	  result = temp + result*arg.coeff;
	  arg.gauge.load(reinterpret_cast<real*>(U.data), idx, dir, arg.parity); 
	  temp = U * result;
	  arg.force.save(reinterpret_cast<real*>(temp.data), idx, dir, arg.parity); 
	}


      } // dir
      idx += gridSize;
    }
    return;
  } // interiorOprodKernel
  
#ifdef MULTI_GPU
  template<typename Complex, typename Output, typename Gauge, typename InputA, typename InputB> 
  __global__ void exteriorOprodKernel(CloverForceArg<Complex, Output, Gauge, InputA, InputB> arg)
    {
      int cb_idx = blockIdx.x*blockDim.x + threadIdx.x;
      const int gridSize = gridDim.x*blockDim.x;

      typedef typename RealTypeId<Complex>::Type real;
      Complex A[12];
      Complex A_shift[12];
      Complex B[12];
      Complex B_shift[12];
      Matrix<Complex,3> result;
      Matrix<Complex,3> temp;
      Matrix<Complex,3> U;

      int x[4];
      while(cb_idx<arg.length){
        coordsFromIndex<1>(x, cb_idx, arg.X, arg.dir, arg.displacement, arg.parity); 
        const unsigned int bulk_cb_idx = ((((x[3]*arg.X[2] + x[2])*arg.X[1] + x[1])*arg.X[0] + x[0]) >> 1);
        arg.inA.load(A, bulk_cb_idx);
        arg.inB.load(B, bulk_cb_idx);

        const unsigned int ghost_idx = arg.ghostOffset[arg.dir] + ghostIndexFromCoords<1,3>(x, arg.X, arg.dir, arg.displacement);
        arg.inB.loadGhost(B_shift, ghost_idx, arg.dir);
        outerProd(B_shift,A,&temp);

        arg.inA.loadGhost(A_shift, ghost_idx, arg.dir);
        outerProd(A_shift,B,&result);
	result += temp;

        arg.force.load(reinterpret_cast<real*>(temp.data), bulk_cb_idx, arg.dir, arg.parity); 
        result = temp + result*arg.coeff; 
	arg.gauge.load(reinterpret_cast<real*>(U.data), bulk_cb_idx, arg.dir, arg.parity); 
	temp = U * result;
        arg.force.save(reinterpret_cast<real*>(temp.data), bulk_cb_idx, arg.dir, arg.parity); 

        cb_idx += gridSize;
      }
      return;
    }
#endif // MULTI_GPU
  
  template<typename Complex, typename Output, typename Gauge, typename InputA, typename InputB> 
  class CloverForce : public Tunable {
    
  private:
    CloverForceArg<Complex,Output,Gauge,InputA,InputB> &arg;
    const GaugeField &meta;
    QudaFieldLocation location; // location of the lattice fields
    
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }
    
    unsigned int minThreads() const { return arg.length; }
    bool tunedGridDim() const { return false; }
    
  public:
    CloverForce(CloverForceArg<Complex,Output,Gauge,InputA,InputB> &arg,
		const GaugeField &meta, QudaFieldLocation location)
      : arg(arg), meta(meta), location(location) {
      writeAuxString("threads=%d,prec=%lu,stride=%d",arg.length,sizeof(Complex)/2,arg.inA.Stride());
      // this sets the communications pattern for the packing kernel
      int comms[QUDA_MAX_DIM] = { commDimPartitioned(0), commDimPartitioned(1), commDimPartitioned(2), commDimPartitioned(3) };
      setPackComms(comms);
    } 
    
    virtual ~CloverForce() {}
    
    void apply(const cudaStream_t &stream){
      if(location == QUDA_CUDA_FIELD_LOCATION){
	// Disable tuning for the time being
	TuneParam tp = tuneLaunch(*this, QUDA_TUNE_NO, getVerbosity());
	if(arg.kernelType == OPROD_INTERIOR_KERNEL){
	  interiorOprodKernel<<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	} else if(arg.kernelType == OPROD_EXTERIOR_KERNEL) {
#ifdef MULTI_GPU
	  exteriorOprodKernel<<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
#endif
	} else {
	  errorQuda("Kernel type not supported\n");
	}
      }else{ // run the CPU code
	errorQuda("No CPU support for staggered outer-product calculation\n");
      }
    } // apply
    
    void preTune(){
      this->arg.force.save();
    }
    void postTune(){
      this->arg.force.load();
    }
  
    long long flops() const { return 0; }
    long long bytes() const { return 0; }
  
    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux);}
  }; // CloverForce
  
  template<typename Complex, typename Output, typename Gauge, typename InputA, typename InputB>
  void computeCloverForceCuda(Output force, Gauge gauge, cudaGaugeField& out, InputA& inA, InputB& inB, 
			      cudaColorSpinorField& src1, cudaColorSpinorField& src2,
			      const unsigned int parity, const int faceVolumeCB[4], 
			      const unsigned int ghostOffset[4], const double coeff)
    {

      int dag = 1;

      cudaEventRecord(oprodStart, streams[Nstream-1]);

      // Create the arguments for the interior kernel 
      CloverForceArg<Complex,Output,Gauge,InputA,InputB> arg(parity, 0, ghostOffset, 1, OPROD_INTERIOR_KERNEL, coeff, inA, inB, gauge, force, out);
      CloverForce<Complex,Output,Gauge,InputA,InputB> oprod(arg, out, QUDA_CUDA_FIELD_LOCATION);

#ifdef MULTI_GPU
      bool pack=false;
      for(int i=3; i>=0; i--){
        if(commDimPartitioned(i) && (i!=3 || getKernelPackT())){
          pack = true;
          break;
        }
      } // i=3,..,0

      // source, dir(+/-1), parity, dagger, stream_ptr
      // packing is all done in streams[Nstream-1]
      // always call pack since this also sets the stream pointer even if not packing
      src1.pack(1, 1-parity, dag, Nstream-1);
      src2.pack(1, 1-parity, !dag, Nstream-1);
      if(pack){
        cudaEventRecord(packEnd, streams[Nstream-1]);
      }

      for(int i=3; i>=0; i--){
        if(commDimPartitioned(i)){

          cudaEvent_t &event = (i!=3 || getKernelPackT()) ? packEnd : oprodStart;
          cudaStreamWaitEvent(streams[2*i], event, 0); // wait in stream 2*i for event to complete

          // Initialize the host transfer from the source spinor
          src1.gather(1, dag, 2*i); 
          src2.gather(1, !dag, 2*i); 
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
        unsigned int completeSum=0;
        while(completeSum < commDimTotal){

          for(int i=3; i>=0; i--){
            if(!commDimPartitioned(i)) continue;

            if(!gatherCompleted[i] && gatherCompleted[previousDir[i]]){
              cudaError_t event_test = cudaEventQuery(gatherEnd[i]);

              if(event_test == cudaSuccess){
                gatherCompleted[i] = 1;
                completeSum++;
                src1.commsStart(1, 2*i, dag);
                src2.commsStart(1, 2*i, !dag);
              }
            }

            // Query if comms has finished 
            if(!commsCompleted[i] && commsCompleted[previousDir[i]] && gatherCompleted[i]){
              int comms_test = (src1.commsQuery(2*i, dag) && src2.commsQuery(2*i, !dag));
              if(comms_test){
                commsCompleted[i] = 1;
                completeSum++;
                src1.scatter(1, dag, 2*i);
                src2.scatter(1, !dag, 2*i);
              }
            }

            // enqueue the boundary oprod kernel as soon as the scatters have been enqueud
            if(!oprodCompleted[i] && commsCompleted[i]){
              cudaEventRecord(scatterEnd[i], streams[2*i]);
              cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[i],0);

	      // update parameters for this exterior kernel
	      arg.kernelType = OPROD_EXTERIOR_KERNEL;
              arg.dir = i;
	      arg.length = faceVolumeCB[i];
	      arg.displacement = 1;

	      oprod.apply(streams[Nstream-1]); 

              oprodCompleted[i] = 1;
            }

          } // i=3,..,0 
        } // completeSum < commDimTotal
      } // if commDimTotal
#endif
    } // computeCloverForceCuda

#endif // GPU_CLOVER_FORCE

  void computeCloverForce(cudaGaugeField& force,
			  const cudaGaugeField& U,
			  cudaColorSpinorField& x,  
			  cudaColorSpinorField& p,
			  const unsigned int parity, const double coeff)
  {

#ifdef GPU_CLOVER_DIRAC

    if(force.Order() != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Unsupported output ordering: %d\n", force.Order());    

    unsigned int ghostOffset[4] = {0,0,0,0};
#ifdef MULTI_GPU
    const unsigned int Npad = x.Ncolor()*x.Nspin()*2/x.FieldOrder();
    for(int dir=0; dir<4; ++dir){
      ghostOffset[dir] = Npad*(x.GhostOffset(dir) + x.Stride()); 
    }
#endif

    if(x.Precision() != force.Precision()) errorQuda("Mixed precision not supported: %d %d\n", x.Precision(), force.Precision());

    cudaColorSpinorField& inA = (parity&1) ? x.Odd() : x.Even();
    cudaColorSpinorField& inB = (parity&1) ? p.Even(): p.Odd();

    if(x.Precision() == QUDA_DOUBLE_PRECISION){
      Spinor<double2, double2, double2, 12, 0, 0> spinorA(inA);
      Spinor<double2, double2, double2, 12, 0, 1> spinorB(inB);
      if (U.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	computeCloverForceCuda<double2>(FloatNOrder<double, 18, 2, 18>(force), 
					FloatNOrder<double,18, 2, 18>(U),
					force, spinorA, spinorB, inA, inB, parity, 
					inB.GhostFace(), ghostOffset, coeff);
      } else if (U.Reconstruct() == QUDA_RECONSTRUCT_12) {
	computeCloverForceCuda<double2>(FloatNOrder<double, 18, 2, 18>(force), 
					FloatNOrder<double,18, 2, 12>(U),
					force, spinorA, spinorB, inA, inB, parity, 
					inB.GhostFace(), ghostOffset, coeff);
      } else {
	errorQuda("Unsupported recontruction type");
      }
    }else if(x.Precision() == QUDA_SINGLE_PRECISION){
#if 0
      Spinor<float4, float4, float4, 6, 0, 0> spinorA(inA);
      Spinor<float4, float4, float4, 6, 0, 1> spinorB(inB);
      if (U.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	computeCloverForceCuda<float2>(FloatNOrder<float, 18, 2, 18>(force), 
				       FloatNOrder<float, 18, 2, 18>(U), 
				       force, spinorA, spinorB, inA, inB, parity, 
				       inB.GhostFace(), ghostOffset, coeff);
      } else if (U.Reconstruct() == QUDA_RECONSTRUCT_12) {
	computeCloverForceCuda<float2>(FloatNOrder<float, 18, 2, 18>(force), 
				       FloatNOrder<float, 18, 4, 12>(U), 
				       force, spinorA, spinorB, inA, inB, parity, 
				       inB.GhostFace(), ghostOffset, coeff);
      }
#endif
    } else {
      errorQuda("Unsupported precision: %d\n", x.Precision());
    }

#else // GPU_CLOVER_DIRAC not defined
   errorQuda("Clover Dirac operator has not been built!"); 
#endif

    return;
  } // computeCloverForce



} // namespace quda
