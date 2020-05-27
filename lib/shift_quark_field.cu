#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <quda_internal.h>

namespace quda {

  template<typename Output, typename Input>
    struct ShiftColorSpinorFieldArg {
      const unsigned int length;
      unsigned int X[4];
#ifdef MULTI_GPU
      const usigned int ghostOffset; // depends on the direction
#endif
      const unsigned int parity;
      const unsigned int dir;
      bool partitioned[4];
      const int shift;
      Input in;
      Output out;
      ShiftColorSpinorFieldArg(const unsigned int length, 
          const unsigned int X[4],
          const unsigned int ghostOffset,
          const unsigned int parity,
          const unsigned int dir,
          const int shift,   
          const Input& in,
          const Output& out) : length(length),
#ifdef MULTI_GPU
      ghostOffset(ghostOffset),
#endif
      parity(parity), dir(dir), shift(shift),  in(in), out(out) 
      {
        for(int i=0; i<4; ++i) this->X[i] = X[i];
        for(int i=0; i<4; ++i) partitioned[i] = commDimPartitioned(i) ? true : false;
      }
    };

  template<IndexType idxType, typename Int>
    __device__ __forceinline__
    int neighborIndex(const unsigned int& cb_idx, const int (&shift)[4], const bool (&partitioned)[4], const unsigned int& parity){

      int idx;
      Int x, y, z, t;

      coordsFromIndex(full_idx, x, y, z, t, cb_idx, parity);

#ifdef MULTI_GPU
      if(partitioned[0])
        if( (x+shift[0])<0 || (x+shift[0])>=X1) return -1;
      if(partitioned[1])
        if( (y+shift[1])<0 || (y+shift[1])>=X2) return -1;
      if(partitioned[2])
        if( (z+shift[2])<0 || (z+shift[2])>=X3) return -1;
      if(partitioned[3])
        if( (z+shift[3])<0 || (z+shift[3])>=X4) return -1;
#endif

      x = shift[0] ? (x + shift[0] + X1) % X1 : x;
      y = shift[1] ? (y + shift[1] + X2) % X2 : y;
      z = shift[2] ? (z + shift[2] + X3) % X3 : z;
      t = shift[3] ? (t + shift[3] + X4) % X4 : t;
      return  (((t*X3 + z)*X2 + y)*X1 + x) >> 1;
    }


  template <typename FloatN, int N, typename Output, typename Input>
    __global__ void shiftColorSpinorFieldKernel(ShiftQuarkArg<Output,Input> arg){

      int shift[4] = {0,0,0,0};
      shift[arg.dir] = arg.shift;

      unsigned int idx = blockIdx.x*(blockDim.x) + threadIdx.x;
      unsigned int gridSize = gridDim.x*blockDim.x;

      FloatN x[N];
      while(idx<arg.length){
        const int new_idx = neighborIndex(idx, shift, arg.partitioned, arg.parity);
#ifdef MULTI_GPU
        if(new_idx > 0){
#endif
          arg.in.load(x, new_idx);
          arg.out.save(x, idx);
#ifdef MULTI_GPU
        }
#endif       
        idx += gridSize;
      }  
      return;
    }

  template<typename FloatN, int N, typename Output, typename Input>
    __global__ void shiftColorSpinorFieldExternalKernel(ShiftQuarkArg<Output,Input> arg){

      unsigned int idx = blockIdx.x*(blockDim.x) + threadIdx.x;
      unsigned int gridSize = gridDim.x*blockDim.x;

      Float x[N];
      unsigned int coord[4];
      while(idx<arg.length){

        // compute the coordinates in the ghost zone 
        coordsFromIndex<1>(coord, idx, arg.X, arg.dir, arg.parity);

        unsigned int ghost_idx = arg.ghostOffset + ghostIndexFromCoords<3,3>(arg.X, coord, arg.dir, arg.shift);

        arg.in.load(x, ghost_idx);
        arg.out.save(x, idx);

        idx += gridSize;
      }


      return;
    }

  template<typename Output, typename Input> 
    class ShiftColorSpinorField : public Tunable {

      private:
        ShiftColorSpinorFieldArg<Output,Input> arg;
        const int *X; // pointer to lattice dimensions

        int sharedBytesPerThread() const { return 0; }
        int sharedBytesPerBlock(const TuneParam &) cont { return 0; }

        // don't tune the grid dimension
        bool advanceGridDim(TuneParam & param) const { return false; }

        bool advanceBlockDim(TuneParam &param) const 
        {
          const unsigned int max_threads = deviceProp.maxThreadsDim[0];
          const unsigned int max_blocks = deviceProp.maxGridSize[0];
          const unsigned int max_shared = 16384;
          const int step = deviceProp.warpSize;
          const int threads = arg.length;
          bool ret;

          param.block.x += step;
          if(param.block.x > max_threads || sharedBytesPerThread()*param.block.x > max_shared){
            param.block = dim3((threads+max_blocks-1)/max_blocks, 1, 1); // ensure the blockDim is large enough given the limit on gridDim
            param.block.x = ((param.block.x+step-1)/step)*step;
            if(param.block.x > max_threads) errorQuda("Local lattice volume is too large for device");
            ret = false;
          }else{
            ret = true;
          }
          param.grid = dim3((threads+param.block.x-1)/param.block.x,1,1);
          return ret;
        }


      public:
        ShiftColorSpinorField(const ShiftColorSpinorField<Output,Input> &arg, 
            QudaFieldLocation location)
          : arg(arg), location(location)  {}
        virtual ~ShiftColorSpinorField() {}

        void apply(const qudaStream_t &stream){
          if(location == QUDA_CUDA_FIELD_LOCATION){
            TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
            shiftColorSpinorFieldKernel<Output,Input><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#ifdef MULTI_GPU
            // Need to perform some communication and call exterior kernel, I guess
#endif
          }else{ // run the CPU code
            errorQuda("ShiftColorSpinorField is not yet implemented on the CPU\n");
          }
        } // apply

        virtual void initTuneParam(TuneParam &param) const
        {
          const unsigned int max_threads = deviceProp.maxThreadsDim[0];
          const unsigned int max_blocks = deviceProp.maxGridSize[0];
          const int threads = arg.length;
          const int step = deviceProp.warpSize;
          param.block = dim3((threads+max_blocks-1)/max_blocks, 1, 1); // ensure the blockDim is large enough, given the limit on gridDim
          param.block.x = ((param.block.x+step-1) / step) * step; // round up to the nearest "step"
          if (param.block.x > max_threads) errorQuda("Local lattice volume is too large for device");
          param.grid = dim3((threads+param.block.x-1)/param.block.x, 1, 1);
          param.shared_bytes = sharedBytesPerThread()*param.block.x > sharedBytesPerBlock(param) ?
            sharedBytesPerThread()*param.block.x : sharedBytesPerBlock(param);
        }

        /** sets default values for when tuning is disabled */
        void defaultTuneParam(TuneParam &param) const {
          initTuneParam(param);
        }

        long long flops() const { return 0; } // fixme
        long long bytes() const { return 0; } // fixme

        TuneKey tuneKey() const {
          std::stringstream vol, aux;
          vol << X[0] << "x";
          vol << X[1] << "x";
          vol << X[2] << "x";
          vol << X[3] << "x";
          aux << "threads=" << 2*arg.in.volumeCB << ",prec=" << sizeof(Complex)/2;
          aux << "stride=" << arg.in.stride;
          return TuneKey(vol.str(), typeid(*this).name(), aux.str());
        }
    };


  // Should really have a parity
  void shiftColorSpinorField(cudaColorSpinorField &dst, const cudaColorSpinorField &src, const unsigned int parity, const unsigned int dim, const int shift) {

    if(&src == &dst){
      errorQuda("destination field is the same as source field\n");
      return;
    }

    if(src.Nspin() != 1 && src.Nspin() !=4) errorQuda("nSpin(%d) not supported\n", src.Nspin());

    if(src.SiteSubset() != dst.SiteSubset())
      errorQuda("Spinor fields do not have matching subsets\n");

    if(src.SiteSubset() == QUDA_FULL_SITE_SUBSET){
      if(shift&1){
        shiftColorSpinorField(dst.Even(), src.Odd(), 0, dim, shift);
        shiftColorSpinorField(dst.Odd(), src.Even(), 1, dim, shift);
      }else{
        shiftColorSpinorField(dst.Even(), src.Even(), 0, dim, shift);
        shiftColorSpinorField(dst.Odd(), src.Odd(), 1, dim, shift);
      }
      return;
    }

#ifdef MULTI_GPU
    const int dir = (shift>0) ? QUDA_BACKWARDS : QUDA_FORWARDS; // pack the start of the field if shift is positive
    const int offset = (shift>0) ? 0 : 1;
#endif


    if(dst.Precision() == QUDA_DOUBLE_PRECISION && src.Precision() == QUDA_DOUBLE_PRECISION){
      if(src.Nspin() == 1){
        Spinor<double2, double2, double2, 3, 0, 0> src_tex(src);
        Spinor<double2, double2, double2, 3, 1> dst_spinor(dst);
        ShiftColorSpinorFieldArg arg(src.Volume(), parity, dim, shift, dst_spinor, src_tex);
        ShiftColorSpinorField shiftColorSpinor(arg, QUDA_CPU_FIELD_LOCATION);

#ifdef MULTI_GPU
        if(commDimPartitioned(dim) && dim!=3){
          face->pack(src, 1-parity, dagger, dim, dir, streams); // pack in stream[1]
          qudaEventRecord(packEnd, streams[1]);
          qudaStreamWaitEvent(streams[1], packEnd, 0); // wait for pack to end in stream[1]
          face->gather(src, dagger, 2*dim+offset, 1); // copy packed data from device buffer to host and do this in stream[1] 
          qudaEventRecord(gatherEnd, streams[1]); // record the completion of face->gather
        }
#endif

        shiftColorSpinor.apply(0); // shift the field in the interior region

#ifdef MULTI_GPU
        if(commDimPartitioned(dim) && dim!=3){
          while(1){
            cudaError_t eventQuery = cudaEventQuery(gatherEnd);
            if(eventQuery == cudaSuccess){
              face->commsStart(2*dim + offset); // if argument is even, send backwards, else send forwards
              break;
            }
          }

          // after communication, load data back on to device
          // do this in stream[1]
          while(1){
            if(face->commsQuery(2*dim + offset)){
              face->scatter(src, dagger, 2*dim+offset, 1);
              break;
            }
          } // while(1) 
          qudaEventRecord(scatterEnd, streams[1]);
          qudaStreamWaitEvent(streams[1], scatterEnd, 0);
          shiftColorSpinor.apply(1);
        }
#endif

      }else{
        errorQuda("Only staggered fermions are currently supported\n");
      }
    }else if(dst.Precision() == QUDA_SINGLE_PRECISION && src.Precision() == QUDA_SINGLE_PRECISION){
      if(src.Nspin() == 1 ){
        Spinor<float2, float2, float2, 3, 0, 0> src_tex(src);
        Spinor<float2, float2, float2, 3, 1> dst_spinor(dst);
        ShiftColorSpinorFieldArg arg(src.Volume(), parity, dim, shift, dst_spinor, src_tex);
        ShiftColorSpinorField shiftColorSpinor(arg, QUDA_CPU_FIELD_LOCATION);
      }else{
        errorQuda("Only staggered fermions are currently supported\n");
      }
    }
    return;
  }


} // namespace quda

