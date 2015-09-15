#ifdef GPU_CONTRACT

#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>

#include <color_spinor_field.h>
#include <clover_field.h>

// these control the Wilson-type actions

#include <quda_internal.h>
#include <dslash_quda.h>
#include <sys/time.h>
#include <blas_quda.h>
#include <face_quda.h>

#include <inline_ptx.h>


namespace quda
{
  namespace covdev
  {
    #include <dslash_constants.h>
    #include <dslash_textures.h>
    #include <dslash_index.cuh>

    // Enable shared memory dslash for Fermi architecture
    //#define SHARED_WILSON_DSLASH
    //#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

    #include <dslash_quda.cuh>

    #include	"covDev.h"	//Covariant derivative definitions
  } // end namespace covDev

  // declare the dslash events
  #include <dslash_events.cuh>

  using namespace covdev;

  /**
     This macros try to mimic dslash definitions, because of the similarities between the covariant derivative and the dslash
  */

  #define MORE_GENERIC_COVDEV(FUNC, dir, DAG, kernel_type, gridDim, blockDim, shared, stream, param,  ...) \
    if		(reconstruct == QUDA_RECONSTRUCT_NO) {			\
      switch	(dir) {							\
      case 0:								\
        FUNC ## 018 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
        break;								\
      case 1:								\
        FUNC ## 118 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
        break;								\
      case 2:								\
        FUNC ## 218 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
        break;								\
      case 3:								\
        FUNC ## 318 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
        break;								\
      }									\
    } else if	(reconstruct == QUDA_RECONSTRUCT_12) {			\
      switch	(dir) {							\
      case 0:								\
        FUNC ## 012 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
        break;								\
      case 1:								\
        FUNC ## 112 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
        break;								\
      case 2:								\
        FUNC ## 212 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
        break;								\
      case 3:								\
        FUNC ## 312 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
        break;								\
      }									\
    } else if	(reconstruct == QUDA_RECONSTRUCT_8) {			\
      switch	(dir) {							\
      case 0:								\
        FUNC ## 08 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
        break;								\
      case 1:								\
        FUNC ## 18 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
        break;								\
      case 2:								\
        FUNC ## 28 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
        break;								\
      case 3:								\
        FUNC ## 38 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
        break;								\
      }									\
    }

  #define GENERIC_COVDEV(FUNC, dir, DAG, gridDim, blockDim, shared, stream, param,  ...) \
    switch(param.kernel_type) {						\
      case INTERIOR_KERNEL:							\
        MORE_GENERIC_COVDEV(FUNC, dir, DAG, INTERIOR_KERNEL,   gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
        break;									\
      case EXTERIOR_KERNEL_X:						\
        MORE_GENERIC_COVDEV(FUNC, dir, DAG, EXTERIOR_KERNEL_X, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
        break;									\
      case EXTERIOR_KERNEL_Y:						\
        MORE_GENERIC_COVDEV(FUNC, dir, DAG, EXTERIOR_KERNEL_Y, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
        break;									\
      case EXTERIOR_KERNEL_Z:						\
        MORE_GENERIC_COVDEV(FUNC, dir, DAG, EXTERIOR_KERNEL_Z, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
        break;									\
      case EXTERIOR_KERNEL_T:						\
        MORE_GENERIC_COVDEV(FUNC, dir, DAG, EXTERIOR_KERNEL_T, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
        break;									\
    }

  #define COVDEV(FUNC, mu, gridDim, blockDim, shared, stream, param, ...)	\
    if (mu < 4) {								\
      GENERIC_COVDEV(FUNC, mu, , gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    } else {								\
      int nMu = mu - 4;							\
      GENERIC_COVDEV(FUNC, nMu, Dagger, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    }

  #define PROFILE(f, profile, idx)		\
    profile.TPSTART(idx);			\
    f;						\
    profile.TPSTOP(idx); 

  /**
     This is a simpler version of the dslashCuda function to call the right kernels
  */

  void covDevCuda(DslashCuda &dslash, const size_t regSize, const int mu, TimeProfile &profile)
  {
    const int	dir = mu%4;

    dslashParam.kernel_type = INTERIOR_KERNEL;

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

    checkCudaError();

    #ifdef MULTI_GPU
      if(comm_dim(dir) > 1)
      {
        dslashParam.kernel_type = static_cast<KernelType>(dir);
        dslashParam.ghostDim[dir] = commDimPartitioned(dir);     // determines whether to use regular or ghost indexing at boundary
        dslashParam.commDim[dir] = commDimPartitioned(dir);      // switch off comms if override = 0
			
        PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

        checkCudaError();

        dslashParam.ghostDim[dir] = 0;                           // not sure whether neccessary 
        dslashParam.commDim[dir] = 0; 
      }
    #endif // MULTI_GPU
    cudaStreamSynchronize(streams[Nstream-1]);
  }

  /**
     Class for covariant derivative, is based on SharedDslashCuda
  */

  template <typename Float, typename Float2>
  class CovDevCuda : public SharedDslashCuda
  {
    private:
      const cudaGaugeField *gauge;
      const int mu;                     // Direction to apply the covariant derivative. Anything beyond 3 is understood as dagger and moving backwards (from x to x-(mu%4))
      const int dir;                    // This is the real direction xyzt, that is, mu%4
      const int parity;                 // The current implementation works on parity spinors

      void *gauge0, *gauge1;

      bool binded;
			
      #ifdef MULTI_GPU
        Float *ghostBuffer;             // For the ghosts, I did my own implementation, jsut because the number of functions to overload was absurd
        int ghostVolume;
        int ghostBytes;
        int offset;
        int Npad;
        int Nvec;
        int Nint;
      #endif

      #ifdef USE_TEXTURE_OBJECTS
        cudaTextureObject_t tex;
      #endif

    protected:
      unsigned int minThreads() const
      {
        #ifdef MULTI_GPU
          if(dslashParam.kernel_type == INTERIOR_KERNEL) { return in->Volume(); } else { return ghostVolume; }
        #else
          return in->Volume();
        #endif
      }

      unsigned int sharedBytesPerThread() const { return 0; }

      dim3 createGrid(const dim3 &block) const	         // Tuning is supported in a one-dimensional grid, because other grids gave me wrong results
      {
        unsigned int gx = (dslashParam.threads + block.x - 1) / block.x;
        unsigned int gy = 1;
        unsigned int gz = 1;

        return dim3(gx, gy, gz);
      }

      /**
         Advance 1-d block size, accounting for the differences of the covariant derivative (I could not make the 3-d block work reliably)
      */

      bool advanceBlockDim(TuneParam &param) const
      {
        //if(dslashParam.kernel_type != INTERIOR_KERNEL) return DslashCuda::advanceBlockDim(param);
        const unsigned int min_threads = 2;
        const unsigned int max_threads = 512;            // FIXME: use deviceProp.maxThreadsDim[0];
    
        param.block.x += 2;
        param.block.y = 1;  
        param.block.z = 1;  
        param.grid = createGrid(param.block);

        if((param.block.x > min_threads) && (param.block.x < max_threads))
          return true;
        else
          return false;
      }

      #ifdef MULTI_GPU

        /**
           Allocates ghosts for multi-GPU
        */
        void allocateGhosts()
        {
          ghostBuffer = (Float*)device_malloc(ghostBytes);
        }

        /**
          Sends the ghosts, I only checked it for two gpus and partitioning in one dimension. The current implementation works for zt-splitting, but
          for z-splitting is not very efficient. I tried to make something that worked for xy-splitting as well, but it didn't and I never understood
          why. You're welcome to improve this piece of code.
        */

        void exchangeGhosts()
        {
          const int rel = (mu < 4) ? 1 : -1;

          // Send buffers:
          void *send = pinned_malloc(ghostBytes);

          // Receive buffers:
          void *recv = pinned_malloc(ghostBytes);

          switch(mu) {
            default:
              break;

            case 0: {                                              // x from point p to p + x
              void *sendFacePtr = (char*) in->V();
              size_t len = Nvec*sizeof(Float);
              size_t skip = len*in->X(0);
              size_t dpitch = ghostVolume*Nvec*sizeof(Float);
              size_t spitch = in->Stride()*Nvec*sizeof(Float);

              for(int t=0;t<ghostVolume;t++) {                     // I know this is a crime...
                cudaMemcpy2DAsync((void*) (((char*)send)+len*t), dpitch, (void*) (((char*)sendFacePtr)+skip*t),
                                  spitch, len, Npad, cudaMemcpyDeviceToHost, streams[0]);
                cudaStreamSynchronize(streams[0]);
              }
            }
            break;

            case 1: {                                             // y from point p to p + y
              void *sendFacePtr = (char*)in->V();
              size_t len = in->X(0)*Nvec*sizeof(Float);
              size_t skip = len*in->X(1);
              size_t dpitch = ghostVolume*Nvec*sizeof(Float);
              size_t spitch = in->Stride()*Nvec*sizeof(Float);

              for(int tz=0;tz<(in->X(2)*in->X(3));tz++) {         // Although is terribly inefficient, it should work. The problem is that it doesn't
                cudaMemcpy2DAsync((void*) (((char*)send)+len*tz), dpitch, (void*) (((char*)sendFacePtr)+skip*tz),
                                  spitch, len, Npad, cudaMemcpyDeviceToHost, streams[0]);
                cudaStreamSynchronize(streams[0]);
              }
            }
            break;

            case 2: {                                             // z from point p to p + z
              void *sendFacePtr = (char*) in->V();
              size_t len = ghostVolume*Nvec*sizeof(Float)/in->X(3);
              size_t skip = len*in->X(2);
              size_t dpitch = ghostVolume*Nvec*sizeof(Float);
              size_t spitch = in->Stride()*Nvec*sizeof(Float);

              for(int t=0;t<in->X(3);t++) {                       // I'm sure this can be improved
                cudaMemcpy2DAsync((void*) (((char*)send)+len*t), dpitch, (void*) (((char*)sendFacePtr)+skip*t),
                                  spitch, len, Npad, cudaMemcpyDeviceToHost, streams[0]);
                cudaStreamSynchronize(streams[0]);
              }
            }
            break;

            case 3: {                                               // t from point p to p + t
              void *sendFacePtr = (char*)in->V();
              size_t len = ghostVolume*Nvec*sizeof(Float);
              size_t spitch = in->Stride()*Nvec*sizeof(Float);
              cudaMemcpy2DAsync(send, len, sendFacePtr, spitch, len, Npad, cudaMemcpyDeviceToHost, streams[0]);
              cudaStreamSynchronize(streams[0]);
            }
            break;

            // Dagger versions (mu >= 4) follow

            case 4: {                                              // x dagger from point p to p - x
              void *sendFacePtr = (char*) in->V() + offset*Nvec*sizeof(Float);
              size_t len = Nvec*sizeof(Float);
              size_t skip = len*in->X(0);
              size_t dpitch = ghostVolume*Nvec*sizeof(Float);
              size_t spitch = in->Stride()*Nvec*sizeof(Float);

              for(int t=0;t<ghostVolume;t++) {
  	      cudaMemcpy2DAsync((void*) (((char*)send)+len*t), dpitch, (void*) (((char*)sendFacePtr)+skip*t),
                                  spitch, len, Npad, cudaMemcpyDeviceToHost, streams[0]);
                cudaStreamSynchronize(streams[0]);
              }
            }
            break;

            case 5: {                                              // y dagger from point p to p - y
              void *sendFacePtr = ((char*) in->V()) + offset*Nvec*sizeof(Float);
              size_t len = in->X(0)*Nvec*sizeof(Float);
              size_t skip = len*in->X(1);
              size_t dpitch = ghostVolume*Nvec*sizeof(Float);
              size_t spitch = in->Stride()*Nvec*sizeof(Float);

              for(int tz=0;tz<(in->X(2)*in->X(3));tz++) {
                cudaMemcpy2DAsync((void*) (((char*)send)+len*tz), dpitch, (void*) (((char*)sendFacePtr)+skip*tz),
                                  spitch, len, Npad, cudaMemcpyDeviceToHost, streams[0]);
                cudaStreamSynchronize(streams[0]);
              }
            }
            break;

            case 6: {                                              // z dagger from point p to p - z
              void *sendFacePtr = (((char*)in->V()) + offset*Nvec*sizeof(Float));
              size_t len = ghostVolume*Nvec*sizeof(Float)/in->X(3);
              size_t skip = len*in->X(2);
              size_t dpitch = ghostVolume*Nvec*sizeof(Float);
              size_t spitch = in->Stride()*Nvec*sizeof(Float);

              for(int t=0;t<in->X(3);t++) {
                cudaMemcpy2DAsync((void*) (((char*)send)+len*t), dpitch, (void*) (((char*)sendFacePtr)+skip*t),
      	                        spitch, len, Npad, cudaMemcpyDeviceToHost, streams[0]);
                cudaStreamSynchronize(streams[0]);
              }
            }
            break;

            case 7: {                                              // t dagger from point p to p - t
              void *sendFacePtr = (char*)in->V() + offset*Nvec*sizeof(Float);
              size_t len = ghostVolume*Nvec*sizeof(Float);
              size_t spitch = in->Stride()*Nvec*sizeof(Float);

              cudaMemcpy2DAsync(send, len, sendFacePtr, spitch, len, Npad, cudaMemcpyDeviceToHost, streams[0]);
              cudaStreamSynchronize(streams[0]);
            }
            break;
          }

          // Send buffers to neighbors:

          MsgHandle *mh_send;
          MsgHandle *mh_from;

          mh_send = comm_declare_send_relative(send, dir, rel*(-1), ghostBytes);
          mh_from = comm_declare_receive_relative(recv, dir, rel, ghostBytes);
          comm_start (mh_send);
          comm_start (mh_from);
          comm_wait (mh_send);
          comm_wait (mh_from);
          comm_free (mh_send);
          comm_free (mh_from);

          // Send buffers to GPU:
          cudaMemcpy(ghostBuffer, recv, ghostBytes, cudaMemcpyHostToDevice);
          cudaDeviceSynchronize();

          host_free(send);
          host_free(recv);
        }


        void freeGhosts() { device_free(ghostBuffer); }

        void bindGhosts()
        {
          if(binded == false) {       // bind only once
            #ifdef USE_TEXTURE_OBJECTS
              cudaChannelFormatDesc desc;
              memset(&desc, 0, sizeof(cudaChannelFormatDesc));
              if (in->Precision() == QUDA_SINGLE_PRECISION) desc.f = cudaChannelFormatKindFloat;
              else desc.f = cudaChannelFormatKindSigned; // half is short, double is int2

              // staggered fields in half and single are always two component
              if (in->Nspin() == 1 && (in->Precision() == QUDA_SINGLE_PRECISION)) {
                desc.x = 8*in->Precision();
                desc.y = 8*in->Precision();
                desc.z = 0;
                desc.w = 0;
              } else { // all others are four component
                desc.x = (in->Precision() == QUDA_DOUBLE_PRECISION) ? 32 : 8*in->Precision();
                desc.y = (in->Precision() == QUDA_DOUBLE_PRECISION) ? 32 : 8*in->Precision();
                desc.z = (in->Precision() == QUDA_DOUBLE_PRECISION) ? 32 : 8*in->Precision();
                desc.w = (in->Precision() == QUDA_DOUBLE_PRECISION) ? 32 : 8*in->Precision();
              }

              cudaResourceDesc resDesc;
              memset(&resDesc, 0, sizeof(resDesc));
              resDesc.resType = cudaResourceTypeLinear;
              resDesc.res.linear.devPtr = ghostBuffer;
              resDesc.res.linear.desc = desc;
              resDesc.res.linear.sizeInBytes = Nint * ghostVolume * sizeof(Float);

              cudaTextureDesc texDesc;
              memset(&texDesc, 0, sizeof(texDesc));
              texDesc.readMode = cudaReadModeElementType;

              cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

              dslashParam.inTex = tex;
            #else
              if(in->Precision() == QUDA_DOUBLE_PRECISION)
                cudaBindTexture(0, spinorTexDouble, (Float2*)ghostBuffer, ghostBytes);
              else if(in->Precision() == QUDA_SINGLE_PRECISION)
                cudaBindTexture(0, spinorTexSingle, (Float2*)ghostBuffer, ghostBytes);
              else
                errorQuda("Half precision for covariant derivative not supported.");
            #endif
            checkCudaError();
            binded = true;
          }
        }

        void unbindGhosts() {
          if(binded == true) {
            #ifdef USE_TEXTURE_OBJECTS
              cudaDestroyTextureObject(tex);
            #else
              if(in->Precision() == QUDA_DOUBLE_PRECISION)
                cudaUnbindTexture(spinorTexDouble);
              else
                cudaUnbindTexture(spinorTexSingle);
            #endif
            checkCudaError();
            binded = false;
          }
        }
      #endif               /* MULTI_GPU */

      void unbindGauge() {
        unbindGaugeTex(*gauge);
        checkCudaError();
      }

	
    public:
      CovDevCuda(cudaColorSpinorField *out, const cudaGaugeField *gauge, const cudaColorSpinorField *in, const int parity, const int mu)
      : SharedDslashCuda(out, in, 0, gauge->Reconstruct(), mu<4 ? 0 : 1), gauge(gauge), parity(parity), mu(mu), dir(mu%4), binded(false)
      { 
        bindSpinorTex<Float2>(in, out); 
        bindGaugeTex(*gauge, parity, &gauge0, &gauge1);

        #ifdef MULTI_GPU
          if(comm_dim(dir) > 1) {
            Nvec = sizeof(Float2)/sizeof(Float);
            Nint = in->Ncolor()*in->Nspin()*Nvec;
            Npad = Nint/Nvec;

            switch(dir) {
              case 0:
                ghostVolume = in->X(1)*in->X(2)*in->X(3)/2;
                offset = in->X(0) - 1;
                break;

              case 1:
                ghostVolume = in->X(0)*in->X(2)*in->X(3);
                offset = in->X(0)*(in->X(1) - 1);
                break;

              case 2:
                ghostVolume = in->X(0)*in->X(1)*in->X(3);
                offset = in->X(0)*in->X(1)*(in->X(2) - 1);
                break;

              case 3:
                ghostVolume = in->X(0)*in->X(1)*in->X(2);
                offset = in->Volume() - ghostVolume;
                break;
            }	

            ghostBytes = ghostVolume*Nint*sizeof(Float);
            allocateGhosts();
          }
        #endif
      }
	
      virtual ~CovDevCuda() {
        #ifdef MULTI_GPU
          if(comm_dim(dir) > 1) {
            unbindGhosts();
            freeGhosts();
          }
        #endif
        unbindGauge();
      }

      void apply(const cudaStream_t &stream) {
        #ifdef SHARED_WILSON_DSLASH
          if(dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
            errorQuda("Shared dslash (covariant derivative) does not yet support X-dimension partitioning");
        #endif
        if((dslashParam.kernel_type == EXTERIOR_KERNEL_X) || (dslashParam.kernel_type == EXTERIOR_KERNEL_Y))
          errorQuda("Covariant derivative does not yet support X or Y-dimension partitioning");

        dslashParam.parity = parity;

        for(int i=0; i<4; i++) {
          dslashParam.ghostDim[i] = 0;
          dslashParam.ghostOffset[i] = 0;
          dslashParam.ghostNormOffset[i] = 0;
          dslashParam.commDim[i] = 0;
        }

        if(dslashParam.kernel_type != INTERIOR_KERNEL) {
          #ifdef MULTI_GPU
            dslashParam.threads = ghostVolume;
            exchangeGhosts();                           // We must exchange ghosts here because the covDevCuda function requires a Dslash class as parameter
            bindGhosts();                               // and the dslash class doesn't have these specific functions to exchange ghosts

            // Maybe I should rebind the spinors for the INTERIOR kernels after this???

            TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
            COVDEV(covDevM, mu, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam, (Float2*)out->V(), (Float2*)gauge0, (Float2*)gauge1, (Float2*)in->V());
          #endif
        } else {
          dslashParam.threads = in->Volume();
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          COVDEV(covDevM, mu, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam, (Float2*)out->V(), (Float2*)gauge0, (Float2*)gauge1, (Float2*)in->V());
        }
      }

      long long flops() const { return 144 * in->Volume(); } // Correct me if I'm wrong
  };


  /**
     The function applies the color matrices of a gauge configuration to a spinor, so it is not exactly a covariant derivative, although a combination of calls to
     this function will give you a covariant derivative. The output is the new spinor (out), and the inputs are the input spinor (in), the parity of the spinor (at
     this moment, the function works for parity spinors only), the direction of the color matrices (mu) and a profiler for timing. The value mu ranges from 0 to 7,
     being anything higher than 3 understood as direction (mu - 4) backwards, and therefore applying the dagger version of the color matrices

     Again a half precision implementation was straightforwad, but I don't think it would be useful
  */

  void covDev(cudaColorSpinorField *out, cudaGaugeField &gauge, const cudaColorSpinorField *in, const int parity, const int mu, TimeProfile &profile) {
    DslashCuda *covdev = 0;
    size_t regSize = sizeof(float);

      #ifdef GPU_CONTRACT
        if(in->Precision() == QUDA_HALF_PRECISION)
          errorQuda("Error: Half precision not supported");

        if(in->Precision() != gauge.Precision())
          errorQuda("Mixing gauge %d and spinor %d precision not supported", gauge.Precision(), in->Precision());

        profile.TPSTART(QUDA_PROFILE_TOTAL);
        profile.TPSTART(QUDA_PROFILE_INIT);

        if(in->Precision() == QUDA_SINGLE_PRECISION)
          covdev = new CovDevCuda<float, float4>(out, &gauge, in, parity, mu);
        else if(in->Precision	() == QUDA_DOUBLE_PRECISION) {
          #if (__COMPUTE_CAPABILITY__ >= 130)
            covdev = new CovDevCuda<double, double2>(out, &gauge, in, parity, mu);
            regSize = sizeof(double);
          #else
            errorQuda("Error: Double precision not supported by hardware");
          #endif
        }
        profile.TPSTOP(QUDA_PROFILE_INIT);

        covDevCuda(*covdev, regSize, mu, profile);

        profile.TPSTART(QUDA_PROFILE_EPILOGUE);
        delete covdev;
        checkCudaError();
        profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
        profile.TPSTOP(QUDA_PROFILE_TOTAL);
      #else
        errorQuda("Contraction kernels have not been built");
      #endif
    }
  }
#endif
