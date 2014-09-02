
  static FaceBuffer *face[2];

  void setFace(const FaceBuffer &Face1, const FaceBuffer &Face2) {
    face[0] = (FaceBuffer*)&(Face1); 
    face[1] = (FaceBuffer*)&(Face2); // nasty
  }

#define MORE_GENERIC_DSLASH(FUNC, DAG, X, kernel_type, gridDim, blockDim, shared, stream, param,  ...) \
  if (x==0) {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {								\
      FUNC ## 8 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
      FUNC ## 8 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  }


#define MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, kernel_type, gridDim, blockDim, shared, stream, param,  ...) \
  if (x==0) {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_13) {			\
      FUNC ## 13 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_9) {								\
      FUNC ## 9 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {								\
      FUNC ## 8 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_13) {			\
      FUNC ## 13 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_9) {			\
      FUNC ## 9 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
      FUNC ## 8 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }                                                                   \
  }


#ifndef MULTI_GPU

#define GENERIC_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
    case INTERIOR_KERNEL:							\
                                                                                MORE_GENERIC_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    default:								\
                                                                        errorQuda("KernelType %d not defined for single GPU", param.kernel_type); \
  }

#define GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
    case INTERIOR_KERNEL:							\
                                                                                MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    default:								\
                                                                        errorQuda("KernelType %d not defined for single GPU", param.kernel_type); \
  }


#else

#define GENERIC_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
    case INTERIOR_KERNEL:							\
                                                                                MORE_GENERIC_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL,   gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_X:						\
                                                                        MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_X, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_Y:						\
                                                                        MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Y, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_Z:						\
                                                                        MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Z, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_T:						\
                                                                        MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_T, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
  }

#define GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
    case INTERIOR_KERNEL:							\
                                                                                MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL,   gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_X:						\
                                                                        MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_X, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_Y:						\
                                                                        MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Y, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_Z:						\
                                                                        MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Z, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_T:						\
                                                                        MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_T, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
  }


#endif

  // macro used for dslash types with dagger kernel defined (Wilson, domain wall, etc.)
#define DSLASH(FUNC, gridDim, blockDim, shared, stream, param, ...)	\
  if (!dagger) {							\
    GENERIC_DSLASH(FUNC, , Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
  } else {								\
    GENERIC_DSLASH(FUNC, Dagger, Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
  }

  // macro used for staggered dslash
#define STAGGERED_DSLASH(gridDim, blockDim, shared, stream, param, ...)	\
  GENERIC_STAGGERED_DSLASH(staggeredDslash, , Axpy, gridDim, blockDim, shared, stream, param, __VA_ARGS__)

#define IMPROVED_STAGGERED_DSLASH(gridDim, blockDim, shared, stream, param, ...) \
  GENERIC_STAGGERED_DSLASH(improvedStaggeredDslash, , Axpy, gridDim, blockDim, shared, stream, param, __VA_ARGS__) 

#define MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, kernel_type, gridDim, blockDim, shared, stream, param,  ...) \
  if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
    FUNC ## 18 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
  } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
    FUNC ## 12 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
    FUNC ## 8 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
  }									

#ifndef MULTI_GPU

#define GENERIC_ASYM_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
    case INTERIOR_KERNEL:							\
                                                                                MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    default:								\
                                                                        errorQuda("KernelType %d not defined for single GPU", param.kernel_type); \
  }

#else

#define GENERIC_ASYM_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
    case INTERIOR_KERNEL:							\
                                                                                MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL,   gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_X:						\
                                                                        MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_X, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_Y:						\
                                                                        MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Y, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_Z:						\
                                                                        MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Z, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_T:						\
                                                                        MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_T, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
  }

#endif

  // macro used for dslash types with dagger kernel defined (Wilson, domain wall, etc.)
#define ASYM_DSLASH(FUNC, gridDim, blockDim, shared, stream, param, ...) \
  if (!dagger) {							\
    GENERIC_ASYM_DSLASH(FUNC, , Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
  } else {								\
    GENERIC_ASYM_DSLASH(FUNC, Dagger, Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
  }



  //macro used for twisted mass dslash:

#define MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, kernel_type, gridDim, blockDim, shared, stream, param,  ...) \
  if (x == 0 && d == 0) {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## Twist ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## Twist ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else {								\
      FUNC ## 8 ## DAG ## Twist ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else if (x != 0 && d == 0) {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## Twist ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## Twist ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
      FUNC ## 8 ## DAG ## Twist ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else if (x == 0 && d != 0) {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else {								\
      FUNC ## 8 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else{								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
      FUNC ## 8 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  }

#ifndef MULTI_GPU

#define GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
    case INTERIOR_KERNEL:							\
                                                                                MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    default:								\
                                                                        errorQuda("KernelType %d not defined for single GPU", param.kernel_type); \
  }

#else

#define GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
    case INTERIOR_KERNEL:							\
                                                                                MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL,   gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_X:						\
                                                                        MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_X, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_Y:						\
                                                                        MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Y, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_Z:						\
                                                                        MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Z, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
    case EXTERIOR_KERNEL_T:						\
                                                                        MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_T, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
    break;								\
  }

#endif

#define NDEG_TM_DSLASH(FUNC, gridDim, blockDim, shared, stream, param, ...)	\
  if (!dagger) {							\
    GENERIC_NDEG_TM_DSLASH(FUNC, , Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
  } else {								\
    GENERIC_NDEG_TM_DSLASH(FUNC, Dagger, Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
  }
  //end of tm dslash macro


  // Use an abstract class interface to drive the different CUDA dslash
  // kernels. All parameters are curried into the derived classes to
  // allow a simple interface.
  class DslashCuda : public Tunable {

  protected:
    cudaColorSpinorField *out;
    const cudaColorSpinorField *in;
    const cudaColorSpinorField *x;
    const QudaReconstructType reconstruct;
    char *saveOut, *saveOutNorm;

    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return dslashConstants.VolumeCB(); }
    char aux[6][256];

    void fillAux(KernelType kernel_type, const char *kernel_str) {
      strcpy(aux[kernel_type],kernel_str);
#ifdef MULTI_GPU
      char comm[5];
      comm[0] = (dslashParam.commDim[0] ? '1' : '0');
      comm[1] = (dslashParam.commDim[1] ? '1' : '0');
      comm[2] = (dslashParam.commDim[2] ? '1' : '0');
      comm[3] = (dslashParam.commDim[3] ? '1' : '0');
      comm[4] = '\0'; 
      strcat(aux[kernel_type],",comm=");
      strcat(aux[kernel_type],comm);
      if (kernel_type == INTERIOR_KERNEL) {
	char ghost[5];
	ghost[0] = (dslashParam.ghostDim[0] ? '1' : '0');
	ghost[1] = (dslashParam.ghostDim[1] ? '1' : '0');
	ghost[2] = (dslashParam.ghostDim[2] ? '1' : '0');
	ghost[3] = (dslashParam.ghostDim[3] ? '1' : '0');
	ghost[4] = '\0';
	strcat(aux[kernel_type],",ghost=");
	strcat(aux[kernel_type],ghost);
      }
#endif

      if (reconstruct == QUDA_RECONSTRUCT_NO) 
	strcat(aux[kernel_type],",reconstruct=18");
      else if (reconstruct == QUDA_RECONSTRUCT_13) 
	strcat(aux[kernel_type],",reconstruct=13");
      else if (reconstruct == QUDA_RECONSTRUCT_12) 
	strcat(aux[kernel_type],",reconstruct=12");
      else if (reconstruct == QUDA_RECONSTRUCT_9) 
	strcat(aux[kernel_type],",reconstruct=9");
      else if (reconstruct == QUDA_RECONSTRUCT_8) 
	strcat(aux[kernel_type],",reconstruct=8");

      if (x) strcat(aux[kernel_type],",Xpay");
    }

  public:
    DslashCuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
	       const cudaColorSpinorField *x, const QudaReconstructType reconstruct) 
      : out(out), in(in), x(x), reconstruct(reconstruct), saveOut(0), saveOutNorm(0) { 

#ifdef MULTI_GPU 
      fillAux(INTERIOR_KERNEL, "type=interior");
      fillAux(EXTERIOR_KERNEL_X, "type=exterior_x");
      fillAux(EXTERIOR_KERNEL_Y, "type=exterior_y");
      fillAux(EXTERIOR_KERNEL_Z, "type=exterior_z");
      fillAux(EXTERIOR_KERNEL_T, "type=exterior_t");
#else
      fillAux(INTERIOR_KERNEL, "type=single-GPU");
#endif // MULTI_GPU

      dslashParam.sp_stride = in->Stride();

      // this sets the communications pattern for the packing kernel
      setPackComms(dslashParam.commDim);
    }

    virtual ~DslashCuda() { }
    virtual TuneKey tuneKey() const  
    { return TuneKey(in->VolString(), typeid(*this).name(), aux[dslashParam.kernel_type]); }

    std::string paramString(const TuneParam &param) const // Don't bother printing the grid dim.
    {
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }
    virtual int Nface() { return 2; }

    virtual void preTune()
    {
      if (dslashParam.kernel_type < 5) { // exterior kernel
	saveOut = new char[out->Bytes()];
	cudaMemcpy(saveOut, out->V(), out->Bytes(), cudaMemcpyDeviceToHost);
	if (out->Precision() == QUDA_HALF_PRECISION) {
	  saveOutNorm = new char[out->NormBytes()];
	  cudaMemcpy(saveOutNorm, out->Norm(), out->NormBytes(), cudaMemcpyDeviceToHost);
	}
      }
    }
    
    virtual void postTune()
    {
      if (dslashParam.kernel_type < 5) { // exterior kernel
	cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
	delete[] saveOut;
	if (out->Precision() == QUDA_HALF_PRECISION) {
	  cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
	  delete[] saveOutNorm;
	}
      }
    }

  };

  /** This derived class is specifically for driving the Dslash kernels
    that use shared memory blocking.  This only applies on Fermi and
    upwards, and only for the interior kernels. */
#if (__COMPUTE_CAPABILITY__ >= 200 && defined(SHARED_WILSON_DSLASH)) 
  class SharedDslashCuda : public DslashCuda {
    protected:
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; } // FIXME: this isn't quite true, but works
      bool advanceSharedBytes(TuneParam &param) const { 
        if (dslashParam.kernel_type != INTERIOR_KERNEL) return DslashCuda::advanceSharedBytes(param);
        else return false;
      } // FIXME - shared memory tuning only supported on exterior kernels

      /** Helper function to set the shared memory size from the 3-d block size */
      int sharedBytes(const dim3 &block) const { 
        int warpSize = 32; // FIXME - query from device properties
        int block_xy = block.x*block.y;
        if (block_xy % warpSize != 0) block_xy = ((block_xy / warpSize) + 1)*warpSize;
        return block_xy*block.z*sharedBytesPerThread();
      }

      /** Helper function to set the 3-d grid size from the 3-d block size */
      dim3 createGrid(const dim3 &block) const {
        unsigned int gx = ((dslashConstants.x[0]/2)*dslashConstants.x[3] + block.x - 1) / block.x;
        unsigned int gy = (dslashConstants.x[1] + block.y - 1 ) / block.y;	
        unsigned int gz = (dslashConstants.x[2] + block.z - 1) / block.z;
        return dim3(gx, gy, gz);
      }

      /** Advance the 3-d block size. */
      bool advanceBlockDim(TuneParam &param) const {
        if (dslashParam.kernel_type != INTERIOR_KERNEL) return DslashCuda::advanceBlockDim(param);
        const unsigned int min_threads = 2;
        const unsigned int max_threads = 512; // FIXME: use deviceProp.maxThreadsDim[0];
        const unsigned int max_shared = 16384*3; // FIXME: use deviceProp.sharedMemPerBlock;

        // set the x-block dimension equal to the entire x dimension
        bool set = false;
        dim3 blockInit = param.block;
        blockInit.z++;
        for (unsigned bx=blockInit.x; bx<=dslashConstants.x[0]/2; bx++) {
          //unsigned int gx = (dslashConstants.x[0]*dslashConstants.x[3] + bx - 1) / bx;
          for (unsigned by=blockInit.y; by<=dslashConstants.x[1]; by++) {
            unsigned int gy = (dslashConstants.x[1] + by - 1 ) / by;	

            if (by > 1 && (by%2) != 0) continue; // can't handle odd blocks yet except by=1

            for (unsigned bz=blockInit.z; bz<=dslashConstants.x[2]; bz++) {
              unsigned int gz = (dslashConstants.x[2] + bz - 1) / bz;

              if (bz > 1 && (bz%2) != 0) continue; // can't handle odd blocks yet except bz=1
              if (bx*by*bz > max_threads) continue;
              if (bx*by*bz < min_threads) continue;
              // can't yet handle the last block properly in shared memory addressing
              if (by*gy != dslashConstants.x[1]) continue;
              if (bz*gz != dslashConstants.x[2]) continue;
              if (sharedBytes(dim3(bx, by, bz)) > max_shared) continue;

              param.block = dim3(bx, by, bz);	  
              set = true; break;
            }
            if (set) break;
            blockInit.z = 1;
          }
          if (set) break;
          blockInit.y = 1;
        }

        if (param.block.x > dslashConstants.x[0]/2 && param.block.y > dslashConstants.x[1] &&
            param.block.z > dslashConstants.x[2] || !set) {
          //||sharedBytesPerThread()*param.block.x > max_shared) {
          param.block = dim3(dslashConstants.x[0]/2, 1, 1);
          return false;
        } else { 
          param.grid = createGrid(param.block);
          param.shared_bytes = sharedBytes(param.block);
          return true; 
        }
      }

  public:
      SharedDslashCuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		       const cudaColorSpinorField *x, const QudaReconstructType reconstruct) 
	: DslashCuda(out, in, x, reconstruct) { ; }
      virtual ~SharedDslashCuda() { ; }
      std::string paramString(const TuneParam &param) const // override and print out grid as well
      {
	std::stringstream ps;
	ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
	ps << "grid=(" << param.grid.x << "," << param.grid.y << "," << param.grid.z << "), ";
	ps << "shared=" << param.shared_bytes;
	return ps.str();
      }

      virtual void initTuneParam(TuneParam &param) const
      {
	if (dslashParam.kernel_type != INTERIOR_KERNEL) return DslashCuda::initTuneParam(param);

	param.block = dim3(dslashConstants.x[0]/2, 1, 1);
	param.grid = createGrid(param.block);
	param.shared_bytes = sharedBytes(param.block);
      }

      /** Sets default values for when tuning is disabled - this is guaranteed to work, but will be slow */
      virtual void defaultTuneParam(TuneParam &param) const
      {
	if (dslashParam.kernel_type != INTERIOR_KERNEL) DslashCuda::defaultTuneParam(param);
	else initTuneParam(param);
      }
    };
#else /** For pre-Fermi architectures */
    class SharedDslashCuda : public DslashCuda {
    public:
      SharedDslashCuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		       const cudaColorSpinorField *x, QudaReconstructType reconstruct) 
	: DslashCuda(out, in, x, reconstruct) { }
      virtual ~SharedDslashCuda() { }
    };
#endif
