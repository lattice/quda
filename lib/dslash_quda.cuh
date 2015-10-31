
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
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
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
      FUNC ## 18 ## 18 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_13) {			\
      FUNC ## 18 ## 13 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 18 ## 12 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_9) {			\
      FUNC ## 18 ## 9 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
      FUNC ## 18 ## 8 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## 18 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_13) {			\
      FUNC ## 18 ## 13 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 18 ## 12 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_9) {			\
      FUNC ## 18 ## 9 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
      FUNC ## 18 ## 8 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
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
  case EXTERIOR_KERNEL_ALL:						\
    MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_ALL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
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
  case EXTERIOR_KERNEL_ALL:						\
    MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_ALL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
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
  GENERIC_DSLASH(staggeredDslash, , Axpy, gridDim, blockDim, shared, stream, param, __VA_ARGS__)

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
  case EXTERIOR_KERNEL_ALL:						\
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_ALL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
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
  if (x == 0 && d == 0) {						\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## Twist ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## Twist ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else {								\
      FUNC ## 8 ## DAG ## Twist ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else if (x != 0 && d == 0) {					\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## Twist ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## Twist ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
      FUNC ## 8 ## DAG ## Twist ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else if (x == 0 && d != 0) {					\
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
  case EXTERIOR_KERNEL_ALL:						\
    MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_ALL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  }

#endif

#define NDEG_TM_DSLASH(FUNC, gridDim, blockDim, shared, stream, param, ...) \
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
  const int dagger;

  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  // all dslashes expect a 4-d volume here (dwf Ls is y thread dimension)
  unsigned int minThreads() const { return in->X(0) * in->X(1) * in->X(2) * in->X(3); } 
  char aux[7][TuneKey::aux_n];

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
    if (dagger) strcat(aux[kernel_type],",dagger");
  }

public:
  DslashCuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
	     const cudaColorSpinorField *x, const QudaReconstructType reconstruct,
	     const int dagger) 
    : out(out), in(in), x(x), reconstruct(reconstruct), 
      dagger(dagger), saveOut(0), saveOutNorm(0) { 

#ifdef MULTI_GPU 
    fillAux(INTERIOR_KERNEL, "type=interior");
    fillAux(EXTERIOR_KERNEL_ALL, "type=exterior_all");
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

    for (int i=0; i<4; i++) dslashParam.X[i] = in->X(i);
#ifdef GPU_DOMAIN_WALL_DIRAC
    dslashParam.Ls = in->X(4); // needed by tuneLaunch()
#endif
    // this is a c/b field so double the x dimension
    dslashParam.X[0] *= 2;
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
    if (dslashParam.kernel_type != INTERIOR_KERNEL) { // exterior kernel
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
    if (dslashParam.kernel_type != INTERIOR_KERNEL) { // exterior kernel
      cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (out->Precision() == QUDA_HALF_PRECISION) {
	cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
	delete[] saveOutNorm;
      }
    }
  }

  /*void launch_auxiliary(cudaStream_t &stream) {
    auxiliary.apply(stream);
    }*/
  
  /*
    per direction / dimension flops
    spin project flops = Nc * Ns
    SU(3) matrix-vector flops = (8 Nc - 2) * Nc
    spin reconstruction flops = 2 * Nc * Ns (just an accumulation to all components)
    xpay = 2 * 2 * Nc * Ns
      
    So for the full dslash we have, where for the final spin
    reconstruct we have -1 since the first direction does not
    require any accumulation.
      
    flops = (2 * Nd * Nc * Ns)  +  (2 * Nd * (Ns/2) * (8*Nc-2) * Nc)  +  ((2 * Nd - 1) * 2 * Nc * Ns)
    flops_xpay = flops + 2 * 2 * Nc * Ns
      
    For Wilson this should give 1344 for Nc=3,Ns=2 and 1368 for the xpay equivalent
  */
  virtual long long flops() const {
    int mv_flops = (8 * in->Ncolor() - 2) * in->Ncolor(); // SU(3) matrix-vector flops
    int num_mv_multiply = in->Nspin() == 4 ? 2 : 1;
    int ghost_flops = (num_mv_multiply * mv_flops + 2*in->Ncolor()*in->Nspin());
    int xpay_flops = 2 * 2 * in->Ncolor() * in->Nspin(); // multiply and add per real component
    int num_dir = 2 * 4;

    long long flops;
    switch(dslashParam.kernel_type) {
    case EXTERIOR_KERNEL_X:
    case EXTERIOR_KERNEL_Y:
    case EXTERIOR_KERNEL_Z:
    case EXTERIOR_KERNEL_T:
      flops = (ghost_flops + (x ? xpay_flops : 0)) * 2 * in->GhostFace()[dslashParam.kernel_type];
      break;
    case EXTERIOR_KERNEL_ALL:
      {
	long long ghost_sites = 2 * (in->GhostFace()[0]+in->GhostFace()[1]+in->GhostFace()[2]+in->GhostFace()[3]);
	flops = (ghost_flops + (x ? xpay_flops : 0)) * ghost_sites;
	break;
      }
    case INTERIOR_KERNEL:
      {
	long long sites = in->VolumeCB();
	flops = (num_dir*(in->Nspin()/4)*in->Ncolor()*in->Nspin() +   // spin project (=0 for staggered)
		 num_dir*num_mv_multiply*mv_flops +                   // SU(3) matrix-vector multiplies
		 ((num_dir-1)*2*in->Ncolor()*in->Nspin())) * sites;   // accumulation
	if (x) flops += xpay_flops * sites;

	// now correct for flops done by exterior kernel
	long long ghost_sites = 0;
	for (int d=0; d<4; d++) if (dslashParam.commDim[d]) ghost_sites += 2 * in->GhostFace()[d];
	flops -= (ghost_flops + (x ? xpay_flops : 0)) * ghost_sites;
	  
	break;
      }
    }
    return flops;
  }

  virtual long long bytes() const {
    int gauge_bytes = reconstruct * in->Precision();
    bool isHalf = in->Precision() == sizeof(short) ? true : false;
    int spinor_bytes = 2 * in->Ncolor() * in->Nspin() * in->Precision() + (isHalf ? sizeof(float) : 0);
    int proj_spinor_bytes = (in->Nspin()==4 ? 1 : 2) * in->Ncolor() * in->Nspin() * in->Precision() + (isHalf ? sizeof(float) : 0);
    int ghost_bytes = (proj_spinor_bytes + gauge_bytes) + spinor_bytes;
    int num_dir = 2 * 4; // set to 4 dimensions since we take care of 5-d fermions in derived classes where necessary

    long long bytes;
    switch(dslashParam.kernel_type) {
    case EXTERIOR_KERNEL_X:
    case EXTERIOR_KERNEL_Y:
    case EXTERIOR_KERNEL_Z:
    case EXTERIOR_KERNEL_T:
      bytes = (ghost_bytes + (x ? spinor_bytes : 0)) * 2 * in->GhostFace()[dslashParam.kernel_type];
      break;
    case EXTERIOR_KERNEL_ALL:
      {
	long long ghost_sites = 2 * (in->GhostFace()[0]+in->GhostFace()[1]+in->GhostFace()[2]+in->GhostFace()[3]);
	bytes = (ghost_bytes + (x ? spinor_bytes : 0)) * ghost_sites;
	break;
      }
    case INTERIOR_KERNEL:
      {
	long long sites = in->VolumeCB();
	bytes = (num_dir*gauge_bytes + ((num_dir-2)*spinor_bytes + 2*proj_spinor_bytes) + spinor_bytes)*sites;
	if (x) bytes += spinor_bytes;

	// now correct for bytes done by exterior kernel
	long long ghost_sites = 0;
	for (int d=0; d<4; d++) if (dslashParam.commDim[d]) ghost_sites += 2*in->GhostFace()[d];
	bytes -= (ghost_bytes + (x ? spinor_bytes : 0)) * ghost_sites;
	  
	break;
      }
    }
    return bytes;
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
    unsigned int gx = (in->X(0)*in->X(3) + block.x - 1) / block.x;
    unsigned int gy = (in->X(1) + block.y - 1 ) / block.y;	
    unsigned int gz = (in->X(2) + block.z - 1) / block.z;
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
    for (unsigned bx=blockInit.x; bx<=in->X(0); bx++) {
      //unsigned int gx = (in->X(0)*in->x(3) + bx - 1) / bx;
      for (unsigned by=blockInit.y; by<=in->X(1); by++) {
	unsigned int gy = (in->X(1) + by - 1 ) / by;	

	if (by > 1 && (by%2) != 0) continue; // can't handle odd blocks yet except by=1

	for (unsigned bz=blockInit.z; bz<=in->X(2); bz++) {
	  unsigned int gz = (in->X(2) + bz - 1) / bz;

	  if (bz > 1 && (bz%2) != 0) continue; // can't handle odd blocks yet except bz=1
	  if (bx*by*bz > max_threads) continue;
	  if (bx*by*bz < min_threads) continue;
	  // can't yet handle the last block properly in shared memory addressing
	  if (by*gy != in->X(1)) continue;
	  if (bz*gz != in->X(2)) continue;
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

    if (param.block.x > in->X(0) && param.block.y > in->X(1) && param.block.z > in->X(2) || !set) {
      //||sharedBytesPerThread()*param.block.x > max_shared) {
      param.block = dim3(in->X(0), 1, 1);
      return false;
    } else { 
      param.grid = createGrid(param.block);
      param.shared_bytes = sharedBytes(param.block);
      return true; 
    }
  }

public:
  SharedDslashCuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		   const cudaColorSpinorField *x, QudaReconstructType reconstruct, int dagger) 
    : DslashCuda(out, in, x, reconstruct, dagger) { ; }
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

    param.block = dim3(in->X(0), 1, 1);
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
		   const cudaColorSpinorField *x, QudaReconstructType reconstruct, int dagger) 
    : DslashCuda(out, in, x, reconstruct, dagger) { }
  virtual ~SharedDslashCuda() { }
};
#endif
