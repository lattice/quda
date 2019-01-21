#pragma once

#include <color_spinor_field.h>
#include <gauge_field.h>
#include <register_traits.h>
#include <index_helper.cuh>
#include <tune_quda.h>
#include <dslash_quda.h>

namespace quda {

  void setPackComms(const int *); // original packing kernel
  void setPackComms2(const int *); // rewriten packing kernel

  /**
     @brief Helper function to determine if we should do halo
     computation
     @param[in] dim Dimension we are working on.  If dim=-1 (default
     argument) then we return true if type is any halo kernel.
  */
  template <KernelType type>
  __host__ __device__ inline bool doHalo(int dim=-1) {
    switch(type) {
    case EXTERIOR_KERNEL_ALL: return true;
    case EXTERIOR_KERNEL_X:   return dim==0 || dim==-1 ? true : false;
    case EXTERIOR_KERNEL_Y:   return dim==1 || dim==-1 ? true : false;
    case EXTERIOR_KERNEL_Z:   return dim==2 || dim==-1 ? true : false;
    case EXTERIOR_KERNEL_T:   return dim==3 || dim==-1 ? true : false;
    case INTERIOR_KERNEL:     return false;
    }
    return false;
  }

  /**
     @brief Helper function to determine if we should do interior
     computation
     @param[in] dim Dimension we are working on
  */
  template <KernelType type>
  __host__ __device__ inline bool doBulk() {
    switch(type) {
    case EXTERIOR_KERNEL_ALL:
    case EXTERIOR_KERNEL_X:
    case EXTERIOR_KERNEL_Y:
    case EXTERIOR_KERNEL_Z:
    case EXTERIOR_KERNEL_T:
      return false;
    case INTERIOR_KERNEL:
      return true;
    }
    return false;
  }

  /**
     @brief Helper functon to determine if the application of the
     derivative in the dslash is complete
     @param[in] Argument parameter struct
     @param[in] Checkerboard space-time index
     @param[in] Parity we are acting on
  */
  template <KernelType type, typename Arg>
  __host__ __device__ inline bool isComplete(const Arg &arg, int coord[]) {

    int incomplete = 0; // Have all 8 contributions been computed for this site?

    switch(type) { // intentional fall-through
    case EXTERIOR_KERNEL_ALL:
      incomplete = false; break; // all active threads are complete
    case INTERIOR_KERNEL:
      incomplete = incomplete || (arg.ghostDim[3] && (coord[3]==0 || coord[3]==(arg.dc.X[3]-1)));
    case EXTERIOR_KERNEL_T:
      incomplete = incomplete || (arg.ghostDim[2] && (coord[2]==0 || coord[2]==(arg.dc.X[2]-1)));
    case EXTERIOR_KERNEL_Z:
      incomplete = incomplete || (arg.ghostDim[1] && (coord[1]==0 || coord[1]==(arg.dc.X[1]-1)));
    case EXTERIOR_KERNEL_Y:
      incomplete = incomplete || (arg.ghostDim[0] && (coord[0]==0 || coord[0]==(arg.dc.X[0]-1)));
    case EXTERIOR_KERNEL_X:
      incomplete = incomplete;
    }

    return !incomplete;
  }

  /**
     @brief Compute the space-time coordinates we are at.
     @param[out] coord The computed space-time coordinates
     @param[in] arg DslashArg struct
     @param[in,out] idx Space-time index (usually equal to global
     x-thread index).  When doing EXTERIOR kernels we overwrite this
     with the index into our face (ghost index).
     @param[in] parity Field parity
     @param[out] the dimension we are working on (fused kernel only)
     @return checkerboard space-time index
  */
  template <int nDim, QudaDWFPCType pc_type, KernelType kernel_type, typename Arg, int nface_=1>
  __host__ __device__ inline int getCoords(int coord[], const Arg &arg, int &idx, int parity, int &dim) {

    
    //MWTODO - needed for improved staggered
    // constexpr int nface_=3;
    int x_cb, X;
    dim = kernel_type; // keep compiler happy
    if (kernel_type == INTERIOR_KERNEL) {
      x_cb = idx;
      getCoordsCB(coord, idx, arg.dim, arg.X0h, parity);
    } else if (kernel_type != EXTERIOR_KERNEL_ALL) {

      // compute face index and then compute coords
      const int face_num = idx >= nface_*arg.dc.ghostFaceCB[kernel_type];
      idx -= face_num*nface_*arg.dc.ghostFaceCB[kernel_type]; //MW: maybe this be FaceVolumeCB
      coordsFromFaceIndex<nDim,pc_type,kernel_type,nface_>(X, x_cb, coord, idx, face_num, parity, arg);

    } else { // fused kernel

      // work out which dimension this thread corresponds to, then compute coords
      if (idx < arg.threadDimMapUpper[0]) { // x face
        dim = 0;
        const int face_num = idx >= nface_*arg.dc.ghostFaceCB[0];
        idx -= nface_*face_num*arg.dc.ghostFaceCB[0];
        coordsFromFaceIndex<nDim,pc_type,0,nface_>(X, x_cb, coord, idx, face_num, parity, arg);
      } else if (idx < arg.threadDimMapUpper[1]) { // y face
        dim = 1;
        idx -= arg.threadDimMapLower[1];
        const int face_num = idx >= nface_*arg.dc.ghostFaceCB[1];
        idx -= nface_*face_num*arg.dc.ghostFaceCB[1];
        coordsFromFaceIndex<nDim,pc_type,1,nface_>(X, x_cb, coord, idx, face_num, parity, arg);
      } else if (idx < arg.threadDimMapUpper[2]){ // z face
        dim = 2;
        idx -= arg.threadDimMapLower[2];
        const int face_num = idx >= nface_*arg.dc.ghostFaceCB[2];
        idx -= nface_*face_num*arg.dc.ghostFaceCB[2];
        coordsFromFaceIndex<nDim,pc_type,2,nface_>(X, x_cb, coord, idx, face_num, parity, arg);
      } else { // t face
        dim = 3;
        idx -= arg.threadDimMapLower[3];
        const int face_num = idx >= nface_*arg.dc.ghostFaceCB[3];
        idx -= nface_*face_num*arg.dc.ghostFaceCB[3];
        coordsFromFaceIndex<nDim,pc_type,3,nface_>(X, x_cb, coord, idx, face_num, parity, arg);
      }

    }

    return x_cb;
  }

  /**
     @brief Compute whether the provided coordinate is within the halo
     region boundary of a given dimension.

     @param[in] coord Coordinates
     @param[in] Arg Dslash argument struct
     @return True if in boundary, else false
  */
  template <int dim, typename Arg>
  inline __host__ __device__ bool inBoundary(const int coord[], const Arg &arg){
    return ( (coord[dim] >= arg.dim[dim] - arg.nFace) || (coord[dim] < arg.nFace) );
  }

  /**
     @brief Compute whether this thread should be active for updating
     the a given offsetDim halo.  For non-fused halo update kernels
     this is a trivial kernel that just checks if the given dimension
     is partitioned and if so, return true.

     For fused halo region update kernels: here every thread has a
     prescribed dimension it is tasked with updating, but for the
     edges and vertices, the thread responsible for the entire update
     is the "greatest" one.  Hence some threads may be labelled as a
     given dimension, but they have to update other dimensions too.
     Conversely, a given thread may be labeled for a given dimension,
     but if that thread lies at en edge or vertex, and we have
     partitioned a higher dimension, then that thread will cede to the
     higher thread.

     @param[in,out] Whether this thread is "cumulatively" active
     (cumulative over all dimensions)
     @param[in] threadDim Prescribed dimension of this thread
     @param[in] offsetDim The dimension we are querying whether this
     thread should be responsible
     @param[in] offset The size of the hop
     @param[in] y Site coordinate
     @param[in] partitioned Array of which dimensions have been partitioned
     @param[in] X Lattice dimensions
     @return true if this thread is active
  */
  template <KernelType kernel_type, typename Arg>
  inline __device__ bool isActive(bool &active, int threadDim, int offsetDim, const int coord[], const Arg &arg)
  {
    // Threads with threadDim = t can handle t,z,y,x offsets
    // Threads with threadDim = z can handle z,y,x offsets
    // Threads with threadDim = y can handle y,x offsets
    // Threads with threadDim = x can handle x offsets
    if (!arg.ghostDim[offsetDim]) return false;

    if (kernel_type == EXTERIOR_KERNEL_ALL) {
      if (threadDim < offsetDim) return false;

      switch (threadDim) {
      case 3: // threadDim = T
        break;

      case 2: // threadDim = Z
        if (!arg.ghostDim[3]) break;
        if (arg.ghostDim[3] && inBoundary<3>(coord, arg)) return false;
        break;

      case 1: // threadDim = Y
        if ((!arg.ghostDim[3]) && (!arg.ghostDim[2])) break;
        if (arg.ghostDim[3] && inBoundary<3>(coord, arg)) return false;
        if (arg.ghostDim[2] && inBoundary<2>(coord, arg)) return false;
        break;

      case 0: // threadDim = X
        if ((!arg.ghostDim[3]) && (!arg.ghostDim[2]) && (!arg.ghostDim[1])) break;
        if (arg.ghostDim[3] && inBoundary<3>(coord, arg)) return false;
        if (arg.ghostDim[2] && inBoundary<2>(coord, arg)) return false;
        if (arg.ghostDim[1] && inBoundary<1>(coord, arg)) return false;
        break;

      default:
        break;
      }
    }

    active = true;
    return true;
  }

  template <typename Float>
  struct DslashArg {

    typedef typename mapper<Float>::type real;

    const int parity;     // only use this for single parity fields
    const int nParity;    // number of parities we're working on
    const int nFace;      // hard code to 1 for now
    const QudaReconstructType reconstruct;

    const int_fastdiv X0h;
    const int_fastdiv dim[5]; // full lattice dimensions
    const int volumeCB;       // checkerboarded volume
    int commDim[4];           // whether a given dimension is partitioned or not (potentially overridden for Schwarz)
    int ghostDim[4];          // always equal to actual dimension partitioning (used inside kernel to ensure corect indexing)

    const real kappa;     // kappa parameter = 1/(8+m)
    const bool dagger;    // dagger
    const bool xpay;      // whether we are doing xpay or not

    real t_proj_scale;    // factor to correct for T-dimensional spin projection

    DslashConstant dc; // pre-computed dslash constants for optimized indexing
    KernelType kernel_type;  // interior, exterior_t, etc.
    bool remote_write;       // used by the autotuner to switch on/off remote writing vs using copy engines

    int threads;              // number of threads in x-thread dimension
    int threadDimMapLower[4];
    int threadDimMapUpper[4];

    // these are set with symmetric preconditioned twisted-mass dagger
    // operator for the packing (which needs to a do a twist)
    real twist_a; // scale factor
    real twist_b; // chiral twist
    real twist_c; // flavor twist


// constructor needed for staggered to set xpay from derived class
   DslashArg(const ColorSpinorField &in, const GaugeField &U, double kappa, int parity, bool dagger, bool xpay, int nFace, const int *comm_override)
      : parity(parity), nParity(in.SiteSubset()), nFace(nFace), reconstruct(U.Reconstruct()),
        X0h(nParity == 2 ? in.X(0)/2 : in.X(0)), dim{ (3-nParity) * in.X(0), in.X(1), in.X(2), in.X(3), 1 },
        volumeCB(in.VolumeCB()), kappa(kappa), dagger(dagger), xpay(xpay),
        kernel_type(INTERIOR_KERNEL), threads(in.VolumeCB()), threadDimMapLower{ }, threadDimMapUpper{ },
        twist_a(0.0), twist_b(0.0), twist_c(0.0)
    {
      for (int d=0; d<4; d++) {
        ghostDim[d] = comm_dim_partitioned(d);
        commDim[d] = (!comm_override[d]) ? 0 : comm_dim_partitioned(d);
      }

      if (in.Location() == QUDA_CUDA_FIELD_LOCATION) {
        // create comms buffers - need to do this before we grab the dslash constants
        ColorSpinorField *in_ = const_cast<ColorSpinorField*>(&in);
        static_cast<cudaColorSpinorField*>(in_)->createComms(nFace);
      }
      dc = in.getDslashConstant();
    }

    DslashArg(const ColorSpinorField &in, const GaugeField &U, double kappa, int parity, bool dagger, const int *comm_override)
      : DslashArg(in, U, kappa, parity, dagger, kappa == 0.0 ? false : true, 1, comm_override)
      {
      };

  };


  template <typename Float>
  std::ostream& operator<<(std::ostream& out, const DslashArg<Float> &arg)
  {
    out << "parity = " << arg.parity << std::endl;
    out << "nParity = " << arg.nParity << std::endl;
    out << "nFace = " << arg.nFace << std::endl;
    out << "reconstruct = " << arg.reconstruct << std::endl;
    out << "X0h = " << arg.X0h << std::endl;
    out << "dim = { "; for (int i=0; i<5; i++) out << arg.dim[i] << (i<4 ? ", " : " }"); out << std::endl;
    out << "commDim = { "; for (int i=0; i<4; i++) out << arg.commDim[i] << (i<3 ? ", " : " }"); out << std::endl;
    out << "ghostDim = { "; for (int i=0; i<4; i++) out << arg.ghostDim[i] << (i<3 ? ", " : " }"); out << std::endl;
    out << "volumeCB = " << arg.volumeCB << std::endl;
    out << "kappa = " << arg.kappa << std::endl;
    out << "dagger = " << arg.dagger << std::endl;
    out << "xpay = " << arg.xpay << std::endl;
    out << "kernel_type = " << arg.kernel_type << std::endl;
    out << "remote_write = " << arg.remote_write << std::endl;
    out << "threads = " << arg.threads << std::endl;
    out << "threadDimMapLower = { ";
    for (int i=0; i<4; i++) out << arg.threadDimMapLower[i] << (i<3 ? ", " : " }"); out << std::endl;
    out << "threadDimMapUpper = { ";
    for (int i=0; i<4; i++) out << arg.threadDimMapUpper[i] << (i<3 ? ", " : " }"); out << std::endl;
    out << "twist_a = " << arg.twist_a;
    out << "twist_b = " << arg.twist_b;
    out << "twist_c = " << arg.twist_c;
    return out;
  }

  template <typename Float>
  class Dslash : public TunableVectorYZ {

  protected:

    DslashArg<Float> &arg;
    const ColorSpinorField &out;
    const ColorSpinorField &in;

    const int nDimComms;

    char aux_base[TuneKey::aux_n];
    char aux[8][TuneKey::aux_n];

    /**
       @brief Set the base strings used by the different dslash kernel
       types for autotuning.
    */
    inline void fillAuxBase() {
      char comm[5];
      comm[0] = (arg.commDim[0] ? '1' : '0');
      comm[1] = (arg.commDim[1] ? '1' : '0');
      comm[2] = (arg.commDim[2] ? '1' : '0');
      comm[3] = (arg.commDim[3] ? '1' : '0');
      comm[4] = '\0';
      strcpy(aux_base,",commDim=");
      strcat(aux_base,comm);

      if (arg.xpay) strcat(aux_base,",xpay");
      if (arg.dagger) strcat(aux_base,",dagger");
    }

    /**
       @brief Specialize the auxiliary strings for each kernel type
       @param[in] kernel_type The kernel_type we are generating the string got
       @param[in] kernel_str String corresponding to the kernel type
    */
    inline void fillAux(KernelType kernel_type, const char *kernel_str) {
      strcpy(aux[kernel_type],kernel_str);
      if (kernel_type == INTERIOR_KERNEL) strcat(aux[kernel_type],comm_dim_partitioned_string());
      strcat(aux[kernel_type],aux_base);
    }

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

    template <typename Arg>
    inline void setParam(Arg &arg) {
      arg.t_proj_scale = getKernelPackT() ? 1.0 : 2.0;

      // Need to reset ghost pointers prior to every call since the
      // ghost buffer may have been changed during policy tuning.
      // Also, the accessor constructor calls Ghost(), which uses
      // ghost_buf, but this is only presently set with the
      // synchronous exchangeGhost.
      static void *ghost[8]; // needs to be persistent across interior and exterior calls
      for (int dim=0; dim<4; dim++) {

        for (int dir=0; dir<2; dir++) {
          // if doing interior kernel, then this is the initial call,
          // so we set all ghost pointers else if doing exterior
          // kernel, then we only have to update the non-p2p ghosts,
          // since these may have been assigned to zero-copy memory
          if (!comm_peer2peer_enabled(1-dir, dim) || arg.kernel_type == INTERIOR_KERNEL) {
            ghost[2*dim+dir] = (Float*)((char*)in.Ghost2() + in.GhostOffset(dim,dir)*in.GhostPrecision());
          }
        }
      }

      arg.in.resetGhost(in, ghost);
    }

    virtual int tuningIter() const { return 10; }

  public:

    template <typename T, typename Arg>
    inline void launch(T *f, const TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      if (1) { // test on Volta
	this->setMaxDynamicSharedBytesPerBlock(f);
      }
      void *args[] = { &arg };
      qudaLaunchKernel((const void *)f, tp.grid, tp.block, args, tp.shared_bytes, stream);
    }

    /**
       @brief This instantiate function is used to instantiate the
       the KernelType template required for the multi-GPU dslash kernels.
       @param[in] tp The tuning parameters to use for this kernel
       @param[in,out] arg The argument struct for the kernel
       @param[in] stream The cudaStream_t where the kernel will run
     */
    template < template <typename,int,int,int,bool,bool,KernelType,typename> class Launch,
               int nDim, int nColor, int nParity, bool dagger, bool xpay, typename Arg>
    inline void instantiate(TuneParam &tp, Arg &arg, const cudaStream_t &stream) {

      if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
        errorQuda("Not implemented");
      } else {
        switch(arg.kernel_type) {
        case INTERIOR_KERNEL:
          Launch<Float,nDim,nColor,nParity,dagger,xpay,INTERIOR_KERNEL,Arg>::launch(*this, tp, arg, stream); break;
        case EXTERIOR_KERNEL_X:
          Launch<Float,nDim,nColor,nParity,dagger,xpay,EXTERIOR_KERNEL_X,Arg>::launch(*this, tp, arg, stream); break;
        case EXTERIOR_KERNEL_Y:
          Launch<Float,nDim,nColor,nParity,dagger,xpay,EXTERIOR_KERNEL_Y,Arg>::launch(*this, tp, arg, stream); break;
        case EXTERIOR_KERNEL_Z:
          Launch<Float,nDim,nColor,nParity,dagger,xpay,EXTERIOR_KERNEL_Z,Arg>::launch(*this, tp, arg, stream); break;
        case EXTERIOR_KERNEL_T:
          Launch<Float,nDim,nColor,nParity,dagger,xpay,EXTERIOR_KERNEL_T,Arg>::launch(*this, tp, arg, stream); break;
        case EXTERIOR_KERNEL_ALL:
          Launch<Float,nDim,nColor,nParity,dagger,xpay,EXTERIOR_KERNEL_ALL,Arg>::launch(*this, tp, arg, stream); break;
        default: errorQuda("Unexpected kernel type %d", arg.kernel_type);
        }
      }
    }

    /**
       @brief This instantiate function is used to instantiate the
       the dagger template
       @param[in] tp The tuning parameters to use for this kernel
       @param[in,out] arg The argument struct for the kernel
       @param[in] stream The cudaStream_t where the kernel will run
     */
    template < template <typename,int,int,int,bool,bool,KernelType,typename> class Launch,
               int nDim, int nColor, int nParity, bool xpay, typename Arg>
    inline void instantiate(TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      if (arg.dagger) instantiate<Launch,nDim,nColor,nParity, true,xpay>(tp, arg, stream);
      else            instantiate<Launch,nDim,nColor,nParity,false,xpay>(tp, arg, stream);
    }

    /**
       @brief This instantiate function is used to instantiate the
       the nParity template
       @param[in] tp The tuning parameters to use for this kernel
       @param[in,out] arg The argument struct for the kernel
       @param[in] stream The cudaStream_t where the kernel will run
     */
    template < template <typename,int,int,int,bool,bool,KernelType,typename> class Launch,
               int nDim, int nColor, bool xpay, typename Arg>
    inline void instantiate(TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      switch (arg.nParity) {
      case 1: instantiate<Launch,nDim,nColor,1,xpay>(tp, arg, stream); break;
      case 2: instantiate<Launch,nDim,nColor,2,xpay>(tp, arg, stream); break;
      default:
        errorQuda("nParity = %d undefined\n", arg.nParity);
      }
    }

    DslashArg<Float> &dslashParam; // temporary addition for policy compatibility

    Dslash(DslashArg<Float> &arg, const ColorSpinorField &out, const ColorSpinorField &in)
      : TunableVectorYZ(1,arg.nParity), arg(arg), out(out), in(in), nDimComms(4), dslashParam(arg)
    {
      // this sets the communications pattern for the packing kernel
      setPackComms(arg.commDim);
      setPackComms2(arg.commDim);

      //strcpy(aux, in.AuxString());
      fillAuxBase();
#ifdef MULTI_GPU
      fillAux(INTERIOR_KERNEL, "policy_kernel=interior");
      fillAux(EXTERIOR_KERNEL_ALL, "policy_kernel=exterior_all");
      fillAux(EXTERIOR_KERNEL_X, "policy_kernel=exterior_x");
      fillAux(EXTERIOR_KERNEL_Y, "policy_kernel=exterior_y");
      fillAux(EXTERIOR_KERNEL_Z, "policy_kernel=exterior_z");
      fillAux(EXTERIOR_KERNEL_T, "policy_kernel=exterior_t");
#else
      fillAux(INTERIOR_KERNEL, "policy_kernel=single-GPU");
#endif // MULTI_GPU
      fillAux(KERNEL_POLICY, "policy");
    }

    int Nface() const { return 2*arg.nFace; } // factor of 2 is for forwards/backwards (convention used in dslash policy)
    int Dagger() const { return arg.dagger; }

    const char* getAux(KernelType type) const {
      return aux[type];
    }

    void setAux(KernelType type, const char *aux_) {
      strcpy(aux[type], aux_);
    }

    void augmentAux(KernelType type, const char *extra) {
      strcat(aux[type], extra);
    }

    /**
       @brief Save the output field since the output field is both
       read from and written to in the exterior kernels
     */
    virtual void preTune() { if (arg.kernel_type != INTERIOR_KERNEL && arg.kernel_type != KERNEL_POLICY) out.backup(); }

    /**
       @brief Restore the output field if doing exterior kernel
     */
    virtual void postTune() { if (arg.kernel_type != INTERIOR_KERNEL && arg.kernel_type != KERNEL_POLICY) out.restore(); }

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
    virtual long long flops() const
    {
      int mv_flops = (8 * in.Ncolor() - 2) * in.Ncolor(); // SU(3) matrix-vector flops
      int num_mv_multiply = in.Nspin() == 4 ? 2 : 1;
      int ghost_flops = (num_mv_multiply * mv_flops + 2*in.Ncolor()*in.Nspin());
      int xpay_flops = 2 * 2 * in.Ncolor() * in.Nspin(); // multiply and add per real component
      int num_dir = 2*in.Ndim();

      long long flops_ = 0;

      switch(arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        flops_ = (ghost_flops + (arg.xpay ? xpay_flops : xpay_flops/2)) * 2 * in.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL:
        {
          long long ghost_sites = 2 * (in.GhostFace()[0]+in.GhostFace()[1]+in.GhostFace()[2]+in.GhostFace()[3]);
          flops_ = (ghost_flops + (arg.xpay ? xpay_flops : xpay_flops/2)) * ghost_sites;
          break;
        }
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
        {
          long long sites = in.Volume();
          flops_ = (num_dir*(in.Nspin()/4)*in.Ncolor()*in.Nspin() +   // spin project (=0 for staggered)
                    num_dir*num_mv_multiply*mv_flops +                // SU(3) matrix-vector multiplies
                    ((num_dir-1)*2*in.Ncolor()*in.Nspin())) * sites;  // accumulation
          if (arg.xpay) flops_ += xpay_flops * sites;

          if (arg.kernel_type == KERNEL_POLICY) break;
          // now correct for flops done by exterior kernel
          long long ghost_sites = 0;
          for (int d=0; d<4; d++) if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
          flops_ -= ghost_flops * ghost_sites;

          break;
        }
      }

      return flops_;
    }

    long long bytes() const
    {
      int gauge_bytes = arg.reconstruct * in.Precision();
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() + (isFixed ? sizeof(float) : 0);
      int proj_spinor_bytes = in.Ncolor() * in.Nspin() * in.Precision() + (isFixed ? sizeof(float) : 0);
      int ghost_bytes = (proj_spinor_bytes + gauge_bytes) + 2*spinor_bytes; // 2 since we have to load the partial
      int num_dir = 2 * in.Ndim(); // set to 4 dimensions since we take care of 5-d fermions in derived classes where necessary

      long long bytes_=0;

      switch(arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        bytes_ = ghost_bytes * 2 * in.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL:
        {
          long long ghost_sites = 2 * (in.GhostFace()[0]+in.GhostFace()[1]+in.GhostFace()[2]+in.GhostFace()[3]);
          bytes_ = ghost_bytes * ghost_sites;
          break;
        }
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
        {
          long long sites = in.Volume();
          bytes_ = (num_dir*gauge_bytes + ((num_dir-2)*spinor_bytes + 2*proj_spinor_bytes) + spinor_bytes)*sites;
          if (arg.xpay) bytes_ += spinor_bytes;

          if (arg.kernel_type == KERNEL_POLICY) break;
          // now correct for bytes done by exterior kernel
          long long ghost_sites = 0;
          for (int d=0; d<4; d++) if (arg.commDim[d]) ghost_sites += 2*in.GhostFace()[d];
          bytes_ -= ghost_bytes * ghost_sites;

          break;
        }
      }
      return bytes_;
    }

  };

}
