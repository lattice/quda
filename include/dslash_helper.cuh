#pragma once

#include <register_traits.h>
#include <index_helper.cuh>
#include <tune_quda.h>

namespace quda {

  void setPackComms(const int *);

  enum KernelType {
    INTERIOR_KERNEL = 5,
    EXTERIOR_KERNEL_ALL = 6,
    EXTERIOR_KERNEL_X = 0,
    EXTERIOR_KERNEL_Y = 1,
    EXTERIOR_KERNEL_Z = 2,
    EXTERIOR_KERNEL_T = 3,
    KERNEL_POLICY = 7
  };

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
    case INTERIOR_KERNEL:
      incomplete = incomplete || (arg.commDim[3] && (coord[3]==0 || coord[3]==(arg.dc.X[3]-1)));
    case EXTERIOR_KERNEL_T:
      incomplete = incomplete || (arg.commDim[2] && (coord[2]==0 || coord[2]==(arg.dc.X[2]-1)));
    case EXTERIOR_KERNEL_Z:
      incomplete = incomplete || (arg.commDim[1] && (coord[1]==0 || coord[1]==(arg.dc.X[1]-1)));
    case EXTERIOR_KERNEL_Y:
      incomplete = incomplete || (arg.commDim[0] && (coord[0]==0 || coord[0]==(arg.dc.X[0]-1)));
    case EXTERIOR_KERNEL_X:
      incomplete = incomplete;
    }

    return !incomplete;
  }

  template <typename Float>
  struct DslashArg {

    typedef typename mapper<Float>::type real;

    const int parity;     // only use this for single parity fields
    const int nParity;    // number of parities we're working on
    const int nFace;      // hard code to 1 for now
    const QudaReconstructType reconstruct;

    const int_fastdiv X0h;
    const int_fastdiv dim[5];     // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;   // checkerboarded volume

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

    // compatibility with dslash_policy for now
    real twist_a;
    real twist_b;

    DslashArg(const ColorSpinorField &in, const GaugeField &U, double kappa, int parity, bool dagger)
      : parity(parity), nParity(in.SiteSubset()), nFace(1), reconstruct(U.Reconstruct()),
        X0h(nParity == 2 ? in.X(0)/2 : in.X(0)),
        dim{ (3-nParity) * in.X(0), in.X(1), in.X(2), in.X(3), 1 },
        commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
        volumeCB(in.VolumeCB()), kappa(kappa), dagger(dagger), xpay(kappa == 0.0 ? false : true),
        kernel_type(INTERIOR_KERNEL), threads(in.VolumeCB()), threadDimMapLower{ }, threadDimMapUpper{ },
        twist_a(0.0), twist_b(0.0)
    {
      if (in.Location() == QUDA_CUDA_FIELD_LOCATION) {
        // create comms buffers - need to do this before we grab the dslash constants
        ColorSpinorField *in_ = const_cast<ColorSpinorField*>(&in);
        static_cast<cudaColorSpinorField*>(in_)->createComms(1);
      }
      dc = in.getDslashConstant();
    }

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
    return out;
  }

  //static declarations
  static bool init = false;
  static char ghost_str[TuneKey::aux_n]; // string with ghostDim information

  template <typename Float>
  class Dslash : public TunableVectorY {

  protected:

    DslashArg<Float> &arg;
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
      strcpy(aux_base,",comm=");
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
      if (kernel_type == INTERIOR_KERNEL) strcat(aux[kernel_type],ghost_str);
      strcat(aux[kernel_type],aux_base);
    }

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

    template <typename T, typename Arg>
    inline void launch(T *f, const TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      if (1) { // test on Volta
	this->setMaxDynamicSharedBytesPerBlock(f);
      }
      void *args[] = { &arg };
      qudaLaunchKernel((const void *)f, tp.grid, tp.block, args, tp.shared_bytes, stream);
    }

  public:
    DslashArg<Float> &dslashParam; // temporary addition for policy compatibility

    Dslash(DslashArg<Float> &arg, const ColorSpinorField &in)
      : TunableVectorY(arg.nParity), arg(arg), in(in), nDimComms(4), dslashParam(arg)
    {
      // this sets the communications pattern for the packing kernel
      setPackComms(arg.commDim);

      if (!init) { // these parameters are constant across all dslash instances for a given run
        char ghost[5]; // set the ghost string
        for (int dim=0; dim<nDimComms; dim++) ghost[dim] = (comm_dim_partitioned(dim) ? '1' : '0');
        ghost[4] = '\0';
        strcpy(ghost_str,",ghost=");
        strcat(ghost_str,ghost);
        init = true;
      }

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

    int Nface() const { return 2; }
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
