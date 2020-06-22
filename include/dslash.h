#pragma once

#include <typeinfo>

#include <color_spinor_field.h>
#include <tune_quda.h>
#include <dslash_quda.h>
#include <dslash_helper.cuh>
#include <jitify_helper.cuh>
#include <instantiate.h>

namespace quda
{

  /**
     @brief This is the generic driver for launching Dslash kernels
     (the base kernel of which is defined in dslash_helper.cuh).  This
     is templated on the a template template parameter which is the
     underlying operator wrapped in a class,

     @tparam D A class that defines the linear operator we wish to
     apply.  This class should define an operator() method that is
     used to apply the operator by the dslash kernel.  See the wilson
     class in the file kernels/dslash_wilson.cuh as an exmaple.

     @tparam Arg The argument struct that is used to parameterize the
     kernel.  For the wilson class example above, the WilsonArg class
     defined in the same file is the corresponding argument class.
  */
  template <template <int, bool, bool, KernelType, typename> class D, typename Arg>
  class Dslash : public TunableVectorYZ
  {

  protected:
    Arg &arg;
    const ColorSpinorField &out;
    const ColorSpinorField &in;

    const int nDimComms;

    char aux_base[TuneKey::aux_n - 32];
    char aux[8][TuneKey::aux_n];
    char aux_pack[TuneKey::aux_n];

    // pointers to ghost buffers we are packing to
    void *packBuffer[2 * QUDA_MAX_DIM];

    std::string kernel_file;
    /**
       @brief Set the base strings used by the different dslash kernel
       types for autotuning.
    */
    inline void fillAuxBase()
    {
      char comm[5];
      comm[0] = (arg.commDim[0] ? '1' : '0');
      comm[1] = (arg.commDim[1] ? '1' : '0');
      comm[2] = (arg.commDim[2] ? '1' : '0');
      comm[3] = (arg.commDim[3] ? '1' : '0');
      comm[4] = '\0';
      strcpy(aux_base, ",commDim=");
      strcat(aux_base, comm);

      if (arg.xpay) strcat(aux_base, ",xpay");
      if (arg.dagger) strcat(aux_base, ",dagger");
    }

    /**
       @brief Specialize the auxiliary strings for each kernel type
       @param[in] kernel_type The kernel_type we are generating the string got
       @param[in] kernel_str String corresponding to the kernel type
    */
    inline void fillAux(KernelType kernel_type, const char *kernel_str)
    {
      strcpy(aux[kernel_type], kernel_str);
      if (kernel_type == INTERIOR_KERNEL) strcat(aux[kernel_type], comm_dim_partitioned_string());
      strncat(aux[kernel_type], aux_base, TuneKey::aux_n - 1);
    }

    virtual bool tuneGridDim() const { return false; }
    virtual unsigned int minThreads() const { return arg.threads; }

    inline void setParam(TuneParam &tp)
    {
      arg.t_proj_scale = getKernelPackT() ? 1.0 : 2.0;

      // Need to reset ghost pointers prior to every call since the
      // ghost buffer may have been changed during policy tuning.
      // Also, the accessor constructor calls Ghost(), which uses
      // ghost_buf, but this is only presently set with the
      // synchronous exchangeGhost.
      static void *ghost[8] = {}; // needs to be persistent across interior and exterior calls
      for (int dim = 0; dim < 4; dim++) {

        for (int dir = 0; dir < 2; dir++) {
          // if doing interior kernel, then this is the initial call,
          // so we set all ghost pointers else if doing exterior
          // kernel, then we only have to update the non-p2p ghosts,
          // since these may have been assigned to zero-copy memory
          if (!comm_peer2peer_enabled(dir, dim) || arg.kernel_type == INTERIOR_KERNEL) {
            ghost[2 * dim + dir]
              = (typename Arg::Float *)((char *)in.Ghost2() + in.GhostOffset(dim, dir) * in.GhostPrecision());
          }
        }
      }

      arg.in.resetGhost(in, ghost);

      if (arg.pack_threads && arg.kernel_type == INTERIOR_KERNEL) {
        arg.blocks_per_dir = tp.aux.x;
        arg.setPack(true); // need to recompute for updated block_per_dir
        arg.in.resetGhost(in, this->packBuffer);
        tp.grid.x += arg.pack_blocks;
      }
    }

    virtual int tuningIter() const { return 10; }

    virtual int blockStep() const { return 16; }
    virtual int blockMin() const { return 16; }

    unsigned int maxSharedBytesPerBlock() const { return maxDynamicSharedBytesPerBlock(); }

    virtual bool advanceAux(TuneParam &param) const
    {
      if (arg.pack_threads && arg.kernel_type == INTERIOR_KERNEL) {
        // if doing the fused kernel we tune how many blocks to use for
        // communication
        constexpr int max_blocks_per_dir = 4;
        if (param.aux.x + 1 <= max_blocks_per_dir) {
          param.aux.x++;
          return true;
        } else {
          param.aux.x = 1;
          return false;
        }
      } else {
        return false;
      }
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::initTuneParam(param);
      if (arg.pack_threads && arg.kernel_type == INTERIOR_KERNEL) param.aux.x = 1; // packing blocks per direction
    }

    virtual void defaultTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::defaultTuneParam(param);
      if (arg.pack_threads && arg.kernel_type == INTERIOR_KERNEL) param.aux.x = 1; // packing blocks per direction
    }

    template <typename T> inline void launch(T *f, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (deviceProp.major >= 7) { // should test whether this is always optimal on Volta
        this->setMaxDynamicSharedBytesPerBlock(f);
      }
      void *args[] = {&arg};
      qudaLaunchKernel((const void *)f, tp.grid, tp.block, args, tp.shared_bytes, stream);
    }

    /**
       @brief This is a helper class that is used to instantiate the
       correct templated kernel for the dslash.  This can be used for
       all dslash types, though in some cases we specialize to reduce
       compilation time.
    */
    template <template <bool, QudaPCType, typename> class P, int nParity, bool dagger, bool xpay, KernelType kernel_type>
    inline void Launch(TuneParam &tp, const qudaStream_t &stream)
    {
      launch(dslashGPU<D, P, nParity, dagger, xpay, kernel_type, Arg>, tp, stream);
    }

#ifdef JITIFY
    /**
       @brief Return a jitify kernel instance
    */
    template <template <bool, QudaPCType, typename> class P> auto kernel_instance()
    {
      if (!program) errorQuda("Jitify program has not been created");
      using namespace jitify::reflection;
      const auto kernel = "quda::dslashGPU";

      // we need this hackery to get the naked unbound template class parameters
      auto D_instance = reflect<D<0, false, false, INTERIOR_KERNEL, Arg>>();
      auto D_naked = D_instance.substr(0, D_instance.find("<"));
      auto P_instance = reflect<P<false, QUDA_4D_PC, Arg>>();
      auto P_naked = P_instance.substr(0, P_instance.find("<"));

      // Since we pass the operator and packer classes as strings to
      // jitify, we need to handle the reflection for all other
      // template parameters here as well as opposed to leaving this
      // to jitify.
      auto instance = program->kernel(kernel).instantiate({D_naked, P_naked, reflect(arg.nParity), reflect(arg.dagger),
                                                           reflect(arg.xpay), reflect(arg.kernel_type), reflect<Arg>()});

      return instance;
    }
#endif

  public:
    /**
       @brief This instantiate function is used to instantiate the
       the KernelType template required for the multi-GPU dslash kernels.
       @param[in] tp The tuning parameters to use for this kernel
       @param[in] stream The qudaStream_t where the kernel will run
     */
    template <template <bool, QudaPCType, typename> class P, int nParity, bool dagger, bool xpay>
    inline void instantiate(TuneParam &tp, const qudaStream_t &stream)
    {
      if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
        errorQuda("Not implemented");
      } else {
#ifdef JITIFY
        Tunable::jitify_error = kernel_instance<P>().configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
        switch (arg.kernel_type) {
        case INTERIOR_KERNEL: Launch<P, nParity, dagger, xpay, INTERIOR_KERNEL>(tp, stream); break;
#ifdef MULTI_GPU
        case EXTERIOR_KERNEL_X: Launch<P, nParity, dagger, xpay, EXTERIOR_KERNEL_X>(tp, stream); break;
        case EXTERIOR_KERNEL_Y: Launch<P, nParity, dagger, xpay, EXTERIOR_KERNEL_Y>(tp, stream); break;
        case EXTERIOR_KERNEL_Z: Launch<P, nParity, dagger, xpay, EXTERIOR_KERNEL_Z>(tp, stream); break;
        case EXTERIOR_KERNEL_T: Launch<P, nParity, dagger, xpay, EXTERIOR_KERNEL_T>(tp, stream); break;
        case EXTERIOR_KERNEL_ALL: Launch<P, nParity, dagger, xpay, EXTERIOR_KERNEL_ALL>(tp, stream); break;
        default: errorQuda("Unexpected kernel type %d", arg.kernel_type);
#else
        default: errorQuda("Unexpected kernel type %d for single-GPU build", arg.kernel_type);
#endif
        }
#endif // JITIFY
      }
    }

    /**
       @brief This instantiate function is used to instantiate the
       the dagger template
       @param[in] tp The tuning parameters to use for this kernel
       @param[in] stream The qudaStream_t where the kernel will run
     */
    template <template <bool, QudaPCType, typename> class P, int nParity, bool xpay>
    inline void instantiate(TuneParam &tp, const qudaStream_t &stream)
    {
#ifdef JITIFY
      Tunable::jitify_error = kernel_instance<P>().configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      if (arg.dagger)
        instantiate<P, nParity, true, xpay>(tp, stream);
      else
        instantiate<P, nParity, false, xpay>(tp, stream);
#endif
    }

    /**
       @brief This instantiate function is used to instantiate the
       the nParity template
       @param[in] tp The tuning parameters to use for this kernel
       @param[in] stream The qudaStream_t where the kernel will run
     */
    template <template <bool, QudaPCType, typename> class P, bool xpay>
    inline void instantiate(TuneParam &tp, const qudaStream_t &stream)
    {
#ifdef JITIFY
      Tunable::jitify_error = kernel_instance<P>().configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      switch (arg.nParity) {
      case 1: instantiate<P, 1, xpay>(tp, stream); break;
      case 2: instantiate<P, 2, xpay>(tp, stream); break;
      default: errorQuda("nParity = %d undefined\n", arg.nParity);
      }
#endif
    }

    /**
       @brief This instantiate function is used to instantiate the
       the xpay template
       @param[in] tp The tuning parameters to use for this kernel
       @param[in] stream The qudaStream_t where the kernel will run
     */
    template <template <bool, QudaPCType, typename> class P>
    inline void instantiate(TuneParam &tp, const qudaStream_t &stream)
    {
#ifdef JITIFY
      Tunable::jitify_error = kernel_instance<P>().configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      if (arg.xpay)
        instantiate<P, true>(tp, stream);
      else
        instantiate<P, false>(tp, stream);
#endif
    }

    Arg &dslashParam; // temporary addition for policy compatibility

    Dslash(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
      TunableVectorYZ(1, arg.nParity),
      arg(arg),
      out(out),
      in(in),
      nDimComms(4),
      dslashParam(arg)
    {
      if (checkLocation(out, in) == QUDA_CPU_FIELD_LOCATION)
        errorQuda("CPU Fields not supported in Dslash framework yet");

      // this sets the communications pattern for the packing kernel
      setPackComms(arg.commDim);

      // strcpy(aux, in.AuxString());
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

      // extract the filename from the template template class (do
      // this regardless of jitify to ensure a build error if filename
      // helper isn't defined)
      using D_ = D<0, false, false, INTERIOR_KERNEL, Arg>;
      kernel_file = std::string("kernels/") + D_::filename();
#ifdef JITIFY
      create_jitify_program(kernel_file);
#endif
    }

    void setPack(bool pack, MemoryLocation location)
    {
      arg.setPack(pack);
      if (!pack) return;

      for (int dim = 0; dim < 4; dim++) {
        for (int dir = 0; dir < 2; dir++) {
          if ((location & Remote) && comm_peer2peer_enabled(dir, dim)) { // pack to p2p remote
            packBuffer[2 * dim + dir]
              = static_cast<char *>(in.remoteFace_d(dir, dim)) + in.Precision() * in.GhostOffset(dim, 1 - dir);
          } else if (location & Host && !comm_peer2peer_enabled(dir, dim)) { // pack to cpu memory
            packBuffer[2 * dim + dir] = in.myFace_hd(dir, dim);
          } else { // pack to local gpu memory
            packBuffer[2 * dim + dir] = in.myFace_d(dir, dim);
          }
        }
      }

      // set the tuning string for the fused interior + packer kernel
      strcpy(aux_pack, aux[arg.kernel_type]);
      strcat(aux_pack, "");

      // label the locations we are packing to
      // location label is nonp2p-p2p
      switch ((int)location) {
      case Device | Remote: strcat(aux_pack, ",device-remote"); break;
      case Host | Remote: strcat(aux_pack, ",host-remote"); break;
      case Device: strcat(aux_pack, ",device-device"); break;
      case Host: strcat(aux_pack, comm_peer2peer_enabled_global() ? ",host-device" : ",host-host"); break;
      default: errorQuda("Unknown pack target location %d\n", location);
      }
    }

    int Nface() const
    {
      return 2 * arg.nFace;
    } // factor of 2 is for forwards/backwards (convention used in dslash policy)
    int Dagger() const { return arg.dagger; }

    const char *getAux(KernelType type) const { return aux[type]; }

    void setAux(KernelType type, const char *aux_) { strcpy(aux[type], aux_); }

    void augmentAux(KernelType type, const char *extra) { strcat(aux[type], extra); }

    virtual TuneKey tuneKey() const
    {
      auto aux_ = (arg.pack_blocks > 0 && arg.kernel_type == INTERIOR_KERNEL) ? aux_pack : aux[arg.kernel_type];
      return TuneKey(in.VolString(), typeid(*this).name(), aux_);
    }

    /**
       @brief Save the output field since the output field is both
       read from and written to in the exterior kernels
     */
    virtual void preTune()
    {
      if (arg.kernel_type != INTERIOR_KERNEL && arg.kernel_type != KERNEL_POLICY) out.backup();
    }

    /**
       @brief Restore the output field if doing exterior kernel
     */
    virtual void postTune()
    {
      if (arg.kernel_type != INTERIOR_KERNEL && arg.kernel_type != KERNEL_POLICY) out.restore();
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
      int ghost_flops = (num_mv_multiply * mv_flops + 2 * in.Ncolor() * in.Nspin());
      int xpay_flops = 2 * 2 * in.Ncolor() * in.Nspin(); // multiply and add per real component
      int num_dir = 2 * 4; // set to 4-d since we take care of 5-d fermions in derived classes where necessary
      int pack_flops = (in.Nspin() == 4 ? 2 * in.Nspin() / 2 * in.Ncolor() : 0); // only flops if spin projecting

      long long flops_ = 0;

      // FIXME - should we count the xpay flops in the derived kernels
      // since some kernels require the xpay in the exterior (preconditiond clover)

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        flops_ = (ghost_flops + (arg.xpay ? xpay_flops : xpay_flops / 2)) * 2 * in.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = 2 * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        flops_ = (ghost_flops + (arg.xpay ? xpay_flops : xpay_flops / 2)) * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
        if (arg.pack_threads) { flops_ += pack_flops * arg.nParity * in.getDslashConstant().Ls * arg.pack_threads; }
      case KERNEL_POLICY: {
        long long sites = in.Volume();
        flops_ = (num_dir * (in.Nspin() / 4) * in.Ncolor() * in.Nspin() + // spin project (=0 for staggered)
                  num_dir * num_mv_multiply * mv_flops +                  // SU(3) matrix-vector multiplies
                  ((num_dir - 1) * 2 * in.Ncolor() * in.Nspin()))
          * sites; // accumulation
        if (arg.xpay) flops_ += xpay_flops * sites;

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
        flops_ -= ghost_flops * ghost_sites;

        break;
      }
      }

      return flops_;
    }

    virtual long long bytes() const
    {
      int gauge_bytes = arg.reconstruct * in.Precision();
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() + (isFixed ? sizeof(float) : 0);
      int proj_spinor_bytes = in.Nspin() == 4 ? spinor_bytes / 2 : spinor_bytes;
      int ghost_bytes = (proj_spinor_bytes + gauge_bytes) + 2 * spinor_bytes; // 2 since we have to load the partial
      int num_dir = 2 * 4; // set to 4-d since we take care of 5-d fermions in derived classes where necessary
      int pack_bytes = 2 * ((in.Nspin() == 4 ? in.Nspin() / 2 : in.Nspin()) + in.Nspin()) * in.Ncolor() * in.Precision();

      long long bytes_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T: bytes_ = ghost_bytes * 2 * in.GhostFace()[arg.kernel_type]; break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = 2 * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        bytes_ = ghost_bytes * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
        if (arg.pack_threads) { bytes_ += pack_bytes * arg.nParity * in.getDslashConstant().Ls * arg.pack_threads; }
      case KERNEL_POLICY: {
        long long sites = in.Volume();
        bytes_ = (num_dir * gauge_bytes + ((num_dir - 2) * spinor_bytes + 2 * proj_spinor_bytes) + spinor_bytes) * sites;
        if (arg.xpay) bytes_ += spinor_bytes;

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
        bytes_ -= ghost_bytes * ghost_sites;

        break;
      }
      }
      return bytes_;
    }
  };

  /**
     @brief This instantiate function is used to instantiate the reconstruct types used
     @param[out] out Output result field
     @param[in] in Input field
     @param[in] U Gauge field
     @param[in] args Additional arguments for different dslash kernels
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon, typename Float, int nColor,
            typename... Args>
  inline void instantiate(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, Args &&... args)
  {
    if (U.Reconstruct() == Recon::recon[0]) {
#if QUDA_RECONSTRUCT & 4
      Apply<Float, nColor, Recon::recon[0]>(out, in, U, args...);
#else
      errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-18", QUDA_RECONSTRUCT);
#endif
    } else if (U.Reconstruct() == Recon::recon[1]) {
#if QUDA_RECONSTRUCT & 2
      Apply<Float, nColor, Recon::recon[1]>(out, in, U, args...);
#else
      errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12/13", QUDA_RECONSTRUCT);
#endif
    } else if (U.Reconstruct() == Recon::recon[2]) {
#if QUDA_RECONSTRUCT & 1
      Apply<Float, nColor, Recon::recon[2]>(out, in, U, args...);
#else
      errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8/9", QUDA_RECONSTRUCT);
#endif
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  /**
     @brief This instantiate function is used to instantiate the colors
     @param[out] out Output result field
     @param[in] in Input field
     @param[in] U Gauge field
     @param[in] args Additional arguments for different dslash kernels
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon, typename Float, typename... Args>
  inline void instantiate(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, Args &&... args)
  {
    if (in.Ncolor() == 3) {
      instantiate<Apply, Recon, Float, 3>(out, in, U, args...);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

  /**
     @brief This instantiate function is used to instantiate the precisions
     @param[out] out Output result field
     @param[in] in Input field
     @param[in] U Gauge field
     @param[in] args Additional arguments for different dslash kernels
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon = WilsonReconstruct, typename... Args>
  inline void instantiate(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, Args &&... args)
  {
    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
#if QUDA_PRECISION & 8
      instantiate<Apply, Recon, double>(out, in, U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
#endif
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
      instantiate<Apply, Recon, float>(out, in, U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
      instantiate<Apply, Recon, short>(out, in, U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
      instantiate<Apply, Recon, char>(out, in, U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
  }

  /**
     @brief This instantiatePrecondtiioner function is used to
     instantiate the precisions for a preconditioner.  This is the
     same as the instantiate helper above, except it only handles half
     and quarter precision.
     @param[out] out Output result field
     @param[in] in Input field
     @param[in] U Gauge field
     @param[in] args Additional arguments for different dslash kernels
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon = WilsonReconstruct, typename... Args>
  inline void instantiatePreconditioner(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                        Args &&... args)
  {
    if (U.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
      instantiate<Apply, Recon, short>(out, in, U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
      instantiate<Apply, Recon, char>(out, in, U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
  }

} // namespace quda
