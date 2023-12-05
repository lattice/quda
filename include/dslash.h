#pragma once

#include <typeinfo>

#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <dslash_helper.cuh>
#include <tunable_nd.h>
#include <instantiate.h>
#include <instantiate_dslash.h>

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
  class Dslash : public TunableKernel3D
  {

  protected:
    Arg &arg;
    const ColorSpinorField &out;
    const ColorSpinorField &in;

    const int nDimComms;

    char aux_base[TuneKey::aux_n - 32];
    char aux[8][TuneKey::aux_n];
    char aux_pack[TuneKey::aux_n];
    char aux_barrier[TuneKey::aux_n];

    // pointers to ghost buffers we are packing to
    void *packBuffer[4 * QUDA_MAX_DIM];

    /**
       @brief Set the base strings used by the different dslash kernel
       types for autotuning.
    */
    inline void fillAuxBase(const std::string &app_base)
    {
      strcpy(aux_base, TunableKernel3D::aux);
      char comm[5];
      comm[0] = (arg.commDim[0] ? '1' : '0');
      comm[1] = (arg.commDim[1] ? '1' : '0');
      comm[2] = (arg.commDim[2] ? '1' : '0');
      comm[3] = (arg.commDim[3] ? '1' : '0');
      comm[4] = '\0';
      strcat(aux_base, ",commDim=");
      strcat(aux_base, comm);
      strcat(aux_base, app_base.c_str());

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
      strncat(aux[kernel_type], aux_base, TuneKey::aux_n - 1);
      if (kernel_type == INTERIOR_KERNEL) strcat(aux[kernel_type], comm_dim_partitioned_string());
    }

    virtual bool tuneGridDim() const override { return arg.kernel_type == EXTERIOR_KERNEL_ALL && arg.shmem > 0; }
    virtual unsigned int minThreads() const override { return arg.threads; }

    virtual unsigned int minGridSize() const override
    {
      /* when using nvshmem we perform the exterior Dslash using a grid strided loop and uniquely assign communication
       * directions to CUDA block and have all communication directions resident. We therefore figure out the number of
       * communicating dimensions and make sure that the number of blocks is a multiple of the communicating directions (2*dim)
       */
      if (arg.kernel_type == EXTERIOR_KERNEL_ALL && arg.shmem > 0) {
        int nDimComms = 0;
        for (int d = 0; d < 4; d++) nDimComms += arg.commDim[d];
        return (device::processor_count() / (2 * nDimComms)) * (2 * nDimComms);
      } else {
        return TunableKernel3D::minGridSize();
      }
    }

    virtual int gridStep() const override
    {
      /* see comment for minGridSize above for gridStep choice when using nvshmem */
      if (arg.kernel_type == EXTERIOR_KERNEL_ALL && arg.shmem > 0) {
        int nDimComms = 0;
        for (int d = 0; d < 4; d++) nDimComms += arg.commDim[d];
        return (device::processor_count() / (2 * nDimComms)) * (2 * nDimComms);
      } else {
        return TunableKernel3D::gridStep();
      }
    }

    inline void setParam(TuneParam &tp)
    {
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
          if (!comm_peer2peer_enabled(dir, dim) || arg.kernel_type == INTERIOR_KERNEL || arg.kernel_type == UBER_KERNEL) {
            ghost[2 * dim + dir] = (typename Arg::Float *)((char *)in.Ghost2() + in.GhostOffset(dim, dir));
          }
        }
      }

      arg.in.resetGhost(ghost);

      if (arg.pack_threads && (arg.kernel_type == INTERIOR_KERNEL || arg.kernel_type == UBER_KERNEL)) {
        arg.blocks_per_dir = tp.aux.x;
        arg.setPack(true, this->packBuffer); // need to recompute for updated block_per_dir
        arg.in_pack.resetGhost(this->packBuffer);
        tp.grid.x += arg.pack_blocks;
        arg.counter = dslash::get_dslash_shmem_sync_counter();
      }
      if (arg.shmem > 0 && arg.kernel_type == EXTERIOR_KERNEL_ALL) {
        // if we are doing tuning we should not wait on the sync_arr to be set.
        arg.counter = (activeTuning() && !policyTuning()) ? 2 : dslash::get_dslash_shmem_sync_counter();
      }
      if (arg.shmem > 0 && (arg.kernel_type == INTERIOR_KERNEL || arg.kernel_type == UBER_KERNEL)) {
        arg.counter = activeTuning() ? (uberTuning() && !policyTuning() ? dslash::inc_dslash_shmem_sync_counter() :
                                                                          dslash::get_dslash_shmem_sync_counter()) :
                                       dslash::get_dslash_shmem_sync_counter();
        arg.exterior_blocks = ((arg.shmem & 64) && arg.exterior_dims > 0) ?
          (device::processor_count() / (2 * arg.exterior_dims)) * (2 * arg.exterior_dims * tp.aux.y) :
          0;
        tp.grid.x += arg.exterior_blocks;
      }
    }

    virtual int blockStep() const override { return 16; }
    virtual int blockMin() const override { return 16; }

    unsigned int maxSharedBytesPerBlock() const override { return maxDynamicSharedBytesPerBlock(); }

    virtual bool advanceAux(TuneParam &param) const override
    {
      if (arg.pack_threads && (arg.kernel_type == INTERIOR_KERNEL || arg.kernel_type == UBER_KERNEL)) {

        int max_threads_per_dir = 0;
        for (int i = 0; i < 4; ++i) {
          max_threads_per_dir = std::max(max_threads_per_dir, (arg.threadDimMapUpper[i] - arg.threadDimMapLower[i]) / 2);
        }
        int nDimComms = 0;
        for (int d = 0; d < 4; d++) nDimComms += arg.commDim[d];

        /* if doing the fused packing + interior kernel we tune how many blocks to use for communication */
        // use up to a quarter of the GPU for packing (but at least up to 4 blocks per dir)
        const int max_blocks_per_dir = std::max(device::processor_count() / (8 * nDimComms), 4u);
        if (param.aux.x + 1 <= max_blocks_per_dir
            && (param.aux.x + 1) * param.block.x < (max_threads_per_dir + param.block.x - 1)) {
          param.aux.x++;
          return true;
        } else {
          param.aux.x = 1;
          if (arg.exterior_dims > 0 && arg.shmem & 64) {
            /* if doing a fused interior+exterior kernel we use aux.y to control the number of blocks we add for the
             * exterior. We make sure to use multiple blocks per communication direction.
             */
            if (param.aux.y < 4) {
              param.aux.y++;
              return true;
            } else {
              param.aux.y = 1;
              return false;
            }
          }
          return false;
        }
      } else {
        return false;
      }
    }

    virtual bool advanceTuneParam(TuneParam &param) const override
    {
      return advanceAux(param) || advanceSharedBytes(param) || advanceBlockDim(param) || advanceGridDim(param);
    }

    virtual void initTuneParam(TuneParam &param) const override
    {
      /* for nvshmem uber kernels the current synchronization requires use to keep the y and z dimension local to the
       * block. This can be removed when we introduce a finer grained synchronization which takes into account the y and
       * z components explicitly */
      if (arg.shmem & 64) {
        step_y = vector_length_y;
        step_z = vector_length_z;
      }
      TunableKernel3D::initTuneParam(param);
      if (arg.pack_threads && (arg.kernel_type == INTERIOR_KERNEL || arg.kernel_type == UBER_KERNEL))
        param.aux.x = 1;                                                        // packing blocks per direction
      if (arg.exterior_dims && arg.kernel_type == UBER_KERNEL) param.aux.y = 1; // exterior blocks
    }

    virtual void defaultTuneParam(TuneParam &param) const override
    {
      /* for nvshmem uber kernels the current synchronization requires use to keep the y and z dimension local to the
       * block. This can be removed when we introduce a finer grained synchronization which takes into account the y and
       * z components explicitly. */
      if (arg.shmem & 64) {
        step_y = vector_length_y;
        step_z = vector_length_z;
      }
      TunableKernel3D::defaultTuneParam(param);
      if (arg.pack_threads && (arg.kernel_type == INTERIOR_KERNEL || arg.kernel_type == UBER_KERNEL))
        param.aux.x = 1;                                                        // packing blocks per direction
      if (arg.exterior_dims && arg.kernel_type == UBER_KERNEL) param.aux.y = 1; // exterior blocks
    }

    /**
       @brief This is a helper class that is used to instantiate the
       correct templated kernel for the dslash.  This can be used for
       all dslash types, though in some cases we specialize to reduce
       compilation time.
    */
    template <template <bool, QudaPCType, typename> class P, int nParity, bool dagger, bool xpay, KernelType kernel_type>
    inline void launch(TuneParam &tp, const qudaStream_t &stream)
    {
      tp.set_max_shared_bytes = true;
      launch_device<dslash_functor>(
        tp, stream, dslash_functor_arg<D, P, nParity, dagger, xpay, kernel_type, Arg>(arg, tp.block.x * tp.grid.x));
    }

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
        switch (arg.kernel_type) {
        case INTERIOR_KERNEL: launch<P, nParity, dagger, xpay, INTERIOR_KERNEL>(tp, stream); break;
#ifdef MULTI_GPU
#ifdef NVSHMEM_COMMS
        case UBER_KERNEL: launch<P, nParity, dagger, xpay, UBER_KERNEL>(tp, stream); break;
#endif
        case EXTERIOR_KERNEL_X: launch<P, nParity, dagger, xpay, EXTERIOR_KERNEL_X>(tp, stream); break;
        case EXTERIOR_KERNEL_Y: launch<P, nParity, dagger, xpay, EXTERIOR_KERNEL_Y>(tp, stream); break;
        case EXTERIOR_KERNEL_Z: launch<P, nParity, dagger, xpay, EXTERIOR_KERNEL_Z>(tp, stream); break;
        case EXTERIOR_KERNEL_T: launch<P, nParity, dagger, xpay, EXTERIOR_KERNEL_T>(tp, stream); break;
        case EXTERIOR_KERNEL_ALL: launch<P, nParity, dagger, xpay, EXTERIOR_KERNEL_ALL>(tp, stream); break;
        default: errorQuda("Unexpected kernel type %d", arg.kernel_type);
#else
        default: errorQuda("Unexpected kernel type %d for single-GPU build", arg.kernel_type);
#endif
        }
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
      if (arg.dagger)
        instantiate<P, nParity, true, xpay>(tp, stream);
      else
        instantiate<P, nParity, false, xpay>(tp, stream);
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
      switch (arg.nParity) {
      case 1: instantiate<P, 1, xpay>(tp, stream); break;
      case 2: instantiate<P, 2, xpay>(tp, stream); break;
      default: errorQuda("nParity = %d undefined\n", arg.nParity);
      }
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
      if (arg.xpay)
        instantiate<P, true>(tp, stream);
      else
        instantiate<P, false>(tp, stream);
    }

    Arg &dslashParam; // temporary addition for policy compatibility

    Dslash(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in, const std::string &app_base = "") :
      TunableKernel3D(in, 1, arg.nParity), arg(arg), out(out), in(in), nDimComms(4), dslashParam(arg)
    {
      if (checkLocation(out, in) == QUDA_CPU_FIELD_LOCATION)
        errorQuda("CPU Fields not supported in Dslash framework yet");

      // this sets the communications pattern for the packing kernel
      setPackComms(arg.commDim);
      // strcpy(aux, in.AuxString().c_str());
      fillAuxBase(app_base);
#ifdef MULTI_GPU
      fillAux(INTERIOR_KERNEL, "policy_kernel=interior,");
      fillAux(UBER_KERNEL, "policy_kernel=uber,");
      fillAux(EXTERIOR_KERNEL_ALL, "policy_kernel=exterior_all,");
      fillAux(EXTERIOR_KERNEL_X, "policy_kernel=exterior_x,");
      fillAux(EXTERIOR_KERNEL_Y, "policy_kernel=exterior_y,");
      fillAux(EXTERIOR_KERNEL_Z, "policy_kernel=exterior_z,");
      fillAux(EXTERIOR_KERNEL_T, "policy_kernel=exterior_t,");
#else
      fillAux(INTERIOR_KERNEL, "policy_kernel=single,");
#endif // MULTI_GPU
      fillAux(KERNEL_POLICY, "policy,");

#ifdef NVSHMEM_COMMS
      strcpy(aux_barrier, aux[EXTERIOR_KERNEL_ALL]);
      strcat(aux_barrier, ",shmem");
#endif
    }

#ifdef NVSHMEM_COMMS
    void setShmem(int shmem)
    {
      arg.shmem = shmem;
      setUberTuning(arg.shmem & 64);
    }
#else
    void setShmem(int) { setUberTuning(arg.shmem & 64); }
#endif

    void setPack(bool pack, MemoryLocation location)
    {
      if (!pack) {
        arg.setPack(pack, packBuffer);
        return;
      }

      for (int dim = 0; dim < 4; dim++) {
        for (int dir = 0; dir < 2; dir++) {
          if ((location & Remote) && comm_peer2peer_enabled(dir, dim)) { // pack to p2p remote
            packBuffer[2 * dim + dir] = static_cast<char *>(in.remoteFace_d(dir, dim)) + in.GhostOffset(dim, 1 - dir);
          } else if (location & Host && !comm_peer2peer_enabled(dir, dim)) { // pack to cpu memory
            packBuffer[2 * dim + dir] = in.myFace_hd(dir, dim);
          } else if (location & Shmem) {
            // we check whether we can directly pack into the in.remoteFace_d(dir, dim) buffer on the remote GPU
            // pack directly into remote or local memory
            packBuffer[2 * dim + dir] = in.remoteFace_d(dir, dim) ?
              static_cast<char *>(in.remoteFace_d(dir, dim)) + in.GhostOffset(dim, 1 - dir) :
              in.myFace_d(dir, dim);
            // whether we need to shmem_putmem into the receiving buffer
            packBuffer[2 * QUDA_MAX_DIM + 2 * dim + dir] = in.remoteFace_d(dir, dim) ?
              nullptr :
              static_cast<char *>(in.remoteFace_r()) + in.GhostOffset(dim, 1 - dir);
          } else { // pack to local gpu memory
            packBuffer[2 * dim + dir] = in.myFace_d(dir, dim);
          }
        }
      }

      arg.setPack(pack, packBuffer);
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
      case Shmem:
        strcat(aux_pack, arg.exterior_dims > 0 ? ",shmemuber" : ",shmem");
        strcat(aux_pack, (arg.shmem & 1 && arg.shmem & 2) ? "3" : "1");
        break;

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

    virtual TuneKey tuneKey() const override
    {
      auto aux_ = (arg.pack_blocks > 0 && (arg.kernel_type == INTERIOR_KERNEL || arg.kernel_type == UBER_KERNEL)) ?
        aux_pack :
        ((arg.shmem > 0 && arg.kernel_type == EXTERIOR_KERNEL_ALL) ? aux_barrier : aux[arg.kernel_type]);
      return TuneKey(in.VolString().c_str(), typeid(*this).name(), aux_);
    }

    /**
       @brief Save the output field since the output field is both
       read from and written to in the exterior kernels
     */
    virtual void preTune() override
    {
      if (arg.kernel_type != INTERIOR_KERNEL && arg.kernel_type != UBER_KERNEL && arg.kernel_type != KERNEL_POLICY)
        out.backup();
    }

    /**
       @brief Restore the output field if doing exterior kernel
     */
    virtual void postTune() override
    {
      if (arg.kernel_type != INTERIOR_KERNEL && arg.kernel_type != UBER_KERNEL && arg.kernel_type != KERNEL_POLICY)
        out.restore();
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
    virtual long long flops() const override
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
      case UBER_KERNEL:
      case KERNEL_POLICY: {
        if (arg.pack_threads && (arg.kernel_type == INTERIOR_KERNEL || arg.kernel_type == UBER_KERNEL))
          flops_ += pack_flops * arg.nParity * in.getDslashConstant().Ls * arg.pack_threads;
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

        if (arg.kernel_type == INTERIOR_KERNEL && arg.pack_threads)
          flops_ += pack_flops * arg.nParity * in.getDslashConstant().Ls * arg.pack_threads;
        break;
      }
      }

      return flops_;
    }

    virtual long long bytes() const override
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
      case UBER_KERNEL:
      case KERNEL_POLICY: {
        if (arg.pack_threads && (arg.kernel_type == INTERIOR_KERNEL || arg.kernel_type == UBER_KERNEL))
          bytes_ += pack_bytes * arg.nParity * in.getDslashConstant().Ls * arg.pack_threads;
        long long sites = in.Volume();
        bytes_ = (num_dir * gauge_bytes + ((num_dir - 2) * spinor_bytes + 2 * proj_spinor_bytes) + spinor_bytes) * sites;
        if (arg.xpay) bytes_ += spinor_bytes;

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
        bytes_ -= ghost_bytes * ghost_sites;

        if (arg.kernel_type == INTERIOR_KERNEL && arg.pack_threads)
          bytes_ += pack_bytes * arg.nParity * in.getDslashConstant().Ls * arg.pack_threads;
        break;
      }
      }
      return bytes_;
    }
  };

} // namespace quda
