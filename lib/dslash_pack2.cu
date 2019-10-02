#include <color_spinor_field.h>

// STRIPED - spread the blocks throughout the workload to ensure we
// work on all directions/dimensions simultanesouly to maximize NVLink saturation
#define STRIPED
// if not STRIPED then this means we assign one thread block per direction / dimension

#include <dslash_quda.h>
#include <kernels/dslash_pack.cuh>

namespace quda
{

  static int commDim[QUDA_MAX_DIM];

  int* getPackComms() { return commDim; }

  void setPackComms(const int *comm_dim)
  {
    for (int i = 0; i < 4; i++) commDim[i] = comm_dim[i];
    for (int i = 4; i < QUDA_MAX_DIM; i++) commDim[i] = 0;
  }

  template <typename Float, int nSpin, int nColor, bool spin_project>
  std::ostream &operator<<(std::ostream &out, const PackArg<Float, nSpin, nColor, spin_project> &arg)
  {
    out << "parity = " << arg.parity << std::endl;
    out << "nParity = " << arg.nParity << std::endl;
    out << "pc_type = " << arg.pc_type << std::endl;
    out << "nFace = " << arg.nFace << std::endl;
    out << "dagger = " << arg.dagger << std::endl;
    out << "a = " << arg.a << std::endl;
    out << "b = " << arg.b << std::endl;
    out << "c = " << arg.c << std::endl;
    out << "twist = " << arg.twist << std::endl;
    out << "threads = " << arg.threads << std::endl;
    out << "threadDimMapLower = { ";
    for (int i = 0; i < 4; i++) out << arg.threadDimMapLower[i] << (i < 3 ? ", " : " }");
    out << std::endl;
    out << "threadDimMapUpper = { ";
    for (int i = 0; i < 4; i++) out << arg.threadDimMapUpper[i] << (i < 3 ? ", " : " }");
    out << std::endl;
    out << "sites_per_block = " << arg.sites_per_block << std::endl;
    return out;
  }

  // FIXME - add CPU variant

  template <typename Float, int nColor, bool spin_project> class Pack : TunableVectorYZ
  {

protected:
    void **ghost;
    const ColorSpinorField &in;
    MemoryLocation location;
    const int nFace;
    const bool dagger; // only has meaning for nSpin=4
    const int parity;
    const int nParity;
    int threads;
    const double a;
    const double b;
    const double c;
    int twist; // only has meaning for nSpin=4

    bool tuneGridDim() const { return true; } // If striping, always tune grid dimension

    unsigned int maxGridSize() const
    {
      if (location & Host) {
#ifdef STRIPED
        // if zero-copy policy then set a maximum number of blocks to be
        // the 3 * number of dimensions we are communicating
        int max = 3;
#else
        // if zero-copy policy then assign exactly up to four thread blocks
        // per direction per dimension (effectively no grid-size tuning)
        int max = 2 * 4;
#endif
        int nDimComms = 0;
        for (int d = 0; d < in.Ndim(); d++) nDimComms += commDim[d];
        return max * nDimComms;
      } else {
        return TunableVectorYZ::maxGridSize();
      }
    } // use no more than a quarter of the GPU

    unsigned int minGridSize() const
    {
      if (location & Host) {
#ifdef STRIPED
        // if zero-copy policy then set a minimum number of blocks to be
        // the 1 * number of dimensions we are communicating
        int min = 3;
#else
        // if zero-copy policy then assign exactly one thread block
        // per direction per dimension (effectively no grid-size tuning)
        int min = 2;
#endif
        int nDimComms = 0;
        for (int d = 0; d < in.Ndim(); d++) nDimComms += commDim[d];
        return min * nDimComms;
      } else {
        return TunableVectorYZ::minGridSize();
      }
    }

    int gridStep() const
    {
#ifdef STRIPED
      return TunableVectorYZ::gridStep();
#else
      if (location & Host) {
        // the shmem kernel must ensure the grid size autotuner
        // increments in steps of 2 * number partitioned dimensions
        // for equal division of blocks to each direction/dimension
        int nDimComms = 0;
        for (int d = 0; d < in.Ndim(); d++) nDimComms += commDim[d];
        return 2 * nDimComms;
      } else {
        return TunableVectorYZ::gridStep();
      }
#endif
    }

    bool tuneAuxDim() const { return true; } // Do tune the aux dimensions.
    unsigned int minThreads() const { return threads; }

    void fillAux()
    {
      strcpy(aux, "policy_kernel,");
      strcat(aux, in.AuxString());
      char comm[5];
      for (int i = 0; i < 4; i++) comm[i] = (commDim[i] ? '1' : '0');
      comm[4] = '\0';
      strcat(aux, ",comm=");
      strcat(aux, comm);
      strcat(aux, comm_dim_topology_string());
      if (in.PCType() == QUDA_5D_PC) { strcat(aux, ",5D_pc"); }
      if (dagger && in.Nspin() == 4) { strcat(aux, ",dagger"); }
      if (getKernelPackT()) { strcat(aux, ",kernelPackT"); }
      switch (nFace) {
      case 1: strcat(aux, ",nFace=1"); break;
      case 3: strcat(aux, ",nFace=3"); break;
      default: errorQuda("Number of faces not supported");
      }

      twist = ((b != 0.0) ? (c != 0.0 ? 2 : 1) : 0);
      if (twist && a == 0.0) errorQuda("Twisted packing requires non-zero scale factor a");
      if (twist) strcat(aux, twist == 2 ? ",twist-doublet" : ",twist-singlet");

#ifndef STRIPED
      if (location & Host) strcat(aux, ",shmem");
#endif

      // label the locations we are packing to
      // location label is nonp2p-p2p
      switch ((int)location) {
      case Device | Remote: strcat(aux, ",device-remote"); break;
      case Host | Remote: strcat(aux, ",host-remote"); break;
      case Device: strcat(aux, ",device-device"); break;
      case Host: strcat(aux, comm_peer2peer_enabled_global() ? ",host-device" : ",host-host"); break;
      default: errorQuda("Unknown pack target location %d\n", location);
      }
    }

public:
    Pack(void *ghost[], const ColorSpinorField &in, MemoryLocation location, int nFace, bool dagger, int parity,
        double a, double b, double c) :
        TunableVectorYZ((in.Ndim() == 5 ? in.X(4) : 1), in.SiteSubset()),
        ghost(ghost),
        in(in),
        location(location),
        nFace(nFace),
        dagger(dagger),
        parity(parity),
        nParity(in.SiteSubset()),
        threads(0),
        a(a),
        b(b),
        c(c)
    {
      fillAux();

      // compute number of threads - really number of active work items we have to do
      for (int i = 0; i < 4; i++) {
        if (!commDim[i]) continue;
        if (i == 3 && !getKernelPackT()) continue;
        threads += 2 * nFace * in.getDslashConstant().ghostFaceCB[i]; // 2 for forwards and backwards faces
      }
    }

    virtual ~Pack() {}

    template <typename T, typename Arg>
    inline void launch(T *f, const TuneParam &tp, Arg &arg, const qudaStream_t &stream)
    {
      if (deviceProp.major >= 7) { // enable max shared memory mode on GPUs that support it
        this->setMaxDynamicSharedBytesPerBlock(f);
      }

      void *args[] = {&arg};
      qudaLaunchKernel((const void *)f, tp.grid, tp.block, args, tp.shared_bytes, stream);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (in.Nspin() == 4) {
        using Arg = PackArg<Float, nColor, 4, spin_project>;
        Arg arg(ghost, in, nFace, dagger, parity, threads, a, b, c);
        arg.swizzle = tp.aux.x;
        arg.sites_per_block = (arg.threads + tp.grid.x - 1) / tp.grid.x;
        arg.blocks_per_dir = tp.grid.x / (2 * arg.active_dims); // set number of blocks per direction

#ifdef STRIPED
        if (in.PCType() == QUDA_4D_PC) {
          if (arg.dagger) {
            switch (arg.twist) {
            case 0: launch(packKernel<true, 0, QUDA_4D_PC, Arg>, tp, arg, stream); break;
            case 1: launch(packKernel<true, 1, QUDA_4D_PC, Arg>, tp, arg, stream); break;
            case 2: launch(packKernel<true, 2, QUDA_4D_PC, Arg>, tp, arg, stream); break;
            }
          } else {
            switch (arg.twist) {
            case 0: launch(packKernel<false, 0, QUDA_4D_PC, Arg>, tp, arg, stream); break;
            default: errorQuda("Twisted packing only for dagger");
            }
          }
        } else if (arg.pc_type == QUDA_5D_PC) {
          if (arg.twist) errorQuda("Twist packing not defined");
          if (arg.dagger) {
            launch(packKernel<true, 0, QUDA_5D_PC, Arg>, tp, arg, stream);
          } else {
            launch(packKernel<false, 0, QUDA_5D_PC, Arg>, tp, arg, stream);
          }
        } else {
          errorQuda("Unexpected preconditioning type %d", in.PCType());
        }
#else
        if (in.PCType() == QUDA_4D_PC) {
          if (arg.dagger) {
            switch (arg.twist) {
            case 0:
              launch(location & Host ? packShmemKernel<true, 0, QUDA_4D_PC, Arg> : packKernel<true, 0, QUDA_4D_PC, Arg>,
                  tp, arg, stream);
              break;
            case 1:
              launch(location & Host ? packShmemKernel<true, 1, QUDA_4D_PC, Arg> : packKernel<true, 0, QUDA_4D_PC, Arg>,
                  tp, arg, stream);
              break;
            case 2:
              launch(location & Host ? packShmemKernel<true, 2, QUDA_4D_PC, Arg> : packKernel<true, 2, QUDA_4D_PC, Arg>,
                  tp, arg, stream);
              break;
            }
          } else {
            switch (arg.twist) {
            case 0:
              launch(location & Host ? packShmemKernel<false, 0, QUDA_4D_PC, Arg> : packKernel<false, 0, QUDA_4D_PC, Arg>,
                  tp, arg, stream);
              break;
            default: errorQuda("Twisted packing only for dagger");
            }
          }
        } else if (arg.pc_type == QUDA_5D_PC) {
          if (arg.twist) errorQuda("Twist packing not defined");
          if (arg.dagger) {
            launch(packKernel<true, 0, QUDA_5D_PC, Arg>, tp, arg, stream);
          } else {
            launch(packKernel<false, 0, QUDA_5D_PC, Arg>, tp, arg, stream);
          }
        }
#endif
      } else if (in.Nspin() == 1) {
        using Arg = PackArg<Float, nColor, 1, false>;
        Arg arg(ghost, in, nFace, dagger, parity, threads, a, b, c);
        arg.swizzle = tp.aux.x;
        arg.sites_per_block = (arg.threads + tp.grid.x - 1) / tp.grid.x;
        arg.blocks_per_dir = tp.grid.x / (2 * arg.active_dims); // set number of blocks per direction

#ifdef STRIPED
        launch(packStaggeredKernel<Arg>, tp, arg, stream);
#else
        launch(location & Host ? packStaggeredShmemKernel<Arg> : packStaggeredKernel<Arg>, tp, arg, stream);
#endif
      } else {
        errorQuda("Unsupported nSpin = %d\n", in.Nspin());
      }
    }

    bool tuneSharedBytes() const { return false; }

#if 0
    // not used at present, but if tuneSharedBytes is enabled then
    // this allows tuning up the full dynamic shared memory if needed
    unsigned int maxSharedBytesPerBlock() const { return maxDynamicSharedBytesPerBlock(); }
#endif

    void initTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::initTuneParam(param);
      // if doing a zero-copy policy then ensure that each thread block
      // runs exclusively on a given SM - this is to ensure quality of
      // service for the packing kernel when running concurrently.
      if (location & Host) param.shared_bytes = maxDynamicSharedBytesPerBlock() / 2 + 1;
#ifndef STRIPED
      if (location & Host) param.grid.x = minGridSize();
#endif
    }

    void defaultTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::defaultTuneParam(param);
      // if doing a zero-copy policy then ensure that each thread block
      // runs exclusively on a given SM - this is to ensure quality of
      // service for the packing kernel when running concurrently.
      if (location & Host) param.shared_bytes = maxDynamicSharedBytesPerBlock() / 2 + 1;
#ifndef STRIPED
      if (location & Host) param.grid.x = minGridSize();
#endif
    }

    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), aux); }

    int tuningIter() const { return 3; }

    long long flops() const
    {
      // unless we are spin projecting (nSpin = 4), there are no flops to do
      return in.Nspin() == 4 ? 2 * in.Nspin() / 2 * nColor * nParity * in.getDslashConstant().Ls * threads : 0;
    }

    long long bytes() const
    {
      size_t precision = sizeof(Float);
      size_t faceBytes = 2 * ((in.Nspin() == 4 ? in.Nspin() / 2 : in.Nspin()) + in.Nspin()) * nColor * precision;
      if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
        faceBytes += 2 * sizeof(float); // 2 is from input and output
      return faceBytes * nParity * in.getDslashConstant().Ls * threads;
    }
  };

  template <typename Float, int nColor>
  void PackGhost(void *ghost[], const ColorSpinorField &in, MemoryLocation location, int nFace, bool dagger, int parity,
                 bool spin_project, double a, double b, double c, const qudaStream_t &stream)
  {
    if (spin_project) {
      Pack<Float, nColor, true> pack(ghost, in, location, nFace, dagger, parity, a, b, c);
      pack.apply(stream);
    } else {
      Pack<Float, nColor, false> pack(ghost, in, location, nFace, dagger, parity, a, b, c);
      pack.apply(stream);
    }
  }

  // template on the number of colors
  template <typename Float>
  void PackGhost(void *ghost[], const ColorSpinorField &in, MemoryLocation location, int nFace, bool dagger, int parity,
                 bool spin_project, double a, double b, double c, const qudaStream_t &stream)
  {
    if (in.Ncolor() == 3) {
      PackGhost<Float, 3>(ghost, in, location, nFace, dagger, parity, spin_project, a, b, c, stream);
    } else {
      errorQuda("Unsupported number of colors %d\n", in.Ncolor());
    }
  }

  // Pack the ghost for the Dslash operator
  void PackGhost(void *ghost[2 * QUDA_MAX_DIM], const ColorSpinorField &in, MemoryLocation location, int nFace,
                 bool dagger, int parity, bool spin_project, double a, double b, double c, const qudaStream_t &stream)
  {
    int nDimPack = 0;
    for (int d = 0; d < 4; d++) {
      if (!commDim[d]) continue;
      if (d != 3 || getKernelPackT()) nDimPack++;
    }

    if (!nDimPack) return; // if zero then we have nothing to pack

    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
#if QUDA_PRECISION & 8
      PackGhost<double>(ghost, in, location, nFace, dagger, parity, spin_project, a, b, c, stream);
#else
      errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
#endif
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
      PackGhost<float>(ghost, in, location, nFace, dagger, parity, spin_project, a, b, c, stream);
#else
      errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
      PackGhost<short>(ghost, in, location, nFace, dagger, parity, spin_project, a, b, c, stream);
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
      PackGhost<char>(ghost, in, location, nFace, dagger, parity, spin_project, a, b, c, stream);
#else
      errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
    } else {
      errorQuda("Unsupported precision %d\n", in.Precision());
    }
  }

} // namespace quda
