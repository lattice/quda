#include <color_spinor_field.h>

// STRIPED - spread the blocks throughout the workload to ensure we
// work on all directions/dimensions simultanesouly to maximize NVLink saturation
// if not STRIPED then this means we assign one thread block per direction / dimension
// currently does not work with NVSHMEM
#ifndef NVSHMEM_COMMS
#define STRIPED 1
#endif

#include <dslash_quda.h>
#include <kernels/dslash_pack.cuh>
#include <tunable_nd.h>
#include <instantiate.h>

namespace quda
{

  static int comm_dim_pack[QUDA_MAX_DIM];

  int* getPackComms() { return comm_dim_pack; }

  void setPackComms(const int *comm_dim)
  {
    for (int i = 0; i < 4; i++) comm_dim_pack[i] = comm_dim[i];
    for (int i = 4; i < QUDA_MAX_DIM; i++) comm_dim_pack[i] = 0;
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
    out << "work_items = " << arg.work_items << std::endl;
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

  template <typename Float, int nColor, bool spin_project> class Pack : TunableKernel3D
  {

protected:
    void **ghost;
    const ColorSpinorField &in;
    MemoryLocation location;
    const int nFace;
    const bool dagger; // only has meaning for nSpin=4
    const int parity;
    const int nParity;
    int work_items;
    const double a;
    const double b;
    const double c;
    int twist; // only has meaning for nSpin=4
#ifdef NVSHMEM_COMMS
    const int shmem;
#else
    static constexpr int shmem = 0;
#endif

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
        for (int d = 0; d < in.Ndim(); d++) nDimComms += comm_dim_pack[d];
        return max * nDimComms;
      } else {
        return TunableKernel3D::maxGridSize();
      }
    } // use no more than a quarter of the GPU

    unsigned int minGridSize() const
    {
      if (location & Host || location & Shmem) {
#ifdef STRIPED
        // if zero-copy policy then set a minimum number of blocks to be
        // the 1 * number of dimensions we are communicating
        int min = 1;
#else
        // if zero-copy policy then assign exactly one thread block
        // per direction per dimension (effectively no grid-size tuning)
        int min = 2;
#endif
        int nDimComms = 0;
        for (int d = 0; d < in.Ndim(); d++) nDimComms += comm_dim_pack[d];
        return min * nDimComms;
      } else {
        return TunableKernel3D::minGridSize();
      }
    }

    int gridStep() const
    {
#ifdef STRIPED
      return TunableKernel3D::gridStep();
#else
      if (location & Host || location & Shmem) {
        // the shmem kernel must ensure the grid size autotuner
        // increments in steps of 2 * number partitioned dimensions
        // for equal division of blocks to each direction/dimension
        int nDimComms = 0;
        for (int d = 0; d < in.Ndim(); d++) nDimComms += comm_dim_pack[d];
        return 2 * nDimComms;
      } else {
        return TunableKernel3D::gridStep();
      }
#endif
    }

    unsigned int minThreads() const { return work_items; }

    void fillAux()
    {
      strcpy(aux, "policy_kernel,");
      strcat(aux, in.AuxString().c_str());
      char comm[5];
      for (int i = 0; i < 4; i++) comm[i] = (comm_dim_pack[i] ? '1' : '0');
      comm[4] = '\0';
      strcat(aux, ",comm=");
      strcat(aux, comm);
      strcat(aux, comm_dim_topology_string());
      if (in.PCType() == QUDA_5D_PC) { strcat(aux, ",5D_pc"); }
      if (dagger && in.Nspin() == 4) { strcat(aux, ",dagger"); }
      switch (nFace) {
      case 1: strcat(aux, ",nFace=1"); break;
      case 3: strcat(aux, ",nFace=3"); break;
      default: errorQuda("Number of faces not supported");
      }

      if (twist && a == 0.0) errorQuda("Twisted packing requires non-zero scale factor a");
      if (twist) strcat(aux, twist == 2 ? ",twist-doublet" : ",twist-singlet");

      // label the locations we are packing to
      // location label is nonp2p-p2p
      switch ((int)location) {
      case Device | Remote: strcat(aux, ",device-remote"); break;
      case Host | Remote: strcat(aux, ",host-remote"); break;
      case Device: strcat(aux, ",device-device"); break;
      case Host: strcat(aux, comm_peer2peer_enabled_global() ? ",host-device" : ",host-host"); break;
      case Shmem: strcat(aux, ",shmem"); break;
      default: errorQuda("Unknown pack target location %d\n", location);
      }
    }

public:
  Pack(void *ghost[], const ColorSpinorField &in, MemoryLocation location, int nFace, bool dagger, int parity, double a,
       double b, double c,
#ifdef NVSHMEM_COMMS
       int shmem) :
#else
       int) :
#endif
    TunableKernel3D(in, (in.Ndim() == 5 ? in.X(4) : 1), in.SiteSubset()),
    ghost(ghost),
    in(in),
    location(location),
    nFace(nFace),
    dagger(dagger),
    parity(parity),
    nParity(in.SiteSubset()),
    work_items(0),
    a(a),
    b(b),
    c(c),
    twist((b != 0.0) ? (c != 0.0 ? 2 : 1) : 0)
#ifdef NVSHMEM_COMMS
    ,
    shmem(shmem)
#endif
  {
    fillAux();

    // compute number of number of work items we have to do
    for (int i = 0; i < 4; i++) {
      if (!comm_dim_pack[i]) continue;
      work_items += 2 * nFace * in.getDslashConstant().ghostFaceCB[i]; // 2 for forwards and backwards faces
    }
  }

  template <int nSpin, bool dagger = false, int twist = 0, QudaPCType pc_type = QUDA_4D_PC> using Arg =
    PackArg<Float, nColor, nSpin, spin_project, dagger, twist, pc_type>;

  void apply(const qudaStream_t &stream)
  {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    // enable max shared memory mode on GPUs that support it
    tp.set_max_shared_bytes = true;

    if (in.Nspin() == 4) {

#ifdef STRIPED
        if (in.PCType() == QUDA_4D_PC) {
          if (dagger) {
            switch (twist) {
            case 0: launch_device<pack_wilson>(tp, stream, Arg<4, true, 0>(ghost, in, nFace, parity, work_items,
                                                                           a, b, c, tp.block.x, tp.grid.x, shmem));
              break;
            case 1: launch_device<pack_wilson>(tp, stream, Arg<4, true, 1>(ghost, in, nFace, parity, work_items,
                                                                           a, b, c, tp.block.x, tp.grid.x, shmem));
              break;
            case 2: launch_device<pack_wilson>(tp, stream, Arg<4, true, 2>(ghost, in, nFace, parity, work_items,
                                                                           a, b, c, tp.block.x, tp.grid.x, shmem));
              break;
            }
          } else {
            switch (twist) {
            case 0: launch_device<pack_wilson>(tp, stream, Arg<4, false, 0>(ghost, in, nFace, parity, work_items,
                                                                            a, b, c, tp.block.x, tp.grid.x, shmem));
              break;
            default: errorQuda("Twisted packing only for dagger");
            }
          }
        } else if (in.PCType() == QUDA_5D_PC) {
          if (twist) errorQuda("Twist packing not defined");
          if (dagger) {
            launch_device<pack_wilson>(tp, stream, Arg<4, true, 0, QUDA_5D_PC>(ghost, in, nFace, parity, work_items,
                                                                               a, b, c, tp.block.x, tp.grid.x, shmem));
          } else {
            launch_device<pack_wilson>(tp, stream, Arg<4, false, 0, QUDA_5D_PC>(ghost, in, nFace, parity, work_items,
                                                                                a, b, c, tp.block.x, tp.grid.x, shmem));
          }
        } else {
          errorQuda("Unexpected preconditioning type %d", in.PCType());
        }
#else
        if (in.PCType() == QUDA_4D_PC) {
          if (dagger) {
            switch (twist) {
            case 0:
              if (location & Host || location & Shmem)
                launch_device<pack_wilson_shmem>(tp, stream, Arg<4, true, 0>(ghost, in, nFace, parity, work_items,
                                                                             a, b, c, tp.block.x, tp.grid.x, shmem));
              else
                launch_device<pack_wilson>(tp, stream, Arg<4, true, 0>(ghost, in, nFace, parity, work_items,
                                                                       a, b, c, tp.block.x, tp.grid.x, shmem));
              break;
            case 1:
              if (location & Host || location & Shmem)
                launch_device<pack_wilson_shmem>(tp, stream, Arg<4, true, 1>(ghost, in, nFace, parity, work_items,
                                                                             a, b, c, tp.block.x, tp.grid.x, shmem));
              else
                launch_device<pack_wilson>(tp, stream, Arg<4, true, 1>(ghost, in, nFace, parity, work_items,
                                                                       a, b, c, tp.block.x, tp.grid.x, shmem));
              break;
            case 2:
              if (location & Host || location & Shmem)
                launch_device<pack_wilson_shmem>(tp, stream, Arg<4, true, 2>(ghost, in, nFace, parity, work_items,
                                                                             a, b, c, tp.block.x, tp.grid.x, shmem));
              else
                launch_device<pack_wilson>(tp, stream, Arg<4, true, 2>(ghost, in, nFace, parity, work_items,
                                                                       a, b, c, tp.block.x, tp.grid.x, shmem));
              break;
            }
          } else {
            switch (twist) {
            case 0:
              if (location & Host || location & Shmem)
                launch_device<pack_wilson_shmem>(tp, stream, Arg<4, false, 0>(ghost, in, nFace, parity, work_items,
                                                                              a, b, c, tp.block.x, tp.grid.x, shmem));
              else
                launch_device<pack_wilson>(tp, stream, Arg<4, false, 0>(ghost, in, nFace, parity, work_items,
                                                                        a, b, c, tp.block.x, tp.grid.x, shmem));
              break;
            default: errorQuda("Twisted packing only for dagger");
            }
          }
        } else if (in.PCType() == QUDA_5D_PC) {
          if (twist) errorQuda("Twist packing not defined");
          if (dagger) {
            launch_device<pack_wilson_shmem>(tp, stream, Arg<4, true, 0, QUDA_5D_PC>(ghost, in, nFace, parity, work_items,
                                                                                     a, b, c, tp.block.x, tp.grid.x, shmem));
          } else {
            launch_device<pack_wilson_shmem>(tp, stream, Arg<4, false, 0, QUDA_5D_PC>(ghost, in, nFace, parity, work_items,
                                                                                      a, b, c, tp.block.x, tp.grid.x, shmem));
          }
        }
#endif

    } else if (in.Nspin() == 1) {

#ifdef STRIPED
        launch_device<pack_staggered>(tp, stream, Arg<1>(ghost, in, nFace, parity, work_items,
                                                         a, b, c, tp.block.x, tp.grid.x, shmem));
#else
        if (location & Host || location & Shmem)
          launch_device<pack_staggered_shmem>(tp, stream, Arg<1>(ghost, in, nFace, parity, work_items,
                                                                 a, b, c, tp.block.x, tp.grid.x, shmem));
        else
          launch_device<pack_staggered>(tp, stream, Arg<1>(ghost, in, nFace, parity, work_items,
                                                           a, b, c, tp.block.x, tp.grid.x, shmem));
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
      TunableKernel3D::initTuneParam(param);
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
      TunableKernel3D::defaultTuneParam(param);
      // if doing a zero-copy policy then ensure that each thread block
      // runs exclusively on a given SM - this is to ensure quality of
      // service for the packing kernel when running concurrently.
      if (location & Host) param.shared_bytes = maxDynamicSharedBytesPerBlock() / 2 + 1;
#ifndef STRIPED
      if (location & Host) param.grid.x = minGridSize();
#endif
    }

    long long flops() const
    {
      // unless we are spin projecting (nSpin = 4), there are no flops to do
      return in.Nspin() == 4 ? 2 * in.Nspin() / 2 * nColor * nParity * in.getDslashConstant().Ls * work_items : 0;
    }

    long long bytes() const
    {
      size_t precision = sizeof(Float);
      size_t faceBytes = 2 * ((in.Nspin() == 4 ? in.Nspin() / 2 : in.Nspin()) + in.Nspin()) * nColor * precision;
      if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
        faceBytes += 2 * sizeof(float); // 2 is from input and output
      return faceBytes * nParity * in.getDslashConstant().Ls * work_items;
    }
  };

  template <typename Float, int nColor> struct GhostPack {
    GhostPack(const ColorSpinorField &in, void *ghost[], MemoryLocation location, int nFace, bool dagger, int parity,
              bool spin_project, double a, double b, double c, int shmem, const qudaStream_t &stream)
    {
      if (spin_project) {
        Pack<Float, nColor, true> pack(ghost, in, location, nFace, dagger, parity, a, b, c, shmem);
        pack.apply(stream);
      } else {
        Pack<Float, nColor, false> pack(ghost, in, location, nFace, dagger, parity, a, b, c, shmem);
        pack.apply(stream);
      }
    }
  };

  // Pack the ghost for the Dslash operator
  void PackGhost(void *ghost[2 * QUDA_MAX_DIM], const ColorSpinorField &in, MemoryLocation location, int nFace,
                 bool dagger, int parity, bool spin_project, double a, double b, double c, int shmem,
                 const qudaStream_t &stream)
  {
    int nDimPack = 0;
    for (int d = 0; d < 4; d++) {
      if (!comm_dim_pack[d]) continue;
      nDimPack++;
    }
    if (!nDimPack) return; // if zero then we have nothing to pack

    instantiate<GhostPack>(in, ghost, location, nFace, dagger, parity, spin_project, a, b, c, shmem, stream);
  }

} // namespace quda
