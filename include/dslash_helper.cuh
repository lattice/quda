#pragma once

#include <color_spinor_field.h>
#include <gauge_field.h>
#include <register_traits.h>
#include <index_helper.cuh>

namespace quda
{

  /**
     @brief Helper function to determine if we should do halo
     computation
     @param[in] dim Dimension we are working on.  If dim=-1 (default
     argument) then we return true if type is any halo kernel.
  */
  template <KernelType type> __host__ __device__ inline bool doHalo(int dim = -1)
  {
    switch (type) {
    case EXTERIOR_KERNEL_ALL: return true;
    case EXTERIOR_KERNEL_X: return dim == 0 || dim == -1 ? true : false;
    case EXTERIOR_KERNEL_Y: return dim == 1 || dim == -1 ? true : false;
    case EXTERIOR_KERNEL_Z: return dim == 2 || dim == -1 ? true : false;
    case EXTERIOR_KERNEL_T: return dim == 3 || dim == -1 ? true : false;
    case INTERIOR_KERNEL: return false;
    }
    return false;
  }

  /**
     @brief Helper function to determine if we should do interior
     computation
     @param[in] dim Dimension we are working on
  */
  template <KernelType type> __host__ __device__ inline bool doBulk()
  {
    switch (type) {
    case EXTERIOR_KERNEL_ALL:
    case EXTERIOR_KERNEL_X:
    case EXTERIOR_KERNEL_Y:
    case EXTERIOR_KERNEL_Z:
    case EXTERIOR_KERNEL_T: return false;
    case INTERIOR_KERNEL: return true;
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
  template <KernelType type, typename Arg, typename Coord> __host__ __device__ inline bool isComplete(const Arg &arg, const Coord &coord)
  {
    int incomplete = 0; // Have all 8 contributions been computed for this site?

    switch (type) {                                      // intentional fall-through
    case EXTERIOR_KERNEL_ALL: incomplete = false; break; // all active threads are complete
    case INTERIOR_KERNEL:
      incomplete = incomplete || (arg.commDim[3] && (coord[3] == 0 || coord[3] == (arg.dc.X[3] - 1)));
    case EXTERIOR_KERNEL_T:
      incomplete = incomplete || (arg.commDim[2] && (coord[2] == 0 || coord[2] == (arg.dc.X[2] - 1)));
    case EXTERIOR_KERNEL_Z:
      incomplete = incomplete || (arg.commDim[1] && (coord[1] == 0 || coord[1] == (arg.dc.X[1] - 1)));
    case EXTERIOR_KERNEL_Y:
      incomplete = incomplete || (arg.commDim[0] && (coord[0] == 0 || coord[0] == (arg.dc.X[0] - 1)));
    case EXTERIOR_KERNEL_X: break;
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
  template <QudaPCType pc_type, KernelType kernel_type, typename Arg, int nface_ = 1>
  __host__ __device__ inline auto getCoords(const Arg &arg, int &idx, int s, int parity, int &dim)
  {
    constexpr auto nDim = Arg::nDim;
    Coord<nDim> coord;
    dim = kernel_type; // keep compiler happy

    // only for 5-d checkerboarding where we need to include the fifth dimension
    const int Ls = (nDim == 5 && pc_type == QUDA_5D_PC ? (int)arg.dim[4] : 1);

    if (kernel_type == INTERIOR_KERNEL) {
      coord.x_cb = idx;
      if (nDim == 5)
        coord.X = getCoords5CB(coord, idx, arg.dim, arg.X0h, parity, pc_type);
      else
        coord.X = getCoordsCB(coord, idx, arg.dim, arg.X0h, parity);
    } else if (kernel_type != EXTERIOR_KERNEL_ALL) {

      // compute face index and then compute coords
      const int face_size = nface_ * arg.dc.ghostFaceCB[kernel_type] * Ls;
      const int face_num = idx >= face_size;
      idx -= face_num * face_size;
      coordsFromFaceIndex<nDim, pc_type, kernel_type, nface_>(coord.X, coord.x_cb, coord, idx, face_num, parity, arg);

    } else { // fused kernel

      // work out which dimension this thread corresponds to, then compute coords
      if (idx < arg.threadDimMapUpper[0] * Ls) { // x face
        dim = 0;
        const int face_size = nface_ * arg.dc.ghostFaceCB[dim] * Ls;
        const int face_num = idx >= face_size;
        idx -= face_num * face_size;
        coordsFromFaceIndex<nDim, pc_type, 0, nface_>(coord.X, coord.x_cb, coord, idx, face_num, parity, arg);
      } else if (idx < arg.threadDimMapUpper[1] * Ls) { // y face
        dim = 1;
        idx -= arg.threadDimMapLower[1] * Ls;
        const int face_size = nface_ * arg.dc.ghostFaceCB[dim] * Ls;
        const int face_num = idx >= face_size;
        idx -= face_num * face_size;
        coordsFromFaceIndex<nDim, pc_type, 1, nface_>(coord.X, coord.x_cb, coord, idx, face_num, parity, arg);
      } else if (idx < arg.threadDimMapUpper[2] * Ls) { // z face
        dim = 2;
        idx -= arg.threadDimMapLower[2] * Ls;
        const int face_size = nface_ * arg.dc.ghostFaceCB[dim] * Ls;
        const int face_num = idx >= face_size;
        idx -= face_num * face_size;
        coordsFromFaceIndex<nDim, pc_type, 2, nface_>(coord.X, coord.x_cb, coord, idx, face_num, parity, arg);
      } else { // t face
        dim = 3;
        idx -= arg.threadDimMapLower[3] * Ls;
        const int face_size = nface_ * arg.dc.ghostFaceCB[dim] * Ls;
        const int face_num = idx >= face_size;
        idx -= face_num * face_size;
        coordsFromFaceIndex<nDim, pc_type, 3, nface_>(coord.X, coord.x_cb, coord, idx, face_num, parity, arg);
      }
    }
    coord.s = s;
    return coord;
  }

  /**
     @brief Compute whether the provided coordinate is within the halo
     region boundary of a given dimension.

     @param[in] coord Coordinates
     @param[in] Arg Dslash argument struct
     @return True if in boundary, else false
  */
  template <int dim, typename Coord, typename Arg> inline __host__ __device__ bool inBoundary(const Coord &coord, const Arg &arg)
  {
    return ((coord[dim] >= arg.dim[dim] - arg.nFace) || (coord[dim] < arg.nFace));
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
  template <KernelType kernel_type, typename Coord, typename Arg>
  inline __device__ bool isActive(bool &active, int threadDim, int offsetDim, const Coord &coord, const Arg &arg)
  {
    // Threads with threadDim = t can handle t,z,y,x offsets
    // Threads with threadDim = z can handle z,y,x offsets
    // Threads with threadDim = y can handle y,x offsets
    // Threads with threadDim = x can handle x offsets
    if (!arg.commDim[offsetDim]) return false;

    if (kernel_type == EXTERIOR_KERNEL_ALL) {
      if (threadDim < offsetDim) return false;

      switch (threadDim) {
      case 3: // threadDim = T
        break;

      case 2: // threadDim = Z
        if (!arg.commDim[3]) break;
        if (arg.commDim[3] && inBoundary<3>(coord, arg)) return false;
        break;

      case 1: // threadDim = Y
        if ((!arg.commDim[3]) && (!arg.commDim[2])) break;
        if (arg.commDim[3] && inBoundary<3>(coord, arg)) return false;
        if (arg.commDim[2] && inBoundary<2>(coord, arg)) return false;
        break;

      case 0: // threadDim = X
        if ((!arg.commDim[3]) && (!arg.commDim[2]) && (!arg.commDim[1])) break;
        if (arg.commDim[3] && inBoundary<3>(coord, arg)) return false;
        if (arg.commDim[2] && inBoundary<2>(coord, arg)) return false;
        if (arg.commDim[1] && inBoundary<1>(coord, arg)) return false;
        break;

      default: break;
      }
    }

    active = true;
    return true;
  }

  template <typename Float_, int nDim_> struct DslashArg {

    using Float = Float_;
    using real = typename mapper<Float>::type;
    static constexpr int nDim = nDim_;

    const int parity;  // only use this for single parity fields
    const int nParity; // number of parities we're working on
    const int nFace;   // hard code to 1 for now
    const QudaReconstructType reconstruct;

    const int_fastdiv X0h;
    const int_fastdiv dim[5]; // full lattice dimensions
    const int volumeCB;       // checkerboarded volume
    int commDim[4];           // whether a given dimension is partitioned or not (potentially overridden for Schwarz)

    const bool dagger; // dagger
    const bool xpay;   // whether we are doing xpay or not

    real t_proj_scale; // factor to correct for T-dimensional spin projection

    DslashConstant dc;      // pre-computed dslash constants for optimized indexing
    KernelType kernel_type; // interior, exterior_t, etc.
    bool remote_write;      // used by the autotuner to switch on/off remote writing vs using copy engines

    int_fastdiv threads; // number of threads in x-thread dimension
    int threadDimMapLower[4];
    int threadDimMapUpper[4];

    const bool spin_project; // whether to spin project nSpin=4 fields (generally true, except for, e.g., covariant derivative)

    // these are set with symmetric preconditioned twisted-mass dagger
    // operator for the packing (which needs to a do a twist)
    real twist_a; // scale factor
    real twist_b; // chiral twist
    real twist_c; // flavor twist

    int pack_threads; // really number of face sites we have to pack
    int_fastdiv blocks_per_dir;
    int dim_map[4];
    int active_dims;
    int pack_blocks; // total number of blocks used for packing in the dslash

    // constructor needed for staggered to set xpay from derived class
    DslashArg(const ColorSpinorField &in, const GaugeField &U, int parity, bool dagger, bool xpay, int nFace,
              int spin_project, const int *comm_override) :
      parity(parity),
      nParity(in.SiteSubset()),
      nFace(nFace),
      reconstruct(U.Reconstruct()),
      X0h(nParity == 2 ? in.X(0) / 2 : in.X(0)),
      dim {(3 - nParity) * in.X(0), in.X(1), in.X(2), in.X(3), in.Ndim() == 5 ? in.X(4) : 1},
      volumeCB(in.VolumeCB()),
      dagger(dagger),
      xpay(xpay),
      kernel_type(INTERIOR_KERNEL),
      threads(in.VolumeCB()),
      threadDimMapLower {},
      threadDimMapUpper {},
      spin_project(spin_project),
      twist_a(0.0),
      twist_b(0.0),
      twist_c(0.0),
      pack_threads(0),
      blocks_per_dir(1),
      dim_map {},
      active_dims(0),
      pack_blocks(0)
    {
      for (int d = 0; d < 4; d++) {
        commDim[d] = (comm_override[d] == 0) ? 0 : comm_dim_partitioned(d);
      }

      if (in.Location() == QUDA_CUDA_FIELD_LOCATION) {
        // create comms buffers - need to do this before we grab the dslash constants
        ColorSpinorField *in_ = const_cast<ColorSpinorField *>(&in);
        static_cast<cudaColorSpinorField *>(in_)->createComms(nFace, spin_project);
      }
      dc = in.getDslashConstant();
    }

    void setPack(bool pack)
    {
      if (pack) {
        // set packing parameters
        // for now we set one block per direction / dimension
        int d = 0;
        pack_threads = 0;
        for (int i = 0; i < 4; i++) {
          if (!commDim[i]) continue;
          if (i == 3 && !getKernelPackT()) continue;
          pack_threads += 2 * nFace * dc.ghostFaceCB[i]; // 2 for fwd/back faces
          dim_map[d++] = i;
        }
        active_dims = d;
        pack_blocks = active_dims * blocks_per_dir * 2;
      } else {
        pack_threads = 0;
        pack_blocks = 0;
        active_dims = 0;
      }
    }
  };

  template <typename Float, int nDim> std::ostream &operator<<(std::ostream &out, const DslashArg<Float, nDim> &arg)
  {
    out << "parity = " << arg.parity << std::endl;
    out << "nParity = " << arg.nParity << std::endl;
    out << "nFace = " << arg.nFace << std::endl;
    out << "reconstruct = " << arg.reconstruct << std::endl;
    out << "X0h = " << arg.X0h << std::endl;
    out << "dim = { ";
    for (int i = 0; i < 5; i++) out << arg.dim[i] << (i < 4 ? ", " : " }");
    out << std::endl;
    out << "commDim = { ";
    for (int i = 0; i < 4; i++) out << arg.commDim[i] << (i < 3 ? ", " : " }");
    out << std::endl;
    out << "volumeCB = " << arg.volumeCB << std::endl;
    out << "dagger = " << arg.dagger << std::endl;
    out << "xpay = " << arg.xpay << std::endl;
    out << "kernel_type = " << arg.kernel_type << std::endl;
    out << "remote_write = " << arg.remote_write << std::endl;
    out << "threads = " << arg.threads << std::endl;
    out << "threadDimMapLower = { ";
    for (int i = 0; i < 4; i++) out << arg.threadDimMapLower[i] << (i < 3 ? ", " : " }");
    out << std::endl;
    out << "threadDimMapUpper = { ";
    for (int i = 0; i < 4; i++) out << arg.threadDimMapUpper[i] << (i < 3 ? ", " : " }");
    out << std::endl;
    out << "twist_a = " << arg.twist_a << std::endl;
    out << "twist_b = " << arg.twist_b << std::endl;
    out << "twist_c = " << arg.twist_c << std::endl;
    out << "pack_threads = " << arg.pack_threads << std::endl;
    out << "blocks_per_dir = " << arg.blocks_per_dir << std::endl;
    out << "dim_map = { ";
    for (int i = 0; i < 4; i++) out << arg.dim_map[i] << (i < 3 ? ", " : " }");
    out << std::endl;
    out << "active_dims = " << arg.active_dims << std::endl;
    out << "pack_blocks = " << arg.pack_blocks << std::endl;

    return out;
  }

  /**
     @brief Base class that set common types for dslash
     implementations.  Where necessary, we specialize in the derived
     classed.
   */
  struct dslash_default {
    constexpr QudaPCType pc_type() const { return QUDA_4D_PC; }
    constexpr int twist_pack() const { return 0; }
  };

  /**
     @brief This is a helper routine for spawning a CPU function for
     applying a Dslash kernel.  The dslash to be applied is passed as
     template template class (template parameter D), which is a
     functor that can apply the dslash.
   */
  template <template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
            class D,
            typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  void dslashCPU(Arg arg)
  {
    D<Float, nDim, nColor, nParity, dagger, xpay, kernel_type, Arg> dslash;

    for (int parity = 0; parity < nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = nParity == 2 ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume
        dslash(arg, x_cb, 0, parity);
      } // 4-d volumeCB
    }   // parity
  }

  /**
     @brief This is a helper routine for spawning a GPU kernel for
     applying a Dslash kernel.  The dslash to be applied is passed as
     template template class (template parameter D), which is a
     functor that can apply the dslash.  The packing routine (P) to be
     used is similarly passed.

     When running an interior kernel, the first few "pack_blocks" CTAs
     are reserved for data packing, which may include communication to
     neighboring processes.
   */
  template <template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg> class D,
            template <bool dagger, QudaPCType pc, typename Arg> class P, int nParity, bool dagger, bool xpay,
            KernelType kernel_type, typename Arg>
  __global__ void dslashGPU(Arg arg)
  {
    D<nParity, dagger, xpay, kernel_type, Arg> dslash(arg);

    int s = blockDim.y * blockIdx.y + threadIdx.y;
    if (s >= arg.dc.Ls) return;

    // for full fields set parity from z thread index else use arg setting
    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;

    if (kernel_type == INTERIOR_KERNEL && blockIdx.x < arg.pack_blocks) {
      // first few blocks do packing kernel
      P<dagger, dslash.pc_type(), Arg> packer;
      packer(arg, s, 1 - parity, dslash.twist_pack()); // flip parity since pack is on input
    } else {
      const int dslash_block_offset = (kernel_type == INTERIOR_KERNEL ? arg.pack_blocks : 0);
      int x_cb = (blockIdx.x - dslash_block_offset) * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;

#ifdef QUDA_DSLASH_FAST_COMPILE
      dslash(x_cb, s, parity);
#else
      switch (parity) {
      case 0: dslash(x_cb, s, 0); break;
      case 1: dslash(x_cb, s, 1); break;
      }
#endif
    }
  }

} // namespace quda
