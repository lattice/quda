#pragma once

#include <color_spinor_field.h>
#include <gauge_field.h>
#include <register_traits.h>
#include <index_helper.cuh>
#include <shmem_helper.cuh>
#include <fast_intdiv.h>
#include <dslash_quda.h>
#include <dslash_shmem.h>
#include <shmem_pack_helper.cuh>
#include <kernel_helper.h>
#include <tune_quda.h>
#include <domain_decomposition_helper.cuh>
#include <kernel_ops.h>
#include <tma_helper.hpp>

constexpr quda::use_kernel_arg_p use_kernel_arg = quda::use_kernel_arg_p::TRUE;

#include <kernel.h>

namespace quda
{

#ifdef QUDA_DSLASH_DOUBLE_STORE
  constexpr bool dslash_double_store() { return true; }
#else
  constexpr bool dslash_double_store() { return false; }
#endif

  constexpr PrefetchType dslash_prefetch_type()
  {
#if defined(QUDA_DSLASH_PREFETCH_TYPE_NONE)
    return PrefetchType::NONE;
#elif defined(QUDA_DSLASH_PREFETCH_TYPE_THREAD)
    return PrefetchType::THREAD;
#elif defined(QUDA_DSLASH_PREFETCH_TYPE_BULK)
    return PrefetchType::BULK;
#elif defined(QUDA_DSLASH_PREFETCH_TYPE_TENSOR)
    return PrefetchType::TENSOR;
#else
#error "Invalid or missing QUDA_DSLASH_PREFETCH_TYPE"
#endif
    return PrefetchType::NONE;
  }

#if defined(NVSHMEM_COMMS) && (defined(QUDA_DSLASH_PREFETCH_TYPE_BULK) || defined(QUDA_DSLASH_PREFETCH_TYPE_TENSOR))
#error NVSHMEM cannot be used in combination with TMA prefetching at present
#endif

  constexpr bool dslash_prefetch_tma()
  {
    return (dslash_prefetch_type() == PrefetchType::BULK || dslash_prefetch_type() == PrefetchType::TENSOR);
  }

  static_assert(!dslash_prefetch_tma() || dslash_double_store(),
                "Cannot use TMA prefetching unless QUDA_DSLASH_DOUBLE_STORE is enabled");

  /**
     @brief Helper function to determine if we should do halo
     computation
     @param[in] dim Dimension we are working on.  If dim=-1 (default
     argument) then we return true if type is any halo kernel.
  */
  template <KernelType type> __host__ __device__ __forceinline__ constexpr bool doHalo(int dim = -1)
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
  template <KernelType type> __host__ __device__ __forceinline__ constexpr bool doBulk()
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
  template <KernelType type, typename Arg, typename Coord>
  __host__ __device__ __forceinline__ bool isComplete(const Arg &arg, const Coord &coord)
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
  template <QudaPCType pc_type, KernelType kernel_type, typename Arg>
  __host__ __device__ __forceinline__ auto getCoords(const Arg &arg, int &idx, int s, int parity, int &dim,
                                                     int block_x = 0)
  {
    constexpr auto nDim = Arg::nDim;
    Coord<nDim> coord;
    for (auto i = 0; i < nDim; i++) coord.gDim[i] = arg.gDim[i];
    dim = kernel_type; // keep compiler happy

    // only for 5-d checkerboarding where we need to include the fifth dimension
    const int Ls = (nDim == 5 && pc_type == QUDA_5D_PC ? (int)arg.dc.X[4] : 1);

    if (kernel_type == INTERIOR_KERNEL) {
      coord.x_cb = idx;
      if constexpr (Arg::work_steal) {
        coord.x_cb_0 = (block_x - arg.pack_blocks) * target::block_dim().x;
      } else {
        coord.x_cb_0 = (target::block_idx<Arg>().x - arg.pack_blocks) * target::block_dim().x;
      }
      if (nDim == 5)
        coord.X = getCoords5CB(coord, idx, arg.dc.X, arg.X0h, parity, pc_type);
      else
        coord.X = getCoordsCB(coord, idx, arg.dc.X, arg.X0h, parity);
    } else if (kernel_type != EXTERIOR_KERNEL_ALL) {

      // compute face index and then compute coords
      const int face_size = arg.dc.ghostFaceCB[kernel_type] * Ls;
      const int face_num = idx >= face_size;
      idx -= face_num * face_size;
      coordsFromFaceIndex<nDim, pc_type, kernel_type, Arg::nFace>(coord.X, coord.x_cb, coord, idx, face_num, parity, arg);

    } else { // fused kernel

      // work out which dimension this thread corresponds to, then compute coords
      if (idx < arg.threadDimMapUpper[0] * Ls) { // x face
        dim = 0;
        const int face_size = arg.dc.ghostFaceCB[dim] * Ls;
        const int face_num = idx >= face_size;
        idx -= face_num * face_size;
        coordsFromFaceIndex<nDim, pc_type, 0, Arg::nFace>(coord.X, coord.x_cb, coord, idx, face_num, parity, arg);
      } else if (idx < arg.threadDimMapUpper[1] * Ls) { // y face
        dim = 1;
        idx -= arg.threadDimMapLower[1] * Ls;
        const int face_size = arg.dc.ghostFaceCB[dim] * Ls;
        const int face_num = idx >= face_size;
        idx -= face_num * face_size;
        coordsFromFaceIndex<nDim, pc_type, 1, Arg::nFace>(coord.X, coord.x_cb, coord, idx, face_num, parity, arg);
      } else if (idx < arg.threadDimMapUpper[2] * Ls) { // z face
        dim = 2;
        idx -= arg.threadDimMapLower[2] * Ls;
        const int face_size = arg.dc.ghostFaceCB[dim] * Ls;
        const int face_num = idx >= face_size;
        idx -= face_num * face_size;
        coordsFromFaceIndex<nDim, pc_type, 2, Arg::nFace>(coord.X, coord.x_cb, coord, idx, face_num, parity, arg);
      } else { // t face
        dim = 3;
        idx -= arg.threadDimMapLower[3] * Ls;
        const int face_size = arg.dc.ghostFaceCB[dim] * Ls;
        const int face_num = idx >= face_size;
        idx -= face_num * face_size;
        coordsFromFaceIndex<nDim, pc_type, 3, Arg::nFace>(coord.X, coord.x_cb, coord, idx, face_num, parity, arg);
      }
    }
    for (int i = 0; i < nDim; i++) { coord.gx[i] = arg.commCoord[i] + coord.x[i]; }
    coord.s = s;

#pragma unroll
    for (int d = 0; d < nDim; d++) {
      coord.in_boundary[1][d] = -(coord[d] + arg.nFace >= arg.dc.X[d]);
      coord.in_boundary[0][d] = -(coord[d] - arg.nFace < 0);
    }

    return coord;
  }

  /**
     @brief Compute the checkerboard 1-d index for the nearest
     neighbor
     @param[in] lattice coordinates
     @param[in] mu dimension in which to add 1
     @param[in] dir direction (+1 or -1)
     @param[in] arg parameter struct
     @return 1-d checkboard index
   */
  template <int nFace = 1, typename Coord, typename Arg>
  __device__ __host__ inline int getNeighborIndexCB(const Coord &x, int mu, int dir, const Arg &arg)
  {
    switch (nFace) {
    case 1:
      switch (dir) {
      case +1: // positive direction
        switch (mu) {
        case 0: return (x.X + 1 - (x.in_boundary[1][0] & arg.X[0])) >> 1;
        case 1: return (x.X + arg.X[0] - (x.in_boundary[1][1] & arg.X2X1)) >> 1;
        case 2: return (x.X + arg.X2X1 - (x.in_boundary[1][2] & arg.X3X2X1)) >> 1;
        case 3: return (x.X + arg.X3X2X1 - (x.in_boundary[1][3] & arg.X4X3X2X1)) >> 1;
        case 4: return (x.X + arg.X4X3X2X1 - (x.in_boundary[1][4] & arg.X5X4X3X2X1)) >> 1;
        }
      case -1:
        switch (mu) {
        case 0: return (x.X - 1 + (x.in_boundary[0][0] & arg.X[0])) >> 1;
        case 1: return (x.X - arg.X[0] + (x.in_boundary[0][1] & arg.X2X1)) >> 1;
        case 2: return (x.X - arg.X2X1 + (x.in_boundary[0][2] & arg.X3X2X1)) >> 1;
        case 3: return (x.X - arg.X3X2X1 + (x.in_boundary[0][3] & arg.X4X3X2X1)) >> 1;
        case 4: return (x.X - arg.X4X3X2X1 + (x.in_boundary[0][4] & arg.X5X4X3X2X1)) >> 1;
        }
      }
    case 3:
      switch (dir) {
      case +1: // positive direction
        switch (mu) {
        case 0: return (x.X + 3 - (x.in_boundary[1][0] & arg.X[0])) >> 1;
        case 1: return (x.X + 3 * arg.X[0] - (x.in_boundary[1][1] & arg.X2X1)) >> 1;
        case 2: return (x.X + 3 * arg.X2X1 - (x.in_boundary[1][2] & arg.X3X2X1)) >> 1;
        case 3: return (x.X + 3 * arg.X3X2X1 - (x.in_boundary[1][3] & arg.X4X3X2X1)) >> 1;
        case 4: return (x.X + 3 * arg.X4X3X2X1 - (x.in_boundary[1][4] & arg.X5X4X3X2X1)) >> 1;
        }
      case -1:
        switch (mu) {
        case 0: return (x.X - 3 + (x.in_boundary[0][0] & arg.X[0])) >> 1;
        case 1: return (x.X - 3 * arg.X[0] + (x.in_boundary[0][1] & arg.X2X1)) >> 1;
        case 2: return (x.X - 3 * arg.X2X1 + (x.in_boundary[0][2] & arg.X3X2X1)) >> 1;
        case 3: return (x.X - 3 * arg.X3X2X1 + (x.in_boundary[0][3] & arg.X4X3X2X1)) >> 1;
        case 4: return (x.X - 3 * arg.X4X3X2X1 + (x.in_boundary[0][4] & arg.X5X4X3X2X1)) >> 1;
        }
      }
    }
    return 0; // should never reach here
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
        if (arg.commDim[3] && (coord.in_boundary[0][3] || coord.in_boundary[1][3])) return false;
        break;

      case 1: // threadDim = Y
        if ((!arg.commDim[3]) && (!arg.commDim[2])) break;
        if (arg.commDim[3] && (coord.in_boundary[0][3] || coord.in_boundary[1][3])) return false;
        if (arg.commDim[2] && (coord.in_boundary[0][2] || coord.in_boundary[1][2])) return false;
        break;

      case 0: // threadDim = X
        if ((!arg.commDim[3]) && (!arg.commDim[2]) && (!arg.commDim[1])) break;
        if (arg.commDim[3] && (coord.in_boundary[0][3] || coord.in_boundary[1][3])) return false;
        if (arg.commDim[2] && (coord.in_boundary[0][2] || coord.in_boundary[1][2])) return false;
        if (arg.commDim[1] && (coord.in_boundary[0][1] || coord.in_boundary[1][1])) return false;
        break;

      default: break;
      }
    }

    active = true;
    return true;
  }

  template <typename Float_, int nDim_, typename DDArg, int nFace_ = 1, int n_src_tile_ = 1> struct DslashArg {

    using Float = Float_;
    using real = typename mapper<Float>::type;
    static constexpr int nDim = nDim_;
    static constexpr int nFace = nFace_;
    static constexpr int n_src_tile = n_src_tile_; // how many RHS per thread
    static constexpr int max_regs = 0;             // by default we don't limit register count
    static constexpr bool spill_shared = false;    // whether a given kernel should use shared memory spilling
    static constexpr bool work_steal = QUDA_WORK_STEAL_DSLASH;
    static constexpr int prefetch_distance = 0;    // whether we are using prefetching in the dslash
    static constexpr PrefetchType prefetch_type = dslash_prefetch_type();
    const int parity;  // only use this for single parity fields
    const int nParity; // number of parities we're working on
    const QudaReconstructType reconstruct;

    const int_fastdiv X0h;
    const int dim[5];         // full lattice dimensions
    const int gDim[5];        // global full lattice dimensions
    int commDim[4];           // whether a given dimension is partitioned or not (potentially overridden for Schwarz)

    const int commCoord[5];
    const int globalDim3;

    const bool dagger; // dagger
    const bool xpay;   // whether we are doing xpay or not

    DslashConstant dc;      // pre-computed dslash constants for optimized indexing
    KernelType kernel_type; // interior, exterior_t, etc.
    bool remote_write;      // used by the autotuner to switch on/off remote writing vs using copy engines

    int_fastdiv threads; // number of threads in x-thread dimension
    int_fastdiv exterior_threads = 0; // number of threads in x-thread dimension for fused exterior dslash
    int threadDimMapLower[4] = {};
    int threadDimMapUpper[4] = {};

    int_fastdiv n_src;
    int_fastdiv Ls;

    // these are set with symmetric preconditioned twisted-mass dagger
    // operator for the packing (which needs to a do a twist)
    real twist_a = 0.0; // scale factor
    real twist_b = 0.0; // chiral twist
    real twist_c = 0.0; // flavor twist

    int pack_threads = 0; // really number of face sites we have to pack
    int_fastdiv blocks_per_dir = 1;
    int sites_per_block;
    int dim_map[4] = {};
    int active_dims = 0;
    int pack_blocks = 0;   // total number of blocks used for packing in the dslash
    int exterior_dims = 0; // dimension to run in the exterior Dslash
    int exterior_blocks = 0;
    int block_size = 0;

    DDArg dd_out;
    DDArg dd_in;
    DDArg dd_x;

    // for shmem ...
    static constexpr bool packkernel = false;
    void *packBuffer[4 * QUDA_MAX_DIM];
    int neighbor_ranks[2 * QUDA_MAX_DIM];
    int bytes[2 * QUDA_MAX_DIM];
#ifndef NVSHMEM_COMMS
    static constexpr int shmem = 0;
    dslash::shmem_sync_t counter = 0;
#else
    int shmem;
    dslash::shmem_sync_t counter = 0;
    dslash::shmem_sync_t *sync_arr;
    dslash::shmem_interior_done_t &interior_done;
    dslash::shmem_interior_count_t &interior_count;
    dslash::shmem_retcount_intra_t *retcount_intra;
    dslash::shmem_retcount_inter_t *retcount_inter;
#endif

    // constructor needed for staggered to set xpay from derived class
    DslashArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
              const GaugeField &U, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, bool xpay,
              int spin_project, const int *comm_override,
#ifdef NVSHMEM_COMMS
              int shmem_ = 0) :
#else
              int = 0) :
#endif
      parity(parity),
      nParity(in.SiteSubset()),
      reconstruct(U.Reconstruct()),
      X0h(nParity == 2 ? in.X(0) / 2 : in.X(0)),
      dim {(3 - nParity) * in.X(0), in.X(1), in.X(2), in.X(3), in.Ndim() == 5 ? in.X(4) : 1},
      gDim {comm_dim(0) * dim[0], comm_dim(1) * dim[1], comm_dim(2) * dim[2], comm_dim(3) * dim[3], dim[4]},
      commCoord {comm_coord(0) * dim[0], comm_coord(1) * dim[1], comm_coord(2) * dim[2], comm_coord(3) * dim[3], dim[4]},
      globalDim3(comm_dim(3) * this->dim[3]),
      dagger(dagger),
      xpay(xpay),
      kernel_type(INTERIOR_KERNEL),
      threads(in.VolumeCB()),
      n_src(in.size()),
      Ls(halo.X(4) / in.size()),
      dd_out(out.DD()),
      dd_in(in.DD()),
      dd_x(x.DD())
#ifdef NVSHMEM_COMMS
      ,
      shmem(shmem_),
      counter(dslash::get_dslash_shmem_sync_counter()),
      sync_arr(dslash::get_dslash_shmem_sync_arr()),
      interior_done(*dslash::get_shmem_interior_done()),
      interior_count(*dslash::get_shmem_interior_count()),
      retcount_intra(dslash::get_shmem_retcount_intra()),
      retcount_inter(dslash::get_shmem_retcount_inter())
#endif
    {
      if (out.size() > get_max_multi_rhs())
        errorQuda("vector set size %lu greater than max size %d", out.size(), get_max_multi_rhs());
      for (auto i = 0u; i < in.size(); i++)
        if (in[i].data() == out[i].data()) errorQuda("Aliasing pointers");
      checkOrder(out, in, x);        // check all orders match
      checkLocation(out, in, x, U);  // check all locations match
      checkDD(out, in, x);           // check all DD match
      checkNative(in, U);

      for (int d = 0; d < 4; d++) {
        commDim[d] = (comm_override[d] == 0) ? 0 : (comm_dim_partitioned(d) * dd_out.commDim(d, dd_in, *this));
      }

      if (in.Location() == QUDA_CUDA_FIELD_LOCATION) {
        // create comms buffers - need to do this before we grab the dslash constants
        halo.createComms(nFace, spin_project);
      }
      dc = halo.getDslashConstant();

      for (int dim = 0; dim < 4; dim++) {
        dc.ghostFaceCB[dim] *= nFace;
        for (int dir = 0; dir < 2; dir++) {
          neighbor_ranks[2 * dim + dir] = commDim[dim] ? comm_neighbor_rank(dir, dim) : -1;
          bytes[2 * dim + dir] = halo.GhostFaceBytes(dim);
        }
      }
    }

    void setPack(bool pack, void *packBuffer_[4 * QUDA_MAX_DIM])
    {
      if (pack) {
        // set packing parameters
        // for now we set one block per direction / dimension
        int d = 0;
        pack_threads = 0;
        for (int i = 0; i < 4; i++) {
          if (!commDim[i]) continue;
          pack_threads += 2 * dc.ghostFaceCB[i]; // 2 for fwd/back faces
          dim_map[d++] = i;
        }
        active_dims = d;
        pack_blocks = active_dims * blocks_per_dir * 2;
        for (int i = 0; i < 4 * QUDA_MAX_DIM; i++) { packBuffer[i] = packBuffer_[i]; }
      } else {
        // we need dim_map for the grid-stride exterior kernel used in shmem
        int d = 0;
        for (int i = 0; i < 4; i++) {
          if (!commDim[i]) continue;
          dim_map[d++] = i;
        }
        pack_threads = 0;
        pack_blocks = 0;
        active_dims = 0;
      }
    }

    void setExteriorDims(bool exterior)
    {
      if (exterior) {
        int nDimComms = 0;
        for (int d = 0; d < 4; d++) nDimComms += commDim[d];
        exterior_dims = nDimComms;
      } else {
        exterior_dims = 0;
      }
    }
  };

  template <typename Float, int nDim, typename DDArg, int nFace, int n_src_tile>
  std::ostream &operator<<(std::ostream &out, const DslashArg<Float, nDim, DDArg, nFace, n_src_tile> &arg)
  {
    out << "parity = " << arg.parity << std::endl;
    out << "nParity = " << arg.nParity << std::endl;
    out << "nFace = " << arg.nFace << std::endl;
    out << "reconstruct = " << arg.reconstruct << std::endl;
    out << "X0h = " << arg.X0h << std::endl;
    out << "dc.X = { ";
    for (int i = 0; i < 5; i++) out << arg.dc.X[i] << (i < 4 ? ", " : " }");
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
    out << "exterior_threads = " << arg.exterior_threads << std::endl;
    out << "exterior_blocks = " << arg.exterior_blocks << std::endl;
    out << "dd_out: " << arg.dd_out << std::endl;
    out << "dd_in: " << arg.dd_in << std::endl;
    out << "dd_x: " << arg.dd_x << std::endl;

    return out;
  }

  /**
     @brief Base class that set common types for dslash
     implementations.  Where necessary, we specialize in the derived
     classed.
   */
  struct dslash_default {
    dim3 block_idx; /**< logical block index (set from dslash_functor when launched via Kernel3D) */

    // By default the dslash types do not have __syncthreads() in their operator();
    constexpr static bool use_syncthreads = false;

    constexpr QudaPCType pc_type() const { return QUDA_4D_PC; }
    constexpr int twist_pack() const { return 0; }
  };

  /**
     @brief This is a helper routine for spawning a CPU function for
     applying a Dslash kernel.  The dslash to be applied is passed as
     template template class (template parameter D), which is a
     functor that can apply the dslash.
   */
  template <template <typename Float, int nDim, int nColor, bool dagger, bool xpay, KernelType kernel_type, typename Arg> class D,
            typename Float, int nDim, int nColor, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  void dslashCPU(Arg arg)
  {
    D<Float, nDim, nColor, dagger, xpay, kernel_type, Arg> dslash;

    for (int parity = 0; parity < arg.nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = arg.nParity == 2 ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume
        dslash(arg, x_cb, 0, parity);
      } // 4-d volumeCB
    }   // parity
  }

  template <KernelType kernel_type, bool allthreads = false, class D>
  __forceinline__ __device__ void apply_dslash(D &dslash, int x_cb, int s, int parity, bool alive = true)
  {
    if constexpr (allthreads)
      dslash.template operator()<kernel_type, true>(x_cb, s, parity, alive);
    else
      dslash.template operator()<kernel_type>(x_cb, s, parity);
  }

#ifdef NVSHMEM_COMMS
  /**
   * @brief helper function for nvshmem uber kernel to signal that the interior kernel has completed.
      This function is supposed to be called only by the last thread of the block.
   */
  template <typename Arg> void __device__ inline shmem_signalinterior(const Arg &arg)
  {
    int amlast = arg.interior_count.fetch_add(1, cuda::std::memory_order_acq_rel); // ensure that my block is done
    if (amlast
        == (target::grid_dim().x - arg.pack_blocks - arg.exterior_blocks) * target::grid_dim().y * target::grid_dim().z
          - 1) {
      arg.interior_done.store(arg.counter, cuda::std::memory_order_release);
      arg.interior_done.notify_all();
      arg.interior_count.store(0, cuda::std::memory_order_relaxed);
    }
  }

  template <KernelType kernel_type, class D, typename Arg>
  void __device__ __forceinline__ shmem_exterior(D &dslash, const Arg &arg, int s, dim3 block_idx)
  {
    // shmem exterior kernel with grid-strided loop
    if (kernel_type == UBER_KERNEL || kernel_type == EXTERIOR_KERNEL_ALL) {
      // figure out some details on blocks
      const bool shmem_interiordone = (arg.shmem & 64);
      const int myblockidx
        = arg.exterior_blocks > 0 ? block_idx.x - (target::grid_dim().x - arg.exterior_blocks) : block_idx.x;
      const int nComm = arg.commDim[0] + arg.commDim[1] + arg.commDim[2] + arg.commDim[3];
      const int blocks_per_dim = (arg.exterior_blocks > 0 ? arg.exterior_blocks : target::grid_dim().x) / (nComm);
      const int blocks_per_dir = blocks_per_dim / 2;

      int dir = (myblockidx % blocks_per_dim) / (blocks_per_dir);
      // this is the dimdir we are working on ...
      int dim;
      int threadl;
      int threads_my_dir;
      switch (myblockidx / blocks_per_dim) {
      case 0: dim = arg.dim_map[0]; break;
      case 1: dim = arg.dim_map[1]; break;
      case 2: dim = arg.dim_map[2]; break;
      case 3: dim = arg.dim_map[3]; break;
      default: dim = -1;
      }

      switch (dim) {
      case 0:
        threads_my_dir = (arg.threadDimMapUpper[0] - arg.threadDimMapLower[0]) / 2;
        threadl = arg.threadDimMapLower[0];
        break;
      case 1:
        threads_my_dir = (arg.threadDimMapUpper[1] - arg.threadDimMapLower[1]) / 2;
        threadl = arg.threadDimMapLower[1];
        break;
      case 2:
        threads_my_dir = (arg.threadDimMapUpper[2] - arg.threadDimMapLower[2]) / 2;
        threadl = arg.threadDimMapLower[2];
        break;
      case 3:
        threads_my_dir = (arg.threadDimMapUpper[3] - arg.threadDimMapLower[3]) / 2;
        threadl = arg.threadDimMapLower[3];
        break;
      default: threadl = 0; threads_my_dir = 0;
      }
      int dimdir = 2 * dim + dir;
      constexpr bool shmembarrier = true; // always true for now (arg.shmem & 16);

      if (shmembarrier) {

        if (shmem_interiordone && target::thread_idx().x == target::block_dim().x - 1 && target::thread_idx().y == 0 && target::thread_idx().z == 0) {
          auto tst_val = arg.interior_done.load(cuda::std::memory_order_relaxed);
          while (tst_val < arg.counter - 1) {
            arg.interior_done.compare_exchange_strong(tst_val, arg.counter - 1, cuda::std::memory_order_relaxed,
                                                      cuda::std::memory_order_relaxed);
          }
          arg.interior_done.wait(arg.counter - 1, cuda::std::memory_order_acquire);
        }

        if (target::thread_idx().x < 8 && target::thread_idx().y == 0 && target::thread_idx().z == 0) {
          /* the first 8 threads of each block are used for spinning on halo data coming
            in from the 4*2 (dim*dir) neighbors. We figure out next on which neighbors the
            block actually needs to wait
          */

          // for now we can only spin per dimdir for 4d indexing as it ensure unique block->dimdir assignment
          bool spin = (dslash.pc_type() == QUDA_5D_PC) || (target::thread_idx().x == dimdir);
          // figure out which other directions also to spin for (to make corners work)
          switch (dim) {
          case 3:
            if (arg.commDim[3]) {
              spin = target::thread_idx().x / 2 < 3 ? arg.commDim[2] : spin;
              spin = target::thread_idx().x / 2 < 2 ? arg.commDim[1] : spin;
              spin = target::thread_idx().x / 2 < 1 ? arg.commDim[0] : spin;
            } else {
              spin = false;
            }
            break;
          case 2:
            if (arg.commDim[2]) {
              if (arg.commDim[1]) spin = target::thread_idx().x / 2 < 2 ? true : spin;
              if (arg.commDim[0]) spin = target::thread_idx().x / 2 < 1 ? true : spin;
            }
            break;
          case 1:
            if (arg.commDim[1]) {
              if (arg.commDim[0]) spin = target::thread_idx().x / 2 < 1 ? true : spin;
            }
            break;
          case 0: break;
          }

          if (getNeighborRank(target::thread_idx().x, arg) >= 0) {
            if (spin) { nvshmem_signal_wait_until((arg.sync_arr + target::thread_idx().x), NVSHMEM_CMP_GE, arg.counter); }
          }
        }

        // wait for all threads here as not all threads spin
        __syncthreads();
        // do exterior
      }

      int local_tid = target::thread_idx().x + target::block_dim().x * (myblockidx % (blocks_per_dir)); // index within the block
      int tid = local_tid + threadl + dir * threads_my_dir; // global index corresponding to local_tid

      while (local_tid < threads_my_dir) {
        // for full fields set parity from z thread index else use arg setting
        int parity = arg.nParity == 2 ? target::block_dim().z * block_idx.z + target::thread_idx().z : arg.parity;
        apply_dslash<EXTERIOR_KERNEL_ALL>(dslash, tid, s, parity);
        local_tid += target::block_dim().x * blocks_per_dir;
        tid += target::block_dim().x * blocks_per_dir;
      }
    }
  }

#endif // NVSHMEM_COMMS

  /**
    @brief This is the wrapper arg struct for driving the dslash_functor.  The dslash to
    be applied is passed as a template template class (template
    parameter D), which is a functor that can apply the dslash.  The
    packing routine (P) to be used is similarly passed.
   */
  template <template <bool dagger, bool xpay, KernelType kernel_type, typename Arg> class D_,
            template <bool dagger, QudaPCType pc, typename Arg> class P_, bool dagger_, bool xpay_,
            KernelType kernel_type_, typename Arg_>
  struct dslash_functor_arg : kernel_param<use_kernel_arg, true, Arg_::work_steal> {
    using Arg = Arg_;
    using D = D_<dagger_, xpay_, kernel_type_, Arg>;
    template <QudaPCType pc> using P = P_<dagger_, pc, Arg>;
    static constexpr bool dagger = dagger_;
    static constexpr bool xpay = xpay_;
    static constexpr KernelType kernel_type = kernel_type_;
    static constexpr int max_regs = Arg::max_regs;
    static constexpr bool spill_shared = Arg::spill_shared;
    static constexpr bool is_dslash = true;
    Arg arg;

    dslash_functor_arg(const Arg &arg, unsigned int threads_x) :
      kernel_param<use_kernel_arg, true, Arg_::work_steal>(
        dim3(threads_x, (arg.dc.Ls + Arg::n_src_tile - 1) / Arg::n_src_tile, arg.nParity)),
      arg(arg)
    {
    }
  };

  /**
    @brief This is the functor for the dslash stencils.

    When running an interior kernel, the first few "pack_blocks" CTAs
    are reserved for data packing, which may include communication to
    neighboring processes.
   */
  template <typename Arg> struct dslash_functor : getKernelOps<typename Arg::D> {
    const typename Arg::Arg &arg;
    dim3 block_idx; /**< logical block index (set by kernel launch, valid when Arg::is_dslash) */
    static constexpr bool dagger = Arg::dagger;
    static constexpr KernelType kernel_type = Arg::kernel_type;
    static constexpr const char *filename() { return Arg::D::filename(); }
    using typename getKernelOps<typename Arg::D>::KernelOpsT;
    template <typename... OpsArgs>
    constexpr dslash_functor(const Arg &arg, const OpsArgs &...ops) : KernelOpsT(ops...), arg(arg.arg)
    {
    }

    template <bool allthreads = false> // true if all threads in block will enter, even if out of range
    __forceinline__ __device__ void operator()(int, int s, int parity, bool alive = true) const
    {
      typename Arg::D dslash(*this);

      if constexpr (dslash_prefetch_tma()) {
        // FIXME need warp uniform parity which is not composable with
        // NVSHMEM since the latter requires blockDim.y and blockDim.z to
        // cover the entire extent
        parity = block_idx.z; // ensure parity is warp uniform
      }

      // for full fields set parity from z thread index else use arg setting
      if (arg.nParity == 1) parity = arg.parity;

      if ((kernel_type == INTERIOR_KERNEL || kernel_type == UBER_KERNEL)
          && block_idx.x < static_cast<unsigned int>(arg.pack_blocks)) {
        if (!allthreads || alive) {
          // first few blocks do packing kernel
          typename Arg::template P<dslash.pc_type()> packer;
          packer.block_idx = block_idx;
          packer(arg, s, 1 - parity, dslash.twist_pack()); // flip parity since pack is on input
        }
        // we use that when running the exterior -- this is either
        // * an explicit call to the exterior when not merged with the interior or
        // * the interior with exterior_blocks > 0
#ifdef NVSHMEM_COMMS
      } else if (arg.shmem > 0
                 && ((kernel_type == EXTERIOR_KERNEL_ALL && arg.exterior_blocks == 0)
                     || (kernel_type == UBER_KERNEL && arg.exterior_blocks > 0
                         && block_idx.x >= (target::grid_dim().x - arg.exterior_blocks)))) {
        shmem_exterior<kernel_type>(dslash, arg, s, block_idx);
#endif
      } else {
        const int dslash_block_offset
          = ((kernel_type == INTERIOR_KERNEL || kernel_type == UBER_KERNEL) ? arg.pack_blocks : 0);
        int x_cb = (block_idx.x - dslash_block_offset) * target::block_dim().x + target::thread_idx().x;

#ifdef NVSHMEM_COMMS
        constexpr bool use_nvshmem_comms = true;
#else
        constexpr bool use_nvshmem_comms = false;
#endif
        if constexpr (use_nvshmem_comms && Arg::D::use_syncthreads) {
#ifdef NVSHMEM_COMMS
          // Initialize a shared memory counter for the threads in the block
          __shared__ cuda::atomic<int, cuda::thread_scope_block> block_counter;
          if (target::thread_idx().x == 0 && target::thread_idx().y == 0 && target::thread_idx().z == 0) {
            block_counter.store(0, cuda::std::memory_order_relaxed);
          }
          __syncthreads();

          if (x_cb < arg.threads) {
            apply_dslash<kernel_type == UBER_KERNEL ? INTERIOR_KERNEL : kernel_type>(dslash, x_cb, s, parity);
          }
          // Use the shared memory counter to see if is the last thread in the block.
          // If yes, signal that interior is done for this block.
          int am_last_thread = block_counter.fetch_add(1, cuda::std::memory_order_acq_rel);
          if constexpr (kernel_type == UBER_KERNEL) {
            if (am_last_thread == (target::block_dim().x * target::block_dim().y * target::block_dim().z - 1))
              shmem_signalinterior(arg);
          }
#endif
        } else {
          if (x_cb >= arg.threads) {
            if constexpr (allthreads)
              alive = false;
            else
              return;
          }
          apply_dslash<kernel_type == UBER_KERNEL ? INTERIOR_KERNEL : kernel_type, allthreads>(dslash, x_cb, s, parity,
                                                                                               alive);
          if constexpr (use_nvshmem_comms && kernel_type == UBER_KERNEL) {
            __syncthreads();
            if (target::thread_idx().x == 0 && target::thread_idx().y == 0 && target::thread_idx().z == 0)
              shmem_signalinterior(arg);
          }
        }
      }
    }
  };

} // namespace quda
