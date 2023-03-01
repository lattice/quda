#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <kernels/color_spinor_pack.cuh>
#include <instantiate.h>
#include <multigrid.h>

/**
   @file color_spinor_pack.cu

   @brief This is the implementation of the color-spinor halo packer
   for an arbitrary field.  This implementation uses the fine-grained
   accessors and should support all field types reqgardless of
   precision, number of color or spins etc.

   Using a different precision of the field and of the halo is
   supported, though only QUDA_SINGLE_PRECISION fields with
   QUDA_HALF_PRECISION or QUDA_QUARTER_PRECISION halos are
   instantiated. When an integer format is requested for the halos
   then block-float format is used.

   As well as tuning basic block sizes, the autotuner also tunes for
   the dimensions to assign to each thread.  E.g., dim_thread=1 means
   we have one thread for all dimensions, dim_thread=4 means we have
   four threads (e.g., one per dimension).  We always uses seperate
   threads for forwards and backwards directions.  Dimension,
   direction and parity are assigned to the z thread dimension.

   If doing block-float format, since all spin and color components of
   a given site have to reside in the same thread block (to allow us
   to compute the max element) we override the autotuner to keep the z
   thread dimensions in the grid and not the block, and allow for
   smaller tuning increments of the thread block dimension in x to
   ensure that we can always fit within a single thread block.  It is
   this constraint that gives rise for the need to cap the limit for
   block-float support, e.g., max_block_float_nc.
 */

namespace quda {

  // this is the maximum number of colors for which we support block-float format

  template <typename store_t, typename ghost_store_t, QudaFieldOrder order, int nSpin, int nColor>
  class GhostPack : public TunableKernel3D {
    void **ghost;
    const ColorSpinorField &a;
    cvector_ref<const ColorSpinorField> &v;
    const QudaParity parity;
    const int nFace;
    const int dagger;
    static constexpr bool block_float = sizeof(store_t) == QUDA_SINGLE_PRECISION && isFixed<ghost_store_t>::value;
    size_t work_items;
    int shmem;
    static constexpr int get_max_block_float_nc() { return 6144; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const
    {
      if (block_float) {
        auto thread_width_x = a.isNative() ?
          ((param.block.x + device::warp_size() - 1) / device::warp_size()) * device::warp_size() :
          param.block.x;
        return sizeof(store_t) * thread_width_x * param.block.y * param.block.z;
      } else {
        return 0;
      }
    }

    bool tuneSharedBytes() const { return false; }
    unsigned int minThreads() const { return work_items; }

  public:
    GhostPack(void **ghost, const ColorSpinorField &a, QudaParity parity, int nFace, int dagger,
              MemoryLocation *destination, int shmem_, cvector_ref<const ColorSpinorField> &v) :
      TunableKernel3D(a, (a.Nspin() / spins_per_thread(a)) * (a.Ncolor() / colors_per_thread(a)), a.SiteSubset()),
      ghost(ghost),
      a(a),
      v(v),
      parity(parity),
      nFace(nFace),
      dagger(dagger),
      work_items(0),
      shmem(shmem_)
    {
      // if doing block float then all spin-color components must be within the same block
      if (block_float) resizeStep((a.Nspin()/spins_per_thread(a))*(a.Ncolor()/colors_per_thread(a)), step_z);
      switch (a.GhostPrecision()) {
      case QUDA_DOUBLE_PRECISION:  strcat(aux,",halo_prec=8"); break;
      case QUDA_SINGLE_PRECISION:  strcat(aux,",halo_prec=4"); break;
      case QUDA_HALF_PRECISION:    strcat(aux,",halo_prec=2"); break;
      case QUDA_QUARTER_PRECISION: strcat(aux,",halo_prec=1"); break;
      default: errorQuda("Unexpected precision = %d", a.GhostPrecision());
      }
      strcat(aux,comm_dim_partitioned_string());
      strcat(aux,comm_dim_topology_string());

      // record the location of where each pack buffer is in [2*dim+dir] ordering
      // 0 - no packing
      // 1 - pack to local GPU memory
      // 2 - pack to local mapped CPU memory
      // 3 - pack to remote mapped GPU memory
      char label[15] = ",dest=";
      for (int dim=0; dim<4; dim++) {
	for (int dir=0; dir<2; dir++) {
	  label[2*dim+dir+6] = !comm_dim_partitioned(dim) ? '0' : destination[2*dim+dir] == Device ? '1' : destination[2*dim+dir] == Host ? '2' : '3';
	}
      }
      label[14] = '\0';
      strcat(aux, label);
      strcat(aux, ",nFace=");
      u32toa(aux + strlen(aux), nFace);
      strcat(aux, ",spins_per_thread=");
      u32toa(aux + strlen(aux), spins_per_thread(a));
      strcat(aux, ",colors_per_thread=");
      u32toa(aux + strlen(aux), colors_per_thread(a));
      strcat(aux, ",shmem=");
      u32toa(aux + strlen(aux), shmem);
      if (v.size()) strcat(aux, ",batched");

      // compute number of number of work items we have to do
      // unlike the dslash kernels, we include the fifth dimension here
      for (int i = 0; i < 4; i++) {
        if (!comm_dim_partitioned(i)) continue;
        work_items += 2 * nFace * a.getDslashConstant().ghostFaceCB[i] * a.getDslashConstant().Ls; // 2 for forwards and backwards faces
      }

      apply(device::get_default_stream());
    }

    template <int nDim> using Arg = PackGhostArg<store_t, ghost_store_t, nSpin, nColor, nDim, order>;

    template <bool enable>
    std::enable_if_t<enable, void> launch_(const TuneParam &tp, const qudaStream_t &stream)
    {
      if (a.Ndim() == 5)
        launch<GhostPacker, true>(tp, stream, Arg<5>(a, work_items, ghost, parity, nFace, dagger, shmem, v));
      else
        launch<GhostPacker, true>(tp, stream, Arg<4>(a, work_items, ghost, parity, nFace, dagger, shmem, v));
    }

    template <bool enable>
    std::enable_if_t<!enable, void> launch_(TuneParam &, const qudaStream_t &)
    {
      errorQuda("block-float halo format not supported for nColor = %d", nColor);
    }

    void apply(const qudaStream_t &stream)
    {
      auto tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch_<(!block_float || nColor <= get_max_block_float_nc())>(tp, stream);
    }

    int blockStep() const { return block_float ? 1 : TunableKernel3D::blockStep(); }
    int blockMin() const { return block_float ? 1 : TunableKernel3D::blockMin(); }

    long long bytes() const { return work_items * 2 * a.Nspin() * a.Ncolor() * (a.Precision() + a.GhostPrecision()); }
  };

  template <int...> struct IntList { };

  template <typename Float, typename ghostFloat, int Ns, bool native, int fineColor, int coarseColor, int...N>
  bool genericPackGhostC(void **ghost, const ColorSpinorField &a, QudaParity parity, int nFace, int dagger,
                         MemoryLocation *destination, int shmem, cvector_ref<const ColorSpinorField> &v,
                         IntList<coarseColor, N...>)
  {
    constexpr int Nc = fineColor * coarseColor;
    if (a.Ncolor() == Nc) {
      
      // don't compile if
      // 1. double precision MG unless enabled
      // 2. block-float format with arbitrary colors
      constexpr bool do_not_compile =
        (std::is_same_v<Float, double> && Nc != 3 && !is_enabled_multigrid_double()) ||
        (std::is_same_v<Float, float> && std::is_same_v<ghostFloat, short> && Nc != 3 && Ns != 2) ||
        (std::is_same_v<Float, float> && std::is_same_v<ghostFloat, int8_t> && Nc != 3 && Ns != 2);

      if constexpr (!do_not_compile) {
        constexpr QudaFieldOrder order = native ? colorspinor::getNative<Float>(Ns) : QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        GhostPack<Float, ghostFloat, order, Ns, Nc>(ghost, a, parity, nFace, dagger, destination, shmem, v);
      } else {
        errorQuda("Not supported (Nc = %d, Ns = %d, Precision = %d, Ghost Precision = %d)",
                  a.Ncolor(), a.Nspin(), a.Precision(), a.GhostPrecision());
      }
      return true;
    } else {
      if constexpr (sizeof...(N) > 0) {
        return genericPackGhostC<Float, ghostFloat, Ns, native, fineColor>
          (ghost, a, parity, nFace, dagger, destination, shmem, v, IntList<N...>());
      }
    }
    return false;
  }

  template <typename Float, typename ghostFloat, int Ns, bool native, int fineColor, int...N>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity, int nFace, int dagger,
                        MemoryLocation *destination, int shmem, cvector_ref<const ColorSpinorField> &v,
                        IntList<fineColor, N...>)
  {
    // 1 ensures we generate templates for just the fineColor with no multiplication by coarseColor
    IntList<1, @QUDA_MULTIGRID_MRHS_LIST@, @QUDA_MULTIGRID_NC_NVEC_LIST@> coarseColors;

    if (!genericPackGhostC<Float, ghostFloat, Ns, native, fineColor>
        (ghost, a, parity, nFace, dagger, destination, shmem, v, coarseColors)) {
      if constexpr (sizeof...(N) > 0) {
        genericPackGhost<Float, ghostFloat, Ns, native>
          (ghost, a, parity, nFace, dagger, destination, shmem, v, IntList<N...>());
      } else {
        errorQuda("Nc = %d has not been instantiated", a.Ncolor());        
      }
    }
  }

  template <typename Float, typename ghostFloat, bool native>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity, int nFace, int dagger,
                        MemoryLocation *destination, int shmem, cvector_ref<const ColorSpinorField> &v)
  {
    if (!is_enabled_spin(a.Nspin())) errorQuda("nSpin=%d not enabled for this build", a.Nspin());

    IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@> fineColors;
    if (a.Nspin() == 4) {
      if constexpr (is_enabled_spin(4))
        genericPackGhost<Float, ghostFloat, 4, native>(ghost, a, parity, nFace, dagger, destination, shmem, v, fineColors);
    } else if (a.Nspin() == 2) {
      if constexpr (is_enabled_spin(2))
        genericPackGhost<Float, ghostFloat, 2, native>(ghost, a, parity, nFace, dagger, destination, shmem, v, fineColors);
    } else if (a.Nspin() == 1) {
      if constexpr (is_enabled_spin(1))
        genericPackGhost<Float, ghostFloat, 1, native>(ghost, a, parity, nFace, dagger, destination, shmem, v, fineColors);
    } else {
      errorQuda("Unsupported nSpin = %d", a.Nspin());
    }
  }

  template <typename Float, typename ghostFloat>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity, int nFace, int dagger,
                        MemoryLocation *destination, int shmem, cvector_ref<const ColorSpinorField> &v)
  {
    if (a.isNative()) {
      genericPackGhost<Float, ghostFloat, true>(ghost, a, parity, nFace, dagger, destination, shmem, v);
    } else if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      genericPackGhost<Float, ghostFloat, false>(ghost, a, parity, nFace, dagger, destination, shmem, v);
    } else {
      errorQuda("Unsupported field order = %d", a.FieldOrder());
    }
  }

  void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity, int nFace, int dagger,
                        MemoryLocation *destination_, int shmem, cvector_ref<const ColorSpinorField> v)
  {
    if (a.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
      errorQuda("Field order %d not supported", a.FieldOrder());
    }

    // set default location to match field type
    MemoryLocation destination[2*QUDA_MAX_DIM];
    for (int i=0; i<4*2; i++) {
      destination[i] = destination_ ? destination_[i] : a.Location() == QUDA_CUDA_FIELD_LOCATION ? Device : Host;
    }

    // only do packing if one of the dimensions is partitioned
    bool partitioned = false;
    for (int d=0; d<4; d++)
      if (comm_dim_partitioned(d)) partitioned = true;
    if (!partitioned) return;

    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      if (a.GhostPrecision() == QUDA_DOUBLE_PRECISION) {
        genericPackGhost<double, double>(ghost, a, parity, nFace, dagger, destination, shmem, v);
      } else {
        errorQuda("precision = %d and ghost precision = %d not supported", a.Precision(), a.GhostPrecision());
      }
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      if (a.GhostPrecision() == QUDA_SINGLE_PRECISION) {
        genericPackGhost<float, float>(ghost, a, parity, nFace, dagger, destination, shmem, v);
      } else if (a.GhostPrecision() == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION))
          genericPackGhost<float, short>(ghost, a, parity, nFace, dagger, destination, shmem, v);
        else
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      } else if (a.GhostPrecision() == QUDA_QUARTER_PRECISION) {
        if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
          genericPackGhost<float, int8_t>(ghost, a, parity, nFace, dagger, destination, shmem, v);
        else
          errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
      } else {
        errorQuda("precision = %d and ghost precision = %d not supported", a.Precision(), a.GhostPrecision());
      }
    } else if (a.Precision() == QUDA_HALF_PRECISION) {
      if (a.GhostPrecision() == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION))
          genericPackGhost<short, short>(ghost, a, parity, nFace, dagger, destination, shmem, v);
        else
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      } else {
        errorQuda("precision = %d and ghost precision = %d not supported", a.Precision(), a.GhostPrecision());
      }
    } else {
      errorQuda("Unsupported precision %d", a.Precision());
    }
  }

} // namespace quda
