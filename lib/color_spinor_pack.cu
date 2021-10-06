#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <kernels/color_spinor_pack.cuh>

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
  constexpr int max_block_float_nc = 96;

  template <typename store_t, typename ghost_store_t, QudaFieldOrder order, int nSpin, int nColor>
  class GhostPack : public TunableKernel3D {
    void **ghost;
    const ColorSpinorField &a;
    const QudaParity parity;
    const int nFace;
    const int dagger;
    static constexpr bool block_float = sizeof(store_t) == QUDA_SINGLE_PRECISION && isFixed<ghost_store_t>::value;
    size_t work_items;

    unsigned int sharedBytesPerBlock(const TuneParam &) const
    {
      if (block_float) {
        auto max_block_size_x = device::max_threads_per_block() / (vector_length_y * vector_length_z);
        auto thread_width_x = ((max_block_size_x + device::shared_memory_bank_width() - 1) /
                               device::shared_memory_bank_width()) * device::shared_memory_bank_width();
        return sizeof(store_t) * thread_width_x * vector_length_y * vector_length_z;
      } else {
        return 0;
      }
    }

    bool tuneSharedBytes() const { return false; }
    unsigned int minThreads() const { return work_items; }

  public:
    GhostPack(void **ghost, const ColorSpinorField &a, QudaParity parity,
              int nFace, int dagger, MemoryLocation *destination) :
      TunableKernel3D(a, (a.Nspin()/spins_per_thread(a))*(a.Ncolor()/colors_per_thread(a)), a.SiteSubset()),
      ghost(ghost),
      a(a),
      parity(parity),
      nFace(nFace),
      dagger(dagger),
      work_items(0)
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
      strcat(aux, ",spins_per_thread=");
      u32toa(label, spins_per_thread(a));
      strcat(aux, label);
      strcat(aux, ",colors_per_thread=");
      u32toa(label, colors_per_thread(a));
      strcat(aux, label);

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
      if (a.Ndim() == 5) launch<GhostPacker, true>(tp, stream, Arg<5>(a, work_items, ghost, parity, nFace, dagger));
      else               launch<GhostPacker, true>(tp, stream, Arg<4>(a, work_items, ghost, parity, nFace, dagger));
    }

    template <bool enable>
    std::enable_if_t<!enable, void> launch_(TuneParam &, const qudaStream_t &)
    {
      errorQuda("block-float halo format not supported for nColor = %d", nColor);
    }

    void apply(const qudaStream_t &stream)
    {
      auto tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch_<(!block_float || nColor <= max_block_float_nc)>(tp, stream);
    }

    int blockStep() const { return block_float ? 2 : TunableKernel3D::blockStep(); }
    int blockMin() const { return block_float ? 2 : TunableKernel3D::blockMin(); }

    long long bytes() const { return work_items * 2 * a.Nspin() * a.Ncolor() * (a.Precision() + a.GhostPrecision()); }
  };

  // traits used to ensure we only instantiate arbitrary colors for nSpin=2,4 fields, and only 3 colors otherwise
  template<typename T, typename G, int nSpin, int nColor_> struct precision_spin_color_mapper { static constexpr int nColor = nColor_; };
#ifndef NSPIN1
  template<typename T, typename G, int nColor_> struct precision_spin_color_mapper<T,G,1,nColor_> { static constexpr int nColor = 3; };
#endif

#ifdef NSPIN4
  // never need block-float format with nSpin=4 fields for arbitrary colors
  template<int nColor_> struct precision_spin_color_mapper<float,short,4,nColor_> { static constexpr int nColor = 3; };
  template<int nColor_> struct precision_spin_color_mapper<float,int8_t,4,nColor_> { static constexpr int nColor = 3; };
#endif

#ifdef NSPIN1
  // never need block-float format with nSpin=4 fields for arbitrary colors
  template<int nColor_> struct precision_spin_color_mapper<float,short,1,nColor_> { static constexpr int nColor = 3; };
  template<int nColor_> struct precision_spin_color_mapper<float,int8_t,1,nColor_> { static constexpr int nColor = 3; };
#endif

#ifndef GPU_MULTIGRID_DOUBLE
#ifdef NSPIN1
  template<int nColor_> struct precision_spin_color_mapper<double,double,1,nColor_> { static constexpr int nColor = 3; };
#endif
#ifdef NSPIN2
  template<int nColor_> struct precision_spin_color_mapper<double,double,2,nColor_> { static constexpr int nColor = 3; };
#endif
#ifdef NSPIN4
  template<int nColor_> struct precision_spin_color_mapper<double,double,4,nColor_> { static constexpr int nColor = 3; };
#endif
#endif

  template <typename Float, typename ghostFloat, QudaFieldOrder order, int Ns>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
                        int nFace, int dagger, MemoryLocation *destination)
  {
#ifndef NSPIN1
    if (a.Ncolor() != 3 && a.Nspin() == 1)
      errorQuda("Ncolor = %d not supported for Nspin = %d fields", a.Ncolor(), a.Nspin());
#endif
    if (a.Ncolor() != 3 && a.Nspin() == 4 && a.Precision() == QUDA_SINGLE_PRECISION &&
        (a.GhostPrecision() == QUDA_HALF_PRECISION || a.GhostPrecision() == QUDA_QUARTER_PRECISION) )
      errorQuda("Ncolor = %d not supported for Nspin = %d fields with precision = %d and ghost_precision = %d",
                a.Ncolor(), a.Nspin(), a.Precision(), a.GhostPrecision());
#ifndef GPU_MULTIGRID_DOUBLE
    if ( a.Ncolor() != 3 && a.Precision() == QUDA_DOUBLE_PRECISION)
      errorQuda("Ncolor = %d not supported for double precision fields", a.Ncolor());
#endif

    if (a.Ncolor() == 3) {
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,3>::nColor>(ghost, a, parity, nFace, dagger, destination);
#ifdef GPU_MULTIGRID
    } else if (a.Ncolor() == 6) {
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,6>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 18) { // Needed for two level free field Wilson
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,18>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 24) { // Needed for K-D staggered Wilson
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,24>::nColor>(ghost, a, parity, nFace, dagger, destination);
#ifdef NSPIN4
    } else if (a.Ncolor() == 32) { // Needed for Wilson
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,32>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 36) { // Needed for three level free field Wilson
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,36>::nColor>(ghost, a, parity, nFace, dagger, destination);
#endif // NSPIN4
#ifdef NSPIN1
    } else if (a.Ncolor() == 64) { // Needed for staggered Nc = 64
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,64>::nColor>(ghost, a, parity, nFace, dagger, destination);
#endif // NSPIN1
    } else if (a.Ncolor() == 72) { // wilson 3 -> 24 nvec, or staggered 3 -> 24 nvec, which could end up getting used for Laplace...
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,72>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 96) { // wilson 3 -> 32 nvec, or staggered Nc = 96
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,96>::nColor>(ghost, a, parity, nFace, dagger, destination);
#ifdef NSPIN1
    } else if (a.Ncolor() == 192) { // staggered 3 -> 64 Nvec
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,192>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 288) { // staggered 3 -> 96 Nvec
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,288>::nColor>(ghost, a, parity, nFace, dagger, destination);
#endif // NSPIN1
    } else if (a.Ncolor() == 576) { // staggered KD free-field or wilson 24 -> 24 nvec
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,576>::nColor>(ghost, a, parity, nFace, dagger, destination);
#ifdef NSPIN4
    } else if (a.Ncolor() == 768) { // wilson 24 -> 32 nvec
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,768>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 1024) { // wilson 32 -> 32 nvec
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,1024>::nColor>(ghost, a, parity, nFace, dagger, destination);
#endif // NSPIN4
#ifdef NSPIN1
    } else if (a.Ncolor() == 1536) { // staggered KD 24 -> 64 nvec
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,1536>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 2304) { // staggered KD 24 -> 96 nvec
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,2304>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 4096) { // staggered 64 -> 64
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,4096>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 6144) { // staggered 64 -> 96 nvec
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,6144>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 9216) { // staggered 96 -> 96 nvec
      GhostPack<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,9216>::nColor>(ghost, a, parity, nFace, dagger, destination);
#endif // NSPIN1
#endif // GPU_MULTIGRID
    } else {
      errorQuda("Unsupported nColor = %d", a.Ncolor());
    }
  }

  // traits used to ensure we only instantiate float4 for spin=4 fields
  template<int nSpin,QudaFieldOrder order_> struct spin_order_mapper { static constexpr QudaFieldOrder order = order_; };
  template<> struct spin_order_mapper<2,QUDA_FLOAT4_FIELD_ORDER> { static constexpr QudaFieldOrder order = QUDA_FLOAT2_FIELD_ORDER; };
  template<> struct spin_order_mapper<1,QUDA_FLOAT4_FIELD_ORDER> { static constexpr QudaFieldOrder order = QUDA_FLOAT2_FIELD_ORDER; };

  template <typename Float, typename ghostFloat, QudaFieldOrder order>
#if defined(NSPIN1) || defined(NSPIN2) || defined(NSPIN4)
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination)
#else
  inline void genericPackGhost(void **, const ColorSpinorField &a, QudaParity, int, int, MemoryLocation *)
#endif
  {
    if (a.Nspin() == 4) {
#ifdef NSPIN4
      genericPackGhost<Float,ghostFloat,order,4>(ghost, a, parity, nFace, dagger, destination);
#else
      errorQuda("nSpin=4 not enabled for this build");
#endif
    } else if (a.Nspin() == 2) {
#ifdef NSPIN2
      if (order == QUDA_FLOAT4_FIELD_ORDER) errorQuda("Field order %d with nSpin = %d not supported", order, a.Nspin());
      genericPackGhost<Float,ghostFloat,spin_order_mapper<2,order>::order,2>(ghost, a, parity, nFace, dagger, destination);
#else
      errorQuda("nSpin=2 not enabled for this build");
#endif
    } else if (a.Nspin() == 1) {
#ifdef NSPIN1
      if (order == QUDA_FLOAT4_FIELD_ORDER) errorQuda("Field order %d with nSpin = %d not supported", order, a.Nspin());
      genericPackGhost<Float,ghostFloat,spin_order_mapper<1,order>::order,1>(ghost, a, parity, nFace, dagger, destination);
#else
      errorQuda("nSpin=1 not enabled for this build");
#endif
    } else {
      errorQuda("Unsupported nSpin = %d", a.Nspin());
    }

  }

  // traits used to ensure we only instantiate double and float templates for non-native fields
  template<typename> struct non_native_precision_mapper { };
  template<> struct non_native_precision_mapper<double> { typedef double type; };
  template<> struct non_native_precision_mapper<float> { typedef float type; };
  template<> struct non_native_precision_mapper<short> { typedef float type; };
  template<> struct non_native_precision_mapper<int8_t> { typedef float type; };

  // traits used to ensure we only instantiate float and lower precision for float4 fields
  template<typename T> struct float4_precision_mapper { typedef T type; };
  template<> struct float4_precision_mapper<double> { typedef float type; };
  template<> struct float4_precision_mapper<short> { typedef float type; };
  template<> struct float4_precision_mapper<int8_t> { typedef float type; };

  template <typename Float, typename ghostFloat>
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination) {

    if (a.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {

      // all precisions, color and spin can use this order
      genericPackGhost<Float,ghostFloat,QUDA_FLOAT2_FIELD_ORDER>(ghost, a, parity, nFace, dagger, destination);

    } else if (a.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {

      // never have double fields here
      if (typeid(Float) != typeid(typename float4_precision_mapper<Float>::type))
        errorQuda("Precision %d not supported for field type %d", a.Precision(), a.FieldOrder());
      if (typeid(ghostFloat) != typeid(typename float4_precision_mapper<ghostFloat>::type))
        errorQuda("Ghost precision %d not supported for field type %d", a.GhostPrecision(), a.FieldOrder());
      genericPackGhost<typename float4_precision_mapper<Float>::type,
                       typename float4_precision_mapper<ghostFloat>::type,
                       QUDA_FLOAT4_FIELD_ORDER>(ghost, a, parity, nFace, dagger, destination);

    } else if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
#ifndef GPU_MULTIGRID // with MG mma we need half-precision AoS exchange support
      if (typeid(Float) != typeid(typename non_native_precision_mapper<Float>::type))
        errorQuda("Precision %d not supported for field type %d", a.Precision(), a.FieldOrder());
      if (typeid(ghostFloat) != typeid(typename non_native_precision_mapper<ghostFloat>::type))
        errorQuda("Ghost precision %d not supported for field type %d", a.GhostPrecision(), a.FieldOrder());
      genericPackGhost<typename non_native_precision_mapper<Float>::type,
                       typename non_native_precision_mapper<ghostFloat>::type,
                       QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(ghost, a, parity, nFace, dagger, destination);
#else
      genericPackGhost<Float, ghostFloat, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(ghost, a, parity, nFace, dagger,
                                                                             destination);
#endif
    } else {
      errorQuda("Unsupported field order = %d", a.FieldOrder());
    }

  }

  void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			int nFace, int dagger, MemoryLocation *destination_) {

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
        genericPackGhost<double,double>(ghost, a, parity, nFace, dagger, destination);
      } else {
        errorQuda("precision = %d and ghost precision = %d not supported", a.Precision(), a.GhostPrecision());
      }
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      if (a.GhostPrecision() == QUDA_SINGLE_PRECISION) {
        genericPackGhost<float,float>(ghost, a, parity, nFace, dagger, destination);
      } else if (a.GhostPrecision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
        genericPackGhost<float,short>(ghost, a, parity, nFace, dagger, destination);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else if (a.GhostPrecision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
        genericPackGhost<float,int8_t>(ghost, a, parity, nFace, dagger, destination);
#else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("precision = %d and ghost precision = %d not supported", a.Precision(), a.GhostPrecision());
      }
    } else if (a.Precision() == QUDA_HALF_PRECISION) {
      if (a.GhostPrecision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
        genericPackGhost<short,short>(ghost, a, parity, nFace, dagger, destination);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("precision = %d and ghost precision = %d not supported", a.Precision(), a.GhostPrecision());
      }
    } else {
      errorQuda("Unsupported precision %d", a.Precision());
    }

  }

} // namespace quda
