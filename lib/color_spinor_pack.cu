#include <color_spinor_field.h>
#include <tune_quda.h>

#include <jitify_helper.cuh>
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

   At present we launch a volume of threads (actually multiples
   thereof for direction / dimension) and thus we have coalesced reads
   but not coalesced writes.  A more optimal implementation will
   launch a surface of threads for each halo giving coalesced writes.
 */

namespace quda {

  template <typename Float, bool block_float, int Ns, int Ms, int Nc, int Mc, typename Arg>
  class GenericPackGhostLauncher : public TunableVectorYZ {
    Arg &arg;
    const ColorSpinorField &meta;
    unsigned int minThreads() const { return arg.volumeCB; }
    bool tuneGridDim() const { return false; }
    bool tuneAuxDim() const { return true; }

  public:
    inline GenericPackGhostLauncher(Arg &arg, const ColorSpinorField &meta, MemoryLocation *destination)
      : TunableVectorYZ((Ns/Ms)*(Nc/Mc), 2*arg.nParity), arg(arg), meta(meta) {

      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        create_jitify_program("kernels/color_spinor_pack.cuh");
#endif
      }

      strcpy(aux,compile_type_str(meta));
      strcat(aux,meta.AuxString());
      switch(meta.GhostPrecision()) {
      case QUDA_DOUBLE_PRECISION:  strcat(aux,",halo_prec=8"); break;
      case QUDA_SINGLE_PRECISION:  strcat(aux,",halo_prec=4"); break;
      case QUDA_HALF_PRECISION:    strcat(aux,",halo_prec=2"); break;
      case QUDA_QUARTER_PRECISION: strcat(aux,",halo_prec=1"); break;
      default: errorQuda("Unexpected precision = %d", meta.GhostPrecision());
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
      strcat(aux,label);
    }

    inline void apply(const qudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	if (arg.nDim == 5) GenericPackGhost<Float,block_float,Ns,Ms,Nc,Mc,5,Arg>(arg);
	else GenericPackGhost<Float,block_float,Ns,Ms,Nc,Mc,4,Arg>(arg);
      } else {
	const TuneParam &tp = tuneLaunch(*this, getTuning(), getVerbosity());
	arg.nParity2dim_threads = arg.nParity*2*tp.aux.x;
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::GenericPackGhostKernel")
          .instantiate(Type<Float>(),block_float,Ns,Ms,Nc,Mc,arg.nDim,(int)tp.aux.x,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        switch(tp.aux.x) {
        case 1:
	  if (arg.nDim == 5) GenericPackGhostKernel<Float,block_float,Ns,Ms,Nc,Mc,5,1,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else GenericPackGhostKernel<Float,block_float,Ns,Ms,Nc,Mc,4,1,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  break;
	case 2:
	  if (arg.nDim == 5) GenericPackGhostKernel<Float,block_float,Ns,Ms,Nc,Mc,5,2,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else GenericPackGhostKernel<Float,block_float,Ns,Ms,Nc,Mc,4,2,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  break;
	case 4:
	  if (arg.nDim == 5) GenericPackGhostKernel<Float,block_float,Ns,Ms,Nc,Mc,5,4,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else GenericPackGhostKernel<Float,block_float,Ns,Ms,Nc,Mc,4,4,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  break;
        }
#endif
      }
    }

    // if doing block float then all spin-color components must be within the same block
    void setColorSpinBlock(TuneParam &param) const {
      param.block.y = (Ns/Ms)*(Nc/Mc);
      param.grid.y = 1;
      param.block.z = 1;
      param.grid.z = arg.nParity*2*param.aux.x;
    }

    bool advanceBlockDim(TuneParam &param) const {
      if (!block_float) {
	return TunableVectorYZ::advanceBlockDim(param);
      } else {
	bool advance = Tunable::advanceBlockDim(param);
	setColorSpinBlock(param); // if doing block float then all spin-color components must be within the same block
	return advance;
      }
    }

    int blockStep() const { return block_float ? 2 : TunableVectorYZ::blockStep(); }
    int blockMin() const { return block_float ? 2 : TunableVectorYZ::blockMin(); }

    bool advanceAux(TuneParam &param) const {
      if (param.aux.x < 4) {
	param.aux.x *= 2;
	const_cast<GenericPackGhostLauncher*>(this)->resizeVector((Ns/Ms)*(Nc/Mc), arg.nParity*2*param.aux.x);
	TunableVectorYZ::initTuneParam(param);
	if (block_float) setColorSpinBlock(param);
	return true;
      }
      param.aux.x = 1;
      const_cast<GenericPackGhostLauncher*>(this)->resizeVector((Ns/Ms)*(Nc/Mc), arg.nParity*2*param.aux.x);
      TunableVectorYZ::initTuneParam(param);
      if (block_float) setColorSpinBlock(param);
      return false;
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }

    virtual void initTuneParam(TuneParam &param) const {
      TunableVectorYZ::initTuneParam(param);
      param.aux = make_int4(1,1,1,1);
      if (block_float) setColorSpinBlock(param);
    }

    virtual void defaultTuneParam(TuneParam &param) const {
      TunableVectorYZ::defaultTuneParam(param);
      param.aux = make_int4(1,1,1,1);
      if (block_float) setColorSpinBlock(param);
    }

    long long flops() const { return 0; }
    long long bytes() const {
      size_t totalBytes = 0;
      for (int d=0; d<4; d++) {
	if (!comm_dim_partitioned(d)) continue;
	totalBytes += arg.nFace*2*Ns*Nc*meta.SurfaceCB(d)*(meta.Precision() + meta.GhostPrecision());
      }
      return totalBytes;
    }
  };

  template <typename Float, typename ghostFloat, QudaFieldOrder order, int Ns, int Nc>
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination)
  {
    typedef typename mapper<Float>::type RegFloat;
    typedef typename colorspinor::FieldOrderCB<RegFloat,Ns,Nc,1,order,Float,ghostFloat> Q;
    Q field(a, nFace, 0, ghost);

    constexpr int spins_per_thread = Ns == 1 ? 1 : 2; // make this autotunable?
    constexpr int colors_per_thread = Nc%2 == 0 ? 2 : 1;
    PackGhostArg<Q> arg(field, a, parity, nFace, dagger);

    constexpr bool block_float_requested = sizeof(Float) == QUDA_SINGLE_PRECISION &&
      (sizeof(ghostFloat) == QUDA_HALF_PRECISION || sizeof(ghostFloat) == QUDA_QUARTER_PRECISION);

    // if we only have short precision for the ghost then this means we have block-float
    constexpr bool block_float = block_float_requested && Nc <= max_block_float_nc;

    // ensure we only compile supported block-float kernels
    constexpr int Nc_ = (block_float_requested &&  Nc > max_block_float_nc) ? max_block_float_nc : Nc;

    if (block_float_requested && Nc > max_block_float_nc)
      errorQuda("Block-float format not supported for Nc = %d", Nc);

    GenericPackGhostLauncher<RegFloat,block_float,Ns,spins_per_thread,Nc_,colors_per_thread,PackGhostArg<Q> >
      launch(arg, a, destination);

    launch.apply(0);
  }

  // traits used to ensure we only instantiate arbitrary colors for nSpin=2,4 fields, and only 3 colors otherwise
  template<typename T, typename G, int nSpin, int nColor_> struct precision_spin_color_mapper { static constexpr int nColor = nColor_; };
#ifndef NSPIN1
  template<typename T, typename G, int nColor_> struct precision_spin_color_mapper<T,G,1,nColor_> { static constexpr int nColor = 3; };
#endif

#ifdef NSPIN4
  // never need block-float format with nSpin=4 fields for arbitrary colors
  template<int nColor_> struct precision_spin_color_mapper<float,short,4,nColor_> { static constexpr int nColor = 3; };
  template<int nColor_> struct precision_spin_color_mapper<float,char,4,nColor_> { static constexpr int nColor = 3; };
#endif

#ifdef NSPIN1
  // never need block-float format with nSpin=4 fields for arbitrary colors
  template<int nColor_> struct precision_spin_color_mapper<float,short,1,nColor_> { static constexpr int nColor = 3; };
  template<int nColor_> struct precision_spin_color_mapper<float,char,1,nColor_> { static constexpr int nColor = 3; };
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
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination) {

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
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,3>::nColor>(ghost, a, parity, nFace, dagger, destination);
#ifdef GPU_MULTIGRID
    } else if (a.Ncolor() == 6) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,6>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 18) { // Needed for two level free field Wilson
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,18>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 24) { // Needed for K-D staggered Wilson
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,24>::nColor>(ghost, a, parity, nFace, dagger, destination);
#ifdef NSPIN4
    } else if (a.Ncolor() == 32) { // Needed for Wilson
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,32>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 36) { // Needed for three level free field Wilson
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,36>::nColor>(ghost, a, parity, nFace, dagger, destination);
#endif // NSPIN4
#ifdef NSPIN1
    } else if (a.Ncolor() == 64) { // Needed for staggered Nc = 64
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,64>::nColor>(ghost, a, parity, nFace, dagger, destination);
#endif // NSPIN1
    } else if (a.Ncolor() == 72) { // wilson 3 -> 24 nvec, or staggered 3 -> 24 nvec, which could end up getting used for Laplace...
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,72>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 96) { // wilson 3 -> 32 nvec, or staggered Nc = 96
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,96>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 576) { // staggered KD free-field or wilson 24 -> 24 nvec
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,576>::nColor>(ghost, a, parity, nFace, dagger, destination);
#ifdef NSPIN4
    } else if (a.Ncolor() == 768) { // wilson 24 -> 32 nvec
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,768>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 1024) { // wilson 32 -> 32 nvec
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,1024>::nColor>(ghost, a, parity, nFace, dagger, destination);
#endif // NSPIN4
#ifdef NSPIN1
    } else if (a.Ncolor() == 1536) { // staggered KD 24 -> 64 nvec
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,1536>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 2304) { // staggered KD 24 -> 96 nvec
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,2304>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 4096) { // staggered 64 -> 64
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,4096>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 6144) { // staggered 64 -> 96 nvec
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,6144>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 9216) { // staggered 96 -> 96 nvec
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,9216>::nColor>(ghost, a, parity, nFace, dagger, destination);
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
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination) {

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
  template<> struct non_native_precision_mapper<char> { typedef float type; };

  // traits used to ensure we only instantiate float and lower precision for float4 fields
  template<typename T> struct float4_precision_mapper { typedef T type; };
  template<> struct float4_precision_mapper<double> { typedef float type; };
  template<> struct float4_precision_mapper<short> { typedef float type; };
  template<> struct float4_precision_mapper<char> { typedef float type; };

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
      if (typeid(Float) != typeid(typename non_native_precision_mapper<Float>::type))
        errorQuda("Precision %d not supported for field type %d", a.Precision(), a.FieldOrder());
      if (typeid(ghostFloat) != typeid(typename non_native_precision_mapper<ghostFloat>::type))
        errorQuda("Ghost precision %d not supported for field type %d", a.GhostPrecision(), a.FieldOrder());
      genericPackGhost<typename non_native_precision_mapper<Float>::type,
                       typename non_native_precision_mapper<ghostFloat>::type,
                       QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(ghost, a, parity, nFace, dagger, destination);
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
        genericPackGhost<float,char>(ghost, a, parity, nFace, dagger, destination);
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
