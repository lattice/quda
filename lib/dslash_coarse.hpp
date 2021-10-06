
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <uint_to_char.h>
#include <worker.h>
#include <tunable_nd.h>
#include <kernels/dslash_coarse.cuh>

namespace quda {

  template <typename Float, typename yFloat, typename ghostFloat, int Ns, int Nc, bool dslash, bool clover, bool dagger,
            DslashType type>
  class DslashCoarse : public TunableKernel3D
  {

    static constexpr int nDim = 4;

    ColorSpinorField &out;
    const ColorSpinorField &inA;
    const ColorSpinorField &inB;
    const GaugeField &Y;
    const GaugeField &X;
    const double kappa;
    const int parity;
    const int nParity;
    const int nSrc;

    const int max_color_col_stride = 8;
    mutable int color_col_stride;
    mutable int dim_threads;

    long long flops() const
    {
      return ((dslash*2*nDim+clover*1)*(8*Ns*Nc*Ns*Nc)-2*Ns*Nc)*nParity*(long long)out.VolumeCB();
    }
    long long bytes() const
    {
     return (dslash||clover) * out.Bytes() + dslash*8*inA.Bytes() + clover*inB.Bytes() +
       nSrc*nParity*(dslash*Y.Bytes()*Y.VolumeCB()/(2*Y.Stride()) + clover*X.Bytes()/2);
    }

    unsigned int sharedBytesPerThread() const { return (sizeof(complex<Float>) * colors_per_thread(Nc, dim_threads)); }
    bool tuneAuxDim() const { return true; } // Do tune the aux dimensions
    unsigned int minThreads() const { return color_col_stride * X.VolumeCB(); }

    /**
       @param Helper function to check that the present launch parameters are valid
    */
    bool checkParam(const TuneParam &param) const
    {
      return ((color_col_stride == 1 || minThreads() % (unsigned)device::warp_size() == 0)
              &&                                          // active threads must be a multiple of the warp
              param.block.x % device::warp_size() == 0 && // block size must be a multiple of the warp
              Nc % color_col_stride == 0 &&               // number of colors must be divisible by the split
              param.grid.x < device::max_grid_size(0));   // ensure the resulting grid size valid
    }

    bool advanceColorStride(TuneParam &param) const
    {
      bool valid = false;

      while (param.aux.x < max_color_col_stride) {
        param.aux.x *= 2;
        color_col_stride = param.aux.x;
        param.grid.x
          = (minThreads() + param.block.x - 1) / param.block.x; // grid size changed since minThreads has been updated
        valid = checkParam(param);
        if (valid) break;
      }

      if (!valid) {
        // reset color column stride if too large or not divisible
        param.aux.x = 1;
        color_col_stride = param.aux.x;
        param.grid.x
          = (minThreads() + param.block.x - 1) / param.block.x; // grid size changed since minThreads has been updated
      }

      return valid;
    }

    bool advanceDimThreads(TuneParam &param) const
    {
      bool rtn;
      if (2 * param.aux.y <= nDim && param.block.x * param.block.y * dim_threads * 2 <= device::max_threads_per_block()) {
        param.aux.y *= 2;
        rtn = true;
      } else {
        param.aux.y = 1;
        rtn = false;
      }

      dim_threads = param.aux.y;
      // need to reset z-block/grid size/shared_bytes since dim_threads has changed
      resizeStep(step_y, 2 * dim_threads);
      resizeVector(vector_length_y, 2 * dim_threads * 2 * (Nc / colors_per_thread(Nc, dim_threads)));
      TunableKernel3D::initTuneParam(param);

      return rtn;
    }

#ifndef QUDA_FAST_COMPILE_DSLASH
    bool advanceAux(TuneParam &param) const { return advanceColorStride(param) || advanceDimThreads(param); }
#else
    bool advanceAux(TuneParam &) const { return false; }
#endif

    void initTuneParam(TuneParam &param) const
    {
      color_col_stride = 1;
      dim_threads = 1;
      resizeStep(step_y, 2 * dim_threads); // 2 is forwards/backwards
      resizeVector(vector_length_y, 2 * dim_threads * 2 * (Nc / colors_per_thread(Nc, dim_threads)));
      TunableKernel3D::initTuneParam(param);
      param.aux = make_int4(color_col_stride, dim_threads, 1, 1);
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      color_col_stride = 1;
      dim_threads = 1;
      resizeStep(step_y, 2 * dim_threads); // 2 is forwards/backwards
      resizeVector(vector_length_y, 2 * dim_threads * 2 * (Nc / colors_per_thread(Nc, dim_threads)));
      TunableKernel3D::defaultTuneParam(param);
      param.aux = make_int4(color_col_stride, dim_threads, 1, 1);

      // ensure that the default x block size is divisible by the warpSize
      param.block.x = device::warp_size();
      param.grid.x = (minThreads() + param.block.x - 1) / param.block.x;
      param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
    }

  public:
    DslashCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB, const GaugeField &Y,
                 const GaugeField &X, double kappa, int parity, MemoryLocation *halo_location) :
      TunableKernel3D(out, out.SiteSubset() * (out.Ndim() == 5 ? out.X(4) : 1), 1),
      out(out),
      inA(inA),
      inB(inB),
      Y(Y),
      X(X),
      kappa(kappa),
      parity(parity),
      nParity(out.SiteSubset()),
      nSrc(out.Ndim() == 5 ? out.X(4) : 1),
      color_col_stride(-1)
    {
      strcpy(aux, (std::string("policy_kernel,") + aux).c_str());
      strcat(aux, comm_dim_partitioned_string());

      switch(type) {
      case DSLASH_INTERIOR: strcat(aux,",interior"); break;
      case DSLASH_EXTERIOR: strcat(aux,",exterior"); break;
      case DSLASH_FULL:     strcat(aux,",full"); break;
      }

      // record the location of where each pack buffer is in [2*dim+dir] ordering
      // 0 - no packing
      // 1 - pack to local GPU memory
      // 2 - pack to local mapped CPU memory
      // 3 - pack to remote mapped GPU memory
      if (doHalo<type>()) {
        char label[15] = ",halo=";
        for (int dim=0; dim<4; dim++) {
          for (int dir=0; dir<2; dir++) {
            label[2*dim+dir+6] = !comm_dim_partitioned(dim) ? '0' : halo_location[2*dim+dir] == Device ? '1' : halo_location[2*dim+dir] == Host ? '2' : '3';
          }
        }
        label[14] = '\0';
        strcat(aux,label);
      }

#ifdef QUDA_FAST_COMPILE_DSLASH
      strcat(aux, ",fast_compile");
#endif

      apply(device::get_default_stream());
    }

    template <int color_stride, int dim_stride, QudaFieldOrder csOrder = QUDA_FLOAT2_FIELD_ORDER,
              QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER>
    using Arg = DslashCoarseArg<dslash, clover, dagger, type, color_stride, dim_stride, Float, yFloat, ghostFloat, Ns,
                                Nc, csOrder, gOrder>;

    void apply(const qudaStream_t &stream)
    {
      const TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      color_col_stride = tp.aux.x;
      dim_threads = tp.aux.y;
      resizeVector(vector_length_y, 2 * dim_threads * 2 * (Nc / colors_per_thread(Nc, dim_threads)));
      if (!checkParam(tp)) errorQuda("Invalid launch param");

      if (out.Location() == QUDA_CPU_FIELD_LOCATION) {
        if (out.FieldOrder() != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER || Y.FieldOrder() != QUDA_QDP_GAUGE_ORDER)
          errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", inA.FieldOrder(), Y.FieldOrder());

        launch_host<CoarseDslash>(
          tp, stream,
          Arg<1, 1, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER, QUDA_QDP_GAUGE_ORDER>(out, inA, inB, Y, X, (Float)kappa, parity));
      } else {
        if (out.FieldOrder() != QUDA_FLOAT2_FIELD_ORDER || Y.FieldOrder() != QUDA_FLOAT2_GAUGE_ORDER)
          errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", inA.FieldOrder(), Y.FieldOrder());

        switch (tp.aux.y) { // dimension gather parallelisation
        case 1:
          switch (tp.aux.x) { // this is color_col_stride
          case 1: launch_device<CoarseDslash>(tp, stream, Arg<1, 1>(out, inA, inB, Y, X, (Float)kappa, parity)); break;
#ifndef QUDA_FAST_COMPILE_DSLASH
          case 2: launch_device<CoarseDslash>(tp, stream, Arg<2, 1>(out, inA, inB, Y, X, (Float)kappa, parity)); break;
          case 4: launch_device<CoarseDslash>(tp, stream, Arg<4, 1>(out, inA, inB, Y, X, (Float)kappa, parity)); break;
          case 8: launch_device<CoarseDslash>(tp, stream, Arg<8, 1>(out, inA, inB, Y, X, (Float)kappa, parity)); break;
#endif
          default: errorQuda("Color column stride %d not valid", tp.aux.x);
          }
          break;
#ifndef QUDA_FAST_COMPILE_DSLASH
        case 2:
          switch (tp.aux.x) { // this is color_col_stride
          case 1: launch_device<CoarseDslash>(tp, stream, Arg<1, 2>(out, inA, inB, Y, X, (Float)kappa, parity)); break;
          case 2: launch_device<CoarseDslash>(tp, stream, Arg<2, 2>(out, inA, inB, Y, X, (Float)kappa, parity)); break;
          case 4: launch_device<CoarseDslash>(tp, stream, Arg<4, 2>(out, inA, inB, Y, X, (Float)kappa, parity)); break;
          case 8: launch_device<CoarseDslash>(tp, stream, Arg<8, 2>(out, inA, inB, Y, X, (Float)kappa, parity)); break;
          default: errorQuda("Color column stride %d not valid", tp.aux.x);
          }
          break;
        case 4:
          switch (tp.aux.x) { // this is color_col_stride
          case 1: launch_device<CoarseDslash>(tp, stream, Arg<1, 4>(out, inA, inB, Y, X, (Float)kappa, parity)); break;
          case 2: launch_device<CoarseDslash>(tp, stream, Arg<2, 4>(out, inA, inB, Y, X, (Float)kappa, parity)); break;
          case 4: launch_device<CoarseDslash>(tp, stream, Arg<4, 4>(out, inA, inB, Y, X, (Float)kappa, parity)); break;
          case 8: launch_device<CoarseDslash>(tp, stream, Arg<8, 4>(out, inA, inB, Y, X, (Float)kappa, parity)); break;
          default: errorQuda("Color column stride %d not valid", tp.aux.x);
          }
          break;
#endif
        default:
          errorQuda("Invalid dimension thread splitting %d", tp.aux.y);
        }
      }
    }

    void preTune() { out.backup(); }
    void postTune() { out.restore(); }
  };

  template <typename Float, typename yFloat, typename ghostFloat, bool dagger, int coarseColor, int coarseSpin>
  inline void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
			  const GaugeField &Y, const GaugeField &X, double kappa, int parity, bool dslash,
			  bool clover, DslashType type, MemoryLocation *halo_location)
  {
    // for now we never instantiate any template except DSLASH_FULL, since never overlap comms and compute and do not properly support additive Schwarz here
    if (dslash) {
      if (clover) {

        if (type == DSLASH_FULL) {
          DslashCoarse<Float, yFloat, ghostFloat, coarseSpin, coarseColor, true, true, dagger, DSLASH_FULL> dslash(
            out, inA, inB, Y, X, kappa, parity, halo_location);
        } else { errorQuda("Dslash type %d not instantiated", type); }

      } else { // plain dslash

        if (type == DSLASH_FULL) {
          DslashCoarse<Float, yFloat, ghostFloat, coarseSpin, coarseColor, true, false, dagger, DSLASH_FULL> dslash(
            out, inA, inB, Y, X, kappa, parity, halo_location);
        } else { errorQuda("Dslash type %d not instantiated", type); }

      }
    } else {

      if (type == DSLASH_EXTERIOR) errorQuda("Cannot call halo on pure clover kernel");
      if (clover) {
        DslashCoarse<Float, yFloat, ghostFloat, coarseSpin, coarseColor, false, true, dagger, DSLASH_FULL> dslash(
          out, inA, inB, Y, X, kappa, parity, halo_location);
      } else {
        errorQuda("Unsupported dslash=false clover=false");
      }

    }
  }

  // template on the number of coarse colors
  template <typename Float, typename yFloat, typename ghostFloat, bool dagger>
  inline void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
			  const GaugeField &Y, const GaugeField &X, double kappa, int parity, bool dslash,
			  bool clover, DslashType type, MemoryLocation *halo_location)
  {
    if (Y.FieldOrder() != X.FieldOrder())
      errorQuda("Field order mismatch Y = %d, X = %d", Y.FieldOrder(), X.FieldOrder());

    if (inA.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch inA = %d, out = %d", inA.FieldOrder(), out.FieldOrder());

    if (inA.Nspin() != 2)
      errorQuda("Unsupported number of coarse spins %d\n", inA.Nspin());

#ifdef NSPIN4
    if (inA.Ncolor() == 6) { // free field Wilson
      ApplyCoarse<Float,yFloat,ghostFloat,dagger,6,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, type, halo_location);
    } else
#endif // NSPIN4
    if (inA.Ncolor() == 24) {
      ApplyCoarse<Float,yFloat,ghostFloat,dagger,24,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, type, halo_location);
#ifdef NSPIN4
    } else if (inA.Ncolor() == 32) {
      ApplyCoarse<Float,yFloat,ghostFloat,dagger,32,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, type, halo_location);
#endif // NSPIN4
#ifdef NSPIN1
    } else if (inA.Ncolor() == 64) {
      ApplyCoarse<Float,yFloat,ghostFloat,dagger,64,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, type, halo_location);
    } else if (inA.Ncolor() == 96) {
      ApplyCoarse<Float,yFloat,ghostFloat,dagger,96,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, type, halo_location);
#endif // NSPIN1
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // this is the Worker pointer that may have issue additional work
  // while we're waiting on communication to finish
  namespace dslash {
    extern Worker* aux_worker;
  }

  enum class DslashCoarsePolicy {
    DSLASH_COARSE_BASIC,          // stage both sends and recvs in host memory using memcpys
    DSLASH_COARSE_ZERO_COPY_PACK, // zero copy write pack buffers
    DSLASH_COARSE_ZERO_COPY_READ, // zero copy read halos in dslash kernel
    DSLASH_COARSE_ZERO_COPY,      // full zero copy
    DSLASH_COARSE_GDR_SEND,       // GDR send
    DSLASH_COARSE_GDR_RECV,       // GDR recv
    DSLASH_COARSE_GDR,             // full GDR
    DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV, // zero copy write and GDR recv
    DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ, // GDR send and zero copy read
    DSLASH_COARSE_POLICY_DISABLED
  };

  template <bool dagger>
  struct DslashCoarseLaunch {

    ColorSpinorField &out;
    const ColorSpinorField &inA;
    const ColorSpinorField &inB;
    const GaugeField &Y;
    const GaugeField &X;
    double kappa;
    int parity;
    bool dslash;
    bool clover;
    const int *commDim;
    const QudaPrecision halo_precision;

    inline DslashCoarseLaunch(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
			      const GaugeField &Y, const GaugeField &X, double kappa, int parity,
			      bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision)
      : out(out), inA(inA), inB(inB), Y(Y), X(X), kappa(kappa), parity(parity),
	dslash(dslash), clover(clover), commDim(commDim),
        halo_precision(halo_precision == QUDA_INVALID_PRECISION ? Y.Precision() : halo_precision) { }

    /**
       @brief Execute the coarse dslash using the given policy
     */
    inline void operator()(DslashCoarsePolicy policy)
    {
      if (inA.V() == out.V()) errorQuda("Aliasing pointers");

      // check all precisions match
      QudaPrecision precision = checkPrecision(out, inA, inB);
      checkPrecision(Y, X);

      // check all locations match
      checkLocation(out, inA, inB, Y, X);

      int comm_sum = 4;
      if (commDim) for (int i=0; i<4; i++) comm_sum -= (1-commDim[i]);
      if (comm_sum != 4 && comm_sum != 0) errorQuda("Unsupported comms %d", comm_sum);
      bool comms = comm_sum;

      MemoryLocation pack_destination[2*QUDA_MAX_DIM]; // where we will pack the ghost buffer to
      MemoryLocation halo_location[2*QUDA_MAX_DIM]; // where we load the halo from
      for (int i=0; i<2*QUDA_MAX_DIM; i++) {
	pack_destination[i] = (policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY ||
			       policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV) ? Host : Device;
	halo_location[i] = (policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_READ || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY ||
			    policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ) ? Host : Device;
      }
      bool gdr_send = (policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND || policy == DslashCoarsePolicy::DSLASH_COARSE_GDR ||
		       policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ) ? true : false;
      bool gdr_recv = (policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_RECV || policy == DslashCoarsePolicy::DSLASH_COARSE_GDR ||
		       policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV) ? true : false;

      // disable peer-to-peer if doing a zero-copy policy (temporary)
      if ( policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK ||
	   policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_READ ||
	   policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY ||
	   policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV ||
	   policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ) comm_enable_peer2peer(false);

      if (dslash && comm_partitioned() && comms) {
	const int nFace = 1;
        inA.exchangeGhost((QudaParity)(inA.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? (1 - parity) : 0), nFace, dagger,
                          pack_destination, halo_location, gdr_send, gdr_recv, halo_precision);
      }

      if (dslash::aux_worker) dslash::aux_worker->apply(device::get_default_stream());

      if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
	if (Y.Precision() != QUDA_DOUBLE_PRECISION)
          errorQuda("Y Precision %d not supported", Y.Precision());
	if (halo_precision != QUDA_DOUBLE_PRECISION)
          errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_precision, precision, Y.Precision());
	ApplyCoarse<double,double,double,dagger>(out, inA, inB, Y, X, kappa, parity, dslash, clover,
                                                 comms ? DSLASH_FULL : DSLASH_INTERIOR, halo_location);
#else
	errorQuda("Double precision multigrid has not been enabled");
#endif
      } else if (precision == QUDA_SINGLE_PRECISION) {
        if (Y.Precision() == QUDA_SINGLE_PRECISION) {
          if (halo_precision == QUDA_SINGLE_PRECISION) {
            ApplyCoarse<float,float,float,dagger>(out, inA, inB, Y, X, kappa, parity, dslash, clover,
                                                  comms ? DSLASH_FULL : DSLASH_INTERIOR, halo_location);
          } else {
            errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_precision, precision, Y.Precision());
          }
        } else if (Y.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
          if (halo_precision == QUDA_HALF_PRECISION) {
            ApplyCoarse<float,short,short,dagger>(out, inA, inB, Y, X, kappa, parity, dslash, clover,
                                                  comms ? DSLASH_FULL : DSLASH_INTERIOR, halo_location);
          } else if (halo_precision == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
            ApplyCoarse<float, short, int8_t, dagger>(out, inA, inB, Y, X, kappa, parity, dslash, clover,
                                                      comms ? DSLASH_FULL : DSLASH_INTERIOR, halo_location);
#else
            errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
          } else {
            errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_precision, precision, Y.Precision());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
        } else {
          errorQuda("Unsupported precision %d\n", Y.Precision());
        }
      } else {
	errorQuda("Unsupported precision %d\n", Y.Precision());
      }

      if (dslash && comm_partitioned() && comms) inA.bufferIndex = (1 - inA.bufferIndex);
      comm_enable_peer2peer(true);
    }
  };

  static bool dslash_init = false;
  static std::vector<DslashCoarsePolicy> policies(static_cast<int>(DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED), DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED);
  static int first_active_policy=static_cast<int>(DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED);

  // string used as a tunekey to ensure we retune if the dslash policy env changes
  static char policy_string[TuneKey::aux_n];

  static inline void enable_policy(DslashCoarsePolicy p) { policies[static_cast<std::size_t>(p)] = p; }

  static inline void disable_policy(DslashCoarsePolicy p)
  {
    policies[static_cast<std::size_t>(p)] = DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED;
  }

  template <typename Launch>
  class DslashCoarsePolicyTune : public Tunable {

   Launch &dslash;

   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   bool tuneAuxDim() const { return true; } // Do tune the aux dimensions.
   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

 public:
   inline DslashCoarsePolicyTune(Launch &dslash) : dslash(dslash)
   {
      if (!dslash_init) {

	static char *dslash_policy_env = getenv("QUDA_ENABLE_DSLASH_COARSE_POLICY");

	if (dslash_policy_env) { // set the policies to tune for explicitly
	  std::stringstream policy_list(dslash_policy_env);

	  int policy_;
	  while (policy_list >> policy_) {
	    DslashCoarsePolicy dslash_policy = static_cast<DslashCoarsePolicy>(policy_);

	    // check this is a valid policy choice
	    if ( (dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND ||
            dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_RECV ||
		        dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_GDR ||
            dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV ||
		        dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ) && !comm_gdr_enabled() ) {
	      errorQuda("Cannot select a GDR policy %d unless QUDA_ENABLE_GDR is set", static_cast<int>(dslash_policy));
	    }

	    enable_policy(dslash_policy);
	    first_active_policy = policy_ < first_active_policy ? policy_ : first_active_policy;
	    if (policy_list.peek() == ',') policy_list.ignore();
	  }
	  if(first_active_policy == static_cast<int>(DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED)) errorQuda("No valid policy found in QUDA_ENABLE_DSLASH_COARSE_POLICY");
	} else {
	  first_active_policy = 0;
	  enable_policy(DslashCoarsePolicy::DSLASH_COARSE_BASIC);
	  enable_policy(DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK);
	  enable_policy(DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_READ);
	  enable_policy(DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY);
	  if (comm_gdr_enabled()) {
	    enable_policy(DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND);
	    enable_policy(DslashCoarsePolicy::DSLASH_COARSE_GDR_RECV);
	    enable_policy(DslashCoarsePolicy::DSLASH_COARSE_GDR);
	    enable_policy(DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV);
	    enable_policy(DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ);
	  }
	}

        // construct string specifying which policies have been enabled
        strcat(policy_string, ",pol=");
        for (int i = 0; i < (int)DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED; i++) {
          strcat(policy_string, (int)policies[i] == i ? "1" : "0");
        }

        dslash_init = true;
      }

      strcpy(aux, "policy,");
      if (dslash.dslash) strcat(aux, "dslash");
      strcat(aux, dslash.clover ? "clover," : ",");
      strcat(aux, dslash.inA.AuxString());
      strcat(aux, ",gauge_prec=");

      char prec_str[8];
      i32toa(prec_str, dslash.Y.Precision());
      strcat(aux, prec_str);
      strcat(aux, ",halo_prec=");
      i32toa(prec_str, dslash.halo_precision);
      strcat(aux, prec_str);
      strcat(aux, comm_dim_partitioned_string(dslash.commDim));
      strcat(aux, comm_dim_topology_string());
      strcat(aux, comm_config_string()); // and change in P2P/GDR will be stored as a separate tunecache entry
      strcat(aux, policy_string);        // any change in policies enabled will be stored as a separate entry

      int comm_sum = 4;
      if (dslash.commDim)
        for (int i = 0; i < 4; i++) comm_sum -= (1 - dslash.commDim[i]);
      strcat(aux, comm_sum ? ",full" : ",interior");

#ifdef QUDA_FAST_COMPILE_DSLASH
      strcat(aux, ",fast_compile");
#endif

      // before we do policy tuning we must ensure the kernel
      // constituents have been tuned since we can't do nested tuning
      if (!tuned()) {
        disableProfileCount();
	for (auto &i : policies) if(i!= DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED) dslash(i);
	enableProfileCount();
	setPolicyTuning(true);
      }
   }

   virtual ~DslashCoarsePolicyTune() { setPolicyTuning(false); }

   inline void apply(const qudaStream_t &)
   {
     TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

     if (tp.aux.x >= (int)policies.size()) errorQuda("Requested policy that is outside of range");
     if (policies[tp.aux.x] == DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED ) errorQuda("Requested policy is disabled");
     dslash(policies[tp.aux.x]);
   }

   int tuningIter() const { return 10; }

   bool advanceAux(TuneParam &param) const
   {
    while ((unsigned)param.aux.x < policies.size()-1) {
      param.aux.x++;
      if (policies[param.aux.x] != DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED) return true;
    }
    param.aux.x = 0;
    return false;
   }

   bool advanceTuneParam(TuneParam &param) const { return advanceAux(param); }

   void initTuneParam(TuneParam &param) const
   {
     Tunable::initTuneParam(param);
     param.aux = make_int4(first_active_policy, 0, 0, 0);
   }

   void defaultTuneParam(TuneParam &param) const
   {
     Tunable::defaultTuneParam(param);
     param.aux = make_int4(first_active_policy, 0, 0, 0);
   }

   TuneKey tuneKey() const {
     return TuneKey(dslash.inA.VolString(), typeid(*this).name(), aux);
   }

   long long flops() const {
     int nDim = 4;
     int Ns = dslash.inA.Nspin();
     int Nc = dslash.inA.Ncolor();
     int nParity = dslash.inA.SiteSubset();
     long long volumeCB = dslash.inA.VolumeCB();
     return ((dslash.dslash*2*nDim+dslash.clover*1)*(8*Ns*Nc*Ns*Nc)-2*Ns*Nc)*nParity*volumeCB;
   }

   long long bytes() const {
     int nParity = dslash.inA.SiteSubset();
     return (dslash.dslash||dslash.clover) * dslash.out.Bytes() +
       dslash.dslash*8*dslash.inA.Bytes() + dslash.clover*dslash.inB.Bytes() +
       nParity*(dslash.dslash*dslash.Y.Bytes()*dslash.Y.VolumeCB()/(2*dslash.Y.Stride())
		+ dslash.clover*dslash.X.Bytes()/2);
     // multiply Y by volume / stride to correct for pad
   }
  };

} // namespace quda
