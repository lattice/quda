#include <color_spinor_field.h>
#include <uint_to_char.h>
#include <vector>
#include <assert.h>
#include <utility>

#include <kernels/block_orthogonalize.cuh>
#include <tunable_block_reduction.h>
#include <target_device.h>
#include <power_of_two_array.h>
namespace quda {

  struct OrthoAggregates {
    // List of block sizes we wish to instantiate.  The required block
    // size is equal to number of fine points per aggregate, rounded
    // up to a whole power of two.  So for example, 2x2x2x2 and
    // 3x3x3x1 aggregation would both use the same block size 32
#ifndef QUDA_FAST_COMPILE_REDUCE
    using array_type = PowerOfTwoArray<device::warp_size(), device::max_block_size()>;
#else
    using array_type = PowerOfTwoArray<device::max_block_size(), device::max_block_size()>;
#endif

    static constexpr array_type block = array_type();


    /**
     *  @brief Return the first power of two block that is larger than the required size
     */
    static unsigned int block_mapper(unsigned int raw_block)
    {
      for (unsigned int b=0; b < block.size();  b++) if (raw_block <= block[b]) return block[b];
        errorQuda("Invalid raw block size %d\n", raw_block);
        return 0;
      }
  };

  constexpr OrthoAggregates::array_type  OrthoAggregates::block;

  using namespace quda::colorspinor;

  // B fields in general use float2 ordering except for fine-grid Wilson
  template <typename store_t, int nSpin, int nColor> struct BOrder { static constexpr QudaFieldOrder order = QUDA_FLOAT2_FIELD_ORDER; };
  template<> struct BOrder<float, 4, 3> { static constexpr QudaFieldOrder order = QUDA_FLOAT4_FIELD_ORDER; };
#ifdef FLOAT8
  template<> struct BOrder<short, 4, 3> { static constexpr QudaFieldOrder order = QUDA_FLOAT8_FIELD_ORDER; };
#else
  template<> struct BOrder<short, 4, 3> { static constexpr QudaFieldOrder order = QUDA_FLOAT4_FIELD_ORDER; };
#endif

  template <typename vFloat, typename bFloat, int nSpin, int spinBlockSize, int nColor_, int coarseSpin, int nVec>
  class BlockOrtho : public TunableBlock2D {

    using real = typename mapper<vFloat>::type;
    // we only support block-format on fine grid where Ncolor=3
    static constexpr int nColor = isFixed<bFloat>::value ? 3 : nColor_;
    static constexpr int chiral_blocks = nSpin == 1 ? 2 : nSpin / spinBlockSize;
    template <bool is_device, typename Rotator, typename Vector> using Arg = BlockOrthoArg<is_device, vFloat, Rotator, Vector, nSpin, nColor, coarseSpin, nVec>;

    ColorSpinorField &V;
    const std::vector<ColorSpinorField*> B;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const int *geo_bs;
    const int n_block_ortho;
    int aggregate_size;
    int nBlock;

  public:
    BlockOrtho(ColorSpinorField &V, const std::vector<ColorSpinorField *> B, const int *fine_to_coarse,
               const int *coarse_to_fine, const int *geo_bs, const int n_block_ortho) :
      TunableBlock2D(V, chiral_blocks),
      V(V),
      B(B),
      fine_to_coarse(fine_to_coarse),
      coarse_to_fine(coarse_to_fine),
      geo_bs(geo_bs),
      n_block_ortho(n_block_ortho)
    {
      if (nColor_ != nColor)
        errorQuda("Number of colors %d not supported with this precision %lu\n", nColor_, sizeof(bFloat));

      strcat(aux,",block_size=");

      aggregate_size = 1;
      char geo_str[16];
      for (int d = 0; d < V.Ndim(); d++) {
        aggregate_size *= geo_bs[d];
        i32toa(geo_str, geo_bs[d]);
        strcat(aux, geo_str);
        if (d < V.Ndim() - 1) strcat(aux, "x");
      }
      nBlock = (V.Volume()/aggregate_size) * chiral_blocks;

      strcat(aux, ",n_block_ortho=");
      char n_ortho_str[2];
      i32toa(n_ortho_str, n_block_ortho);
      strcat(aux, n_ortho_str);
      strcat(aux, ",mVec=");
      char mvec_str[3];
      int active_x_threads = (aggregate_size / 2) * (nSpin == 1 ? 1 : V.SiteSubset());
      i32toa(mvec_str, tile_size<nColor, nVec>(OrthoAggregates::block_mapper(active_x_threads)));
      strcat(aux, mvec_str);
    }

    template <typename Rotator, typename Vector, std::size_t... S>
    void CPU(const TuneParam &tp, const qudaStream_t &stream,
             const std::vector<ColorSpinorField*> &B, std::index_sequence<S...>)
    {
      Arg<false, Rotator, Vector> arg(V, fine_to_coarse, coarse_to_fine, QUDA_INVALID_PARITY, geo_bs, n_block_ortho, V, B[S]...);
      launch_host<BlockOrtho_, OrthoAggregates>(tp, stream, arg);
    }

    template <typename Rotator, typename Vector, std::size_t... S>
    void GPU(const TuneParam &tp, const qudaStream_t &stream,
             const std::vector<ColorSpinorField*> &B, std::index_sequence<S...>)
    {
      Arg<true, Rotator, Vector> arg(V, fine_to_coarse, coarse_to_fine, QUDA_INVALID_PARITY, geo_bs, n_block_ortho, V, B[S]...);
      arg.swizzle_factor = tp.aux.x;
      launch_device<BlockOrtho_, OrthoAggregates>(tp, stream, arg);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (V.Location() == QUDA_CPU_FIELD_LOCATION) {
        if (V.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER && B[0]->FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
          typedef FieldOrderCB<real,nSpin,nColor,nVec,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,vFloat,vFloat,DISABLE_GHOST> Rotator;
          typedef FieldOrderCB<real,nSpin,nColor,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,bFloat,bFloat,DISABLE_GHOST> Vector;
          CPU<Rotator,Vector>(tp, stream, B, std::make_index_sequence<nVec>());
        } else {
          errorQuda("Unsupported field order %d\n", V.FieldOrder());
        }
      } else {
        if (V.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER && B[0]->FieldOrder() == BOrder<bFloat,nSpin,nColor>::order) {
          typedef FieldOrderCB<real,nSpin,nColor,nVec,QUDA_FLOAT2_FIELD_ORDER,vFloat,vFloat,DISABLE_GHOST> Rotator;
          typedef FieldOrderCB<real,nSpin,nColor,1,BOrder<bFloat,nSpin,nColor>::order,bFloat,bFloat,DISABLE_GHOST,isFixed<bFloat>::value> Vector;
          GPU<Rotator,Vector>(tp, stream, B, std::make_index_sequence<nVec>());
        } else {
          errorQuda("Unsupported field order V=%d B=%d\n", V.FieldOrder(), B[0]->FieldOrder());
        }
      }
    }

#ifdef SWIZZLE
    bool advanceAux(TuneParam &param) const
    {
      if (param.aux.x < 2 * device::processor_count()) {
        param.aux.x++;
	return true;
      } else {
        param.aux.x = 1;
	return false;
      }
    }
#else
    bool advanceAux(TuneParam &) const { return false; }
#endif

    bool advanceTuneParam(TuneParam &param) const
    {
      if (V.Location() == QUDA_CUDA_FIELD_LOCATION) {
	return advanceSharedBytes(param) || advanceAux(param);
      } else {
	return false;
      }
    }

    /** sets default values for when tuning is disabled */
    void initTuneParam(TuneParam &param) const
    {
      TunableBlock2D::initTuneParam(param);
      int active_x_threads = (aggregate_size / 2) * (nSpin == 1 ? 1 : V.SiteSubset());
      param.block = dim3(OrthoAggregates::block_mapper(active_x_threads), 1, 1);
      param.grid = dim3(V.Volume() / (nSpin == 1 ? 2 : active_x_threads), chiral_blocks, 1);
      param.aux.x = 1; // swizzle factor
    }

    void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

    long long flops() const
    {
      auto n = nVec;
      auto k = (aggregate_size / 2) * (spinBlockSize == 0 ? 1 : 2 * spinBlockSize) * nColor;
      auto L = 8l + 8l; // dot + caxpy
      auto D = 4l + 2l; // norm + scale
      return n_block_ortho * nBlock * k * ((n - 1) * n / 2 * L + n * D);
    }

    long long bytes() const
    {
      return nVec * B[0]->Bytes() + (nVec - 1) * nVec / 2 * V.Bytes() / nVec + V.Bytes()
        + (n_block_ortho - 1) * (V.Bytes() + (nVec - 1) * nVec / 2 * V.Bytes() / nVec + V.Bytes());
    }

    void preTune() { V.backup(); }
    void postTune() { V.restore(); }
  };

  template <typename vFloat, typename bFloat, int nSpin, int spinBlockSize, int nColor, int nVec>
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField *> &B, const int *fine_to_coarse,
                          const int *coarse_to_fine, const int *geo_bs, const int n_block_ortho)
  {
    int geo_blocksize = 1;
    for (int d = 0; d < V.Ndim(); d++) geo_blocksize *= geo_bs[d];

    int blocksize = geo_blocksize * V.Ncolor();
    if (spinBlockSize == 0) { blocksize /= 2; } else { blocksize *= spinBlockSize; }
    int chiralBlocks = (spinBlockSize == 0) ? 2 : V.Nspin() / spinBlockSize; //always 2 for staggered.
    int numblocks = (V.Volume()/geo_blocksize) * chiralBlocks;
    constexpr int coarseSpin = (nSpin == 4 || nSpin == 2 || spinBlockSize == 0) ? 2 : 1;

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("Block Orthogonalizing %d blocks of %d length and width %d repeating %d times\n", numblocks, blocksize,
                 nVec, n_block_ortho);

    V.Scale(1.0); // by definition this is true
    BlockOrtho<vFloat, bFloat, nSpin, spinBlockSize, nColor, coarseSpin, nVec> ortho(
      V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho);
    ortho.apply(device::get_default_stream());
  }

  template <typename vFloat, typename bFloat>
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField *> &B, const int *fine_to_coarse,
                          const int *coarse_to_fine, const int *geo_bs, int spin_bs, int n_block_ortho)
  {
    const int Nvec = B.size();
    if (V.Ncolor()/Nvec == 3) {
#ifdef NSPIN4
      if (V.Nspin() == 4) {
        constexpr int nColor = 3;
        constexpr int nSpin = 4;
        if (spin_bs != 2) errorQuda("Unexpected spin block size = %d", spin_bs);
        constexpr int spinBlockSize = 2;

        if (Nvec == 6) { // for Wilson free field
          BlockOrthogonalize<vFloat, bFloat, nSpin, spinBlockSize, nColor, 6>(V, B, fine_to_coarse, coarse_to_fine,
                                                                              geo_bs, n_block_ortho);
        } else if (Nvec == 24) {
          BlockOrthogonalize<vFloat, bFloat, nSpin, spinBlockSize, nColor, 24>(V, B, fine_to_coarse, coarse_to_fine,
                                                                               geo_bs, n_block_ortho);
        } else if (Nvec == 32) {
          BlockOrthogonalize<vFloat, bFloat, nSpin, spinBlockSize, nColor, 32>(V, B, fine_to_coarse, coarse_to_fine,
                                                                               geo_bs, n_block_ortho);
        } else {
          errorQuda("Unsupported nVec %d\n", Nvec);
        }
      } else
#endif // NSPIN4
#ifdef NSPIN1
      if (V.Nspin() == 1) {
        constexpr int nColor = 3;
        constexpr int nSpin = 1;
        if (spin_bs != 0) errorQuda("Unexpected spin block size = %d", spin_bs);
        constexpr int spinBlockSize = 0;

        if (Nvec == 24) {
          BlockOrthogonalize<vFloat, bFloat, nSpin, spinBlockSize, nColor, 24>(V, B, fine_to_coarse, coarse_to_fine,
                                                                               geo_bs, n_block_ortho);
        } else if (Nvec == 64) {
          BlockOrthogonalize<vFloat, bFloat, nSpin, spinBlockSize, nColor, 64>(V, B, fine_to_coarse, coarse_to_fine,
                                                                               geo_bs, n_block_ortho);
        } else if (Nvec == 96) {
          BlockOrthogonalize<vFloat, bFloat, nSpin, spinBlockSize, nColor, 96>(V, B, fine_to_coarse, coarse_to_fine,
                                                                               geo_bs, n_block_ortho);
        } else {
          errorQuda("Unsupported nVec %d\n", Nvec);
        }

      } else
#endif // NSPIN1
      {
        errorQuda("Unexpected nSpin = %d", V.Nspin());
      }

    } else { // Nc != 3
      if (V.Nspin() != 2) errorQuda("Unexpected nSpin = %d", V.Nspin());
      constexpr int nSpin = 2;
      if (spin_bs != 1) errorQuda("Unexpected spin block size = %d", spin_bs);
      constexpr int spinBlockSize = 1;

#ifdef NSPIN4
      if (V.Ncolor()/Nvec == 6) {
        constexpr int nColor = 6;
        if (Nvec == 6) {
          BlockOrthogonalize<vFloat, bFloat, nSpin, spinBlockSize, nColor, 6>(V, B, fine_to_coarse, coarse_to_fine,
                                                                              geo_bs, n_block_ortho);
        } else {
          errorQuda("Unsupported nVec %d\n", Nvec);
        }
      } else
#endif // NSPIN4
      if (V.Ncolor()/Nvec == 24) {
        constexpr int nColor = 24;
        if (Nvec == 24) {
          BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,24>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho);
#ifdef NSPIN4
        } else if (Nvec == 32) {
          BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,32>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho);
#endif // NSPIN4
#ifdef NSPIN1
        } else if (Nvec == 64) {
          BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,64>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho);
        } else if (Nvec == 96) {
          BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,96>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho);
#endif // NSPIN1
        } else {
          errorQuda("Unsupported nVec %d\n", Nvec);
        }
#ifdef NSPIN4
      } else if (V.Ncolor()/Nvec == 32) {
        constexpr int nColor = 32;
        if (Nvec == 32) {
          BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,32>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho);
        } else {
          errorQuda("Unsupported nVec %d\n", Nvec);
        }
#endif // NSPIN4
#ifdef NSPIN1
      } else if (V.Ncolor()/Nvec == 64) {
        constexpr int nColor = 64;
        if (Nvec == 64) {
          BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,64>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho);
        } else if (Nvec == 96) {
          BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,96>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho);
        } else {
          errorQuda("Unsupported nVec %d\n", Nvec);
        }
      } else if (V.Ncolor()/Nvec == 96) {
        constexpr int nColor = 96;
        if (Nvec == 96) {
          BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,96>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho);
        } else {
          errorQuda("Unsupported nVec %d\n", Nvec);
        }
#endif // NSPIN1
      } else {
        errorQuda("Unsupported nColor %d\n", V.Ncolor()/Nvec);
      }
    } // Nc != 3
  }

#ifdef GPU_MULTIGRID
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField *> &B, const int *fine_to_coarse,
                          const int *coarse_to_fine, const int *geo_bs, const int spin_bs, const int n_block_ortho)
  {
    if (B[0]->V() == nullptr) {
      warningQuda("Trying to BlockOrthogonalize staggered transform, skipping...");
      return;
    }
    if (V.Precision() == QUDA_DOUBLE_PRECISION && B[0]->Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      BlockOrthogonalize<double>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (V.Precision() == QUDA_SINGLE_PRECISION && B[0]->Precision() == QUDA_SINGLE_PRECISION) {
      BlockOrthogonalize<float, float>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho);
    } else if (V.Precision() == QUDA_HALF_PRECISION && B[0]->Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 2
      BlockOrthogonalize<short, float>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho);
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    } else if (V.Precision() == QUDA_HALF_PRECISION && B[0]->Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
      BlockOrthogonalize<short, short>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho);
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    } else {
      errorQuda("Unsupported precision combination V=%d B=%d\n", V.Precision(), B[0]->Precision());
    }
  }
#else
  void BlockOrthogonalize(ColorSpinorField &, const std::vector<ColorSpinorField *> &, const int *,
                          const int *, const int *, const int, const int)
  {
    errorQuda("Multigrid has not been built");
  }
#endif // GPU_MULTIGRID

} // namespace quda
