#include <color_spinor_field.h>
#include <uint_to_char.h>
#include <vector>
#include <assert.h>
#include <utility>

#include <power_of_two_array.h>
#include <kernels/block_orthogonalize.cuh>
#include <tunable_block_reduction.h>
#include <instantiate.h>
#include <multigrid.h>

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
       @brief Return the first power of two block that is larger than the required size
    */
    static unsigned int block_mapper(unsigned int raw_block)
    {
      for (unsigned int b = 0; b < block.size();  b++) if (raw_block <= block[b]) return block[b];
      errorQuda("Invalid raw block size %d\n", raw_block);
      return 0;
    }
  };

  constexpr OrthoAggregates::array_type OrthoAggregates::block;

  using namespace quda::colorspinor;

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
    bool two_pass;
    int iter;
    double max;

  public:
    BlockOrtho(ColorSpinorField &V, const std::vector<ColorSpinorField *> B, const int *fine_to_coarse,
               const int *coarse_to_fine, const int *geo_bs, int n_block_ortho, bool two_pass) :
      TunableBlock2D(V, false, chiral_blocks),
      V(V),
      B(B),
      fine_to_coarse(fine_to_coarse),
      coarse_to_fine(coarse_to_fine),
      geo_bs(geo_bs),
      n_block_ortho(n_block_ortho),
      two_pass(two_pass),
      iter(0),
      max(1.0)
    {
      if (nColor_ != nColor)
        errorQuda("Number of colors %d not supported with this precision %lu\n", nColor_, sizeof(bFloat));

#ifdef QUDA_FAST_COMPILE_REDUCE
      strcat(aux, ",fast_compile");
#endif
      strcat(aux,",block_size=");

      aggregate_size = 1;
      for (int d = 0; d < V.Ndim(); d++) {
        aggregate_size *= geo_bs[d];
        i32toa(aux + strlen(aux), geo_bs[d]);
        if (d < V.Ndim() - 1) strcat(aux, "x");
      }

      if (aggregate_size == 1) errorQuda("Invalid MG aggregate size %d, cannot be 1", aggregate_size);
      if (aggregate_size % 2 != 0) errorQuda("Invalid MG aggregate size %d, must be even", aggregate_size);
      if (aggregate_size > 1024) errorQuda("Invalid MG aggregate size %d, must be <= 1024", aggregate_size);

      nBlock = (V.Volume()/aggregate_size) * chiral_blocks;

      strcat(aux, ",n_block_ortho=");
      i32toa(aux + strlen(aux), n_block_ortho);
      strcat(aux, ",mVec=");
      int active_x_threads = (aggregate_size / 2) * (nSpin == 1 ? 1 : V.SiteSubset());
      i32toa(aux + strlen(aux), tile_size<nColor, nVec>(OrthoAggregates::block_mapper(active_x_threads)));

      V.Scale(max); // by definition this is true
      apply(device::get_default_stream());

      auto margin = 1.05; // the 1.05 gives us some margin
      if (two_pass && V.Precision() < QUDA_SINGLE_PRECISION && (margin * max < 1.0)) {  // recompute for more precision
        iter++;
        V.Scale(margin * max);
        apply(device::get_default_stream());
      }
    }

    template <typename Rotator, typename Vector, std::size_t... S>
    void launch_host_(const TuneParam &tp, const qudaStream_t &stream,
                     const std::vector<ColorSpinorField*> &B, std::index_sequence<S...>)
    {
      Arg<false, Rotator, Vector> arg(V, fine_to_coarse, coarse_to_fine, QUDA_INVALID_PARITY, geo_bs, n_block_ortho, V, B[S]...);
      launch_host<BlockOrtho_, OrthoAggregates>(tp, stream, arg);
      if (two_pass && iter == 0 && V.Precision() < QUDA_SINGLE_PRECISION && !activeTuning()) max = Rotator(V).abs_max(V);
    }

    template <typename Rotator, typename Vector, std::size_t... S>
    void launch_device_(const TuneParam &tp, const qudaStream_t &stream,
                        const std::vector<ColorSpinorField*> &B, std::index_sequence<S...>)
    {
      Arg<true, Rotator, Vector> arg(V, fine_to_coarse, coarse_to_fine, QUDA_INVALID_PARITY, geo_bs, n_block_ortho, V, B[S]...);
      arg.swizzle_factor = tp.aux.x;
      launch_device<BlockOrtho_, OrthoAggregates>(tp, stream, arg);
      if (two_pass && iter == 0 && V.Precision() < QUDA_SINGLE_PRECISION && !activeTuning()) max = Rotator(V).abs_max(V);
    }

    void apply(const qudaStream_t &stream)
    {
      constexpr bool disable_ghost = true;
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (V.Location() == QUDA_CPU_FIELD_LOCATION) {
        if (V.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER && B[0]->FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
          typedef FieldOrderCB<real,nSpin,nColor,nVec,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,vFloat,vFloat,disable_ghost> Rotator;
          typedef FieldOrderCB<real,nSpin,nColor,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,bFloat,bFloat,disable_ghost> Vector;
          launch_host_<Rotator, Vector>(tp, stream, B, std::make_index_sequence<nVec>());
        } else {
          errorQuda("Unsupported field order %d", V.FieldOrder());
        }
      } else {
        constexpr auto vOrder = colorspinor::getNative<vFloat>(nSpin);
        constexpr auto bOrder = colorspinor::getNative<bFloat>(nSpin);
        if (V.FieldOrder() == vOrder && B[0]->FieldOrder() == bOrder) {
          typedef FieldOrderCB<real,nSpin,nColor,nVec,vOrder,vFloat,vFloat,disable_ghost> Rotator;
          typedef FieldOrderCB<real,nSpin,nColor,1,bOrder,bFloat,bFloat,disable_ghost,isFixed<bFloat>::value> Vector;
          launch_device_<Rotator, Vector>(tp, stream, B, std::make_index_sequence<nVec>());
        } else {
          errorQuda("Unsupported field order V=%d B=%d", V.FieldOrder(), B[0]->FieldOrder());
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
      param.grid = dim3((nSpin == 1 ? V.VolumeCB() : V.Volume()) / active_x_threads, 1, chiral_blocks);
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
                          const int *coarse_to_fine, const int *geo_bs, int n_block_ortho, bool two_pass)
  {
    int geo_blocksize = 1;
    for (int d = 0; d < V.Ndim(); d++) geo_blocksize *= geo_bs[d];

    int blocksize = geo_blocksize * V.Ncolor();
    if (spinBlockSize == 0) { blocksize /= 2; } else { blocksize *= spinBlockSize; }
    int chiralBlocks = (spinBlockSize == 0) ? 2 : V.Nspin() / spinBlockSize; //always 2 for staggered.
    int numblocks = (V.Volume()/geo_blocksize) * chiralBlocks;
    constexpr int coarseSpin = (nSpin == 4 || nSpin == 2 || spinBlockSize == 0) ? 2 : 1;

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("Block Orthogonalizing %d blocks of %d length and width %d repeating %d times, two_pass = %d\n",
                 numblocks, blocksize, nVec, n_block_ortho, two_pass);

    BlockOrtho<vFloat, bFloat, nSpin, spinBlockSize, nColor, coarseSpin, nVec>
      ortho(V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho, two_pass);
  }

  template <typename vFloat, typename bFloat, int fineColor, int coarseColor>
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField *> &B, const int *fine_to_coarse,
                          const int *coarse_to_fine, const int *geo_bs, int spin_bs, int n_block_ortho, bool two_pass)
  {
    if (!is_enabled_spin(V.Nspin())) errorQuda("nSpin %d has not been built", V.Nspin());

    if (V.Nspin() == 2) {
      constexpr int nSpin = 2;
      if (spin_bs != 1) errorQuda("Unexpected spin block size = %d", spin_bs);
      constexpr int spinBlockSize = 1;
      BlockOrthogonalize<vFloat, bFloat, nSpin, spinBlockSize, fineColor, coarseColor>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho, two_pass);
    } else if constexpr (fineColor == 3) {
      if (V.Nspin() == 4) {
        constexpr int nSpin = 4;
        if (spin_bs != 2) errorQuda("Unexpected spin block size = %d", spin_bs);
        if constexpr (is_enabled_spin(nSpin)) {
          constexpr int spinBlockSize = 2;
          BlockOrthogonalize<vFloat, bFloat, nSpin, spinBlockSize, fineColor, coarseColor>
            (V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho, two_pass);
        }
      } else if (V.Nspin() == 1) {
        constexpr int nSpin = 1;
        if (spin_bs != 0) errorQuda("Unexpected spin block size = %d", spin_bs);
        if constexpr (is_enabled_spin(nSpin)) {
          constexpr int spinBlockSize = 0;
          BlockOrthogonalize<vFloat, bFloat, nSpin, spinBlockSize, fineColor, coarseColor>
            (V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho, two_pass);
        }
      } else {
        errorQuda("Unexpected nSpin = %d", V.Nspin());
      }
    }
  }

  constexpr int fineColor = @QUDA_MULTIGRID_NC_NVEC@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC2@;

  template <>
  void BlockOrthogonalize<fineColor, coarseColor>(ColorSpinorField &V, const std::vector<ColorSpinorField *> &B, const int *fine_to_coarse,
                                                  const int *coarse_to_fine, const int *geo_bs, int spin_bs, int n_block_ortho, bool two_pass)
  {
    if (!is_enabled(V.Precision()) || !is_enabled(B[0]->Precision()))
      errorQuda("QUDA_PRECISION=%d does not enable required precision combination (V = %d B = %d)",
                QUDA_PRECISION, V.Precision(), B[0]->Precision());

    if constexpr (is_enabled_multigrid()) {
      if (B[0]->V() == nullptr) {
        warningQuda("Trying to BlockOrthogonalize staggered transform, skipping...");
        return;
      }
      if (V.Precision() == QUDA_DOUBLE_PRECISION && B[0]->Precision() == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double())
          BlockOrthogonalize<double, double, fineColor, coarseColor>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho, two_pass);
        else
          errorQuda("Double precision multigrid has not been enabled");
      } else if (V.Precision() == QUDA_SINGLE_PRECISION && B[0]->Precision() == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
          BlockOrthogonalize<float, float, fineColor, coarseColor>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho, two_pass);
      } else if (V.Precision() == QUDA_HALF_PRECISION && B[0]->Precision() == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION) && is_enabled(QUDA_SINGLE_PRECISION))
          BlockOrthogonalize<short, float, fineColor, coarseColor>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho, two_pass);
      } else if (V.Precision() == QUDA_HALF_PRECISION && B[0]->Precision() == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION))
          BlockOrthogonalize<short, short, fineColor, coarseColor>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs, n_block_ortho, two_pass);
      } else {
        errorQuda("Unsupported precision combination V=%d B=%d\n", V.Precision(), B[0]->Precision());
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
