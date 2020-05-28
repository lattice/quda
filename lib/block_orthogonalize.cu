#include <color_spinor_field.h>
#include <tune_quda.h>
#include <uint_to_char.h>
#include <vector>
#include <assert.h>
#include <utility>

#include <launch_kernel.cuh>
#include <jitify_helper.cuh>
#include <kernels/block_orthogonalize.cuh>

namespace quda {

  using namespace quda::colorspinor;

  template <typename sumType, typename vFloat, typename bFloat, int nSpin, int spinBlockSize, int nColor_, int coarseSpin, int nVec>
  class BlockOrtho : public Tunable {

    // we only support block-format on fine grid where Ncolor=3
    static constexpr int nColor = isFixed<bFloat>::value ? 3 : nColor_;

    typedef typename mapper<vFloat>::type RegType;
    ColorSpinorField &V;
    const std::vector<ColorSpinorField*> B;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const int *geo_bs;
    const int n_block_ortho;
    int geoBlockSize;
    int nBlock;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    unsigned int minThreads() const { return V.VolumeCB(); } // fine parity is the block y dimension

  public:
      BlockOrtho(ColorSpinorField &V, const std::vector<ColorSpinorField *> B, const int *fine_to_coarse,
                 const int *coarse_to_fine, const int *geo_bs, const int n_block_ortho) :
        V(V),
        B(B),
        fine_to_coarse(fine_to_coarse),
        coarse_to_fine(coarse_to_fine),
        geo_bs(geo_bs),
        n_block_ortho(n_block_ortho)
      {
        if (nColor_ != nColor)
          errorQuda("Number of colors %d not supported with this precision %lu\n", nColor_, sizeof(bFloat));

        if (V.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
          create_jitify_program("kernels/block_orthogonalize.cuh");
#endif
        }
      strcat(aux, compile_type_str(V));
      strcat(aux, V.AuxString());
      strcat(aux,",block_size=");

      geoBlockSize = 1;
      char geo_str[16];
      for (int d = 0; d < V.Ndim(); d++) {
        geoBlockSize *= geo_bs[d];
        i32toa(geo_str, geo_bs[d]);
        strcat(aux, geo_str);
        if (d < V.Ndim() - 1) strcat(aux, "x");
      }

      strcat(aux, ",n_block_ortho=");
      char n_ortho_str[2];
      i32toa(n_ortho_str, n_block_ortho);
      strcat(aux, n_ortho_str);

      if (V.Location() == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());

      int chiralBlocks = (nSpin==1) ? 2 : V.Nspin() / spinBlockSize; //always 2 for staggered.
      nBlock = (V.Volume()/geoBlockSize) * chiralBlocks;
      }

    /**
       @brief Helper function for expanding the std::vector into a
       parameter pack that we can use to instantiate the const arrays
       in BlockOrthoArg and then call the CPU variant of the block
       orthogonalization.
     */
    template <typename Rotator, typename Vector, std::size_t... S>
    void CPU(const std::vector<ColorSpinorField*> &B, std::index_sequence<S...>) {
      typedef BlockOrthoArg<Rotator,Vector,nSpin,spinBlockSize,coarseSpin,nVec> Arg;
      Arg arg(V, fine_to_coarse, coarse_to_fine, QUDA_INVALID_PARITY, geo_bs, n_block_ortho, V, B[S]...);
      blockOrthoCPU<sumType,RegType,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg>(arg);
    }

    /**
       @brief Helper function for expanding the std::vector into a
       parameter pack that we can use to instantiate the const arrays
       in BlockOrthoArg and then call the GPU variant of the block
       orthogonalization.
     */
    template <typename Rotator, typename Vector, std::size_t... S>
    void GPU(const TuneParam &tp, const qudaStream_t &stream, const std::vector<ColorSpinorField*> &B, std::index_sequence<S...>) {
      typedef typename mapper<vFloat>::type RegType; // need to redeclare typedef (WAR for CUDA 7 and 8)
      typedef BlockOrthoArg<Rotator,Vector,nSpin,spinBlockSize,coarseSpin,nVec> Arg;
      Arg arg(V, fine_to_coarse, coarse_to_fine, QUDA_INVALID_PARITY, geo_bs, n_block_ortho, V, B[S]...);
      arg.swizzle = tp.aux.x;
#ifdef JITIFY
      using namespace jitify::reflection;
      auto instance = program->kernel("quda::blockOrthoGPU")
        .instantiate((int)tp.block.x,Type<sumType>(),Type<RegType>(),nSpin,spinBlockSize,nColor,coarseSpin,nVec,Type<Arg>());
      cuMemcpyHtoDAsync(instance.get_constant_ptr("quda::B_array_d"), B_array_h, MAX_MATRIX_SIZE, stream);
      jitify_error = instance.configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
      cudaMemcpyToSymbolAsync(B_array_d, B_array_h, MAX_MATRIX_SIZE, 0, cudaMemcpyHostToDevice, stream);
      LAUNCH_KERNEL_MG_BLOCK_SIZE(blockOrthoGPU,tp,stream,arg,sumType,RegType,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg);
#endif
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (V.Location() == QUDA_CPU_FIELD_LOCATION) {
        if (V.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER && B[0]->FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
          typedef FieldOrderCB<RegType,nSpin,nColor,nVec,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,vFloat,vFloat,DISABLE_GHOST> Rotator;
          typedef FieldOrderCB<RegType,nSpin,nColor,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,bFloat,bFloat,DISABLE_GHOST> Vector;
          CPU<Rotator,Vector>(B, std::make_index_sequence<nVec>());
        } else {
          errorQuda("Unsupported field order %d\n", V.FieldOrder());
        }
            } else {
        if (V.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER && B[0]->FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
          typedef FieldOrderCB<RegType,nSpin,nColor,nVec,QUDA_FLOAT2_FIELD_ORDER,vFloat,vFloat,DISABLE_GHOST> Rotator;
          typedef FieldOrderCB<RegType,nSpin,nColor,1,QUDA_FLOAT2_FIELD_ORDER,bFloat,bFloat,DISABLE_GHOST,isFixed<bFloat>::value> Vector;
                GPU<Rotator,Vector>(tp,stream,B,std::make_index_sequence<nVec>());
        } else {
          errorQuda("Unsupported field order V=%d B=%d\n", V.FieldOrder(), B[0]->FieldOrder());
        }
      }
    }

    bool advanceAux(TuneParam &param) const
    {
#ifdef SWIZZLE
      if (param.aux.x < 2*deviceProp.multiProcessorCount) {
        param.aux.x++;
	return true;
      } else {
        param.aux.x = 1;
	return false;
      }
#else
      return false;
#endif
    }

    bool advanceTuneParam(TuneParam &param) const {
      if (V.Location() == QUDA_CUDA_FIELD_LOCATION) {
	return advanceSharedBytes(param) || advanceAux(param);
      } else {
	return false;
      }
    }

    TuneKey tuneKey() const { return TuneKey(V.VolString(), typeid(*this).name(), aux); }

    void initTuneParam(TuneParam &param) const { defaultTuneParam(param); }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const {
      param.block = dim3(geoBlockSize/2, V.SiteSubset(), 1);
      param.grid = dim3((minThreads() + param.block.x - 1) / param.block.x, 1, coarseSpin);
      param.shared_bytes = 0;
      param.aux.x = 1; // swizzle factor
    }

    long long flops() const
    {
      return n_block_ortho * nBlock * (geoBlockSize / 2) * (spinBlockSize == 0 ? 1 : 2 * spinBlockSize) / 2 * nColor
        * (nVec * ((nVec - 1) * (8l + 8l)) + 6l);
    }

    long long bytes() const
    {
      return nVec * B[0]->Bytes() + (nVec - 1) * nVec / 2 * V.Bytes() / nVec + V.Bytes()
        + (n_block_ortho - 1) * (V.Bytes() + (nVec - 1) * nVec / 2 * V.Bytes() / nVec + V.Bytes());
    }

    char *saveOut, *saveOutNorm;

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
    BlockOrtho<double, vFloat, bFloat, nSpin, spinBlockSize, nColor, coarseSpin, nVec> ortho(
      V, B, fine_to_coarse, coarse_to_fine, geo_bs, n_block_ortho);
    ortho.apply(0);
    checkCudaError();
  }

  template <typename vFloat, typename bFloat>
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField *> &B, const int *fine_to_coarse,
                          const int *coarse_to_fine, const int *geo_bs, int spin_bs, int n_block_ortho)
  {
    const int Nvec = B.size();
    if (V.Ncolor()/Nvec == 3) {
      if (V.Nspin() != 4) errorQuda("Unexpected nSpin = %d", V.Nspin());
#ifdef NSPIN4
      constexpr int nSpin = 4;
      if (spin_bs != 2) errorQuda("Unexpected spin block size = %d", spin_bs);
      constexpr int spinBlockSize = 2;
      constexpr int nColor = 3;

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
#endif // NSPIN4

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

  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField *> &B, const int *fine_to_coarse,
                          const int *coarse_to_fine, const int *geo_bs, const int spin_bs, const int n_block_ortho)
  {
#ifdef GPU_MULTIGRID
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
#else
    errorQuda("Multigrid has not been built");
#endif // GPU_MULTIGRID
  }

} // namespace quda
