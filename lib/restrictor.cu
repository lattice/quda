#include <color_spinor_field.h>
#include <tune_quda.h>
#include <launch_kernel.cuh>

#include <jitify_helper.cuh>
#include <kernels/restrictor.cuh>

namespace quda {

  template <typename Float, typename vFloat, int fineSpin, int fineColor, int coarseSpin, int coarseColor,
            int coarse_colors_per_thread>
  class RestrictLaunch : public Tunable {

  protected:
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const ColorSpinorField &v;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const int parity;
    const QudaFieldLocation location;
    const int block_size;
    char vol[TuneKey::volume_n];

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    bool tuneAuxDim() const { return true; } // Do tune the aux dimensions.
    unsigned int minThreads() const { return in.VolumeCB(); } // fine parity is the block y dimension

  public:
    RestrictLaunch(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                   const int *fine_to_coarse, const int *coarse_to_fine, int parity)
      : out(out), in(in), v(v), fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
        parity(parity), location(checkLocation(out,in,v)), block_size(in.VolumeCB()/(2*out.VolumeCB()))
    {
      if (v.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        create_jitify_program("kernels/restrictor.cuh");
#endif
      }
      strcpy(aux, compile_type_str(in));
      strcat(aux, out.AuxString());
      strcat(aux, ",");
      strcat(aux, in.AuxString());

      strcpy(vol, out.VolString());
      strcat(vol, ",");
      strcat(vol, in.VolString());
    } // block size is checkerboard fine length / full coarse length

    void apply(const qudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
        if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
          RestrictArg<Float,vFloat,fineSpin,fineColor,coarseSpin,coarseColor,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>
            arg(out, in, v, fine_to_coarse, coarse_to_fine, parity);
          Restrict<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread>(arg);
        } else {
          errorQuda("Unsupported field order %d", out.FieldOrder());
        }
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
          typedef RestrictArg<Float,vFloat,fineSpin,fineColor,coarseSpin,coarseColor,QUDA_FLOAT2_FIELD_ORDER> Arg;
          Arg arg(out, in, v, fine_to_coarse, coarse_to_fine, parity);
          arg.swizzle = tp.aux.x;

#ifdef JITIFY
          using namespace jitify::reflection;
          jitify_error = program->kernel("quda::RestrictKernel")
            .instantiate((int)tp.block.x,Type<Float>(),fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Type<Arg>())
            .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
          LAUNCH_KERNEL_MG_BLOCK_SIZE(RestrictKernel,tp,stream,arg,Float,fineSpin,fineColor,
                                      coarseSpin,coarseColor,coarse_colors_per_thread,Arg);
#endif
        } else {
          errorQuda("Unsupported field order %d", out.FieldOrder());
        }
      }
    }

    // This block tuning tunes for the optimal amount of color
    // splitting between blockDim.z and gridDim.z.  However, enabling
    // blockDim.z > 1 gives incorrect results due to cub reductions
    // being unable to do independent sliced reductions along
    // blockDim.z.  So for now we only split between colors per thread
    // and grid.z.
    bool advanceBlockDim(TuneParam &param) const
    {
      // let's try to advance spin/block-color
      while(param.block.z <= coarseColor/coarse_colors_per_thread) {
        param.block.z++;
        if ( (coarseColor/coarse_colors_per_thread) % param.block.z == 0) {
          param.grid.z = (coarseColor/coarse_colors_per_thread) / param.block.z;
          break;
        }
      }

      // we can advance spin/block-color since this is valid
      if (param.block.z <= (coarseColor/coarse_colors_per_thread) ) { //
        return true;
      } else { // we have run off the end so let's reset
        param.block.z = 1;
        param.grid.z = coarseColor/coarse_colors_per_thread;
        return false;
      }
    }

    int tuningIter() const { return 3; }

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

    // only tune shared memory per thread (disable tuning for block.z for now)
    bool advanceTuneParam(TuneParam &param) const { return advanceSharedBytes(param) || advanceAux(param); }

    TuneKey tuneKey() const { return TuneKey(vol, typeid(*this).name(), aux); }

    void initTuneParam(TuneParam &param) const { defaultTuneParam(param); }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const {
      param.block = dim3(block_size, in.SiteSubset(), 1);
      param.grid = dim3( (minThreads()+param.block.x-1) / param.block.x, 1, 1);
      param.shared_bytes = 0;

      param.block.z = 1;
      param.grid.z = coarseColor / coarse_colors_per_thread;
      param.aux.x = 1; // swizzle factor
    }

    long long flops() const { return 8 * fineSpin * fineColor * coarseColor * in.SiteSubset()*(long long)in.VolumeCB(); }

    long long bytes() const {
      size_t v_bytes = v.Bytes() / (v.SiteSubset() == in.SiteSubset() ? 1 : 2);
      return in.Bytes() + out.Bytes() + v_bytes + in.SiteSubset()*in.VolumeCB()*sizeof(int);
    }

  };

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                const int *fine_to_coarse, const int *coarse_to_fine, int parity) {

    // for fine grids (Nc=3) have more parallelism so can use more coarse strategy
    constexpr int coarse_colors_per_thread = fineColor != 3 ? 2 : coarseColor >= 4 && coarseColor % 4 == 0 ? 4 : 2;
    //coarseColor >= 8 && coarseColor % 8 == 0 ? 8 : coarseColor >= 4 && coarseColor % 4 == 0 ? 4 : 2;

    if (v.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
      RestrictLaunch<Float, short, fineSpin, fineColor, coarseSpin, coarseColor, coarse_colors_per_thread>
        restrictor(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      restrictor.apply(0);
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    } else if (v.Precision() == in.Precision()) {
      RestrictLaunch<Float, Float, fineSpin, fineColor, coarseSpin, coarseColor, coarse_colors_per_thread>
        restrictor(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      restrictor.apply(0);
    } else {
      errorQuda("Unsupported V precision %d", v.Precision());
    }

    if (checkLocation(out, in, v) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }

  template <typename Float>
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                int nVec, const int *fine_to_coarse, const int *coarse_to_fine, const int * const * spin_map, int parity)
  {
    if (out.Nspin() != 2) errorQuda("Unsupported nSpin %d", out.Nspin());
    constexpr int coarseSpin = 2;

    // Template over fine color
    if (in.Ncolor() == 3) { // standard QCD
      if (in.Nspin() != 4) errorQuda("Unexpected nSpin = %d", in.Nspin());
#ifdef NSPIN4
      constexpr int fineSpin = 4;
      constexpr int fineColor = 3;

      // first check that the spin_map matches the spin_mapper
      spin_mapper<fineSpin,coarseSpin> mapper;
      for (int s=0; s<fineSpin; s++)
        for (int p=0; p<2; p++)
          if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

      if (nVec == 6) { // free field Wilson
        Restrict<Float,fineSpin,fineColor,coarseSpin,6>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else if (nVec == 24) {
        Restrict<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else if (nVec == 32) {
        Restrict<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
#endif // NSPIN4

    } else { // Nc != 3

      if (in.Nspin() != 2) errorQuda("Unexpected nSpin = %d", in.Nspin());
      constexpr int fineSpin = 2;

      // first check that the spin_map matches the spin_mapper
      spin_mapper<fineSpin,coarseSpin> mapper;
      for (int s=0; s<fineSpin; s++)
        for (int p=0; p<2; p++)
          if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

#ifdef NSPIN4
      if (in.Ncolor() == 6) { // Coarsen coarsened Wilson free field
        const int fineColor = 6;
        if (nVec == 6) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,6>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else {
          errorQuda("Unsupported nVec %d", nVec);
        }
      } else
#endif // NSPIN4
      if (in.Ncolor() == 24) { // to keep compilation under control coarse grids have same or more colors
        const int fineColor = 24;
        if (nVec == 24) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
#ifdef NSPIN4
        } else if (nVec == 32) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
#endif // NSPIN4
#ifdef NSPIN1
        } else if (nVec == 64) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else if (nVec == 96) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
#endif // NSPIN1
        } else {
          errorQuda("Unsupported nVec %d", nVec);
        }
#ifdef NSPIN4
      } else if (in.Ncolor() == 32) {
        const int fineColor = 32;
        if (nVec == 32) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else {
          errorQuda("Unsupported nVec %d", nVec);
        }
#endif // NSPIN4
#ifdef NSPIN1
      } else if (in.Ncolor() == 64) {
        const int fineColor = 64;
        if (nVec == 64) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else if (nVec == 96) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else {
          errorQuda("Unsupported nVec %d", nVec);
        }
      } else if (in.Ncolor() == 96) {
        const int fineColor = 96;
        if (nVec == 96) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else {
          errorQuda("Unsupported nVec %d", nVec);
        }
#endif // NSPIN1
      } else {
        errorQuda("Unsupported nColor %d", in.Ncolor());
      }
    } // Nc != 3
  }

  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                int Nvec, const int *fine_to_coarse, const int *coarse_to_fine, const int * const * spin_map, int parity)
  {
#ifdef GPU_MULTIGRID
    if (out.FieldOrder() != in.FieldOrder() ||        out.FieldOrder() != v.FieldOrder())
      errorQuda("Field orders do not match (out=%d, in=%d, v=%d)",
                out.FieldOrder(), in.FieldOrder(), v.FieldOrder());

    QudaPrecision precision = checkPrecision(out, in);

    if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      Restrict<double>(out, in, v, Nvec, fine_to_coarse, coarse_to_fine, spin_map, parity);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (precision == QUDA_SINGLE_PRECISION) {
      Restrict<float>(out, in, v, Nvec, fine_to_coarse, coarse_to_fine, spin_map, parity);
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }
#else
    errorQuda("Multigrid has not been built");
#endif
  }

} // namespace quda
