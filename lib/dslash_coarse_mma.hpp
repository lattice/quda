#include <gauge_field.h>
#include <color_spinor_field.h>
#include <uint_to_char.h>
#include <worker.h>
#include <tunable_nd.h>
#include <kernels/dslash_coarse.cuh>
#include <shmem_helper.cuh>
#include <dslash_quda.h>
#include <dslash_shmem.h>
#include <multigrid.h>

namespace quda {

  template <typename Float, typename yFloat, typename ghostFloat, int Ns, int Nc, bool dslash, bool clover, bool dagger,
            DslashType type>
  class DslashCoarseMma : public TunableKernel3D
  {
    static constexpr int nDim = 4;

    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &inA;
    cvector_ref<const ColorSpinorField> &inB;
    const GaugeField &Y;
    const GaugeField &X;
    const double kappa;
    const int parity;
    const int nParity;
    const int nSrc;
    const ColorSpinorField &halo;

    const int max_color_col_stride = 8;
    mutable int color_col_stride;
    mutable int dim_threads;

    long long flops() const
    {
      return ((dslash*2*nDim+clover*1)*(8*Ns*Nc*Ns*Nc)-2*Ns*Nc)*nParity*(long long)out[0].VolumeCB() * out.size();
    }
    long long bytes() const
    {
      return ((dslash||clover) * out[0].Bytes() + dslash*8*inA[0].Bytes() + clover*inB[0].Bytes() +
              nSrc*nParity*(dslash*Y.Bytes()*Y.VolumeCB()/(2*Y.Stride()) + clover*X.Bytes()/2)) * out.size();
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
              (color_col_stride == 1 || param.block.x % device::warp_size() == 0) && // block must be a multiple of the warp
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
    DslashCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                 cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y,
                 const GaugeField &X, double kappa, int parity, MemoryLocation *halo_location,
                 const ColorSpinorField &halo) :
      TunableKernel3D(out[0], out[0].SiteSubset() * out.size(), 1),
      out(out),
      inA(inA),
      inB(inB),
      Y(Y),
      X(X),
      kappa(kappa),
      parity(parity),
      nParity(out[0].SiteSubset()),
      nSrc(out[0].Ndim() == 5 ? out[0].X(4) : 1),
      halo(halo),
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

      strcat(aux, ",n_rhs=");
      char rhs_str[8];
      i32toa(rhs_str, out.size());
      strcat(aux, rhs_str);

#ifdef QUDA_FAST_COMPILE_DSLASH
      strcat(aux, ",fast_compile");
#endif

      apply(device::get_default_stream());
    }

    template <int color_stride, int dim_stride, bool native = true>
    using Arg = DslashCoarseArg<dslash, clover, dagger, type, color_stride, dim_stride, Float, yFloat, ghostFloat, Ns,
                                Nc, native>;

    void apply(const qudaStream_t &stream)
    {
      const TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      color_col_stride = tp.aux.x;
      dim_threads = tp.aux.y;
      resizeVector(vector_length_y, 2 * dim_threads * 2 * (Nc / colors_per_thread(Nc, dim_threads)));
      if (!checkParam(tp)) errorQuda("Invalid launch param");

      if (out[0].Location() == QUDA_CPU_FIELD_LOCATION) {
        errorQuda("Not enabled");
      } else {
        checkNative(out[0], inA[0], inB[0], Y, X);

        switch (tp.aux.y) { // dimension gather parallelisation
        case 1:
          switch (tp.aux.x) { // this is color_col_stride
          case 1: launch_device<CoarseDslash>(tp, stream, Arg<1, 1>(out, inA, inB, Y, X, (Float)kappa, parity, halo)); break;
#ifndef QUDA_FAST_COMPILE_DSLASH
          case 2: launch_device<CoarseDslash>(tp, stream, Arg<2, 1>(out, inA, inB, Y, X, (Float)kappa, parity, halo)); break;
          case 4: launch_device<CoarseDslash>(tp, stream, Arg<4, 1>(out, inA, inB, Y, X, (Float)kappa, parity, halo)); break;
          case 8: launch_device<CoarseDslash>(tp, stream, Arg<8, 1>(out, inA, inB, Y, X, (Float)kappa, parity, halo)); break;
#endif
          default: errorQuda("Color column stride %d not valid", static_cast<int>(tp.aux.x));
          }
          break;
#ifndef QUDA_FAST_COMPILE_DSLASH
        case 2:
          switch (tp.aux.x) { // this is color_col_stride
          case 1: launch_device<CoarseDslash>(tp, stream, Arg<1, 2>(out, inA, inB, Y, X, (Float)kappa, parity, halo)); break;
          case 2: launch_device<CoarseDslash>(tp, stream, Arg<2, 2>(out, inA, inB, Y, X, (Float)kappa, parity, halo)); break;
          case 4: launch_device<CoarseDslash>(tp, stream, Arg<4, 2>(out, inA, inB, Y, X, (Float)kappa, parity, halo)); break;
          case 8: launch_device<CoarseDslash>(tp, stream, Arg<8, 2>(out, inA, inB, Y, X, (Float)kappa, parity, halo)); break;
          default: errorQuda("Color column stride %d not valid", static_cast<int>(tp.aux.x));
          }
          break;
        case 4:
          switch (tp.aux.x) { // this is color_col_stride
          case 1: launch_device<CoarseDslash>(tp, stream, Arg<1, 4>(out, inA, inB, Y, X, (Float)kappa, parity, halo)); break;
          case 2: launch_device<CoarseDslash>(tp, stream, Arg<2, 4>(out, inA, inB, Y, X, (Float)kappa, parity, halo)); break;
          case 4: launch_device<CoarseDslash>(tp, stream, Arg<4, 4>(out, inA, inB, Y, X, (Float)kappa, parity, halo)); break;
          case 8: launch_device<CoarseDslash>(tp, stream, Arg<8, 4>(out, inA, inB, Y, X, (Float)kappa, parity, halo)); break;
          default: errorQuda("Color column stride %d not valid", static_cast<int>(tp.aux.x));
          }
          break;
#endif
        default: errorQuda("Invalid dimension thread splitting %d", static_cast<int>(tp.aux.y));
        }
      }
    }

    void preTune() { for (auto i = 0u; i < out.size(); i++) out[i].backup(); }
    void postTune() { for (auto i = 0u; i < out.size(); i++) out[i].restore(); }
  };

  template <typename Float, typename yFloat, typename ghostFloat, bool dagger, int coarseColor>
  inline void ApplyCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA, cvector_ref<const ColorSpinorField> &inB,
                          const GaugeField &Y, const GaugeField &X, double kappa, int parity, bool dslash, bool clover,
                          DslashType type, MemoryLocation *halo_location, const ColorSpinorField &halo)
  {
    if (Y.FieldOrder() != X.FieldOrder())
      errorQuda("Field order mismatch Y = %d, X = %d", Y.FieldOrder(), X.FieldOrder());

    if (inA[0].FieldOrder() != out[0].FieldOrder())
      errorQuda("Field order mismatch inA = %d, out = %d", inA[0].FieldOrder(), out[0].FieldOrder());

    if (inA[0].Nspin() != 2)
      errorQuda("Unsupported number of coarse spins %d", inA[0].Nspin());

    constexpr int coarseSpin = 2;

    if (dslash) {
      if (clover) {
        switch (type) {
        case DSLASH_FULL: {
          DslashCoarse<Float, yFloat, ghostFloat, coarseSpin, coarseColor, true, true, dagger, DSLASH_FULL> dslash(
                                                                                                                   out, inA, inB, Y, X, kappa, parity, halo_location, halo);
          break;
        }
        case DSLASH_EXTERIOR: {
          DslashCoarse<Float, yFloat, ghostFloat, coarseSpin, coarseColor, true, true, dagger, DSLASH_EXTERIOR> dslash(
                                                                                                                       out, inA, inB, Y, X, kappa, parity, halo_location, halo);
          break;
        }
        case DSLASH_INTERIOR: {
          DslashCoarse<Float, yFloat, ghostFloat, coarseSpin, coarseColor, true, true, dagger, DSLASH_INTERIOR> dslash(
                                                                                                                       out, inA, inB, Y, X, kappa, parity, halo_location, halo);
          break;
        }
        default: errorQuda("Dslash type %d not instantiated", type);
        }

      } else { // plain dslash

        switch (type) {
        case DSLASH_FULL: {
          DslashCoarse<Float, yFloat, ghostFloat, coarseSpin, coarseColor, true, false, dagger, DSLASH_FULL> dslash(
                                                                                                                    out, inA, inB, Y, X, kappa, parity, halo_location, halo);
          break;
        }
        case DSLASH_EXTERIOR: {
          DslashCoarse<Float, yFloat, ghostFloat, coarseSpin, coarseColor, true, false, dagger, DSLASH_EXTERIOR> dslash(
                                                                                                                        out, inA, inB, Y, X, kappa, parity, halo_location, halo);
          break;
        }
        case DSLASH_INTERIOR: {
          DslashCoarse<Float, yFloat, ghostFloat, coarseSpin, coarseColor, true, false, dagger, DSLASH_INTERIOR> dslash(
                                                                                                                        out, inA, inB, Y, X, kappa, parity, halo_location, halo);
          break;
        }
        default: errorQuda("Dslash type %d not instantiated", type);
        }
      }
    } else {

      if (type == DSLASH_EXTERIOR) errorQuda("Cannot call halo on pure clover kernel");
      if (clover) {
        DslashCoarse<Float, yFloat, ghostFloat, coarseSpin, coarseColor, false, true, dagger, DSLASH_FULL> dslash(
                                                                                                                  out, inA, inB, Y, X, kappa, parity, halo_location, halo);
      } else {
        errorQuda("Unsupported dslash=false clover=false");
      }
    }
  }

} // namespace quda
