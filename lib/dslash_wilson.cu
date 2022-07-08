#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <algorithm>

#include <dslash_policy.cuh>
#include <kernels/dslash_wilson.cuh>

/**
   This is the basic gauged Wilson operator
   TODO
   - gauge fix support
*/

namespace quda
{
  constexpr int num_buckets = 4;
  using array_t = std::array<int, num_buckets>;

  template <class T>
  static int encode(const array_t &a, T dim[]) {
    return ((a[3] * (dim[2] + 1) + a[2]) * (dim[1] + 1) + a[1]) * (dim[0] + 1) + a[0];
  }

  template <class T>
  static auto decode(int code, T dim[]) {
    array_t a;
    for (int d = 0; d < 4; d++) {
      a[d] = code % (dim[d] + 1);
      code /= (dim[d] + 1);
    }
    return a;
  }

  template <class T>
  static int get_grid(const TuneParam &tp, T dim[]) {
    auto a = decode(tp.aux.z, dim);
    return ((dim[0] + 1) / a[0]) * ((dim[1] + 1) / a[1]) * ((dim[2] + 1) / a[2]) * ((dim[3] + 1) / a[3]);
  }

  template <typename Arg> class Wilson : public Dslash<wilson, Arg>
  {
    using Dslash = Dslash<wilson, Arg>;
    using Dslash::arg;

  public:
    Wilson(Arg &arg_, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg_, out, in)
    {
      if(in.Ndim() == 5) {
        TunableKernel3D::resizeVector(in.X(4), arg_.nParity);
      }
    }

    void set_shared_grid(TuneParam &tp, const array_t &p) const {
      tp.aux.z = encode(p, arg.dim);
      tp.shared_bytes = sharedBytesPerBlock(tp);
      tp.grid.x = get_grid(tp, arg.dim);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      if (arg.kernel_type == INTERIOR_KERNEL) {
        auto p = decode(tp.aux.z, arg.dim);

        arg.tb.X0h = p[0] / 2;
        arg.tb.Xex0h = (p[0] + 2) / 2;
        for (int d = 0; d < 4; d++) {
          arg.tb.dim[d] = p[d];
          arg.tb.dim_ex[d] = p[d] + 2;
          arg.tb.grid_dim[d] = arg.dim[d] / p[d];
        }
        printf("p = %d, %d, %d, %d\n", p[0], p[1], p[2], p[3]);
        arg.tb.X1 = p[0] + 2;
        arg.tb.X2X1 = (p[1] + 2) * (p[0] + 2);
        arg.tb.X3X2X1 = (p[2] + 2) * (p[1] + 2) * (p[0] + 2);

        arg.tb.volume_4d_cb = p[3] * p[2] * p[1] * p[0] / 2;
        arg.tb.volume_4d_cb_ex = (p[3] + 2) * (p[2] + 2) * (p[1] + 2) * (p[0] + 2) / 2;

        arg.threads = tp.block.x * tp.grid.x;
        tp.set_max_shared_bytes = true;
      }
      Dslash::template instantiate<packShmem>(tp, stream);
    }

    virtual unsigned int sharedBytesPerBlock(const TuneParam &tp) const
    {
      if (arg.kernel_type == INTERIOR_KERNEL) {
        auto p = decode(tp.aux.z, arg.dim);
        return (p[0] + 2) * (p[1] + 2) * (p[2] + 2) * (p[3] + 2) * 24 * 4 / 2;
      } else {
        return 0;
      }
    }

    virtual bool advanceAux(TuneParam & tp) const {
      if (arg.kernel_type != INTERIOR_KERNEL) {
        return Dslash::advanceAux(tp);
      }
      if (Dslash::advanceAux(tp)) {
        return true;
      } else {
        auto p = decode(tp.aux.z, arg.dim);
        bool ret = false;
        for (int d = 0; d < 4; d++) {
          p[d] *= 2;
          set_shared_grid(tp, p);
          if (arg.dim[d] % (p[d] * 2) == 0 && tp.shared_bytes <= this->maxSharedBytesPerBlock()) {
            return true;
          } else {
            p[d] = 2;
          }
        }

        for (int d = 0; d < 4; d++) {
          p[d] = 2;
        }
        set_shared_grid(tp, p);
        return false;
      }
    }

    virtual bool tuneGridDim() const { return true; }
    virtual bool tuneSharedBytes() const { return false; }

    virtual bool advanceGridDim(TuneParam &param) const
    {
      return false;
    }

    virtual void initTuneParam(TuneParam &param) const {
      Dslash::initTuneParam(param);
      if (arg.kernel_type == INTERIOR_KERNEL) {
        array_t p;
        for (int d = 0; d < 4; d++) {
          p[d] = 2;
        }
        set_shared_grid(param, p);
      }
    }

    virtual void defaultTuneParam(TuneParam &param) const {
      initTuneParam(param);
    }

  };

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonApply {

    inline WilsonApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                       const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      WilsonArg<Float, nColor, nDim, recon> arg(out, in, U, a, x, parity, dagger, comm_override);
      Wilson<decltype(arg)> wilson(arg, out, in);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, in.VolumeCB(), in.GhostFaceCB(), profile);
    }
  };

  // Apply the Wilson operator
  // out(x) = M*in = - a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the a normalization for the Wilson operator.
#ifdef GPU_WILSON_DIRAC
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                   const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    instantiate<WilsonApply, WilsonReconstruct>(out, in, U, a, x, parity, dagger, comm_override, profile);
  }
#else
  void ApplyWilson(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, double,
                   const ColorSpinorField &, int, bool, const int *, TimeProfile &)
  {
    errorQuda("Wilson dslash has not been built");
  }
#endif // GPU_WILSON_DIRAC

} // namespace quda
