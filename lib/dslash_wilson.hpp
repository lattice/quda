#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <algorithm>

#include <dslash_policy.hpp>
#include <kernels/dslash_wilson.cuh>

/**
   This is the basic gauged Wilson operator
   TODO
   - gauge fix support
*/

namespace quda
{

  constexpr bool cache_ext = false;

  constexpr int num_buckets = 4;
  using array_t = std::array<int, num_buckets>;

  template <class T>
  static int encode(const array_t &a, T dim[]) {
    return ((a[3] * (dim[2] + 1) + a[2]) * (dim[1] + 1) + a[1]) * (dim[0] + 1) + a[0];
  }

  template <class T>
  static auto decode(int code, T dim[]) {
    array_t a;
    int code_ = code;
    for (int d = 0; d < 4; d++) {
      a[d] = code % (dim[d] + 1);
      code /= (dim[d] + 1);
    }
    // printf("code = %d, dim = %d %d %d %d, a = %d, %d, %d, %d\n", code_, int(dim[0]), int(dim[1]), int(dim[2]), int(dim[3]), a[0], a[1], a[2], a[3]);
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
    Wilson(Arg &arg_, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
           const ColorSpinorField &halo) :
      Dslash(arg_, out, in, halo)
    {
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
        arg.tb.Xex0h = arg.dim[0] == p[0] ? p[0] / 2 : (p[0] + 2) / 2;
        int parity_bit = 0;
        for (int d = 0; d < 4; d++) {
          arg.tb.dim[d] = p[d];
          if (cache_ext) {
            if (arg.dim[d] == p[d]) {
              arg.tb.dim_ex[d] = p[d];
            } else {
              parity_bit++;
              arg.tb.dim_ex[d] = p[d] + 2;
            }
          } else {
            arg.tb.dim_ex[d] = p[d];
          }
          arg.tb.parity_bit = parity_bit % 2;
          arg.tb.grid_dim[d] = arg.dim[d] / p[d];
        }
        printf("aux.z = %d, p = %d, %d, %d, %d\n", tp.aux.z, p[0], p[1], p[2], p[3]);
        arg.tb.cache_ext = cache_ext;
        if (cache_ext) {
          arg.tb.X1 = arg.tb.dim_ex[0]; // p[0] + 2;
          arg.tb.X2X1 = arg.tb.dim_ex[1] * arg.tb.dim_ex[0]; // (p[1] + 2) * (p[0] + 2);
          arg.tb.X3X2X1 = arg.tb.dim_ex[2] * arg.tb.dim_ex[1] * arg.tb.dim_ex[0]; // (p[2] + 2) * (p[1] + 2) * (p[0] + 2);

          arg.tb.X2X1mX1 = (arg.tb.dim_ex[1] - 1) * arg.tb.dim_ex[0]; // p[1] - 1) * p[0];
          arg.tb.X3X2X1mX2X1 = (arg.tb.dim_ex[2] - 1) * arg.tb.dim_ex[1] * arg.tb.dim_ex[0]; // (p[2] - 1) * p[1] * p[0];
          arg.tb.X4X3X2X1mX3X2X1 = (arg.tb.dim_ex[3] - 1) * arg.tb.dim_ex[2] * arg.tb.dim_ex[1] * arg.tb.dim_ex[0]; // (p[3] - 1) * p[2] * p[1] * p[0];
        } else {
          arg.tb.X1 = p[0];
          arg.tb.X2X1 = p[1] * p[0];
          arg.tb.X3X2X1 = p[2] * p[1] * p[0];

          arg.tb.X2X1mX1 = (p[1] - 1) * p[0];
          arg.tb.X3X2X1mX2X1 = (p[2] - 1) * p[1] * p[0];
          arg.tb.X4X3X2X1mX3X2X1 = (p[3] - 1) * p[2] * p[1] * p[0];
        }

        arg.tb.volume_4d_cb = p[3] * p[2] * p[1] * p[0] / 2;
        arg.tb.volume_4d_cb_ex = arg.tb.dim_ex[3] * arg.tb.dim_ex[2] * arg.tb.dim_ex[1] * arg.tb.dim_ex[0] / 2;

        std::array<size_t, 5> tensor_size = {arg.dim[0] / 2 * 16, arg.dim[1], arg.dim[2], arg.dim[3], 6};
        std::array<size_t, 5> box_size = {p[0] / 2 * 16, p[1], p[2], p[3], 6};

        printf("tensor = %lu %lu %lu %lu %lu, box = %lu %lu %lu %lu %lu\n",
          tensor_size[0], tensor_size[1], tensor_size[2], tensor_size[3], tensor_size[4],
          box_size[0], box_size[1], box_size[2], box_size[3], box_size[4]);

        tma_descriptor_key_t<5> key = {tensor_size, box_size, arg.in[0].field};
        arg.tma_desc = get_tma_descriptor<int8_t, 5>(key);

        arg.threads = tp.block.x * tp.grid.x;
        tp.set_max_shared_bytes = true;
      }

      Dslash::template instantiate<packShmem>(tp, stream);
    }

    virtual unsigned int sharedBytesPerBlock(const TuneParam &tp) const
    {
      if (arg.kernel_type == INTERIOR_KERNEL) {
        auto p = decode(tp.aux.z, arg.dim);
        int prod = 1;
        for (int d = 0; d < 4; d++) {
          arg.tb.dim[d] = p[d];
          if (cache_ext) {
            prod *= (arg.dim[d] == p[d] ? p[d] : p[d] + 2);
          } else {
            prod *= p[d];
          }
        }
        int smem_size = sizeof(typename Arg::Float) * 24 * prod / 2;
        if (isFixed<typename Arg::Float>::value) {
          smem_size += sizeof(float) * prod / 2;
        }
        return smem_size + 8;
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
        for (int d = 0; d < 4; d++) {
          p[d] *= 2;
          if (arg.dim[d] % p[d] == 0 && p[d] <= arg.dim[d]) {
            set_shared_grid(tp, p);
            if (tp.shared_bytes <= this->maxSharedBytesPerBlock()) {
              return true;
            } else {
              p[d] = d == 0 ? 2 : 1;
            }
          } else {
            p[d] = d == 0 ? 2 : 1;
          }
        }

        for (int d = 0; d < 4; d++) {
          p[d] = d == 0 ? 2 : 1;
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
          p[d] = d == 0 ? 2 : 1;
        }
        set_shared_grid(param, p);
      }
    }

    virtual void defaultTuneParam(TuneParam &param) const {
      initTuneParam(param);
    }

  };

  template <bool distance_pc> struct DistanceType {
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonApply {

    template <bool distance_pc>
    WilsonApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double alpha0, int t0,
                int parity, bool dagger, const int *comm_override, DistanceType<distance_pc>, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      WilsonArg<Float, nColor, nDim, recon, distance_pc> arg(out, in, halo, U, a, x, parity, dagger, comm_override,
                                                             alpha0, t0);
      Wilson<decltype(arg)> wilson(arg, out, in, halo);
      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, halo, profile);
    }
  };

} // namespace quda
