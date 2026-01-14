#include <math_helper.cuh>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <kernel.h>

namespace quda
{
  using namespace colorspinor;

  template <typename store_t, int nColor_, QudaChirality Chirality_>
  struct ChiralReconstructSpinorArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nSpin = 4;
    static constexpr int nColor = nColor_;
    static constexpr QudaChirality Chirality = Chirality_;
    using Vout = typename colorspinor_mapper<store_t, nSpin, nColor>::type;
    using Vin = typename colorspinor_mapper<store_t, nSpin / 2, nColor>::type;

    Vout out;
    const Vin in_left;
    const Vin in_right;
    ChiralReconstructSpinorArg(ColorSpinorField &out, const ColorSpinorField &in_left, const ColorSpinorField &in_right) :
      kernel_param(dim3(out.VolumeCB(), out.SiteSubset(), 1)), out(out), in_left(in_left), in_right(in_right)
    {
    }
  };

  template <typename Arg> struct ChiralReconstructSpinor {
    const Arg &arg;
    constexpr ChiralReconstructSpinor(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;
      using HalfVector = ColorSpinor<real, Arg::nColor, Arg::nSpin / 2>;
      const real invsqrt2 = (real)(1.0 / sqrt(2.0));

      Vector out;
      HalfVector in;
      if constexpr (Arg::Chirality == QUDA_LEFT_CHIRALITY || Arg::Chirality == QUDA_INVALID_CHIRALITY) {
        in = arg.in_left(x_cb, parity);
        out += in.chiral_reconstruct(1);
      }
      if constexpr (Arg::Chirality == QUDA_RIGHT_CHIRALITY || Arg::Chirality == QUDA_INVALID_CHIRALITY) {
        in = arg.in_right(x_cb, parity);
        out += in.chiral_reconstruct(0);
      }
      out.toNonRel();
      out *= invsqrt2;
      arg.out(x_cb, parity) = out;
    }
  };

  template <typename store_t, int nColor_, QudaChirality Chirality_> struct ChiralProjectSpinorArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nSpin = 4;
    static constexpr int nColor = nColor_;
    static constexpr QudaChirality Chirality = Chirality_;
    using Vout = typename colorspinor_mapper<store_t, nSpin / 2, nColor>::type;
    using Vin = typename colorspinor_mapper<store_t, nSpin, nColor>::type;

    Vout out_left;
    Vout out_right;
    const Vin in;
    ChiralProjectSpinorArg(ColorSpinorField &out_left, ColorSpinorField &out_right, const ColorSpinorField &in) :
      kernel_param(dim3(in.VolumeCB(), in.SiteSubset(), 1)), out_left(out_left), out_right(out_right), in(in)
    {
    }
  };

  template <typename Arg> struct ChiralProjectSpinor {
    const Arg &arg;
    constexpr ChiralProjectSpinor(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using HalfVector = ColorSpinor<real, Arg::nColor, Arg::nSpin / 2>;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;
      const real invsqrt2 = (real)(1.0 / sqrt(2.0));

      HalfVector out;
      Vector in = arg.in(x_cb, parity);
      in.toRel();
      in *= invsqrt2;
      if constexpr (Arg::Chirality == QUDA_LEFT_CHIRALITY || Arg::Chirality == QUDA_INVALID_CHIRALITY) {
        out = in.chiral_project(1);
        arg.out_left(x_cb, parity) = out;
      }
      if constexpr (Arg::Chirality == QUDA_RIGHT_CHIRALITY || Arg::Chirality == QUDA_INVALID_CHIRALITY) {
        out = in.chiral_project(0);
        arg.out_right(x_cb, parity) = out;
      }
    }
  };

} // namespace quda
