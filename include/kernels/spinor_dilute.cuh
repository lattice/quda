#include <color_spinor_field_order.h>
#include <constant_kernel_arg.h>
#include <kernel.h>

namespace quda {

  using namespace colorspinor;

  /**
     @brief Helper function that returns the dilution set size, given
     the dilution type.
     @param type The dilution type
   */
  template <int nSpin, int nColor> constexpr auto get_size(QudaDilutionType type)
  {
    switch (type) {
    case QUDA_DILUTION_SPIN: return nSpin;
    case QUDA_DILUTION_COLOR: return nColor;
    case QUDA_DILUTION_SPIN_COLOR: return nSpin * nColor;
    case QUDA_DILUTION_SPIN_COLOR_EVEN_ODD: return nSpin * nColor * 2;
    default: return 1;
    }
  }

  template <typename store_t, int nSpin_, int nColor_, QudaDilutionType type_>
  struct SpinorDiluteArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    static constexpr QudaDilutionType type = type_;
    static constexpr int dilution_size = get_size<nSpin, nColor>(type);
    using V = typename colorspinor_mapper<store_t, nSpin, nColor>::type;
    V v[dilution_size];
    V src;

    /**
       @brief Constructor for the dilution arg
       @param v The output diluted set
       @param src The source vector we are diluting
     */
    template <std::size_t... S>
    SpinorDiluteArg(std::vector<ColorSpinorField> &v, const ColorSpinorField &src, std::index_sequence<S...>) :
      kernel_param(dim3(src.VolumeCB(), src.SiteSubset(), 1)),
      v{v[S]...},
      src(src)
    {
    }
  };

  /**
     Functor for diluting the src vector
   */
  template <typename Arg> struct DiluteSpinor {
    const Arg &arg;
    constexpr DiluteSpinor(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    /**
       @brief Helper function that returns true if we should write the
       source to this set for the given spin and color indices
       @param[in] i Set index
       @param[in] s Spin index
       @param[in] c Color index
    */
    constexpr bool write_source(int i, int s, int c, int parity)
    {
      switch (Arg::type) {
      case QUDA_DILUTION_SPIN: return s == i;
      case QUDA_DILUTION_COLOR: return c == i;
      case QUDA_DILUTION_SPIN_COLOR: return (s * Arg::nColor + c) == i;
      case QUDA_DILUTION_SPIN_COLOR_EVEN_ODD: return ((parity * Arg::nSpin + s) * Arg::nColor + c) == i;
      }
      return 0;
    }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      using vector = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
      vector src = arg.src(x_cb, parity);

      for (int i = 0; i < Arg::dilution_size; i++) {
        vector v;

        for (int s = 0; s < Arg::nSpin; s++) {
          for (int c = 0; c < Arg::nColor; c++) {
            v(s, c) = write_source(i, s, c, parity) ? src(s, c) : complex<typename Arg::real>(0.0, 0.0);
          }
        }

        arg.v[i](x_cb, parity) = v;
      }
    }

  };

}
