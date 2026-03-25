#include <color_spinor_field_order.h>
#include <kernel.h>

namespace quda {

  using namespace colorspinor;

  template <typename store_t, int nSpin_, int nColor_>
  struct SpinorDuplicateArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    using V = typename colorspinor_mapper<store_t, nSpin, nColor, false, false, true>::type;
    V v[nSpin];
    V src;

    /**
       @brief Constructor for the duplication arg
       @param v The spin duplicated set
       @param src The source vector we are duplicating
     */
    SpinorDuplicateArg(cvector_ref<ColorSpinorField> &v, const ColorSpinorField &src) :
      kernel_param(dim3(src.VolumeCB(), src.SiteSubset(), 1)),
      src(src)
    {
      for (auto i = 0u; i < v.size(); i++) this->v[i] = V(v[i]);
    }
  };

  /**
     Functor for spin duplicating the src vector
   */
  template <typename Arg> struct DuplicateSpinor {
    const Arg &arg;
    constexpr DuplicateSpinor(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      using vector = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
      vector src = arg.src(x_cb, parity);

      for (int i = 0; i < Arg::nSpin; i++) {
        vector v;

        for (int s = 0; s < Arg::nSpin; s++) {
          for (int c = 0; c < Arg::nColor; c++) {
            v(s, c) = src(i, c);
          }
        }

        arg.v[i](x_cb, parity) = v;
      }
    }

  };

}
