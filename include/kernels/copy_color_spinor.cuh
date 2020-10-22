#include <color_spinor_field_order.h>
#include <kernel.h>

#define PRESERVE_SPINOR_NORM

#ifdef PRESERVE_SPINOR_NORM // Preserve the norm regardless of basis
#define kP (1.0/sqrt(2.0))
#define kU (1.0/sqrt(2.0))
#else // More numerically accurate not to preserve the norm between basis
#define kP (0.5)
#define kU (1.0)
#endif

namespace quda {

  using namespace colorspinor;

  template <typename FloatOut, typename FloatIn, int nSpin_, int nColor_, typename Out, typename In, template <typename> class Basis_, typename param_t>
  struct CopyColorSpinorArg {
    template <typename Arg_> using Basis = Basis_<Arg_>;
    using realOut = typename mapper<FloatOut>::type;
    using realIn = typename mapper<FloatIn>::type;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    Out out;
    const In in;
    const int outParity;
    const int inParity;
    dim3 threads;
    CopyColorSpinorArg(ColorSpinorField &out, const ColorSpinorField &in, param_t &param) :
      out(out, 1, std::get<0>(param), std::get<2>(param)),
      in(in, 1, std::get<1>(param), std::get<3>(param)),
      outParity(out.SiteOrder()==QUDA_ODD_EVEN_SITE_ORDER ? 1 : 0),
      inParity(in.SiteOrder()==QUDA_ODD_EVEN_SITE_ORDER ? 1 : 0),
      threads(in.VolumeCB(), in.SiteSubset(), 1)
    { }
  };

  /** Straight copy with no basis change */
  template <typename Arg>
  struct PreserveBasis {
    static constexpr int Ns = Arg::nSpin;
    static constexpr int Nc = Arg::nColor;
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns*Nc], const complex<FloatIn> in[Ns*Nc]) const {
      for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++) out[s*Nc+c] = in[s*Nc+c];
    }
  };

  /** Transform from relativistic into non-relavisitic basis */
  template <typename Arg>
  struct NonRelBasis {
    static constexpr int Ns = Arg::nSpin;
    static constexpr int Nc = Arg::nColor;
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns*Nc], const complex<FloatIn> in[Ns*Nc]) const {
      int s1[4] = {1, 2, 3, 0};
      int s2[4] = {3, 0, 1, 2};
      FloatOut K1[4] = {static_cast<FloatOut>(kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(-kP)};
      FloatOut K2[4] = {static_cast<FloatOut>(kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP)};
      for (int s=0; s<Ns; s++) {
	for (int c=0; c<Nc; c++) {
	  out[s*Nc+c] = K1[s]*static_cast<complex<FloatOut> >(in[s1[s]*Nc+c]) + K2[s]*static_cast<complex<FloatOut> >(in[s2[s]*Nc+c]);
	}
      }
    }
  };

  /** Transform from non-relativistic into relavisitic basis */
  template <typename Arg>
  struct RelBasis {
    static constexpr int Ns = Arg::nSpin;
    static constexpr int Nc = Arg::nColor;
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns*Nc], const complex<FloatIn> in[Ns*Nc]) const {
      int s1[4] = {1, 2, 3, 0};
      int s2[4] = {3, 0, 1, 2};
      FloatOut K1[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU),  static_cast<FloatOut>(kU)};
      FloatOut K2[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(-kU), static_cast<FloatOut>(-kU)};
      for (int s=0; s<Ns; s++) {
	for (int c=0; c<Nc; c++) {
	  out[s*Nc+c] = K1[s]*static_cast<complex<FloatOut> >(in[s1[s]*Nc+c]) + K2[s]*static_cast<complex<FloatOut> >(in[s2[s]*Nc+c]);
	}
      }
    }
  };

  /** Transform from chiral into non-relavisitic basis */
  template <typename Arg>
  struct ChiralToNonRelBasis {
    static constexpr int Ns = Arg::nSpin;
    static constexpr int Nc = Arg::nColor;
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns*Nc], const complex<FloatIn> in[Ns*Nc]) const {
      int s1[4] = {0, 1, 0, 1};
      int s2[4] = {2, 3, 2, 3};
      FloatOut K1[4] = {static_cast<FloatOut>(-kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP)};
      FloatOut K2[4] = {static_cast<FloatOut>(kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP)};
      for (int s=0; s<Ns; s++) {
	for (int c=0; c<Nc; c++) {
	  out[s*Nc+c] = K1[s]*static_cast<complex<FloatOut> >(in[s1[s]*Nc+c]) + K2[s]*static_cast<complex<FloatOut> >(in[s2[s]*Nc+c]);
	}
      }
    }
  };

  /** Transform from non-relativistic into chiral basis */
  template <typename Arg>
  struct NonRelToChiralBasis {
    static constexpr int Ns = Arg::nSpin;
    static constexpr int Nc = Arg::nColor;
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns*Nc], const complex<FloatIn> in[Ns*Nc]) const {
      int s1[4] = {0, 1, 0, 1};
      int s2[4] = {2, 3, 2, 3};
      FloatOut K1[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU)};
      FloatOut K2[4] = {static_cast<FloatOut>(kU),static_cast<FloatOut>(kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU)};
      for (int s=0; s<Ns; s++) {
	for (int c=0; c<Nc; c++) {
	  out[s*Nc+c] = K1[s]*static_cast<complex<FloatOut> >(in[s1[s]*Nc+c]) + K2[s]*static_cast<complex<FloatOut> >(in[s2[s]*Nc+c]);
	}
      }
    }
  };

  template <typename Arg> struct CopyColorSpinor_ {
    Arg &arg;
    constexpr CopyColorSpinor_(Arg &arg): arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      ColorSpinor<typename Arg::realIn, Arg::nColor, Arg::nSpin> in = arg.in(x_cb, (parity+arg.inParity)&1);
      ColorSpinor<typename Arg::realOut, Arg::nColor, Arg::nSpin> out;
      typename Arg::Basis<Arg> basis;
      basis(out.data, in.data);
      arg.out(x_cb, (parity+arg.outParity)&1) = out;
    }
  };

}
