#include <stdlib.h>
#include <stdio.h>
#include <cstring> // needed for memset

#include <tune_quda.h>

#include <quda_internal.h>
#include <float_vector.h>
#include <blas_quda.h>
#include <color_spinor_field.h>

#include <jitify_helper.cuh>
#include <kernels/blas_core.cuh>

namespace quda {

  namespace blas {

#include <generic_blas.cuh>

    unsigned long long flops;
    unsigned long long bytes;

    static qudaStream_t *blasStream;

    template <typename FloatN, int M, typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW,
        typename SpinorV, typename Functor>
    class BlasCuda : public Tunable
    {

  private:
      const int nParity; // for composite fields this includes the number of composites
      mutable BlasArg<SpinorX, SpinorY, SpinorZ, SpinorW, SpinorV, Functor> arg;

      const ColorSpinorField &x, &y, &z, &w, &v;

      // host pointers used for backing up fields when tuning
      // dont't these curry these in to minimize Arg size
      char *X_h, *Y_h, *Z_h, *W_h, *V_h;
      char *Xnorm_h, *Ynorm_h, *Znorm_h, *Wnorm_h, *Vnorm_h;

      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; }

      // for these streaming kernels, there is no need to tune the grid size, just use max
      unsigned int minGridSize() const { return maxGridSize(); }

    public:
      BlasCuda(SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, SpinorV &V, Functor &f, ColorSpinorField &x,
          ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, int length) :
          nParity((x.IsComposite() ? x.CompositeDim() : 1) * x.SiteSubset()), // must be first
          arg(X, Y, Z, W, V, f, length / nParity),
          x(x),
          y(y),
          z(z),
          w(w),
          v(v),
          X_h(0),
          Y_h(0),
          Z_h(0),
          W_h(0),
          V_h(0),
          Xnorm_h(0),
          Ynorm_h(0),
          Znorm_h(0),
          Wnorm_h(0),
          Vnorm_h(0)
      {
        strcpy(aux, x.AuxString());
        if (x.Precision() != y.Precision()) {
          strcat(aux, ",");
          strcat(aux, y.AuxString());
        }

#ifdef JITIFY
        ::quda::create_jitify_program("kernels/blas_core.cuh");
#endif
      }

      virtual ~BlasCuda() {}

      inline TuneKey tuneKey() const { return TuneKey(x.VolString(), typeid(arg.f).name(), aux); }

      inline void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::blas::blasKernel")
                           .instantiate(Type<FloatN>(), M, Type<decltype(arg)>())
                           .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                           .launch(arg);
#else
        blasKernel<FloatN, M><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
#endif
      }

      void preTune()
      {
        arg.X.backup(&X_h, &Xnorm_h, x.Bytes(), x.NormBytes());
        arg.Y.backup(&Y_h, &Ynorm_h, y.Bytes(), y.NormBytes());
        arg.Z.backup(&Z_h, &Znorm_h, z.Bytes(), z.NormBytes());
        arg.W.backup(&W_h, &Wnorm_h, w.Bytes(), w.NormBytes());
        arg.V.backup(&V_h, &Vnorm_h, v.Bytes(), v.NormBytes());
      }

      void postTune()
      {
        arg.X.restore(&X_h, &Xnorm_h, x.Bytes(), x.NormBytes());
        arg.Y.restore(&Y_h, &Ynorm_h, y.Bytes(), y.NormBytes());
        arg.Z.restore(&Z_h, &Znorm_h, z.Bytes(), z.NormBytes());
        arg.W.restore(&W_h, &Wnorm_h, w.Bytes(), w.NormBytes());
        arg.V.restore(&V_h, &Vnorm_h, v.Bytes(), v.NormBytes());
      }

      void initTuneParam(TuneParam &param) const
      {
        Tunable::initTuneParam(param);
        param.grid.y = nParity;
      }

      void defaultTuneParam(TuneParam &param) const
      {
        Tunable::initTuneParam(param);
        param.grid.y = nParity;
      }

      long long flops() const { return arg.f.flops() * vec_length<FloatN>::value * arg.length * nParity * M; }
      long long bytes() const
      {
        // the factor two here assumes we are reading and writing to the high precision vector
        // this will evaluate correctly for non-mixed kernels since the +2/-2 will cancel out
        return (arg.f.streams() - 2) * x.Bytes() + 2 * y.Bytes();
      }
      int tuningIter() const { return 3; }
    };

    template <typename RegType, typename StoreType, typename yType, int M, template <typename, typename> class Functor,
        int writeX, int writeY, int writeZ, int writeW, int writeV>
    void nativeBlas(const double2 &a, const double2 &b, const double2 &c, ColorSpinorField &x, ColorSpinorField &y,
        ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, int length)
    {

      checkLength(x, y);
      checkLength(x, z);
      checkLength(x, w);
      checkLength(x, v);

      Spinor<RegType, StoreType, M, writeX> X(x);
      Spinor<RegType, yType, M, writeY> Y(y);
      Spinor<RegType, StoreType, M, writeZ> Z(z);
      Spinor<RegType, StoreType, M, writeW> W(w);
      Spinor<RegType, yType, M, writeV> V(v);

      typedef typename scalar<RegType>::type Float;
      typedef typename vector<Float, 2>::type Float2;
      typedef vector<Float, 2> vec2;
      Functor<Float2, RegType> f((Float2)vec2(a), (Float2)vec2(b), (Float2)vec2(c));

      BlasCuda<RegType, M, decltype(X), decltype(Y), decltype(Z), decltype(W), decltype(V), Functor<Float2, RegType>> blas(
          X, Y, Z, W, V, f, x, y, z, w, v, length);
      blas.apply(*blasStream);

      blas::bytes += blas.bytes();
      blas::flops += blas.flops();

      checkCudaError();
    }

    /**
       Driver for generic blas routine with four loads and two store.  All
       fields must have matching precisions.
     */
    template <template <typename Float, typename FloatN> class Functor, int writeX = 0, int writeY = 0, int writeZ = 0,
        int writeW = 0, int writeV = 0>
    void uni_blas(const double2 &a, const double2 &b, const double2 &c, ColorSpinorField &x, ColorSpinorField &y,
        ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v)
    {
      checkPrecision(x, y, z, w, v);

      if (checkLocation(x, y, z, w, v) == QUDA_CUDA_FIELD_LOCATION) {

        if (!x.isNative() && x.FieldOrder() != QUDA_FLOAT2_FIELD_ORDER && x.FieldOrder() != QUDA_FLOAT8_FIELD_ORDER) {
          warningQuda("Device blas on non-native fields is not supported (prec = %d, order = %d)", x.Precision(),
                      x.FieldOrder());
          return;
        }

        if (x.Precision() == QUDA_DOUBLE_PRECISION) {

#if QUDA_PRECISION & 8
#if defined(NSPIN4) || defined(NSPIN2) || defined(NSPIN1)
          const int M = 1;
          nativeBlas<double2, double2, double2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
              a, b, c, x, y, z, w, v, x.Length() / (2 * M));
#else
          errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

        } else if (x.Precision() == QUDA_SINGLE_PRECISION) {

#if QUDA_PRECISION & 4
          if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
#if defined(NSPIN4)
            const int M = 1;
            nativeBlas<float4, float4, float4, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                a, b, c, x, y, z, w, v, x.Length() / (4 * M));
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 1 || x.Nspin() == 2 || (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER)) {
#if defined(NSPIN1) || defined(NSPIN2)
            const int M = 1;
            nativeBlas<float2, float2, float2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                a, b, c, x, y, z, w, v, x.Length() / (2 * M));
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else {
            errorQuda("nSpin=%d is not supported\n", x.Nspin());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

        } else if (x.Precision() == QUDA_HALF_PRECISION) {

#if QUDA_PRECISION & 2
          if (x.Ncolor() != 3) { errorQuda("nColor = %d is not supported", x.Ncolor()); }
          if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) { // wilson
#if defined(NSPIN4)
            const int M = 6;
            nativeBlas<float4, short4, short4, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                a, b, c, x, y, z, w, v, x.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) { // wilson
#if defined(GPU_MULTIGRID) // FIXME eventually we should get rid of this and use float4 ordering
            const int M = 12;
            nativeBlas<float2, short2, short2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                a, b, c, x, y, z, w, v, x.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT8_FIELD_ORDER) { // wilson
#if defined(NSPIN4) && defined(FLOAT8)
            const int M = 3;
            nativeBlas<float8, short8, short8, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                a, b, c, x, y, z, w, v, x.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
            const int M = 3;
            nativeBlas<float2, short2, short2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                a, b, c, x, y, z, w, v, x.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else {
            errorQuda("nSpin=%d is not supported\n", x.Nspin());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

        } else if (x.Precision() == QUDA_QUARTER_PRECISION) {

#if QUDA_PRECISION & 1
          if (x.Ncolor() != 3) { errorQuda("nColor = %d is not supported", x.Ncolor()); }
          if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) { // wilson
#if defined(NSPIN4)
            const int M = 6;
            nativeBlas<float4, char4, char4, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                a, b, c, x, y, z, w, v, x.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) { // wilson
#if defined(GPU_MULTIGRID) // FIXME eventually we should get rid of this and use float4 ordering
            const int M = 12;
            nativeBlas<float2, char2, char2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(a, b, c, x, y, z, w, v,
                                                                                                 x.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT8_FIELD_ORDER) { // wilson
#if defined(NSPIN4) && defined(FLOAT8)
            const int M = 3;
            nativeBlas<float8, char8, char8, M, Functor, writeX, writeY, writeZ, writeW, writeV>(a, b, c, x, y, z, w, v,
                                                                                                 x.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
            const int M = 3;
            nativeBlas<float2, char2, char2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                a, b, c, x, y, z, w, v, x.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else {
            errorQuda("nSpin=%d is not supported\n", x.Nspin());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

        } else {
          errorQuda("precision=%d is not supported\n", x.Precision());
        }
      } else { // fields on the cpu
        if (x.Precision() == QUDA_DOUBLE_PRECISION) {
          Functor<double2, double2> f(a, b, c);
          genericBlas<double, double, writeX, writeY, writeZ, writeW, writeV>(x, y, z, w, v, f);
        } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
          Functor<float2, float2> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
          genericBlas<float, float, writeX, writeY, writeZ, writeW, writeV>(x, y, z, w, v, f);
        } else {
          errorQuda("Not implemented");
        }
      }
    }

    /**
       Driver for generic blas routine with four loads and two store.
       This is the mixed-precision driver which supports a different
       precision for (x,z,w) and (y,v), where the former is the low
       precision and the latter is the high precision.
    */
    template <template <typename Float, typename FloatN> class Functor, int writeX = 0, int writeY = 0, int writeZ = 0,
        int writeW = 0, int writeV = 0>
    void mixed_blas(const double2 &a, const double2 &b, const double2 &c, ColorSpinorField &x, ColorSpinorField &y,
        ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v)
    {

      checkPrecision(x, z, w);
      checkPrecision(y, v);

      if (checkLocation(x, y, z, w, v) == QUDA_CUDA_FIELD_LOCATION) {

        if (!x.isNative()
            && !(x.FieldOrder() == QUDA_FLOAT8_FIELD_ORDER && y.FieldOrder() == QUDA_FLOAT8_FIELD_ORDER
                 && x.Precision() == QUDA_QUARTER_PRECISION && y.Precision() == QUDA_HALF_PRECISION)) {
          warningQuda("Device blas on non-native fields is not supported (prec = %d, order = %d)", x.Precision(),
                      x.FieldOrder());
          return;
        }

        if (x.Precision() == QUDA_SINGLE_PRECISION && y.Precision() == QUDA_DOUBLE_PRECISION) {

#if QUDA_PRECISION & 4
          if (x.Nspin() == 4) {
#if defined(NSPIN4)
            const int M = 12;
            nativeBlas<double2, float4, double2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                a, b, c, x, y, z, w, v, x.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 1) {
#if defined(NSPIN1)
            const int M = 3;
            nativeBlas<double2, float2, double2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                a, b, c, x, y, z, w, v, x.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

        } else if (x.Precision() == QUDA_HALF_PRECISION) {

#if QUDA_PRECISION & 2
          if (y.Precision() == QUDA_DOUBLE_PRECISION) {

#if QUDA_PRECISION & 8
            if (x.Nspin() == 4) {
#if defined(NSPIN4)
              const int M = 12;
              nativeBlas<double2, short4, double2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                  a, b, c, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) {
#if defined(NSPIN1)
              const int M = 3;
              nativeBlas<double2, short2, double2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                  a, b, c, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, y.Precision());
#endif

          } else if (y.Precision() == QUDA_SINGLE_PRECISION) {

#if QUDA_PRECISION & 4
            if (x.Nspin() == 4) {
#if defined(NSPIN4)
              const int M = 6;
              nativeBlas<float4, short4, float4, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                  a, b, c, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) {
#if defined(NSPIN1)
              const int M = 3;
              nativeBlas<float2, short2, float2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                  a, b, c, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, y.Precision());
#endif

          } else {
            errorQuda("Not implemented for this precision combination %d %d", x.Precision(), y.Precision());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

        } else if (x.Precision() == QUDA_QUARTER_PRECISION) {

#if QUDA_PRECISION & 1

          if (y.Precision() == QUDA_DOUBLE_PRECISION) {

#if QUDA_PRECISION & 8
            if (x.Nspin() == 4) {
#if defined(NSPIN4)
              const int M = 12;
              nativeBlas<double2, char4, double2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                  a, b, c, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) {
#if defined(NSPIN1)
              const int M = 3;
              nativeBlas<double2, char2, double2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                  a, b, c, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, y.Precision());
#endif

          } else if (y.Precision() == QUDA_SINGLE_PRECISION) {

#if QUDA_PRECISION & 4
            if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
#if defined(NSPIN4)
              const int M = 6;
              nativeBlas<float4, char4, float4, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                  a, b, c, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) {
#if defined(NSPIN1)
              const int M = 3;
              nativeBlas<float2, char2, float2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                  a, b, c, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, y.Precision());
#endif

          } else if (y.Precision() == QUDA_HALF_PRECISION) {

#if QUDA_PRECISION & 2
            if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
#if defined(NSPIN4)
              const int M = 6;
              nativeBlas<float4, char4, short4, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                  a, b, c, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            }
            if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT8_FIELD_ORDER && y.FieldOrder() == QUDA_FLOAT8_FIELD_ORDER) {
#if defined(NSPIN4)
              const int M = 3;
              nativeBlas<float8, char8, short8, M, Functor, writeX, writeY, writeZ, writeW, writeV>(a, b, c, x, y, z, w,
                                                                                                    v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) {
#if defined(NSPIN1)
              const int M = 3;
              nativeBlas<float2, char2, short2, M, Functor, writeX, writeY, writeZ, writeW, writeV>(
                  a, b, c, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, y.Precision());
#endif

          } else {
            errorQuda("Not implemented for this precision combination %d %d", x.Precision(), y.Precision());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

        } else {
          errorQuda("Not implemented for this precision combination %d %d", x.Precision(), y.Precision());
        }

      } else { // fields on the cpu
        using namespace quda::colorspinor;
        if (x.Precision() == QUDA_SINGLE_PRECISION && y.Precision() == QUDA_DOUBLE_PRECISION) {
          Functor<double2, double2> f(a, b, c);
          genericBlas<float, double, writeX, writeY, writeZ, writeW, writeV>(x, y, z, w, v, f);
        } else {
          errorQuda("Not implemented");
        }
      }
    }

    void zero(ColorSpinorField &a) {
      if (typeid(a) == typeid(cudaColorSpinorField)) {
	static_cast<cudaColorSpinorField&>(a).zero();
      } else {
	static_cast<cpuColorSpinorField&>(a).zero();
      }
    }

    void initReduce();
    void endReduce();

    void init()
    {
      blasStream = &streams[Nstream-1];
      initReduce();
    }

    void end(void)
    {
      endReduce();
    }

    qudaStream_t* getStream() { return blasStream; }

    void axpbyz(double a, ColorSpinorField &x, double b,
                ColorSpinorField &y, ColorSpinorField &z) {
      if (x.Precision() != y.Precision()) {
	// call hacked mixed precision kernel
        mixed_blas<axpbyz_, 0, 0, 0, 0, 1>(
            make_double2(a, 0.0), make_double2(b, 0.0), make_double2(0.0, 0.0), x, y, x, x, z);
      } else {
        uni_blas<axpbyz_, 0, 0, 0, 0, 1>(
            make_double2(a, 0.0), make_double2(b, 0.0), make_double2(0.0, 0.0), x, y, x, x, z);
      }
    }

    void ax(double a, ColorSpinorField &x) {
      uni_blas<ax_, 1>(make_double2(a, 0.0), make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, x, x, x, x);
    }

    void caxpy(const Complex &a, ColorSpinorField &x, ColorSpinorField &y) {
      if (x.Precision() != y.Precision()) {
        mixed_blas<caxpy_, 0, 1>(
            make_double2(real(a), imag(a)), make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, y);
      } else {
        uni_blas<caxpy_, 0, 1>(
            make_double2(real(a), imag(a)), make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, y);
      }
    }


    void caxpby(const Complex &a, ColorSpinorField &x, const Complex &b, ColorSpinorField &y) {
      uni_blas<caxpby_, 0, 1>(
          make_double2(REAL(a), IMAG(a)), make_double2(REAL(b), IMAG(b)), make_double2(0.0, 0.0), x, y, x, x, y);
    }

    void caxpbypczw(const Complex &a, ColorSpinorField &x, const Complex &b, ColorSpinorField &y, const Complex &c,
                    ColorSpinorField &z, ColorSpinorField &w)
    {
      uni_blas<caxpbypczw_, 0, 0, 0, 1>(make_double2(REAL(a), IMAG(a)), make_double2(REAL(b), IMAG(b)),
                                        make_double2(REAL(c), IMAG(c)), x, y, z, w, y);
    }

    void cxpaypbz(ColorSpinorField &x, const Complex &a, ColorSpinorField &y,
		  const Complex &b, ColorSpinorField &z) {
      uni_blas<caxpbypczw_, 0, 0, 0, 1>(make_double2(1.0, 0.0), make_double2(REAL(a), IMAG(a)),
                                        make_double2(REAL(b), IMAG(b)), x, y, z, z, y);
    }

    void axpyBzpcx(double a, ColorSpinorField& x, ColorSpinorField& y, double b,
		   ColorSpinorField& z, double c) {
      if (x.Precision() != y.Precision()) {
	// call hacked mixed precision kernel
        mixed_blas<axpyBzpcx_, 1, 1>(make_double2(a, 0.0), make_double2(b, 0.0), make_double2(c, 0.0), x, y, z, x, y);
      } else {
	// swap arguments around
        uni_blas<axpyBzpcx_, 1, 1>(make_double2(a, 0.0), make_double2(b, 0.0), make_double2(c, 0.0), x, y, z, x, y);
      }
    }

    void axpyZpbx(double a, ColorSpinorField& x, ColorSpinorField& y,
		  ColorSpinorField& z, double b) {
      if (x.Precision() != y.Precision()) {
	// call hacked mixed precision kernel
        mixed_blas<axpyZpbx_, 1, 1>(make_double2(a, 0.0), make_double2(b, 0.0), make_double2(0.0, 0.0), x, y, z, x, y);
      } else {
	// swap arguments around
        uni_blas<axpyZpbx_, 1, 1>(make_double2(a, 0.0), make_double2(b, 0.0), make_double2(0.0, 0.0), x, y, z, x, y);
      }
    }

    void caxpyBzpx(const Complex &a, ColorSpinorField &x,
		      ColorSpinorField &y, const Complex &b, ColorSpinorField &z) {
      if (x.Precision() != y.Precision()) {
        mixed_blas<caxpyBzpx_, 1, 1>(
            make_double2(REAL(a), IMAG(a)), make_double2(REAL(b), IMAG(b)), make_double2(0.0, 0.0), x, y, z, x, y);
      } else {
        uni_blas<caxpyBzpx_, 1, 1>(
            make_double2(REAL(a), IMAG(a)), make_double2(REAL(b), IMAG(b)), make_double2(0.0, 0.0), x, y, z, x, y);
      }
    }

    void caxpyBxpz(const Complex &a, ColorSpinorField &x,
		      ColorSpinorField &y, const Complex &b, ColorSpinorField &z) {
      if (x.Precision() != y.Precision()) {
        mixed_blas<caxpyBxpz_, 0, 1, 1>(
            make_double2(REAL(a), IMAG(a)), make_double2(REAL(b), IMAG(b)), make_double2(0.0, 0.0), x, y, z, x, y);
      } else {
        uni_blas<caxpyBxpz_, 0, 1, 1>(
            make_double2(REAL(a), IMAG(a)), make_double2(REAL(b), IMAG(b)), make_double2(0.0, 0.0), x, y, z, x, y);
      }
    }

    void caxpbypzYmbw(const Complex &a, ColorSpinorField &x, const Complex &b,
		      ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w) {
      uni_blas<caxpbypzYmbw_, 0, 1, 1>(
          make_double2(REAL(a), IMAG(a)), make_double2(REAL(b), IMAG(b)), make_double2(0.0, 0.0), x, y, z, w, y);
    }

    void cabxpyAx(double a, const Complex &b, ColorSpinorField &x, ColorSpinorField &y) {
      // swap arguments around
      uni_blas<cabxpyAx_, 1, 1>(
          make_double2(a, 0.0), make_double2(REAL(b), IMAG(b)), make_double2(0.0, 0.0), x, y, x, x, y);
    }

    void caxpyXmaz(const Complex &a, ColorSpinorField &x,
		   ColorSpinorField &y, ColorSpinorField &z) {
      uni_blas<caxpyxmaz_, 1, 1>(
          make_double2(REAL(a), IMAG(a)), make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, z, x, y);
    }

    void caxpyXmazMR(const Complex &a, ColorSpinorField &x,
		     ColorSpinorField &y, ColorSpinorField &z) {
      if (!commAsyncReduction())
	errorQuda("This kernel requires asynchronous reductions to be set");
      if (x.Location() == QUDA_CPU_FIELD_LOCATION)
	errorQuda("This kernel cannot be run on CPU fields");

      uni_blas<caxpyxmazMR_, 1, 1>(
          make_double2(REAL(a), IMAG(a)), make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, z, x, y);
    }

    void tripleCGUpdate(double a, double b, ColorSpinorField &x,
			ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w) {
      if (x.Precision() != y.Precision()) {
      // call hacked mixed precision kernel
      mixed_blas<tripleCGUpdate_, 0, 1, 1, 1>(
          make_double2(a, 0.0), make_double2(b, 0.0), make_double2(0.0, 0.0), x, y, z, w, y);
      } else {
        uni_blas<tripleCGUpdate_, 0, 1, 1, 1>(
            make_double2(a, 0.0), make_double2(b, 0.0), make_double2(0.0, 0.0), x, y, z, w, y);
      }
    }

    void doubleCG3Init(double a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      uni_blas<doubleCG3Init_, 1, 1, 0, 0>(
          make_double2(a, 0.0), make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, z, z, y);
    }

    void doubleCG3Update(double a, double b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      uni_blas<doubleCG3Update_, 1, 1, 0, 0>(
          make_double2(a, 0.0), make_double2(b, 1.0 - b), make_double2(0.0, 0.0), x, y, z, z, y);
    }

  } // namespace blas

} // namespace quda
