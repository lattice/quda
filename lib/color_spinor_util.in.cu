#include <tuple>
#include <memory>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <blas_quda.h>
#include <instantiate.h>

namespace quda {

  using namespace colorspinor;

  /**
     Random number insertion over all field elements
  */
  template <class T>
  void random(T &t) {
    for (int parity=0; parity<t.Nparity(); parity++) {
      for (int x_cb=0; x_cb<t.VolumeCB(); x_cb++) {
      	for (int s=0; s<t.Nspin(); s++) {
      	  for (int c=0; c<t.Ncolor(); c++) {
            t(parity,x_cb,s,c) = complex<typename T::real>(comm_drand(), comm_drand());
      	  }
      	}
      }
    }
  }

  /**
     Create a point source at spacetime point x, spin s and colour c
  */
  template <class T>
  void point(T &t, int x, int s, int c) { t(x%2, x/2, s, c) = 1.0; }

  /**
     Set all space-time real elements at spin s and color c of the
     field equal to k
  */
  template <class T>
  void constant(T &t, int k, int s, int c) {
    for (int parity=0; parity<t.Nparity(); parity++) {
      for (int x_cb=0; x_cb<t.VolumeCB(); x_cb++) {
      	// set all color-spin components to zero
      	for (int s2=0; s2<t.Nspin(); s2++) {
      	  for (int c2=0; c2<t.Ncolor(); c2++) {
      	    t(parity,x_cb,s2,c2) = 0.0;
      	  }
      	}
        t(parity,x_cb,s,c) = k; // now set the one we want
      }
    }
  }

  /**
     Insert a sinusoidal wave sin ( n * (x[d] / X[d]) * pi ) in dimension d
   */
  template <class P>
  void sin(P &p, int d, int n, int offset, const ColorSpinorField &meta) {
    int coord[4];
    int X[4] = { meta.X(0), meta.X(1), meta.X(2), meta.X(3) };
    X[0] *= (p.Nparity() == 1) ? 2 : 1; // need full lattice dims

    for (int parity=0; parity<p.Nparity(); parity++) {
      for (int x_cb=0; x_cb<p.VolumeCB(); x_cb++) {
        getCoords(coord, x_cb, X, parity);

        double mode = n * (double)coord[d] / X[d];
        double k = (double)offset + sin (M_PI * mode);

        for (int s=0; s<p.Nspin(); s++)
          for (int c=0; c<p.Ncolor(); c++)
            p(parity, x_cb, s, c) = k;
      }
    }
  }

  /**
     Create a corner source with value "v" on color "c"
     on a single corner overloaded into "s". "s" is
     encoded via a bitmap: 1010 -> x = 0, y = 1, z = 0, t = 1
     corner, for ex.
  */
  template <class T>
  void corner(T &p, int v, int s, int c, const ColorSpinorField &meta) {
    if (p.Nspin() != 1) errorQuda("corner() is only defined for Nspin = 1 fields");

    int coord[4];
    int X[4] = { meta.X(0), meta.X(1), meta.X(2), meta.X(3) };
    X[0] *= (p.Nparity() == 1) ? 2 : 1; // need full lattice dims

    for (int parity=0; parity<p.Nparity(); parity++) {
      for (int x_cb=0; x_cb<p.VolumeCB(); x_cb++) {

        // get coords
        getCoords(coord, x_cb, X, parity);

        // Figure out corner of current site.
        int corner = 8*(coord[3]%2)+4*(coord[2]%2)+2*(coord[1]%2)+(coord[0]%2);

        // set all color components to zero
        for (int c2=0; c2<p.Ncolor(); c2++) {
          p(parity,x_cb,0,c2) = 0.0;
        }
        // except the corner and color we want
        if (s == corner)
          p(parity,x_cb,0,c) = (double)v;
      }
    }
  }

  // print out the vector at volume point x
  template <typename Float, int nSpin, int nColor, QudaFieldOrder order, typename pack_t>
  void genericSource(const pack_t &pack)
  {
    auto &a = std::get<0>(pack);
    auto &sourceType = std::get<1>(pack);
    auto &x = std::get<2>(pack);
    auto &s = std::get<3>(pack);
    auto &c = std::get<4>(pack);

    FieldOrderCB<Float,nSpin,nColor,1,order> A(a);
    if (sourceType == QUDA_RANDOM_SOURCE) random(A);
    else if (sourceType == QUDA_POINT_SOURCE) point(A, x, s, c);
    else if (sourceType == QUDA_CONSTANT_SOURCE) constant(A, x, s, c);
    else if (sourceType == QUDA_SINUSOIDAL_SOURCE) sin(A, x, s, c, a);
    else if (sourceType == QUDA_CORNER_SOURCE) corner(A, x, s, c, a);
    else errorQuda("Unsupported source type %d", sourceType);
  }

  template <int...> struct IntList { };

  template <typename Float, int nSpin, QudaFieldOrder order, typename pack_t, int nColor, int...N>
  void genericSource(const pack_t &pack, IntList<nColor, N...>)
  {
    auto &a = std::get<0>(pack);

    if (a.Ncolor() == nColor) {
      genericSource<Float, nSpin, nColor, order>(pack);
    } else {
      if constexpr (sizeof...(N) > 0) {
        genericSource<Float, nSpin, order>(pack, IntList<N...>());
      } else {
        errorQuda("Unsupported nColor=%d", a.Ncolor());
      }
    }
  }

  template <typename Float, QudaFieldOrder order, typename pack_t>
  void genericSource(const pack_t &pack)
  {
    auto &a = std::get<0>(pack);
    if (!is_enabled_spin(a.Nspin())) errorQuda("nSpin=%d not enabled for this build", a.Nspin());

    if (a.Nspin() == 1) {
      if constexpr (is_enabled_spin(1)) genericSource<Float, 1, order>(pack, IntList<3>());
    } else if (a.Nspin() == 2) {
      //if constexpr (is_enabled_spin(2)) genericSource<Float, 2, order>(pack, IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@>());
    } else if (a.Nspin() == 4) {
      if constexpr (is_enabled_spin(4)) genericSource<Float, 4 ,order>(pack, IntList<3>());
    } else {
      errorQuda("Unsupported nSpin=%d", a.Nspin());
    }
  }

  template <typename Float, typename pack_t>
  void genericSource(const pack_t &pack)
  {
    auto &a = std::get<0>(pack);
    if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      genericSource<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(pack);
    } else {
      errorQuda("Unsupported field order %d", a.FieldOrder());
    }
  }

  void genericSource(ColorSpinorField &a, QudaSourceType sourceType, int x, int s, int c)
  {
    if (a.Location() == QUDA_CUDA_FIELD_LOCATION) errorQuda("device field not implemented");
    using pack_t = std::tuple<ColorSpinorField&, QudaSourceType, int, int, int>;
    pack_t pack(a, sourceType, x, s, c);
    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      genericSource<double>(pack);
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      genericSource<float>(pack);
    } else {
      errorQuda("Precision not supported");
    }
  }

  template <class U, class V>
  int compareSpinor(const U &u, const V &v, const int tol)
  {
    int fail_check = 16*tol;
    std::vector<int> fail(fail_check);
    for (int f=0; f<fail_check; f++) fail[f] = 0;

    int N = 2*u.Nspin()*u.Ncolor();
    std::vector<int> iter(N);
    for (int i=0; i<N; i++) iter[i] = 0;

    for (int parity=0; parity<v.Nparity(); parity++) {
      for (int x_cb=0; x_cb<u.VolumeCB(); x_cb++) {

	for (int s=0; s<u.Nspin(); s++) {
	  for (int c=0; c<u.Ncolor(); c++) {
            complex<double> u_ = u(parity, x_cb, s, c);
            complex<double> v_ = v(parity, x_cb, s, c);

            double diff_real = fabs(u_.real() - v_.real());
            double diff_imag = fabs(u_.imag() - v_.imag());

            for (int f=0; f<fail_check; f++) {
              if (diff_real > pow(10.0,-(f+1)/(double)tol) || std::isnan(diff_real)) fail[f]++;
              if (diff_imag > pow(10.0,-(f+1)/(double)tol) || std::isnan(diff_imag)) fail[f]++;
            }

            int j = (s * u.Ncolor() + c) * 2;
            if (diff_real > 1e-3 || std::isnan(diff_real)) iter[j+0]++;
            if (diff_imag > 1e-3 || std::isnan(diff_imag)) iter[j+1]++;
	  }
	}
      }
    }

    // reduce over all processes
    for (int i=0; i<N; i++) comm_allreduce_int(iter[i]);
    for (int f=0; f<fail_check; f++) comm_allreduce_int(fail[f]);

    for (int i=0; i<N; i++) printfQuda("%d fails = %d\n", i, iter[i]);

    int accuracy_level =0;
    for (int f=0; f<fail_check; f++) {
      if (fail[f] == 0) accuracy_level = f+1;
    }

    size_t total = u.Nparity()*u.VolumeCB()*N*comm_size();
    for (int f=0; f<fail_check; f++) {
      printfQuda("%e Failures: %d / %lu  = %e\n", pow(10.0,-(f+1)/(double)tol),
		 fail[f], total, fail[f] / (double)total);
    }

    return accuracy_level;
  }

  template <typename oFloat, typename iFloat, QudaFieldOrder order>
  int genericCompare(const ColorSpinorField &a, const ColorSpinorField &b, int tol) {
    int ret = 0;
    if (a.Ncolor() == 3) {
      constexpr int Nc = 3;
      if (a.Nspin() == 4) {
        constexpr int Ns = 4;
        FieldOrderCB<oFloat,Ns,Nc,1,order> A(a);
	FieldOrderCB<iFloat,Ns,Nc,1,order> B(b);

        double rescale = 1.0 / A.abs_max(a);

        auto a_(a), b_(b);
        blas::ax(rescale, a_);
        blas::ax(rescale, b_);
        FieldOrderCB<oFloat, Ns, Nc, 1, order> A_(a_);
        FieldOrderCB<iFloat, Ns, Nc, 1, order> B_(b_);

        ret = compareSpinor(A_, B_, tol);
      } else if (a.Nspin() == 1) {
        constexpr int Ns = 1;
        FieldOrderCB<oFloat,Ns,Nc,1,order> A(a);
	FieldOrderCB<iFloat,Ns,Nc,1,order> B(b);

        double rescale = 1.0 / A.abs_max(a);

        auto a_(a), b_(b);
        blas::ax(rescale, a_);
        blas::ax(rescale, b_);
        FieldOrderCB<oFloat, Ns, Nc, 1, order> A_(a_);
        FieldOrderCB<iFloat, Ns, Nc, 1, order> B_(b_);

        ret = compareSpinor(A_, B_, tol);
      }
    } else {
      errorQuda("Number of colors %d not supported", a.Ncolor());
    }
    return ret;
  }


  template <typename oFloat, typename iFloat>
  int genericCompare(const ColorSpinorField &a, const ColorSpinorField &b, int tol) {
    int ret = 0;
    if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER && b.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      ret = genericCompare<oFloat,iFloat,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(a, b, tol);
    } else {
      errorQuda("Unsupported field order %d", a.FieldOrder());
    }
    return ret;
  }


  template <typename oFloat>
  int genericCompare(const ColorSpinorField &a, const ColorSpinorField &b, int tol) {
    int ret = 0;
    if (b.Precision() == QUDA_DOUBLE_PRECISION) {
      ret = genericCompare<oFloat,double>(a, b, tol);
    } else if (b.Precision() == QUDA_SINGLE_PRECISION) {
      ret = genericCompare<oFloat,float>(a, b, tol);
    } else {
      errorQuda("Precision not supported");
    }
    return ret;
  }


  int genericCompare(const ColorSpinorField &a, const ColorSpinorField &b, int tol) {
    if (a.Location() == QUDA_CUDA_FIELD_LOCATION) errorQuda("device field not implemented");
    int ret = 0;
    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      ret = genericCompare<double>(a, b, tol);
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      ret = genericCompare<float>(a, b, tol);
    } else {
      errorQuda("Precision not supported");
    }
    return ret;
  }

  template <class Order>
  void print_vector(const Order &o, int parity, unsigned int x_cb)
  {
    for (int s = 0; s < o.Nspin(); s++) {
      printf("rank = %d x = %u, s = %d, { ", comm_rank(), x_cb, s);
      for (int c = 0; c < o.Ncolor(); c++) {
        auto value = complex<double>(o(parity, x_cb, s, c));
        printf("(%f,%f) ", value.real(), value.imag());
      }
      printf("}\n");
    }
  }

  template <typename Float, int nSpin, int nColor>
  void genericPrintVector(const ColorSpinorField &a, int parity, unsigned int x_cb)
  {
    if (a.isNative()) {
      constexpr auto order = colorspinor::getNative<Float>(nSpin);
      print_vector(FieldOrderCB<double, nSpin, nColor, 1, order, Float, Float, false, true>(a), parity, x_cb);
    } else if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      constexpr auto order = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      print_vector(FieldOrderCB<double, nSpin, nColor, 1, order, Float, Float, false, true>(a), parity, x_cb);
    } else {
      errorQuda("Unsupported field order %d", a.FieldOrder());
    }
  }

  template <typename Float, int nSpin, int nColor, int...N>
  void genericPrintVector(const ColorSpinorField &a, int parity, unsigned int x_cb, IntList<nColor, N...>)
  {
    if (a.Ncolor() == nColor) {
      genericPrintVector<Float, nSpin, nColor>(a, parity, x_cb);
    } else {
      if constexpr (sizeof...(N) > 0) {
        genericPrintVector<Float, nSpin, N...>(a, parity, x_cb, IntList<N...>());
      } else {
        errorQuda("Not supported Ncolor = %d", a.Ncolor());
      }
    }
  }

  template <typename Float> void genericPrintVector(const ColorSpinorField &a, int parity, unsigned int x_cb)
  {
    if (!is_enabled_spin(a.Nspin())) errorQuda("Nspin = %d not enabled", a.Nspin());

    if (a.Nspin() == 1) {
      if constexpr (is_enabled_spin(1)) genericPrintVector<Float, 1>(a, parity, x_cb, IntList<3>());
    } else if (a.Nspin() == 2) {
      if constexpr (is_enabled_spin(2)) genericPrintVector<Float, 2>(a, parity, x_cb, IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@>());
    } else if (a.Nspin() == 4) {
      if constexpr (is_enabled_spin(4)) genericPrintVector<Float, 4>(a, parity, x_cb, IntList<3>());
    }
  }

  void genericPrintVector(const ColorSpinorField &a, int parity, unsigned int x_cb, int rank)
  {
    if (rank != comm_rank()) return;

    ColorSpinorParam param(a);
    param.location = QUDA_CPU_FIELD_LOCATION;
    param.create = QUDA_COPY_FIELD_CREATE;
    // if field is a pinned device field then we need to clone it on the host
    bool host_clone = (a.Location() == QUDA_CUDA_FIELD_LOCATION && a.MemType() == QUDA_MEMORY_DEVICE && !use_managed_memory()) ? true : false;
    std::unique_ptr<ColorSpinorField> clone_a = !host_clone ? nullptr : std::make_unique<ColorSpinorField>(param);
    const ColorSpinorField &a_ = !host_clone ? a : *clone_a.get();

    switch (a.Precision()) {
    case QUDA_DOUBLE_PRECISION:  genericPrintVector<double>(a_, parity, x_cb); break;
    case QUDA_SINGLE_PRECISION:  genericPrintVector<float>(a_, parity, x_cb); break;
    case QUDA_HALF_PRECISION:    genericPrintVector<short>(a_, parity, x_cb); break;
    case QUDA_QUARTER_PRECISION: genericPrintVector<int8_t>(a_, parity, x_cb); break;
    default: errorQuda("Precision %d not implemented", a.Precision());
    }
  }

  void resize(std::vector<ColorSpinorField> &v, size_t new_size, const ColorSpinorParam &param)
  {
    auto old_size = v.size();
    v.resize(new_size);
    for (auto k = old_size; k < new_size; k++) v[k] = ColorSpinorField(param);
  }

  void resize(std::vector<ColorSpinorField> &v, size_t new_size, QudaFieldCreate create, const ColorSpinorField &src)
  {
    ColorSpinorParam param;
    if (v.size()) {
      param = ColorSpinorParam(v[0]);
    } else if (src.Volume() > 0) {
      param = ColorSpinorParam(src);
    } else {
      errorQuda("Cannot resize an empty vector unless a src is passed");
    }

    param.create = create;
    if (create == QUDA_COPY_FIELD_CREATE) param.field = &const_cast<ColorSpinorField&>(src);
    else if (create == QUDA_REFERENCE_FIELD_CREATE) param.v = const_cast<ColorSpinorField&>(src).V();

    resize(v, new_size, param);
  }

} // namespace quda
