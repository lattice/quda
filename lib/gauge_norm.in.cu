#include <gauge_field_order.h>
#include <instantiate.h>
#include <memory>

namespace quda {

  using namespace gauge;

  enum norm_type_ {
    NORM1,
    NORM2,
    ABS_MAX,
    ABS_MIN
  };

  template <typename reg_type, typename real, int Nc, QudaGaugeFieldOrder order>
  double norm(const GaugeField &u, int d, norm_type_ type) {
    double norm_ = 0.0;
    switch(type) {
    case   NORM1: norm_ = FieldOrder<reg_type,Nc,1,order,true,real>(const_cast<GaugeField &>(u)).norm1(d);   break;
    case   NORM2: norm_ = FieldOrder<reg_type,Nc,1,order,true,real>(const_cast<GaugeField &>(u)).norm2(d);   break;
    case ABS_MAX: norm_ = FieldOrder<reg_type,Nc,1,order,true,real>(const_cast<GaugeField &>(u)).abs_max(d); break;
    case ABS_MIN: norm_ = FieldOrder<reg_type,Nc,1,order,true,real>(const_cast<GaugeField &>(u)).abs_min(d); break;
    }
    return norm_;
  }

  template <typename T, bool fixed> struct type_mapper {
    using reg_t = typename mapper<T>::type;
    using store_t = T;
  };

  // fixed-point single-precision field
  template <> struct type_mapper<float, true> {
    using reg_t = float;
    using store_t = int;
  };

  template <typename T, bool fixed, int nColor>
  double norm(const GaugeField &u, int d, norm_type_ type) {
    using reg_t = typename type_mapper<T, fixed>::reg_t;
    using store_t = typename type_mapper<T, fixed>::store_t;
    double norm_ = 0.0;
    switch (u.FieldOrder()) {
    case QUDA_FLOAT2_GAUGE_ORDER: norm_ = norm<reg_t, store_t, nColor, QUDA_FLOAT2_GAUGE_ORDER>(u, d, type); break;
    case QUDA_QDP_GAUGE_ORDER:    norm_ = norm<reg_t, store_t, nColor, QUDA_QDP_GAUGE_ORDER>(u, d, type); break;
    case QUDA_MILC_GAUGE_ORDER:   norm_ = norm<reg_t, store_t, nColor, QUDA_MILC_GAUGE_ORDER>(u, d, type); break;
    default: errorQuda("Gauge field %d order not supported", u.Order());
    }
    return norm_;
  }

  template <int...> struct IntList { };

  template <typename T, bool fixed, int nColor, int...N>
  double norm(const GaugeField &u, int d, norm_type_ type, IntList<nColor, N...>)
  {
    double norm_ = 0.0;
    if (u.Ncolor() == 3 && u.Ncolor() == nColor) {
      norm_ = norm<T, fixed, nColor>(u, d, type);
    } else if (u.Ncolor() > 3 && u.Ncolor() / 2 == nColor) {
      norm_ = norm<T, fixed, 2 * nColor>(u, d, type); // factor of two to account for spin with MG fields
    } else {
      if constexpr (sizeof...(N) > 0) {
        norm_ = norm<T, fixed>(u, d, type, IntList<N...>());
      } else {
        errorQuda("Nc = %d has not been instantiated", u.Ncolor());
      }
    }
    return norm_;
  }

  template <typename T> struct Norm {
    Norm(const GaugeField &u, double &nrm, int d, bool fixed, norm_type_ type)
    {
      if (fixed && u.Precision() > QUDA_SINGLE_PRECISION)
        errorQuda("Fixed point override only enabled for 8-bit, 16-bit and 32-bit fields");

      if (fixed) nrm = norm<T,  true>(u, d, type, IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@>());
      else       nrm = norm<T, false>(u, d, type, IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@>());
    }
  };

  double GaugeField::norm1(int d, bool fixed) const {
    if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct=%d", reconstruct);
    double nrm = 0.0;
    instantiatePrecision<Norm>(*this, nrm, d, fixed, NORM1);
    return nrm;
  }

  double GaugeField::norm2(int d, bool fixed) const {
    if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct=%d", reconstruct);
    double nrm = 0.0;
    instantiatePrecision<Norm>(*this, nrm, d, fixed, NORM2);
    return nrm;
  }

  double GaugeField::abs_max(int d, bool fixed) const {
    if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct=%d", reconstruct);
    double nrm = 0.0;
    instantiatePrecision<Norm>(*this, nrm, d, fixed, ABS_MAX);
    return nrm;
  }

  double GaugeField::abs_min(int d, bool fixed) const {
    if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct=%d", reconstruct);
    double nrm = std::numeric_limits<double>::infinity();
    instantiatePrecision<Norm>(*this, nrm, d, fixed, ABS_MIN);
    return nrm;
  }

  template <class Order> void print_matrix(const Order &o, int d, int parity, unsigned int x_cb)
  {
    for (int r = 0; r < o.Ncolor(); r++) {
      printf("rank %d parity %d x %u row %d", comm_rank(), parity, x_cb, r);
      for (int c = 0; c < o.Ncolor(); c++) {
        auto value = complex<double>(o(d, parity, x_cb, r, c));
        printf(" (%g,%g)", value.real(), value.imag());
      }
      printf("\n");
    }
  }

  template <typename Float, int nColor>
  void genericPrintMatrix(const GaugeField &a, int d, int parity, unsigned int x_cb)
  {
    switch (a.FieldOrder()) {
    case QUDA_FLOAT2_GAUGE_ORDER:
      print_matrix(FieldOrder<double, nColor, 1, QUDA_FLOAT2_GAUGE_ORDER, true, Float>(a), d, parity, x_cb);
      break;
    case QUDA_QDP_GAUGE_ORDER:
      print_matrix(FieldOrder<double, nColor, 1, QUDA_QDP_GAUGE_ORDER, true, Float>(a), d, parity, x_cb);
      break;
    case QUDA_MILC_GAUGE_ORDER:
      print_matrix(FieldOrder<double, nColor, 1, QUDA_MILC_GAUGE_ORDER, true, Float>(a), d, parity, x_cb);
      break;
    default: errorQuda("Unsupported field order %d", a.FieldOrder());
    }
  }

  template <typename Float, int nColor, int... N>
  void genericPrintMatrix(const GaugeField &a, int d, int parity, unsigned int x_cb, IntList<nColor, N...>)
  {
    if (a.Ncolor() == nColor) {
      genericPrintMatrix<Float, nColor>(a, d, parity, x_cb);
    } else {
      if constexpr (sizeof...(N) > 0) {
        genericPrintMatrix<Float, N...>(a, d, parity, x_cb, IntList<N...>());
      } else {
        errorQuda("Not supported Ncolor = %d", a.Ncolor());
      }
    }
  }

  template <typename Float> void genericPrintMatrix(const GaugeField &a, int d, int parity, unsigned int x_cb)
  {
    genericPrintMatrix<Float>(a, d, parity, x_cb, IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@>());
  }

  void genericPrintMatrix(const GaugeField &a, int d, int parity, unsigned int x_cb, int rank)
  {
    GaugeFieldParam param(a);
    param.field = const_cast<GaugeField *>(&a);
    param.location = QUDA_CPU_FIELD_LOCATION;
    param.create = QUDA_COPY_FIELD_CREATE;
    // if field is a pinned device field then we need to clone it on the host
    bool host_clone
      = (a.Location() == QUDA_CUDA_FIELD_LOCATION && a.MemType() == QUDA_MEMORY_DEVICE && !use_managed_memory()) ? true :
                                                                                                                   false;
    std::unique_ptr<GaugeField> clone_a = !host_clone ? nullptr : std::make_unique<GaugeField>(param);
    const GaugeField &a_ = !host_clone ? a : *clone_a.get();

    if (rank != comm_rank()) return; // rank returns after potential copy to host to prevent tuning hang

    switch (a.Precision()) {
    case QUDA_DOUBLE_PRECISION: genericPrintMatrix<double>(a_, d, parity, x_cb); break;
    case QUDA_SINGLE_PRECISION: genericPrintMatrix<float>(a_, d, parity, x_cb); break;
    case QUDA_HALF_PRECISION: genericPrintMatrix<short>(a_, d, parity, x_cb); break;
    case QUDA_QUARTER_PRECISION: genericPrintMatrix<int8_t>(a_, d, parity, x_cb); break;
    default: errorQuda("Precision %d not implemented", a.Precision());
    }
  }

} // namespace quda
