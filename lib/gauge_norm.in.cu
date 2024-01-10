#include <gauge_field_order.h>
#include <instantiate.h>

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

} // namespace quda
