#include <gauge_field_order.h>

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

  template <typename reg_type, typename real, int Nc>
  double norm(const GaugeField &u, int d, norm_type_ type) {
    double norm_ = 0.0;
    switch (u.FieldOrder()) {
    case QUDA_FLOAT2_GAUGE_ORDER: norm_ = norm<reg_type, real,Nc,QUDA_FLOAT2_GAUGE_ORDER>(u, d, type); break;
    case QUDA_QDP_GAUGE_ORDER:    norm_ = norm<reg_type, real,Nc,   QUDA_QDP_GAUGE_ORDER>(u, d, type); break;
    case QUDA_MILC_GAUGE_ORDER:   norm_ = norm<reg_type, real,Nc,  QUDA_MILC_GAUGE_ORDER>(u, d, type); break;
    default: errorQuda("Gauge field %d order not supported", u.Order());
    }
    return norm_;
  }

  template <typename reg_type, typename real>
  double norm(const GaugeField &u, int d, norm_type_ type) {
    double norm_ = 0.0;
    switch(u.Ncolor()) {
    case  3: norm_ = norm<reg_type, real, 3>(u, d, type); break;
#ifdef GPU_MULTIGRID
    case 48: norm_ = norm<reg_type, real, 48>(u, d, type); break;
#ifdef NSPIN4
    case 12: norm_ = norm<reg_type, real, 12>(u, d, type); break;
    case 64: norm_ = norm<reg_type, real, 64>(u, d, type); break;
#endif // NSPIN4
#ifdef NSPIN1
    case 128: norm_ = norm<reg_type, real, 128>(u, d, type); break;
    case 192: norm_ = norm<reg_type, real, 192>(u, d, type); break;
#endif // NSPIN1
#endif // GPU_MULTIGRID
    default: errorQuda("Unsupported color %d", u.Ncolor());
    }
    return norm_;
  }

  double norm(const GaugeField &u, int d, bool fixed, norm_type_ type) {
    if (fixed && u.Precision() > QUDA_SINGLE_PRECISION)
      errorQuda("Fixed point override only enabled for 8-bit, 16-bit and 32-bit fields");

    double nrm = 0.0;
    switch(u.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      nrm = norm<typename mapper<double>::type, double>(u, d, type); break;
    case QUDA_SINGLE_PRECISION:
#if QUDA_PRECISION & 4
      nrm = fixed ? norm<float, int>(u, d, type) : norm<typename mapper<float>::type, float>(u, d, type); break;
#else
      errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
    case QUDA_HALF_PRECISION:
#if QUDA_PRECISION & 2
      nrm = norm<typename mapper<short>::type, short>(u, d, type); break;
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    case QUDA_QUARTER_PRECISION:
#if QUDA_PRECISION & 1
      nrm = norm<typename mapper<char>::type, char>(u, d, type); break;
#else
      errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
    default: errorQuda("Unsupported precision %d", u.Precision());
    }
    return nrm;
  }

  double GaugeField::norm1(int d, bool fixed) const {
    if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct=%d", reconstruct);
    return norm(*this, d, fixed, NORM1);
  }

  double GaugeField::norm2(int d, bool fixed) const {
    if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct=%d", reconstruct);
    return norm(*this, d, fixed, NORM2);
  }

  double GaugeField::abs_max(int d, bool fixed) const {
    if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct=%d", reconstruct);
    return norm(*this, d, fixed, ABS_MAX);
  }

  double GaugeField::abs_min(int d, bool fixed) const {
    if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct=%d", reconstruct);
    return norm(*this, d, fixed, ABS_MIN);
  }

} // namespace quda
