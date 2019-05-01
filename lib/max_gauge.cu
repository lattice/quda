#include <gauge_field_order.h>

namespace quda {

  using namespace gauge;

  enum norm_type_ {
    NORM1,
    NORM2,
    ABS_MAX,
    ABS_MIN
  };

  template<typename real, int Nc, QudaGaugeFieldOrder order>
  double norm(const GaugeField &u, int d, norm_type_ type) {
    typedef typename mapper<real>::type reg_type;
    double norm_ = 0.0;
    switch(type) {
    case   NORM1: norm_ = FieldOrder<reg_type,Nc,1,order,true,real>(const_cast<GaugeField &>(u)).norm1(d);   break;
    case   NORM2: norm_ = FieldOrder<reg_type,Nc,1,order,true,real>(const_cast<GaugeField &>(u)).norm2(d);   break;
    case ABS_MAX: norm_ = FieldOrder<reg_type,Nc,1,order,true,real>(const_cast<GaugeField &>(u)).abs_max(d); break;
    case ABS_MIN: norm_ = FieldOrder<reg_type,Nc,1,order,true,real>(const_cast<GaugeField &>(u)).abs_min(d); break;
    }
    return norm_;
  }

  template<typename real, int Nc>
  double norm(const GaugeField &u, int d, norm_type_ type) {
    double norm_ = 0.0;
    switch (u.FieldOrder()) {
    case QUDA_FLOAT2_GAUGE_ORDER: norm_ = norm<real,Nc,QUDA_FLOAT2_GAUGE_ORDER>(u, d, type); break;
    case QUDA_QDP_GAUGE_ORDER:    norm_ = norm<real,Nc,   QUDA_QDP_GAUGE_ORDER>(u, d, type); break;
    case QUDA_MILC_GAUGE_ORDER:   norm_ = norm<real,Nc,  QUDA_MILC_GAUGE_ORDER>(u, d, type); break;
    default: errorQuda("Gauge field %d order not supported", u.Order());
    }
    return norm_;
  }

  template<typename real>
  double norm(const GaugeField &u, int d, norm_type_ type) {
    double norm_ = 0.0;
    switch(u.Ncolor()) {
    case  3: norm_ = norm<real, 3>(u, d, type); break;
    case  8: norm_ = norm<real, 8>(u, d, type); break;
    case 12: norm_ = norm<real,12>(u, d, type); break;
    case 16: norm_ = norm<real,16>(u, d, type); break;
    case 24: norm_ = norm<real,24>(u, d, type); break;
    case 32: norm_ = norm<real,32>(u, d, type); break;
    case 40: norm_ = norm<real,40>(u, d, type); break;
    case 48: norm_ = norm<real,48>(u, d, type); break;
    case 56: norm_ = norm<real,56>(u, d, type); break;
    case 64: norm_ = norm<real,64>(u, d, type); break;
    default: errorQuda("Unsupported color %d", u.Ncolor());
    }
    return norm_;
  }

  double GaugeField::norm1(int d) const {
    if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct=%d", reconstruct);
    double nrm1 = 0.0;
    switch(precision) {
    case QUDA_DOUBLE_PRECISION: nrm1 = norm<double>(*this, d, NORM1); break;
    case QUDA_SINGLE_PRECISION: nrm1 = norm<float>(*this, d, NORM1); break;
    case QUDA_HALF_PRECISION: nrm1 = norm<short>(*this, d, NORM1); break;
    case QUDA_QUARTER_PRECISION: nrm1 = norm<char>(*this, d, NORM1); break;
    default: errorQuda("Unsupported precision %d", precision);
    }
    return nrm1;
  }

  double GaugeField::norm2(int d) const {
    if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct=%d", reconstruct);
    double nrm2 = 0.0;
    switch(precision) {
    case QUDA_DOUBLE_PRECISION: nrm2 = norm<double>(*this, d, NORM2); break;
    case QUDA_SINGLE_PRECISION: nrm2 = norm<float>(*this, d, NORM2); break;
    case QUDA_HALF_PRECISION: nrm2 = norm<short>(*this, d, NORM2); break;
    case QUDA_QUARTER_PRECISION: nrm2 = norm<char>(*this, d, NORM2); break;
    default: errorQuda("Unsupported precision %d", precision);
    }
    return nrm2;
  }

  double GaugeField::abs_max(int d) const {
    if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct=%d", reconstruct);
    double max = 0.0;
    switch(precision) {
    case QUDA_DOUBLE_PRECISION: max = norm<double>(*this, d, ABS_MAX); break;
    case QUDA_SINGLE_PRECISION: max = norm<float>(*this, d, ABS_MAX); break;
    case QUDA_HALF_PRECISION: max = norm<short>(*this, d, ABS_MAX); break;
    case QUDA_QUARTER_PRECISION: max = norm<char>(*this, d, ABS_MAX); break;
    default: errorQuda("Unsupported precision %d", precision);
    }
    return max;
  }

  double GaugeField::abs_min(int d) const {
    if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct=%d", reconstruct);
    double min = 0.0;
    switch(precision) {
    case QUDA_DOUBLE_PRECISION: min = norm<double>(*this, d, ABS_MIN); break;
    case QUDA_SINGLE_PRECISION: min = norm<float>(*this, d, ABS_MIN); break;
    case QUDA_HALF_PRECISION: min = norm<short>(*this, d, ABS_MIN); break;
    case QUDA_QUARTER_PRECISION: min = norm<char>(*this, d, ABS_MIN); break;
    default: errorQuda("Unsupported precision %d", precision);
    }
    return min;
  }

} // namespace quda
