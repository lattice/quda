#include <clover_field_order.h>

namespace quda {

  using namespace clover;

  enum norm_type_ {
    NORM1,
    NORM2,
    ABS_MAX,
    ABS_MIN
  };

  template<typename real, int Nc, QudaCloverFieldOrder order>
  double norm(const CloverField &u, bool inverse, norm_type_ type) {
    constexpr int Ns = 4;
    typedef typename mapper<real>::type reg_type;
    double norm_ = 0.0;
    switch(type) {
    case   NORM1: norm_ = FieldOrder<reg_type,Nc,Ns,order>(const_cast<CloverField &>(u), inverse).norm1();   break;
    case   NORM2: norm_ = FieldOrder<reg_type,Nc,Ns,order>(const_cast<CloverField &>(u), inverse).norm2();   break;
    case ABS_MAX: norm_ = FieldOrder<reg_type,Nc,Ns,order>(const_cast<CloverField &>(u), inverse).abs_max(); break;
    case ABS_MIN: norm_ = FieldOrder<reg_type,Nc,Ns,order>(const_cast<CloverField &>(u), inverse).abs_min(); break;
    }
    return norm_;
  }

  template<typename real, int Nc>
  double norm(const CloverField &u, bool inverse, norm_type_ type) {
    double norm_ = 0.0;
    switch (u.Order()) {
    case QUDA_FLOAT2_CLOVER_ORDER: norm_ = norm<real,Nc,QUDA_FLOAT2_CLOVER_ORDER>(u, inverse, type); break;
    case QUDA_FLOAT4_CLOVER_ORDER: norm_ = norm<real,Nc,QUDA_FLOAT4_CLOVER_ORDER>(u, inverse, type); break;
    case QUDA_PACKED_CLOVER_ORDER: norm_ = norm<real,Nc,QUDA_PACKED_CLOVER_ORDER>(u, inverse, type); break;
    default: errorQuda("Clover field %d order not supported", u.Order());
    }
    return norm_;
  }

  template<typename real>
  double _norm(const CloverField &u, bool inverse, norm_type_ type) {
    double norm_ = 0.0;
    switch(u.Ncolor()) {
    case  3: norm_ = norm<real, 3>(u, inverse, type); break;
    default: errorQuda("Unsupported color %d", u.Ncolor());
    }
    return norm_;
  }

#ifdef GPU_CLOVER_DIRAC
  double _norm(const CloverField &u, bool inverse, norm_type_ type)
  {
    if (!u.V(inverse)) errorQuda("reqeusted clover is_inverse=%d, but not allocated", inverse);
    double nrm = 0.0;
    switch(u.Precision()) {
    case QUDA_DOUBLE_PRECISION: nrm = _norm<double>(u, inverse, type); break;
    case QUDA_SINGLE_PRECISION: nrm = _norm< float>(u, inverse, type); break;
    default: errorQuda("Unsupported precision %d", u.Precision());
    }
    return nrm;
  }
#else
  double _norm(const CloverField &, bool, norm_type_)
  {
    errorQuda("Clover dslash has not been built");
    return 0.0;;
  }
#endif

  double CloverField::norm1(bool inverse) const {
    return _norm(*this, inverse, NORM1);
  }

  double CloverField::norm2(bool inverse) const {
    return _norm(*this, inverse, NORM2);
  }

  double CloverField::abs_max(bool inverse) const {
    return _norm(*this, inverse, ABS_MAX);
  }

  double CloverField::abs_min(bool inverse) const {
    return _norm(*this, inverse, ABS_MIN);
  }

} // namespace quda
