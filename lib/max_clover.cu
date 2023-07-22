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
  real_t norm(const CloverField &u, bool inverse, norm_type_ type) {
    constexpr int Ns = 4;
    typedef typename mapper<real>::type reg_type;
    real_t norm_ = 0.0;
    switch(type) {
    case   NORM1: norm_ = FieldOrder<reg_type,Nc,Ns,order>(const_cast<CloverField &>(u), inverse).norm1();   break;
    case   NORM2: norm_ = FieldOrder<reg_type,Nc,Ns,order>(const_cast<CloverField &>(u), inverse).norm2();   break;
    case ABS_MAX: norm_ = FieldOrder<reg_type,Nc,Ns,order>(const_cast<CloverField &>(u), inverse).abs_max(); break;
    case ABS_MIN: norm_ = FieldOrder<reg_type,Nc,Ns,order>(const_cast<CloverField &>(u), inverse).abs_min(); break;
    }
    return norm_;
  }

  template<typename real, int Nc>
  real_t norm(const CloverField &u, bool inverse, norm_type_ type) {
    real_t norm_ = 0.0;
    switch (u.Order()) {
    case QUDA_FLOAT2_CLOVER_ORDER: norm_ = norm<real,Nc,QUDA_FLOAT2_CLOVER_ORDER>(u, inverse, type); break;
    case QUDA_FLOAT4_CLOVER_ORDER: norm_ = norm<real,Nc,QUDA_FLOAT4_CLOVER_ORDER>(u, inverse, type); break;
    case QUDA_PACKED_CLOVER_ORDER: norm_ = norm<real,Nc,QUDA_PACKED_CLOVER_ORDER>(u, inverse, type); break;
    default: errorQuda("Clover field %d order not supported", u.Order());
    }
    return norm_;
  }

  template<typename real>
  real_t _norm(const CloverField &u, bool inverse, norm_type_ type) {
    real_t norm_ = 0.0;
    switch(u.Ncolor()) {
    case  3: norm_ = norm<real, 3>(u, inverse, type); break;
    default: errorQuda("Unsupported color %d", u.Ncolor());
    }
    return norm_;
  }

#ifdef GPU_CLOVER_DIRAC
  real_t _norm(const CloverField &u, bool inverse, norm_type_ type)
  {
    if (!u.data(inverse)) errorQuda("reqeusted clover is_inverse=%d, but not allocated", inverse);
    real_t nrm = 0.0;
    switch(u.Precision()) {
    case QUDA_DOUBLE_PRECISION: nrm = _norm<double>(u, inverse, type); break;
    case QUDA_SINGLE_PRECISION: nrm = _norm< float>(u, inverse, type); break;
    default: errorQuda("Unsupported precision %d", u.Precision());
    }
    return nrm;
  }
#else
  real_t _norm(const CloverField &, bool, norm_type_)
  {
    errorQuda("Clover dslash has not been built");
    return 0.0;;
  }
#endif

  real_t CloverField::norm1(bool inverse) const {
    return _norm(*this, inverse, NORM1);
  }

  real_t CloverField::norm2(bool inverse) const {
    return _norm(*this, inverse, NORM2);
  }

  real_t CloverField::abs_max(bool inverse) const {
    return _norm(*this, inverse, ABS_MAX);
  }

  real_t CloverField::abs_min(bool inverse) const {
    return _norm(*this, inverse, ABS_MIN);
  }

} // namespace quda
