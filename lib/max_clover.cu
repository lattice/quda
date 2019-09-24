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
  double norm(const CloverField &u, norm_type_ type) {
    constexpr int Ns = 4;
    typedef typename mapper<real>::type reg_type;
    double norm_ = 0.0;
    switch(type) {
    case   NORM1: norm_ = FieldOrder<reg_type,Nc,Ns,order>(const_cast<CloverField &>(u)).norm1();   break;
    case   NORM2: norm_ = FieldOrder<reg_type,Nc,Ns,order>(const_cast<CloverField &>(u)).norm2();   break;
    case ABS_MAX: norm_ = FieldOrder<reg_type,Nc,Ns,order>(const_cast<CloverField &>(u)).abs_max(); break;
    case ABS_MIN: norm_ = FieldOrder<reg_type,Nc,Ns,order>(const_cast<CloverField &>(u)).abs_min(); break;
    }
    return norm_;
  }

  template<typename real, int Nc>
  double norm(const CloverField &u, norm_type_ type) {
    double norm_ = 0.0;
    switch (u.Order()) {
    case QUDA_FLOAT2_CLOVER_ORDER: norm_ = norm<real,Nc,QUDA_FLOAT2_CLOVER_ORDER>(u, type); break;
    case QUDA_FLOAT4_CLOVER_ORDER: norm_ = norm<real,Nc,QUDA_FLOAT4_CLOVER_ORDER>(u, type); break;
    default: errorQuda("Clover field %d order not supported", u.Order());
    }
    return norm_;
  }

  template<typename real>
  double _norm(const CloverField &u, norm_type_ type) {
    double norm_ = 0.0;
    switch(u.Ncolor()) {
    case  3: norm_ = norm<real, 3>(u, type); break;
    default: errorQuda("Unsupported color %d", u.Ncolor());
    }
    return norm_;
  }

  double _norm(const CloverField &u, norm_type_ type)
  {
    double nrm = 0.0;
#ifdef GPU_CLOVER_DIRAC
    switch(u.Precision()) {
    case QUDA_DOUBLE_PRECISION: nrm = _norm<double>(u, type); break;
    case QUDA_SINGLE_PRECISION: nrm = _norm< float>(u, type); break;
    default: errorQuda("Unsupported precision %d", u.Precision());
    }
#else
    errorQuda("Clover dslash has not been built");
#endif
    return nrm;
  }

  double CloverField::norm1() const {
    return _norm(*this, NORM1);
  }

  double CloverField::norm2() const {
    return _norm(*this, NORM2);
  }

  double CloverField::abs_max() const {
    return _norm(*this, ABS_MAX);
  }

  double CloverField::abs_min() const {
    return _norm(*this, ABS_MIN);
  }

} // namespace quda
