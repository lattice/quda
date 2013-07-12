#include <gauge_field_order.h>

namespace quda {

  /**
     Generic CPU function find the gauge maximum
  */
  template <typename Float, int Nc, typename Order>
    double maxGauge(const Order order, int volume, int nDim) {  
    typedef typename mapper<Float>::type RegType;
    RegType max = 0.0;

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<nDim; d++) {
	for (int x=0; x<volume/2; x++) {
	  RegType v[Nc*Nc*2];
	  order.load(v, x, d, parity);
	  for (int i=0; i<Nc*Nc*2; i++) if (abs(v[i]) > max) { max = v[i]; }
	}
      }

    }

    return max;
  }

  template <typename Float>
    double maxGauge(const GaugeField &u) {

    if (typeid(u) != typeid(cpuGaugeField)) errorQuda("Field type not supported");

    if (u.Ncolor() != 3) errorQuda("Unsupported number of colors; Nc=%d", u.Ncolor());
    const int Nc = 3;

    double max;
    // max only supported on external fields currently
    if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
      max = maxGauge<Float,Nc>(QDPOrder<Float,2*Nc*Nc>(u, (Float*)u.Gauge_p()),u.Volume(),4);
    } else if (u.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {
      max = maxGauge<Float,Nc>(CPSOrder<Float,2*Nc*Nc>(u, (Float*)u.Gauge_p()),u.Volume(),4);
    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {
      max = maxGauge<Float,Nc>(MILCOrder<Float,2*Nc*Nc>(u, (Float*)u.Gauge_p()),u.Volume(),4);
    } else if (u.Order() == QUDA_BQCD_GAUGE_ORDER) {
      max = maxGauge<Float,Nc>(BQCDOrder<Float,2*Nc*Nc>(u, (Float*)u.Gauge_p()),u.Volume(),4);
    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }

    reduceMaxDouble(max);
    return max;
  }

  double maxGauge(const GaugeField &u) {
    double max = 0;
    if (u.Precision() == QUDA_DOUBLE_PRECISION) {
      max = maxGauge<double>(u);
    } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
      max = maxGauge<float>(u);
    } else {
      errorQuda("Precision %d undefined", u.Precision());
    }
    return max;
  }

} // namespace quda
