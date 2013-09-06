#ifndef _GAUGE_UPDATE_QUDA_H_
#define _GAUGE_UPDATE_QUDA_H_

namespace quda {

  /**
     Evolve the gauge field by step size dt using the momentuim field
     @param out Updated gauge field
     @param dt Step size 
     @param in Input gauge field
     @param mom Momentum field
   */
  void updateGaugeField(GaugeField &out, double dt, 
			const GaugeField& in, const GaugeField& mom);

} // namespace quda

#endif // _GAUGE_UPDATE_QUDA_H_
