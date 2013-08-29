#ifndef _GAUGE_UPDATE_QUDA_H_
#define _GAUGE_UPDATE_QUDA_H_

namespace quda {

  void updateGaugeFieldCuda(cudaGaugeField* const gauge, double eps, const cudaGaugeField& momentum);

} // namespace quda

#endif // _GAUGE_UPDATE_QUDA_H_
