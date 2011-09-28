#ifndef _GAUGE_QUDA_H
#define _GAUGE_QUDA_H

#include <quda_internal.h>
#include <quda.h>

class cpuGaugeField;
class cudaGaugeField;

class GaugeField {

 protected:
  size_t bytes; // bytes allocated per clover full field 
  size_t norm_bytes; // sizeof each norm full field
  size_t total_bytes; // total bytes allocated
  QudaPrecision precision;
  int length;
  int real_length;
  int volume;
  int volumeCB;
  int X[QUDA_MAX_DIM];
  int Nc;
  int Ns;
  int pad;
  int stride;

 public:
  GaugeField();
  virtual ~GaugeField();

};

class cudaGaugeField : public GaugeField {

 public:
  cudaGaugeField();
  virtual ~cudaGaugeField();

  void loadCPUGaugeField(const cpuGaugeField &);
  void saveCPUGaugeField(cpuGaugeField &) const;

};

class cpuGaugeField : public GaugeField {

 public:
  cpuGaugeField();
  virtual ~cpuGaugeField();

};

#ifdef __cplusplus
extern "C" {
#endif

  void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, QudaPrecision cuda_prec, QudaPrecision cpu_prec,
			GaugeFieldOrder gauge_order, ReconstructType reconstruct, GaugeFixed gauge_fixed,
			Tboundary t_boundary, int *XX, double anisotropy, double tadpole_coeff, int pad, 
			QudaLinkType type);

  void restoreGaugeField(void *cpuGauge, FullGauge *cudaGauge, QudaPrecision cpu_prec, GaugeFieldOrder gauge_order);

  void freeGaugeField(FullGauge *cudaGauge);
  
#define gaugeSiteSize 18 // real numbers per gauge field
  
#ifdef __cplusplus
}
#endif

#endif // _GAUGE_QUDA_H
