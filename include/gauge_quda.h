#ifndef _GAUGE_QUDA_H
#define _GAUGE_QUDA_H

#include <quda_internal.h>
#include <quda.h>
#include <lattice_field.h>

struct GaugeFieldParam : public LatticeFieldParam {
  int nColor;
  int nDim;
  int nFace;

  QudaReconstructType reconstruct;
  QudaGaugeFieldOrder order;
  QudaGaugeFixed fixed;
  QudaLinkType link_type;
  QudaTboundary t_boundary;

  double anisotropy;

  void *gauge[QUDA_MAX_DIM]; // used when we use a reference to an external field

  QudaFieldCreate create; // used to determine the type of field created
};

class GaugeField : LatticeField {

 protected:
  size_t bytes; // bytes allocated per clover full field 
  int length;
  int real_length;
  int nColor;
  int nFace;

  QudaReconstructType reconstruct;
  QudaGaugeFieldOrder order;
  QudaGaugeFixed fixed;
  QudaLinkType link_type;
  QudaTboundary t_boundary;

  double anisotropy;

  QudaFieldCreate create; // used to determine the type of field created

 public:
  GaugeField(const GaugeFieldParam &param, const QudaFieldLocation &location);
  virtual ~GaugeField();

  int Ncolor() const { return nColor; }
  QudaReconstructType Reconstruct() const { return reconstruct; }
  double Anisotropy() const { return anisotropy; }
  QudaTboundary TBoundary() const { return t_boundary; }
  QudaLinkType LinkType() const { return link_type; }
  QudaGaugeFixed GaugeFixed() const { return fixed; }
};

class cudaGaugeField : public GaugeField {

 private:
  double fat_link_max;

 public:
  cudaGaugeField(const GaugeFieldParam &);
  virtual ~cudaGaugeField();

  void loadCPUField(const cpuGaugeField &);
  void saveCPUField(cpuGaugeField &) const;

};

class cpuGaugeField : public GaugeField {

 private:
  void *gauge[QUDA_MAX_DIM]; // the actual gauge field
  void *ghost[QUDA_MAX_DIM]; // stores the ghost zone of the gauge field

 public:
  cpuGaugeField(const GaugeFieldParam &);
  virtual ~cpuGaugeField();

  void exchangeGhost();
};

#ifdef __cplusplus
extern "C" {
#endif

  void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, QudaPrecision cuda_prec, QudaPrecision cpu_prec,
			GaugeFieldOrder gauge_order, QudaReconstructType reconstruct, QudaGaugeFixed gauge_fixed,
			Tboundary t_boundary, int *XX, double anisotropy, double tadpole_coeff, int pad, 
			QudaLinkType type);

  void restoreGaugeField(void *cpuGauge, FullGauge *cudaGauge, QudaPrecision cpu_prec, GaugeFieldOrder gauge_order);

  void freeGaugeField(FullGauge *cudaGauge);
  
#define gaugeSiteSize 18 // real numbers per gauge field
  
#ifdef __cplusplus
}
#endif

#endif // _GAUGE_QUDA_H
