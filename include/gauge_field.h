#ifndef _GAUGE_QUDA_H
#define _GAUGE_QUDA_H

#include <quda_internal.h>
#include <quda.h>
#include <lattice_field.h>

struct GaugeFieldParam : public LatticeFieldParam {
  int nColor;
  int nFace;

  QudaReconstructType reconstruct;
  QudaGaugeFieldOrder order;
  QudaGaugeFixed fixed;
  QudaLinkType link_type;
  QudaTboundary t_boundary;

  double anisotropy;
  double tadpole;

  void *gauge; // used when we use a reference to an external field

  QudaFieldCreate create; // used to determine the type of field created

 GaugeFieldParam(void *h_gauge, const QudaGaugeParam &param) : LatticeFieldParam(param),
    nColor(3), nFace(0), reconstruct(QUDA_RECONSTRUCT_NO),
    order(param.gauge_order), fixed(param.gauge_fix), link_type(param.type), 
    t_boundary(param.t_boundary), anisotropy(param.anisotropy), tadpole(param.tadpole_coeff),
    gauge(h_gauge), create(QUDA_REFERENCE_FIELD_CREATE) {

    if (link_type == QUDA_WILSON_LINKS || link_type == QUDA_ASQTAD_FAT_LINKS) nFace = 1;
    else if (link_type == QUDA_ASQTAD_LONG_LINKS) nFace = 3;
  }
};

std::ostream& operator<<(std::ostream& output, const GaugeFieldParam& param);

class GaugeField : public LatticeField {

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
  double tadpole;

  QudaFieldCreate create; // used to determine the type of field created

 public:
  GaugeField(const GaugeFieldParam &param, const QudaFieldLocation &location);
  virtual ~GaugeField();

  int Ncolor() const { return nColor; }
  QudaReconstructType Reconstruct() const { return reconstruct; }
  QudaGaugeFieldOrder Order() const { return order; }
  double Anisotropy() const { return anisotropy; }
  double Tadpole() const { return tadpole; }
  QudaTboundary TBoundary() const { return t_boundary; }
  QudaLinkType LinkType() const { return link_type; }
  QudaGaugeFixed GaugeFixed() const { return fixed; }

  virtual void checkField(const GaugeField &);
};

class cudaGaugeField : public GaugeField {

  friend void bindGaugeTex(const cudaGaugeField &gauge, const int oddBit, 
			   void **gauge0, void **gauge1);
  friend void unbindGaugeTex(const cudaGaugeField &gauge);
  friend void bindFatGaugeTex(const cudaGaugeField &gauge, const int oddBit, 
			      void **gauge0, void **gauge1);
  friend void unbindFatGaugeTex(const cudaGaugeField &gauge);
  friend void bindLongGaugeTex(const cudaGaugeField &gauge, const int oddBit, 
			       void **gauge0, void **gauge1);
  friend void unbindLongGaugeTex(const cudaGaugeField &gauge);

 private:
  void *gauge;
  void *even;
  void *odd;

  double fat_link_max;

 public:
  cudaGaugeField(const GaugeFieldParam &);
  virtual ~cudaGaugeField();

  void loadCPUField(const cpuGaugeField &);
  void saveCPUField(cpuGaugeField &) const;

  double LinkMax() const { return fat_link_max; }
};

class cpuGaugeField : public GaugeField {

  friend void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu);
  friend void cudaGaugeField::saveCPUField(cpuGaugeField &cpu) const;

 private:
  void *gauge[QUDA_MAX_DIM]; // the actual gauge field
  mutable void *ghost[QUDA_MAX_DIM]; // stores the ghost zone of the gauge field

 public:
  cpuGaugeField(const GaugeFieldParam &);
  virtual ~cpuGaugeField();

  void exchangeGhost() const;
  const void** Ghost() const { return (const void**)ghost; }
};

#define gaugeSiteSize 18 // real numbers per gauge field
  
#endif // _GAUGE_QUDA_H
