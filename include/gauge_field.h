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

  QudaFieldGeometry geometry; // whether the field is a scale, vector or tensor
  int pinned; //used in cpu field only, where the host memory is pinned

  // Default constructor
  GaugeFieldParam(void* const h_gauge=NULL) : LatticeFieldParam(),
    nColor(3),
    nFace(0),
    reconstruct(QUDA_RECONSTRUCT_NO),
    order(QUDA_INVALID_GAUGE_ORDER),
    fixed(QUDA_GAUGE_FIXED_NO),
    link_type(QUDA_WILSON_LINKS),
    t_boundary(QUDA_INVALID_T_BOUNDARY),
    anisotropy(1.0),
    tadpole(1.0),
    gauge(h_gauge),
    create(QUDA_REFERENCE_FIELD_CREATE), 
    geometry(QUDA_VECTOR_GEOMETRY),
    pinned(0)
        {
	  // variables declared in LatticeFieldParam
	  precision = QUDA_INVALID_PRECISION;
	  verbosity = QUDA_SILENT;
	  nDim = 4;
	  pad  = 0;
	  for(int dir=0; dir<nDim; ++dir) x[dir] = 0;
	}
	
  GaugeFieldParam(const int *x, const QudaPrecision precision, const QudaReconstructType reconstruct,
		  const int pad, const QudaFieldGeometry geometry) : LatticeFieldParam(), nColor(3), nFace(0), 
    reconstruct(reconstruct), order(QUDA_INVALID_GAUGE_ORDER), fixed(QUDA_GAUGE_FIXED_NO), 
    link_type(QUDA_WILSON_LINKS), t_boundary(QUDA_INVALID_T_BOUNDARY), anisotropy(1.0), 
    tadpole(1.0), gauge(0), create(QUDA_NULL_FIELD_CREATE), geometry(geometry), pinned(0)
    {
      // variables declared in LatticeFieldParam
      this->precision = precision;
      this->verbosity = QUDA_SILENT;
      this->nDim = 4;
      this->pad = pad;
      for(int dir=0; dir<nDim; ++dir) this->x[dir] = x[dir];
    }
  
 GaugeFieldParam(void *h_gauge, const QudaGaugeParam &param) : LatticeFieldParam(param),
  nColor(3), nFace(0), reconstruct(QUDA_RECONSTRUCT_NO), order(param.gauge_order), 
  fixed(param.gauge_fix), link_type(param.type), t_boundary(param.t_boundary), 
  anisotropy(param.anisotropy), tadpole(param.tadpole_coeff), gauge(h_gauge), 
  create(QUDA_REFERENCE_FIELD_CREATE), geometry(QUDA_VECTOR_GEOMETRY), pinned(0) {

    if (link_type == QUDA_WILSON_LINKS || link_type == QUDA_ASQTAD_FAT_LINKS) nFace = 1;
    else if (link_type == QUDA_ASQTAD_LONG_LINKS) nFace = 3;
    else errorQuda("Error: invalid link type(%d)\n", link_type);
  }
};

std::ostream& operator<<(std::ostream& output, const GaugeFieldParam& param);

class GaugeField : public LatticeField {

 protected:
  size_t bytes; // bytes allocated per full field 
  int length;
  int real_length;
  int nColor;
  int nFace;
  QudaFieldGeometry geometry; // whether the field is a scale, vector or tensor

  QudaReconstructType reconstruct;
  QudaGaugeFieldOrder order;
  QudaGaugeFixed fixed;
  QudaLinkType link_type;
  QudaTboundary t_boundary;

  double anisotropy;
  double tadpole;

  QudaFieldCreate create; // used to determine the type of field created
  
 public:
  GaugeField(const GaugeFieldParam &param);
  virtual ~GaugeField();

  int Length() const { return length; }
  int Ncolor() const { return nColor; }
  QudaReconstructType Reconstruct() const { return reconstruct; }
  QudaGaugeFieldOrder Order() const { return order; }
  double Anisotropy() const { return anisotropy; }
  double Tadpole() const { return tadpole; }
  QudaTboundary TBoundary() const { return t_boundary; }
  QudaLinkType LinkType() const { return link_type; }
  QudaGaugeFixed GaugeFixed() const { return fixed; }

  void checkField(const GaugeField &);

  const size_t& Bytes() const { return bytes; }

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

  void loadCPUField(const cpuGaugeField &, const QudaFieldLocation &);
  void saveCPUField(cpuGaugeField &, const QudaFieldLocation &) const;

  double LinkMax() const { return fat_link_max; }

  // (ab)use with care
  void* Gauge_p() { return gauge; }
  void* Even_p() { return even; }
  void* Odd_p() { return odd; }

  const void* Gauge_p() const { return gauge; }
  const void* Even_p() const { return even; }
  const void* Odd_p() const { return odd; }	

  mutable char *backup_h;
  mutable bool backed_up;
  // backs up the cudaGaugeField to CPU memory
  void backup() const;
  // restores the cudaGaugeField to CUDA memory
  void restore();

};

class cpuGaugeField : public GaugeField {

  friend void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu, const QudaFieldLocation &);
  friend void cudaGaugeField::saveCPUField(cpuGaugeField &cpu, const QudaFieldLocation &) const;

 private:
  void **gauge; // the actual gauge field

  mutable void **ghost; // stores the ghost zone of the gauge field
  int pinned;
  
 public:
  cpuGaugeField(const GaugeFieldParam &);
  virtual ~cpuGaugeField();

  void exchangeGhost() const;
  const void** Ghost() const { return (const void**)ghost; }

  void* Gauge_p() { return gauge; }
  void setGauge(void** _gauge); //only allowed when create== QUDA_REFERENCE_FIELD_CREATE
};

#define gaugeSiteSize 18 // real numbers per gauge field
  
#endif // _GAUGE_QUDA_H
