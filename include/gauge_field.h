#ifndef _GAUGE_QUDA_H
#define _GAUGE_QUDA_H

#include <quda_internal.h>
#include <quda.h>
#include <lattice_field.h>

namespace quda {

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
    double fat_link_max;
    
    QudaFieldCreate create; // used to determine the type of field created

    bool ghostExchange; // whether we have exchanged the ghost or not

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
    QudaGaugeFieldOrder FieldOrder() const { return order; }
    QudaFieldGeometry Geometry() const { return geometry; }

    const double& LinkMax() const { return fat_link_max; }
    int Nface() const { return nFace; }
  
    void checkField(const GaugeField &);

    const size_t& Bytes() const { return bytes; }

    virtual void* Gauge_p() { errorQuda("Not implemented"); return (void*)0;}
    virtual void* Even_p() { errorQuda("Not implemented"); return (void*)0;}
    virtual void* Odd_p() { errorQuda("Not implemented"); return (void*)0;}

    virtual const void* Gauge_p() const { errorQuda("Not implemented"); return (void*)0;}
    virtual const void* Even_p() const { errorQuda("Not implemented"); return (void*)0;}
    virtual const void* Odd_p() const { errorQuda("Not implemented"); return (void*)0;}

    virtual const void** Ghost() const { errorQuda("Not implemented"); return (const void**)0; }
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

#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t evenTex;
    cudaTextureObject_t oddTex;
    void createTexObject(cudaTextureObject_t &tex, void *gauge);
    void destroyTexObject();
#endif

  public:
    cudaGaugeField(const GaugeFieldParam &);
    virtual ~cudaGaugeField();

    void exchangeGhost(); // exchange the ghost and store store in the padded region

    void copy(const GaugeField &);     // generic gauge field copy
    void loadCPUField(const cpuGaugeField &, const QudaFieldLocation &);
    void saveCPUField(cpuGaugeField &, const QudaFieldLocation &) const;

    // (ab)use with care
    void* Gauge_p() { return gauge; }
    void* Even_p() { return even; }
    void* Odd_p() { return odd; }

    const void* Gauge_p() const { return gauge; }
    const void* Even_p() const { return even; }
    const void* Odd_p() const { return odd; }	

#ifdef USE_TEXTURE_OBJECTS
    const cudaTextureObject_t& EvenTex() const { return evenTex; }
    const cudaTextureObject_t& OddTex() const { return oddTex; }
#endif

    mutable char *backup_h;
    mutable bool backed_up;
    // backs up the cudaGaugeField to CPU memory
    void backup() const;
    // restores the cudaGaugeField to CUDA memory
    void restore();

    void setGauge(void* _gauge); //only allowed when create== QUDA_REFERENCE_FIELD_CREATE
  };

  class cpuGaugeField : public GaugeField {

    friend void cudaGaugeField::copy(const GaugeField &cpu);
    friend void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu, const QudaFieldLocation &);
    friend void cudaGaugeField::saveCPUField(cpuGaugeField &cpu, const QudaFieldLocation &) const;

  private:
    void **gauge; // the actual gauge field
    mutable void **ghost; // stores the ghost zone of the gauge field
    int pinned;
  
  public:
    cpuGaugeField(const GaugeFieldParam &);
    virtual ~cpuGaugeField();

    void exchangeGhost();
    const void** Ghost() const { return (const void**)ghost; }

    void* Gauge_p() { return gauge; }
    const void* Gauge_p() const { return gauge; }
    void setGauge(void** _gauge); //only allowed when create== QUDA_REFERENCE_FIELD_CREATE
  };

  /**
     This is a debugging function, where we cast a gauge field into a
     spinor field so we can compute its L2 norm.
     @param u The gauge field that we want the norm of
     @return The L2 norm squared of the gauge field
  */
  double norm2(const cudaGaugeField &u);

  /**
     This function is used for  extracting the gauge ghost zone from a
     gauge field array.  Defined in copy_gauge.cu.
     @param out The output field to which we are copying
     @param in The input field from which we are copying
     @param location The location of where we are doing the copying (CPU or CUDA)
     @param Out The output buffer (optional)
     @param In The input buffer (optional)
     @param ghostOut The output ghost buffer (optional)
     @param ghostIn The input ghost buffer (optional)
     @param type The type of copy we doing (0 body and ghost else ghost only)
  */
  // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location, 
			void *Out=0, void *In=0, void **ghostOut=0, void **ghostIn=0, int type=0);
  /**
     This function is used for  extracting the gauge ghost zone from a
     gauge field array.  Defined in extract_gauge_ghost.cu.
     @param u The gauge field from which we want to extract the ghost zone
     @param ghost The array where we want to pack the ghost zone into
  */
  void extractGaugeGhost(const GaugeField &u, void **ghost);

  /**
     This function is used to calculate the maximum absolute value of
     a gauge field array.  Defined in max_gauge.cu.  

     @param u The gauge field from which we want to compute the max
  */
  double maxGauge(const GaugeField &u);

} // namespace quda


//FIXME remove this legacy macro
#define gaugeSiteSize 18 // real numbers per gauge field
  
#endif // _GAUGE_QUDA_H
