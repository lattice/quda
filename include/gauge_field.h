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

    // whether we need to compute the fat link maxima
    // FIXME temporary flag until we have a kernel that can do this, then we just do this in copy()
    // always set to false, requires external override
    bool compute_fat_link_max; 

    /** The staggered phase convention to use */
    QudaStaggeredPhase staggeredPhaseType;

    /** Whether the staggered phase factor has been applied */
    bool staggeredPhaseApplied;

    /*** Experimental for staggered only! ***/
    bool staggered_u1_emulation;
    bool staggered_2link_term;

    /** Imaginary chemical potential */
    double i_mu;

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
      compute_fat_link_max(false),
      staggeredPhaseType(QUDA_STAGGERED_PHASE_NO),
      staggeredPhaseApplied(false),
      staggered_u1_emulation(false),
      staggered_2link_term(false),
      i_mu(0.0)
	{ }

    GaugeFieldParam(const GaugeField &u);

  GaugeFieldParam(const int *x, const QudaPrecision precision, const QudaReconstructType reconstruct,
		  const int pad, const QudaFieldGeometry geometry,
		  const QudaGhostExchange ghostExchange=QUDA_GHOST_EXCHANGE_PAD) 
    : LatticeFieldParam(4, x, pad, precision, ghostExchange), nColor(3), nFace(0), reconstruct(reconstruct),
      order(QUDA_INVALID_GAUGE_ORDER), fixed(QUDA_GAUGE_FIXED_NO),
      link_type(QUDA_WILSON_LINKS), t_boundary(QUDA_INVALID_T_BOUNDARY), anisotropy(1.0),
      tadpole(1.0), gauge(0), create(QUDA_NULL_FIELD_CREATE), geometry(geometry),
      compute_fat_link_max(false), staggeredPhaseType(QUDA_STAGGERED_PHASE_NO),
      staggeredPhaseApplied(false), staggered_u1_emulation(false), staggered_2link_term(false),  i_mu(0.0)
      { }

  GaugeFieldParam(void *h_gauge, const QudaGaugeParam &param, QudaLinkType link_type_=QUDA_INVALID_LINKS)
    : LatticeFieldParam(param), nColor(3), nFace(0), reconstruct(QUDA_RECONSTRUCT_NO),
      order(param.gauge_order), fixed(param.gauge_fix),
      link_type(link_type_ != QUDA_INVALID_LINKS ? link_type_ : param.type), t_boundary(param.t_boundary),
      anisotropy(param.anisotropy), tadpole(param.tadpole_coeff), gauge(h_gauge),
      create(QUDA_REFERENCE_FIELD_CREATE), geometry(QUDA_VECTOR_GEOMETRY),
      compute_fat_link_max(false), staggeredPhaseType(param.staggered_phase_type),
      staggeredPhaseApplied(param.staggered_phase_applied), staggered_u1_emulation(param._2d_u1_emulation), staggered_2link_term(param._2link_term), i_mu(param.i_mu)
	{
	  if (link_type == QUDA_WILSON_LINKS || link_type == QUDA_ASQTAD_FAT_LINKS) nFace = 1;
	  else if (link_type == QUDA_ASQTAD_LONG_LINKS) nFace = 3;
	  else if (link_type == QUDA_INVALID_LINKS) errorQuda("Error: invalid link type(%d)\n", link_type);
	}
    
    /**
       @brief Helper function for setting the precision and corresponding
       field order for QUDA internal fields.
       @param precision The precision to use 
     */
    void setPrecision(QudaPrecision precision) {
      this->precision = precision;
      order = (precision == QUDA_DOUBLE_PRECISION || reconstruct == QUDA_RECONSTRUCT_NO) ? 
	QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER; 
    }

  };

  std::ostream& operator<<(std::ostream& output, const GaugeFieldParam& param);

  class GaugeField : public LatticeField {

  protected:
    size_t bytes; // bytes allocated per full field 
    int phase_offset; // offset in bytes to gauge phases - useful to keep track of texture alignment
    int phase_bytes;  // bytes needed to store the phases
    int length;
    int real_length;
    int nColor;
    int nFace;
    QudaFieldGeometry geometry; // whether the field is a scale, vector or tensor

    QudaReconstructType reconstruct;
    int nInternal; // number of degrees of freedom per link matrix
    QudaGaugeFieldOrder order;
    QudaGaugeFixed fixed;
    QudaLinkType link_type;
    QudaTboundary t_boundary;

    double anisotropy;
    double tadpole;
    double fat_link_max;

    QudaFieldCreate create; // used to determine the type of field created

    mutable void *ghost[2*QUDA_MAX_DIM]; // stores the ghost zone of the gauge field (non-native fields only)

    /** The staggered phase convention to use */
    QudaStaggeredPhase staggeredPhaseType;

    /** Whether the staggered phase factor has been applied */
    bool staggeredPhaseApplied;

    /*** Experimental for staggered only! ***/
    bool staggered_u1_emulation;
    bool staggered_2link_term;
 
    /**
       @brief Exchange the buffers across all dimensions in a given direction
       @param recv[out] Reicve buffer
       @param send[in] Send buffer
       @param dir[in] Direction in which we are sending (forwards OR backwards only)
    */
    void exchange(void **recv, void **send, QudaDirection dir) const;

    /** Imaginary chemical potential */
    double i_mu;

  public:
    GaugeField(const GaugeFieldParam &param);
    virtual ~GaugeField();

    virtual void exchangeGhost(QudaLinkDirection = QUDA_LINK_BACKWARDS) = 0;
    virtual void injectGhost(QudaLinkDirection = QUDA_LINK_BACKWARDS) = 0;

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
    QudaStaggeredPhase StaggeredPhase() const { return staggeredPhaseType; }
    bool StaggeredPhaseApplied() const { return staggeredPhaseApplied; }
    /**Experimental! */
    bool StaggeredU1Emulation() const { return staggered_u1_emulation; }
    bool Staggered2LinkTerm() const { return staggered_2link_term; }


    /**
       Apply the staggered phase factors to the gauge field.
    */
    void applyStaggeredPhase();

    /**
       Remove the staggered phase factors from the gauge field.
    */
    void removeStaggeredPhase();

    /**
       Return the imaginary chemical potential applied to this field
    */
    double iMu() const { return i_mu; }

    const double& LinkMax() const { return fat_link_max; }
    int Nface() const { return nFace; }

    void checkField(const LatticeField &) const;

    /**
       This function returns true if the field is stored in an
       internal field order for the given precision.
    */ 
    bool isNative() const;

    size_t Bytes() const { return bytes; }
    size_t PhaseBytes() const { return phase_bytes; }
    size_t PhaseOffset() const { return phase_offset; }

    virtual void* Gauge_p() { errorQuda("Not implemented"); return (void*)0;}
    virtual void* Even_p() { errorQuda("Not implemented"); return (void*)0;}
    virtual void* Odd_p() { errorQuda("Not implemented"); return (void*)0;}

    virtual const void* Gauge_p() const { errorQuda("Not implemented"); return (void*)0;}
    virtual const void* Even_p() const { errorQuda("Not implemented"); return (void*)0;}
    virtual const void* Odd_p() const { errorQuda("Not implemented"); return (void*)0;}

    const void** Ghost() const {
      if ( isNative() ) errorQuda("No ghost zone pointer for quda-native gauge fields");
      return (const void**)ghost;
    }

    void** Ghost() {
      if ( isNative() ) errorQuda("No ghost zone pointer for quda-native gauge fields");
      return ghost;
    }

    /**
       Set all field elements to zero (virtual)
    */
    virtual void zero() = 0;

    /**
     * Generic gauge field copy
     * @param[in] src Source from which we are copying
     */
    virtual void copy(const GaugeField &src) = 0;

    /**
       @brief Backs up the cpuGaugeField
    */
    virtual void backup() const = 0;

    /**
       @brief Restores the cpuGaugeField
    */
    virtual void restore() = 0;
    
  };

  class cudaGaugeField : public GaugeField {

  private:
    void *gauge;
    void *even;
    void *odd;

    /**
       @brief Initialize the padded region to 0
     */
    void zeroPad();

#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t evenTex;
    cudaTextureObject_t oddTex;
    cudaTextureObject_t evenPhaseTex;
    cudaTextureObject_t oddPhaseTex;
    void createTexObject(cudaTextureObject_t &tex, void *gauge, int isPhase=0);
    void destroyTexObject();
#endif

  public:
    cudaGaugeField(const GaugeFieldParam &);
    virtual ~cudaGaugeField();

    /**
       @brief Exchange the ghost and store store in the padded region
       @param[in] link_direction Which links are we exchanging: this
       flag only applies to bi-directional coarse-link fields
     */
    void exchangeGhost(QudaLinkDirection link_direction = QUDA_LINK_BACKWARDS);

    /**
       @brief The opposite of exchangeGhost: take the ghost zone on x,
       send to node x-1, and inject back into the field
       @param[in] link_direction Which links are we injecting: this
       flag only applies to bi-directional coarse-link fields
     */
    void injectGhost(QudaLinkDirection link_direction = QUDA_LINK_BACKWARDS);

    /**
       @brief This does routine will populate the border / halo region of a
       gauge field that has been created using copyExtendedGauge.  

       @param R The thickness of the extended region in each dimension
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
    */
    void exchangeExtendedGhost(const int *R, bool no_comms_fill=false);

    /**
     * Generic gauge field copy
     * @param[in] src Source from which we are copying
     */
    void copy(const GaugeField &src);

    void loadCPUField(const cpuGaugeField &);
    void saveCPUField(cpuGaugeField &) const;

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
    const cudaTextureObject_t& EvenPhaseTex() const { return evenPhaseTex; }
    const cudaTextureObject_t& OddPhaseTex() const { return oddPhaseTex; }
#endif

    mutable char *backup_h;
    mutable bool backed_up;

    /**
       @brief Backs up the cudaGaugeField to CPU memory
    */
    void backup() const;

    /**
       @brief Restores the cudaGaugeField to CUDA memory
    */
    void restore();

    void setGauge(void* _gauge); //only allowed when create== QUDA_REFERENCE_FIELD_CREATE

    /**
       Set all field elements to zero
    */
    void zero();
  };

  class cpuGaugeField : public GaugeField {

    friend void cudaGaugeField::copy(const GaugeField &cpu);
    friend void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu);
    friend void cudaGaugeField::saveCPUField(cpuGaugeField &cpu) const;

  private:
    void **gauge; // the actual gauge field

  public:
    /**
       @brief Constructor for cpuGaugeField from a GaugeFieldParam
       @param[in,out] param Parameter struct - note that in the case
       that we are wrapping host-side extended fields, this param is
       modified for subsequent creation of fields that are not
       extended.
    */
    cpuGaugeField(const GaugeFieldParam &param);
    virtual ~cpuGaugeField();

    /**
       @brief Exchange the ghost and store store in the padded region
       @param[in] link_direction Which links are we extracting: this
       flag only applies to bi-directional coarse-link fields
     */
    void exchangeGhost(QudaLinkDirection link_direction = QUDA_LINK_BACKWARDS);

    /**
       @brief The opposite of exchangeGhost: take the ghost zone on x,
       send to node x-1, and inject back into the field
       @param[in] link_direction Which links are we injecting: this
       flag only applies to bi-directional coarse-link fields
     */
    void injectGhost(QudaLinkDirection link_direction = QUDA_LINK_BACKWARDS);

    /**
       @brief This does routine will populate the border / halo region of a
       gauge field that has been created using copyExtendedGauge.  

       @param R The thickness of the extended region in each dimension
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimenions
    */
    void exchangeExtendedGhost(const int *R, bool no_comms_fill=false);

    /**
     * Generic gauge field copy
     * @param[in] src Source from which we are copying
     */
    void copy(const GaugeField &src);

    void* Gauge_p() { return gauge; }
    const void* Gauge_p() const { return gauge; }

    mutable char *backup_h;
    mutable bool backed_up;

    /**
     @brief Backs up the cpuGaugeField
    */
    void backup() const;

    /**
       @brief Restores the cpuGaugeField
    */
    void restore();

    void setGauge(void** _gauge); //only allowed when create== QUDA_REFERENCE_FIELD_CREATE

    /**
       Set all field elements to zero
    */
    void zero();
  };

  /**
     This is a debugging function, where we cast a gauge field into a
     spinor field so we can compute its L1 norm.
     @param u The gauge field that we want the norm of
     @return The L1 norm of the gauge field
  */
  double norm1(const GaugeField &u);

  /**
     This is a debugging function, where we cast a gauge field into a
     spinor field so we can compute its L2 norm.
     @param u The gauge field that we want the norm of
     @return The L2 norm squared of the gauge field
  */
  double norm2(const GaugeField &u);

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
  void copyGenericGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location, 
			void *Out=0, void *In=0, void **ghostOut=0, void **ghostIn=0, int type=0);
  /**
     This function is used for copying the gauge field into an
     extended gauge field.  Defined in copy_extended_gauge.cu.
     @param out The extended output field to which we are copying
     @param in The input field from which we are copying
     @param location The location of where we are doing the copying (CPU or CUDA)
     @param Out The output buffer (optional)
     @param In The input buffer (optional)
  */
  void copyExtendedGauge(GaugeField &out, const GaugeField &in,
			 QudaFieldLocation location, void *Out=0, void *In=0);

  /**
     This function is used for  extracting the gauge ghost zone from a
     gauge field array.  Defined in extract_gauge_ghost.cu.
     @param u The gauge field from which we want to extract the ghost zone
     @param ghost The array where we want to pack the ghost zone into
     @param extract Where we are extracting into ghost or injecting from ghost
     @param offset By default we exchange the nDim site-vector of
     links in the first nDim dimensions; offset allows us to instead
     exchange the links in nDim+offset dimensions.  This is used to
     faciliate sending bi-directional links which is needed for the
     coarse links.
  */
  void extractGaugeGhost(const GaugeField &u, void **ghost, bool extract=true, int offset=0);

  /**
     This function is used for  extracting the gauge ghost zone from a
     gauge field array.  Defined in extract_gauge_ghost.cu.
     @param u The gauge field from which we want to extract/pack the ghost zone
     @param dim The dimension in which we are packing/unpacking
     @param ghost The array where we want to pack/unpack the ghost zone into/from
     @param extract Whether we are extracting into ghost or injecting from ghost
  */
  void extractExtendedGaugeGhost(const GaugeField &u, int dim, const int *R, 
				 void **ghost, bool extract);

  /**
     This function is used to calculate the maximum absolute value of
     a gauge field array.  Defined in max_gauge.cu.  

     @param u The gauge field from which we want to compute the max
  */
  double maxGauge(const GaugeField &u);

  /** 
      Apply the staggered phase factor to the gauge field.

      @param u The gauge field to which we apply the staggered phase factors
  */
  void applyGaugePhase(GaugeField &u);

} // namespace quda

#endif // _GAUGE_QUDA_H
