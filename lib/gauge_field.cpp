#include <gauge_field.h>
#include <typeinfo>
#include <blas_quda.h>

namespace quda {

  GaugeFieldParam::GaugeFieldParam(const GaugeField &u) :
    LatticeFieldParam(u),
    location(u.Location()),
    nColor(u.Ncolor()),
    nFace(u.Nface()),
    reconstruct(u.Reconstruct()),
    order(u.Order()),
    fixed(u.GaugeFixed()),
    link_type(u.LinkType()),
    t_boundary(u.TBoundary()),
    anisotropy(u.Anisotropy()),
    tadpole(u.Tadpole()),
    gauge(NULL),
    create(QUDA_NULL_FIELD_CREATE),
    geometry(u.Geometry()),
    compute_fat_link_max(false),
    staggeredPhaseType(u.StaggeredPhase()),
    staggeredPhaseApplied(u.StaggeredPhaseApplied()),
    i_mu(u.iMu()),
    site_offset(u.SiteOffset()),
    site_size(u.SiteSize())
  { }

  GaugeField::GaugeField(const GaugeFieldParam &param) :
    LatticeField(param),
    bytes(0),
    phase_offset(0),
    phase_bytes(0),
    nColor(param.nColor),
    nFace(param.nFace),
    geometry(param.geometry),
    reconstruct(param.reconstruct),
    nInternal(reconstruct != QUDA_RECONSTRUCT_NO ? reconstruct : nColor * nColor * 2),
    order(param.order),
    fixed(param.fixed),
    link_type(param.link_type),
    t_boundary(param.t_boundary),
    anisotropy(param.anisotropy),
    tadpole(param.tadpole),
    fat_link_max(link_type == QUDA_ASQTAD_FAT_LINKS ? 0.0 : 1.0),
    create(param.create),
    staggeredPhaseType(param.staggeredPhaseType),
    staggeredPhaseApplied(param.staggeredPhaseApplied),
    i_mu(param.i_mu),
    site_offset(param.site_offset),
    site_size(param.site_size)
  {
    if (order == QUDA_NATIVE_GAUGE_ORDER) errorQuda("Invalid gauge order %d", order);
    if (ghost_precision != precision) ghost_precision = precision; // gauge fields require matching precision

    if (link_type != QUDA_COARSE_LINKS && nColor != 3)
      errorQuda("nColor must be 3, not %d for this link type", nColor);
    if (nDim != 4)
      errorQuda("Number of dimensions must be 4 not %d", nDim);
    if (link_type != QUDA_WILSON_LINKS && anisotropy != 1.0)
      errorQuda("Anisotropy only supported for Wilson links");
    if (link_type != QUDA_WILSON_LINKS && fixed == QUDA_GAUGE_FIXED_YES)
      errorQuda("Temporal gauge fixing only supported for Wilson links");
    if (geometry == QUDA_SCALAR_GEOMETRY) {
      real_length = volume*nInternal;
      length = 2*stride*nInternal; // two comes from being full lattice
    } else if (geometry == QUDA_VECTOR_GEOMETRY) {
      real_length = nDim*volume*nInternal;
      length = 2*nDim*stride*nInternal; // two comes from being full lattice
    } else if (geometry == QUDA_TENSOR_GEOMETRY) {
      real_length = (nDim*(nDim-1)/2)*volume*nInternal;
      length = 2*(nDim*(nDim-1)/2)*stride*nInternal; // two comes from being full lattice
    } else if (geometry == QUDA_COARSE_GEOMETRY) {
      real_length = 2*nDim*volume*nInternal;
      length = 2*2*nDim*stride*nInternal;  //two comes from being full lattice
    }

    if ((reconstruct == QUDA_RECONSTRUCT_12 || reconstruct == QUDA_RECONSTRUCT_8) && link_type != QUDA_SU3_LINKS) {
      errorQuda("Cannot request a 12/8 reconstruct type without SU(3) link type");
    }

    if (reconstruct == QUDA_RECONSTRUCT_9 || reconstruct == QUDA_RECONSTRUCT_13) {
      // Need to adjust the phase alignment as well.
      int half_phase_bytes
        = (length / (2 * reconstruct)) * precision; // number of bytes needed to store phases for a single parity
      int half_gauge_bytes = (length / 2) * precision
        - half_phase_bytes; // number of bytes needed to store the gauge field for a single parity excluding the phases
      // Adjust the alignments for the gauge and phase separately
      half_phase_bytes = ((half_phase_bytes + (512-1))/512)*512;
      half_gauge_bytes = ((half_gauge_bytes + (512-1))/512)*512;
    
      phase_offset = half_gauge_bytes;
      phase_bytes = half_phase_bytes*2;
      bytes = (half_gauge_bytes + half_phase_bytes)*2;      
    } else {
      bytes = length * precision;
      if (isNative()) bytes = 2*ALIGNMENT_ADJUST(bytes/2);
    }
    total_bytes = bytes;

    setTuningString();
  }

  GaugeField::~GaugeField() {

  }

  void GaugeField::setTuningString() {
    LatticeField::setTuningString();
    int aux_string_n = TuneKey::aux_n / 2;
    int check = snprintf(aux_string, aux_string_n, "vol=%lu,stride=%lu,precision=%d,geometry=%d,Nc=%d", volume, stride,
                         precision, geometry, nColor);
    if (check < 0 || check >= aux_string_n) errorQuda("Error writing aux string");
  }

  void GaugeField::createGhostZone(const int *R, bool no_comms_fill, bool bidir) const
  {
    if (typeid(*this) == typeid(cpuGaugeField)) return;

    // if this is not a bidirectional exchange then we are doing a
    // scalar exchange, e.g., only the link matrix in the direcion we
    // are exchanging is exchanged, and none of the orthogonal links
    QudaFieldGeometry geometry_comms = bidir ? (geometry == QUDA_COARSE_GEOMETRY ? QUDA_VECTOR_GEOMETRY : geometry) : QUDA_SCALAR_GEOMETRY;

    // calculate size of ghost zone required
    ghost_bytes_old = ghost_bytes; // save for subsequent resize checking
    ghost_bytes = 0;
    for (int i=0; i<nDim; i++) {
      ghost_face_bytes[i] = 0;
      if ( !(comm_dim_partitioned(i) || (no_comms_fill && R[i])) ) ghostFace[i] = 0;
      else ghostFace[i] = surface[i] * R[i]; // includes the radius (unlike ColorSpinorField)

      ghostOffset[i][0] = (i == 0) ? 0 : ghostOffset[i-1][1] + ghostFace[i-1]*geometry_comms*nInternal;
      ghostOffset[i][1] = (bidir ? ghostOffset[i][0] + ghostFace[i]*geometry_comms*nInternal : ghostOffset[i][0]);

      ghost_face_bytes[i] = ghostFace[i] * geometry_comms * nInternal * ghost_precision;
      ghost_bytes += (bidir ? 2 : 1 ) * ghost_face_bytes[i]; // factor of two from direction
    }

    if (isNative()) ghost_bytes = ALIGNMENT_ADJUST(ghost_bytes);
  } // createGhostZone

  void GaugeField::applyStaggeredPhase(QudaStaggeredPhase phase) {
    if (staggeredPhaseApplied) errorQuda("Staggered phases already applied");

    if (phase != QUDA_STAGGERED_PHASE_INVALID) staggeredPhaseType = phase;
    applyGaugePhase(*this);
    if (ghostExchange==QUDA_GHOST_EXCHANGE_PAD) {
      if (typeid(*this)==typeid(cudaGaugeField)) {
	static_cast<cudaGaugeField&>(*this).exchangeGhost();
      } else {
	static_cast<cpuGaugeField&>(*this).exchangeGhost();
      }
    }
    staggeredPhaseApplied = true;
  }

  void GaugeField::removeStaggeredPhase() {
    if (!staggeredPhaseApplied) errorQuda("No staggered phases to remove");
    applyGaugePhase(*this);
    if (ghostExchange==QUDA_GHOST_EXCHANGE_PAD) {
      if (typeid(*this)==typeid(cudaGaugeField)) {
	static_cast<cudaGaugeField&>(*this).exchangeGhost();
      } else {
	static_cast<cpuGaugeField&>(*this).exchangeGhost();
      }
    }
    staggeredPhaseApplied = false;
  }

  bool GaugeField::isNative() const { return gauge::isNative(order, precision, reconstruct); }

  void GaugeField::exchange(void **ghost_link, void **link_sendbuf, QudaDirection dir) const {
    MsgHandle *mh_send[4];
    MsgHandle *mh_recv[4];
    size_t bytes[4];

    for (int i=0; i<nDimComms; i++) bytes[i] = 2*nFace*surfaceCB[i]*nInternal*precision;

    // in general (standard ghost exchange) we always do the exchange
    // even if a dimension isn't partitioned.  However, this breaks
    // GaugeField::injectGhost(), so when transferring backwards we
    // only exchange if a dimension is partitioned.  FIXME: this
    // should probably be cleaned up.
    bool no_comms_fill = (dir == QUDA_BACKWARDS) ? false : true;

    void *send[4];
    void *receive[4];
    if (Location() == QUDA_CPU_FIELD_LOCATION) {
      for (int i=0; i<nDimComms; i++) {
	if (comm_dim_partitioned(i)) {
	  send[i] = link_sendbuf[i];
	  receive[i] = ghost_link[i];
	} else {
	  if (no_comms_fill) memcpy(ghost_link[i], link_sendbuf[i], bytes[i]);
	}
      }
    } else { // FIXME for CUDA field copy back to the CPU
      for (int i=0; i<nDimComms; i++) {
	if (comm_dim_partitioned(i)) {
	  send[i] = pool_pinned_malloc(bytes[i]);
	  receive[i] = pool_pinned_malloc(bytes[i]);
	  qudaMemcpy(send[i], link_sendbuf[i], bytes[i], cudaMemcpyDeviceToHost);
	} else {
	  if (no_comms_fill) qudaMemcpy(ghost_link[i], link_sendbuf[i], bytes[i], cudaMemcpyDeviceToDevice);
	}
      }
    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      if (dir == QUDA_FORWARDS) {
	mh_send[i] = comm_declare_send_relative(send[i], i, +1, bytes[i]);
	mh_recv[i] = comm_declare_receive_relative(receive[i], i, -1, bytes[i]);
      } else if (dir == QUDA_BACKWARDS) {
	mh_send[i] = comm_declare_send_relative(send[i], i, -1, bytes[i]);
	mh_recv[i] = comm_declare_receive_relative(receive[i], i, +1, bytes[i]);
      } else {
	errorQuda("Unsuported dir=%d", dir);
      }

    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_start(mh_send[i]);
      comm_start(mh_recv[i]);
    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_wait(mh_send[i]);
      comm_wait(mh_recv[i]);
    }

    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      for (int i=0; i<nDimComms; i++) {
	if (!comm_dim_partitioned(i)) continue;
	qudaMemcpy(ghost_link[i], receive[i], bytes[i], cudaMemcpyHostToDevice);
	pool_pinned_free(send[i]);
	pool_pinned_free(receive[i]);
      }
    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_free(mh_send[i]);
      comm_free(mh_recv[i]);
    }

  }

  void GaugeField::checkField(const LatticeField &l) const {
    LatticeField::checkField(l);
    try {
      const GaugeField &g = dynamic_cast<const GaugeField&>(l);
      if (g.link_type != link_type) errorQuda("link_type does not match %d %d", link_type, g.link_type);
      if (g.nColor != nColor) errorQuda("nColor does not match %d %d", nColor, g.nColor);
      if (g.nFace != nFace) errorQuda("nFace does not match %d %d", nFace, g.nFace);
      if (g.fixed != fixed) errorQuda("fixed does not match %d %d", fixed, g.fixed);
      if (g.t_boundary != t_boundary) errorQuda("t_boundary does not match %d %d", t_boundary, g.t_boundary);
      if (g.anisotropy != anisotropy) errorQuda("anisotropy does not match %e %e", anisotropy, g.anisotropy);
      if (g.tadpole != tadpole) errorQuda("tadpole does not match %e %e", tadpole, g.tadpole);
    }
    catch(std::bad_cast &e) {
      errorQuda("Failed to cast reference to GaugeField");
    }
  }

  std::ostream& operator<<(std::ostream& output, const GaugeFieldParam& param) {
    output << static_cast<const LatticeFieldParam &>(param);
    output << "nColor = " << param.nColor << std::endl;
    output << "nFace = " << param.nFace << std::endl;
    output << "reconstruct = " << param.reconstruct << std::endl;
    int nInternal = (param.reconstruct != QUDA_RECONSTRUCT_NO ? 
		     param.reconstruct : param.nColor * param.nColor * 2);
    output << "nInternal = " << nInternal << std::endl;
    output << "order = " << param.order << std::endl;
    output << "fixed = " << param.fixed << std::endl;
    output << "link_type = " << param.link_type << std::endl;
    output << "t_boundary = " << param.t_boundary << std::endl;
    output << "anisotropy = " << param.anisotropy << std::endl;
    output << "tadpole = " << param.tadpole << std::endl;
    output << "create = " << param.create << std::endl;
    output << "geometry = " << param.geometry << std::endl;
    output << "staggeredPhaseType = " << param.staggeredPhaseType << std::endl;
    output << "staggeredPhaseApplied = " << param.staggeredPhaseApplied << std::endl;

    return output;  // for multiple << operators.
  }

  ColorSpinorParam colorSpinorParam(const GaugeField &a) {
   if (a.FieldOrder() == QUDA_QDP_GAUGE_ORDER || a.FieldOrder() == QUDA_QDPJIT_GAUGE_ORDER)
     errorQuda("Not implemented for this order %d", a.FieldOrder());

    if (a.LinkType() == QUDA_COARSE_LINKS) errorQuda("Not implemented for coarse-link type");
    if (a.Ncolor() != 3) errorQuda("Not implemented for Ncolor = %d", a.Ncolor());

    if (a.Precision() == QUDA_HALF_PRECISION || a.Precision() == QUDA_QUARTER_PRECISION)
      errorQuda("Casting a GaugeField into ColorSpinorField not possible in half or quarter precision");

    ColorSpinorParam spinor_param;
    spinor_param.nColor = (a.Geometry()*a.Reconstruct())/2;
    spinor_param.nSpin = 1;
    spinor_param.nDim = a.Ndim();
    for (int d=0; d<a.Ndim(); d++) spinor_param.x[d] = a.X()[d];
    spinor_param.setPrecision(a.Precision());
    spinor_param.pad = a.Pad();
    spinor_param.siteSubset = QUDA_FULL_SITE_SUBSET;
    spinor_param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    spinor_param.fieldOrder = (a.Precision() == QUDA_DOUBLE_PRECISION || spinor_param.nSpin == 1) ?
      QUDA_FLOAT2_FIELD_ORDER : QUDA_FLOAT4_FIELD_ORDER;
    spinor_param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    spinor_param.create = QUDA_REFERENCE_FIELD_CREATE;
    spinor_param.v = (void*)a.Gauge_p();
    spinor_param.location = a.Location();
    return spinor_param;
  }

  // Return the L2 norm squared of the gauge field
  double norm2(const GaugeField &a) {
    ColorSpinorField *b = ColorSpinorField::Create(colorSpinorParam(a));
    double nrm2 = blas::norm2(*b);
    delete b;
    return nrm2;
  }

  // Return the L1 norm of the gauge field
  double norm1(const GaugeField &a) {
    ColorSpinorField *b = ColorSpinorField::Create(colorSpinorParam(a));
    double nrm1 = blas::norm1(*b);
    delete b;
    return nrm1;
  }

  // Scale the gauge field by the constant a
  void ax(const double &a, GaugeField &u) {
    ColorSpinorField *b = ColorSpinorField::Create(colorSpinorParam(u));
    blas::ax(a, *b);
    delete b;
  }

  uint64_t GaugeField::checksum(bool mini) const {
    return Checksum(*this, mini);
  }

  GaugeField* GaugeField::Create(const GaugeFieldParam &param) {

    GaugeField *field = nullptr;
    if (param.location == QUDA_CPU_FIELD_LOCATION) {
      field = new cpuGaugeField(param);
    } else if (param.location== QUDA_CUDA_FIELD_LOCATION) {
      field = new cudaGaugeField(param);
    } else {
      errorQuda("Invalid field location %d", param.location);
    }

    return field;
  }

  // helper for creating extended gauge fields
  cudaGaugeField *createExtendedGauge(cudaGaugeField &in, const int *R, TimeProfile &profile, bool redundant_comms,
                                      QudaReconstructType recon)
  {
    profile.TPSTART(QUDA_PROFILE_INIT);
    GaugeFieldParam gParamEx(in);
    gParamEx.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
    gParamEx.pad = 0;
    gParamEx.nFace = 1;
    gParamEx.tadpole = in.Tadpole();
    gParamEx.anisotropy = in.Anisotropy();
    for (int d = 0; d < 4; d++) {
      gParamEx.x[d] += 2 * R[d];
      gParamEx.r[d] = R[d];
    }

    auto *out = new cudaGaugeField(gParamEx);

    // copy input field into the extended device gauge field
    copyExtendedGauge(*out, in, QUDA_CUDA_FIELD_LOCATION);

    profile.TPSTOP(QUDA_PROFILE_INIT);

    // now fill up the halos
    out->exchangeExtendedGhost(R, profile, redundant_comms);

    return out;
  }

  // helper for creating extended (cpu) gauge fields
  cpuGaugeField *createExtendedGauge(void **gauge, QudaGaugeParam &gauge_param, const int *R)
  {
    GaugeFieldParam gauge_field_param(gauge, gauge_param);
    cpuGaugeField cpu(gauge_field_param);

    gauge_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
    gauge_field_param.create = QUDA_ZERO_FIELD_CREATE;
    for (int d = 0; d < 4; d++) {
      gauge_field_param.x[d] += 2 * R[d];
      gauge_field_param.r[d] = R[d];
    }
    cpuGaugeField *padded_cpu = new cpuGaugeField(gauge_field_param);

    copyExtendedGauge(*padded_cpu, cpu, QUDA_CPU_FIELD_LOCATION);
    padded_cpu->exchangeExtendedGhost(R, true); // Do comm to fill halo = true

    return padded_cpu;
  }

} // namespace quda
