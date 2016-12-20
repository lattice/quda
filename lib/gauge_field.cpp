#include <gauge_field.h>
#include <typeinfo>
#include <blas_quda.h>

namespace quda {

  GaugeFieldParam::GaugeFieldParam(const GaugeField &u) : LatticeFieldParam(),
    nColor(3),
    nFace(u.Nface()),
    reconstruct(u.Reconstruct()),
    order(u.Order()),
    fixed(u.GaugeFixed()),
    link_type(u.LinkType()),
    t_boundary(u.TBoundary()),
    anisotropy(u.Anisotropy()),
    tadpole(u.Tadpole()),
    scale(u.Scale()),
    gauge(NULL),
    create(QUDA_NULL_FIELD_CREATE),
    geometry(u.Geometry()),
    compute_fat_link_max(false),
    ghostExchange(u.GhostExchange()),
    staggeredPhaseType(u.StaggeredPhase()),
    staggeredPhaseApplied(u.StaggeredPhaseApplied()),
    i_mu(u.iMu())
      {
	precision = u.Precision();
	nDim = u.Ndim();
	pad = u.Pad();
	siteSubset = QUDA_FULL_SITE_SUBSET;

	for(int dir=0; dir<nDim; ++dir) {
	  x[dir] = u.X()[dir];
	  r[dir] = u.R()[dir];
	}
      }


  GaugeField::GaugeField(const GaugeFieldParam &param) :
    LatticeField(param), bytes(0), phase_offset(0), phase_bytes(0), nColor(param.nColor), nFace(param.nFace),
    geometry(param.geometry), reconstruct(param.reconstruct), 
    nInternal(reconstruct != QUDA_RECONSTRUCT_NO ? reconstruct : nColor * nColor * 2),
    order(param.order), fixed(param.fixed), link_type(param.link_type), t_boundary(param.t_boundary), 
    anisotropy(param.anisotropy), tadpole(param.tadpole), fat_link_max(0.0), scale(param.scale),  
    create(param.create), ghostExchange(param.ghostExchange), 
    staggeredPhaseType(param.staggeredPhaseType), staggeredPhaseApplied(param.staggeredPhaseApplied), i_mu(param.i_mu)
  {
    if (link_type != QUDA_COARSE_LINKS && nColor != 3)
      errorQuda("nColor must be 3, not %d for this link type", nColor);
    if (nDim != 4)
      errorQuda("Number of dimensions must be 4 not %d", nDim);
    if (link_type != QUDA_WILSON_LINKS && anisotropy != 1.0)
      errorQuda("Anisotropy only supported for Wilson links");
    if (link_type != QUDA_WILSON_LINKS && fixed == QUDA_GAUGE_FIXED_YES)
      errorQuda("Temporal gauge fixing only supported for Wilson links");

    if(link_type != QUDA_ASQTAD_LONG_LINKS && (reconstruct ==  QUDA_RECONSTRUCT_13 || reconstruct == QUDA_RECONSTRUCT_9))
      errorQuda("reconstruct %d only supported for staggered long links\n", reconstruct);
       
    if (link_type == QUDA_ASQTAD_MOM_LINKS) scale = 1.0;

    if(geometry == QUDA_SCALAR_GEOMETRY) {
      real_length = volume*nInternal;
      length = 2*stride*nInternal; // two comes from being full lattice
    } else if (geometry == QUDA_VECTOR_GEOMETRY) {
      real_length = nDim*volume*nInternal;
      length = 2*nDim*stride*nInternal; // two comes from being full lattice
    } else if(geometry == QUDA_TENSOR_GEOMETRY){
      real_length = (nDim*(nDim-1)/2)*volume*nInternal;
      length = 2*(nDim*(nDim-1)/2)*stride*nInternal; // two comes from being full lattice
    } else if(geometry == QUDA_COARSE_GEOMETRY){
      real_length = 2*nDim*volume*nInternal;
      length = 2*2*nDim*stride*nInternal;  //two comes from being full lattice
    }

    if (ghostExchange == QUDA_GHOST_EXCHANGE_EXTENDED) {
      for (int d=0; d<nDim; d++) r[d] = param.r[d];
    } else {
      for (int d=0; d<nDim; d++) r[d] = 0;
    }


    if (reconstruct == QUDA_RECONSTRUCT_9 || reconstruct == QUDA_RECONSTRUCT_13) {
      // Need to adjust the phase alignment as well.  
      int half_phase_bytes = ((size_t)length/(2*reconstruct))*precision; // number of bytes needed to store phases for a single parity
      int half_gauge_bytes = ((size_t)length/2)*precision - half_phase_bytes; // number of bytes needed to store the gauge field for a single parity excluding the phases
      // Adjust the alignments for the gauge and phase separately
      half_phase_bytes = ((half_phase_bytes + (512-1))/512)*512;
      half_gauge_bytes = ((half_gauge_bytes + (512-1))/512)*512;
    
      phase_offset = half_gauge_bytes;
      phase_bytes = half_phase_bytes*2;
      bytes = (half_gauge_bytes + half_phase_bytes)*2;      
    } else {
      bytes = (size_t)length*precision;
      if (isNative()) bytes = 2*ALIGNMENT_ADJUST(bytes/2);
    }
    total_bytes = bytes;
  }

  GaugeField::~GaugeField() {

  }

  void GaugeField::applyStaggeredPhase() {
    if (staggeredPhaseApplied) errorQuda("Staggered phases already applied");
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

  bool GaugeField::isNative() const {
    if (precision == QUDA_DOUBLE_PRECISION) {
      if (order  == QUDA_FLOAT2_GAUGE_ORDER) return true;
    } else if (precision == QUDA_SINGLE_PRECISION || 
	       precision == QUDA_HALF_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	if (order == QUDA_FLOAT2_GAUGE_ORDER) return true;
      } else if (reconstruct == QUDA_RECONSTRUCT_12 || reconstruct == QUDA_RECONSTRUCT_13) {
	if (order == QUDA_FLOAT4_GAUGE_ORDER) return true;
      } else if (reconstruct == QUDA_RECONSTRUCT_8 || reconstruct == QUDA_RECONSTRUCT_9) {
	if (order == QUDA_FLOAT4_GAUGE_ORDER) return true;
      } else if (reconstruct == QUDA_RECONSTRUCT_10) {
	if (order == QUDA_FLOAT2_GAUGE_ORDER) return true;
      }
    }
    return false;
  }

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

  void GaugeField::checkField(const GaugeField &a) {
    LatticeField::checkField(a);
    if (a.link_type != link_type) errorQuda("link_type does not match %d %d", link_type, a.link_type);
    if (a.nColor != nColor) errorQuda("nColor does not match %d %d", nColor, a.nColor);
    if (a.nFace != nFace) errorQuda("nFace does not match %d %d", nFace, a.nFace);
    if (a.fixed != fixed) errorQuda("fixed does not match %d %d", fixed, a.fixed);
    if (a.t_boundary != t_boundary) errorQuda("t_boundary does not match %d %d", t_boundary, a.t_boundary);
    if (a.anisotropy != anisotropy) errorQuda("anisotropy does not match %e %e", anisotropy, a.anisotropy);
    if (a.tadpole != tadpole) errorQuda("tadpole does not match %e %e", tadpole, a.tadpole);
    //if (a.scale != scale) errorQuda("scale does not match %e %e", scale, a.scale); 
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
    output << "scale = " << param.scale << std::endl;
    output << "create = " << param.create << std::endl;
    output << "geometry = " << param.geometry << std::endl;
    output << "ghostExchange = " << param.ghostExchange << std::endl;
    for (int i=0; i<param.nDim; i++) {
      output << "r[" << i << "] = " << param.r[i] << std::endl;    
    }
    output << "staggeredPhaseType = " << param.staggeredPhaseType << std::endl;
    output << "staggeredPhaseApplied = " << param.staggeredPhaseApplied << std::endl;

    return output;  // for multiple << operators.
  }

  ColorSpinorParam colorSpinorParam(const GaugeField &a) {
   if (a.FieldOrder() == QUDA_QDP_GAUGE_ORDER || a.FieldOrder() == QUDA_QDPJIT_GAUGE_ORDER)
     errorQuda("Not implemented for this order %d", a.FieldOrder());

    if (a.LinkType() == QUDA_COARSE_LINKS) errorQuda("Not implemented for coarse-link type");
    if (a.Ncolor() != 3) errorQuda("Not implemented for Ncolor = %d", a.Ncolor());

    if (a.Precision() == QUDA_HALF_PRECISION)
      errorQuda("Casting a GaugeField into ColorSpinorField not possible in half precision");

    ColorSpinorParam spinor_param;
    spinor_param.nColor = (a.Geometry()*a.Reconstruct())/2;
    spinor_param.nSpin = 1;
    spinor_param.nDim = a.Ndim();
    for (int d=0; d<a.Ndim(); d++) spinor_param.x[d] = a.X()[d];
    spinor_param.precision = a.Precision();
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

} // namespace quda
