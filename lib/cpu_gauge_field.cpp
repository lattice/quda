#include <quda_internal.h>
#include <gauge_field.h>
#include <face_quda.h>
#include <assert.h>
#include <string.h>

namespace quda {

  cpuGaugeField::cpuGaugeField(const GaugeFieldParam &param) : 
    GaugeField(param), pinned(param.pinned)
  {
    if (precision == QUDA_HALF_PRECISION) {
      errorQuda("CPU fields do not support half precision");
    }
    if (pad != 0) {
      errorQuda("CPU fields do not support non-zero padding");
    }
    if (reconstruct != QUDA_RECONSTRUCT_NO && reconstruct != QUDA_RECONSTRUCT_10) {
      errorQuda("Reconstruction type %d not supported", reconstruct);
    }
    if (reconstruct == QUDA_RECONSTRUCT_10 && order != QUDA_MILC_GAUGE_ORDER) {
      errorQuda("10-reconstruction only supported with MILC gauge order");
    }

    int siteDim=0;
    if (geometry == QUDA_SCALAR_GEOMETRY) siteDim = 1;
    else if (geometry == QUDA_VECTOR_GEOMETRY) siteDim = nDim;
    else if (geometry == QUDA_TENSOR_GEOMETRY) siteDim = nDim * (nDim-1) / 2;

    if (order == QUDA_QDP_GAUGE_ORDER) {

      gauge = (void**) safe_malloc(siteDim * sizeof(void*));

      for (int d=0; d<siteDim; d++) {
	size_t nbytes = volume * reconstruct * precision;
	if (create == QUDA_NULL_FIELD_CREATE || create == QUDA_ZERO_FIELD_CREATE) {
	  gauge[d] = (pinned ? pinned_malloc(nbytes) : safe_malloc(nbytes));
	  if (create == QUDA_ZERO_FIELD_CREATE){
	    memset(gauge[d], 0, nbytes);
	  }
	} else if (create == QUDA_REFERENCE_FIELD_CREATE) {
	  gauge[d] = ((void**)param.gauge)[d];
	} else {
	  errorQuda("Unsupported creation type %d", create);
	}
      }
    
    } else if (order == QUDA_CPS_WILSON_GAUGE_ORDER || order == QUDA_MILC_GAUGE_ORDER  || 
	       order == QUDA_BQCD_GAUGE_ORDER || order == QUDA_TIFR_GAUGE_ORDER) {

      if (create == QUDA_NULL_FIELD_CREATE || create == QUDA_ZERO_FIELD_CREATE) {
	size_t nbytes = siteDim * volume * reconstruct * precision;
	gauge = (void **) (pinned ? pinned_malloc(nbytes) : safe_malloc(nbytes));
	if(create == QUDA_ZERO_FIELD_CREATE){
	  memset(gauge, 0, nbytes);
	}
      } else if (create == QUDA_REFERENCE_FIELD_CREATE) {
	gauge = (void**) param.gauge;
      } else {
	errorQuda("Unsupported creation type %d", create);
      }

    } else {
      errorQuda("Unsupported gauge order type %d", order);
    }
  
    // no need to exchange data if this is a momentum field
    if (link_type != QUDA_ASQTAD_MOM_LINKS) {
      // Ghost zone is always 2-dimensional    
      for (int i=0; i<nDim; i++) {
	size_t nbytes = nFace * surface[i] * reconstruct * precision;
	ghost[i] = safe_malloc(nbytes); // no need to use pinned memory for this
      }  

      if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD) {
	// exchange the boundaries if a non-trivial field
	if (create != QUDA_NULL_FIELD_CREATE && create != QUDA_ZERO_FIELD_CREATE &&
	    geometry == QUDA_VECTOR_GEOMETRY) 
	  exchangeGhost();
      }
    }

    // compute the fat link max now in case it is needed later (i.e., for half precision)
    if (param.compute_fat_link_max) fat_link_max = maxGauge(*this);
  }


  cpuGaugeField::~cpuGaugeField()
  {
    int siteDim = 0;
    if (geometry == QUDA_SCALAR_GEOMETRY) siteDim = 1;
    else if (geometry == QUDA_VECTOR_GEOMETRY) siteDim = nDim;
    else if (geometry == QUDA_TENSOR_GEOMETRY) siteDim = nDim * (nDim-1) / 2;

    if (create == QUDA_NULL_FIELD_CREATE || create == QUDA_ZERO_FIELD_CREATE) {
      if (order == QUDA_QDP_GAUGE_ORDER) {
	for (int d=0; d<siteDim; d++) {
	  if (gauge[d]) host_free(gauge[d]);
	}
	if (gauge) host_free(gauge);
      } else {
	if (gauge) host_free(gauge);
      }
    } else { // QUDA_REFERENCE_FIELD_CREATE 
      if (order == QUDA_QDP_GAUGE_ORDER){
	if (gauge) host_free(gauge);
      }
    }
  
    if (link_type != QUDA_ASQTAD_MOM_LINKS) {
      for (int i=0; i<nDim; i++) {
	if (ghost[i]) host_free(ghost[i]);
      }
    }
  }

  // This does the exchange of the gauge field ghost zone and places it
  // into the ghost array.
  void cpuGaugeField::exchangeGhost() {
    void *send[QUDA_MAX_DIM];
    for (int d=0; d<nDim; d++) send[d] = safe_malloc(nFace*surface[d]*reconstruct*precision);

    // get the links into contiguous buffers
    extractGaugeGhost(*this, send);

    // communicate between nodes
    FaceBuffer faceBuf(x, nDim, reconstruct, nFace, precision);
    faceBuf.exchangeLink(ghost, send, QUDA_CPU_FIELD_LOCATION);

    for (int d=0; d<nDim; d++) host_free(send[d]);
  }

  void cpuGaugeField::exchangeExtendedGhost(const int *R, bool no_comms_fill) {
    
    void *send[QUDA_MAX_DIM];
    void *recv[QUDA_MAX_DIM];
    size_t bytes[QUDA_MAX_DIM];
    // store both parities and directions in each
    for (int d=0; d<nDim; d++) {
      if (!commDimPartitioned(d) && !no_comms_fill) continue;
      bytes[d] = surface[d] * R[d] * geometry * reconstruct * precision;
      send[d] = safe_malloc(2 * bytes[d]);
      recv[d] = safe_malloc(2 * bytes[d]);
    }

    for (int d=0; d<nDim; d++) {
      if (!commDimPartitioned(d) && !no_comms_fill) continue;
      //extract into a contiguous buffer
      extractExtendedGaugeGhost(*this, d, R, send, true);

      if (commDimPartitioned(d)) {
	// do the exchange
	MsgHandle *mh_recv_back;
	MsgHandle *mh_recv_fwd;
	MsgHandle *mh_send_fwd;
	MsgHandle *mh_send_back;
	
	mh_recv_back = comm_declare_receive_relative(recv[d], d, -1, bytes[d]);
	mh_recv_fwd  = comm_declare_receive_relative(((char*)recv[d])+bytes[d], d, +1, bytes[d]);
	mh_send_back = comm_declare_send_relative(send[d], d, -1, bytes[d]);
	mh_send_fwd  = comm_declare_send_relative(((char*)send[d])+bytes[d], d, +1, bytes[d]);
	
	comm_start(mh_recv_back);
	comm_start(mh_recv_fwd);
	comm_start(mh_send_fwd);
	comm_start(mh_send_back);
	
	comm_wait(mh_send_fwd);
	comm_wait(mh_send_back);
	comm_wait(mh_recv_back);
	comm_wait(mh_recv_fwd);
	
	comm_free(mh_send_fwd);
	comm_free(mh_send_back);
	comm_free(mh_recv_back);
	comm_free(mh_recv_fwd);
      } else {
	memcpy(static_cast<char*>(recv[d])+bytes[d], send[d], bytes[d]);
	memcpy(recv[d], static_cast<char*>(send[d])+bytes[d], bytes[d]);
      }      

      // inject back into the gauge field
      extractExtendedGaugeGhost(*this, d, R, recv, false);
    }

    for (int d=0; d<nDim; d++) {
      if (!commDimPartitioned(d) && !no_comms_fill) continue;
      host_free(send[d]);
      host_free(recv[d]);
    }

  }

  void cpuGaugeField::setGauge(void **gauge_)
  {
    if(create != QUDA_REFERENCE_FIELD_CREATE) {
      errorQuda("Setting gauge pointer is only allowed when create="
		"QUDA_REFERENCE_FIELD_CREATE type\n");
    }
    gauge = gauge_;
  }

/*template <typename Float>
void print_matrix(const Float &m, unsigned int x) {

  for (int s=0; s<o.Nspin(); s++) {
    std::cout << "x = " << x << ", s = " << s << ", { ";
    for (int c=0; c<o.Ncolor(); c++) {
      std::cout << " ( " << o(x, s, c, 0) << " , " ;
      if (c<o.Ncolor()-1) std::cout << o(x, s, c, 1) << " ) ," ;
      else std::cout << o(x, s, c, 1) << " ) " ;
    }
    std::cout << " } " << std::endl;
  }

}

// print out the vector at volume point x
void cpuColorSpinorField::PrintMatrix(unsigned int x) {
  
  switch(precision) {
  case QUDA_DOUBLE_PRECISION:
    print_matrix(*order_double, x);
    break;
  case QUDA_SINGLE_PRECISION:
    print_matrix(*order_single, x);
    break;
  default:
    errorQuda("Precision %d not implemented", precision); 
  }

}
*/

} // namespace quda
