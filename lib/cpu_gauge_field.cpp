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

    if (order == QUDA_QDP_GAUGE_ORDER) {

      gauge = (void**) safe_malloc(nDim * sizeof(void*));

      for (int d=0; d<nDim; d++) {
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
    
    } else if (order == QUDA_CPS_WILSON_GAUGE_ORDER || order == QUDA_MILC_GAUGE_ORDER || order == QUDA_BQCD_GAUGE_ORDER) {

      if (create == QUDA_NULL_FIELD_CREATE || create == QUDA_ZERO_FIELD_CREATE) {
	size_t nbytes = nDim * volume * reconstruct * precision;
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
  
    // Ghost zone is always 2-dimensional
    ghost = (void**) safe_malloc(QUDA_MAX_DIM * sizeof(void*));
    for (int i=0; i<nDim; i++) {
      size_t nbytes = nFace * surface[i] * reconstruct * precision;
      ghost[i] = (pinned ? pinned_malloc(nbytes) : safe_malloc(nbytes));
    }  
  }


  cpuGaugeField::~cpuGaugeField()
  {
    if (create == QUDA_NULL_FIELD_CREATE || create == QUDA_ZERO_FIELD_CREATE) {
      if (order == QUDA_QDP_GAUGE_ORDER) {
	for (int d=0; d<nDim; d++) {
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
  
    for (int i=0; i<nDim; i++) {
      if (ghost[i]) host_free(ghost[i]);
    }
    if (ghost) host_free(ghost);
  }

  
  // transpose the matrix
  template <typename Float>
  inline void transpose(Float *gT, const Float *g) {
    for (int ic=0; ic<3; ic++) {
      for (int jc=0; jc<3; jc++) { 
	for (int r=0; r<2; r++) {
	  gT[(ic*3+jc)*2+r] = g[(jc*3+ic)*2+r];
	}
      }
    }
  }

  // FIXME declare here for prototyping
  void extractGhost(const GaugeField &, void **);

  // This does the exchange of the gauge field ghost zone and places it
  // into the ghost array.
  // This should be optimized so it is reused if called multiple times
  void cpuGaugeField::exchangeGhost() const {
    void **send = (void **) safe_malloc(QUDA_MAX_DIM*sizeof(void *));

    for (int d=0; d<nDim; d++) {
      send[d] = safe_malloc(nFace * surface[d] * reconstruct * precision);
    }

    // get the links into contiguous buffers
    extractGhost(*this, send);

    /*
    // get the links into a contiguous buffer
    if (precision == QUDA_DOUBLE_PRECISION) {
      packGhost((double**)send, (const double**)gauge, nFace, x, volumeCB, surfaceCB, order);
    } else {
      packGhost((float**)send, (const float**)gauge, nFace, x, volumeCB, surfaceCB, order);
      }*/

    // communicate between nodes
    FaceBuffer faceBuf(x, nDim, reconstruct, nFace, precision);
    faceBuf.exchangeCpuLink(ghost, send);

    for (int d=0; d<nDim; d++) {
      host_free(send[d]);
    }
    host_free(send);
  }

  void cpuGaugeField::setGauge(void **_gauge)
  {
    if(create != QUDA_REFERENCE_FIELD_CREATE) {
      errorQuda("Setting gauge pointer is only allowed when cpu gauge"
		"is of QUDA_REFERENCE_FIELD_CREATE type\n");
    }
    gauge = _gauge;
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
