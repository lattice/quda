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

  // FIXME - replace this with a functor approach to more easily arbitrary ordering
  template <typename Float>
    void packGhost(Float **ghost, const Float **gauge, const int nFace, const int *X, 
		   const int volumeCB, const int *surfaceCB, const QudaGaugeFieldOrder order) {
    
    if (order != QUDA_QDP_GAUGE_ORDER && order != QUDA_BQCD_GAUGE_ORDER &&
	order != QUDA_CPS_WILSON_GAUGE_ORDER && order != QUDA_MILC_GAUGE_ORDER) 
      errorQuda("packGhost not supported for %d gauge field order", order);
    
    int XY=X[0]*X[1];
    int XYZ=X[0]*X[1]*X[2];
    
    //loop variables: a, b, c with a the most signifcant and c the least significant
    //A, B, C the maximum value
    //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
    int A[4], B[4], C[4];
  
    //X dimension
    A[0] = X[3]; B[0] = X[2]; C[0] = X[1];
  
    //Y dimension
    A[1] = X[3]; B[1] = X[2]; C[1] = X[0];
    
    //Z dimension
    A[2] = X[3]; B[2] = X[1]; C[2] = X[0];
    
    //T dimension
    A[3] = X[2]; B[3] = X[1]; C[3] = X[0];
    
    //multiplication factor to compute index in original cpu memory
    int f[4][4]={
      {XYZ,    XY, X[0],     1},
      {XYZ,    XY,    1,  X[0]},
      {XYZ,  X[0],    1,    XY},
      { XY,  X[0],    1,   XYZ}
    };
    
    for(int dir =0; dir < 4; dir++)
      {
	const Float* even_src;
	const Float* odd_src;
	
	if (order == QUDA_BQCD_GAUGE_ORDER) {
	  // need to add on halo region
	  int mu_offset = X[0]/2 + 2;
	  for (int i=1; i<4; i++) mu_offset *= (X[i] + 2);
	  even_src = (const Float*)gauge + (dir*2+0)*mu_offset*gaugeSiteSize;
	  odd_src = (const Float*)gauge + (dir*2+1)*mu_offset*gaugeSiteSize;
	} else if (order == QUDA_CPS_WILSON_GAUGE_ORDER || order == QUDA_MILC_GAUGE_ORDER) {
	  even_src = (const Float*)gauge + 0*4*volumeCB*gaugeSiteSize;
	  odd_src = (const Float*)gauge + 1*4*volumeCB*gaugeSiteSize;
	} else { // QDP_GAUGE_FIELD_ORDER
	  even_src = gauge[dir];
	  odd_src = gauge[dir] + volumeCB*gaugeSiteSize;
	}
	
	Float* even_dst;
	Float* odd_dst;
	
	//switching odd and even ghost gauge when that dimension size is odd
	//only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
	if((X[dir] % 2 ==0) || (commDim(dir) == 1)){
	  even_dst = ghost[dir];
	  odd_dst = ghost[dir] + nFace*surfaceCB[dir]*gaugeSiteSize;	
	}else{
	  even_dst = ghost[dir] + nFace*surfaceCB[dir]*gaugeSiteSize;
	  odd_dst = ghost[dir];
	}

	int even_dst_index = 0;
	int odd_dst_index = 0;

	int d;
	int a,b,c;
	for(d = X[dir]- nFace; d < X[dir]; d++){
	  for(a = 0; a < A[dir]; a++){
	    for(b = 0; b < B[dir]; b++){
	      for(c = 0; c < C[dir]; c++){
		int index = ( a*f[dir][0] + b*f[dir][1]+ c*f[dir][2] + d*f[dir][3])>> 1;
		if (order == QUDA_CPS_WILSON_GAUGE_ORDER || order == QUDA_MILC_GAUGE_ORDER)
		  index = 4*index + dir;
		int oddness = (a+b+c+d)%2;
		if (oddness == 0){ //even
		  if (order == QUDA_BQCD_GAUGE_ORDER || order == QUDA_CPS_WILSON_GAUGE_ORDER) {
		    // we do transposition here so we can just call packQDPGauge for the ghost zone
		    Float gT[18];
		    transpose(gT, &even_src[18*index]);
		    for(int i=0; i<18; i++){
		      even_dst[18*even_dst_index+i] = gT[i];
		    }		    
		  } else {
		    for(int i=0;i < 18;i++){
		      even_dst[18*even_dst_index+i] = even_src[18*index + i];
		    }
		  }
		  even_dst_index++;
		}else{ //odd
		  if (order == QUDA_BQCD_GAUGE_ORDER || order == QUDA_CPS_WILSON_GAUGE_ORDER) {
		    // we do transposition here so we can just call packQDPGauge for the ghost zone
		    Float gT[18];
		    transpose(gT, &odd_src[18*index]);
		    for(int i=0; i<18; i++){
		      odd_dst[18*odd_dst_index+i] = gT[i];
		    }		    
		  } else {
		    for(int i=0;i < 18;i++){
		      odd_dst[18*odd_dst_index+i] = odd_src[18*index + i];
		    }
		  }
		  odd_dst_index++;
		}
	      }//c
	    }//b
	  }//a
	}//d

	assert( even_dst_index == nFace*surfaceCB[dir]);
	assert( odd_dst_index == nFace*surfaceCB[dir]);
      }

  }


  // This does the exchange of the gauge field ghost zone and places it
  // into the ghost array.
  // This should be optimized so it is reused if called multiple times
  void cpuGaugeField::exchangeGhost() const {
    void **send = (void **) safe_malloc(QUDA_MAX_DIM*sizeof(void *));

    for (int d=0; d<nDim; d++) {
      send[d] = safe_malloc(nFace * surface[d] * reconstruct * precision);
    }

    // get the links into a contiguous buffer
    if (precision == QUDA_DOUBLE_PRECISION) {
      packGhost((double**)send, (const double**)gauge, nFace, x, volumeCB, surfaceCB, order);
    } else {
      packGhost((float**)send, (const float**)gauge, nFace, x, volumeCB, surfaceCB, order);
    }

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
