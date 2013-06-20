#include <gauge_field_order.h>

namespace quda {

  /**
     Generic CPU gauge ghost extraction and packing
     NB This routines is specialized to four dimensions

     FIXME - need to ensure that saveGhost is pointing to the send
     buffer not the ghost zone per se
  */
  template <typename Float, int length, int nDim, typename Order>
    void extractGhost(Order order, int nFace, const int surfaceCB[nDim], const int X[nDim],
			   const int A[nDim], const int B[nDim], const int C[nDim], 
			   const int f[nDim][nDim], const int localParity[nDim]) {  
    typedef typename mapper<Float>::type RegType;

    for (int parity=0; parity<2; parity++) {

      for (int dim=0; dim<nDim; dim++) {

	// linear index used for writing into ghost buffer
	int indexDst = 0;
	// the following 4-way loop means this is specialized for 4 dimensions 

	for (int d=X[dim]-nFace; d<X[dim]; d++) { // loop over last nFace faces in this dimension
	  for (int a=0; a<A[dim]; a++) { // loop over the surface elements of this face
	    for (int b=0; b<B[dim]; b++) { // loop over the surface elements of this face
	      for (int c=0; c<C[dim]; c++) { // loop over the surface elements of this face
		// index is a checkboarded spacetime coordinate
		int indexCB = (a*f[dim][0] + b*f[dim][1]+ c*f[dim][2] + d*f[dim][3]) >> 1;
		// we only do the extraction for parity we are currently working on
		int oddness = (a+b+c+d)%2;
		if (oddness == parity) {
		  RegType u[length];
		  order.load(u, indexCB, dim, parity); // load the ghost element from the bulk
		  order.saveGhost(u, indexDst, dim, (parity+localParity[dim])%2);
		  indexDst++;
		} // oddness == parity
	      } // c
	    } // b
	  } // a
	} // d

	assert(indexDst == nFace*surfaceCB[dim]);
      } // dim

    } // parity

  }

  /** This is the template driver for extractGhost */
  template <typename Float>
    void extractGhost(const GaugeField &u, Float **Ghost) {

    const int nDim = 4;
    const int length = 18;
    const int *X = u.X();

    //loop variables: a, b, c with a the most signifcant and c the least significant
    //A, B, C the maximum value
    //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
    int A[nDim], B[nDim], C[nDim];
    A[0] = X[3]; B[0] = X[2]; C[0] = X[1]; // X dimension face
    A[1] = X[3]; B[1] = X[2]; C[1] = X[0]; // Y dimension face
    A[2] = X[3]; B[2] = X[1]; C[2] = X[0]; // Z dimension face
    A[3] = X[2]; B[3] = X[1]; C[3] = X[0]; // T dimension face    

    //multiplication factor to compute index in original cpu memory
    int f[4][4]={
      {X[0]*X[1]*X[2],  X[0]*X[1], X[0],               1},
      {X[0]*X[1]*X[2],  X[0]*X[1],    1,            X[0]},
      {X[0]*X[1]*X[2],       X[0],    1,       X[0]*X[1]},
      {     X[0]*X[1],       X[0],    1,  X[0]*X[1]*X[2]}
    };

    //set the local processor parity 
    //switching odd and even ghost gauge when that dimension size is odd
    //only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
    // FIXME - I don't understand this, shouldn't it be commDim(dim) == 0 ?
    int localParity[nDim];
    for (int dim=0; dim<nDim; dim++) {
      localParity[dim] = ((X[dim] % 2 ==0) || (commDim(dim) == 1)) ? 0 : 1;
    }

    if (u.Order() == QUDA_FLOAT_GAUGE_ORDER) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(Float)==typeid(short) && u.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  extractGhost<Float,length,nDim>(FloatNOrder<Float,length,1,19>(u, 0, Ghost), 
					  u.Nface(), u.SurfaceCB(), u.X(), 
					  A, B, C, f, localParity);
	} else {
	  extractGhost<Float,length,nDim>(FloatNOrder<Float,length,1,18>(u, 0, Ghost),
					  u.Nface(), u.SurfaceCB(), u.X(), 
					  A, B, C, f, localParity);
	}
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
	extractGhost<Float,length,nDim>(FloatNOrder<Float,length,1,12>(u, 0, Ghost), 
					u.Nface(), u.SurfaceCB(), u.X(), 
					A, B, C, f, localParity);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_8) {
	extractGhost<Float,length,nDim>(FloatNOrder<Float,length,1,8>(u, 0, Ghost), 
					u.Nface(), u.SurfaceCB(), u.X(), 
					A, B, C, f, localParity);
      }
    } else if (u.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(Float)==typeid(short) && u.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  extractGhost<Float,length,nDim>(FloatNOrder<Float,length,2,19>(u, 0, Ghost), 
					  u.Nface(), u.SurfaceCB(), u.X(), 
					  A, B, C, f, localParity);
	} else {
	  extractGhost<Float,length,nDim>(FloatNOrder<Float,length,2,18>(u, 0, Ghost),
					  u.Nface(), u.SurfaceCB(), u.X(), 
					  A, B, C, f, localParity);
	}
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
	extractGhost<Float,length,nDim>(FloatNOrder<Float,length,2,12>(u, 0, Ghost),
					u.Nface(), u.SurfaceCB(), u.X(), 
					A, B, C, f, localParity);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_8) {
	extractGhost<Float,length,nDim>(FloatNOrder<Float,length,2,8>(u, 0, Ghost), 
					u.Nface(), u.SurfaceCB(), u.X(), 
					A, B, C, f, localParity);
					}
    } else if (u.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(Float)==typeid(short) && u.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  extractGhost<Float,length,nDim>(FloatNOrder<Float,length,1,19>(u, 0, Ghost),
					  u.Nface(), u.SurfaceCB(), u.X(), 
					  A, B, C, f, localParity);
	} else {
	  extractGhost<Float,length,nDim>(FloatNOrder<Float,length,1,18>(u, 0, Ghost),
					  u.Nface(), u.SurfaceCB(), u.X(), 
					  A, B, C, f, localParity);
	}
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
	extractGhost<Float,length,nDim>(FloatNOrder<Float,length,4,12>(u, 0, Ghost),
					u.Nface(), u.SurfaceCB(), u.X(), 
					A, B, C, f, localParity);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_8) {
	extractGhost<Float,length,nDim>(FloatNOrder<Float,length,4,8>(u, 0, Ghost),
					u.Nface(), u.SurfaceCB(), u.X(), 
					A, B, C, f, localParity);
					}
    } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
      extractGhost<Float,length,nDim>(QDPOrder<Float,length>(u, 0, Ghost),
				      u.Nface(), u.SurfaceCB(), u.X(), 
				      A, B, C, f, localParity);
    } else if (u.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {
      extractGhost<Float,length,nDim>(CPSOrder<Float,length>(u, 0, Ghost),
				      u.Nface(), u.SurfaceCB(), u.X(), 
				      A, B, C, f, localParity);
    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {
      extractGhost<Float,length,nDim>(MILCOrder<Float,length>(u, 0, Ghost),
				      u.Nface(), u.SurfaceCB(), u.X(), 
				      A, B, C, f, localParity);
    } else if (u.Order() == QUDA_BQCD_GAUGE_ORDER) {
      extractGhost<Float,length,nDim>(BQCDOrder<Float,length>(u, 0, Ghost),
				      u.Nface(), u.SurfaceCB(), u.X(), 
				      A, B, C, f, localParity);
    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }

  }

  void extractGaugeGhost(const GaugeField &u, void **ghost) {
    if (u.Precision() == QUDA_DOUBLE_PRECISION) {
      extractGhost(u, (double**)ghost);
    } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
      extractGhost(u, (float**)ghost);
    } else if (u.Precision() == QUDA_HALF_PRECISION) {
      extractGhost(u, (short**)ghost);      
    } else {
      errorQuda("Unknown precision type %d", u.Precision());
    }
  }

} // namespace quda
