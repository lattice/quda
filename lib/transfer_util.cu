#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

namespace quda {

  template <typename Float>
  ColorSpinorFieldOrder<Float>* createOrder(const cpuColorSpinorField &a) {
    ColorSpinorFieldOrder<Float>* ptr=0;
    if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      ptr = new SpaceSpinColorOrder<Float>(const_cast<cpuColorSpinorField&>(a));
    } else if (a.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      ptr = new SpaceColorSpinOrder<Float>(const_cast<cpuColorSpinorField&>(a));
    } else if (a.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
      ptr = new QOPDomainWallOrder<Float>(const_cast<cpuColorSpinorField&>(a));
    } else {
      errorQuda("Order %d not supported in cpuColorSpinorField", a.FieldOrder());
    }

    return ptr;
  }

  // copy the null-space vectors into the V-field
  template <class V, class B>
  void fill(V &out, const B &in, int v, int Nvec) {
    for (int x=0; x<out.Volume(); x++) {
      for (int s=0; s<in.Nspin(); s++) {
	for (int c=0; c<in.Ncolor(); c++) {
	  out(x, c, s*Nvec + v) = in(x, s, c);
	}
      }
    }
  }

  void FillV(cpuColorSpinorField &V, const cpuColorSpinorField **B, int Nvec) {
    if (V.Precision() == QUDA_DOUBLE_PRECISION) {
      ColorSpinorFieldOrder<double> *vOrder = createOrder<double>(V);
      for (int v=0; v<Nvec; v++) {
	ColorSpinorFieldOrder<double> *bOrder = createOrder<double>(*B[v]);
	fill(*vOrder, *bOrder, v, Nvec);
	delete bOrder;
      }
      delete vOrder;
    } else {
      ColorSpinorFieldOrder<float> *vOrder = createOrder<float>(V);
      for (int v=0; v<Nvec; v++) {
	ColorSpinorFieldOrder<float> *bOrder = createOrder<float>(*B[v]);
	fill(*vOrder, *bOrder, v, Nvec);
	delete bOrder;
      }
      delete vOrder;
    }    
  }

  // Creates a block-ordered version of a ColorSpinorField
  // N.B.: Only works for the V field, as we need to block spin.
  template <bool toBlock, class Complex, class FieldOrder>
  void blockOrderV(Complex *out, FieldOrder &in, int Nvec, 
		   const int *geo_map, const int *geo_bs, int spin_bs,
		   const ColorSpinorField &V) {

    int nColor = in.Nspin();     //spin indices of V correspond to fine color
    int vec_spin = in.Ncolor();  // color indices of V correspond to (v * nSpin + s)
    int nSpin = vec_spin / Nvec;
    int nSpin_coarse = nSpin / spin_bs; // this is number of chiral blocks

    // length of each fine site that is contained in a single block
    int fsite_length = in.Nspin()*spin_bs; 

    //Compute the size of each block
    int geoBlockSize = 1;
    for (int d=0; d<in.Ndim(); d++) geoBlockSize *= geo_bs[d];
    int blockSize = geoBlockSize * nColor * spin_bs; // blockSize includes internal dof

    int x[QUDA_MAX_DIM]; // global coordinates
    
    int checkLength = in.Volume() * in.Ncolor() * in.Nspin();
    int *check = new int[checkLength];
    int count = 0;

    // Run through the fine grid and do the block ordering
    for (int i=0; i<in.Volume(); i++) {
      
      // Get fine grid coordinates
      V.LatticeIndex(x, i);

      //The coordinates within a block
      int y[QUDA_MAX_DIM]; 

      //Geometric Offset within a block
      int blockOffset = 0;

      //Compute the offset within a block 
      // (x fastest direction, t is slowest direction, non-parity ordered)
      for (int d=in.Ndim()-1; d>=0; d--) {
	y[d] = x[d]%geo_bs[d];
	blockOffset *= geo_bs[d];
	blockOffset += y[d];
      }

      //Take the block-ordered offset from the coarse grid offset (geo_map) 
      int offset = geo_map[i]*nSpin_coarse*Nvec*geoBlockSize*nColor*spin_bs;

      for (int vs=0; vs<vec_spin; vs++) {  // loop over spin and vectors
	for (int c=0; c<nColor; c++) { // loop over color
	  int s = vs / Nvec;
	  int v = vs % Nvec;
	  int chirality = s / spin_bs; // chirality is the coarse spin
	  int blockSpin = s % spin_bs; // the remaing spin dof left in each block

	  int ind = offset +                  // which block (excluding chirality)
	    chirality*blockSize*Nvec +        // which chiral block
	    v * blockSize +                   // which vector
	    blockOffset*nSpin_coarse*nColor + // which block size
	    blockSpin * nColor +              // which block spin
	    c;                                // which color

	  if (toBlock) // going to block order
	    out[ind] = in(i, c, vs);
	  else // coming from block order
	    in(i, c, vs) = out[ind];

	  check[count++] = ind;
	}
      }
	
    }
    
    if (count != checkLength) {
      errorQuda("Number of elements packed %d does not match expected value %d", 
		count, checkLength);
    }

    delete []check;
  }

  // Orthogonalise the nc vectors v[] of length n
  // this assumes the ordering v[(b * Nvec + v) * blocksize + i]

  template <class Complex>
  void blockGramSchmidt(Complex *v, int nBlocks, int Nc, int blockSize) {
    
    for (int b=0; b<nBlocks; b++) {
      for (int jc=0; jc<Nc; jc++) {
      
	for (int ic=0; ic<jc; ic++) {
	  // Calculate dot product.
	  Complex dot = 0.0;
	  for (int i=0; i<blockSize; i++) 
	    dot += conj(v[(b*Nc+ic)*blockSize+i]) * v[(b*Nc+jc)*blockSize+i];
	
	  // Subtract the blocks to orthogonalise
	  for (int i=0; i<blockSize; i++) 
	    v[(b*Nc+jc)*blockSize+i] -= dot * v[(b*Nc+ic)*blockSize+i];
	}
      
	// Normalize the block
	// Again, nrm2 is pure real, but need to use Complex because of template.
	Complex nrm2 = 0.0;
	for (int i=0; i<blockSize; i++) nrm2 += norm(v[(b*Nc+jc)*blockSize+i]);
	nrm2 = 1.0/sqrt(nrm2.real());
      
	for (int i=0; i<blockSize; i++) v[(b*Nc+jc)*blockSize+i] = nrm2 * v[(b*Nc+jc)*blockSize+i];

      }
    }

  }

  //Orthogonalize null vectors
  void BlockOrthogonalize(cpuColorSpinorField &V, int Nvec, 
			  const int *geo_bs, const int *geo_map, int spin_bs) {
  
    int nColor = V.Nspin();
    int vec_spin = V.Ncolor();
    int nSpin = vec_spin / Nvec;

    int geo_blocksize = 1;
    for (int d = 0; d < V.Ndim(); d++) geo_blocksize *= geo_bs[d];
    int chiralBlocks = nSpin / spin_bs;
    int numblocks = (V.Volume()/geo_blocksize) * chiralBlocks;
    int blocksize = geo_blocksize * nColor * spin_bs; 

    if (V.Precision() == QUDA_DOUBLE_PRECISION) {
      std::complex<double> *Vblock = new std::complex<double>[V.Volume()*V.Nspin()*V.Ncolor()];
      ColorSpinorFieldOrder<double> *vOrder = createOrder<double>(V);

      blockOrderV<true>(Vblock, *vOrder, Nvec, geo_map, geo_bs, spin_bs, V);
      blockGramSchmidt(Vblock, numblocks, Nvec, blocksize);  
      blockOrderV<false>(Vblock, *vOrder, Nvec, geo_map, geo_bs, spin_bs, V);

      delete vOrder;
      delete []Vblock;
    } else {
      std::complex<float> *Vblock = new std::complex<float>[V.Volume()*V.Nspin()*V.Ncolor()];
      ColorSpinorFieldOrder<float> *vOrder = createOrder<float>(V);

      blockOrderV<true>(Vblock, *vOrder, Nvec, geo_map, geo_bs, spin_bs, V);
      blockGramSchmidt(Vblock, numblocks, Nvec, blocksize);  
      blockOrderV<false>(Vblock, *vOrder, Nvec, geo_map, geo_bs, spin_bs, V);

      delete vOrder;
      delete []Vblock;
    }
  }

} // namespace quda
