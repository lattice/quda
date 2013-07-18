#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <typeinfo>

namespace quda {

  // copy the null-space vectors into the V-field
  template <class V, class B>
  void fill(V &out, const B &in, int v, int Nvec) {
    for (int x=0; x<out.Volume(); x++) {
      for (int s=0; s<in.Nspin(); s++) {
	for (int c=0; c<in.Ncolor(); c++) {
	  out(x, v, s, c) = in(x, s, c);
	}
      }
    }
  }

  void FillV(cpuColorSpinorField &V, cpuColorSpinorField **B, int Nvec) {
    if (V.Precision() == QUDA_DOUBLE_PRECISION) {
      ColorSpinorFieldOrder<double> *vOrder = createOrder<double>(V, Nvec);
      for (int v=0; v<Nvec; v++) {
	ColorSpinorFieldOrder<double> *bOrder = createOrder<double>(*B[v]);
	fill(*vOrder, *bOrder, v, Nvec);
	delete bOrder;
      }
      delete vOrder;
    } else {
      ColorSpinorFieldOrder<float> *vOrder = createOrder<float>(V, Nvec);
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
		   const cpuColorSpinorField &V) {

    int nSpin_coarse = in.NspinPacked() / spin_bs; // this is number of chiral blocks

    //Compute the size of each block
    int geoBlockSize = 1;
    for (int d=0; d<in.Ndim(); d++) geoBlockSize *= geo_bs[d];
    int blockSize = geoBlockSize * in.NcolorPacked() * spin_bs; // blockSize includes internal dof

    int x[QUDA_MAX_DIM]; // global coordinates
    int y[QUDA_MAX_DIM]; // local coordinates within a block (full site ordering)

    int checkLength = in.Volume() * in.Ncolor() * in.Nspin();
    int *check = new int[checkLength];
    int count = 0;

    // Run through the fine grid and do the block ordering
    for (int i=0; i<in.Volume(); i++) {
      
      // Get fine grid coordinates
      V.LatticeIndex(x, i);

      //Compute the geometric offset within a block 
      // (x fastest direction, t is slowest direction, non-parity ordered)
      int blockOffset = 0;
      for (int d=in.Ndim()-1; d>=0; d--) {
	y[d] = x[d]%geo_bs[d];
	blockOffset *= geo_bs[d];
	blockOffset += y[d];
      }

      //Take the block-ordered offset from the coarse grid offset (geo_map) 
      int offset = geo_map[i]*nSpin_coarse*Nvec*geoBlockSize*in.NcolorPacked()*spin_bs;

      for (int v=0; v<in.NvecPacked(); v++) {
	for (int s=0; s<in.NspinPacked(); s++) {
	  for (int c=0; c<in.NcolorPacked(); c++) {
	    int chirality = s / spin_bs; // chirality is the coarse spin
	    int blockSpin = s % spin_bs; // the remainingg spin dof left in each block

	    int index = offset +                                              // geo block
	      chirality * Nvec * geoBlockSize * spin_bs * in.NcolorPacked() + // chiral block
	                     v * geoBlockSize * spin_bs * in.NcolorPacked() + // vector
	                          blockOffset * spin_bs * in.NcolorPacked() + // local geometry
	                                        blockSpin*in.NcolorPacked() + // block spin
	                                                                 c;   // color

	    if (toBlock) out[index] = in(i, v, s, c); // going to block order
	    else in(i, v, s, c) = out[index]; // coming from block order
	    
	    check[count++] = index;
	  }
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
  
    int geo_blocksize = 1;
    for (int d = 0; d < V.Ndim(); d++) geo_blocksize *= geo_bs[d];

    if (V.Precision() == QUDA_DOUBLE_PRECISION) {
      std::complex<double> *Vblock = new std::complex<double>[V.Volume()*V.Nspin()*V.Ncolor()];
      ColorSpinorFieldOrder<double> *vOrder = createOrder<double>(V, Nvec);

      int blocksize = geo_blocksize * vOrder->NcolorPacked() * spin_bs; 
      int chiralBlocks = vOrder->NspinPacked() / spin_bs;
      int numblocks = (V.Volume()/geo_blocksize) * chiralBlocks;

      blockOrderV<true>(Vblock, *vOrder, Nvec, geo_map, geo_bs, spin_bs, V);
      blockGramSchmidt(Vblock, numblocks, Nvec, blocksize);  
      blockOrderV<false>(Vblock, *vOrder, Nvec, geo_map, geo_bs, spin_bs, V);

      delete vOrder;
      delete []Vblock;
    } else {
      std::complex<float> *Vblock = new std::complex<float>[V.Volume()*V.Nspin()*V.Ncolor()];
      ColorSpinorFieldOrder<float> *vOrder = createOrder<float>(V, Nvec);

      int blocksize = geo_blocksize * vOrder->NcolorPacked() * spin_bs; 
      int chiralBlocks = vOrder->NspinPacked() / spin_bs;
      int numblocks = (V.Volume()/geo_blocksize) * chiralBlocks;

      blockOrderV<true>(Vblock, *vOrder, Nvec, geo_map, geo_bs, spin_bs, V);
      blockGramSchmidt(Vblock, numblocks, Nvec, blocksize);  
      blockOrderV<false>(Vblock, *vOrder, Nvec, geo_map, geo_bs, spin_bs, V);

      delete vOrder;
      delete []Vblock;
    }
  }

} // namespace quda
