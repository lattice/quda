#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

namespace quda {

  template <typename Float>
  ColorSpinorFieldOrder<Float>* createOrder(const cpuColorSpinorField &a) {
    ColorSpinorFieldOrder<Float>* ptr=0;
    if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) 
      ptr = new SpaceSpinColorOrder<Float>(const_cast<cpuColorSpinorField&>(a));
    else if (a.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) 
      ptr = new SpaceColorSpinOrder<Float>(const_cast<cpuColorSpinorField&>(a));
    else if (a.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) 
      ptr = new QOPDomainWallOrder<Float>(const_cast<cpuColorSpinorField&>(a));
    else
      errorQuda("Order %d not supported in cpuColorSpinorField", a.FieldOrder());
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

  void FillV(ColorSpinorField &V, const ColorSpinorField **B, int Nvec) {
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
  template <class Complex, class FieldOrder>
  void blockOrderV(Complex *out, const FieldOrder &in, int Nvec, 
		   const int *geo_map, const int *geo_bs, int spin_bs,
		   const ColorSpinorField &V) {

    int fspin_components = in.Ncolor()/Nvec;

    //fsite_length = V->Nspin()*spin_bs
    int fsite_length = in.Nspin()*spin_bs;


    //Compute the size of each block
    int blockSize = 1;
    for (int d=0; d<in.Ndim(); d++) {
      blockSize *= geo_bs[d];
    }
    blockSize *= fsite_length;

    int x[QUDA_MAX_DIM];
    
    // Run through the fine grid and do the block ordering
    for (int i=0; i<in.Volume(); i++) {
      
      // Get fine grid coordinates
      V.LatticeIndex(x, i);

      //Take the block-ordered offset from the coarse grid offset (geo_map) 
      int offset = geo_map[i]*blockSize*Nvec*fspin_components/spin_bs;

      //The coordinates within a block
      int y[QUDA_MAX_DIM];

      //Geometric Offset within a block
      int block_offset = 0;

      //Compute the offset within a block (x fastest direction, t is slowest direction, non-parity ordered)
      for (int d=in.Ndim()-1; d>=0; d--) {
	y[d] = x[d]%geo_bs[d];
	block_offset *= geo_bs[d];
	block_offset += y[d];
      }

      //Loop over spin and color indices of V field.
      //color indices of V correspond to spin and number of null vectors, with spin component changing fastest
      //spin indices of V correspond to fine color
      for (int c=0; c<in.Ncolor(); c++) {
	for (int s=0; s<in.Nspin(); s++) {
	  int ind = offset + (c*fspin_components/spin_bs)*blockSize + (block_offset*fsite_length + in.Nspin()*(c%(fspin_components/spin_bs))+s);
	  out[ind] = in(i, s, c);
	}
      }
	
    }
  }

  // 
  template <class FieldOrder, class Complex>
  void undoblockOrderV(FieldOrder &out, Complex *in, int Nvec, 
		       const int *geo_map, const int *geo_bs, int spin_bs, 
		       const ColorSpinorField &V) {
    int fspin_components = out.Ncolor()/Nvec;

    //fsite_length = V->Nspin()*spin_bs = Nc*spin_bs
    int fsite_length = out.Nspin()*spin_bs;

    //Compute the size of each block
    int blockSize = 1;
    for (int d=0; d<out.Ndim(); d++) {
      blockSize *= geo_bs[d];
    }
    blockSize *= fsite_length;

    int x[QUDA_MAX_DIM];
    
    // Run through the fine grid and do the block ordering
    for (int i=0; i<out.Volume(); i++) {
      
      // Get fine grid coordinates
      V.LatticeIndex(x, i);

      //Take the block-ordered offset from the coarse grid offset (geo_map) 
      int offset = geo_map[i]*blockSize*Nvec*fspin_components/spin_bs;

      //The coordinates within a block
      int y[QUDA_MAX_DIM];

      //Geometric Offset within a block
      int block_offset = 0;

      //Compute the offset within a block (x fastest direction, t is slowest direction, non-parity ordered)
      for (int d=out.Ndim()-1; d>=0; d--) {
	y[d] = x[d]%geo_bs[d];
	block_offset *= geo_bs[d];
	block_offset += y[d];
      }

      //Loop over spin and color indices of V field.
      //color indices of V correspond to spin and number of null vectors, with spin component changing fastest
      //spin indices of V correspond to fine color
      for (int c=0; c<out.Ncolor(); c++) {
	for (int s=0; s<out.Nspin(); s++) {
	  int ind = offset + (c*spin_bs/fspin_components)*blockSize + (block_offset*fsite_length + out.Nspin()*(c%(fspin_components/spin_bs))+s);
	  out(i,s,c) = in[ind];
	}
      }
	
    }
  }


  // Orthogonalise the nc vectors v[] of length n
  template <class Complex>
  void blockGramSchmidt(Complex *v, int nBlocks, int Nc, int blockSize) {
    
    for (int b=0; b<nBlocks; b++) {
      for (int jc=0; jc<Nc; jc++) {
      
	for (int ic=0; ic<jc; ic++) {
	  // Calculate dot product.
	  Complex dot = 0.0;
	  for (int i=0; i<blockSize; i++) dot += conj(v[(b*Nc+ic)*blockSize+i]) * v[(b*Nc+jc)*blockSize+i];
	
	  // Subtract the blocks to orthogonalise
	  for (int i=0; i<blockSize; i++) v[(b*Nc+jc)*blockSize+i] -= dot * v[(b*Nc+ic)*blockSize+i];
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

  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, const int *geo_bs, const int *geo_map, int spin_bs) {

  
    //Orthogonalize null vectors
    
    int geo_blocksize = 1;
    for (int d = 0; d < V.Ndim(); d++) geo_blocksize *= geo_bs[d];
    int numblocks = V.Volume()/geo_blocksize;
    int fsite_length = V.Nspin()*spin_bs;

    if (V.Precision() == QUDA_DOUBLE_PRECISION) {
      std::complex<double> *Vblock = new std::complex<double>[V.Volume()*V.Nspin()*V.Ncolor()];
      ColorSpinorFieldOrder<double> *vOrder = createOrder<double>(V);

      blockOrderV(Vblock, *vOrder, Nvec, geo_map, geo_bs, spin_bs, V);
      blockGramSchmidt(Vblock, numblocks, V.Ncolor()/spin_bs, geo_blocksize*fsite_length);  
      undoblockOrderV(*vOrder, Vblock, Nvec, geo_map, geo_bs, spin_bs, V);

      delete vOrder;
      delete []Vblock;
    } else {
      std::complex<float> *Vblock = new std::complex<float>[V.Volume()*V.Nspin()*V.Ncolor()];
      ColorSpinorFieldOrder<float> *vOrder = createOrder<float>(V);

      blockOrderV(Vblock, *vOrder, Nvec, geo_map, geo_bs, spin_bs, V);
      blockGramSchmidt(Vblock, numblocks, V.Ncolor()/spin_bs, geo_blocksize*fsite_length);  
      undoblockOrderV(*vOrder, Vblock, Nvec, geo_map, geo_bs, spin_bs, V);

      delete vOrder;
      delete []Vblock;
    }
  }

} // namespace quda
