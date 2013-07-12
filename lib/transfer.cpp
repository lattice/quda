#include <transfer.h>
#include <blas_quda.h>

namespace quda {

  Transfer::Transfer(cpuColorSpinorField **B, int Nvec, int *geo_bs, int spin_bs)
    : B(B), Nvec(Nvec), V(0), tmp(0), geo_map(0), spin_map(0) 
  {

    // create the storage for the final block orthogonal elements
    ColorSpinorParam param(*B[0]); // takes the geometry from the null-space vectors
    param.nSpin = B[0]->Ncolor(); // the spin dimension corresponds to fine nColor
    param.nColor = B[0]->Nspin() * Nvec; // nColor = number of spin components * number of null-space vectors
    param.create = QUDA_ZERO_FIELD_CREATE;
    // the V field is defined on all sites regardless of B field (maybe the B fields are always full?)
    if (param.siteSubset == QUDA_PARITY_SITE_SUBSET) {
      param.siteSubset = QUDA_FULL_SITE_SUBSET;
      param.x[0] *= 2;
    }
    V = new cpuColorSpinorField(param);
    fillV(); // copy the null space vectors into V

    // create the storage for the intermediate temporary vector
    param.nSpin = B[0]->Nspin(); // tmp has same nSpin has the fine dimension
    param.nColor = Nvec; // tmp has nColor equal to the number null-space vectors
    tmp = new cpuColorSpinorField(param);

    // allocate and compute the fine-to-coarse site map
    geo_map = new int[B[0]->Volume()];
    createGeoMap(geo_bs);

    // allocate the fine-to-coarse spin map
    spin_map = new int[B[0]->Nspin()];
    createSpinMap(spin_bs);

    int geo_blocksize = 1;
    for (int d = 0; d < tmp->Ndim(); d++) {
      geo_blocksize *= geo_bs[d];
    }
    int numblocks = B[0]->Volume()/geo_blocksize;
    int fsite_length = V->Nspin()*V->Ncolor()/(spin_bs*Nvec);


   //Orthogonalize null vectors
    if (V->Precision() == QUDA_DOUBLE_PRECISION) {
      std::complex<double> * Vblock;
      Vblock = (std::complex<double> *) malloc(V->Volume()*param.nSpin*param.nColor*Nvec*sizeof(std::complex<double>));
      blockOrderV(Vblock, *(V->order_double), geo_bs, spin_bs);
      blockGramSchmidt(Vblock, numblocks, spin_bs*Nvec, geo_blocksize*fsite_length);  
      undoblockOrderV(*(V->order_double), Vblock, geo_bs, spin_bs);
      free(Vblock);
    }
    else {
      std::complex<float> * Vblock;
      Vblock = (std::complex<float> *) malloc(V->Volume()*param.nSpin*param.nColor*Nvec*sizeof(std::complex<float>));
      blockOrderV(Vblock,*(V->order_single), geo_bs, spin_bs);
      blockGramSchmidt(Vblock, numblocks, spin_bs*Nvec, geo_blocksize*fsite_length);      
      undoblockOrderV(*(V->order_single), Vblock, geo_bs, spin_bs);
      free(Vblock);
    }
  }

  Transfer::~Transfer() {

    if (spin_map) delete [] spin_map;
    if (geo_map) delete [] geo_map;

    if (V) delete V;
    if (tmp) delete tmp;

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

  void Transfer::fillV() {
    if (V->Precision() == QUDA_DOUBLE_PRECISION) {
      for (int v=0; v<Nvec; v++) fill(*(V->order_double), *(B[v]->order_double), v, Nvec);
    } else {
      for (int v=0; v<Nvec; v++) fill(*(V->order_single), *(B[v]->order_single), v, Nvec);
    }    
    printfQuda("V fill check %e\n", norm2(*V));
  }

  // compute the fine-to-coarse site map
  void Transfer::createGeoMap(int *geo_bs) {

    int x[QUDA_MAX_DIM];

    // create a spinor with coarse geometry so we can use its OffsetIndex member function
    ColorSpinorParam param(*tmp);
    param.nColor = 1;
    param.nSpin = 1;
    param.create = QUDA_ZERO_FIELD_CREATE;
    for (int d=0; d<param.nDim; d++) param.x[d] /= geo_bs[d];
    cpuColorSpinorField coarse(param);

    //std::cout << coarse;

    // compute the coarse grid point for every site (assuming parity ordering currently)
    for (int i=0; i<tmp->Volume(); i++) {
      // compute the lattice-site index for this offset index
      tmp->LatticeIndex(x, i);

      printf("fine idx %d = fine (%d,%d,%d,%d), ", i, x[0], x[1], x[2], x[3]);

      // compute the corresponding coarse-grid index given the block size
      for (int d=0; d<tmp->Ndim(); d++) x[d] /= geo_bs[d];

      // compute the coarse-offset index and store in the geo_map
      int k;
      coarse.OffsetIndex(k, x); // this index is parity ordered
      geo_map[i] = k;

      printf("coarse (%d,%d,%d,%d), coarse idx %d\n", x[0], x[1], x[2], x[3], k);
    }

  }

  // compute the fine spin to coarse spin map
  void Transfer::createSpinMap(int spin_bs) {

    for (int s=0; s<B[0]->Nspin(); s++) {
      spin_map[s] = s / spin_bs;
    }

  }

  // Applies the grid prolongation operator (coarse to fine)
  template <class FineSpinor, class CoarseSpinor>
  void prolongate(FineSpinor &out, const CoarseSpinor &in, const int *geo_map, const int *spin_map) {

    for (int x=0; x<out.Volume(); x++) {
      for (int s=0; s<out.Nspin(); s++) {
	for (int c=0; c<out.Ncolor(); c++) {
	  out(x, s, c) = in(geo_map[x], spin_map[s], c);
	}
      }
    }

  }

  // Applies the grid restriction operator (fine to coarse)
  template <class CoarseSpinor, class FineSpinor>
  void restrict(CoarseSpinor &out, const FineSpinor &in, const int* geo_map, const int* spin_map) {

    // We need to zero all elements first, since this is a reduction operation
    for (int x=0; x<in.Volume(); x++) {
      for (int s=0; s<in.Nspin(); s++) {
	for (int c=0; c<in.Ncolor(); c++) {
	  out(geo_map[x], spin_map[s], c) = 0.0;
	}
      }
    }

    for (int x=0; x<in.Volume(); x++) {
      for (int s=0; s<in.Nspin(); s++) {
	for (int c=0; c<in.Ncolor(); c++) {
	  out(geo_map[x], spin_map[s], c) += in(x, s, c);
	}
      }
    }

  }

  /*
    Rotates from the coarse-color basis into the fine-color basis.  This
    is the second step of applying the prolongator.
  */
  template <class FineColor, class CoarseColor, class Rotator>
  void rotateFineColor(FineColor &out, const CoarseColor &in, const Rotator &V) {

    for(int x=0; x<in.Volume(); x++) {

      for (int s=0; s<out.Nspin(); s++) for (int i=0; i<out.Ncolor(); i++) out(x, s, i) = 0.0;

      for (int i=0; i<out.Ncolor(); i++) {
	for (int s=0; s<in.Nspin(); s++) {
	  for (int j=0; j<in.Ncolor(); j++) { 
	    // V is a ColorMatrixField with dimension [out.Nc][in.Ns*in.Nc] - the rotation has spin dependence
	    out(x, s, i) += V(x, i, s*in.Ncolor() + j) * in(x, s, j);
	  }
	}
      }
      
    }

  }

  /*
    Rotates from the fine-color basis into the coarse-color basis.
  */
  template <class CoarseColor, class FineColor, class Rotator>
  void rotateCoarseColor(CoarseColor &out, const FineColor &in, const Rotator &V) {

    for(int x=0; x<in.Volume(); x++) {

      for (int s=0; s<out.Nspin(); s++) for (int i=0; i<out.Ncolor(); i++) out(x, s, i) = 0.0;

      for (int i=0; i<out.Ncolor(); i++) {
	for (int s=0; s<out.Nspin(); s++) {
	  for (int j=0; j<in.Ncolor(); j++) {
	    out(x, s, j) += std::conj(V(x, i, s*in.Ncolor() + j)) * in(x, s, i);
	  }
	}
      }
    }

  }


  // apply the prolongator
  void Transfer::P(cpuColorSpinorField &out, const cpuColorSpinorField &in) {

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      prolongate(*(tmp->order_double), *(in.order_double), geo_map, spin_map);
      rotateFineColor(*(out.order_double), *(tmp->order_double), *(V->order_double));
    } else {
      prolongate(*(tmp->order_single), *(in.order_single), geo_map, spin_map);
      rotateFineColor(*(out.order_single), *(tmp->order_single), *(V->order_single));
    }

  }

  // apply the restrictor
  void Transfer::R(cpuColorSpinorField &out, const cpuColorSpinorField &in) {

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      rotateCoarseColor(*(tmp->order_double), *(in.order_double), *(V->order_double));
      restrict(*(out.order_double), *(tmp->order_double), geo_map, spin_map);
    } else {
      rotateCoarseColor(*(tmp->order_single), *(in.order_single), *(V->order_single));
      restrict(*(out.order_single), *(tmp->order_single), geo_map, spin_map);
    }

  }

  // Creates a block-ordered version of a ColorSpinorField
  // N.B.: Only works for the V field, as we need to block spin.
  template <class Complex, class FieldOrder>
  void Transfer::blockOrderV(Complex *out, const FieldOrder &in, int *geo_bs, int spin_bs) {

    //fsite_length = V->Nspin()*V->Ncolor()/(Nvec*spin_bs) = Nc*Ns/spin_bs
    int fsite_length = V->Nspin()*V->Ncolor()/(Nvec*spin_bs);

    //Compute the size of each block
    int blockSize = 1;
    for (int d=0; d<V->Ndim(); d++) {
      blockSize *= geo_bs[d];
    }
    blockSize *= fsite_length;

    int x[QUDA_MAX_DIM];
    
   // Run through the fine grid and do the block ordering
    for (int i=0; i<V->Volume(); i++) {
      
      // Get fine grid coordinates
      V->LatticeIndex(x, i);

      //Take the block-ordered offset from the coarse grid offset (geo_map) 
      int offset = geo_map[i]*blockSize;

      //The coordinates within a block
      int y[QUDA_MAX_DIM];

      //Geometric Offset within a block
      int block_offset = 0;

      //Compute the offset within a block (x fastest direction, t is slowest direction, non-parity ordered)
      for (int d=tmp->Ndim(); d>=0; d--) {
	y[d] = x[d]%geo_bs[d];
	block_offset *= geo_bs[d];
	block_offset += y[d];
      }

      //Loop over spin and color indices of V field.
      //color indices of V correspond to spin and number of null vectors, with spin component changing fastest
      //spin indices of V correspond to fine color
      for (int c=0; c<V->Ncolor(); c++) {
	for (int s=0; s<V->Nspin(); s++) {
	  out[offset + (c/spin_bs)*block_offset*fsite_length + (s*(c%spin_bs)+s)] = in(i, s, c);
	}
      }
	
    }
  }

  // 
  template <class FieldOrder, class Complex>
  void Transfer::undoblockOrderV(FieldOrder &out, Complex *in, int *geo_bs, int spin_bs) {

    //fsite_length = V->Nspin()*V->Ncolor()/(Nvec*spin_bs) = Nc*Ns/spin_bs
    int fsite_length = V->Nspin()*V->Ncolor()/(Nvec*spin_bs);

    //Compute the size of each block
    int blockSize = 1;
    for (int d=0; d<V->Ndim(); d++) {
      blockSize *= geo_bs[d];
    }
    blockSize *= fsite_length;

    int x[QUDA_MAX_DIM];
    
   // Run through the fine grid and do the block ordering
    for (int i=0; i<V->Volume(); i++) {
      
      // Get fine grid coordinates
      V->LatticeIndex(x, i);

      //Take the block-ordered offset from the coarse grid offset (geo_map) 
      int offset = geo_map[i]*blockSize;

      //The coordinates within a block
      int y[QUDA_MAX_DIM];

      //Geometric Offset within a block
      int block_offset = 0;

      //Compute the offset within a block (x fastest direction, t is slowest direction, non-parity ordered)
      for (int d=tmp->Ndim(); d>=0; d--) {
	y[d] = x[d]%geo_bs[d];
	block_offset *= geo_bs[d];
	block_offset += y[d];
      }

      //Loop over spin and color indices of V field.
      //color indices of V correspond to spin and number of null vectors, with spin component changing fastest
      //spin indices of V correspond to fine color
      for (int c=0; c<V->Ncolor(); c++) {
	for (int s=0; s<V->Nspin(); s++) {
	  out(i,s,c) = in[offset + (c/spin_bs)*block_offset*fsite_length + (s*(c%spin_bs)+s)];
	}
      }
	
    }
  }


  // Orthogonalise the nc vectors v[] of length n
  template <class Complex>
  void Transfer::blockGramSchmidt(Complex *v, int nBlocks, int Nc, int blockSize) {
    
    for (int b=0; b<nBlocks; b++) {
      for (int jc=0; jc<Nc; jc++) {
      
	for (int ic=0; ic<jc; ic++) {
	  // Calculate dot product.
	  // dot should be double, but doesn't play well with complex<float>.  Use Complex type instead
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

} // namespace quda
