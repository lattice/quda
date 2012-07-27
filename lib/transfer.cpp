#include <transfer.h>

Transfer::Transfer(cpuColorSpinorField *B, int Nvec, int *geo_bs, int spin_bs)
  : V(0), tmp(0), geo_map(0), spin_map(0) 
{

  // create the storage for the final block orthogonal elements
  ColorSpinorParam param(B[0]); // takes the geometry from the null-space vectors
  param.nSpin = B[0].Ncolor(); // the spin dimension corresponds to fine nColor
  param.nColor = B[0].Nspin() * Nvec; // nColor = number of spin components * number of null-space vectors
  V = new cpuColorSpinorField(param);

  // create the storage for the intermediate temporary vector
  param.nSpin = B[0].Nspin(); // tmp has same nSpin has the fine dimension
  param.nColor = Nvec; // tmp has nColor equal to the number null-space vectors
  tmp = new cpuColorSpinorField(param);

  // allocate and compute the fine-to-coarse site map
  geo_map = new int[B[0].Ndim()];
  createGeoMap(geo_bs);

  // allocate the fine-to-coarse spin map
  spin_map = new int[B[0].Nspin() / spin_bs];
  createSpinMap(spin_bs);
}

Transfer::~Transfer() {

  if (spin_map) delete [] spin_map;
  if (geo_map) delete [] geo_map;

  if (V) delete V;
  if (tmp) delete tmp;

}

// compute the fine-to-coarse site map
void Transfer::createGeoMap(int *geo_bs) {

  int x[QUDA_MAX_DIM];

  // create a spinor with coarse geometry so we can use its OffsetIndex member function
  ColorSpinorParam param(tmp);
  param.nColor = 1;
  param.nSpin = 1;
  cpuColorSpinorField coarse(param);

  // compute the coarse grid point for every site
  for (int i=0; i<tmp.Volume(); i++) {

    // compute the lattice-site index for this offet index
    tmp.LatticeIndex(x, i);
    
    // compute the corresponding coarse-grid index given the block size
    for (int d=0; d<tmp.Ndim(); d++) x[d] /= geo_bs[d];

    // compute the coarse-offset index and store in the geo_map
    int k;
    coarse.SiteIndex(k, x);
    geo_map[i] = k;
  }

}

// compute the fine spin to coarse spin map
void Transfer::createSpinMap(int spin_bs) {

  for (int s=0; s<B[0].Nspin(); s++) {
    spin_map[s] = s / spin_bs;
  }

}

// apply the prolongator
void Transfer::P(cpuColorSpinorField &out, cpuColorSpinorField &in) {

  if (out.Precision() == QUDA_DOUBLE_PRECISION) {
    prolongate(tmp->order_double, in->order_double, geo_map, spin_map);
    rotateFineColor(out->order_double, tmp->order_double, V->order_double);
  } else {
    prolongate(tmp->order_single, in->order_single, geo_map, spin_map);
    rotateFineColor(out->order_single, tmp->order_single, V->order_single);
  }

}

// apply the restrictor
void Transfer::R(cpuColorSpinorField &out, cpuColorSpinorField &in) {

  if (out.Precision() == QUDA_DOUBLE_PRECISION) {
    rotateCoarseColor(tmp->order_double, in->order_double, V->order_double);
    restrict(out->order_double, tmp->order_double, geo_map, spin_map);
  } else {
    rotateCoarseColor(tmp->order_single, in->order_single, V->order_single);
    restrict(out->order_single, tmp->order_single, geo_map, spin_map);
  }

}

// Applies the grid prolongation operator (coarse to fine)
template <class FineSpinor, class CoarseSpinor>
void prolongate(FineSpinor &out, const CoarseSpinor &in, const uint *geo_map, const uint* spin_map) {

  for (uint x=0; x<out.Volume(); x++) {
    for (uint s=0; s<out.Nspin(); s++) {
      for (uint c=0; c<out.Ncolor(); c++) {
	out(x, s, c) = in(geo_map[x], spin_map[s], c);
      }
    }
  }

}

// Applies the grid restriction operator (fine to coarse)
template <class CoarseSpinor, class FineSpinor>
void restrict(CoarseSpinor &out, const FineSpinor &in, const uint* geo_map, const uint* spin_map) {

  // We need to zero all elements first, since this is a summation operation
  for (uint x=0; x<in.Volume(); x++) {
    for (uint s=0; s<in.Nspin(); s++) {
      for (uint c=0; c<in.Ncolor(); c++) {
	out(geo_map[is], spin_map[s], c) = 0.0;
      }
    }
  }

  for (uint x=0; x<in.Volume(); x++) {
    for (uint s=0; s<in.Nspin(); s++) {
      for (uint c=0; c<in.Ncolor(); c++) {
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
  Rotates from the fine-color basis into the color-color basis.
 */
template <class CoarseColor, class FineColor, class Rotator>
void rotateCoarseColor(CoarseColor &out, const FineColor &in, const Rotator &V) {

  for(int x=0; x<in.Volume(); x++) {

    for (int s=0; s<out.Nspin(); s++) for (int i=0; i<out.Ncolor(); i++) out(x, s, i) = 0.0;

    for (int i=0; i<out.Ncolor(); i++) {
      for (int s=0; s<out.Nspin(); s++) {
	for (int j=0; j<in.Ncolor(); j++) {
	  out(x, s, j) += conj(V(x, i, s*in.Ncolor() + j)) * in(x, s, i);
	}
      }
    }

}

/*
  Orthogonalise the nc vectors v[] of length n
*/
template <class Complex>
void blockGramSchmidt(Complex ***v, uint nBlocks, uint Nc, uint blockSize) {

  for (uint b=0; b<nBlocks; b++) {
    for (int jc=0; jc<Nc; jc++) {
      
      for (int ic=0; ic<jc; ic++) {
	// Calculate dot product
	complex<double> dot = 0.0;
	for (int i=0; i<blockSize; i++) dot += v[b][ic][i] * v[b][jc][i];
	
	// Subtract the blocks to orthogonalise
	for (int i=0; i<blockSize; i++) v[b][jc][i] -= dot * v[b][ic][i];
      }
      
      // Normalize the block
      double nrm2 = 0.0;
      for (int i=0; i<blockSize; i++) 
	norm2 += real(v[b][jc][i])*real(v[b][jc][i]) + imag(v[b][jc][i])*imag(v[b][jc][i]);
      nrm2 = 1.0/sqrt(nrm2);
      
      for (int i=0; i<blockSize; i++) v[b][jc][i] = nrm2 * v[b][jc][i];
    }

  }

}
